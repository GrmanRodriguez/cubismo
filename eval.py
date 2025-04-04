import math
import numpy as np
import torch
from PIL import Image
from torch import nn
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

def get_sinusoidal_positional_embedding(t, embedding_dim):
    """
    Create sinusoidal embedding of scalar t.
 
    Args:
        t: tensor of shape [B] or [B,1], values in [0,1]
        embedding_dim: size of the output embedding
 
    Returns:
        A tensor of shape [B, embedding_dim].
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)  # [B, 1]
    half_dim = embedding_dim // 2
    # Exponential decay frequencies
    freq = torch.exp(
        torch.linspace(
            0, math.log(10000), half_dim, device=t.device
        )
    )
    # shape: [half_dim]
    # Outer product -> (B, half_dim)
    args = t * freq.unsqueeze(0)
 
    sin = torch.sin(args)
    cos = torch.cos(args)
    emb = torch.cat([sin, cos], dim=1)  # (B, embedding_dim)
    return emb
 
class TimeLabelEmbedding(nn.Module):
    """
    Combines a sinusoidal time embedding and a label embedding, 
    then processes them with an MLP.
    """
    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        # A small MLP to merge time emb + label emb
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
 
    def forward(self, t, y):
        """
        t: (B,) or (B,1) time steps in [0,1]
        y: (B,) class labels
        """
        # Sinusoidal time emb
        time_emb = get_sinusoidal_positional_embedding(t, self.embedding_dim)
        # Label emb
        label_emb = self.label_emb(y)  # [B, embedding_dim]
        # Concatenate
        combined = torch.cat([time_emb, label_emb], dim=1)  # (B, 2*embedding_dim)
        # MLP
        cond = self.mlp(combined)  # (B, embedding_dim)
        return cond
 
class DoubleConv(nn.Module):
    """
    Double 3x3 conv layers with conditioning-based FiLM.
 
    in_channels -> out_channels,
    using two (Conv -> BN -> ReLU) in sequence.
    """
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
 
        self.act = nn.ReLU(inplace=True)
 
        # For FiLM parameters (scale + shift) from the embedding
        self.scale_shift1 = nn.Linear(emb_dim, 2 * out_channels)  # gamma1, beta1
        self.scale_shift2 = nn.Linear(emb_dim, 2 * out_channels)  # gamma2, beta2
 
    def forward(self, x, emb):
        """
        x: (B, in_channels, H, W)
        emb: (B, emb_dim)
        """
        # ----- First Conv -----
        x = self.conv1(x)
        x = self.bn1(x)
 
        # Get gamma,beta from embedding for the first conv
        gamma1, beta1 = self.scale_shift1(emb).chunk(2, dim=1)  # each (B, out_channels)
        # Reshape/broadcast to (B, out_channels, 1, 1)
        gamma1 = gamma1.unsqueeze(-1).unsqueeze(-1)
        beta1 = beta1.unsqueeze(-1).unsqueeze(-1)
 
        # FiLM: x = x * (1 + gamma) + beta
        x = x * (1 + gamma1) + beta1
        x = self.act(x)
 
        # ----- Second Conv -----
        x = self.conv2(x)
        x = self.bn2(x)
 
        # Get gamma,beta from embedding for the second conv
        gamma2, beta2 = self.scale_shift2(emb).chunk(2, dim=1)
        gamma2 = gamma2.unsqueeze(-1).unsqueeze(-1)
        beta2 = beta2.unsqueeze(-1).unsqueeze(-1)
 
        x = x * (1 + gamma2) + beta2
        x = self.act(x)
        return x
 
class Down(nn.Module):
    """
    Downscale by factor of 2 (MaxPool) then DoubleConv with conditioning.
    """
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, emb_dim)
 
    def forward(self, x, emb):
        x = self.pool(x)
        x = self.conv(x, emb)
        return x
 
 
class Up(nn.Module):
    """
    Upscale by factor of 2 (ConvTranspose2d), then concat the skip connection,
    then DoubleConv with conditioning.
    """
    def __init__(self, in_channels, out_channels, emb_dim, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            # Classic transposed conv
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(2 * out_channels, out_channels, emb_dim)
 
    def forward(self, x1, x2, emb):
        """
        x1: the 'lower' resolution feature
        x2: the skip-connection feature from earlier in the encoder
        emb: the conditioning embedding
        """
        x1 = self.up(x1)
 
        # Handle any dimension mismatches (odd input sizes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(
                x1, 
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )
 
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, emb)
        return x
 
class OutConv(nn.Module):
    """
    Final 1x1 convolution to get the desired output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        x = self.conv(x)
        return x
 
 
class UNetCond(nn.Module):
    """
    U-Net that accepts:
      - x: the input image/tensor
      - t: a scalar time in [0,1] (for diffusion steps, etc.)
      - y: a class label for conditional generation
 
    Channels double as we go down, then halve as we go up, 
    with FiLM-like conditioning in each block based on (t,y).
    """
    def __init__(
        self,
        num_classes=10,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        emb_dim=128,
        bilinear=False
    ):
        super().__init__()
        # For time + label conditioning
        self.time_label_emb = TimeLabelEmbedding(num_classes, embedding_dim=emb_dim)
 
        # Encoder/down
        self.inc = DoubleConv(in_channels, base_channels, emb_dim)
        self.down1 = Down(base_channels, base_channels * 2, emb_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, emb_dim)
        self.down3 = Down(base_channels * 4, base_channels * 8, emb_dim)
 
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, emb_dim)
 
        # Decoder/up
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, emb_dim, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, emb_dim, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, emb_dim, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, emb_dim, bilinear)
        self.outc = OutConv(base_channels, out_channels)
 
    def forward(self, x, t, y):
        """
        x: (B, in_channels, H, W)
        t: (B,) or (B,1) time in [0,1]
        y: (B,) class labels
        """
        # 1) Get combined time+label embedding
        emb = self.time_label_emb(t, y)  # (B, emb_dim)
 
        # 2) Encoder
        x1 = self.inc(x, emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)
        x5 = self.down4(x4, emb)
 
        # 3) Decoder
        x = self.up1(x5, x4, emb)
        x = self.up2(x, x3, emb)
        x = self.up3(x, x2, emb)
        x = self.up4(x, x1, emb)
 
        # 4) Final output
        logits = self.outc(x)
        return logits
    
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, y):
        return self.model(x, torch.full(x.shape[:1], fill_value=t, device="cuda"), y.to("cuda"))
 
@torch.no_grad()
def run_flow_discrete(flow_model, starting_point, t_initial, t_final, labels, device="cuda"):
    solver = ODESolver(flow_model)
    times = torch.linspace(t_initial, t_final, 10)
    return solver.sample(x_init = starting_point, method="euler", time_grid=times, step_size=0.1, y=labels)

model = UNetCond(num_classes=2, in_channels=3, out_channels=3, base_channels=64, emb_dim=128, bilinear=False)

model.load_state_dict(torch.load("checkpoints/celeba_1743749607/epoch=4_loss=0.031844.ckpt"))

wrapped_model = WrappedModel(model).cuda()
wrapped_model.eval()

size = [1,3,218,177]
label = {
    "man": 1,
    "woman": 0
}



output = run_flow_discrete(wrapped_model, torch.randn(size).cuda(), 0, 1, torch.tensor([label["man"]]).cuda())
image_array = output.squeeze(0).permute(1,2,0).cpu().numpy()
image = Image.fromarray((image_array * 255).astype(np.uint8))

image.show()