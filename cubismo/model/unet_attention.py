import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal time embedding (unchanged)
def get_sinusoidal_positional_embedding(t, embedding_dim):
    if t.dim() == 1:
        t = t.unsqueeze(1)  # [B, 1]
    half_dim = embedding_dim // 2
    freq = torch.exp(torch.linspace(0, math.log(10000), half_dim, device=t.device))
    args = t * freq.unsqueeze(0)
    sin = torch.sin(args)
    cos = torch.cos(args)
    emb = torch.cat([sin, cos], dim=1)
    return emb

class TimeLabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
 
    def forward(self, t, y):
        time_emb = get_sinusoidal_positional_embedding(t, self.embedding_dim)
        label_emb = self.label_emb(y)
        combined = torch.cat([time_emb, label_emb], dim=1)
        cond = self.mlp(combined)
        return cond

# FiLM-conditioned double conv block (unchanged)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.scale_shift1 = nn.Linear(emb_dim, 2 * out_channels)
        self.scale_shift2 = nn.Linear(emb_dim, 2 * out_channels)
 
    def forward(self, x, emb):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        gamma1, beta1 = self.scale_shift1(emb).chunk(2, dim=1)
        gamma1 = gamma1.unsqueeze(-1).unsqueeze(-1)
        beta1 = beta1.unsqueeze(-1).unsqueeze(-1)
        x = self.act(x * (1 + gamma1) + beta1)
 
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        gamma2, beta2 = self.scale_shift2(emb).chunk(2, dim=1)
        gamma2 = gamma2.unsqueeze(-1).unsqueeze(-1)
        beta2 = beta2.unsqueeze(-1).unsqueeze(-1)
        x = self.act(x * (1 + gamma2) + beta2)
        return x

# Self-attention block on 2D feature maps.
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Use GroupNorm for stability
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        # 1x1 convolutions to compute queries, keys, and values
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
 
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
 
        q = self.q(h).view(B, C, -1)   # (B, C, N)
        k = self.k(h).view(B, C, -1)   # (B, C, N)
        v = self.v(h).view(B, C, -1)   # (B, C, N)
 
        # Compute attention map
        attn = torch.bmm(q.permute(0, 2, 1), k)   # (B, N, N)
        attn = attn / math.sqrt(C)
        attn = torch.softmax(attn, dim=-1)
 
        # Weighted sum
        out = torch.bmm(v, attn.permute(0, 2, 1))   # (B, C, N)
        out = out.view(B, C, H, W)
        out = self.proj(out)
        return x + out

# Down block: pooling + conv, optionally with attention
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, use_attn=False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, emb_dim)
        self.use_attn = use_attn
        if use_attn:
            self.attn = SelfAttention(out_channels)
 
    def forward(self, x, emb):
        x = self.pool(x)
        x = self.conv(x, emb)
        if self.use_attn:
            x = self.attn(x)
        return x

# Up block: upsample, concat skip connection, conv, optionally with attention
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, bilinear=False, use_attn=False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(2 * out_channels, out_channels, emb_dim)
        self.use_attn = use_attn
        if use_attn:
            self.attn = SelfAttention(out_channels)
 
    def forward(self, x1, x2, emb):
        x1 = self.up(x1)
        # Pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
 
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, emb)
        if self.use_attn:
            x = self.attn(x)
        return x

# Final 1x1 conv for output
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)

# UNet with attention modules added in the encoder/decoder at lower resolutions.
class UNetCondAttn(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        emb_dim=128,
        bilinear=False
    ):
        super().__init__()
        self.time_label_emb = TimeLabelEmbedding(num_classes, embedding_dim=emb_dim)
 
        # Encoder: add attention in later stages (where feature maps are smaller)
        self.inc = DoubleConv(in_channels, base_channels, emb_dim)
        self.down1 = Down(base_channels, base_channels * 2, emb_dim, use_attn=False)
        self.down2 = Down(base_channels * 2, base_channels * 4, emb_dim, use_attn=True)  # attention added here
        self.down3 = Down(base_channels * 4, base_channels * 8, emb_dim, use_attn=True)
 
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, emb_dim, use_attn=True)
 
        # Decoder: mirror the encoder, adding attention blocks at matching resolutions
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, emb_dim, bilinear, use_attn=True)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, emb_dim, bilinear, use_attn=True)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, emb_dim, bilinear, use_attn=False)
        self.up4 = Up(base_channels * 2, base_channels, emb_dim, bilinear, use_attn=False)
        self.outc = OutConv(base_channels, out_channels)
 
    def forward(self, x, t, y):
        # Conditioning embedding
        emb = self.time_label_emb(t, y)
 
        # Encoder
        x1 = self.inc(x, emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)
        x5 = self.down4(x4, emb)
 
        # Decoder
        x = self.up1(x5, x4, emb)
        x = self.up2(x, x3, emb)
        x = self.up3(x, x2, emb)
        x = self.up4(x, x1, emb)
 
        logits = self.outc(x)
        return logits
