import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Efficient Self-Attention Block
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        out = attn_out.transpose(1, 2).view(B, C, H, W)
        return out + x  # Residual

# Cross-Attention with embedding
class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, emb_dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(emb_dim, channels)
        self.v_proj = nn.Linear(emb_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x, cond_emb):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        q = self.q_proj(self.norm(x_flat))
        k = self.k_proj(cond_emb).unsqueeze(1)  # (B, 1, C)
        v = self.v_proj(cond_emb).unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        out = attn_out.transpose(1, 2).view(B, C, H, W)
        return out + x

class DoubleConvAttn(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, use_self_attn=False, use_cross_attn=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )
        self.cross_attn = CrossAttentionBlock(out_channels, emb_dim) if use_cross_attn else None
        self.self_attn = SelfAttentionBlock(out_channels) if use_self_attn else None

    def forward(self, x, emb):
        x = self.conv(x)
        if self.cross_attn:
            x = self.cross_attn(x, emb)
        if self.self_attn:
            x = self.self_attn(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, use_self_attn=False, use_cross_attn=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConvAttn(in_channels, out_channels, emb_dim, use_self_attn, use_cross_attn)

    def forward(self, x, emb):
        x = self.pool(x)
        return self.block(x, emb)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, bilinear=False, use_self_attn=False, use_cross_attn=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.block = DoubleConvAttn(out_channels * 2, out_channels, emb_dim, use_self_attn, use_cross_attn)

    def forward(self, x1, x2, emb):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x, emb)

class UNetCondAttn(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, out_channels=3, base_channels=64, emb_dim=128, bilinear=False):
        super().__init__()
        self.time_label_emb = TimeLabelEmbedding(num_classes, embedding_dim=emb_dim)

        self.inc = DoubleConvAttn(in_channels, base_channels, emb_dim)
        self.down1 = Down(base_channels, base_channels * 2, emb_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, emb_dim, use_self_attn=True, use_cross_attn=True)
        self.down3 = Down(base_channels * 4, base_channels * 8, emb_dim, use_self_attn=True, use_cross_attn=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, emb_dim, use_self_attn=True, use_cross_attn=True)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, emb_dim, bilinear, use_self_attn=True, use_cross_attn=True)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, emb_dim, bilinear, use_self_attn=True, use_cross_attn=True)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, emb_dim, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, emb_dim, bilinear)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t, y):
        emb = self.time_label_emb(t, y)
        x1 = self.inc(x, emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)
        x5 = self.down4(x4, emb)
        x = self.up1(x5, x4, emb)
        x = self.up2(x, x3, emb)
        x = self.up3(x, x2, emb)
        x = self.up4(x, x1, emb)
        return self.outc(x)
