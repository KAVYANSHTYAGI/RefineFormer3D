import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, GroupNorm
from typing import List, Tuple

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim, shape):
        super().__init__()
        D, H, W = shape
        self.pos_table = nn.Parameter(torch.zeros(1, embed_dim, D, H, W))
        nn.init.trunc_normal_(self.pos_table, std=0.02)

    def forward(self, x):
        return x + self.pos_table

# -----------------------------
# DropPath with residual scaling
# -----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_residual=1.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_residual = scale_residual

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x * self.scale_residual
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor * self.scale_residual



# -----------------------------
# Dynamic GroupNorm helper
# -----------------------------
def dynamic_group_norm(channels, groups=16):
    return GroupNorm(min(groups, channels), channels)

# -----------------------------
# Placeholder for window_partition and reverse (will be completed with full encoder)
# -----------------------------
def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    x = x.view(B,
               D // Wd, Wd,
               H // Wh, Wh,
               W // Ww, Ww,
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, Wd, Wh, Ww, C)
    return windows

def window_reverse(windows, window_size, D, H, W):
    Wd, Wh, Ww = window_size
    B = int(windows.shape[0] / (D // Wd * H // Wh * W // Ww))
    x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_attention_mask(D, H, W, window_size, shift_size, device, B):
    img_mask = torch.zeros((B, D, H, W, 1), device=device)
    cnt = 0
    for d in range(0, D, window_size[0]):
        for h in range(0, H, window_size[1]):
            for w in range(0, W, window_size[2]):
                img_mask[:, d:d+window_size[0], h:h+window_size[1], w:w+window_size[2], :] = cnt
                cnt += 1

    if any(s > 0 for s in shift_size):
        img_mask = torch.roll(img_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask




class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        relative_coords = self.compute_relative_coords()
        self.register_buffer("relative_position_index", relative_coords)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def compute_relative_coords(self):
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords += torch.tensor([(self.window_size[0] - 1),
                                         (self.window_size[1] - 1),
                                         (self.window_size[2] - 1)])
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        ).permute(2, 0, 1).unsqueeze(0)

        attn = attn + relative_bias

        if mask is not None:
            attn_mask, nW = mask
            B_ = x.shape[0]
            if B_ % nW != 0:
                raise RuntimeError(f"[WindowAttention3D] Mismatch: B_={B_}, nW={nW}, expected B_ % nW == 0")
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            if attn.shape[1] != attn_mask.shape[0]:
                raise RuntimeError(
                    f"[WindowAttention3D] Shape mismatch in attn_mask:attn shape={attn.shape}, attn_mask shape={attn_mask.shape}, B_={B_}, nW={nW}"
                )
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)










# (previous code remains unchanged)

# -----------------------------
# Feedforward Network (Mix-FFN with Depthwise Conv3D)
# -----------------------------
class MixFFN3D(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

# -----------------------------
# Transformer Block for Encoder
# -----------------------------
class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path, scale_residual=0.5)
        self.norm2 = LayerNorm(dim)
        self.mlp = MixFFN3D(dim, int(dim * mlp_ratio), drop)

        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)

        # Disable shifted windows and masking
        # (we skip torch.roll and mask generation)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            C
        )

        # Run attention without any mask
        attn_windows = self.attn(x_windows, mask=None)

        x = window_reverse(
            attn_windows.view(-1, *self.window_size, C),
            self.window_size, D, H, W
        )

        x = x.view(B, N, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x


# -----------------------------
# Encoder Backbone (Patch Embedding + Positional Encoding + Transformer Stages)
# -----------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, D, H, W]
        D, H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return self.norm(x), (D, H, W)

class HierarchicalEncoder3D(nn.Module):
    def __init__(self, in_ch, embed_dims, depths, num_heads, window_sizes, mlp_ratios,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        self.stages = nn.ModuleList()
        self.embeds = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(embed_dims)):
            embed = PatchEmbed3D(in_ch if i == 0 else embed_dims[i-1], embed_dims[i])
            blocks = nn.ModuleList([
                TransformerBlock3D(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    shift_size=(0, 0, 0) if j % 2 == 0 else tuple(w // 2 for w in window_sizes[i]),
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j]
                ) for j in range(depths[i])
            ])
            cur += depths[i]
            self.embeds.append(embed)
            self.stages.append(blocks)
            self.norms.append(LayerNorm(embed_dims[i]))

    def forward(self, x):
        features = []
        for embed, blocks, norm in zip(self.embeds, self.stages, self.norms):
            x, (D, H, W) = embed(x)
            for blk in blocks:
                x = blk(x, D, H, W)
            feat = norm(x).view(-1, D, H, W, x.shape[-1]).permute(0, 4, 1, 2, 3).contiguous()
            features.append(feat)
            x = feat
        return features


# -----------------------------
# Decoder Block with Attention Fusion
# -----------------------------
class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.attn_fuse = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 1),
            dynamic_group_norm(out_ch),
            nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch),
            dynamic_group_norm(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=1)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.attn_fuse(x)
        return self.conv(x)

# -----------------------------
# Refined Decoder Head
# -----------------------------
class RefinedDecoderHead3D(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes):
        super().__init__()
        self.decode3 = DecoderBlock3D(encoder_channels[3], encoder_channels[2], decoder_channels[0])
        self.decode2 = DecoderBlock3D(decoder_channels[0], encoder_channels[1], decoder_channels[1])
        self.decode1 = DecoderBlock3D(decoder_channels[1], encoder_channels[0], decoder_channels[2])
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1),
            dynamic_group_norm(decoder_channels[3]),
            nn.GELU()
        )
        self.seg_head = nn.Conv3d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        x = self.decode3(c4, c3)
        x = self.decode2(x, c2)
        x = self.decode1(x, c1)
        x = self.final_up(x)
        return {"main": self.seg_head(x)}

# -----------------------------
# Full Model Wrapper
# -----------------------------
class RefineFormer3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=3,
                 embed_dims=[64, 128, 320, 512],
                 depths=[2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 window_sizes=[(4, 4, 4), (2, 4, 4), (2, 2, 2), (1, 2, 2)],
                 mlp_ratios=[4, 4, 4, 4],
                 decoder_channels=[256, 128, 64, 32],
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.encoder = HierarchicalEncoder3D(
            in_ch=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_sizes=window_sizes,
            mlp_ratios=mlp_ratios,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        self.decoder = RefinedDecoderHead3D(
            encoder_channels=embed_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)

    def exportable_forward(self, x):
        with torch.no_grad():
            return self.forward(x)
