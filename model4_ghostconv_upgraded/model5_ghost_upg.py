import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, GroupNorm
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint

# ------------------------------------
# Replace the existing CrossFuse class
# ------------------------------------
class CrossFuse(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, ratio=2):
        super().__init__()
        # 1×1 GhostConv fusion (cheap!)
        self.fuse = GhostConv3D(in_ch + skip_ch, out_ch, ratio=ratio, kernel_size=1)
        # Normalization + activation
        self.norm = nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch)
        self.act  = nn.GELU()
        # Squeeze-and-Excitation for lightweight channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                    # → [B, out_ch, 1,1,1]
            nn.Conv3d(out_ch, out_ch // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(out_ch // 4, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        # x: [B, C1, D, H, W], skip: [B, C2, D, H, W]
        x = torch.cat([x, skip], dim=1)  # [B, C1+C2, D, H, W]
        x = self.fuse(x)                 # [B, out_ch, D, H, W]
        x = self.act(self.norm(x))       # normalize + non-linear

        # channel attention
        weight = self.se(x)              # [B, out_ch, 1,1,1]
        return x * weight                # re-scale channels



# -----------------------------
# DropPath (stochastic depth)
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
# Ghost Conv3D (GhostNet-style)
# -----------------------------
class GhostConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, dw_kernel_size=3, stride=1, padding=0):
        super().__init__()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Conv3d(in_channels, init_channels, kernel_size, stride, padding, bias=False)
        self.cheap_operation = nn.Conv3d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2,
                                        groups=init_channels, bias=False)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

# -----------------------------
# Low-Rank Linear Approximation
# -----------------------------
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.B(self.A(x))

# -----------------------------
# Dynamic GroupNorm helper
# -----------------------------
def dynamic_group_norm(channels, groups=16):
    # For conv outputs [B, C, D, H, W], use GroupNorm
    num_groups = min(groups, channels)
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


# -----------------------------
# Window partition / reverse
# -----------------------------
def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    x = x.view(B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C)
    windows = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, Wd, Wh, Ww, C)
    return windows

def window_reverse(windows, window_size, D, H, W):
    Wd, Wh, Ww = window_size
    B = int(windows.shape[0] / (D//Wd * H//Wh * W//Ww))
    x = windows.view(B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, -1)
    x = x.permute(0,1,4,2,5,3,6,7).contiguous().view(B, D, H, W, -1)
    return x

# -----------------------------
# Attention Mask
# -----------------------------
def get_attention_mask(D, H, W, window_size, shift_size, device, B):
    # 1) Create a numbering mask over the 3D volume
    img_mask = torch.zeros((B, D, H, W, 1), device=device)
    cnt = 0
    for d in range(0, D, window_size[0]):
        for h in range(0, H, window_size[1]):
            for w in range(0, W, window_size[2]):
                img_mask[:, d:d+window_size[0],
                            h:h+window_size[1],
                            w:w+window_size[2], :] = cnt
                cnt += 1
    # 2) Shift if needed
    if any(s > 0 for s in shift_size):
        img_mask = torch.roll(img_mask,
                              shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                              dims=(1, 2, 3))

    # 3) Partition into windows
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)  # [B·num_windows, Wd,Wh,Ww]

    # 4) Flatten each window to a vector of length N
    N = window_size[0] * window_size[1] * window_size[2]
    mask_windows = mask_windows.view(-1, N)  # [B·num_windows, N]

    # 5) Build the N×N mask per window
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)  # [B·num_windows, N, N]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                         .masked_fill(attn_mask == 0, float(0.0))
    return attn_mask  # shape: [B·num_windows, N, N]


# -----------------------------
# WindowAttention3D
# -----------------------------
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (D, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias indexing
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size[0]), torch.arange(window_size[1]), torch.arange(window_size[2]), indexing='ij'
        ))  # [3, D, H, W]
        coords_flat = coords.flatten(1)  # [3, N]
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # [3, N, N]
        relative = relative.permute(1, 2, 0).contiguous()  # [N, N, 3]
        relative += torch.tensor([
            window_size[0] - 1,
            window_size[1] - 1,
            window_size[2] - 1
        ])
        relative[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative[:, :, 1] *= (2 * window_size[2] - 1)
        index = relative.sum(-1)  # [N, N]
        self.register_buffer('rel_idx', index)

        # Learnable bias for each relative index
        self.relative_bias = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1), num_heads)
        )

    def forward(self, x, mask=None):
        B_, N, C = x.shape  # B_ = B * num_windows, N = window_size^3

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, heads, N, head_dim]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, heads, N, N]

        # Add relative position bias
        bias = self.relative_bias[self.rel_idx.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias.to(attn.dtype)

        if mask is not None:
            # attn: [B·num_windows, heads, N, N]
            # mask: [B·num_windows, N, N]
            attn = attn + mask.unsqueeze(1)  # -> [B·num_windows, heads, N, N]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


# -----------------------------
# Mix-FFN with Low-Rank
# -----------------------------
class MixFFN3D(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        rank1 = max(hidden // 2, 64)  # Ensure minimum rank capacity
        rank2 = max(dim // 2, 64)

        self.fc1 = LowRankLinear(dim, hidden, rank=rank1)
        self.dw = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)

        self.fc2 = LowRankLinear(hidden, dim, rank=rank2)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dw(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)

# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.1):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn  = WindowAttention3D(dim, window_size, num_heads, qkv_bias=True,
                                      attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path==0 else DropPath(drop_path)
        self.norm2 = LayerNorm(dim)
        self.mlp   = MixFFN3D(dim, mlp_ratio, drop)
        self.window_size, self.shift_size = window_size, shift_size

    def forward(self, x, D, H, W):
        B,N,C = x.shape
        shortcut = x
        x = self.norm1(x).view(B,D,H,W,C)
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size[0]*self.window_size[1]*self.window_size[2], C)
        # compute & cache mask once per shape
        B = x.shape[0]

        if not hasattr(self, 'attn_mask') or self.attn_mask_shape != (B, D, H, W):
            self.attn_mask = get_attention_mask(D, H, W, self.window_size, self.shift_size, x.device, B)
            self.attn_mask_shape = (B, D, H, W)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        x = window_reverse(attn_windows.view(-1,*self.window_size,C), self.window_size,D,H,W)
        x = x.view(B,N,C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x

# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        # Ghost-based convolutional patch embedding (stride=2)
        self.proj = GhostConv3D(in_ch, embed_dim, ratio=2, kernel_size=3, stride=2, padding=1)
        self.norm = LayerNorm(embed_dim)
        self.pos_conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)

    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.proj(x)  # [B, embed_dim, D/2, H/2, W/2]
        B, C, D, H, W = x.shape

        # Apply positional encoding in 3D space
        x = self.pos_conv(x)  # [B, C, D, H, W]

        # Flatten for transformer input
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)

        return x, (D, H, W)

class HierarchicalEncoder3D(nn.Module):
    def __init__(self, in_ch, embed_dims: List[int], depths: List[int],
                 num_heads: List[int], window_sizes: List[Tuple[int,int,int]], mlp_ratios: List[float],
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        total_blocks = sum(depths)
        dpr = list(torch.linspace(0, drop_path_rate, total_blocks))
        cur = 0
        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for i, dim in enumerate(embed_dims):
            in_channels = in_ch if i == 0 else embed_dims[i-1]
            # patch embedding
            self.embeds.append(PatchEmbed3D(in_channels, dim))
            # transformer blocks
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                shift = tuple(s//2 for s in window_sizes[i]) if (j % 2 == 1) else (0,0,0)
                blocks.append(TransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    shift_size=shift,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur+j]
                ))
            self.stages.append(blocks)
            self.norms.append(LayerNorm(dim))
            cur += depths[i]

    def forward(self, x):
        B = x.shape[0]
        features = []

        for embed, blocks, norm in zip(self.embeds, self.stages, self.norms):
            # Patch embed → (B, N, C) and spatial dims
            x, (D, H, W) = embed(x)

            # Transformer blocks with checkpointing
            for blk in blocks:
                x = checkpoint(blk, x, D, H, W)

            # Normalize then reshape back to 3D volume
            x_norm = norm(x)                                        # [B, N, C]
            feat = x_norm.view(B, D, H, W, x_norm.shape[-1])        # [B, D, H, W, C]
            feat = feat.permute(0, 4, 1, 2, 3).contiguous()         # [B, C, D, H, W]

            features.append(feat)
            x = feat  # pass to next stage

        return features
    

class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.attn_fuse = CrossFuse(in_ch, skip_ch, out_ch)

        self.conv = nn.Sequential(
            GhostConv3D(out_ch, out_ch, ratio=2, kernel_size=3, padding=1),
            dynamic_group_norm(out_ch),
            nn.GELU()
        )

    def forward(self, x, skip):
        # upsample + skip connect
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = self.attn_fuse(x, skip)
        return self.conv(x)
    

class RefinedDecoderHead3D(nn.Module):
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int], num_classes: int):
        super().__init__()
        # decode3: in=encoder[3], skip=encoder[2], out=decoder[0]
        self.decode3 = DecoderBlock3D(encoder_channels[3], encoder_channels[2], decoder_channels[0])
        # decode2: in=decoder[0], skip=encoder[1], out=decoder[1]
        self.decode2 = DecoderBlock3D(decoder_channels[0], encoder_channels[1], decoder_channels[1])
        # decode1: in=decoder[1], skip=encoder[0], out=decoder[2]
        self.decode1 = DecoderBlock3D(decoder_channels[1], encoder_channels[0], decoder_channels[2])

        # AUX heads must match the *output* channels of decode2 and decode1
        self.aux2 = nn.Conv3d(decoder_channels[1], num_classes, kernel_size=1)
        self.aux3 = nn.Conv3d(decoder_channels[2], num_classes, kernel_size=1)

        # final up‐sampling + conv to decoder_channels[3]
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            GhostConv3D(decoder_channels[2], decoder_channels[3], ratio=2, kernel_size=3, padding=1),
            dynamic_group_norm(decoder_channels[3]),
            nn.GELU()
        )
        self.seg_head = nn.Conv3d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features

        # Stage 1
        x = self.decode3(c4, c3)         # → [B, decoder_channels[0], D, H, W]
        # Stage 2
        x = self.decode2(x, c2)          # → [B, decoder_channels[1], D, H, W]
        aux2 = self.aux2(x)              # AUX head on decode2 output

        # Stage 3
        x = self.decode1(x, c1)          # → [B, decoder_channels[2], D, H, W]
        aux3 = self.aux3(x)              # AUX head on decode1 output

        # Final upsample + main head
        x = self.final_up(x)             # → [B, decoder_channels[3], 2D, 2H, 2W]
        main_out = self.seg_head(x)      # → [B, num_classes, 2D, 2H, 2W]

        return {
            "main": main_out,
            "aux2": aux2,
            "aux3": aux3
        }


class RefineFormer3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=3,
                 embed_dims=[64,128,320,512],
                 depths=[2,2,2,2],
                 num_heads=[1,2,4,8],
                 window_sizes=[(4,4,4),(2,4,4),(2,2,2),(1,2,2)],
                 mlp_ratios=[4,4,4,4],
                 decoder_channels=[256,128,64,32],
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
