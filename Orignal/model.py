import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- NEW CUBE UTILITY ---
def compute_DHW(N, C):
    n = round(N ** (1/3))
    return n, n, N // (n * n)

# --- _MLP and DWConv ---
class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, D=None, H=None, W=None):
        x = self.fc1(x)
        if D is not None and H is not None and W is not None:
            x = self.dwconv(x, D, H, W)
        else:
            x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, D=None, H=None, W=None):
        B, N, C = x.shape
        if D is None or H is None or W is None:
            D, H, W = compute_DHW(N, C)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# --- PatchEmbedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, patch_size=7, stride=4, padding=3):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        D, H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (D, H, W)

# --- DeformableSelfAttention3D ---
class DeformableSelfAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, sr_ratio, offset_range_factor=2, qkv_bias=False, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.offset_range_factor = offset_range_factor

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

        self.offset_conv = nn.Conv3d(embed_dim, 3 * num_heads, kernel_size=3, stride=1, padding=1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, D, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_)
            D_, H_, W_ = x_.shape[2:]
            x_ = x_.flatten(2).transpose(1, 2)
            x_ = self.sr_norm(x_)
        else:
            x_ = x
            D_, H_, W_ = D, H, W

        kv = self.kv(x_)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        offset = self.offset_conv(x_.transpose(1, 2).reshape(B, C, D_, H_, W_))
        offset = offset.view(B, self.num_heads, 3, D_, H_, W_)
        offset = torch.tanh(offset) * self.offset_range_factor

        base_z = torch.linspace(-1, 1, steps=D_, device=x.device)
        base_y = torch.linspace(-1, 1, steps=H_, device=x.device)
        base_x = torch.linspace(-1, 1, steps=W_, device=x.device)
        grid_z, grid_y, grid_x = torch.meshgrid(base_z, base_y, base_x, indexing='ij')
        grid = torch.stack([grid_z, grid_y, grid_x], dim=0).unsqueeze(0).unsqueeze(0)

        sampling_grid = (grid + offset).permute(0, 1, 3, 4, 5, 2)  # (B, num_heads, D, H, W, 3)
        sampling_grid = sampling_grid.reshape(B * self.num_heads, D_, H_, W_, 3)

        x_k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, D_, H_, W_)
        x_v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, D_, H_, W_)

        sampled_k = F.grid_sample(x_k, sampling_grid, mode='bilinear', align_corners=True)
        sampled_v = F.grid_sample(x_v, sampling_grid, mode='bilinear', align_corners=True)

        sampled_k = sampled_k.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        sampled_v = sampled_v.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        attn = (q @ sampled_k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ sampled_v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out

# --- TransformerBlock3D ---
import torch.utils.checkpoint

class TransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, num_heads, sr_ratio, qkv_bias=False, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DeformableSelfAttention3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=proj_dropout)

    def forward(self, x, D, H, W):
        x = x + self.attn(self.norm1(x), D, H, W)
        x = x + self.mlp(self.norm2(x), D, H, W)
        return x





    





def cube_root(n):
    return round(n ** (1/3))

class MixVisionTransformer3D(nn.Module):
    def __init__(self, in_channels=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], depths=[2, 2, 2, 2], patch_kernel_sizes=[7, 3, 3, 3], patch_strides=[4, 2, 2, 2], patch_paddings=[3, 1, 1, 1]):
        super().__init__()

        self.patch_embed1 = PatchEmbedding(in_channels, embed_dims[0], patch_kernel_sizes[0], patch_strides[0], patch_paddings[0])
        self.block1 = nn.ModuleList([TransformerBlock3D(embed_dims[0], mlp_ratios[0], num_heads[0], sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.patch_embed2 = PatchEmbedding(embed_dims[0], embed_dims[1], patch_kernel_sizes[1], patch_strides[1], patch_paddings[1])
        self.block2 = nn.ModuleList([TransformerBlock3D(embed_dims[1], mlp_ratios[1], num_heads[1], sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.patch_embed3 = PatchEmbedding(embed_dims[1], embed_dims[2], patch_kernel_sizes[2], patch_strides[2], patch_paddings[2])
        self.block3 = nn.ModuleList([TransformerBlock3D(embed_dims[2], mlp_ratios[2], num_heads[2], sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.patch_embed4 = PatchEmbedding(embed_dims[2], embed_dims[3], patch_kernel_sizes[3], patch_strides[3], patch_paddings[3])
        self.block4 = nn.ModuleList([TransformerBlock3D(embed_dims[3], mlp_ratios[3], num_heads[3], sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        outs = []

        x, (D, H, W) = self.patch_embed1(x)
        B, N, C = x.shape
        for blk in self.block1:
            x = blk(x, D, H, W)
        x = self.norm1(x)
        x1 = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x1)

        x, (D, H, W) = self.patch_embed2(x1)
        B, N, C = x.shape
        for blk in self.block2:
            x = blk(x, D, H, W)
        x = self.norm2(x)
        x2 = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x2)

        x, (D, H, W) = self.patch_embed3(x2)
        B, N, C = x.shape
        for blk in self.block3:
            x = blk(x, D, H, W)
        x = self.norm3(x)
        x3 = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x3)

        x, (D, H, W) = self.patch_embed4(x3)
        B, N, C = x.shape
        for blk in self.block4:
            x = blk(x, D, H, W)
        x = self.norm4(x)
        x4 = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x4)

        return outs








import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock3D, self).__init__()
        self.attention = AttentionGate3D(in_channels, skip_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip):
        skip = self.attention(x, skip)
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class SegmentationHead3D(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=1):
        super(SegmentationHead3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return x


class DropoutSegmentationHead3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, scale_factor=1):
        super(DropoutSegmentationHead3D, self).__init__()
        self.dropout = nn.Dropout3d(dropout_rate)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return x


class RefineFormerDecoderHead3D(nn.Module):
    def __init__(self,
                 encoder_channels=[256, 160, 64, 32],
                 decoder_channels=[256, 128, 64, 32],
                 num_classes=3,
                 dropout_rate=0.5):
        super(RefineFormerDecoderHead3D, self).__init__()

        self.decode4 = DecoderBlock3D(encoder_channels[0], encoder_channels[1], decoder_channels[0])
        self.decode3 = DecoderBlock3D(decoder_channels[0], encoder_channels[2], decoder_channels[1])
        self.decode2 = DecoderBlock3D(decoder_channels[1], encoder_channels[3], decoder_channels[2])

        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm3d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.segmentation_head = SegmentationHead3D(decoder_channels[3], num_classes, scale_factor=2)
        self.aux_head4 = SegmentationHead3D(decoder_channels[0], num_classes, scale_factor=8)
        self.aux_head3 = SegmentationHead3D(decoder_channels[1], num_classes, scale_factor=4)
        self.aux_head2 = SegmentationHead3D(decoder_channels[2], num_classes, scale_factor=2)
        self.uncertainty_head = DropoutSegmentationHead3D(decoder_channels[3], num_classes, dropout_rate, scale_factor=2)

    def forward(self, c1, c2, c3, c4):
        x = self.decode4(c4, c3)
        aux_out4 = self.aux_head4(x)

        x = self.decode3(x, c2)
        aux_out3 = self.aux_head3(x)

        x = self.decode2(x, c1)
        aux_out2 = self.aux_head2(x)

        x = self.decode1(x)

        main_out = self.segmentation_head(x)
        uncertainty_out = self.uncertainty_head(x)

        return main_out, aux_out2, aux_out3, aux_out4, uncertainty_out








class RefineFormer3D(nn.Module):
    def __init__(self,
                 in_channels=4,
                 num_classes=3,
                 embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[4, 2, 1, 1],
                 depths=[2, 2, 2, 2],
                 decoder_channels=[256, 128, 64, 32],
                 dropout_rate=0.5):
        """
        Full RefineFormer3D model
        """
        super(RefineFormer3D, self).__init__()

        # Encoder: Hierarchical Transformer with Deformable Attention
        self.encoder = MixVisionTransformer3D(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            depths=depths,
            patch_kernel_sizes=[7, 3, 3, 3],
            patch_strides=[4, 2, 2, 2],
            patch_paddings=[3, 1, 1, 1],
        )

        # Decoder: Hybrid Decoder with Attention Gating + Conv3D
        self.decoder = RefineFormerDecoderHead3D(
            encoder_channels=embed_dims[::-1],  # c4, c3, c2, c1
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        x: [B, in_channels, D, H, W]
        """
        features = self.encoder(x)  # c1, c2, c3, c4

        c1 = features[0]  # High resolution
        c2 = features[1]
        c3 = features[2]
        c4 = features[3]  # Low resolution, deep features

        main_out, aux_out2, aux_out3, aux_out4, uncertainty_out = self.decoder(c1, c2, c3, c4)

        return {
            "main": main_out,
            "aux2": aux_out2,
            "aux3": aux_out3,
            "aux4": aux_out4,
            "uncertainty": uncertainty_out,
        }