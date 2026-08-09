import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = attn + rel_pos_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        self._attn_mask = None
        self._mask_shape = None

    def _compute_attn_mask(self, H: int, W: int, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** 2, C)

        if self._mask_shape != (H, W):
            self._attn_mask = self._compute_attn_mask(H, W, x.device)
            self._mask_shape = (H, W)

        attn_windows = self.attn(x_windows, mask=self._attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 8,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        residual = x
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.norm(x)

        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)

        return x + residual


class PixelShuffleUpsampler(nn.Module):

    def __init__(self, dim: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.refine = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.refine(self.pixel_shuffle(self.conv(x)))


class SwinIR(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        upscale: int = 1,
        img_range: float = 1.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        depths = depths or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]

        self.img_range = img_range
        self.upscale = upscale
        self.window_size = window_size

        if in_channels == 3:
            self.register_buffer(
                "mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
            )
        else:
            self.register_buffer("mean", torch.zeros(1, in_channels, 1, 1))

        self.shallow_feat = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.rstb_blocks = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for i in range(len(depths))
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.deep_feat_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        if upscale == 1:
            self.upsample = None
            self.reconstruction = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = PixelShuffleUpsampler(embed_dim, upscale)
            self.reconstruction = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pad(self, x: torch.Tensor):
        _, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    @staticmethod
    def _unpad(x: torch.Tensor, H: int, W: int):
        return x[:, :, :H, :W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean * self.img_range) / self.img_range

        x, H_orig, W_orig = self._pad(x)
        _, _, H, W = x.shape

        feat = self.shallow_feat(x)
        feat_shallow = feat

        feat_seq = feat.flatten(2).transpose(1, 2)
        for rstb in self.rstb_blocks:
            feat_seq = rstb(feat_seq, H, W)
        feat_seq = self.norm(feat_seq)

        feat = feat_seq.transpose(1, 2).view(-1, feat.shape[1], H, W)
        feat = self.deep_feat_conv(feat) + feat_shallow

        if self.upsample is not None:
            feat = self.upsample(feat)
        out = self.reconstruction(feat)

        out = out * self.img_range + self.mean * self.img_range

        if self.upscale == 1:
            out = self._unpad(out, H_orig, W_orig)
        else:
            out = self._unpad(out, H_orig * self.upscale, W_orig * self.upscale)

        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_swinir(variant: str = "small", **kwargs) -> SwinIR:
    presets = {
        "tiny": dict(embed_dim=60,  depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4]),
        "small": dict(embed_dim=96,  depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6]),
        "large": dict(embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6]),
    }
    assert variant in presets, f"Unknown variant '{variant}'. Choose from {list(presets)}."
    cfg = {**presets[variant], **kwargs}
    return SwinIR(**cfg)


if __name__ == "__main__":
    model = build_swinir("small", upscale=1)
    print(f"SwinIR-Small  params: {model.parameter_count() / 1e6:.2f} M")
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input : {dummy.shape} -> Output: {out.shape}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("Smoke-test passed")
