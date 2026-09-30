"""Convolutional and attention building blocks.

The designs follow the published YOLO11/YOLO26 (C3k2, SPPF, position-sensitive attention) and YOLOv12 (area attention)
architectures; this is an independent implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """'Same' padding for an odd kernel."""
    if p is None:
        p = d * (k - 1) // 2
    return p


class Conv(nn.Module):
    """Conv2d -> BatchNorm -> SiLU. BN folds into the conv at export time via `fuse()`."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, d: int = 1,
                 act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self) -> None:
        """Fold BatchNorm statistics into the convolution weights."""
        w = self.conv.weight
        std = (self.bn.running_var + self.bn.eps).sqrt()
        scale = self.bn.weight / std
        fused = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size, self.conv.stride,
                          self.conv.padding, self.conv.dilation, self.conv.groups, bias=True).to(w.device, w.dtype)
        fused.weight.copy_(w * scale.view(-1, 1, 1, 1))
        fused.bias.copy_(self.bn.bias - self.bn.running_mean * scale)
        self.conv = fused
        del self.bn
        self.forward = self.forward_fused


class DWConv(Conv):
    """Depth-wise convolution (groups = gcd(c1, c2))."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    """Two convolutions with an optional residual connection."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3),
                 e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """CSP block with `n` bottlenecks of kernel `k` (a deeper inner unit for C3k2)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class MHSA(nn.Module):
    """Global multi-head self-attention on a feature map with a depth-wise conv positional term on V.

    Keys/queries use a reduced dimension (attn_ratio * head_dim), values keep the full head dimension.
    With `area > 1`, the sequence is split into `area` contiguous stripes that attend only within themselves
    (YOLOv12 "area attention"), which cuts the quadratic cost by a factor of `area`.
    """

    def __init__(self, dim: int, num_heads: int = 4, attn_ratio: float = 0.5, area: int = 1, pe_kernel: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim**-0.5
        self.area = area
        self.qkv = Conv(dim, dim + 2 * self.key_dim * num_heads, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, pe_kernel, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, self.num_heads, 2 * self.key_dim + self.head_dim, N)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.head_dim), dim=2)  # (B, h, d, N)
        a = self.area if (self.area > 1 and N % self.area == 0) else 1
        if a > 1:  # fold stripes into the batch dimension; row-major flattening makes stripes contiguous rows
            q, k, v = (t.reshape(B, self.num_heads, t.shape[2], a, N // a).permute(0, 3, 1, 2, 4)
                       .reshape(B * a, self.num_heads, t.shape[2], N // a) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1),
                                             scale=self.scale)  # (B*a, h, N/a, d)
        out = out.transpose(-2, -1)  # (B*a, h, d, N/a)
        if a > 1:
            out = out.reshape(B, a, self.num_heads, self.head_dim, N // a).permute(0, 2, 3, 1, 4)
        out = out.reshape(B, C, H, W)
        v_map = v.reshape(B, a, self.num_heads, self.head_dim, N // a).permute(0, 2, 3, 1, 4) if a > 1 else v
        return self.proj(out + self.pe(v_map.reshape(B, C, H, W)))


class AttnBlock(nn.Module):
    """Pre-activation-free transformer block: x + MHSA(x); x + FFN(x) with 1x1 convs."""

    def __init__(self, c: int, num_heads: int, attn_ratio: float = 0.5, mlp_ratio: float = 2.0, area: int = 1):
        super().__init__()
        self.attn = MHSA(c, num_heads, attn_ratio, area)
        self.ffn = nn.Sequential(Conv(c, int(c * mlp_ratio), 1), Conv(int(c * mlp_ratio), c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C3k2(nn.Module):
    """CSP bottleneck with two convolutions (C2f topology).

    Inner units are plain Bottlenecks, C3k blocks (`c3k=True`) or Bottleneck+attention (`attn=True`).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, attn: bool = False,
                 area: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        heads = max(self.c // 64, 1)

        def unit():
            if attn:
                return nn.Sequential(Bottleneck(self.c, self.c, shortcut), AttnBlock(self.c, heads, area=area))
            if c3k:
                return C3k(self.c, self.c, 2, shortcut)
            return Bottleneck(self.c, self.c, shortcut)

        self.m = nn.ModuleList(unit() for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling (fast): repeated max-pools concatenated, optional residual."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = True):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return x + y if self.add else y


class C2PSA(nn.Module):
    """CSP wrapper around `n` attention blocks on half the channels (position-sensitive attention)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, area: int = 1):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(AttnBlock(self.c, max(self.c // 64, 1), area=area) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class SPDConv(nn.Module):
    """Space-to-depth downsampling (lossless 2x2 pixel unshuffle) followed by a 1x1 projection.

    Unlike a strided conv it keeps every input pixel, which helps preserve small-object detail when a
    high-resolution map is fused into a coarser level.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv = Conv(4 * c1, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv(F.pixel_unshuffle(x, 2))
