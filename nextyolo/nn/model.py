"""nextYOLO model: spec-driven backbone + PAN neck + NMS-free dual-assignment head."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .blocks import C2PSA, MHSA, C3k2, SPPF, Conv, SPDConv
from .head import DetectHead

# (depth multiple, width multiple, max channels) — same compound scaling as YOLO11/YOLO26.
SCALES = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


@dataclass
class ModelConfig:
    nc: int = 80
    scale: str = "n"
    end2end: bool = True          # NMS-free o2o head (training uses both branches)
    levels: tuple = (3, 4, 5)     # detection levels; (3, 5) = YOLO27-n style dual-scale head
    p2_fusion: bool = False       # fuse a lossless space-to-depth P2 map into the P3 neck node
    attn_area: int = 1            # area-attention stripes in the P5 attention blocks (1 = global)
    o2o_grad_scale: float = 0.0   # 0 = o2o branch fully detached from the backbone (YOLOv10/26)
    max_det: int = 300
    extra: dict = field(default_factory=dict)


def _divisible(x: float, d: int = 8) -> int:
    return int(math.ceil(x / d) * d)


def build_spec(cfg: ModelConfig) -> list:
    """Layer spec: (from, repeats, type, args). Channels are pre-scaling values (as in YOLO YAMLs)."""
    a = cfg.attn_area
    spec = [
        # backbone
        (-1, 1, "Conv", [64, 3, 2]),                  # 0 P1/2
        (-1, 1, "Conv", [128, 3, 2]),                 # 1 P2/4
        (-1, 2, "C3k2", [256, False, 0.25]),          # 2
        (-1, 1, "Conv", [256, 3, 2]),                 # 3 P3/8
        (-1, 2, "C3k2", [512, False, 0.25]),          # 4
        (-1, 1, "Conv", [512, 3, 2]),                 # 5 P4/16
        (-1, 2, "C3k2", [512, True]),                 # 6
        (-1, 1, "Conv", [1024, 3, 2]),                # 7 P5/32
        (-1, 2, "C3k2", [1024, True]),                # 8
        (-1, 1, "SPPF", [1024, 5, 3, True]),          # 9
        (-1, 2, "C2PSA", [1024, a]),                  # 10
        # neck (top-down)
        (-1, 1, "Upsample", [2]),                     # 11
        ([-1, 6], 1, "Concat", []),                   # 12
        (-1, 2, "C3k2", [512, True]),                 # 13
        (-1, 1, "Upsample", [2]),                     # 14
        ([-1, 4], 1, "Concat", []),                   # 15
        (-1, 2, "C3k2", [256, True]),                 # 16 P3 out
        # neck (bottom-up)
        (-1, 1, "Conv", [256, 3, 2]),                 # 17
        ([-1, 13], 1, "Concat", []),                  # 18
        (-1, 2, "C3k2", [512, True]),                 # 19 P4 out
        (-1, 1, "Conv", [512, 3, 2]),                 # 20
        ([-1, 10], 1, "Concat", []),                  # 21
        (-1, 1, "C3k2", [1024, True, 0.5, True, a]),  # 22 P5 out
    ]
    outs = {3: 16, 4: 19, 5: 22}
    if cfg.p2_fusion:
        # Append an SPD branch from the P2 stage (layer 2) and fold it into the P3 concat (layer 15).
        spec.append((2, 1, "SPDConv", [256]))         # 23
        spec[15] = ([-1, 4, 23], 1, "Concat", [])
    spec.append(([outs[l] for l in cfg.levels], 1, "Detect", []))
    return spec


class Concat(nn.Module):
    def forward(self, xs):
        return torch.cat(xs, 1)


class NextYOLO(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or ModelConfig(**kw)
        self.cfg = cfg
        depth, width, max_ch = SCALES[cfg.scale]
        spec = build_spec(cfg)
        # Layers may reference later indices (p2_fusion appends a branch consumed earlier), so build in a
        # dependency-respecting order and record execution order separately.
        order = self._topo_order(spec)
        ch: dict[int, int] = {}
        layers: dict[int, nn.Module] = {}
        for i in order:
            f, n, t, args = spec[i]
            n = max(round(n * depth), 1) if n > 1 else n
            c_in = 3 if i == 0 else ch[i - 1] if f == -1 else (ch[f] if isinstance(f, int) else None)
            srcs = [i - 1 if x == -1 else x for x in f] if isinstance(f, list) else None
            if t == "Conv":
                c2 = _divisible(min(args[0], max_ch) * width)
                m = Conv(c_in, c2, *args[1:])
            elif t == "C3k2":
                c2 = _divisible(min(args[0], max_ch) * width)
                c3k = args[1] if len(args) > 1 else False
                c3k = True if cfg.scale in ("m", "l", "x") else c3k
                e = args[2] if len(args) > 2 else 0.5
                attn = args[3] if len(args) > 3 else False
                area = args[4] if len(args) > 4 else 1
                m = C3k2(c_in, c2, n, c3k, e, attn, area)
            elif t == "SPPF":
                c2 = _divisible(min(args[0], max_ch) * width)
                m = SPPF(c_in, c2, *args[1:])
            elif t == "C2PSA":
                c2 = _divisible(min(args[0], max_ch) * width)
                m = C2PSA(c_in, c2, n, area=args[1] if len(args) > 1 else 1)
            elif t == "SPDConv":
                c2 = _divisible(min(args[0], max_ch) * width)
                m = SPDConv(c_in, c2)
            elif t == "Upsample":
                c2 = c_in
                m = nn.Upsample(scale_factor=args[0], mode="nearest")
            elif t == "Concat":
                c2 = sum(ch[s] for s in srcs)
                m = Concat()
            elif t == "Detect":
                c2 = 0
                m = DetectHead(cfg.nc, [ch[s] for s in srcs], cfg.end2end, cfg.max_det, cfg.o2o_grad_scale)
            else:
                raise ValueError(t)
            m.f = srcs if srcs is not None else (i - 1 if f == -1 else f)
            m.i = i
            ch[i] = c2
            layers[i] = m
        self.layers = nn.ModuleList(layers[i] for i in range(len(spec)))
        self.order = order
        self.save = {x for m in self.layers for x in (m.f if isinstance(m.f, list) else [m.f])}
        self.head: DetectHead = self.layers[-1]
        self._init_weights()
        self._init_strides()
        self.head.bias_init()

    @staticmethod
    def _topo_order(spec) -> list[int]:
        deps = {}
        for i, (f, *_rest) in enumerate(spec):
            fs = f if isinstance(f, list) else [f]
            deps[i] = {i - 1 if x == -1 else x for x in fs if not (i == 0 and x == -1)}
        order, done = [], set()
        while len(order) < len(spec):
            for i in range(len(spec)):
                if i not in done and deps[i] <= done:
                    order.append(i)
                    done.add(i)
                    break
        return order

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03

    @torch.no_grad()
    def _init_strides(self, s: int = 256):
        was_training = self.training
        self.train()  # training path returns feature shapes without decoding
        out = self.forward(torch.zeros(1, 3, s, s))
        self.head.stride.copy_(torch.tensor([s / h for h, _ in out["shapes"]]))
        self.train(was_training)

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    def forward(self, x: torch.Tensor):
        cache: dict[int, torch.Tensor] = {}
        inp = x
        for i in self.order:
            m = self.layers[i]
            if i == 0:
                y = m(inp)
            elif isinstance(m.f, list):
                y = m([cache[j] for j in m.f])
            else:
                y = m(cache[m.f])
            cache[i] = y
        return cache[len(self.layers) - 1]

    def fuse(self) -> "NextYOLO":
        """Fold BN into convs and drop the o2m branch for NMS-free inference."""
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.fuse()
        self.head.fuse()
        return self

    def info(self, imgsz: int = 640) -> dict:
        n_params = sum(p.numel() for p in self.parameters())
        return {"params": n_params, "gflops": gflops(self, imgsz)}


def gflops(model: nn.Module, imgsz: int = 640) -> float:
    """Multiply-accumulates of convolutions and attention matmuls for one image, reported as GFLOPs (2 * MACs)."""
    model = copy.deepcopy(model).eval()
    macs = 0

    def conv_hook(m, i, o):
        nonlocal macs
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs += o.numel() * k

    def attn_hook(m, i, o):
        nonlocal macs
        n = o.shape[2] * o.shape[3]
        a = m.area if (m.area > 1 and n % m.area == 0) else 1
        macs += o.shape[0] * m.num_heads * (n * n // a) * (m.key_dim + m.head_dim)

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(attn_hook) for m in model.modules() if isinstance(m, MHSA)]
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    for h in hooks:
        h.remove()
    return 2 * macs / 1e9
