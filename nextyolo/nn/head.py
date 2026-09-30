"""NMS-free dual-assignment detection head.

Two structurally identical branches share the neck features:

* one-to-many (o2m): dense supervision (top-k positives per object). Only used during training.
* one-to-one (o2o): exactly one positive per object, so its scores can be read out with a plain top-k — no NMS.

The o2o branch sees detached features, so the backbone/neck are shaped only by the richer o2m signal while the
o2o branch learns to pick a single anchor per object (the YOLOv10/YOLO26 recipe). Boxes are regressed directly as
(l, t, r, b) distances in stride units — no DFL bins — which keeps the exported graph to convs, a top-k and a gather.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from ..utils.boxes import dist2bbox, make_anchors
from .blocks import Conv, DWConv


class DetectHead(nn.Module):
    def __init__(self, nc: int, ch: list[int], end2end: bool = True, max_det: int = 300,
                 o2o_grad_scale: float = 0.0):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.max_det = max_det
        self.o2o_grad_scale = o2o_grad_scale
        self.register_buffer("stride", torch.zeros(self.nl), persistent=True)
        c_box = max(16, ch[0] // 4)
        c_cls = max(ch[0], min(nc, 100))
        self.box = nn.ModuleList(
            nn.Sequential(Conv(c, c_box, 3), Conv(c_box, c_box, 3), nn.Conv2d(c_box, 4, 1)) for c in ch)
        self.cls = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(c, c, 3), Conv(c, c_cls, 1)),
                          nn.Sequential(DWConv(c_cls, c_cls, 3), Conv(c_cls, c_cls, 1)),
                          nn.Conv2d(c_cls, nc, 1)) for c in ch)
        self.end2end = end2end
        if end2end:
            self.o2o_box = copy.deepcopy(self.box)
            self.o2o_cls = copy.deepcopy(self.cls)
        self._anchor_cache: tuple | None = None

    def bias_init(self, ref_imgsz: int = 640) -> None:
        """Box bias -> ~2 cells per side; class prior ~5 objects per image spread over nc classes."""
        branches = [(self.box, self.cls)] + ([(self.o2o_box, self.o2o_cls)] if self.end2end else [])
        for box, cls in branches:
            if box is None:
                continue
            for b, c, s in zip(box, cls, self.stride.tolist()):
                b[-1].bias.data.fill_(2.0)
                c[-1].bias.data.fill_(math.log(5 / self.nc / (ref_imgsz / s) ** 2))

    @staticmethod
    def _branch(feats: list[torch.Tensor], box: nn.ModuleList, cls: nn.ModuleList):
        bs = feats[0].shape[0]
        boxes = torch.cat([box[i](f).view(bs, 4, -1) for i, f in enumerate(feats)], -1)
        logits = torch.cat([cls[i](f).view(bs, cls[i][-1].out_channels, -1) for i, f in enumerate(feats)], -1)
        return boxes, logits  # (B, 4, A), (B, nc, A)

    def forward(self, feats: list[torch.Tensor]):
        shapes = [tuple(f.shape[2:]) for f in feats]
        if self.training:
            out = {"shapes": shapes, "o2m": self._branch(feats, self.box, self.cls)}
            if self.end2end:
                s = self.o2o_grad_scale
                f2 = [f.detach() if s == 0 else _scale_grad(f, s) for f in feats]
                out["o2o"] = self._branch(f2, self.o2o_box, self.o2o_cls)
            return out
        if self.end2end:
            boxes, logits = self._branch(feats, self.o2o_box, self.o2o_cls)
            return self.topk_decode(*self._decode(boxes, logits, shapes))
        boxes, logits = self._branch(feats, self.box, self.cls)
        return torch.cat(self._decode(boxes, logits, shapes), -1)  # (B, A, 4 + nc) for NMS post-processing

    def _decode(self, boxes: torch.Tensor, logits: torch.Tensor, shapes):
        key = (tuple(shapes), boxes.device, boxes.dtype)
        if self._anchor_cache is None or self._anchor_cache[0] != key:
            pts, st = make_anchors(shapes, self.stride.tolist(), device=boxes.device, dtype=boxes.dtype)
            self._anchor_cache = (key, pts.t().unsqueeze(0), st.t().unsqueeze(0))
        _, pts, st = self._anchor_cache
        xyxy = dist2bbox(boxes, pts, dim=1) * st  # (B, 4, A) pixels
        return xyxy.transpose(1, 2), logits.sigmoid().transpose(1, 2)

    def topk_decode(self, xyxy: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pick the `max_det` best (anchor, class) pairs -> (B, K, 6) [x1, y1, x2, y2, score, class]."""
        k = min(self.max_det, scores.shape[1])
        # First keep the k anchors with the best class score, then rank their (anchor, class) pairs.
        idx = scores.amax(-1).topk(k, dim=1).indices  # (B, k)
        cand = scores.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.nc))  # (B, k, nc)
        conf, flat = cand.flatten(1).topk(k, dim=1)
        anchor = idx.gather(1, torch.div(flat, self.nc, rounding_mode="floor"))
        cls = (flat % self.nc).to(xyxy.dtype)
        box = xyxy.gather(1, anchor.unsqueeze(-1).expand(-1, -1, 4))
        return torch.cat((box, conf.unsqueeze(-1), cls.unsqueeze(-1)), -1)

    def fuse(self) -> None:
        """Drop the training-only o2m branch when running NMS-free."""
        if self.end2end:
            self.box = None
            self.cls = None


class _ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


def _scale_grad(x: torch.Tensor, s: float) -> torch.Tensor:
    return _ScaleGrad.apply(x, s)
