"""Bounding-box primitives: format conversion, IoU variants and anchor-point encoding."""

from __future__ import annotations

import math

import torch


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2) on the last dimension."""
    xy, wh = x[..., :2], x[..., 2:4] / 2
    return torch.cat((xy - wh, xy + wh), -1)


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) -> (cx, cy, w, h) on the last dimension."""
    x1y1, x2y2 = x[..., :2], x[..., 2:4]
    return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), -1)


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Pairwise IoU between xyxy boxes a (N, 4) and b (M, 4) -> (N, M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def bbox_iou(a: torch.Tensor, b: torch.Tensor, kind: str = "ciou", eps: float = 1e-7) -> torch.Tensor:
    """Element-wise IoU between broadcastable xyxy boxes, returned with a trailing singleton dim removed.

    kind: "iou", "giou", "diou" or "ciou".
    """
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    aw, ah = (ax2 - ax1).clamp(min=eps), (ay2 - ay1).clamp(min=eps)
    bw, bh = (bx2 - bx1).clamp(min=eps), (by2 - by1).clamp(min=eps)
    inter = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0) * (
        torch.min(ay2, by2) - torch.max(ay1, by1)
    ).clamp(min=0)
    union = aw * ah + bw * bh - inter + eps
    iou = inter / union
    if kind == "iou":
        return iou
    cw = torch.max(ax2, bx2) - torch.min(ax1, bx1)  # enclosing box
    ch = torch.max(ay2, by2) - torch.min(ay1, by1)
    if kind == "giou":
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    c2 = cw.pow(2) + ch.pow(2) + eps  # squared enclosing diagonal
    rho2 = ((bx1 + bx2 - ax1 - ax2).pow(2) + (by1 + by2 - ay1 - ay2).pow(2)) / 4  # squared centre distance
    if kind == "diou":
        return iou - rho2 / c2
    if kind == "ciou":
        v = (4 / math.pi**2) * (torch.atan(bw / bh) - torch.atan(aw / ah)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    raise ValueError(f"unknown IoU kind {kind!r}")


def make_anchors(shapes: list[tuple[int, int]], strides: list[int], offset: float = 0.5, device=None, dtype=None):
    """Anchor-point centres (in grid units of each level) and per-point strides for a list of (h, w) feature shapes.

    Returns:
        points (A, 2) xy in grid units, stride (A, 1).
    """
    points, stride_t = [], []
    for (h, w), s in zip(shapes, strides):
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((xx, yy), -1).view(-1, 2))
        stride_t.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_t)


def dist2bbox(dist: torch.Tensor, points: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Decode (l, t, r, b) distances from anchor points into xyxy boxes."""
    lt, rb = dist.chunk(2, dim)
    return torch.cat((points - lt, points + rb), dim)


def bbox2dist(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Encode xyxy boxes as (l, t, r, b) distances from anchor points."""
    x1y1, x2y2 = boxes.chunk(2, -1)
    return torch.cat((points - x1y1, x2y2 - points), -1)
