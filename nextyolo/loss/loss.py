"""Detection losses for the dual-assignment head.

Per branch:  L = box_gain * CIoU + l1_gain * L1(normalised ltrb) + cls_gain * L_cls
End-to-end:  L = w_o2m(t) * L_o2m + w_o2o(t) * L_o2o   with w_o2m decaying linearly (ProgLoss)

Classification variants (all use soft IoU-aware targets q from the assigner):
  bce  binary cross-entropy against q                                           (YOLOv8/11/26)
  vfl  varifocal: positives weighted by q, negatives by alpha * p^gamma         (VarifocalNet, PP-YOLOE)
  mal  matchability-aware: target q^gamma, positives weight 1, negatives p^gamma (DEIM)
  qfl  quality focal: BCE(p, q) * |q - p|^beta                                   (GFL)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..utils.boxes import bbox2dist, bbox_iou, dist2bbox, make_anchors, xywh2xyxy
from .assigner import TaskAlignedAssigner


@dataclass
class LossConfig:
    box_gain: float = 7.5
    cls_gain: float = 0.5
    l1_gain: float = 1.5
    cls_loss: str = "bce"          # o2m classification loss
    o2o_cls_loss: str | None = None  # defaults to cls_loss
    stal: bool = True              # small-target-aware candidate selection
    o2m_topk: int = 10
    o2o_topk: int = 7              # o2o picks top-1 out of its top-k pool (consistent matching)
    o2o_assign: str = "self"       # "self": o2o ranks anchors by its own predictions (YOLO26);
                                   # "o2m": AOA — o2o positive = top-1 of the o2m ranking (nextYOLO)
    prog_loss: bool = True         # decay o2m weight 0.8 -> 0.1 across training
    o2m_w0: float = 0.8
    o2m_w1: float = 0.1
    alpha: float = 0.5             # task-alignment exponents
    beta: float = 6.0
    mal_gamma: float = 1.5
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0
    qfl_beta: float = 2.0


def cls_loss_fn(kind: str, logits: torch.Tensor, q: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """Summed classification loss for logits/targets of shape (B, A, C)."""
    q = q.to(logits.dtype)
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, q, reduction="sum")
    p = logits.sigmoid().detach()
    pos = (q > 0).to(logits.dtype)
    if kind == "vfl":
        w = cfg.vfl_alpha * p.pow(cfg.vfl_gamma) * (1 - pos) + q * pos
        return (F.binary_cross_entropy_with_logits(logits, q, reduction="none") * w).sum()
    if kind == "mal":
        t = q.pow(cfg.mal_gamma)
        w = p.pow(cfg.mal_gamma) * (1 - pos) + pos
        return (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * w).sum()
    if kind == "qfl":
        w = (q - p).abs().pow(cfg.qfl_beta)
        return (F.binary_cross_entropy_with_logits(logits, q, reduction="none") * w).sum()
    raise ValueError(f"unknown cls loss {kind!r}")


class BranchLoss:
    """Loss for one prediction branch with its own assigner."""

    def __init__(self, nc: int, strides: list[float], topk: int, topk2: int | None, cls_kind: str, cfg: LossConfig):
        self.nc = nc
        self.strides = strides
        self.cfg = cfg
        self.cls_kind = cls_kind
        min_side = strides[1] if (cfg.stal and len(strides) > 1) else 0.0
        self.assigner = TaskAlignedAssigner(nc, topk, topk2, cfg.alpha, cfg.beta, min_side)

    def __call__(self, boxes, logits, shapes, gt_labels, gt_bboxes, mask_gt, imgsz, assign_src=None):
        """assign_src: optional (boxes, logits) of another branch whose predictions choose the positives (AOA).
        Target quality then comes from this branch's own box IoU at the chosen anchors."""
        B = boxes.shape[0]
        pred_dist = boxes.permute(0, 2, 1).float()        # (B, A, 4) ltrb, stride units
        pred_logits = logits.permute(0, 2, 1).float()     # (B, A, C)
        anchors, stride = make_anchors(shapes, self.strides, device=boxes.device, dtype=pred_dist.dtype)
        pred_bboxes = dist2bbox(pred_dist, anchors)       # xyxy, stride units

        if assign_src is None:
            a_scores, a_boxes = pred_logits.detach().sigmoid(), pred_bboxes.detach()
        else:
            a_scores = assign_src[1].detach().permute(0, 2, 1).float().sigmoid()
            a_boxes = dist2bbox(assign_src[0].detach().permute(0, 2, 1).float(), anchors)
        tb, ts, fg, _ = self.assigner(a_scores, a_boxes * stride, anchors * stride, gt_labels, gt_bboxes, mask_gt)
        if assign_src is not None and fg.any():
            q = bbox_iou(pred_bboxes.detach()[fg], (tb / stride)[fg], "ciou").clamp(0)
            ts[fg] = (ts[fg] > 0).to(ts.dtype) * q[:, None]
        ts_sum = ts.sum().clamp(min=1.0)
        l_cls = cls_loss_fn(self.cls_kind, pred_logits, ts, self.cfg) / ts_sum

        if fg.any():
            w = ts.sum(-1)[fg]                             # (P,)
            tb_s = tb / stride                             # stride units
            iou = bbox_iou(pred_bboxes[fg], tb_s[fg], "ciou")
            l_box = ((1.0 - iou) * w).sum() / ts_sum
            # L1 on ltrb distances normalised by image size (scale-invariant regression target).
            h, w_img = imgsz
            norm = torch.tensor([w_img, h, w_img, h], device=boxes.device, dtype=pred_dist.dtype)
            t_ltrb = bbox2dist(anchors, tb_s)[fg] * stride[fg.nonzero(as_tuple=True)[1]] / norm
            p_ltrb = pred_dist[fg] * stride[fg.nonzero(as_tuple=True)[1]] / norm
            l_l1 = ((p_ltrb - t_ltrb).abs().mean(-1) * w).sum() / ts_sum
        else:
            l_box = l_l1 = pred_dist.sum() * 0.0

        cfg = self.cfg
        parts = torch.stack((l_box * cfg.box_gain, l_cls * cfg.cls_gain, l_l1 * cfg.l1_gain))
        return parts.sum() * B, parts.detach()


def prepare_targets(targets: torch.Tensor, batch_size: int, imgsz: tuple[int, int]):
    """(N, 6) [img_idx, cls, cx, cy, w, h] normalised -> padded (B, M, 1) labels, (B, M, 4) xyxy px, (B, M, 1) mask."""
    device = targets.device
    if targets.numel() == 0:
        z = torch.zeros(batch_size, 0, 1, device=device)
        return z, torch.zeros(batch_size, 0, 4, device=device), z
    idx = targets[:, 0].long()
    counts = torch.bincount(idx, minlength=batch_size)
    M = int(counts.max())
    labels = torch.zeros(batch_size, M, 1, device=device)
    boxes = torch.zeros(batch_size, M, 4, device=device)
    mask = torch.zeros(batch_size, M, 1, device=device)
    order = torch.argsort(idx, stable=True)
    idx_sorted = idx[order]
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(len(idx), device=device) - starts[idx_sorted]
    h, w = imgsz
    scale = torch.tensor([w, h, w, h], device=device, dtype=targets.dtype)
    labels[idx_sorted, slot, 0] = targets[order, 1]
    boxes[idx_sorted, slot] = xywh2xyxy(targets[order, 2:6] * scale)
    mask[idx_sorted, slot, 0] = 1.0
    return labels, boxes, mask


class DetectionLoss:
    """Dual-branch (o2m + o2o) loss with progressive re-weighting."""

    names = ("box", "cls", "l1")

    def __init__(self, nc: int, strides: list[float], cfg: LossConfig | None = None, end2end: bool = True,
                 epochs: int = 100):
        self.cfg = cfg = cfg or LossConfig()
        self.end2end = end2end
        self.epochs = epochs
        self.o2m = BranchLoss(nc, strides, cfg.o2m_topk, None, cfg.cls_loss, cfg)
        self.o2o = BranchLoss(nc, strides, cfg.o2o_topk, 1, cfg.o2o_cls_loss or cfg.cls_loss, cfg) if end2end else None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        if not self.end2end:
            self.w_o2m, self.w_o2o = 1.0, 0.0
        elif cfg.prog_loss:
            t = min(epoch / max(self.epochs - 1, 1), 1.0)
            self.w_o2m = cfg.o2m_w0 + (cfg.o2m_w1 - cfg.o2m_w0) * t
            self.w_o2o = 1.0 - self.w_o2m
        else:
            self.w_o2m = self.w_o2o = 0.5

    def __call__(self, preds: dict, targets: torch.Tensor, imgsz: tuple[int, int]):
        B = preds["o2m"][0].shape[0]
        labels, boxes, mask = prepare_targets(targets.to(preds["o2m"][0].device), B, imgsz)
        l_m, items_m = self.o2m(*preds["o2m"], preds["shapes"], labels, boxes, mask, imgsz)
        if not self.end2end:
            return l_m, {"o2m": items_m}
        src = preds["o2m"] if self.cfg.o2o_assign == "o2m" else None
        l_o, items_o = self.o2o(*preds["o2o"], preds["shapes"], labels, boxes, mask, imgsz, assign_src=src)
        return self.w_o2m * l_m + self.w_o2o * l_o, {"o2m": items_m, "o2o": items_o}
