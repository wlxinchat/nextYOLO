"""Task-aligned label assignment with small-target awareness.

For each ground-truth box the assigner ranks candidate anchor points by the task-alignment metric
    t = s^alpha * u^beta
(s: predicted score of the GT class, u: CIoU of the predicted box with the GT) and keeps the top-k. Anchors claimed by
several GTs go to the GT with the highest IoU. A second optional top-k2 filter (k2 < k) yields the one-to-one
assignment used by the NMS-free branch: the o2o positive is chosen from the same candidate pool, with the same metric,
as the o2m positives ("consistent matching").

Small-target awareness (STAL): candidate anchors must lie inside the GT box, which leaves tiny objects with few or no
candidates on coarse grids. Boxes whose side is below `min_side` pixels are enlarged to `min_side` about their centre
for candidate selection only (regression targets are unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.boxes import bbox_iou


class TaskAlignedAssigner(nn.Module):
    def __init__(self, num_classes: int, topk: int = 10, topk2: int | None = None, alpha: float = 0.5,
                 beta: float = 6.0, min_side: float = 0.0, eps: float = 1e-9):
        super().__init__()
        self.nc = num_classes
        self.topk = topk
        self.topk2 = topk2 or topk
        self.alpha = alpha
        self.beta = beta
        self.min_side = min_side
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchors, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores: (B, A, C) sigmoid scores.
            pd_bboxes: (B, A, 4) predicted xyxy boxes, pixels.
            anchors:   (A, 2) anchor centres, pixels.
            gt_labels: (B, M, 1) class ids.
            gt_bboxes: (B, M, 4) xyxy pixels.
            mask_gt:   (B, M, 1) 1 for real boxes, 0 for padding.
        Returns:
            target_bboxes (B, A, 4), target_scores (B, A, C), fg_mask (B, A) bool, target_gt_idx (B, A).
        """
        B, A, C = pd_scores.shape
        M = gt_bboxes.shape[1]
        if M == 0:
            return (torch.zeros_like(pd_bboxes), torch.zeros_like(pd_scores),
                    torch.zeros(B, A, dtype=torch.bool, device=pd_scores.device),
                    torch.zeros(B, A, dtype=torch.long, device=pd_scores.device))

        valid = mask_gt.squeeze(-1).bool()                                         # (B, M)
        in_gts = self._candidates_in_gts(anchors, gt_bboxes) & valid[..., None]    # (B, M, A)

        # Alignment metric on candidate pairs only (sparse gather keeps memory ~ #candidates).
        b_idx, m_idx, a_idx = in_gts.nonzero(as_tuple=True)
        cls_idx = gt_labels[b_idx, m_idx, 0].long()
        s = pd_scores[b_idx, a_idx, cls_idx]
        u = bbox_iou(gt_bboxes[b_idx, m_idx], pd_bboxes[b_idx, a_idx], "ciou").clamp_(0)
        overlaps = torch.zeros(B, M, A, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        metric = torch.zeros_like(overlaps)
        overlaps[b_idx, m_idx, a_idx] = u
        metric[b_idx, m_idx, a_idx] = s.pow(self.alpha) * u.pow(self.beta)

        # Top-k anchors per GT (restricted to candidates).
        # (a tiny bonus for in-box candidates keeps ties at metric == 0 from selecting anchors outside the box)
        k = min(self.topk, A)
        topk_idx = (metric + in_gts * 1e-30).topk(k, dim=-1).indices              # (B, M, k)
        mask_pos = torch.zeros_like(in_gts).scatter_(-1, topk_idx, True) & in_gts

        # Anchors claimed by several GTs keep only the GT with max IoU.
        multi = mask_pos.sum(1, keepdim=True) > 1                                 # (B, 1, A)
        best_gt = overlaps.argmax(1, keepdim=True)                                # (B, 1, A)
        is_best = torch.zeros_like(mask_pos).scatter_(1, best_gt, True)
        mask_pos = torch.where(multi, is_best & mask_pos, mask_pos)

        # Optional second top-k2 (one-to-one selection from the same pool).
        if self.topk2 < self.topk:
            m2 = metric * mask_pos
            k2_idx = m2.topk(self.topk2, dim=-1).indices
            mask_pos &= torch.zeros_like(mask_pos).scatter_(-1, k2_idx, True)

        fg_mask = mask_pos.any(1)                                                 # (B, A)
        target_gt_idx = mask_pos.float().argmax(1)                                # (B, A)

        # Gather targets.
        batch = torch.arange(B, device=gt_labels.device)[:, None]
        target_bboxes = gt_bboxes[batch, target_gt_idx]                           # (B, A, 4)
        target_labels = gt_labels[batch, target_gt_idx, 0].long()                 # (B, A)

        # Soft targets: metric rescaled per GT so its best positive gets that GT's best IoU.
        metric = metric * mask_pos
        overlaps = overlaps * mask_pos
        pos_metric_max = metric.amax(-1, keepdim=True)
        pos_iou_max = overlaps.amax(-1, keepdim=True)
        norm = (metric * pos_iou_max / (pos_metric_max + self.eps)).amax(1)       # (B, A)
        target_scores = torch.zeros(B, A, C, dtype=pd_scores.dtype, device=pd_scores.device)
        target_scores.scatter_(-1, target_labels.unsqueeze(-1), 1.0)
        target_scores *= (norm * fg_mask).unsqueeze(-1)
        return target_bboxes, target_scores, fg_mask, target_gt_idx

    def _candidates_in_gts(self, anchors: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """(B, M, A) mask of anchor centres strictly inside each (STAL-enlarged) GT box."""
        boxes = gt_bboxes
        if self.min_side > 0:
            c = (boxes[..., :2] + boxes[..., 2:]) / 2
            wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=self.min_side)
            boxes = torch.cat((c - wh / 2, c + wh / 2), -1)
        lt = anchors[None, None] - boxes[..., None, :2]                           # (B, M, A, 2)
        rb = boxes[..., None, 2:] - anchors[None, None]
        return torch.cat((lt, rb), -1).amin(-1) > self.eps
