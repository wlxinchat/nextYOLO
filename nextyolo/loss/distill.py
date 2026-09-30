"""Dense anchor-aligned knowledge distillation for dual-assignment heads.

Teacher and student share strides and input resolution, so their predictions live on the same anchor grid and can be
matched per anchor, with no feature adapters. Each branch is distilled into its counterpart:

* o2m: the teacher's dense class probabilities (soft labels) and its box distances.
* o2o: the same terms. The teacher's one-to-one scores encode *which single anchor should fire* for each object — the
  hardest thing for an NMS-free head to learn from sparse ground truth.

    L_kd = cls_gain * sum KL(Bern(t) || Bern(sigmoid(s / T))) / sum(t),   t = sigmoid(teacher_logits / T)
         + box_gain * sum_a w_a * |ltrb_s - ltrb_t|_1 / sum_a w_a

with w_a = max-class teacher probability at anchor a, zeroed below `box_min_conf` (box distillation only where the
teacher is confident — o2o boxes at anchors that never fire are untrained and carry no information).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillConfig:
    teacher: str | None = None     # checkpoint path (nextYOLO or Ultralytics YOLO26)
    cls_gain: float = 1.0
    box_gain: float = 1.0
    temperature: float = 1.0
    box_min_conf: float = 0.1
    branches: tuple = ("o2m", "o2o")


def distill_loss(student: dict, teacher: dict, cfg: DistillConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss scaled like the detection loss, i.e. * batch size, detached per-part values (cls, box))."""
    total_cls, total_box = 0.0, 0.0
    B = student["o2m"][0].shape[0]
    for br in cfg.branches:
        if br not in student or br not in teacher:
            continue
        s_box, s_logit = student[br]
        t_box, t_logit = teacher[br]
        if s_logit.shape != t_logit.shape:
            raise ValueError(f"teacher/student anchor grids differ: {tuple(t_logit.shape)} vs {tuple(s_logit.shape)}")
        t_prob = (t_logit.float() / cfg.temperature).sigmoid()
        ce = F.binary_cross_entropy_with_logits(s_logit.float() / cfg.temperature, t_prob, reduction="sum")
        ent = F.binary_cross_entropy(t_prob, t_prob, reduction="sum")  # constant: makes the loss a KL (0 at optimum)
        total_cls = total_cls + (ce - ent) / t_prob.sum().clamp(min=1.0)
        w = t_prob.amax(1)                                             # (B, A)
        w = w * (w >= cfg.box_min_conf)
        l_box = ((s_box.float() - t_box.float()).abs().sum(1) * w).sum() / w.sum().clamp(min=1e-6)
        total_box = total_box + l_box
    parts = torch.stack((torch.as_tensor(total_cls) * cfg.cls_gain, torch.as_tensor(total_box) * cfg.box_gain))
    return parts.sum() * B, parts.detach()
