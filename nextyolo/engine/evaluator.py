"""Run a detector over a dataset and compute COCO-style metrics in original-image coordinates."""

from __future__ import annotations

import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from ..data.dataset import YOLODataset, collate
from ..utils.metrics import COCOEvaluator


def nms_postprocess(pred: torch.Tensor, conf: float = 0.001, iou: float = 0.7, max_det: int = 300):
    """(B, A, 4 + nc) decoded dense predictions -> list of (k, 6) after class-aware NMS."""
    out = []
    for p in pred:
        boxes, scores = p[:, :4], p[:, 4:]
        i, j = (scores > conf).nonzero(as_tuple=True)
        b, s, c = boxes[i], scores[i, j], j.float()
        if len(s) > 30000:
            top = s.topk(30000).indices
            b, s, c = b[top], s[top], c[top]
        keep = torchvision.ops.batched_nms(b, s, c.long(), iou)[:max_det]
        out.append(torch.cat((b[keep], s[keep, None], c[keep, None]), 1))
    return out


@torch.no_grad()
def evaluate(model, dataset: YOLODataset, batch_size: int = 32, workers: int = 2, conf: float = 0.001,
             mode: str = "e2e", nms_iou: float = 0.7, max_images: int | None = None, verbose: bool = False) -> dict:
    """mode: 'e2e' (model returns (B, K, 6) NMS-free) or 'nms' (model returns dense (B, A, 4+nc))."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                        pin_memory=torch.cuda.is_available())
    ev = COCOEvaluator(nc=model.cfg.nc if hasattr(model, "cfg") else dataset.nc)
    t_model, n_img = 0.0, 0
    device = next(model.parameters()).device
    for imgs, _, metas in loader:
        x = imgs.to(device, non_blocking=True).float() / 255
        t0 = time.perf_counter()
        pred = model(x)
        t_model += time.perf_counter() - t0
        dets = [p[p[:, 4] > conf] for p in pred] if mode == "e2e" else nms_postprocess(pred, conf, nms_iou)
        for d, meta in zip(dets, metas):
            d = d.cpu().numpy().astype(np.float64)
            px, py = meta["pad"]
            r = meta["ratio"]
            oh, ow = meta["orig_shape"]
            boxes = d[:, :4].copy()
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / r).clip(0, ow)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / r).clip(0, oh)
            gc, gb = dataset.gt_original(meta["index"])
            ev.add(meta["index"], gc, gb, boxes, d[:, 4], d[:, 5])
        n_img += len(imgs)
        if max_images and n_img >= max_images:
            break
    t0 = time.perf_counter()
    res = ev.evaluate()
    res["eval_time_s"] = time.perf_counter() - t0
    res["model_ms_per_img"] = 1000 * t_model / max(n_img, 1)
    return res
