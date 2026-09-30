"""COCO-style detection metrics (AP@[.50:.95], AP50, AP75, APs/m/l) without pycocotools.

Follows COCOeval semantics for non-crowd ground truth: per (image, class) detections are ranked by score and capped at
`max_dets`; each is greedily matched to the unmatched GT with the highest IoU above the threshold (real GTs before
area-ignored ones, ties to the later GT); GTs outside the area range are ignored, as are unmatched detections outside
it; precision is made monotone and sampled at 101 recall points. Verified against pycocotools in
tests/test_metrics.py.
"""

from __future__ import annotations

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
AREAS = {"all": (0, 1e10), "small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e10)}


def _iou_matrix(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    lt = np.maximum(d[:, None, :2], g[None, :, :2])
    rb = np.minimum(d[:, None, 2:], g[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    ad = (d[:, 2:] - d[:, :2]).prod(-1)
    ag = (g[:, 2:] - g[:, :2]).prod(-1)
    return inter / (ad[:, None] + ag[None, :] - inter + 1e-12)


def _area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def _match(ious: np.ndarray | None, d_area: np.ndarray, g_area: np.ndarray, rng) -> tuple:
    """Greedy COCO matching for one (image, class) and one area range.

    Returns tp (T, D) bool, det-ignore (T, D) bool, number of non-ignored GTs.
    """
    T, D = len(IOU_THRS), len(d_area)
    g_ign = (g_area < rng[0]) | (g_area > rng[1])
    tp = np.zeros((T, D), bool)
    ign = np.zeros((T, D), bool)
    if D and len(g_area):
        cand = np.flatnonzero(ious.max(1) >= IOU_THRS[0])  # detections that can match anything
        for t, thr in enumerate(IOU_THRS):
            taken = np.zeros(len(g_area), bool)
            for d in cand:
                row = ious[d]
                ok = ~taken & (row >= thr)
                if not ok.any():
                    continue
                pool = ok & ~g_ign
                if not pool.any():
                    pool = ok  # only ignored GTs left
                m = np.flatnonzero(pool & (row == row[pool].max()))[-1]
                taken[m] = True
                tp[t, d] = True
                ign[t, d] = g_ign[m]
    if D:
        out = (d_area < rng[0]) | (d_area > rng[1])
        ign |= ~tp & out[None, :]
    return tp, ign, int((~g_ign).sum())


class COCOEvaluator:
    def __init__(self, nc: int, max_dets: int = 100):
        self.nc = nc
        self.max_dets = max_dets
        self.gts: dict[tuple[int, int], np.ndarray] = {}
        self.dts: dict[tuple[int, int], np.ndarray] = {}

    def add(self, img_id: int, gt_cls, gt_boxes, det_boxes, det_scores, det_cls) -> None:
        """Register one image: GT (n,) + (n, 4) xyxy and detections (k, 4) xyxy + (k,) scores + (k,) classes."""
        gt_cls, det_cls = np.asarray(gt_cls).astype(int).reshape(-1), np.asarray(det_cls).astype(int).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
        det_boxes = np.asarray(det_boxes, float).reshape(-1, 4)
        det_scores = np.asarray(det_scores, float).reshape(-1)
        for c in np.unique(gt_cls):
            self.gts[(img_id, int(c))] = gt_boxes[gt_cls == c]
        for c in np.unique(det_cls):
            m = det_cls == c
            b, s = det_boxes[m], det_scores[m]
            order = np.argsort(-s, kind="mergesort")[: self.max_dets]
            self.dts[(img_id, int(c))] = np.concatenate((b[order], s[order, None]), 1)

    def evaluate(self) -> dict:
        names = list(AREAS)
        acc = {a: {c: {"scores": [], "tp": [], "ign": [], "npos": 0} for c in range(self.nc)} for a in names}
        for key in sorted(set(self.gts) | set(self.dts)):
            c = key[1]
            if c >= self.nc:
                continue
            gt = self.gts.get(key, np.zeros((0, 4)))
            dt = self.dts.get(key, np.zeros((0, 5)))
            ious = _iou_matrix(dt[:, :4], gt) if len(dt) and len(gt) else None
            d_area, g_area = _area(dt), _area(gt)
            for a in names:
                tp, ign, npos = _match(ious, d_area, g_area, AREAS[a])
                pc = acc[a][c]
                pc["scores"].append(dt[:, 4])
                pc["tp"].append(tp)
                pc["ign"].append(ign)
                pc["npos"] += npos

        results, per_class = {}, None
        for a in names:
            prec = -np.ones((len(IOU_THRS), len(REC_THRS), self.nc))
            for c in range(self.nc):
                pc = acc[a][c]
                if pc["npos"] == 0:
                    continue
                prec[:, :, c] = 0.0
                if not pc["scores"]:
                    continue
                scores = np.concatenate(pc["scores"])
                order = np.argsort(-scores, kind="mergesort")
                tp = np.concatenate(pc["tp"], 1)[:, order]
                ign = np.concatenate(pc["ign"], 1)[:, order]
                tps = np.cumsum(tp & ~ign, 1, dtype=float)
                fps = np.cumsum(~tp & ~ign, 1, dtype=float)
                for t in range(len(IOU_THRS)):
                    if tps.shape[1] == 0:
                        continue
                    rc = tps[t] / pc["npos"]
                    pr = tps[t] / (fps[t] + tps[t] + np.spacing(1))
                    pr = np.maximum.accumulate(pr[::-1])[::-1]
                    inds = np.searchsorted(rc, REC_THRS, side="left")
                    valid = inds < len(pr)
                    q = np.zeros(len(REC_THRS))
                    q[valid] = pr[inds[valid]]
                    prec[t, :, c] = q

            def ap(p):
                p = p[p > -1]
                return float(p.mean()) if p.size else -1.0

            results[a] = (ap(prec), ap(prec[0]), ap(prec[5]))
            if a == "all":
                per_class = [ap(prec[:, :, c]) for c in range(self.nc)]
        return {
            "mAP": results["all"][0], "mAP50": results["all"][1], "mAP75": results["all"][2],
            "APs": results["small"][0], "APm": results["medium"][0], "APl": results["large"][0],
            "per_class_ap": per_class,
        }
