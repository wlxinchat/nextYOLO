"""Pipeline sanity check: overfit a few images without augmentation; mAP on the same images should approach 1.

    python tools/overfit.py --data data/voc.json --n 16 --iters 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nextyolo.data.dataset import YOLODataset, collate  # noqa: E402
from nextyolo.engine.evaluator import evaluate  # noqa: E402
from nextyolo.engine.trainer import build_optimizer  # noqa: E402
from nextyolo.loss.loss import DetectionLoss, LossConfig  # noqa: E402
from nextyolo.nn.model import ModelConfig, NextYOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    data = json.loads(Path(a.data).read_text())
    root = Path(data["root"])
    src = [str(root / data["val"][0])]
    no_aug = dict(mosaic=0.0, translate=0.0, scale=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    ds_train = YOLODataset(src, a.imgsz, train=True, hyp=no_aug, max_images=a.n)
    ds_eval = YOLODataset(src, a.imgsz, train=False, max_images=a.n)
    nc = len(data["names"])
    model = NextYOLO(ModelConfig(nc=nc))
    crit = DetectionLoss(nc, model.stride.tolist(), LossConfig(prog_loss=False), end2end=True, epochs=1)
    opt = build_optimizer(model, a.optimizer, a.lr, 0.9, 0.0)
    loader = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=True, collate_fn=collate)
    it, t0 = 0, time.time()
    while it < a.iters:
        for imgs, targets, _ in loader:
            model.train()
            loss, items = crit(model(imgs.float() / 255), targets, (a.imgsz, a.imgsz))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            it += 1
            if it % 50 == 0:
                print(f"it {it} loss {loss.item():.2f} o2m {items['o2m'].tolist()} o2o {items['o2o'].tolist()} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if it >= a.iters:
                break
    r_e2e = evaluate(model, ds_eval, workers=0)
    model.head.end2end = False
    r_nms = evaluate(model, ds_eval, workers=0, mode="nms")
    model.head.end2end = True
    print(f"overfit mAP (o2o, NMS-free): {r_e2e['mAP']:.3f}  mAP50 {r_e2e['mAP50']:.3f}")
    print(f"overfit mAP (o2m + NMS):     {r_nms['mAP']:.3f}  mAP50 {r_nms['mAP50']:.3f}")


if __name__ == "__main__":
    main()
