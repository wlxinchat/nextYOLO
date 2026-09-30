"""Reference baseline: train Ultralytics YOLO26/YOLO11 from scratch with the same data and budget, then score it with
nextYOLO's evaluator so both detectors are measured identically. Requires `pip install ultralytics` (AGPL-3.0; used
only as an external benchmark, none of its code is part of nextYOLO).

    python tools/baseline_ultralytics.py --data data/voc.json --model yolo26n.yaml --imgsz 256 --epochs 12 \
        --out runs/ul_yolo26n --threads 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class _Wrap(nn.Module):
    """Adapt an Ultralytics model to nextYOLO's evaluator: returns (B, K, 6) or dense (B, A, 4 + nc)."""

    def __init__(self, m, nc, end2end: bool):
        super().__init__()
        self.m, self.end2end = m, end2end
        self.cfg = type("C", (), {"nc": nc})()

    def forward(self, x):
        y = self.m(x)
        y = y[0] if isinstance(y, (tuple, list)) else y
        if self.end2end:
            return y
        # dense (B, 4 + nc, A) xywh -> (B, A, 4 + nc) xyxy
        y = y.transpose(1, 2)
        xy, wh = y[..., :2], y[..., 2:4] / 2
        return torch.cat((xy - wh, xy + wh, y[..., 4:]), -1)


def _resumable(path: Path) -> bool:
    """True if an Ultralytics checkpoint is from an unfinished run (finished runs drop the optimizer state)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26n.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="runs/ul")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--optimizer", default="MuSGD")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--warmup_epochs", type=float, default=3.0)
    ap.add_argument("--close_mosaic", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction of the data (smoke tests)")
    ap.add_argument("--eval_only", default=None, help="path to an Ultralytics .pt to score without training")
    a = ap.parse_args()

    import ultralytics.utils.torch_utils as tu
    from ultralytics import YOLO

    tu.NUM_THREADS = a.threads  # Ultralytics otherwise pins CPU training to cpu_count - 1 threads
    torch.set_num_threads(a.threads)

    spec = json.loads(Path(a.data).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {spec['root']}\ntrain: {json.dumps(spec['train'])}\nval: {json.dumps(spec['val'])}\n"
        f"names: {json.dumps(dict(enumerate(spec['names'])))}\n")

    last = out / "train" / "weights" / "last.pt"
    if (out / "eval_nextyolo_metric.json").exists() and not a.eval_only:
        print("run already complete; nothing to do")
        return
    if a.eval_only:
        weights = a.eval_only
    elif last.exists() and _resumable(last):  # interrupted run: Ultralytics resumes from its own full checkpoint
        YOLO(str(last)).train(resume=True)
        weights = last
    else:
        model = YOLO(a.model)
        model.train(data=str(yaml_path), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device="cpu",
                    workers=a.workers, optimizer=a.optimizer, lr0=a.lr0, warmup_epochs=a.warmup_epochs,
                    close_mosaic=a.close_mosaic, cache="ram", amp=False, plots=False, val=False, seed=0,
                    deterministic=False, pretrained=a.model.endswith(".pt"), project=str(out), name="train",
                    exist_ok=True,
                    compile=a.compile, fraction=a.fraction)
        weights = out / "train" / "weights" / "last.pt"

    from nextyolo.data.dataset import YOLODataset
    from nextyolo.engine.evaluator import evaluate

    torch.set_num_threads(a.threads)
    m = YOLO(str(weights)).model.float().eval()
    head = m.model[-1]
    has_o2o = getattr(head, "one2one_cv2", None) is not None
    root = Path(spec["root"])
    ds = YOLODataset([str(root / v) for v in spec["val"]], a.imgsz, train=False)
    nc = len(spec["names"])
    results = {"weights": str(weights)}
    for mode in (["e2e", "nms"] if has_o2o else ["nms"]):
        if has_o2o:
            head.end2end = mode == "e2e"  # o2o NMS-free branch vs o2m branch + NMS
        res = evaluate(_Wrap(m, nc, mode == "e2e"), ds, workers=a.workers, mode=mode)
        results[mode] = res
        print(mode, json.dumps({k: v for k, v in res.items() if k != "per_class_ap"}), flush=True)
    (out / "eval_nextyolo_metric.json").write_text(json.dumps(results, indent=1))

if __name__ == "__main__":
    main()
