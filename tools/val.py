"""Evaluate a checkpoint with the COCO-style evaluator.

    python tools/val.py --weights runs/voc_n/best.pt --data data/voc.json --imgsz 320
    python tools/val.py --ultralytics yolo26n.pt --data data/voc.json --imgsz 640   # COCO model, zero-shot on VOC

With a model whose classes are a superset of the dataset's (matched by name, with COCO<->VOC aliases), the class
heads are sliced to the dataset's classes (see `subset_classes`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nextyolo.data.dataset import YOLODataset  # noqa: E402
from nextyolo.engine.evaluator import evaluate  # noqa: E402
from nextyolo.engine.trainer import load_model  # noqa: E402
from nextyolo.nn.model import subset_classes  # noqa: E402

ALIASES = {"airplane": "aeroplane", "motorcycle": "motorbike", "couch": "sofa", "potted plant": "pottedplant",
           "dining table": "diningtable", "tv": "tvmonitor"}


def class_index_map(model_names: list[str], data_names: list[str]) -> list[int] | None:
    """Indices into model_names for each data class, or None if the class sets are already identical."""
    if list(model_names) == list(data_names):
        return None
    norm = {ALIASES.get(n, n): i for i, n in enumerate(model_names)}
    missing = [n for n in data_names if n not in norm]
    if missing:
        raise ValueError(f"model lacks dataset classes {missing}")
    return [norm[n] for n in data_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="nextYOLO checkpoint")
    ap.add_argument("--ultralytics", help="Ultralytics YOLO26 checkpoint (loaded into nextYOLO)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", default="e2e", choices=["e2e", "nms", "both"])
    ap.add_argument("--device", default="auto", help="auto = cuda if available, else cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    a = ap.parse_args()
    if a.out and Path(a.out).exists():
        print(f"{a.out} exists; nothing to do")
        return
    if a.threads:
        torch.set_num_threads(a.threads)

    if a.ultralytics:
        from tools.convert_ultralytics import load_ultralytics
        model, names = load_ultralytics(a.ultralytics)
    else:
        model = load_model(a.weights)
        names = model.names
    spec = json.loads(Path(a.data).read_text())
    keep = class_index_map(names, spec["names"])
    if keep is not None:
        model = subset_classes(model, keep)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else "cpu" if a.device == "auto" else a.device)
    model = model.to(device).eval()
    root = Path(spec["root"])
    ds = YOLODataset([str(root / v) for v in spec["val"]], a.imgsz, train=False, max_images=a.max_images)
    res = {}
    for mode in (["e2e", "nms"] if a.mode == "both" else [a.mode]):
        model.head.end2end = mode == "e2e"  # nms: dense o2m branch + NMS
        r = evaluate(model, ds, workers=a.workers, mode=mode, max_images=a.max_images)
        r.update(weights=a.weights or a.ultralytics, imgsz=a.imgsz, mode=mode, device=str(device))
        print(json.dumps({k: v for k, v in r.items() if k != "per_class_ap"}), flush=True)
        res[mode] = r
    res = res[a.mode] if a.mode != "both" else res
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
