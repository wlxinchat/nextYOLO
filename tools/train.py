"""Train nextYOLO.

    python tools/train.py --data data/voc.json --scale n --imgsz 320 --epochs 40 --out runs/voc_n
    python tools/train.py ... --set loss.o2o_cls_loss=mal model.levels=[3,5] optimizer=sgd

`--set` takes dotted overrides into TrainConfig (`model.*` -> ModelConfig, `loss.*` -> LossConfig, `hyp.*` ->
augmentation); values are parsed as JSON when possible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nextyolo.engine.trainer import TrainConfig, Trainer  # noqa: E402


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSON with train/val/names (paths relative to the file allowed)")
    ap.add_argument("--scale", default="n")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    data = json.loads(Path(a.data).read_text())
    root = Path(data.get("root", Path(a.data).parent))
    fix = lambda xs: [str(root / x) for x in ([xs] if isinstance(xs, str) else xs)]  # noqa: E731
    cfg = TrainConfig(train=fix(data["train"]), val=fix(data["val"]), names=data["names"], out=a.out,
                      epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, model={"scale": a.scale})
    for kv in a.set:
        k, v = kv.split("=", 1)
        v = parse_value(v)
        if "." in k:
            group, key = k.split(".", 1)
            if key == "levels":
                v = tuple(v)
            getattr(cfg, group)[key] = v
        else:
            if not hasattr(cfg, k):
                raise KeyError(k)
            setattr(cfg, k, v)
    if cfg.resume and (Path(cfg.out) / "summary.json").exists():
        print(f"{cfg.out}: run already complete; nothing to do")
        return
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
