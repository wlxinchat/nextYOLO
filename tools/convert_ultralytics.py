"""Load Ultralytics YOLO26 weights into nextYOLO (the architectures match parameter-for-parameter).

The mapping is a pure rename: `model.{i}.` -> `layers.{i}.` and, in the head, cv2/cv3 -> box/cls and
one2one_cv2/one2one_cv3 -> o2o_box/o2o_cls. Note that the official weights are AGPL-3.0 licensed; loading them does not
change nextYOLO's code, but models derived from them inherit that licence.

    python tools/convert_ultralytics.py --weights yolo26n.pt --scale n --out yolo26n_nextyolo.pt [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nextyolo.nn.model import ModelConfig, NextYOLO  # noqa: E402

HEAD_RENAMES = {"one2one_cv2": "o2o_box", "one2one_cv3": "o2o_cls", "cv2": "box", "cv3": "cls"}


def convert_state_dict(sd: dict, head_index: int) -> dict:
    out = {}
    for k, v in sd.items():
        if not k.startswith("model."):
            continue
        parts = k.split(".")
        parts[0] = "layers"
        if int(parts[1]) == head_index and parts[2] in HEAD_RENAMES:
            parts[2] = HEAD_RENAMES[parts[2]]
        out[".".join(parts)] = v
    return out


def load_ultralytics(path: str, scale: str | None = None) -> tuple[NextYOLO, list[str]]:
    """Build a nextYOLO model from an Ultralytics YOLO26 detection checkpoint. Returns (model, class names)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    um = (ck.get("ema") or ck["model"]).float()
    names = um.names if isinstance(um.names, list) else [um.names[i] for i in sorted(um.names)]
    scale = scale or um.yaml.get("scale", "n")
    model = NextYOLO(ModelConfig(nc=len(names), scale=scale))
    sd = convert_state_dict(um.state_dict(), len(model.layers) - 1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("head.stride") and "stride" not in k]
    if missing or unexpected:
        raise RuntimeError(f"weight mapping mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.head.stride.copy_(um.stride)
    return model.eval(), names


@torch.no_grad()
def check_against_ultralytics(path: str, model: NextYOLO, imgsz: int = 640) -> float:
    """Max |difference| between Ultralytics' NMS-free output and nextYOLO's on a random image."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    um = (ck.get("ema") or ck["model"]).float().eval()
    um.model[-1].end2end = True
    x = torch.rand(1, 3, imgsz, imgsz)
    ref = um(x)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    ours = model(x)
    return float((ref - ours).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scale", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="compare outputs with Ultralytics (needs ultralytics)")
    a = ap.parse_args()
    model, names = load_ultralytics(a.weights, a.scale)
    print(f"loaded {a.weights}: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M params, {len(names)} classes")
    if a.check:
        print(f"max |ultralytics - nextYOLO| on NMS-free output: {check_against_ultralytics(a.weights, model):.3e}")
    if a.out:
        from dataclasses import asdict
        torch.save({"epoch": -1, "model_cfg": asdict(model.cfg), "names": names, "ema": model.state_dict()}, a.out)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
