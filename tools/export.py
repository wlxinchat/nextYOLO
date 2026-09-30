"""Export a trained nextYOLO checkpoint to ONNX with the NMS-free output baked in.

The graph is convs + a top-k + gathers: output (B, max_det, 6) = [x1, y1, x2, y2, score, class] in input pixels.
No NMS plugin or post-processing is required by the runtime.

    python tools/export.py --weights runs/voc_n/best.pt --imgsz 640 --out nextyolo_n.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nextyolo.engine.trainer import load_model  # noqa: E402
from nextyolo.nn.model import ModelConfig, NextYOLO  # noqa: E402


def export_onnx(model: NextYOLO, path: str, imgsz: int = 640, batch: int = 1, opset: int = 17) -> str:
    model = model.eval().fuse()
    x = torch.rand(batch, 3, imgsz, imgsz)
    try:
        torch.onnx.export(model, (x,), path, input_names=["images"], output_names=["detections"],
                          opset_version=opset, dynamo=False)
    except TypeError:  # older torch without the `dynamo` flag
        torch.onnx.export(model, (x,), path, input_names=["images"], output_names=["detections"],
                          opset_version=opset)
    return path


def check_onnx(model: NextYOLO, path: str, imgsz: int, top: int = 100) -> float:
    """Worst-case distance from each of the `top` ONNX Runtime detections to its closest PyTorch detection.

    Order-invariant: near-tied scores may legitimately swap rank between runtimes.
    """
    import onnxruntime as ort

    x = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        ref = model.eval()(x)[0].numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x.numpy()})[0][0]
    d = np.abs(out[:top, None, :] - ref[None, :, :]).max(-1)  # (top, K) Chebyshev distance
    return float(d.min(1).max())


@torch.no_grad()
def calibrate_bn(model: NextYOLO, imgsz: int) -> NextYOLO:
    """Give an untrained model realistic BN statistics (fresh-init eval activations otherwise vanish)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.momentum = 1.0
    model.train()(torch.rand(4, 3, imgsz, imgsz))
    for bn in bns:
        bn.momentum = 0.03
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint; omit to export a randomly initialised model")
    ap.add_argument("--scale", default="n")
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="nextyolo.onnx")
    a = ap.parse_args()
    model = load_model(a.weights) if a.weights else calibrate_bn(NextYOLO(ModelConfig(nc=a.nc, scale=a.scale)), a.imgsz)
    model = model.eval().fuse()
    export_onnx(model, a.out, a.imgsz)
    print(f"exported {a.out}; max deviation torch vs onnxruntime = {check_onnx(model, a.out, a.imgsz):.2e}")


if __name__ == "__main__":
    main()
