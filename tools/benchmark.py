"""CPU latency benchmark (batch 1) for PyTorch (fused, eager) and ONNX Runtime.

    python tools/benchmark.py --scales n s --imgsz 640 --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nextyolo.nn.model import ModelConfig, NextYOLO, gflops  # noqa: E402
from tools.export import export_onnx  # noqa: E402


def time_fn(fn, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return 1000 * ts[len(ts) // 2]  # median ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["n"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variants", nargs="*", default=["default"],
                    help="default | dual_scale | p2 (model variants to time)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import onnxruntime as ort

    variants = {"default": {}, "dual_scale": {"levels": (3, 5)}, "p2": {"p2_fusion": True}}
    rows = []
    for s in a.scales:
        for v in a.variants:
            m = NextYOLO(ModelConfig(nc=a.nc, scale=s, **variants[v])).eval()
            params = sum(p.numel() for p in m.parameters())
            m.fuse()
            infer_params = sum(p.numel() for p in m.parameters())
            fl = gflops(m, a.imgsz)
            x = torch.rand(1, 3, a.imgsz, a.imgsz)
            with torch.no_grad():
                t_pt = time_fn(lambda: m(x))
            with tempfile.TemporaryDirectory() as d:
                path = export_onnx(m, f"{d}/m.onnx", a.imgsz)
                so = ort.SessionOptions()
                so.intra_op_num_threads = a.threads
                sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                xn = x.numpy()
                t_ort = time_fn(lambda: sess.run(None, {"images": xn}))
            row = dict(scale=s, variant=v, imgsz=a.imgsz, train_params_M=params / 1e6,
                       infer_params_M=infer_params / 1e6, gflops=fl, torch_ms=t_pt, onnxruntime_ms=t_ort,
                       threads=a.threads)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    main()
