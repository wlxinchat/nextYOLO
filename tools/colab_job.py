"""Self-contained GPU job for a fresh Colab (or any CUDA) VM: fetch nextYOLO, prepare VOC, train, archive results.

Run it either way:
    colab run --gpu A100 tools/colab_job.py --preset smoke              # Google Colab CLI, from a terminal
    !python colab_job.py --preset voc_scratch                           # inside a Colab notebook cell

Presets
    smoke        1 short epoch on 512 images at 320 px: proves the CUDA/AMP path end to end (~2 min)
    voc_scratch  nextYOLO-n from scratch, VOC07+12, 640 px, 100 epochs (YOLO26 recipe v2)
    voc_ft       fine-tune from official YOLO26n COCO weights (class-subset heads), 640 px, 30 epochs

Anything after `--set` is passed through to tools/train.py (e.g. `--set loss.o2o_cls_loss=mal seed=1`).
Already-finished steps (clone, download, conversion, completed runs) are skipped, so re-running resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = "https://github.com/wlxinchat/nextYOLO"
ASSETS = "https://github.com/ultralytics/assets/releases/download"
VOC_ZIPS = ["VOCtrainval_06-Nov-2007", "VOCtest_06-Nov-2007", "VOCtrainval_11-May-2012"]
VOC_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

PRESETS = {
    "smoke": dict(imgsz=320, epochs=1, batch=32,
                  sets=["max_train_images=512", "eval_max_images=500", "warmup_epochs=0.5", "close_mosaic=0",
                        "eval_interval=1"]),
    "voc_scratch": dict(imgsz=640, epochs=100, batch=64, sets=["warmup_epochs=3", "close_mosaic=10",
                                                              "eval_interval=10"]),
    "voc_ft": dict(imgsz=640, epochs=30, batch=64, init="yolo26n",
                   sets=["optimizer=adamw", "lr0=0.000417", "momentum=0.9", "warmup_bias_lr=0.0",
                         "warmup_epochs=1", "close_mosaic=5", "eval_interval=5"]),
}


def sh(cmd: list[str] | str, **kw) -> None:
    """Run a command and relay its output line by line (inside a Jupyter kernel a child's stdout would otherwise go
    to the kernel's terminal instead of the cell / `colab exec` stream)."""
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    with subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, **kw) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def fetch(url: str, dst: Path, attempts: int = 6) -> None:
    """Download with retries; an interrupted transfer resumes from the partial file (HTTP Range)."""
    if dst.exists():
        return
    print(f"downloading {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dst)
            return
        except OSError as e:  # URLError, connection reset, timeout
            if k == attempts - 1:
                raise
            print(f"  {type(e).__name__}: {e}; retrying ({tmp.stat().st_size if tmp.exists() else 0} bytes so far)",
                  flush=True)
            time.sleep(2 ** (k + 1))


def get_repo(work: Path, ref: str, repo_dir: str | None) -> Path:
    if repo_dir:
        return Path(repo_dir).resolve()
    d = work / "nextYOLO"
    if not (d / ".git").exists():
        sh(["git", "clone", "--depth", "1", "--branch", ref, REPO, str(d)])
    else:
        sh(["git", "-C", str(d), "pull", "--ff-only"])
    return d


def get_voc(work: Path, repo: Path, data_dir: str | None) -> Path:
    voc = Path(data_dir) if data_dir else work / "datasets" / "VOC"
    if not (voc / "labels" / "test2007").exists():
        raw = work / "datasets" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for z in VOC_ZIPS:
            done = raw / f"{z}.done"  # extracted already (the zip itself is deleted to save disk)
            if done.exists():
                continue
            fetch(f"{ASSETS}/v0.0.0/{z}.zip", raw / f"{z}.zip")
            sh(["unzip", "-q", "-o", str(raw / f"{z}.zip"), "-d", str(raw)])
            (raw / f"{z}.zip").unlink()
            done.touch()
        sh([sys.executable, str(repo / "tools" / "prepare_voc.py"), "--src", str(raw / "VOCdevkit"), "--dst", str(voc)])
    spec = {"root": str(voc), "train": ["images/train2007", "images/val2007", "images/train2012", "images/val2012"],
            "val": ["images/test2007"], "names": VOC_NAMES}
    path = work / "voc_job.json"
    path.write_text(json.dumps(spec))
    return path


def get_init_weights(work: Path, repo: Path, name: str) -> Path:
    """Official YOLO26 COCO weights (AGPL-3.0) converted to a nextYOLO checkpoint."""
    out = work / f"{name}_nextyolo.pt"
    if not out.exists():
        sh([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        pt = work / f"{name}.pt"
        fetch(f"{ASSETS}/v8.4.0/{name}.pt", pt)
        sh([sys.executable, str(repo / "tools" / "convert_ultralytics.py"), "--weights", str(pt), "--out", str(out)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    ap.add_argument("--work", default="/content" if Path("/content").exists() else str(Path.home() / "nextyolo_job"))
    ap.add_argument("--ref", default="claude/awesome-maxwell-zfo3s4", help="git branch/tag to clone")
    ap.add_argument("--repo-dir", default=None, help="use an existing checkout instead of cloning")
    ap.add_argument("--data-dir", default=None, help="use an existing prepared VOC root instead of downloading")
    ap.add_argument("--name", default=None, help="run name (default: the preset)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--scale", default="n")
    ap.add_argument("--init", default=None, help="official weights to fine-tune from, e.g. yolo26s (voc_ft preset)")
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides passed to tools/train.py")
    ap.add_argument("--eval", nargs="*", default=None, metavar="WEIGHTS:IMGSZ",
                    help="evaluate instead of training: official names (yolo26m) or checkpoint paths, e.g. "
                         "yolo26m:640 /content/runs/ft_s640/best.pt:768; reports NMS-free and NMS AP")
    a = ap.parse_args()

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    p = PRESETS[a.preset]
    repo = get_repo(work, a.ref, a.repo_dir)
    sh([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless", "pycocotools"])
    import torch  # noqa: E402  (after installs)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__}, CUDA device: {gpu}, CPUs: {os.cpu_count()}", flush=True)
    data = get_voc(work, repo, a.data_dir)

    if a.eval is not None:
        results = {}
        for spec in a.eval:
            w, _, size = spec.rpartition(":")
            w, size = (w, int(size)) if w else (spec, 640)
            if Path(w).exists():
                src = ["--weights", w]
            else:  # official Ultralytics weights, evaluated zero-shot with class-subset heads
                sh([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
                pt = work / f"{w}.pt"
                fetch(f"{ASSETS}/v8.4.0/{w}.pt", pt)
                src = ["--ultralytics", str(pt)]
            out_json = work / "evals" / f"{Path(w).parent.name + '_' if Path(w).exists() else ''}{Path(w).stem}_{size}.json"
            out_json.parent.mkdir(exist_ok=True)
            sh([sys.executable, str(repo / "tools" / "val.py"), *src, "--data", str(data), "--imgsz", str(size),
                "--mode", "both", "--workers", str(min(4, os.cpu_count() or 2)), "--out", str(out_json)], cwd=repo)
            r = json.loads(out_json.read_text())
            results[spec] = {m: {k: round(100 * r[m][k], 2) for k in ("mAP", "mAP50", "mAP75", "APs", "APm", "APl")}
                             for m in ("e2e", "nms")}
            print("EVAL", spec, json.dumps(results[spec]), flush=True)
        (work / "evals" / "summary.json").write_text(json.dumps(results, indent=1))
        return

    name = a.name or a.preset
    out = work / "runs" / name
    sets = [f"workers={min(8, os.cpu_count() or 2)}", "compile=true", *p["sets"]]
    init = a.init or p.get("init")
    if init:
        sets.append(f"init={get_init_weights(work, repo, init)}")
    sets += a.set
    cmd = [sys.executable, str(repo / "tools" / "train.py"), "--data", str(data), "--scale", a.scale,
           "--imgsz", str(a.imgsz or p["imgsz"]), "--epochs", str(a.epochs or p["epochs"]),
           "--batch", str(a.batch or p["batch"]), "--out", str(out), "--set", *sets]
    sh(cmd, cwd=repo)

    archive = work / f"nextyolo_{name}_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in ("summary.json", "history.json", "log.txt", "best.pt"):
            if (out / f).exists():
                tar.add(out / f, arcname=f"{name}/{f}")
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    best, nms = summary.get("best", {}), summary.get("final_o2m_nms", {})
    print(json.dumps({"run": name, "gpu": gpu, "AP_e2e": best.get("mAP"), "AP50_e2e": best.get("mAP50"),
                      "AP_o2m_nms": nms.get("mAP"), "hours": summary.get("hours"), "archive": str(archive)}),
          flush=True)
    if Path("/content/drive/MyDrive").exists():  # mounted Drive: keep a copy that survives the VM
        shutil.copy(archive, "/content/drive/MyDrive/")


if __name__ == "__main__":
    main()
