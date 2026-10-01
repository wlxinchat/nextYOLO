# nextYOLO

**An independent, hackable implementation of the strongest 2026 real-time detection recipe (NMS-free,
YOLO26-class), verified against the reference, with every design lever exposed as a switch and measured.**

* **Faithful at three levels.**
  * The architecture matches Ultralytics YOLO26 parameter-for-parameter at all five scales.
  * Official YOLO26 weights load directly and reproduce Ultralytics' NMS-free outputs to 2.4e-4.
  * Trained from scratch under an identical budget, it matches Ultralytics within run-to-run noise (22.08 vs 22.02 AP
    NMS-free, 25.02 vs 24.95 AP with NMS).
* **NMS-free end to end.** A dual-assignment head: a one-to-many branch for dense training and a one-to-one branch for
  inference. Inference is `convs → top-k → gather`, and the ONNX graph contains no NMS. DFL-free box regression.
* **Every lever is a flag.** Small-target-aware assignment (STAL), progressive o2m→o2o loss (ProgLoss), MuSGD (Muon +
  SGD), BCE/VFL/MAL/QFL classification losses, YOLO27-style dual-scale head, lossless SPD P2 fusion, area attention,
  class-subset head transfer, dense anchor-aligned distillation, and alternative one-to-one assignments.
* **Measured, not assumed.** A COCO-style evaluator that matches `pycocotools` to 1e-6, a harness that trains the
  Ultralytics reference under the same budget and scores it with the same evaluator, and ablation results reported
  with their noise level.
* **Small and dependency-light.** About 2,000 lines of PyTorch + OpenCV for the library, plus 700 lines of tools. No Ultralytics code is included (Ultralytics is
  used only as an optional external benchmark).

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the survey of the field (YOLO26/27, RF-DETR, DEIMv2, D-FINE, YOLOv12/13,
…) that motivated the design, and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the experiments.

## Quick start

```bash
pip install -r requirements.txt

# Pascal VOC -> YOLO layout (VOCdevkit from the official tarballs)
python tools/prepare_voc.py --src /path/to/VOCdevkit --dst /path/to/VOC   # then set "root" in data/voc.json

# train (any lever can be flipped with --set)
python tools/train.py --data data/voc.json --scale n --imgsz 640 --epochs 300 --out runs/voc_n \
    --set init=yolo26n_nextyolo.pt   # optional: start from COCO weights (class-subset heads)

# evaluate, export (NMS-free ONNX), benchmark
python tools/val.py --weights runs/voc_n/best.pt --data data/voc.json --imgsz 640
python tools/export.py --weights runs/voc_n/best.pt --imgsz 640 --out nextyolo_n.onnx
python tools/benchmark.py --scales n s --imgsz 640

# start from official YOLO26 COCO weights (AGPL-3.0 weights; heads sliced to the dataset's classes)
python tools/convert_ultralytics.py --weights yolo26n.pt --out yolo26n_nextyolo.pt --check
python tools/val.py --ultralytics yolo26n.pt --data data/voc.json --imgsz 640   # zero-shot COCO -> VOC
```

### Running on a GPU (Google Colab)

The trainer uses CUDA automatically when it's available, with bf16 autocast on A100/L4/H100 and fp16 with gradient
scaling on T4. `tools/colab_job.py` is a self-contained job: it clones this repo, downloads and converts VOC, trains a
preset (`smoke`, `voc_scratch`, `voc_ft`), and archives the results. It can be launched two ways:

* **Notebook:** open [`notebooks/nextyolo_colab.ipynb`](https://colab.research.google.com/github/wlxinchat/nextYOLO/blob/claude/awesome-maxwell-zfo3s4/notebooks/nextyolo_colab.ipynb)
  in Colab, select a GPU runtime and run the cells.
* **Terminal / agents:** use Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli) (Python ≥ 3.12):
  `uv tool install google-colab-cli`, then `colab run --gpu A100 tools/colab_job.py --preset voc_scratch`. The CLI
  signs in with a copy-paste OAuth flow that works on headless machines. Network-restricted sandboxes must allow
  `colab.research.google.com` and `*.colab.dev`.

Dataset format: `images/<split>/*.jpg` with `labels/<split>/*.txt` (`cls cx cy w h`, normalised), described by a JSON
file such as [`data/voc.json`](data/voc.json).

## Architecture

```
image ─► Conv s2 ─► Conv s2 ─► C3k2 ─► Conv s2 ─► C3k2 ─► Conv s2 ─► C3k2 ─► Conv s2 ─► C3k2 ─► SPPF ─► C2PSA
                                           │P3                  │P4                               │P5 (attention)
                                           ▼                    ▼                                 ▼
                                    PAN neck (top-down + bottom-up C3k2; P5 output uses C3k2 + attention)
                                           │                    │                                 │
                        ┌──────────────────┴────────────────────┴─────────────────────────────────┘
                        ▼
   one-to-many head (training only, top-10 TAL positives)  ─┐  shared matching metric s^0.5·IoU^6
   one-to-one head  (detached features, top-1 positive)    ─┘  → inference: sigmoid → top-k → boxes
```

| scale | params: train (both heads) | params: inference (fused, o2o only) | GFLOPs @640 | official YOLO26 (params / GFLOPs) |
|---|---|---|---|---|
| n | 2,572,280 | 2.409M | 5.48 | 2.4M / 5.4 |
| s | 10,009,784 | 9.496M | 20.93 | 9.5M / 20.7 |
| m | 21,896,248 | 20.411M | 68.43 | 20.4M / 68.2 |
| l | 26,299,704 | 24.807M | 86.80 | 24.8M / 86.4 |
| x | 58,993,368 | 55.726M | 194.42 | 55.7M / 193.9 |

Training parameter counts are identical to Ultralytics' model summaries for `yolo26{n,s,m,l,x}.yaml`.

## Levers (all in `ModelConfig` / `LossConfig` / `TrainConfig`)

| flag | default | what it does | origin |
|---|---|---|---|
| `model.end2end` | `true` | o2o NMS-free head (+ o2m auxiliary head in training) | YOLOv10, YOLO26 |
| `model.levels` | `[3,4,5]` | `[3,5]` = dual-scale head | YOLO27 n/s |
| `model.p2_fusion` | `false` | lossless space-to-depth P2 map fused into the P3 neck node | YOLO27 "stronger high-res features"; SPD-Conv |
| `model.attn_area` | `1` | area attention (split tokens into stripes) in P5 attention | YOLOv12 |
| `model.o2o_grad_scale` | `0` | let o2o gradients reach the backbone, scaled | ablation |
| `loss.o2o_assign` | `self` | `o2m` = AOA (o2o positive = top-1 of the o2m ranking), `teacher` — **both measured worse; keep `self`** | nextYOLO (negative result) |
| `loss.cls_loss` / `loss.o2o_cls_loss` | `bce` | `vfl`, `mal`, `qfl` — **MAL on o2o measured −4.1 AP e2e** | VarifocalNet, DEIM, GFL |
| `init`, `init_head` | — / `subset` | start from nextYOLO or YOLO26 weights; `subset` slices cls heads by class name — **+7.7 AP e2e** | nextYOLO |
| `distill.*` | off | dense anchor-aligned KD; `distill.branches=["o2m"]` — never distil the o2o branch (−4.3 AP) | nextYOLO |
| `loss.stal` | `true` | small GT boxes enlarged to stride₂ for candidate selection | YOLO26 |
| `loss.prog_loss` | `true` | o2m weight 0.8 → 0.1 linearly over training | YOLO26 |
| `optimizer` | `musgd` | `sgd`, `adamw` | YOLO26 (Muon: K. Jordan et al.) |
| `cls_lr_mult` | `3.0` | lr multiplier for both cls heads under MuSGD — needed for parity (+0.84 AP e2e) | YOLO26 |
| `model.levels=[3,5]` | — | dual-scale head — measured −1.1 AP e2e, −15% ONNX latency | YOLO27 n/s |

## Results

Everything was trained on a 4-core CPU with no GPU, so the experiments use reduced budgets: Pascal VOC 07+12 →
VOC07 test, all models scored by the same pycocotools-equivalent evaluator. They show which levers help under those
budgets, not COCO-scale state of the art. Full details, including what each number does and does not show, are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

**Best models obtained (VOC07 test, AP@[.5:.95] / AP50, NMS-free unless noted):**

| model | 320 px | 640 px | how |
|---|---|---|---|
| YOLO26s COCO weights, class-subset heads | 65.45 / 83.52 | **68.16 / 85.55** | zero-shot, no VOC training |
| YOLO26n COCO weights, class-subset heads | 56.54 / 75.78 | 62.61 / 81.36 | zero-shot, no VOC training |
| nextYOLO-n fine-tuned from YOLO26n (3 epochs) | 56.96 / 76.95 (58.08 / 78.41 with NMS) | — | class-subset head init |
| Ultralytics fine-tune of YOLO26n (3 epochs) | 47.67 / 65.79 (54.55 / 75.28 with NMS) | — | standard trainer, cls heads re-initialised |

**What the experiments established** (noise level between seeds: ~0.2 AP):

| finding | effect |
|---|---|
| Parity with YOLO26 requires its 3× cls-head lr under MuSGD (easy to miss) | +0.84 AP NMS-free |
| **Class-subset head transfer** from COCO weights vs re-initialised heads | **+7.7 AP** NMS-free, +2.7 AP with NMS |
| The **one-to-one head must stay self-consistent**: an external ranking for its assignment (AOA from o2m, or from a teacher) or a distilled o2o score map | −2.5 to −6.5 AP NMS-free |
| Distil only the dense o2m branch (o2o left alone) | neutral (−0.2) with a zero-shot COCO teacher |
| DEIM's MAL loss on a dense o2o head (it down-weights the near-duplicate negatives an NMS-free head must suppress) | −4.1 AP NMS-free |
| YOLO27-style dual-scale head without compensation | −1.1 AP, −15% latency |
| NMS-free inference cost, nextYOLO-n @640, 4-thread CPU, ONNX Runtime | 23.9 ms, no post-processing |

**Recommendation.** For the best accuracy on a new task, take the largest real-time model your latency budget allows,
initialise it from detection-pretrained weights with class-subset heads where the label spaces overlap, and fine-tune
on a GPU with a long schedule. For the best NMS-free nano model, use the YOLO26 recipe as implemented here; spend
extra effort on data and pretraining, not on re-wiring the one-to-one assignment. Beyond the nano scale, the field's
frontier is query-based decoders on foundation-model backbones (RF-DETR, DEIMv2, YOLO27 m/l; see
[`docs/RESEARCH.md`](docs/RESEARCH.md)).

## Tests

```bash
python -m pytest tests -q
```

The tests cover box ops against torchvision, the evaluator against pycocotools, assigner invariants (one positive per
object in the o2o branch, STAL candidates, AOA = o2m top-1), gradient isolation of the o2o branch, BN fusion
equivalence, NMS-free ONNX export parity, MuSGD, distillation, resume after interruption, deadline-aware pausing, and
(optionally) parity with the official YOLO26 weights.

## Licence note

nextYOLO's code is written independently, so the repository owner can choose its licence. Official YOLO26 weights,
and any model fine-tuned from them, remain under AGPL-3.0.
