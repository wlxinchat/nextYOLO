# Experiments

All experiments ran on a **4-core CPU with no GPU** (15 GB RAM). COCO-scale training, with hundreds of GPU-epochs,
was out of reach, so every experiment uses a reduced but identical budget per comparison. These results measure
**relative effects under short schedules**. They are not COCO state-of-the-art numbers, and short-schedule rankings
can differ from long-schedule ones.

**Data:** Pascal VOC 2007+2012 trainval (16,551 images) for training; VOC 2007 test (4,952 images) for evaluation.
Objects flagged `difficult` are dropped. **Metric:** COCO-style AP@[.5:.95] and AP50 from nextYOLO's evaluator, which
matches pycocotools to 1e-6 (tests/test_metrics.py). Every model in a table, Ultralytics included, is scored by the
same evaluator on the same letterboxed inputs. "e2e" means the NMS-free one-to-one branch (top-k, no NMS). "o2m+NMS"
means the dense branch followed by class-aware NMS at IoU 0.7.

_Results are filled in as runs finish; see the status column._

## A. Training from scratch — is nextYOLO a faithful YOLO26?

Setting: 256 px, 12 epochs, batch 16 (nominal 64), MuSGD lr 0.01, 1 warmup epoch, mosaic off for the last 2
epochs, EMA, seed 0.

| run | AP e2e | AP50 e2e | AP o2m+NMS | AP50 o2m+NMS | status |
|---|---|---|---|---|---|
| Ultralytics YOLO26n (reference implementation) | **22.02** | **36.97** | **24.95** | **42.19** | done |
| nextYOLO-n, recipe v1 | 21.15 | 35.25 | 24.71 | 41.46 | done |
| nextYOLO-n, recipe v2 (+3× cls-head lr under MuSGD, as Ultralytics does) | | | | | running |

Recipe v1 matched the reference on the NMS branch (−0.24 AP) but trailed on the NMS-free branch (−0.87 AP).
Re-reading the reference optimiser code showed that Ultralytics trains both classification heads at 3× lr whenever
MuSGD is used. The code comment says "when finetuning", but the boost applies to every MuSGD run. It only affects
classification, which fits an o2o-specific gap. Recipe v2 adds it (`cls_lr_mult=3.0`).

At half schedule (epoch 6), v2 reaches 12.14 AP e2e against 11.40 for v1.

## B. Levers on top of YOLO26 (from scratch, same setting as A)

| lever | AP e2e @ epoch 6 | Δ vs v1 baseline @ epoch 6 | final | status |
|---|---|---|---|---|
| baseline (recipe v1) | 11.38 | — | 21.15 | done |
| **AOA** — o2o positive = top-1 of the o2m ranking (new) | 8.87 | **−2.51** | stopped at half schedule | negative |

**AOA is a negative result.** Forcing the one-to-one head to fire where the *one-to-many* head ranks best was 2.5 AP
worse at half schedule; only AP_S improved (3.12 vs 1.99). The likely explanation: with self-assignment
(YOLO26), the o2o head picks the anchor where *its own* box is already good, so its quality target is high and the
choice reinforces itself. AOA picks anchors where the o2o box is still poor, which lowers the IoU-aware targets and
contradicts the head's own ranking. The comparison is at equal schedule position (same LR schedule, both evaluated at
epoch 6); the run was not continued, to save compute.

## C. Transfer from COCO-pretrained YOLO26 (the practical regime)

VOC's 20 classes are a subset of COCO's 80. nextYOLO can therefore load the official YOLO26 weights and **slice the
classification heads to the 20 matching classes** (`subset_classes`), keeping all class knowledge. Setting: 320 px;
fine-tuning runs use 3 epochs of AdamW (lr 4.17e-4, which is what Ultralytics' `optimizer=auto` selects for this
budget), 0.5 warmup epochs, and mosaic off for the last epoch.

| run | AP e2e | AP50 e2e | AP o2m+NMS | AP50 o2m+NMS | status |
|---|---|---|---|---|---|
| YOLO26n COCO weights, zero-shot (class-subset heads, no VOC training) | 56.54 | 75.78 | — | — | done |
| YOLO26s COCO weights, zero-shot (distillation teacher) | 65.45 | 83.52 | — | — | done |
| Ultralytics fine-tune of yolo26n.pt (its trainer; cls heads re-initialised for 20 classes) | 47.67 | 65.79 | 54.55 | 75.28 | done |
| **nextYOLO fine-tune, class-subset head init** | **56.96** | **76.95** | **58.08** | **78.41** | done |
| nextYOLO fine-tune, re-initialised cls heads (isolates the head-init effect) | | | | | queued |
| nextYOLO fine-tune + dense distillation from YOLO26s (o2o assignment: self) | | | | | running |
| nextYOLO fine-tune + dense distillation, o2o assignment follows the teacher | | | | | queued |

Under the same 3-epoch budget, **class-subset head transfer beats standard fine-tuning by +9.3 AP e2e and +3.5 AP
o2m+NMS.** The gain is largest for the NMS-free branch: a re-initialised o2o head has to relearn "exactly one anchor
per object" from scratch. The re-initialised-head nextYOLO run separates the head-init effect from other
implementation differences.

Fine-tuning curve (e2e AP by epoch): 50.47 → 53.70 → 56.96. The first epoch of mosaic-augmented fine-tuning drops
6 AP below the zero-shot start before recovering. At this budget, fine-tuning only just beats zero-shot (+0.4 AP).

## Reproducing

```bash
python tools/prepare_voc.py --src VOCdevkit --dst VOC
# A: from scratch
python tools/train.py --data data/voc.json --imgsz 256 --epochs 12 --out runs/scratch \
  --set warmup_epochs=1 close_mosaic=2 eval_interval=6
python tools/baseline_ultralytics.py --data data/voc.json --model yolo26n.yaml --imgsz 256 --epochs 12 \
  --warmup_epochs 1 --close_mosaic 2 --out runs/ul_scratch
# C: transfer
python tools/val.py --ultralytics yolo26n.pt --data data/voc.json --imgsz 320
python tools/train.py --data data/voc.json --imgsz 320 --epochs 3 --out runs/ft --set init=yolo26n.pt \
  optimizer=adamw lr0=0.000417 momentum=0.9 warmup_bias_lr=0.0 warmup_epochs=0.5 close_mosaic=1 eval_interval=1
python tools/train.py ... --set init=yolo26n.pt distill.teacher=yolo26s.pt loss.o2o_assign=teacher
```

## Engineering notes (CPU budget)

* `torch.compile` gives +24% training throughput on CPU, and channels-last another +8%. Two 2-thread jobs in parallel
  get ~30% more total throughput than one 4-thread job.
* The JPEG RAM cache (quality 95) uses 6.3× less memory than raw arrays at 0.7 ms per decode. Before it, two 320 px
  jobs ran out of the 15 GB limit.
* Runs checkpoint their full state atomically every epoch and resume automatically. A stop file lets long runs pause
  gracefully at epoch boundaries, so the queue runs in bounded segments on a preemptible machine.
