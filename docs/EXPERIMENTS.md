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

All runs below completed; the per-run summaries, learning curves and eval JSONs are in [`results/`](../results).

## A. Training from scratch — is nextYOLO a faithful YOLO26?

Setting: 256 px, 12 epochs, batch 16 (nominal 64), MuSGD lr 0.01, 1 warmup epoch, mosaic off for the last 2
epochs, EMA, seed 0.

| run | AP e2e | AP50 e2e | AP o2m+NMS | AP50 o2m+NMS | status |
|---|---|---|---|---|---|
| Ultralytics YOLO26n (reference implementation) | **22.02** | **36.97** | **24.95** | **42.19** | done |
| nextYOLO-n, recipe v1 | 21.15 | 35.25 | 24.71 | 41.46 | done |
| **nextYOLO-n, recipe v2** (+3× cls-head lr under MuSGD, as Ultralytics does) | **21.99** | **36.76** | **24.99** | **41.79** | done |
| nextYOLO-n, recipe v2, seed 1 | 22.16 | 37.33 | 25.05 | 42.33 | done |

Recipe v1 matched the reference on the NMS branch (−0.24 AP) but trailed on the NMS-free branch (−0.87 AP).
Re-reading the reference optimiser code showed that Ultralytics trains both classification heads at 3× lr whenever
MuSGD is used. The code comment says "when finetuning", but the boost applies to every MuSGD run. It only affects
classification, which fits an o2o-specific gap. Recipe v2 adds it (`cls_lr_mult=3.0`).

**With v2 the two implementations agree within 0.03–0.04 AP on both branches** (e2e 21.99 vs 22.02, NMS 24.99 vs
24.95). Together with the weight-level check — official YOLO26 weights loaded into nextYOLO reproduce Ultralytics'
outputs to 2.4e-4 — this establishes nextYOLO as a faithful YOLO26 implementation at both the architecture level and
the training-recipe level. The 3× cls-head lr alone was worth +0.84 AP on the NMS-free branch. Recipe v2 is the
baseline for all later from-scratch ablations.

**Noise level:** two seeds of recipe v2 differ by 0.17 AP e2e and 0.06 AP o2m+NMS. Their mean is 22.08 AP e2e and
25.02 AP o2m+NMS; Ultralytics scores 22.02 and 24.95. Below we treat differences under ~0.3 AP as noise.

## B. Levers on top of YOLO26 (from scratch, same setting as A)

| lever | AP e2e @ epoch 6 | AP e2e final | AP o2m+NMS final | Δ e2e vs baseline | verdict |
|---|---|---|---|---|---|
| baseline, recipe v1 | 11.38 | 21.15 | 24.71 | — | |
| **AOA** — o2o positive = top-1 of the o2m ranking (new; vs v1) | 8.87 | stopped at epoch 6 | — | **−2.51** @ epoch 6 | negative |
| baseline, recipe v2 (mean of 2 seeds) | 12.27 | 22.08 | 25.02 | — | |
| **MAL** for the o2o branch (DEIM's matchability-aware loss; vs v2) | 9.84 | 18.02 | 24.46 | **−4.06** | negative |
| YOLO27-style dual-scale head, P3+P5 only (vs v2) | 10.92 | 20.98 | 23.59 | **−1.10** (−1.43 o2m+NMS) | negative for accuracy; −15% latency |

**AOA is a negative result.** Forcing the one-to-one head to fire where the *one-to-many* head ranks best was 2.5 AP
worse at half schedule; only AP_S improved (3.12 vs 1.99). The likely explanation: with self-assignment
(YOLO26), the o2o head picks the anchor where *its own* box is already good, so its quality target is high and the
choice reinforces itself. AOA picks anchors where the o2o box is still poor, which lowers the IoU-aware targets and
contradicts the head's own ranking. The comparison is at equal schedule position (same LR schedule, both evaluated at
epoch 6); the run was not continued, to save compute.

**The dual-scale head trades accuracy for speed.** Dropping the P4 output (YOLO27 n/s style) costs 1.1 AP e2e and
1.4 AP o2m+NMS at this scale, and makes ONNX inference 15% faster (20.4 vs 23.9 ms, section D). YOLO27 pairs the
dual-scale head with "strengthened high-resolution features" to make up the loss; removing P4 alone does not.

**MAL is also a negative result for dense NMS-free heads** (−4.1 AP e2e, −0.6 AP o2m+NMS). MAL down-weights negatives
by p^γ, focal-style. In a dense one-to-one head, the hardest negatives are the near-duplicate neighbours of each
positive, and suppressing them is exactly what makes the head NMS-free, so weakening their gradient leaves
duplicates. In DEIM's DETR decoder, query self-attention does the de-duplication, so MAL's weighting there is harmless
or even useful. The loss is only 0.6 AP on the dense branch, where NMS removes duplicates, which supports this
explanation.

## C. Transfer from COCO-pretrained YOLO26 (the practical regime)

VOC's 20 classes are a subset of COCO's 80. nextYOLO can therefore load the official YOLO26 weights and **slice the
classification heads to the 20 matching classes** (`subset_classes`), keeping all class knowledge. Setting: 320 px;
fine-tuning runs use 3 epochs of AdamW (lr 4.17e-4, which is what Ultralytics' `optimizer=auto` selects for this
budget), 0.5 warmup epochs, and mosaic off for the last epoch.

| run | AP e2e | AP50 e2e | AP o2m+NMS | AP50 o2m+NMS | status |
|---|---|---|---|---|---|
| YOLO26n COCO weights, zero-shot (class-subset heads, no VOC training) | 56.54 | 75.78 | — | — | done |
| YOLO26s COCO weights, zero-shot (distillation teacher) | 65.45 | 83.52 | — | — | done |
| YOLO26n COCO weights, zero-shot, **640 px** | 62.61 | 81.36 | — | — | done |
| YOLO26s COCO weights, zero-shot, **640 px** | **68.16** | **85.55** | — | — | done |
| Ultralytics fine-tune of yolo26n.pt (its trainer; cls heads re-initialised for 20 classes) | 47.67 | 65.79 | 54.55 | 75.28 | done |
| **nextYOLO fine-tune, class-subset head init** | **56.96** | **76.95** | **58.08** | **78.41** | done |
| nextYOLO fine-tune, re-initialised cls heads (isolates the head-init effect) | 49.22 | 67.83 | 55.40 | 76.12 | done |
| nextYOLO fine-tune + dense distillation from YOLO26s (o2o assignment: self) | 52.44 | 71.95 | 57.06 | 77.66 | done |
| nextYOLO fine-tune + dense distillation, o2o assignment follows the teacher | 50.49 | 68.32 | 57.03 | 77.69 | done |
| nextYOLO fine-tune + distillation of the o2m branch only | 56.78 | 77.32 | 57.24 | 77.90 | done |

Under the same 3-epoch budget, **class-subset head transfer beats standard fine-tuning by +9.3 AP e2e and +3.5 AP
o2m+NMS.** Within nextYOLO alone, the head initialisation accounts for **+7.7 AP e2e and +2.7 AP o2m+NMS** (56.96 vs
49.22, 58.08 vs 55.40). The rest (+1.5 / +0.9) comes from differences between the two trainers in this fine-tuning
setting, within the range expected for 3-epoch runs. The gain is largest for the NMS-free branch: a re-initialised
o2o head has to relearn "exactly one anchor per object" from scratch, while the sliced head keeps it.

Fine-tuning curve (e2e AP by epoch): 50.47 → 53.70 → 56.96. The first epoch of mosaic-augmented fine-tuning drops
6 AP below the zero-shot start before recovering. At this budget, fine-tuning only just beats zero-shot (+0.4 AP).

**Distillation with o2o self-assignment hurts:** −4.5 AP e2e and −1.0 AP o2m+NMS against plain fine-tuning (by epoch:
44.20 → 49.07 → 52.44). Most of the loss (about 3.5 of 4.5 AP) is specific to the NMS-free branch. This fits the
conflict predicted before the run: the n student and s teacher fire their o2o outputs at the same anchor only about 40%
of the time, so distilling the teacher's o2o map while the ground-truth loss assigns the o2o positive from the
student's *own* ranking pushes the o2o head in two directions.

**That hypothesis did not hold.** Letting the teacher's o2o ranking choose the o2o positive, so the ground-truth and
distillation losses agree, made the NMS-free branch *worse* (50.49 vs 52.44 AP). It matches the AOA result: in every
variant tried, taking the one-to-one assignment from any ranking other than the o2o head's own hurts. The student did
learn to mimic the teacher (the KL term fell from about 11.8 to 3.1), yet distillation also cost ~1 AP on the dense
branch. The remaining suspect is the teacher's targets themselves. The teacher is a COCO model evaluated zero-shot, so
its soft labels carry COCO annotation conventions (box extents, objects VOC marks `difficult` and drops, near-miss
classes such as truck/car) that conflict with VOC ground truth. The standard remedy, a teacher fine-tuned on VOC, costs
about 4 CPU-hours for YOLO26s and was not run.

**The o2m-only run separates the two effects.** With distillation applied only to the dense branch, the NMS-free AP
is back to 56.78, within noise of plain fine-tuning (56.96). **Distilling the one-to-one score map therefore caused
about 4.3 of the 4.5 AP e2e loss.** The remaining ~0.8 AP loss on the dense branch (57.24 vs 58.08) is what the
zero-shot COCO teacher's targets cost.

### A consistent principle: the one-to-one head must stay self-consistent

Three independent interventions imposed an external ranking on the NMS-free branch, and all three hurt it badly:

| intervention on the o2o branch | Δ AP e2e |
|---|---|
| assignment from the o2m branch's ranking (AOA) | −2.5 (at half schedule) |
| assignment from a teacher's o2o ranking (+ KD) | −6.5 |
| distilling a teacher's o2o score map | −4.3 |
| distilling only the dense o2m branch (o2o left alone) | −0.2 (noise) |

A one-to-one head works by committing to **one** anchor per object and suppressing that anchor's near-identical
neighbours. Which anchor wins is arbitrary but has to be *self-consistent*: the anchor where the head's own box and
score are best, so that targets, scores and box quality reinforce each other. Any external opinion about which anchor
should fire — another branch, another model — breaks that consistency, even when the external opinion comes from a
stronger model. For dual-head NMS-free detectors: **distil or regularise the dense branch, and leave the one-to-one
branch to its self-assigned ground-truth loss.**

## D. Inference cost (NMS-free, batch 1, 4 CPU threads, 640 px)

| scale | variant | params (inference) | GFLOPs | PyTorch eager ms | ONNX Runtime ms |
|---|---|---|---|---|---|
| n | default (P3/P4/P5) | 2.409M | 5.48 | 58.9 | **23.9** |
| n | dual-scale (P3/P5) | 2.363M | 5.33 | 45.2 | 20.4 |
| n | + SPD P2 fusion | 2.429M | 5.74 | 49.9 | 30.4 |
| s | default | 9.496M | 20.93 | 108.0 | **57.6** |
| s | dual-scale | 9.349M | 20.47 | 95.0 | 60.9 |
| s | + SPD P2 fusion | 9.578M | 21.98 | 105.0 | 74.8 |

The exported graph is convolutions + top-k + gathers, with no NMS and no post-processing. Eager PyTorch timings on
CPU are noisy; the ONNX Runtime numbers are the reliable ones. SPD P2 fusion adds only 5% FLOPs but 27% ONNX latency,
because pixel-unshuffle at high resolution is memory-bound. It was not trained, given that cost.

## E. GPU runs (Google Colab, free Tesla T4)

These runs were launched from this sandbox with the Google Colab CLI (`tools/colab_job.py`). Settings: 640 px, 10
epochs, AdamW (lr 4.17e-4), fp16 autocast, batch 32 (s) / 64 (n), mosaic off for the last 2 epochs, class-subset head
initialisation from the official COCO weights. The free T4 has only 2 CPU cores, so the augmentation pipeline caps
training at ~45 img/s at 640 px, and the GPU waits on data.

| model | AP e2e | AP50 e2e | AP o2m+NMS | AP50 o2m+NMS | GPU hours |
|---|---|---|---|---|---|
| YOLO26s COCO weights, zero-shot (reference, section C) | 68.16 | 85.55 | — | — | — |
| **nextYOLO-s fine-tuned on VOC** | **69.72** | **87.63** | **70.79** | **88.70** | 1.09 |
| YOLO26n COCO weights, zero-shot (reference, section C) | 62.61 | 81.36 | — | — | — |
| nextYOLO-n fine-tuned on VOC | 65.97 | 85.02 | 66.75 | 85.93 | 0.95 |
| **nextYOLO-n fine-tuned + distillation from the VOC-tuned nextYOLO-s (o2m branch only)** | **66.70** | **85.44** | **67.54** | **86.48** | 1.03 |

Learning curve (AP e2e by epoch): 54.04 (2) → 60.01 (4) → 64.12 (6) → 68.36 (8) → 69.72 (10). As in the CPU runs,
mosaic fine-tuning first drops below the zero-shot start; the last two epochs, without mosaic and at the lowest lr,
bring the largest gain.

The free VM was reclaimed after about an hour, during epoch 8. The run resumed on a fresh T4 from the epoch-7
checkpoint, which had been backed up to this sandbox every few minutes. The 160 MB checkpoint was uploaded in 20 MB
chunks (single large uploads were cut off) and verified by SHA-256.

**Distillation works once the teacher is adapted and the o2o branch is left alone.** With the VOC-fine-tuned s
model (69.72 AP) as teacher and distillation applied only to the dense o2m branch, the n student gains **+0.73 AP
NMS-free and +0.79 AP with NMS** (AP75 +1.07, AP_S +1.67). It is ahead at every evaluation point: +0.82, +2.68,
+0.46, +0.75 and +0.73 AP at epochs 2/4/6/8/10. The KD loss starts 5× lower than with the zero-shot COCO teacher in
section C (0.57 vs 2.96), which supports the earlier diagnosis that the COCO teacher's targets conflicted with VOC
labels. Caveat: one seed per arm; the per-epoch consistency is the main evidence that the gain is real.

### Accuracy frontier and test resolution (GPU evaluation, VOC07 test)

The GPU evaluation reproduces the CPU numbers exactly (YOLO26s zero-shot 68.16, fine-tuned s 69.72), so results from
the two machines are comparable.

| model @ 640 px | AP e2e | AP50 e2e | AP_S e2e | AP o2m+NMS |
|---|---|---|---|---|
| YOLO26s zero-shot | 68.16 | 85.55 | 23.72 | 68.93 |
| nextYOLO-s fine-tuned (above) | 69.72 | 87.63 | 25.28 | 70.79 |
| YOLO26m zero-shot | 70.44 | 86.88 | 22.47 | 71.21 |
| YOLO26l zero-shot | 71.27 | 87.25 | 25.83 | 71.78 |
| YOLO26x zero-shot | **72.27** | 87.58 | 31.31 | **72.96** |

Capacity dominates: m/l/x beat the fine-tuned s without any VOC training. **Larger test resolution does not help**
models trained at 640 px. Fine-tuned s scores 69.72 / 69.09 / 67.03 / 63.52 AP e2e at 640 / 704 / 768 / 832. AP_S
rises (25.3 → 31.5), but medium and large objects lose more. The distilled n behaves the same (66.70 → 66.38 →
64.96 at 640 / 704 / 768).

### Fine-tuning recipe search (YOLO26s → VOC, 640 px, 10 epochs)

| recipe | AP e2e @ epoch 2 / 4 / 6 / 8 / 10 | final AP o2m+NMS | GPU hours |
|---|---|---|---|
| **mosaic, mosaic off for last 2 epochs, AdamW lr 4.17e-4** (baseline above) | 54.04 / 60.01 / 64.12 / 68.36 / **69.72** | **70.79** | 1.09 |
| no mosaic at all | 48.53 / 58.39 / 62.26 / 66.70 / 68.83 | 69.91 | 0.89 |

**Mosaic helps fine-tuning (+0.9 AP)**, even though it speeds data loading up ~1.4× to remove it on the 2-core T4
VM. The drop below the zero-shot start (68.16 → 48.5–54 AP after 2 epochs) happens with or without mosaic. So it is
an optimisation shock from the learning rate, not an augmentation effect.

**Lowering the fine-tuning lr removes the drop and is the biggest recipe gain found.**

| run | lr | AP e2e by epoch | final AP o2m+NMS |
|---|---|---|---|
| YOLO26s → VOC, 640 px (GPU) | 4.17e-4 | 54.04 (ep 2) … 69.72 (ep 10) | 70.79 |
| YOLO26s → VOC, 640 px (GPU) | **1e-4** | **69.35 (ep 2)**, then the VM was reclaimed | — |
| YOLO26n → VOC, 320 px, 3 ep (CPU) | 4.17e-4 | 50.47 / 53.70 / 56.96 | 58.08 |
| YOLO26n → VOC, 320 px, 3 ep (CPU) | **1e-4** | **56.54 / 58.14 / 58.96** | **59.42** |
| YOLO26n → VOC, 320 px, 3 ep (CPU) | 3e-5 | 56.51 / 57.61 / 57.92 | 58.30 |

The lr that Ultralytics' `optimizer=auto` picks for short schedules (4.17e-4 for 20 classes) is too high to fine-tune
a model that already knows the classes. It wipes out ~6–14 AP of pretrained accuracy in the first epoch, and the rest
of the schedule is spent recovering it. At 1e-4 the start is preserved and every epoch adds accuracy: +2.0 AP e2e at
n/320 after 3 epochs, and after 2 epochs the s/640 run is already level with the old recipe's 10-epoch result. 3e-5
is too conservative.

With lr 1e-4 (YOLO26n → VOC, 320 px, CPU):

| run | AP e2e by epoch | final AP o2m+NMS |
|---|---|---|
| 3 epochs | 56.54 / 58.14 / 58.96 | 59.42 |
| 6 epochs | 56.54 / 57.95 / 58.65 / 59.01 / 59.37 / 60.03 | 60.70 |
| 6 epochs, lr 2e-4 (control) | 55.85 / 56.91 / 57.73 / 58.02 / 58.95 / 59.38 | 60.18 |
| 3 epochs + distillation from the VOC-tuned s (o2m only) | 57.34 / 59.03 / 59.19 | 59.94 |
| **6 epochs + distillation from the VOC-tuned s (o2m only)** | 57.34 / 58.78 / 58.91 / 59.56 / 59.58 / **60.43** | **61.01** (AP50 81.0) |

Longer training still pays at the new lr: +1.1 AP from 3 to 6 epochs, and the curve is still rising. Distillation
adds on top of the lr gain (+0.23 AP e2e, +0.52 AP o2m+NMS; +0.8–0.9 AP at epochs 1–2). The gain is smaller than at
640 px on the GPU (+0.73), probably because this teacher was trained at 640 px and is weaker at 320.

At 6 epochs lr 2e-4 is behind 1e-4 at every epoch (−0.65 AP final), so 1e-4 is the best lr we found, not just a
short-schedule effect. Distillation still adds +0.40 AP e2e / +0.31 AP o2m+NMS at 6 epochs. Its lead is largest
early (+0.8 AP at epochs 1–2) and settles at about +0.4 AP. It costs about 1.7× the training time on CPU for the
teacher forward. Best n/320 fine-tuning recipe: **lr 1e-4, mosaic + close_mosaic, longer schedule, o2m-only KD from a
target-tuned teacher**. Result: 60.43 AP NMS-free (+3.9 over zero-shot, +3.5 over the original recipe).

### Fine-tuning the largest model: YOLO26x → VOC (640 px, lr 1e-4, 3 epochs, Colab T4)

The recipe found above, applied to YOLO26x (58.9M params, 194 GFLOPs): AdamW lr 1e-4, batch 8 (nominal 64),
mosaic with the last epoch mosaic-free, class-subset head init. One run, 1.37 GPU hours.

| epoch | AP e2e | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|
| 0 (zero-shot) | 72.27 | 87.58 | 79.95 | 31.31 | 56.59 | 80.82 |
| 1 (mosaic) | 70.29 | 87.27 | 78.58 | 35.34 | 57.86 | 75.99 |
| 2 (mosaic) | 73.84 | 90.32 | 82.42 | 36.23 | 61.23 | 79.63 |
| **3 (no mosaic)** | **76.10** | **91.67** | **84.81** | **38.22** | **63.38** | **82.08** |
| 3, o2m + NMS | 77.19 | 92.66 | 85.38 | 37.95 | 64.08 | 82.70 |

**+3.8 AP NMS-free over the zero-shot start, and 6.4 AP above the best fine-tuned s.** This is the best model in this
study. The epoch-1 dip (−2.0) is the mosaic distribution shift, not an lr shock: AP_S already rises at epoch 1, and
the mosaic-free last epoch adds +2.3. The curve is still steep at the end, so a longer schedule should pay further.

The first attempt was lost when the free VM was reclaimed 19 minutes into a 21-minute epoch. Since then the trainer
writes a mid-epoch checkpoint every `save_interval_min` minutes. The run above saved one every 8 minutes and the
session synced each to the local machine (0.94 GB, ~80 s), so a reclaim costs at most ~8 minutes of training.

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
