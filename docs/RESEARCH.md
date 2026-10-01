# The state of real-time object detection (September 2026)

This survey motivated nextYOLO's design. It covers the YOLO line (YOLOv10 → YOLO27) and the real-time DETR line
(RT-DETR → D-FINE → DEIM/DEIMv2 → RF-DETR), and ranks the techniques by how much evidence supports them. The COCO
numbers are the ones the authors report; latency figures come from different hardware and harnesses, so compare them
only within a row.

## 1. The frontier at a glance

| Model (release) | Family | Largest COCO AP (val) | Smallest model | Pretraining | License |
|---|---|---|---|---|---|
| YOLO27 l (preview, Sep 2026) | CNN n/s; query decoder m/l | **60.4** @640, 61.2 @800 | n: 42.3 | not public | AGPL (Ultralytics) |
| RF-DETR 2XL (ICLR 2026) | DETR + DINOv2 backbone + NAS | **60.1** | N: 48.4 (384 px, 30.5M params) | DINOv2 + Objects365 | Apache-2.0 |
| DEIMv2 X (Sep 2025) | DETR + DINOv3 features | 57.8 | Atto/Femto/Pico (ultra-light) | DINOv3 distillation | Apache-2.0 |
| YOLO26 x (Sep 2025 / paper Jun 2026) | CNN, NMS-free o2o head | 57.5 (56.9 e2e) | n: 40.9 (40.1 e2e), 2.4M params | Objects365 | AGPL (Ultralytics) |
| D-FINE X (+O365) (ICLR 2025) | DETR, distribution refinement | 59.3 (55.8 COCO only) | N: 42.8 | Objects365 (optional) | Apache-2.0 |
| "YOLOv14" x (Aug 2026, third-party) | CNN + deformable area attention, cross-domain training | 56.5 | s: 49.1 | — | — |
| VajraV1 X (Dec 2025) | CNN, combines prior YOLO components | 56.2 | Nano: 44.3 | none | — |
| YOLOv13 X (Jun 2025) | CNN + hypergraph attention | 54.8 | N: 41.6 | none | AGPL |
| YOLOv12 X (Feb 2025) | CNN + area attention | 55.2 | N: 40.6 | none | AGPL |
| YOLO-ULM (CVPR 2026) | lightweight CNN | L/X +0.7/+0.8 over YOLOv13 | N: 41.6 @ 1.52 ms T4 | none | — |

**What the table says.**
1. The 60-AP barrier fell in 2026. Both routes over it use a **query-based transformer decoder at the large
   scales**: RF-DETR with a DINOv2 backbone, and YOLO27 m/l, which switched to a query-based decoder.
2. **Every recent leader is NMS-free.** It is done either with a DETR decoder or with a one-to-one (o2o) dense head
   (YOLOv10 → YOLO26 → YOLO27 n/s).
3. At the **nano scale, CNNs with o2o heads still win on latency**: YOLO26n, YOLO27n, YOLO-ULM-N at 41–42 AP in about
   1.5–1.7 ms on a T4. RF-DETR-N reaches 48.4 AP but carries 30.5M parameters.
4. **Pretraining is now a large, often under-credited part of the headline numbers.** Objects365 accounts for
   +1.5–3.5 AP (D-FINE-X: 55.8 → 59.3), and foundation-model backbones (DINOv2/v3) add the rest of the DETR-line
   gains.

## 2. The techniques, ranked by evidence

| # | Technique | Where | Evidence of gain | Deployment cost |
|---|---|---|---|---|
| 1 | Foundation-model backbone (DINOv2/DINOv3) | RF-DETR, DEIMv2 | largest single lever; RF-DETR-N is +5.3 AP over D-FINE-N at similar latency | heavy ViT backbone; needs the pretrained weights |
| 2 | Large-scale detection pretraining (Objects365) | D-FINE, RF-DETR, YOLO26 | +1.5–3.5 AP | compute only |
| 3 | NMS-free one-to-one prediction | YOLOv10, YOLO26/27, all DETRs | removes NMS latency and tuning; YOLO26 e2e is only −0.6..−0.8 AP from its NMS branch | none; simplifies export |
| 4 | Dual assignment (o2m auxiliary + o2o inference) with a shared ("consistent") matching metric | YOLOv10, YOLO26 | makes o2o trainable with dense-detector efficiency | o2m head dropped at inference |
| 5 | Better o2o supervision: Dense O2O and Matchability-Aware Loss (MAL) | DEIM/DEIMv2 | faster DETR convergence; DEIM-D-FINE-X +0.7 AP | none |
| 6 | Progressive o2m→o2o loss re-weighting (ProgLoss) | YOLO26 | best e2e AP with the (0.8, 0.2)→(0.1, 0.9) schedule (YOLO26 paper) | none |
| 7 | Fine-grained distribution refinement (FDR) + self-distillation (GO-LSD) | D-FINE, DEIM | +~1–2 AP on DETRs | extra decoder ops |
| 8 | DFL removal: direct ltrb regression | YOLO26 | ≈ accuracy-neutral; simpler, quantisation-friendly graph | negative cost |
| 9 | Attention in the deepest stages (PSA, area attention, hypergraph) | YOLO11/12/13 | about +1–1.5 AP per generation at the n scale | small at P5; area attention for high resolution |
| 10 | Muon-style optimisers (MuSGD: Newton–Schulz-orthogonalised momentum mixed with SGD) | YOLO26 | +0.4 AP and a shorter schedule (600 → 500 epochs) | training only |
| 11 | Small-target-aware assignment (STAL) | YOLO26 | +0.2 AP, +0.6 AP_S (YOLO11s, s_ref = 16) | none |
| 12 | Scale-adaptive design: dual-scale P3+P5 heads for n/s, query decoders for m/l | YOLO27 | fewer head FLOPs at n/s; >60 AP at l | — |
| 13 | NAS over weight-shared super-nets (resolution, patch size, queries, decoder depth) | RF-DETR | a whole Pareto front from one training run | training compute |

## 3. What "most powerful" means depends on the constraint

* **Absolute accuracy at real-time speed on a GPU:** a query-based decoder with a foundation-model backbone
  (RF-DETR-2XL 60.1 AP, YOLO27l 60.4 AP preview).
* **Edge, CPU, or NPU deployment:** CNN + NMS-free o2o head, DFL-free boxes, exportable as plain convolutions plus a
  top-k (YOLO26n/27n class, ~2.4M parameters, ~5.4 GFLOPs).
* **Commercial use without AGPL obligations:** the Apache-2.0 DETR line (RF-DETR, D-FINE, DEIM), or an independent
  implementation of the published CNN recipes — the gap nextYOLO fills.

## 4. nextYOLO's position

nextYOLO targets the second and third bullets. It is an independently written, dependency-light implementation of
the strongest published **CNN + NMS-free** recipe, verified against YOLO26 at three levels:

* parameter counts match at all five scales;
* official weights load and reproduce Ultralytics' outputs to 2.4e-4;
* from-scratch training matches Ultralytics within run-to-run noise.

Every lever is a config switch, so it can be measured rather than assumed. Several promising-looking levers were
tested on top of YOLO26, and most did **not** help a dense one-to-one head:

* **AOA (Aligned One-to-One assignment)**, proposed here: choose the o2o positive from the o2m branch's ranking. It
  scored −2.5 AP at half schedule.
* **MAL (matchability-aware loss)** from DEIM: −4.1 AP on the NMS-free branch.
* **Dense distillation of the one-to-one branch** from a stronger teacher: −4.3 AP.

All three point to the same principle: *a one-to-one head must stay self-consistent*. What did help, by a large
margin, is **class-subset head transfer** from COCO-pretrained weights (+7.7 AP e2e over re-initialised heads).
YOLO27-style dual-scale heads and lossless SPD P2→P3 fusion are also implemented.

The empirical results, and what they do and do not show at CPU-feasible budgets, are in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Sources

* YOLO26: [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models (arXiv 2606.03748)](https://arxiv.org/abs/2606.03748);
  [YOLO26 key architectural enhancements (arXiv 2509.25164)](https://arxiv.org/abs/2509.25164);
  [YOLO26 NMS-free analysis (arXiv 2601.12882)](https://arxiv.org/html/2601.12882v1);
  [YOLO26 docs](https://docs.ultralytics.com/models/yolo26);
  [ProgLoss/STAL/MuSGD blog](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd).
  The implementation details (head, E2E loss schedule, STAL, MuSGD) were read from the `ultralytics` 8.4.166 source.
* YOLO27: [YOLO27 docs](https://docs.ultralytics.com/models/yolo27);
  [Ultralytics YOLO Evolution overview (arXiv 2510.09653)](https://arxiv.org/abs/2510.09653);
  [YOLO Vision 2026 highlights](https://www.ultralytics.com/blog/key-highlights-from-ultralytics-yolo-vision-2026).
* RF-DETR: [roboflow/rf-detr](https://github.com/roboflow/rf-detr);
  [RF-DETR: NAS for Real-Time Detection Transformers (arXiv 2511.09554)](https://arxiv.org/html/2511.09554v2);
  [RF-DETR docs](https://rfdetr.roboflow.com/latest/).
* DEIMv2: [Real-Time Object Detection Meets DINOv3 (arXiv 2509.20787)](https://arxiv.org/html/2509.20787v4).
  DEIM MAL loss read from [DEIM source](https://github.com/ShihuaHuang95/DEIM).
* RT-DETRv4: [arXiv 2510.25257](https://arxiv.org/html/2510.25257v1).
* YOLOv13: [arXiv 2506.17733](https://arxiv.org/html/2506.17733v1).
* VajraV1: [arXiv 2512.13834](https://arxiv.org/abs/2512.13834v1). "YOLOv14" (third-party):
  [arXiv 2608.04720](https://arxiv.org/pdf/2608.04720).
* D-FINE: [Peterande/D-FINE](https://github.com/Peterande/D-FINE); [arXiv 2410.13842](https://arxiv.org/abs/2410.13842).
* YOLO-ULM: [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36363);
  [open access](https://openaccess.thecvf.com/content/CVPR2026/html/Han_YOLO-ULM_Ultra-Lightweight_Models_for_Real-Time_Object_Detection_CVPR_2026_paper.html).
* Overviews: [Roboflow: best object detection models 2026](https://blog.roboflow.com/best-object-detection-models/);
  [JetBrains: best object detection models 2026](https://blog.jetbrains.com/pycharm/2026/07/best-object-detection-models-for-machine-learning-in-2026/).
