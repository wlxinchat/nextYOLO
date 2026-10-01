# All runs (VOC07 test, AP in %, scored by nextYOLO's pycocotools-equivalent evaluator)

Legend: `p1_ny_y26recipe` = recipe v1 (its config predates the `cls_lr_mult` flag, i.e. no 3x cls-head lr); `p1b_ny_y26parity` and all later MuSGD runs = recipe v2 (`cls_lr_mult=3.0`). Fine-tuning runs (`b*`, `c*`) use 320 px / 3 epochs / AdamW from `yolo26n.pt`; from-scratch runs use 256 px / 12 epochs / MuSGD. "first eval" is epoch 6 for from-scratch runs and epoch 1 for fine-tuning runs. The AOA run (`p2_ny_aoa_halfschedule`) was stopped after its epoch-6 evaluation; see its history.json.

| run | impl | change vs. YOLO26 recipe | AP e2e (o2o, NMS-free) | AP50 e2e | AP_S e2e | AP o2m+NMS | AP50 o2m+NMS | AP e2e @ first eval | hours |
|---|---|---|---|---|---|---|---|---|---|
| p1_ul_yolo26n | Ultralytics | reference implementation | 22.02 | 36.97 | 3.11 | 24.95 | 42.19 | — | — |
| p1_ny_y26recipe | nextYOLO | — | 21.15 | 35.25 | 3.30 | 24.71 | 41.46 | 11.40 | 2.6 |
| p1b_ny_y26parity | nextYOLO | — | 21.99 | 36.76 | 4.62 | 24.99 | 41.79 | 12.14 | 2.6 |
| p3_ny_y26recipe_seed1 | nextYOLO | seed=1 | 22.16 | 37.33 | 2.61 | 25.05 | 42.33 | 12.39 | 2.5 |
| p2_ny_mal | nextYOLO | loss.o2o_cls_loss=mal | 18.02 | 28.42 | 2.54 | 24.46 | 40.96 | 9.84 | 2.3 |
| p3_ny_dualscale | nextYOLO | model.levels=[3, 5] | 20.98 | 35.52 | 2.78 | 23.59 | 40.15 | 10.92 | 2.0 |
| b4_ul_ft | Ultralytics | reference implementation | 47.67 | 65.79 | 8.46 | 54.55 | 75.28 | — | — |
| b1_ft_y26recipe | nextYOLO | optimizer=adamw, init=yolo26n.pt | 56.96 | 76.95 | 13.53 | 58.08 | 78.41 | 50.47 | 1.0 |
| b2_ft_randhead | nextYOLO | optimizer=adamw, init=yolo26n.pt (random cls head) | 49.22 | 67.83 | 9.95 | 55.40 | 76.12 | 35.31 | 0.9 |
| c1_ft_kd | nextYOLO | optimizer=adamw, init=yolo26n.pt, distill.teacher=yolo26s.pt | 52.44 | 71.95 | 13.14 | 57.06 | 77.66 | 44.20 | 1.7 |
| c2_ft_kd_tassign | nextYOLO | loss.o2o_assign=teacher, optimizer=adamw, init=yolo26n.pt, distill.teacher=yolo26s.pt | 50.49 | 68.32 | 14.05 | 57.03 | 77.69 | 42.19 | 1.8 |
| c3_ft_kd_o2m_only | nextYOLO | optimizer=adamw, init=yolo26n.pt, distill.teacher=yolo26s.pt, distill.branches=['o2m'] | 56.78 | 77.32 | 15.37 | 57.24 | 77.90 | 51.36 | 1.5 |
