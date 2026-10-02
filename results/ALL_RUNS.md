# All runs (VOC07 test, AP in %, scored by nextYOLO's pycocotools-equivalent evaluator)

Legend: `p1_ny_y26recipe` = recipe v1 (its config predates the `cls_lr_mult` flag, i.e. no 3x cls-head lr); `p1b_ny_y26parity` and all later MuSGD runs = recipe v2 (`cls_lr_mult=3.0`). Fine-tuning runs (`b*`, `c*`) use 320 px / 3 epochs (unless noted) / AdamW from `yolo26n.pt`, lr0 4.17e-4 unless noted; from-scratch runs use 256 px / 12 epochs / MuSGD. "first eval" is epoch 6 for from-scratch runs and epoch 1 for fine-tuning runs. The AOA run (`p2_ny_aoa_halfschedule`) was stopped after its epoch-6 evaluation; see its history.json.

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
| b5_ft_lr1e4 | nextYOLO | optimizer=adamw, init=yolo26n.pt, lr0=1e-4 | 58.96 | 78.48 | 11.78 | 59.42 | 79.37 | 56.54 | 0.9 |
| b6_ft_lr3e5 | nextYOLO | optimizer=adamw, init=yolo26n.pt, lr0=3e-5 | 57.92 | 77.29 | 11.68 | 58.30 | 77.97 | 56.51 | 0.9 |
| b7_ft_lr1e4_6ep | nextYOLO | optimizer=adamw, init=yolo26n.pt, lr0=1e-4, 6 epochs | 60.03 | 79.55 | 12.38 | 60.70 | 80.48 | 56.54 | 1.8 |
| b8_ft_lr2e4_6ep | nextYOLO | optimizer=adamw, init=yolo26n.pt, lr0=2e-4, 6 epochs | 59.38 | 79.21 | 12.78 | 60.18 | 80.25 | 55.85 | 1.8 |
| c4_ft_lr1e4_kd | nextYOLO | optimizer=adamw, init=yolo26n.pt, distill.teacher=ft_s640_voc.pt, distill.branches=['o2m'], lr0=1e-4 | 59.19 | 79.17 | 12.93 | 59.94 | 80.12 | 57.34 | 1.6 |
| c5_ft_lr1e4_6ep_kd | nextYOLO | optimizer=adamw, init=yolo26n.pt, distill.teacher=ft_s640_voc.pt, distill.branches=['o2m'], lr0=1e-4, 6 epochs | 60.43 | 80.34 | 13.28 | 61.01 | 81.00 | 57.34 | 3.1 |
