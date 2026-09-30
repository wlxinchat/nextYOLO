"""Optional: official YOLO26 weights load into nextYOLO and reproduce Ultralytics' NMS-free outputs.

Runs only when `ultralytics` is installed and YOLO26N_WEIGHTS points to yolo26n.pt.
"""

import os

import pytest

WEIGHTS = os.environ.get("YOLO26N_WEIGHTS", "/home/user/weights/yolo26n.pt")


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="yolo26n.pt not available")
def test_yolo26n_weights_reproduce_ultralytics():
    pytest.importorskip("ultralytics")
    from tools.convert_ultralytics import check_against_ultralytics, load_ultralytics

    model, names = load_ultralytics(WEIGHTS)
    assert len(names) == 80
    assert check_against_ultralytics(WEIGHTS, model, imgsz=320) < 1e-2
