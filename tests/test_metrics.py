"""The built-in COCO-style evaluator must agree with pycocotools."""

import contextlib
import io

import numpy as np
import pytest

from nextyolo.utils.metrics import COCOEvaluator

pycocotools = pytest.importorskip("pycocotools")
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402


def _random_problem(seed: int, n_img=40, nc=5):
    rng = np.random.default_rng(seed)
    images, gts, dts = [], [], []
    for i in range(n_img):
        W, H = 640, 480
        k = rng.integers(0, 8)
        xy = rng.uniform(0, [W - 20, H - 20], (k, 2))
        wh = rng.uniform(4, 200, (k, 2))  # spans small / medium / large
        gb = np.concatenate((xy, np.minimum(xy + wh, [W, H])), 1)
        gc = rng.integers(0, nc, k)
        # detections: jittered GT copies (TP-ish), duplicates, and random boxes (FP)
        jit = gb + rng.normal(0, 6, gb.shape)
        dup = gb + rng.normal(0, 15, gb.shape)
        m = rng.integers(0, 6)
        rxy = rng.uniform(0, [W - 20, H - 20], (m, 2))
        rb = np.concatenate((rxy, rxy + rng.uniform(4, 150, (m, 2))), 1)
        db = np.concatenate((jit, dup, rb))
        db[:, 2:] = np.maximum(db[:, 2:], db[:, :2] + 1)
        dc = np.concatenate((gc, np.where(rng.random(k) < 0.8, gc, rng.integers(0, nc, k)), rng.integers(0, nc, m)))
        ds = rng.random(len(db))
        images.append(dict(id=i, width=W, height=H))
        gts.append((gc, gb))
        dts.append((db, ds, dc))
    return images, gts, dts, nc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools(seed):
    images, gts, dts, nc = _random_problem(seed)
    # pycocotools
    anns, aid = [], 1
    for i, (gc, gb) in enumerate(gts):
        for c, b in zip(gc, gb):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append(dict(id=aid, image_id=i, category_id=int(c), bbox=[b[0], b[1], w, h], area=w * h, iscrowd=0))
            aid += 1
    coco = COCO()
    coco.dataset = dict(images=images, annotations=anns, categories=[dict(id=c) for c in range(nc)])
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        res = []
        for i, (db, ds, dc) in enumerate(dts):
            for b, s, c in zip(db, ds, dc):
                res.append(dict(image_id=i, category_id=int(c), bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]],
                                score=float(s)))
        cdt = coco.loadRes(res)
        E = COCOeval(coco, cdt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    ref = E.stats  # AP, AP50, AP75, APs, APm, APl, ...
    # ours
    ev = COCOEvaluator(nc)
    for i, ((gc, gb), (db, ds, dc)) in enumerate(zip(gts, dts)):
        ev.add(i, gc, gb, db, ds, dc)
    r = ev.evaluate()
    ours = [r["mAP"], r["mAP50"], r["mAP75"], r["APs"], r["APm"], r["APl"]]
    np.testing.assert_allclose(ours, ref[:6], atol=1e-6)


def test_perfect_detections():
    ev = COCOEvaluator(2)
    gb = np.array([[10, 10, 60, 60], [100, 100, 300, 250]], float)
    ev.add(0, [0, 1], gb, gb, [0.9, 0.8], [0, 1])
    r = ev.evaluate()
    assert r["mAP"] == pytest.approx(1.0) and r["mAP50"] == pytest.approx(1.0)
