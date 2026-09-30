"""Unit tests for box ops, assigner, head decoding, model fusion, loss and optimizer."""

import torch
import torchvision

from nextyolo.loss.assigner import TaskAlignedAssigner
from nextyolo.loss.loss import DetectionLoss, LossConfig, prepare_targets
from nextyolo.nn.model import ModelConfig, NextYOLO
from nextyolo.optim.musgd import MuSGD, orthogonalize
from nextyolo.utils.boxes import bbox2dist, bbox_iou, box_iou, dist2bbox, make_anchors, xywh2xyxy, xyxy2xywh


def _rand_boxes(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 200
    wh = torch.rand(n, 2, generator=g) * 100 + 1
    return torch.cat((xy, xy + wh), 1)


def test_box_conversions_roundtrip():
    b = _rand_boxes(50)
    torch.testing.assert_close(xywh2xyxy(xyxy2xywh(b)), b)
    pts = (b[:, :2] + b[:, 2:]) / 2
    torch.testing.assert_close(dist2bbox(bbox2dist(pts, b), pts), b)


def test_iou_variants_match_torchvision():
    a, b = _rand_boxes(64, 1), _rand_boxes(64, 2)
    torch.testing.assert_close(box_iou(a, b), torchvision.ops.box_iou(a, b), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(bbox_iou(a, b, "iou"), torchvision.ops.box_iou(a, b).diagonal(), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(bbox_iou(a, b, "giou"), torchvision.ops.generalized_box_iou(a, b).diagonal(),
                               atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(bbox_iou(a, b, "diou"), torchvision.ops.distance_box_iou(a, b).diagonal(),
                               atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(bbox_iou(a, b, "ciou"), torchvision.ops.complete_box_iou(a, b).diagonal(),
                               atol=1e-4, rtol=1e-3)


def test_make_anchors():
    pts, st = make_anchors([(2, 3), (1, 1)], [8, 16])
    assert pts.shape == (7, 2) and st.shape == (7, 1)
    assert pts[0].tolist() == [0.5, 0.5] and pts[2].tolist() == [2.5, 0.5] and pts[3].tolist() == [0.5, 1.5]
    assert st[-1].item() == 16


def _assign_setup(gt_xyxy, strides=(8, 16, 32), imgsz=64):
    shapes = [(imgsz // s, imgsz // s) for s in strides]
    pts, st = make_anchors(shapes, list(strides))
    anchors = pts * st
    A = len(anchors)
    scores = torch.full((1, A, 3), 0.5)
    pd = torch.cat((anchors - 8, anchors + 8), -1)[None]  # 16 px boxes everywhere
    gt = torch.tensor(gt_xyxy, dtype=torch.float)[None]
    labels = torch.zeros(1, gt.shape[1], 1)
    mask = torch.ones(1, gt.shape[1], 1)
    return scores, pd, anchors, labels, gt, mask


def test_assigner_topk_and_one_to_one():
    s, pd, anc, lab, gt, mask = _assign_setup([[8, 8, 40, 40], [30, 30, 60, 62]])
    tb, ts, fg, gi = TaskAlignedAssigner(3, topk=10)(s, pd, anc, lab, gt, mask)
    assert 2 < fg.sum() <= 20
    assert ts[fg].sum(-1).gt(0).all() and ts[~fg].sum() == 0
    tb, ts, fg, gi = TaskAlignedAssigner(3, topk=7, topk2=1)(s, pd, anc, lab, gt, mask)
    assert fg.sum() == 2 and set(gi[fg].tolist()) == {0, 1}  # exactly one anchor per object


def test_stal_gives_tiny_objects_candidates():
    tiny = [[33.0, 33.0, 35.0, 35.0]]  # 2x2 px box, between anchor centres of every level
    s, pd, anc, lab, gt, mask = _assign_setup(tiny)
    pd = gt.expand(1, pd.shape[1], 4).clone()  # a perfect regressor: only candidate selection can fail
    _, _, fg, _ = TaskAlignedAssigner(3, topk=10, min_side=0)(s, pd, anc, lab, gt, mask)
    assert fg.sum() == 0
    _, _, fg, _ = TaskAlignedAssigner(3, topk=10, min_side=16)(s, pd, anc, lab, gt, mask)
    assert fg.sum() > 0


def test_assigner_ignores_padding():
    s, pd, anc, lab, gt, mask = _assign_setup([[8, 8, 40, 40], [0, 0, 0, 0]])
    mask[0, 1] = 0
    _, _, fg, gi = TaskAlignedAssigner(3, topk=10)(s, pd, anc, lab, gt, mask)
    assert (gi[fg] == 0).all()


def test_topk_decode_matches_bruteforce():
    m = NextYOLO(ModelConfig(nc=5, max_det=20)).eval()
    xyxy = torch.rand(2, 100, 4)
    scores = torch.rand(2, 100, 5)
    out = m.head.topk_decode(xyxy, scores)
    for b in range(2):
        flat = scores[b].flatten()
        ref = flat.topk(20).values
        torch.testing.assert_close(out[b, :, 4], ref)
        for k in range(20):
            a, c = int(out[b, k, 5]) , None
            idx = (scores[b, :, a] == out[b, k, 4]).nonzero()
            assert len(idx) and torch.allclose(xyxy[b, idx[0, 0]], out[b, k, :4])


def test_model_train_eval_shapes_and_fuse():
    torch.manual_seed(0)
    m = NextYOLO(ModelConfig(nc=7))
    x = torch.rand(2, 3, 128, 96)
    m.train()
    out = m(x)
    A = (16 * 12) + (8 * 6) + (4 * 3)
    assert out["o2m"][0].shape == (2, 4, A) and out["o2o"][1].shape == (2, 7, A)
    m.eval()
    y = m(x)
    assert y.shape == (2, min(300, A), 6)
    y2 = m.fuse()(x)
    torch.testing.assert_close(y, y2, atol=1e-4, rtol=1e-4)


def test_dual_scale_and_p2_variants_run():
    for kw in (dict(levels=(3, 5)), dict(p2_fusion=True), dict(attn_area=4)):
        m = NextYOLO(ModelConfig(nc=3, **kw)).eval()
        assert m(torch.rand(1, 3, 128, 128)).shape[-1] == 6


def test_loss_backward_and_empty_targets():
    torch.manual_seed(0)
    m = NextYOLO(ModelConfig(nc=4)).train()
    crit = DetectionLoss(4, m.stride.tolist(), LossConfig(), end2end=True, epochs=10)
    x = torch.rand(2, 3, 128, 128)
    targets = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.4], [0, 2, 0.2, 0.3, 0.1, 0.1], [1, 3, 0.6, 0.6, 0.5, 0.5]])
    for kind in ("bce", "vfl", "mal", "qfl"):
        crit.o2m.cls_kind = crit.o2o.cls_kind = kind
        loss, items = crit(m(x), targets, (128, 128))
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
    m.zero_grad()
    loss, _ = crit(m(x), torch.zeros(0, 6), (128, 128))
    loss.backward()
    assert torch.isfinite(loss)
    # o2o branch is detached: its loss alone must not reach the backbone
    m.zero_grad()
    preds = m(x)
    lo, _ = crit.o2o(*preds["o2o"], preds["shapes"], *prepare_targets(targets, 2, (128, 128)), (128, 128))
    lo.backward()
    assert m.layers[0].conv.weight.grad is None or m.layers[0].conv.weight.grad.abs().sum() == 0
    assert m.head.o2o_cls[0][-1].weight.grad.abs().sum() > 0


def test_prog_loss_schedule():
    crit = DetectionLoss(3, [8, 16, 32], LossConfig(), end2end=True, epochs=11)
    crit.set_epoch(0)
    assert abs(crit.w_o2m - 0.8) < 1e-9 and abs(crit.w_o2o - 0.2) < 1e-9
    crit.set_epoch(10)
    assert abs(crit.w_o2m - 0.1) < 1e-9 and abs(crit.w_o2o - 0.9) < 1e-9


def test_orthogonalize_and_musgd():
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    O = orthogonalize(G)
    sv = torch.linalg.svdvals(O)
    assert sv.min() > 0.5 and sv.max() < 1.3
    # MuSGD minimises a least-squares problem
    W = torch.nn.Parameter(torch.zeros(16, 8))
    b = torch.nn.Parameter(torch.zeros(16))
    target = torch.randn(16, 8)
    opt = MuSGD([dict(params=[W], use_muon=True), dict(params=[b], use_muon=False)], lr=0.05, momentum=0.9)
    first = None
    for _ in range(100):
        opt.zero_grad()
        loss = ((W - target) ** 2).sum() + (b - 1).pow(2).sum()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.05 * first


def test_aligned_one_to_one_assignment():
    """AOA: the o2o positive for each object is the top-1 anchor of the o2m ranking."""
    torch.manual_seed(0)
    m = NextYOLO(ModelConfig(nc=4)).train()
    targets = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.4], [0, 2, 0.2, 0.3, 0.2, 0.2], [1, 3, 0.6, 0.6, 0.5, 0.5]])
    preds = m(torch.rand(2, 3, 128, 128))
    crit = DetectionLoss(4, m.stride.tolist(), LossConfig(o2o_assign="o2m"), end2end=True, epochs=10)
    loss, _ = crit(preds, targets, (128, 128))
    assert torch.isfinite(loss)
    lab, box, mask = prepare_targets(targets, 2, (128, 128))
    pts, st = make_anchors(preds["shapes"], m.stride.tolist())

    def assign(branch, topk, topk2):
        b, l = preds[branch]
        pb = dist2bbox(b.detach().permute(0, 2, 1), pts) * st
        return TaskAlignedAssigner(4, topk, topk2, min_side=m.stride[1].item())(
            l.detach().permute(0, 2, 1).sigmoid(), pb, pts * st, lab, box, mask)

    _, _, fg_aoa, gi_aoa = assign("o2m", 7, 1)
    assert fg_aoa.sum() == 3  # one anchor per object
    # the chosen anchor is the o2m positive with the highest alignment metric
    _, ts_m, fg_m, gi_m = assign("o2m", 10, None)
    for b in range(2):
        for g in gi_aoa[b][fg_aoa[b]].unique():
            sel = fg_aoa[b] & (gi_aoa[b] == g)
            cand = fg_m[b] & (gi_m[b] == g)
            assert (sel & cand).sum() == 1
            assert ts_m[b][sel].sum() >= ts_m[b][cand].sum(-1).max() - 1e-6


def test_onnx_export_nms_free(tmp_path):
    import pytest

    pytest.importorskip("onnxruntime")
    import onnx

    from tools.export import calibrate_bn, check_onnx, export_onnx

    torch.manual_seed(0)
    m = calibrate_bn(NextYOLO(ModelConfig(nc=5, max_det=50)), 128).fuse()
    path = export_onnx(m, str(tmp_path / "m.onnx"), 128)
    ops = {n.op_type for n in onnx.load(path).graph.node}
    assert "TopK" in ops and "NonMaxSuppression" not in ops
    assert check_onnx(m, path, 128, top=20) < 1e-2
