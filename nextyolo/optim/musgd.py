"""MuSGD: SGD with an added Muon (orthogonalised-momentum) step for matrix/conv weights.

Muon (K. Jordan et al., 2024) replaces the raw momentum update of a weight matrix by its nearest semi-orthogonal
matrix, computed with a quintic Newton-Schulz iteration. All singular directions then move at the same rate, which
speeds up training of the rarely-updated directions. YOLO26 popularised a hybrid for detectors: every matrix-shaped
weight receives both `muon_scale * lr * Muon(update)` and `sgd_scale * lr * SGD(update)`; all other parameters
(biases, norms) receive plain SGD.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T of G = U S V^T via Newton-Schulz (singular values mapped to ~[0.7, 1.2])."""
    a, b, c = _NS_COEFFS
    X = G.float()
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class MuSGD(Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.937, weight_decay: float = 0.0,
                 nesterov: bool = True, use_muon: bool = False, muon_scale: float = 0.2, sgd_scale: float = 1.0,
                 ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon_scale = muon_scale
        self.sgd_scale = sgd_scale
        self.ns_steps = ns_steps

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, nesterov, wd = group["lr"], group["momentum"], group["nesterov"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                sgd_lr = lr
                if group["use_muon"] and p.ndim >= 2:
                    buf = state.get("muon_buf")
                    if buf is None:
                        buf = state["muon_buf"] = torch.zeros_like(p)
                    buf.mul_(mom).add_(g, alpha=1 - mom)
                    u = g.lerp(buf, mom) if nesterov else buf       # (1-m) g + m buf
                    mat = u.reshape(u.shape[0], -1)
                    o = orthogonalize(mat, self.ns_steps)
                    o *= max(1.0, mat.shape[0] / mat.shape[1]) ** 0.5
                    p.add_(o.view_as(p), alpha=-lr * self.muon_scale)
                    sgd_lr = lr * self.sgd_scale
                # SGD component (weight decay lives here only)
                d = g.add(p, alpha=wd) if wd != 0 else g
                buf = state.get("sgd_buf")
                if buf is None:
                    buf = state["sgd_buf"] = d.clone()
                else:
                    buf.mul_(mom).add_(d)
                upd = d.add(buf, alpha=mom) if nesterov else buf
                p.add_(upd, alpha=-sgd_lr)
        return loss
