"""Training loop: warmup, nominal-batch gradient accumulation, EMA, periodic COCO-style evaluation."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import YOLODataset, collate
from ..loss.loss import DetectionLoss, LossConfig
from ..nn.model import ModelConfig, NextYOLO
from ..optim.musgd import MuSGD
from .evaluator import evaluate


@dataclass
class TrainConfig:
    train: list = field(default_factory=list)   # image dirs / list files
    val: list = field(default_factory=list)
    names: list = field(default_factory=list)
    out: str = "runs/exp"
    epochs: int = 100
    batch: int = 16
    nbs: int = 64                  # nominal batch size (gradient accumulation target)
    imgsz: int = 640
    optimizer: str = "musgd"       # musgd | sgd | adamw
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    close_mosaic: int = 10
    ema_decay: float = 0.9999
    ema_tau: float = 2000
    workers: int = 2
    threads: int = 0               # torch intra-op threads (0 = leave default)
    eval_interval: int = 5
    eval_max_images: int | None = None
    seed: int = 0
    hyp: dict = field(default_factory=dict)      # augmentation overrides
    model: dict = field(default_factory=dict)    # ModelConfig overrides
    loss: dict = field(default_factory=dict)     # LossConfig overrides
    max_train_images: int | None = None
    time_limit_h: float | None = None
    compile: bool = False          # torch.compile the training forward (+~25% CPU throughput)
    channels_last: bool = True


class ModelEMA:
    """Exponential moving average of weights with a warm-up ramp: d = decay * (1 - exp(-updates / tau))."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def build_optimizer(model: nn.Module, name: str, lr: float, momentum: float, decay: float):
    """Three groups (conv/linear weights with decay, norm weights, biases); MuSGD adds Muon to matrix weights."""
    g_w, g_bn, g_b = [], [], []
    for mod in model.modules():
        for pn, p in mod.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pn == "bias":
                g_b.append(p)
            elif isinstance(mod, nn.modules.batchnorm._NormBase) or p.ndim < 2:
                g_bn.append(p)
            else:
                g_w.append(p)
    name = name.lower()
    if name == "musgd":
        groups = [
            dict(params=g_w, weight_decay=decay, use_muon=True, kind="weight"),
            dict(params=g_bn, weight_decay=0.0, use_muon=False, kind="bn"),
            dict(params=g_b, weight_decay=0.0, use_muon=False, kind="bias"),
        ]
        return MuSGD(groups, lr=lr, momentum=momentum, nesterov=True)
    if name == "sgd":
        return torch.optim.SGD([dict(params=g_w, weight_decay=decay, kind="weight"),
                                dict(params=g_bn, weight_decay=0.0, kind="bn"),
                                dict(params=g_b, weight_decay=0.0, kind="bias")],
                               lr=lr, momentum=momentum, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW([dict(params=g_w, weight_decay=decay, kind="weight"),
                                  dict(params=g_bn, weight_decay=0.0, kind="bn"),
                                  dict(params=g_b, weight_decay=0.0, kind="bias")],
                                 lr=lr, betas=(momentum, 0.999))
    raise ValueError(name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        seed_all(cfg.seed)
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        nc = len(cfg.names)
        self.model = NextYOLO(ModelConfig(nc=nc, **cfg.model))
        self.mem_format = torch.channels_last if cfg.channels_last else torch.contiguous_format
        self.model = self.model.to(memory_format=self.mem_format)
        self.fwd = torch.compile(self.model) if cfg.compile else self.model
        self.loss_cfg = LossConfig(**cfg.loss)
        self.criterion = DetectionLoss(nc, self.model.stride.tolist(), self.loss_cfg, self.model.cfg.end2end,
                                       cfg.epochs)
        t0 = time.time()
        self.train_set = YOLODataset(cfg.train, cfg.imgsz, train=True, hyp=cfg.hyp,
                                     max_images=cfg.max_train_images)
        self.val_set = YOLODataset(cfg.val, cfg.imgsz, train=False) if cfg.val else None
        self.log(f"datasets cached in {time.time() - t0:.0f}s: train {len(self.train_set)} "
                 f"val {len(self.val_set) if self.val_set else 0}")
        self.loader = self._loader()
        self.accumulate = max(round(cfg.nbs / cfg.batch), 1)
        wd = cfg.weight_decay * cfg.batch * self.accumulate / cfg.nbs
        self.opt = build_optimizer(self.model, cfg.optimizer, cfg.lr0, cfg.momentum, wd)
        for g in self.opt.param_groups:
            g["initial_lr"] = cfg.lr0
        self.ema = ModelEMA(self.model, cfg.ema_decay, cfg.ema_tau)
        self.history: list[dict] = []
        info = self.model.info(cfg.imgsz)
        self.log(f"model: {info['params'] / 1e6:.3f}M params, {info['gflops']:.2f} GFLOPs@{cfg.imgsz} | "
                 f"cfg {json.dumps(asdict(cfg), default=str)}")

    def _loader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch, shuffle=True, num_workers=self.cfg.workers,
                          collate_fn=collate, drop_last=True, persistent_workers=self.cfg.workers > 0,
                          worker_init_fn=_worker_init)

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.out / "log.txt", "a") as f:
            f.write(line + "\n")

    def lr_factor(self, epoch: float) -> float:
        return (1 - epoch / self.cfg.epochs) * (1.0 - self.cfg.lrf) + self.cfg.lrf  # linear decay

    def train(self) -> dict:
        cfg = self.cfg
        nb = len(self.loader)
        nw = round(min(cfg.warmup_epochs, max(cfg.epochs - 1, 0)) * nb) if cfg.warmup_epochs > 0 else 0
        best, last_step, t_start = -1.0, -1, time.time()
        best_metrics: dict = {}
        for epoch in range(cfg.epochs):
            self.criterion.set_epoch(epoch)
            if epoch == cfg.epochs - cfg.close_mosaic:
                self.train_set.hyp["mosaic"] = 0.0
                self.loader = self._loader()  # restart workers so they see the new hyp
                self.log("closing mosaic")
            self.model.train()
            t_ep = time.time()
            run = torch.zeros(6)
            self.opt.zero_grad(set_to_none=True)
            for i, (imgs, targets, _) in enumerate(self.loader):
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, cfg.nbs / cfg.batch]).round()))
                    for g in self.opt.param_groups:
                        start = cfg.warmup_bias_lr if g.get("kind") == "bias" else 0.0
                        g["lr"] = float(np.interp(ni, xi, [start, g["initial_lr"] * self.lr_factor(epoch)]))
                        if "momentum" in g:
                            g["momentum"] = float(np.interp(ni, xi, [cfg.warmup_momentum, cfg.momentum]))
                x = (imgs.float() / 255).contiguous(memory_format=self.mem_format)
                preds = self.fwd(x)
                loss, items = self.criterion(preds, targets, (cfg.imgsz, cfg.imgsz))
                loss.backward()
                if ni - last_step >= self.accumulate:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update(self.model)
                    last_step = ni
                vec = torch.cat([items["o2m"], items.get("o2o", torch.zeros(3))])
                run = run * (i / (i + 1)) + vec / (i + 1)
                if i % 50 == 0:
                    ips = (i + 1) * cfg.batch / (time.time() - t_ep)
                    self.log(f"ep {epoch + 1}/{cfg.epochs} it {i}/{nb} o2m[box {run[0]:.3f} cls {run[1]:.3f} "
                             f"l1 {run[2]:.3f}] o2o[box {run[3]:.3f} cls {run[4]:.3f} l1 {run[5]:.3f}] "
                             f"{ips:.1f} img/s")
            # epoch end: scheduler step (applies to the next epoch)
            for g in self.opt.param_groups:
                g["lr"] = g["initial_lr"] * self.lr_factor(epoch + 1)
            rec = dict(epoch=epoch + 1, time_s=time.time() - t_ep, loss=run.tolist(),
                       w_o2m=self.criterion.w_o2m, lr=self.opt.param_groups[0]["lr"])
            final = epoch + 1 == cfg.epochs
            out_of_time = cfg.time_limit_h and (time.time() - t_start) / 3600 > cfg.time_limit_h
            if self.val_set and ((epoch + 1) % cfg.eval_interval == 0 or final or out_of_time):
                m = evaluate(self.ema.ema, self.val_set, workers=cfg.workers, max_images=cfg.eval_max_images)
                rec.update({k: v for k, v in m.items() if k != "per_class_ap"})
                self.log(f"eval ep {epoch + 1}: mAP {m['mAP']:.4f} mAP50 {m['mAP50']:.4f} mAP75 {m['mAP75']:.4f} "
                         f"APs {m['APs']:.4f} APm {m['APm']:.4f} APl {m['APl']:.4f} (eval {m['eval_time_s']:.0f}s)")
                if m["mAP"] > best:
                    best, best_metrics = m["mAP"], m
                    self.save("best.pt", epoch)
            self.history.append(rec)
            self.log(f"epoch {epoch + 1} done in {rec['time_s']:.0f}s")
            self.save("last.pt", epoch)
            (self.out / "history.json").write_text(json.dumps(self.history, indent=1))
            if out_of_time:
                self.log("time limit reached")
                break
        summary = dict(best_mAP=best, best=best_metrics, hours=(time.time() - t_start) / 3600,
                       config=asdict(cfg))
        if self.val_set and self.model.cfg.end2end:
            # Also score the dense o2m branch with NMS: the gap to the NMS-free o2o branch is the price of e2e.
            head = self.ema.ema.head
            head.end2end = False
            m = evaluate(self.ema.ema, self.val_set, workers=cfg.workers, mode="nms",
                         max_images=cfg.eval_max_images)
            head.end2end = True
            summary["final_o2m_nms"] = {k: v for k, v in m.items() if k != "per_class_ap"}
            self.log(f"final o2m+NMS: mAP {m['mAP']:.4f} mAP50 {m['mAP50']:.4f}")
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary

    def save(self, name: str, epoch: int) -> None:
        torch.save({"epoch": epoch, "model_cfg": asdict(self.model.cfg), "names": self.cfg.names,
                    "ema": self.ema.ema.state_dict()}, self.out / name)


def _worker_init(worker_id: int) -> None:
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def load_model(path: str) -> NextYOLO:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: (tuple(v) if k == "levels" else v) for k, v in ck["model_cfg"].items()})
    model = NextYOLO(cfg)
    model.load_state_dict(ck["ema"])
    model.names = ck.get("names")
    return model.eval()
