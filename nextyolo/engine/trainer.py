"""Training loop: warmup, nominal-batch gradient accumulation, EMA, periodic COCO-style evaluation."""

from __future__ import annotations

import copy
import json
import os
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from ..data.dataset import YOLODataset, collate
from ..loss.distill import DistillConfig, distill_loss
from ..loss.loss import DetectionLoss, LossConfig
from ..nn.model import ModelConfig, NextYOLO
from ..optim.musgd import MuSGD
from .evaluator import evaluate


STOPPED_EXIT_CODE = 3  # process exit code for a graceful, resumable stop (see NEXTYOLO_STOP_FILE)


def _amp_dtype(mode: str, device: torch.device):
    """Autocast dtype for training: None on CPU / when off; bf16 where supported; fp16 (+GradScaler) otherwise."""
    if mode == "off" or device.type != "cuda":
        return None
    # Native bf16 needs compute capability >= 8.0 (Ampere+). torch.cuda.is_bf16_supported() also reports True for
    # emulated bf16 (e.g. T4 / Turing), which is far slower than fp16 there.
    if mode == "fp16" or (mode == "auto" and torch.cuda.get_device_capability(device)[0] < 8):
        return torch.float16
    return torch.bfloat16


def should_pause(epoch_seconds: float) -> bool:
    """True if NEXTYOLO_STOP_FILE exists, or if another epoch of this length would overrun NEXTYOLO_DEADLINE
    (unix time). Lets a job scheduler split long runs into bounded segments without losing partial epochs."""
    stop = os.environ.get("NEXTYOLO_STOP_FILE")
    if stop and Path(stop).exists():
        return True
    deadline = os.environ.get("NEXTYOLO_DEADLINE")
    return bool(deadline) and time.time() + 1.1 * epoch_seconds + 60 > float(deadline)


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
    cls_lr_mult: float = 3.0       # lr multiplier for classification heads under MuSGD (as in YOLO26)
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
    log_interval: int = 50
    eval_max_images: int | None = None
    seed: int = 0
    hyp: dict = field(default_factory=dict)      # augmentation overrides
    model: dict = field(default_factory=dict)    # ModelConfig overrides
    loss: dict = field(default_factory=dict)     # LossConfig overrides
    max_train_images: int | None = None
    time_limit_h: float | None = None
    distill: dict = field(default_factory=dict)  # DistillConfig overrides; distill.teacher enables KD
    init: str | None = None        # start from a checkpoint (nextYOLO .pt or Ultralytics YOLO26 .pt)
    init_head: str = "subset"      # "subset": slice cls heads to our classes by name; "random": re-init cls heads
    compile: bool = False          # torch.compile the training forward (+~25% CPU throughput)
    resume: bool = True            # continue from <out>/last.pt if it holds a full training state
    save_interval_min: float = 0.0  # also checkpoint mid-epoch every N minutes (preemptible VMs); 0 = epoch ends only
    channels_last: bool = True
    device: str = "auto"           # "auto" = cuda if available, else cpu
    amp: str = "auto"              # "auto": bf16 on GPUs that support it, else fp16 + grad scaling; "off" on CPU


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


def build_optimizer(model: nn.Module, name: str, lr: float, momentum: float, decay: float,
                    boost: set[int] | None = None, boost_mult: float = 1.0):
    """Three groups (conv/linear weights with decay, norm weights, biases); MuSGD adds Muon to matrix weights.

    Parameters whose id() is in `boost` get their own groups with lr * boost_mult (YOLO26 trains the classification
    heads at 3x lr under MuSGD).
    """
    opt = _build_optimizer(model, name, lr, momentum, decay)
    if not boost or boost_mult == 1.0:
        return opt
    groups = []
    for g in opt.param_groups:
        hot = [p for p in g["params"] if id(p) in boost]
        cold = [p for p in g["params"] if id(p) not in boost]
        opts = {k: v for k, v in g.items() if k != "params"}
        groups += [dict(opts, params=cold)] + ([dict(opts, params=hot, lr=g["lr"] * boost_mult)] if hot else [])
    kw = dict(muon_scale=opt.muon_scale, sgd_scale=opt.sgd_scale) if isinstance(opt, MuSGD) else {}
    return type(opt)(groups, **kw) if isinstance(opt, MuSGD) else type(opt)(groups)


def _build_optimizer(model: nn.Module, name: str, lr: float, momentum: float, decay: float):
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


class EpochSampler(Sampler):
    """Shuffled order that is a pure function of (seed, epoch), so a run resumed mid-epoch replays exactly the
    batches it has not trained on yet (`start` = number of samples to skip)."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self) -> int:
        return self.n - self.start


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
        self.model = self._init_model(nc) if cfg.init else NextYOLO(ModelConfig(nc=nc, **cfg.model))
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                                   else "cpu" if cfg.device == "auto" else cfg.device)
        self.amp_dtype = _amp_dtype(cfg.amp, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.mem_format = torch.channels_last if cfg.channels_last else torch.contiguous_format
        self.model = self.model.to(self.device, memory_format=self.mem_format)
        self.fwd = torch.compile(self.model) if cfg.compile else self.model
        self.dcfg = DistillConfig(**cfg.distill)
        self.teacher = None
        if self.dcfg.teacher:
            t, how = load_for_classes(self.dcfg.teacher, cfg.names, "subset")
            t = t.eval().to(self.device, memory_format=self.mem_format)
            for p in t.parameters():
                p.requires_grad_(False)
            t.head.return_raw = True
            self.teacher = t
            self.log(f"distilling from {self.dcfg.teacher} (head: {how}, {sum(p.numel() for p in t.parameters()) / 1e6:.1f}M params)")
        self.loss_cfg = LossConfig(**cfg.loss)
        self.criterion = DetectionLoss(nc, self.model.stride.tolist(), self.loss_cfg, self.model.cfg.end2end,
                                       cfg.epochs)
        t0 = time.time()
        self.train_set = YOLODataset(cfg.train, cfg.imgsz, train=True, hyp=cfg.hyp,
                                     max_images=cfg.max_train_images)
        self.val_set = (YOLODataset(cfg.val, cfg.imgsz, train=False, max_images=cfg.eval_max_images)
                        if cfg.val else None)
        self.log(f"datasets cached in {time.time() - t0:.0f}s: train {len(self.train_set)} "
                 f"val {len(self.val_set) if self.val_set else 0}")
        self.sampler = EpochSampler(len(self.train_set), cfg.seed)
        self.loader = self._loader()
        self.accumulate = max(round(cfg.nbs / cfg.batch), 1)
        wd = cfg.weight_decay * cfg.batch * self.accumulate / cfg.nbs
        head = self.model.head
        boost = {id(p) for m in (head.cls, getattr(head, "o2o_cls", None)) if m is not None for p in m.parameters()}
        mult = cfg.cls_lr_mult if cfg.optimizer.lower() == "musgd" else 1.0
        self.opt = build_optimizer(self.model, cfg.optimizer, cfg.lr0, cfg.momentum, wd, boost, mult)
        for g in self.opt.param_groups:
            g["initial_lr"] = g["lr"]
        self.ema = ModelEMA(self.model, cfg.ema_decay, cfg.ema_tau)
        self.history: list[dict] = []
        dev = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        self.log(f"device: {self.device} ({dev}), amp: {self.amp_dtype or 'off'}, compile: {cfg.compile}, "
                 f"workers: {cfg.workers}")
        info = self.model.info(cfg.imgsz)
        self.log(f"model: {info['params'] / 1e6:.3f}M params, {info['gflops']:.2f} GFLOPs@{cfg.imgsz} | "
                 f"cfg {json.dumps(asdict(cfg), default=str)}")

    def _init_model(self, nc: int) -> NextYOLO:
        model, how = load_for_classes(self.cfg.init, self.cfg.names, self.cfg.init_head)
        self.log(f"initialised from {self.cfg.init} (classification head: {how})")
        return model.train()

    def _loader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch, sampler=self.sampler,
                          num_workers=self.cfg.workers, collate_fn=collate, drop_last=True, persistent_workers=self.cfg.workers > 0,
                          worker_init_fn=_worker_init, pin_memory=self.device.type == "cuda")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.out / "log.txt", "a") as f:
            f.write(line + "\n")

    def lr_factor(self, epoch: float) -> float:
        return (1 - epoch / self.cfg.epochs) * (1.0 - self.cfg.lrf) + self.cfg.lrf  # linear decay

    def _try_resume(self) -> tuple[int, float, dict, int, float]:
        """Restore full state from <out>/last.pt. Returns (start_epoch, best, best_metrics, last_step, hours)."""
        path = self.out / "last.pt"
        if not (self.cfg.resume and path.exists()):
            return 0, -1.0, {}, -1, 0.0
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "optimizer" not in ck:  # EMA-only checkpoint from an older version: cannot resume exactly
            self.log(f"{path} has no optimizer state; starting from scratch")
            return 0, -1.0, {}, -1, 0.0
        self.model.load_state_dict(ck["model"])
        self.ema.ema.load_state_dict(ck["ema"])
        self.ema.updates = ck["ema_updates"]
        self.opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        self.history = ck["history"]
        if ck.get("iter", 0):  # mid-epoch checkpoint: continue inside that epoch
            self._mid = ck
            self.log(f"resumed from {path} in epoch {ck['epoch'] + 1} at iteration {ck['iter']}")
            return ck["epoch"], ck["best"], ck["best_metrics"], ck["last_step"], ck["hours"]
        self.log(f"resumed from {path} after epoch {ck['epoch'] + 1}")
        return ck["epoch"] + 1, ck["best"], ck["best_metrics"], ck["last_step"], ck["hours"]

    def _full_state(self, best: float, best_metrics: dict, last_step: int, t_start: float, **extra) -> dict:
        return dict(model=self.model.state_dict(), optimizer=self.opt.state_dict(), scaler=self.scaler.state_dict(),
                    ema_updates=self.ema.updates, history=self.history, best=best, best_metrics=best_metrics,
                    last_step=last_step, hours=(time.time() - t_start) / 3600, **extra)

    def train(self) -> dict:
        cfg = self.cfg
        if cfg.resume and (self.out / "summary.json").exists():
            self.log("run already complete; nothing to do")
            return json.loads((self.out / "summary.json").read_text())
        nb = len(self.train_set) // cfg.batch  # full epoch (drop_last), also when resuming mid-epoch
        self._mid = None
        nw = round(min(cfg.warmup_epochs, max(cfg.epochs - 1, 0)) * nb) if cfg.warmup_epochs > 0 else 0
        start_epoch, best, best_metrics, last_step, prev_hours = self._try_resume()
        t_start = time.time() - prev_hours * 3600
        for epoch in range(start_epoch, cfg.epochs):
            self.criterion.set_epoch(epoch)
            if epoch >= cfg.epochs - cfg.close_mosaic and self.train_set.hyp["mosaic"] > 0:
                self.train_set.hyp["mosaic"] = 0.0
                self.loader = self._loader()  # restart workers so they see the new hyp
                self.log("closing mosaic")
            self.model.train()
            mid, self._mid = self._mid, None
            i0 = mid["iter"] if mid else 0
            t_ep = time.time() - (mid["epoch_time"] if mid else 0.0)
            t_seg, t_save = time.time(), time.time()
            run = mid["run"] if mid else torch.zeros(6)
            kd_run = mid.get("kd_run") if mid else None
            self.sampler.set_epoch(epoch, i0 * cfg.batch)
            self.opt.zero_grad(set_to_none=True)
            for i, (imgs, targets, _) in enumerate(self.loader, start=i0):
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, cfg.nbs / cfg.batch]).round()))
                    for g in self.opt.param_groups:
                        start = cfg.warmup_bias_lr if g.get("kind") == "bias" else 0.0
                        g["lr"] = float(np.interp(ni, xi, [start, g["initial_lr"] * self.lr_factor(epoch)]))
                        if "momentum" in g:
                            g["momentum"] = float(np.interp(ni, xi, [cfg.warmup_momentum, cfg.momentum]))
                x = (imgs.to(self.device, non_blocking=True).float() / 255).contiguous(memory_format=self.mem_format)
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    preds = self.fwd(x)
                    t_preds = None
                    if self.teacher is not None:
                        with torch.no_grad():
                            t_preds = self.teacher(x)
                loss, items = self.criterion(preds, targets, (cfg.imgsz, cfg.imgsz), teacher=t_preds)
                if t_preds is not None:
                    l_kd, kd_items = distill_loss(preds, t_preds, self.dcfg)
                    loss = loss + l_kd
                    kd_items = kd_items.cpu()
                    kd_run = kd_items if kd_run is None else kd_run * (i / (i + 1)) + kd_items / (i + 1)
                self.scaler.scale(loss).backward()
                if ni - last_step >= self.accumulate:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update(self.model)
                    last_step = ni
                vec = torch.cat([items["o2m"], items.get("o2o", torch.zeros(3, device=items["o2m"].device))]).cpu()
                run = run * (i / (i + 1)) + vec / (i + 1)
                if (cfg.save_interval_min and last_step == ni and i + 1 < nb
                        and time.time() - t_save > cfg.save_interval_min * 60):
                    # right after an optimizer step, so no accumulated gradients are lost
                    self.save("last.pt", epoch, full=self._full_state(
                        best, best_metrics, last_step, t_start, iter=i + 1, run=run, kd_run=kd_run,
                        epoch_time=time.time() - t_ep))
                    self.log(f"checkpoint at epoch {epoch + 1} iteration {i + 1}/{nb}")
                    t_save = time.time()
                if i % self.cfg.log_interval == 0:
                    ips = (i + 1 - i0) * cfg.batch / (time.time() - t_seg)
                    self.log(f"ep {epoch + 1}/{cfg.epochs} it {i}/{nb} o2m[box {run[0]:.3f} cls {run[1]:.3f} "
                             f"l1 {run[2]:.3f}] o2o[box {run[3]:.3f} cls {run[4]:.3f} l1 {run[5]:.3f}] "
                             f"{ips:.1f} img/s" + (f" kd[cls {kd_run[0]:.3f} box {kd_run[1]:.3f}]"
                                                   if self.teacher is not None else ""))
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
            self.save("last.pt", epoch, full=self._full_state(best, best_metrics, last_step, t_start))
            (self.out / "history.json").write_text(json.dumps(self.history, indent=1))
            if epoch + 1 < cfg.epochs and should_pause(time.time() - t_ep):
                self.log("pausing after the epoch checkpoint (stop file or deadline; run is resumable)")
                raise SystemExit(STOPPED_EXIT_CODE)
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

    def save(self, name: str, epoch: int, full: dict | None = None) -> None:
        """Checkpoint with EMA weights (for inference); `full` adds the state needed to resume training."""
        ck = {"epoch": epoch, "model_cfg": asdict(self.model.cfg), "names": self.cfg.names,
              "ema": self.ema.ema.state_dict(), **(full or {})}
        tmp = self.out / f".{name}.tmp"
        torch.save(ck, tmp)
        tmp.replace(self.out / name)  # atomic: an interruption never leaves a truncated checkpoint


def load_for_classes(path: str, names: list[str], head: str = "subset") -> tuple[NextYOLO, str]:
    """Load a nextYOLO or Ultralytics YOLO26 checkpoint and adapt its classification heads to `names`.

    head="subset" slices the heads by class name when the checkpoint's classes are a superset (COCO -> VOC);
    otherwise (or with head="random") the cls heads are re-initialised and everything else is transferred.
    """
    from ..nn.model import subset_classes
    try:
        src = load_model(path)
        src_names = src.names
    except (KeyError, TypeError):  # not a nextYOLO checkpoint: try the Ultralytics format
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.convert_ultralytics import load_ultralytics
        src, src_names = load_ultralytics(path)
    from tools.val import class_index_map
    try:
        keep = class_index_map(src_names, names)
    except ValueError:
        keep = "missing"
    if keep is None:
        return src, "kept"
    if keep != "missing" and head == "subset":
        return subset_classes(src, keep), "subset"
    model = NextYOLO(replace(src.cfg, nc=len(names), cls_hidden=None))  # fresh cls branch, as Ultralytics does
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in src.state_dict().items() if k in own and own[k].shape == v.shape},
                          strict=False)
    return model, "random"


def _worker_init(worker_id: int) -> None:
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def load_model(path: str) -> NextYOLO:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: (tuple(v) if k == "levels" else v) for k, v in ck["model_cfg"].items()})
    if cfg.cls_hidden is None:  # older sliced-head checkpoints: read the cls-branch width off the weights
        w = next((v for k, v in ck["ema"].items() if k.endswith("cls.0.2.weight")), None)
        if w is not None:
            cfg = replace(cfg, cls_hidden=w.shape[1])
    model = NextYOLO(cfg)
    # older checkpoints also carried the head under a duplicate "head." prefix
    model.load_state_dict({k: v for k, v in ck["ema"].items() if not k.startswith("head.")})
    model.names = ck.get("names")
    return model.eval()
