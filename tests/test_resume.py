"""Training resumes from a full-state checkpoint after an interruption, and a finished run is not re-run."""

import cv2
import numpy as np
import pytest

from nextyolo.engine.trainer import TrainConfig, Trainer


def _synthetic_dataset(root, n=8, size=96):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = np.full((size, size, 3), 114, np.uint8)
            rows = []
            for _ in range(2):
                w, h = rng.uniform(0.2, 0.5, 2)
                cx, cy = rng.uniform(0.3, 0.7, 2)
                c = int(rng.integers(0, 2))
                x1, y1, x2, y2 = (np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * size).astype(int)
                img[y1:y2, x1:x2] = (255, 0, 0) if c == 0 else (0, 0, 255)
                rows.append(f"{c} {cx} {cy} {w} {h}")
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            (root / "labels" / split / f"{i}.txt").write_text("\n".join(rows))
    return [str(root / "images" / "train")], [str(root / "images" / "val")]


def test_resume_after_interruption(tmp_path, monkeypatch):
    train, val = _synthetic_dataset(tmp_path / "data")
    cfg = dict(train=train, val=val, names=["a", "b"], out=str(tmp_path / "run"), epochs=2, batch=4, imgsz=64,
               workers=0, warmup_epochs=0, close_mosaic=1, eval_interval=1)

    class Interrupt(Exception):
        pass

    orig_save = Trainer.save

    def crash_after_first_epoch(self, name, epoch, full=None):
        orig_save(self, name, epoch, full)
        if name == "last.pt" and epoch == 0:
            raise Interrupt

    monkeypatch.setattr(Trainer, "save", crash_after_first_epoch)
    with pytest.raises(Interrupt):
        Trainer(TrainConfig(**cfg)).train()
    monkeypatch.setattr(Trainer, "save", orig_save)

    t = Trainer(TrainConfig(**cfg))
    summary = t.train()
    log = (tmp_path / "run" / "log.txt").read_text()
    assert "resumed from" in log and "after epoch 1" in log
    assert [h["epoch"] for h in t.history] == [1, 2]
    assert "closing mosaic" in log and summary["best_mAP"] >= 0
    Trainer(TrainConfig(**cfg)).train()
    assert "nothing to do" in (tmp_path / "run" / "log.txt").read_text()


def test_epoch_sampler_resume_replays_the_rest_of_the_epoch():
    from nextyolo.engine.trainer import EpochSampler

    s = EpochSampler(10, seed=3)
    s.set_epoch(2)
    full = list(s)
    s.set_epoch(2, start=4)
    assert list(s) == full[4:] and len(s) == 6
    s.set_epoch(3)
    assert sorted(list(s)) == list(range(10)) and list(s) != full


def test_resume_mid_epoch(tmp_path, monkeypatch):
    """A preempted run continues from a mid-epoch checkpoint and trains each remaining batch exactly once."""
    from nextyolo.engine import trainer as tr

    train, val = _synthetic_dataset(tmp_path / "data")
    cfg = dict(train=train, val=val, names=["a", "b"], out=str(tmp_path / "run"), epochs=2, batch=2, nbs=2,
               imgsz=64, workers=0, warmup_epochs=0, close_mosaic=0, eval_interval=2, save_interval_min=1e-9)

    class Interrupt(Exception):
        pass

    orig_save = Trainer.save
    seen = []

    def crash_at_second_mid_checkpoint(self, name, epoch, full=None):
        orig_save(self, name, epoch, full)
        if full and full.get("iter") == 2 and epoch == 0:
            seen.append(full["last_step"])
            raise Interrupt

    monkeypatch.setattr(Trainer, "save", crash_at_second_mid_checkpoint)
    with pytest.raises(Interrupt):
        Trainer(TrainConfig(**cfg)).train()
    monkeypatch.setattr(Trainer, "save", orig_save)

    batches = []
    orig_iter = tr.EpochSampler.__iter__
    monkeypatch.setattr(tr.EpochSampler, "__iter__", lambda s: iter(batches.append((s.epoch, s.start)) or
                                                                     orig_iter(s)))
    t = Trainer(TrainConfig(**cfg))
    t.train()
    log = (tmp_path / "run" / "log.txt").read_text()
    assert "resumed from" in log and "in epoch 1 at iteration 2" in log
    assert batches[0] == (0, 4)          # epoch 1 continues after the 2 trained batches (2 images each)
    assert batches[1] == (1, 0)          # and epoch 2 starts from the top
    assert seen == [1] and [h["epoch"] for h in t.history] == [1, 2]
    assert (tmp_path / "run" / "summary.json").exists()


def test_stop_file_exits_resumably(tmp_path, monkeypatch):
    from nextyolo.engine.trainer import STOPPED_EXIT_CODE

    train, val = _synthetic_dataset(tmp_path / "data")
    cfg = dict(train=train, val=val, names=["a", "b"], out=str(tmp_path / "run"), epochs=2, batch=4, imgsz=64,
               workers=0, warmup_epochs=0, close_mosaic=0, eval_interval=1)
    stop = tmp_path / "STOP"
    stop.touch()
    monkeypatch.setenv("NEXTYOLO_STOP_FILE", str(stop))
    with pytest.raises(SystemExit) as e:
        Trainer(TrainConfig(**cfg)).train()
    assert e.value.code == STOPPED_EXIT_CODE
    assert not (tmp_path / "run" / "summary.json").exists()
    stop.unlink()
    Trainer(TrainConfig(**cfg)).train()
    assert (tmp_path / "run" / "summary.json").exists()
    assert "after epoch 1" in (tmp_path / "run" / "log.txt").read_text()


def test_deadline_pauses_before_an_epoch_that_would_not_fit(monkeypatch):
    import time

    from nextyolo.engine.trainer import should_pause

    monkeypatch.delenv("NEXTYOLO_STOP_FILE", raising=False)
    monkeypatch.setenv("NEXTYOLO_DEADLINE", str(time.time() + 600))
    assert not should_pause(100)   # 110 s + 60 s margin fits in 600 s
    assert should_pause(500)       # 550 s + 60 s does not
    monkeypatch.delenv("NEXTYOLO_DEADLINE")
    assert not should_pause(10**6)
