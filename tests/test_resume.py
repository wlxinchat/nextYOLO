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
