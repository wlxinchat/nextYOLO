"""YOLO-format detection dataset with RAM caching and YOLO-style augmentation.

Layout: `<root>/images/<split>/*.jpg` with labels at `<root>/labels/<split>/*.txt`, one `cls cx cy w h` row per
object (normalised). Images are decoded once, resized so the long side equals `imgsz`, and kept in RAM, which makes
mosaic augmentation cheap enough to train on a CPU.
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def img2label_path(p: str) -> str:
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return sb.join(p.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def list_images(sources: str | list[str]) -> list[str]:
    files = []
    for s in [sources] if isinstance(sources, str) else sources:
        p = Path(s)
        if p.is_dir():
            files += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        elif p.suffix == ".txt":
            files += [x.strip() for x in p.read_text().splitlines() if x.strip()]
        else:
            raise FileNotFoundError(s)
    return files


def read_labels(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((0, 5), np.float32)
    rows = [r.split() for r in Path(path).read_text().splitlines() if r.strip()]
    lb = np.array(rows, dtype=np.float32).reshape(-1, 5) if rows else np.zeros((0, 5), np.float32)
    lb[:, 1:] = lb[:, 1:].clip(0, 1)
    keep = (lb[:, 3] > 1e-4) & (lb[:, 4] > 1e-4)
    return lb[keep]


def letterbox(img: np.ndarray, new: int, color=114):
    """Resize long side to `new` (keeping aspect), pad to new x new. Returns img, ratio, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = new / max(h, w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (new - nw) / 2, (new - nh) / 2
    top, bottom = round(py - 0.1), round(py + 0.1)
    left, right = round(px - 0.1), round(px + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, (left, top)


def augment_hsv(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4) -> None:
    if not (hgain or sgain or vgain):
        return
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_affine(img, boxes, out_size: int, border: int = 0, degrees=0.0, translate=0.1, scale=0.5, shear=0.0):
    """Random scale/translate (+optional rotate/shear) of an image and its xyxy pixel boxes."""
    h, w = img.shape[:2]
    C = np.eye(3)
    C[0, 2], C[1, 2] = -w / 2, -h / 2
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D((0, 0), a, s)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    M = T @ S @ R @ C
    img = cv2.warpAffine(img, M[:2], (out_size, out_size), borderValue=(114, 114, 114))
    if len(boxes):
        n = len(boxes)
        pts = np.ones((n * 4, 3))
        pts[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        pts = (pts @ M.T)[:, :2].reshape(n, 8)
        xs, ys = pts[:, [0, 2, 4, 6]], pts[:, [1, 3, 5, 7]]
        new = np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), 1).clip(0, out_size)
        # Keep boxes that survive with enough area and a sane aspect ratio.
        w0, h0 = (boxes[:, 2] - boxes[:, 0]) * s, (boxes[:, 3] - boxes[:, 1]) * s
        w1, h1 = new[:, 2] - new[:, 0], new[:, 3] - new[:, 1]
        ar = np.maximum(w1 / (h1 + 1e-16), h1 / (w1 + 1e-16))
        keep = (w1 > 2) & (h1 > 2) & (w1 * h1 / (w0 * h0 + 1e-16) > 0.1) & (ar < 100)
        return img, new, keep
    return img, boxes, np.zeros(0, bool)


class YOLODataset(Dataset):
    def __init__(self, sources, imgsz: int = 640, train: bool = True, hyp: dict | None = None, cache: bool = True,
                 max_images: int | None = None, workers: int = 8):
        self.files = list_images(sources)
        if max_images:
            self.files = self.files[:max_images]
        self.imgsz = imgsz
        self.train = train
        self.hyp = dict(mosaic=1.0, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, fliplr=0.5,
                        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mixup=0.0)
        self.hyp.update(hyp or {})
        self.labels = [read_labels(img2label_path(f)) for f in self.files]
        self.orig_shapes: list[tuple[int, int]] = [None] * len(self.files)
        self.imgs: list[np.ndarray | None] = [None] * len(self.files)
        if cache:
            with ThreadPoolExecutor(workers) as ex:
                for i, (im, hw) in enumerate(ex.map(self._read_resized, range(len(self.files)))):
                    self.imgs[i], self.orig_shapes[i] = im, hw

    def __len__(self):
        return len(self.files)

    def _read_resized(self, i: int):
        im = cv2.imread(self.files[i])
        if im is None:
            raise FileNotFoundError(self.files[i])
        h, w = im.shape[:2]
        r = self.imgsz / max(h, w)
        if r != 1:
            im = cv2.resize(im, (round(w * r), round(h * r)),
                            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        return im, (h, w)

    def load_image(self, i: int) -> np.ndarray:
        if self.imgs[i] is None:
            im, hw = self._read_resized(i)
            self.orig_shapes[i] = hw
            return im
        return self.imgs[i]

    def _pixel_boxes(self, i: int, w: int, h: int, ox: float = 0, oy: float = 0):
        lb = self.labels[i]
        xy = lb[:, 1:3] * [w, h]
        wh = lb[:, 3:5] * [w, h]
        boxes = np.concatenate((xy - wh / 2, xy + wh / 2), 1) + [ox, oy, ox, oy]
        return lb[:, 0].copy(), boxes

    def _mosaic(self, i: int):
        s = self.imgsz
        yc, xc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        idxs = [i] + random.choices(range(len(self)), k=3)
        canvas = np.full((2 * s, 2 * s, 3), 114, np.uint8)
        cls_all, box_all = [], []
        for k, j in enumerate(idxs):
            im = self.load_image(j)
            h, w = im.shape[:2]
            if k == 0:    # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:         # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
            c, b = self._pixel_boxes(j, w, h, x1a - x1b, y1a - y1b)
            cls_all.append(c)
            box_all.append(b)
        cls = np.concatenate(cls_all)
        boxes = np.concatenate(box_all).clip(0, 2 * s)
        return canvas, cls, boxes

    def __getitem__(self, i: int):
        hyp = self.hyp
        if self.train and random.random() < hyp["mosaic"]:
            img, cls, boxes = self._mosaic(i)
            img, boxes, keep = random_affine(img, boxes, self.imgsz, degrees=hyp["degrees"],
                                             translate=hyp["translate"], scale=hyp["scale"], shear=hyp["shear"])
            cls, boxes = cls[keep], boxes[keep]
            meta = None
        else:
            im = self.load_image(i)
            h, w = im.shape[:2]
            img, r, (px, py) = letterbox(im, self.imgsz)
            cls, boxes = self._pixel_boxes(i, w * r, h * r, px, py)
            if self.train:
                img, boxes, keep = random_affine(img, boxes, self.imgsz, degrees=hyp["degrees"],
                                                 translate=hyp["translate"], scale=hyp["scale"], shear=hyp["shear"])
                cls, boxes = cls[keep], boxes[keep]
                meta = None
            else:
                oh, ow = self.orig_shapes[i] or (h, w)
                meta = dict(index=i, orig_shape=(oh, ow), ratio=r * max(h, w) / max(oh, ow), pad=(px, py))
        if self.train:
            augment_hsv(img, hyp["hsv_h"], hyp["hsv_s"], hyp["hsv_v"])
            if random.random() < hyp["fliplr"]:
                img = np.ascontiguousarray(img[:, ::-1])
                if len(boxes):
                    boxes[:, [0, 2]] = self.imgsz - boxes[:, [2, 0]]
        s = self.imgsz
        n = len(boxes)
        lab = np.zeros((n, 5), np.float32)
        if n:
            lab[:, 0] = cls
            lab[:, 1] = (boxes[:, 0] + boxes[:, 2]) / 2 / s
            lab[:, 2] = (boxes[:, 1] + boxes[:, 3]) / 2 / s
            lab[:, 3] = (boxes[:, 2] - boxes[:, 0]) / s
            lab[:, 4] = (boxes[:, 3] - boxes[:, 1]) / s
        img = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))  # BGR->RGB, HWC->CHW
        return img, torch.from_numpy(lab), meta

    def gt_original(self, i: int):
        """Ground truth for evaluation in original-image pixels: cls (n,), xyxy (n, 4)."""
        oh, ow = self.orig_shapes[i]
        return self._pixel_boxes(i, ow, oh)


def collate(batch):
    imgs, labels, metas = zip(*batch)
    targets = [torch.cat((torch.full((len(lb), 1), float(k)), lb), 1) for k, lb in enumerate(labels)]
    return torch.stack(imgs), torch.cat(targets), list(metas)
