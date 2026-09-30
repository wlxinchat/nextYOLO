"""Convert Pascal VOC (VOCdevkit) to YOLO layout: images/<split>/ (symlinks) + labels/<split>/*.txt.

Splits: train2007, val2007, train2012, val2012 (training: 16,551 images) and test2007 (evaluation: 4,952 images).
Objects flagged `difficult` are dropped, matching the common YOLO convention.

    python tools/prepare_voc.py --src /path/to/VOCdevkit --dst /path/to/VOC
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
         "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(src: Path, dst: Path) -> None:
    for year, split in [("2007", "train"), ("2007", "val"), ("2012", "train"), ("2012", "val"), ("2007", "test")]:
        root = src / f"VOC{year}"
        ids = (root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
        img_dir, lb_dir = dst / "images" / f"{split}{year}", dst / "labels" / f"{split}{year}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lb_dir.mkdir(parents=True, exist_ok=True)
        n_obj = 0
        for i in ids:
            tree = ET.parse(root / "Annotations" / f"{i}.xml").getroot()
            size = tree.find("size")
            w, h = int(size.find("width").text), int(size.find("height").text)
            rows = []
            for obj in tree.iter("object"):
                name = obj.find("name").text.strip()
                diff = obj.find("difficult")
                if name not in NAMES or (diff is not None and int(diff.text) == 1):
                    continue
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
                x1, y1 = x1 - 1, y1 - 1  # VOC pixel indices are 1-based
                rows.append(f"{NAMES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            n_obj += len(rows)
            (lb_dir / f"{i}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            link = img_dir / f"{i}.jpg"
            if not link.exists():
                os.symlink(root / "JPEGImages" / f"{i}.jpg", link)
        print(f"{split}{year}: {len(ids)} images, {n_obj} objects")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    a = ap.parse_args()
    convert(a.src.resolve(), a.dst.resolve())
