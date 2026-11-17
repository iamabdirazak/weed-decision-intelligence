"""The detector's training set, in YOLO format.

Train and validation photos are linked, not copied. Each training photo
also gets one damaged copy under a random field condition, so the detector
has seen blur, low light and glare before it meets them. Plants marked
neither crop nor weed are left out. Resumable, so it can run in chunks.

    python -m src.preparation.detector_data --limit 1500
"""
import argparse
import os
import zlib

import cv2
import numpy as np
import pandas as pd

from src.acquisition import cropandweed as cnw
from src.paths import PROCESSED
from src.preparation.conditions import CONDITIONS, apply
from src.preparation.splitting import SEED

CLASSES = {"crop": 0, "weed": 1}
OUT = PROCESSED / "detector"


def label_lines(boxes):
    lines = []
    for b in boxes:
        if b["label"] in CLASSES:
            cx, cy = (b["left"] + b["right"]) / 2 / cnw.WIDTH, (b["top"] + b["bottom"]) / 2 / cnw.HEIGHT
            w, h = (b["right"] - b["left"]) / cnw.WIDTH, (b["bottom"] - b["top"]) / cnw.HEIGHT
            lines.append(f"{CLASSES[b['label']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    plants = pd.read_csv(PROCESSED / "plants.csv")
    split = plants.drop_duplicates("image").set_index("image").split
    damaged = [c for c in CONDITIONS if c != "clean"]
    rng = np.random.default_rng(SEED)
    choice = {n: damaged[i] for n, i in zip(split.index, rng.integers(0, len(damaged), len(split)))}

    done = 0
    for name, s in split.items():
        if s not in ("train", "val"):
            continue
        images, labels = OUT / "images" / s, OUT / "labels" / s
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        lines = label_lines(cnw.load_boxes(name))
        link = images / f"{name}.jpg"
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(cnw.image_path(name), images), link)
            (labels / f"{name}.txt").write_text(lines)
        if s == "train":
            condition = choice[name]
            copy = images / f"{name}_{condition}.jpg"
            if not copy.exists():
                frame = apply(cv2.imread(str(cnw.image_path(name))), condition, seed=zlib.crc32(name.encode()))
                cv2.imwrite(str(copy), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (labels / f"{name}_{condition}.txt").write_text(lines)
                done += 1
                if args.limit and done >= args.limit:
                    print("Chunk done, run again to continue")
                    return
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: crop\n  1: weed\n")
    print(f"Training set ready in {OUT}")


if __name__ == "__main__":
    main()
