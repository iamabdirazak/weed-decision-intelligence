"""The detector's crop and weed confidence for every labelled plant, under every condition.

Runs the released detector through ONNX Runtime on 4 CPU threads, as on the
machine. A plant counts as reached by a detection when its centre lies
inside the detection box, the rule that matches spraying: a nozzle fired
over the detection reaches the plant. Each plant keeps the strongest weed
and crop confidence of any detection that reaches it, and zero for both when
none does.

    python -m src.evaluation.scores --split calib
    python -m src.evaluation.scores --split test
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from src.acquisition import cropandweed as cnw
from src.model.detector import Detector
from src.paths import MODELS, PROCESSED
from src.preparation.conditions import CONDITIONS, apply

SCORES = PROCESSED / "scores"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "calib", "test"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = ap.parse_args()
    detect = Detector(MODELS / "detector.onnx")
    plants = pd.read_csv(PROCESSED / "plants.csv")
    rows = plants[plants.split == args.split]
    names = list(dict.fromkeys(rows.image))
    SCORES.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        i = list(CONDITIONS).index(condition)
        weed, crop = [], []
        for k, name in enumerate(names):
            frame = apply(cv2.imread(str(cnw.image_path(name))), condition, seed=k * 10 + i)
            gt = rows[rows.image == name][["left", "top", "right", "bottom"]].to_numpy(float)
            box, confidence = detect(frame)
            if len(box):
                cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
                inside = ((cx[:, None] >= box[None, :, 0]) & (cx[:, None] <= box[None, :, 2])
                          & (cy[:, None] >= box[None, :, 1]) & (cy[:, None] <= box[None, :, 3]))
            for c, store in ((1, weed), (0, crop)):
                store.append(np.where(inside, confidence[:, c][None, :], 0.0).max(1) if len(box) else np.zeros(len(gt)))
        np.save(SCORES / f"{args.split}_{condition}_weed.npy", np.concatenate(weed))
        np.save(SCORES / f"{args.split}_{condition}_crop.npy", np.concatenate(crop))
        print(f"{args.split} {condition}: done")


if __name__ == "__main__":
    main()
