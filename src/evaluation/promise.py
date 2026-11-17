"""Does the promised weed miss rate hold field by field?

Every rule sprays a plant when its score clears a threshold, and weeds the
detector never found count as missed. Thresholds are fitted on the
calibration fields and scored on fields never seen before.

    standard cutoff            the detector's usual rule: weed is the top class at 0.25 or more
    conformal, detected weeds  the usual conformal setup: one threshold from clean images,
                               counting only the weeds the detector found
    conformal, pooled plants   one threshold per condition, every weed counted, plants pooled
    per field                  the model: one threshold per condition, fields as the unit

    python -m src.evaluation.promise               # on the test fields, run once
    python -m src.evaluation.promise --split val   # during development
"""
import argparse

import numpy as np
import pandas as pd

from src.evaluation.data import load
from src.evaluation.metrics import MIN_WEEDS, summary
from src.model.calibration import FieldCalibration
from src.paths import REPORTS
from src.preparation.conditions import CONDITIONS
from src.preparation.splitting import SEED

PROMISES = (0.05, 0.10, 0.15, 0.20)
DELTA = 0.10          # the promise holds in at least 90% of new fields
STANDARD = 0.25


def weed_threshold(weed_scores: np.ndarray, alpha: float) -> float:
    """Split conformal on the weed class: at most alpha of the weeds fall below it."""
    s = np.sort(weed_scores)
    k = int(np.floor(alpha * (len(s) + 1))) - 1
    return -np.inf if k < 0 else s[k]


def rules(cal: pd.DataFrame, plants: pd.DataFrame, alpha: float, condition: str, model: FieldCalibration):
    """Spray decisions of every rule for the plants of one condition."""
    found = weed_threshold(cal[(cal.condition == "clean") & (cal.weed == 1) & cal.found].score.to_numpy(), alpha)
    pooled = weed_threshold(cal[(cal.condition == condition) & (cal.weed == 1)].score.to_numpy(), alpha)
    return {
        "standard cutoff": ((plants.w >= STANDARD) & (plants.w >= plants.k)).to_numpy(),
        "conformal, detected weeds": (plants.found & (plants.score >= found)).to_numpy(),
        "conformal, pooled plants": (plants.score >= pooled).to_numpy() if pooled > 0 else np.ones(len(plants), bool),
        "per field": model.spray(plants.score.to_numpy(), plants.condition.to_numpy()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    split = ap.parse_args().split
    cal, new = load("calib"), load(split)
    rng = np.random.default_rng(SEED)
    results, thresholds = [], []
    for alpha in PROMISES:
        model = FieldCalibration(alpha, DELTA, MIN_WEEDS).fit(cal.score.to_numpy(), cal.weed.to_numpy(),
                                                              cal.field.to_numpy(), cal.condition.to_numpy())
        for c in CONDITIONS:
            thresholds.append(dict(promise=alpha, condition=c, achievable=model.achievable(c),
                                   threshold=model.threshold[c], fields=model.fields[c]))
            plants = new[new.condition == c].reset_index(drop=True)
            for rule, spray in rules(cal, plants, alpha, c, model).items():
                results.append(dict(promise=alpha, condition=c, rule=rule, **summary(plants, spray, alpha, rng)))
    REPORTS.mkdir(exist_ok=True)
    suffix = "" if split == "test" else f"_{split}"
    pd.DataFrame(results).to_csv(REPORTS / f"promise{suffix}.csv", index=False)
    pd.DataFrame(thresholds).to_csv(REPORTS / "thresholds.csv", index=False)
    print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
