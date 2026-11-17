"""Does every part of the model earn its place?

Each variant removes one part and is scored on the same test fields with the
same detector confidences. Nothing is retrained.

    full model             faint detections kept, weed-against-crop score,
                           undetected weeds counted, one threshold per condition
    no faint detections    detections below the usual 0.25 confidence removed
    weed confidence only   the score is the weed confidence alone
    undetected ignored     calibration sees only the plants the detector found
    one threshold          a single threshold across all conditions

    python -m src.evaluation.ablations
"""
import numpy as np
import pandas as pd

from src.evaluation.data import load
from src.evaluation.metrics import MIN_WEEDS, summary
from src.evaluation.promise import DELTA, PROMISES, STANDARD
from src.model.calibration import FieldCalibration
from src.paths import REPORTS
from src.preparation.conditions import CONDITIONS
from src.preparation.splitting import SEED

VARIANTS = {
    "full model": dict(),
    "no faint detections": dict(faint=False),
    "weed confidence only": dict(crop_aware=False),
    "undetected ignored": dict(found_only=True),
    "one threshold": dict(single=True),
}


def rescore(plants, faint=True, crop_aware=True):
    w, k = plants.w.to_numpy(), plants.k.to_numpy()
    if not faint:
        keep = np.maximum(w, k) >= STANDARD
        w, k = np.where(keep, w, 0.0), np.where(keep, k, 0.0)
    found = np.maximum(w, k) > 0
    score = w / np.maximum(w + k, 1e-9) if crop_aware else w
    return plants.assign(score=np.where(found, score, 0.0), found=found)


def main():
    cal0, test0 = load("calib"), load("test")
    rng = np.random.default_rng(SEED)
    rows = []
    for name, option in VARIANTS.items():
        scoring = {k: option[k] for k in ("faint", "crop_aware") if k in option}
        cal, test = rescore(cal0, **scoring), rescore(test0, **scoring)
        single, found_only = option.get("single", False), option.get("found_only", False)
        c = cal[cal.found] if found_only else cal
        condition = np.full(len(c), "all") if single else c.condition.to_numpy()
        field = (c.field + "|" + c.condition).to_numpy() if single else c.field.to_numpy()
        for alpha in PROMISES:
            model = FieldCalibration(alpha, DELTA, MIN_WEEDS).fit(c.score.to_numpy(), c.weed.to_numpy(), field, condition)
            for cond in CONDITIONS:
                plants = test[test.condition == cond].reset_index(drop=True)
                key = np.full(len(plants), "all") if single else plants.condition.to_numpy()
                spray = model.spray(plants.score.to_numpy(), key)
                rows.append(dict(variant=name, promise=alpha, condition=cond,
                                 achievable=model.achievable("all" if single else cond),
                                 **summary(plants, spray, alpha, rng)))
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "ablations.csv", index=False)
    cells = df.merge(df[(df.variant == "full model") & df.achievable][["promise", "condition"]], on=["promise", "condition"])
    table = (cells.groupby("variant", sort=False)
             .agg(fields_kept=("fields_kept", "mean"), crops_sprayed=("crops_sprayed", "mean"))
             .join(df.groupby("variant", sort=False).achievable.sum().rename("achievable")))
    print(table.round(3))


if __name__ == "__main__":
    main()
