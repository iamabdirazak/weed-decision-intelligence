"""Two views behind the results: where weeds are lost, and how a threshold is chosen.

    weeds_lost.csv        for each condition, the share of test weeds the standard
                          cutoff keeps, the share the detector saw as weed but the
                          cutoff dropped, and the share it never saw as weed
    field_thresholds.csv  each calibration field's largest safe threshold for a
                          10% promise on clean images, and the threshold chosen

    python -m src.evaluation.breakdown
"""
import pandas as pd

from src.evaluation.data import load
from src.evaluation.metrics import MIN_WEEDS
from src.evaluation.promise import DELTA, STANDARD
from src.model.calibration import FieldCalibration, largest_safe_threshold
from src.model.detector import FAINT
from src.paths import REPORTS
from src.preparation.conditions import CONDITIONS


def weeds_lost(test):
    rows = []
    for c in CONDITIONS:
        t = test[(test.condition == c) & (test.weed == 1)]
        seen = ((t.w >= FAINT) & (t.w >= t.k)).mean()
        kept = ((t.w >= STANDARD) & (t.w >= t.k)).mean()
        rows.append(dict(condition=c, kept_by_cutoff=kept, dropped_by_cutoff=seen - kept, not_seen=1 - seen))
    return pd.DataFrame(rows)


def field_thresholds(cal, alpha=0.10, condition="clean"):
    weeds = cal[(cal.condition == condition) & (cal.weed == 1)]
    safe = [largest_safe_threshold(g.score.to_numpy(), alpha) for _, g in weeds.groupby("field") if len(g) >= MIN_WEEDS]
    model = FieldCalibration(alpha, DELTA, MIN_WEEDS).fit(cal.score.to_numpy(), cal.weed.to_numpy(),
                                                          cal.field.to_numpy(), cal.condition.to_numpy())
    return pd.DataFrame(dict(safe=sorted(safe))).assign(chosen=model.threshold[condition])


def main():
    weeds_lost(load("test")).to_csv(REPORTS / "weeds_lost.csv", index=False)
    field_thresholds(load("calib")).to_csv(REPORTS / "field_thresholds.csv", index=False)
    print("Written to", REPORTS)


if __name__ == "__main__":
    main()
