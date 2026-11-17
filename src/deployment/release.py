"""Fits the released thresholds on the calibration fields and writes the model card.

    models/thresholds.json   one threshold per promise and condition, 0 where not achievable
    models/model.json        what the model is, what it was trained and calibrated on, how it did

    python -m src.deployment.release
"""
import json

import numpy as np
import onnx
import pandas as pd

from src.evaluation.data import load
from src.evaluation.metrics import MIN_WEEDS
from src.evaluation.promise import DELTA, PROMISES
from src.model.calibration import FieldCalibration
from src.paths import MODELS, REPORTS
from src.preparation.conditions import CONDITIONS


def thresholds():
    cal = load("calib")
    out = {}
    for alpha in PROMISES:
        model = FieldCalibration(alpha, DELTA, MIN_WEEDS).fit(cal.score.to_numpy(), cal.weed.to_numpy(),
                                                              cal.field.to_numpy(), cal.condition.to_numpy())
        out[f"{alpha:.2f}"] = {c: model.threshold[c] for c in CONDITIONS}
    return out


def card(table):
    detector = onnx.load(str(MODELS / "detector.onnx"))
    parameters = sum(int(np.prod(t.dims)) for t in detector.graph.initializer)
    results = pd.read_csv(REPORTS / "promise.csv")
    ours = results[results.rule == "per field"]
    held = {f"{r.promise:.2f}/{r.condition}": round(r.fields_kept, 3) for r in ours.itertuples()
            if table[f"{r.promise:.2f}"][r.condition] > 0}
    return {
        "name": "Weed Decision Model",
        "task": "Spray or skip for every plant in a field image, with a promised weed miss rate per field",
        "detector": {"file": "detector.onnx", "architecture": "YOLO11n, fine-tuned on CropAndWeed (CropOrWeed2)",
                     "input": "BGR image, letterboxed to 640 x 640", "classes": ["crop", "weed"],
                     "parameters": parameters, "lowest_confidence_kept": 0.01},
        "calibration": {"file": "thresholds.json", "fields": "149 calibration field visits, 88 with at least 20 weeds",
                        "promise_holds_in": f"at least {1 - DELTA:.0%} of new fields",
                        "conditions": list(CONDITIONS), "promises": [f"{a:.2f}" for a in PROMISES]},
        "test": {"fields": "121 field visits never seen in training or calibration, 69 with at least 20 weeds",
                 "share_of_fields_within_promise": held},
        "limits": ["Calibrated on CropAndWeed field visits; a new crop, region or camera needs new calibration fields.",
                   "Conditions are simulated on the images, not recorded on a moving machine.",
                   "Under strong low light and glare no threshold keeps the promise, and every plant is sprayed."],
    }


def main():
    table = thresholds()
    (MODELS / "thresholds.json").write_text(json.dumps(
        {"promise_holds_in_share_of_fields": 1 - DELTA, "thresholds": table}, indent=2))
    (MODELS / "model.json").write_text(json.dumps(card(table), indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
