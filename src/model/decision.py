"""The weed decision model: a camera frame in, a spray decision per plant out.

Loads the released detector and the calibrated thresholds from models/.
The caller states the field condition, which the machine knows from its
speed and exposure settings, and the weed miss rate it wants to promise.

    model = WeedDecision.load()
    decision = model(frame, condition="blur_17", promise=0.10)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.model.detector import Detector
from src.model.score import weed_score
from src.paths import MODELS


@dataclass
class Decision:
    boxes: np.ndarray        # (n, 4) detected plants, left, top, right, bottom in pixels
    scores: np.ndarray       # (n,) weed score of each plant
    spray: np.ndarray        # (n,) True where the plant is to be sprayed
    achievable: bool         # False when the promise cannot be kept under this condition
    threshold: float         # the threshold used, 0 when every plant is sprayed


class WeedDecision:
    def __init__(self, detector: Detector, thresholds: dict):
        self.detect, self.thresholds = detector, thresholds

    @classmethod
    def load(cls, folder: Path = MODELS, threads: int = 4) -> "WeedDecision":
        thresholds = json.loads((folder / "thresholds.json").read_text())["thresholds"]
        return cls(Detector(folder / "detector.onnx", threads=threads), thresholds)

    def __call__(self, frame_bgr: np.ndarray, condition: str = "clean", promise: float = 0.10) -> Decision:
        threshold = self.thresholds[f"{promise:.2f}"][condition]
        boxes, confidence = self.detect(frame_bgr)
        scores = weed_score(confidence[:, 1], confidence[:, 0])
        achievable = threshold > 0
        spray = scores >= threshold if achievable else np.ones(len(scores), bool)
        return Decision(boxes, scores, spray, achievable, threshold)
