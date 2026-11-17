"""Weed score: how strongly the detector favours weed over crop for one plant.

    s = w / (w + c)

with w and c the strongest weed and crop confidences among the detections
covering the plant. The score is high when the detector leans to weed even
if both confidences are low, which is common under blur and low light. A
plant no detection covers scores zero, so it can never be sprayed and
counts as missed when a weed.
"""
import numpy as np


def weed_score(weed: np.ndarray, crop: np.ndarray) -> np.ndarray:
    found = np.maximum(weed, crop) > 0
    return np.where(found, weed / np.maximum(weed + crop, 1e-9), 0.0)
