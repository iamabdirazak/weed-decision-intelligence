"""Calibration: a spray threshold whose weed miss rate holds field by field.

A plant is sprayed when its weed score reaches the threshold. For one field
the miss rate can only grow as the threshold rises, so each field has a
largest threshold that keeps its own miss rate within the target alpha.

Treating fields as exchangeable units, the threshold is the conformal lower
quantile of those per-field values. A new field then stays within alpha with
probability at least 1 - delta [split conformal prediction, Vovk et al. 2005;
Angelopoulos and Bates 2021]. Weeds the detector never found enter with a
score of zero, so the promise covers them too. One threshold is fitted per
field condition.

When the quantile is zero, even spraying every detected plant would break
the target in too many fields. The condition is then marked not achievable,
rather than given a threshold that cannot keep the promise.
"""
from typing import Dict

import numpy as np


def largest_safe_threshold(weed_scores: np.ndarray, alpha: float) -> float:
    """Largest threshold t with share(weed_scores < t) <= alpha, for one field.

    Zero when more than alpha of the field's weeds were never detected.
    """
    s = np.sort(weed_scores)
    k = int(np.floor(alpha * len(s)))      # weeds that may be missed
    if k >= len(s):
        return np.inf
    t = s[k]                                # spraying at t keeps s[k:] and misses at most s[:k]
    return float(t) if t > 0 else 0.0


def conformal_lower_quantile(values: np.ndarray, delta: float) -> float:
    """Value v with P(new value >= v) >= 1 - delta, for exchangeable values."""
    v = np.sort(values)
    k = int(np.floor(delta * (len(v) + 1))) - 1
    return float(v[k]) if k >= 0 else 0.0


class FieldCalibration:
    """One spray threshold per field condition, valid field by field."""

    def __init__(self, alpha: float = 0.10, delta: float = 0.10, min_weeds: int = 20):
        self.alpha, self.delta, self.min_weeds = alpha, delta, min_weeds

    def fit(self, scores: np.ndarray, is_weed: np.ndarray, field: np.ndarray,
            condition: np.ndarray) -> "FieldCalibration":
        self.threshold: Dict[str, float] = {}
        self.fields: Dict[str, int] = {}
        for c in np.unique(condition):
            safe = [largest_safe_threshold(scores[(condition == c) & (field == f) & (is_weed == 1)], self.alpha)
                    for f in np.unique(field[condition == c])
                    if ((condition == c) & (field == f) & (is_weed == 1)).sum() >= self.min_weeds]
            self.fields[c] = len(safe)
            self.threshold[c] = conformal_lower_quantile(np.array(safe), self.delta)
        return self

    def achievable(self, condition: str) -> bool:
        return self.threshold[condition] > 0

    def spray(self, scores: np.ndarray, condition: np.ndarray) -> np.ndarray:
        """Spray decision per plant. Where the promise is not achievable, every plant is sprayed."""
        t = np.array([self.threshold[c] for c in condition])
        return np.where(t > 0, scores >= t, True)
