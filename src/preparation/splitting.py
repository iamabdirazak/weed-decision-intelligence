"""Splits by field visit, so no field appears on both sides of any comparison.

Photos from one visit share light, soil and growth stage, and are near
copies of each other. A split by photo would let the model and the
calibration see the test fields in advance.
"""
import random
from typing import Dict, List

from src.acquisition.cropandweed import session_of

SEED = 7
SPLITS = {"train": 0.60, "val": 0.10, "calib": 0.15, "test": 0.15}


def split_by_session(names: List[str], fractions: Dict[str, float] = SPLITS, seed: int = SEED) -> Dict[str, str]:
    """Assign every image to a split, keeping each session whole.

    Sessions are shuffled with a fixed seed, then filled into the splits in
    order until each holds about its share of images.
    """
    sessions: Dict[str, List[str]] = {}
    for n in names:
        sessions.setdefault(session_of(n), []).append(n)
    order = sorted(sessions)
    random.Random(seed).shuffle(order)
    bounds, acc = [], 0.0
    for split, fraction in fractions.items():
        acc += fraction
        bounds.append((split, acc * len(names)))
    assigned, out = 0, {}
    for s in order:
        split = next(sp for sp, b in bounds if assigned < b or sp == bounds[-1][0])
        for n in sessions[s]:
            out[n] = split
        assigned += len(sessions[s])
    return out
