"""The released model end to end: a frame in, a decision per plant out."""
import numpy as np

from src.model import WeedDecision
from src.model.score import weed_score

FRAME = np.random.default_rng(0).integers(0, 255, (1088, 1920, 3), np.uint8)


def test_decision_has_one_entry_per_plant():
    d = WeedDecision.load()(FRAME, "clean", 0.10)
    assert d.achievable and len(d.boxes) == len(d.scores) == len(d.spray)


def test_unachievable_condition_sprays_every_plant():
    d = WeedDecision.load()(FRAME, "dark_25", 0.10)
    assert not d.achievable and d.spray.all()


def test_undetected_plant_scores_zero():
    assert weed_score(np.array([0.0, 0.3]), np.array([0.0, 0.1]))[0] == 0.0
