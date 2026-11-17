import numpy as np

from src.model.calibration import FieldCalibration, conformal_lower_quantile, largest_safe_threshold


def test_largest_safe_threshold_misses_at_most_alpha():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.random(rng.integers(20, 200))
        t = largest_safe_threshold(s, 0.1)
        assert (s < t).mean() <= 0.1
        assert (s < np.nextafter(t, 2)).mean() > 0.1 or t == s.max()


def test_undetected_weeds_make_a_condition_unachievable():
    s = np.r_[np.zeros(30), np.ones(70)]        # 30% of the weeds never detected
    assert largest_safe_threshold(s, 0.1) == 0.0


def test_conformal_quantile_rank():
    v = np.arange(1, 20, dtype=float)            # 19 fields
    assert conformal_lower_quantile(v, 0.1) == 2.0


def test_promise_holds_on_new_fields():
    rng = np.random.default_rng(1)
    kept = []
    for _ in range(300):
        fields = [np.clip(rng.beta(5, 2, 40) + rng.normal(0, 0.15), 0, 1) for _ in range(80)]
        cal, new = fields[:60], fields[60:]
        scores = np.concatenate(cal)
        model = FieldCalibration(alpha=0.1, delta=0.1, min_weeds=1).fit(
            scores, np.ones_like(scores, int), np.repeat(np.arange(60), 40), np.array(["c"] * len(scores)))
        kept += [(f < model.threshold["c"]).mean() <= 0.1 for f in new]
    assert np.mean(kept) >= 0.88
