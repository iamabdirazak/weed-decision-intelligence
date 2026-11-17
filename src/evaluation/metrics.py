"""How a spray decision is judged: field by field, with every weed counted."""
import numpy as np
import pandas as pd

MIN_WEEDS = 20      # fields with fewer labelled weeds are too small to score a miss rate
BOOTSTRAP = 1000


def fields_kept(plants: pd.DataFrame, spray: np.ndarray, alpha: float) -> np.ndarray:
    """For each field with enough weeds, whether its weed miss rate stayed within alpha."""
    weeds = plants[plants.weed == 1].assign(miss=~spray[plants.weed.to_numpy() == 1])
    per_field = weeds.groupby("field").miss.agg(["mean", "size"])
    per_field = per_field[per_field["size"] >= MIN_WEEDS]
    return (per_field["mean"] <= alpha).to_numpy()


def summary(plants: pd.DataFrame, spray: np.ndarray, alpha: float, rng: np.random.Generator) -> dict:
    """Share of fields kept with a 95% bootstrap interval over fields, weeds missed and crops sprayed."""
    kept = fields_kept(plants, spray, alpha)
    boot = [rng.choice(kept, len(kept)).mean() for _ in range(BOOTSTRAP)]
    weed = plants.weed.to_numpy() == 1
    return dict(fields=len(kept), fields_kept=kept.mean(), low=np.percentile(boot, 2.5),
                high=np.percentile(boot, 97.5), weeds_missed=(~spray[weed]).mean(), crops_sprayed=spray[~weed].mean())
