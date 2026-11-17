"""Every labelled plant of a split under every condition, with its weed score and field."""
import numpy as np
import pandas as pd

from src.acquisition.cropandweed import session_of
from src.evaluation.scores import SCORES
from src.model.score import weed_score
from src.paths import PROCESSED
from src.preparation.conditions import CONDITIONS


def load(split: str) -> pd.DataFrame:
    plants = pd.read_csv(PROCESSED / "plants.csv", usecols=["image", "split", "label"])
    plants = plants[plants.split == split]
    weed, field = (plants.label == "weed").astype(int).to_numpy(), plants.image.map(session_of).to_numpy()
    frames = []
    for c in CONDITIONS:
        w = np.load(SCORES / f"{split}_{c}_weed.npy")
        k = np.load(SCORES / f"{split}_{c}_crop.npy")
        frames.append(pd.DataFrame(dict(score=weed_score(w, k), w=w, k=k, found=np.maximum(w, k) > 0,
                                        weed=weed, field=field, condition=c)))
    return pd.concat(frames, ignore_index=True)
