"""One row per labelled crop or weed plant, with its split.

Writes data/processed/plants.csv, which every later step reads, and the
dataset summary in reports/splits.csv.

    python -m src.preparation.plants
"""
import pandas as pd

from src.acquisition import cropandweed as cnw
from src.paths import PROCESSED, REPORTS
from src.preparation.splitting import SPLITS, split_by_session


def main():
    names = cnw.image_names()
    split = split_by_session(names)
    rows = []
    for name in names:
        parameters = cnw.load_parameters(name)
        for p in cnw.load_boxes(name):
            if p["label"] in ("crop", "weed"):
                rows.append(dict(image=name, session=cnw.session_of(name), split=split[name],
                                 area=(p["right"] - p["left"]) * (p["bottom"] - p["top"]), **p, **parameters))
    plants = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    plants.to_csv(PROCESSED / "plants.csv", index=False)

    summary = pd.DataFrame([dict(split=s, field_visits=part.session.nunique(), images=part.image.nunique(),
                                 crops=int((part.label == "crop").sum()), weeds=int((part.label == "weed").sum()))
                            for s in SPLITS for part in [plants[plants.split == s]]])
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS / "splits.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
