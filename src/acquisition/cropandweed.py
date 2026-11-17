"""Reading CropAndWeed [Steininger et al., WACV 2023] in its binary CropOrWeed2 variant.

Image names look like 'vwg-0328-0012': image set, recording session, image.
A recording session is one visit to one field, and it is the unit every
split and every result is built on.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List

from src.paths import RAW

VARIANT = "CropOrWeed2"              # labels 0 crop, 1 weed
LABELS = {0: "crop", 1: "weed"}
PARAMETERS = ("moisture", "soil", "lighting", "separability")
WIDTH, HEIGHT = 1920, 1088           # every image in the dataset


def session_of(name: str) -> str:
    """The field visit an image belongs to, for example 'vwg-0328'."""
    return name[:8]


def image_names(root: Path = RAW) -> List[str]:
    """Every image that has plant boxes in the variant."""
    return sorted(p.stem for p in (root / "bboxes" / VARIANT).glob("*.csv"))


def image_path(name: str, root: Path = RAW) -> Path:
    return root / "images" / f"{name}.jpg"


def load_boxes(name: str, root: Path = RAW) -> List[Dict]:
    """The labelled plants of one image: box, label and stem point."""
    rows = []
    with open(root / "bboxes" / VARIANT / f"{name}.csv", newline="") as f:
        for r in csv.reader(f):
            if r:
                left, top, right, bottom, label, sx, sy = (int(float(v)) for v in r[:7])
                rows.append(dict(left=left, top=top, right=right, bottom=bottom,
                                 label=LABELS.get(label, "other"), stem_x=sx, stem_y=sy))
    return rows


def load_parameters(name: str, root: Path = RAW) -> Dict[str, int]:
    """Moisture, soil, lighting and separability of one image, or {} when missing."""
    path = root / "params" / f"{name}.csv"
    if not path.exists():
        return {}
    text = path.read_text()
    named = dict(re.findall(r"(moisture|soil|lighting|separability)\D+(\d)", text))
    if len(named) == 4:
        return {k: int(v) for k, v in named.items()}
    values = re.findall(r"-?\d+", text)
    return dict(zip(PARAMETERS, map(int, values[:4]))) if len(values) >= 4 else {}
