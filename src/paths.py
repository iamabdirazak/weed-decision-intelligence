"""Where everything lives. Paths are relative to the repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "cropandweed"          # the dataset, mapped to CropOrWeed2
PROCESSED = ROOT / "data" / "processed"      # plant index, detector training set and scores
MODELS = ROOT / "models"                     # the released detector, thresholds and model card
REPORTS = ROOT / "reports"                   # evaluation results
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"
