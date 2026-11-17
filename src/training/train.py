"""Fine-tunes YOLO11n on crops and weeds, then exports it to ONNX.

Starts from the published YOLO11n weights and trains for 15 epochs at
640 px, the input size a Raspberry Pi 5 can run, on the clean and damaged
training photos. The exported file is the detector in models/.

    python -m src.training.train                 # train, then export
    python -m src.training.train --device cpu    # mps on Apple silicon is the default
"""
import argparse
import shutil

from src.paths import MODELS, PROCESSED
from src.preparation.splitting import SEED

BASE, SIZE, EPOCHS = "yolo11n.pt", 640, 15


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    runs = PROCESSED / "training"
    YOLO(BASE).train(data=str(PROCESSED / "detector" / "data.yaml"), imgsz=SIZE, epochs=args.epochs,
                     patience=args.epochs, batch=16, device=args.device, project=str(runs), name="detector",
                     exist_ok=True, seed=SEED, workers=4)
    exported = YOLO(str(runs / "detector" / "weights" / "best.pt")).export(format="onnx", imgsz=SIZE)
    MODELS.mkdir(exist_ok=True)
    shutil.copy(exported, MODELS / "detector.onnx")
    print(f"Detector written to {MODELS / 'detector.onnx'}")


if __name__ == "__main__":
    main()
