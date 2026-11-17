"""Time per frame of the detector, and an estimate for a Raspberry Pi 5.

Measured with ONNX Runtime limited to 4 threads, the core count of a Pi 5.
The stock YOLO11n is timed the same way, and Ultralytics' published Pi 5
time for that model (6.79 frames per second with ONNX) scales the
measurement to a Pi 5 estimate. Calibration adds one table lookup and one
comparison per plant, which is negligible next to detection.

    python -m src.evaluation.timing
"""
import platform
import time

import numpy as np
import pandas as pd

from src.model.detector import Detector
from src.paths import MODELS, REPORTS

PI5_STOCK_MS = 1000 / 6.79


def clock(fn, repeat=50):
    fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times))


def main():
    from ultralytics import YOLO
    frame = np.random.default_rng(0).integers(0, 255, (1088, 1920, 3), np.uint8)
    stock = YOLO("yolo11n.pt").export(format="onnx", imgsz=640)
    models = {"stock YOLO11n": Detector(stock), "detector": Detector(MODELS / "detector.onnx")}
    ms = {name: clock(lambda: model(frame)) for name, model in models.items()}
    scale = PI5_STOCK_MS / ms["stock YOLO11n"]
    rows = [dict(model=k, milliseconds=v, pi5_estimate=v * scale, machine=platform.machine()) for k, v in ms.items()]
    pd.DataFrame(rows).to_csv(REPORTS / "timing.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
