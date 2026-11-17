"""The field conditions the model is calibrated for, applied to the whole photo.

Each condition copies something a moving sprayer meets. It is applied at the
camera's full resolution before detection, so the detector sees the same
damage it would see in the field.

    blur     the robot moves while the shutter is open, so the ground slides
             down the image and the blur runs vertically (9, 17, 31 px)
    dark     short exposure under cloud or at dusk: less signal, more noise
    bright   overexposure in direct sun, highlights clipped to white
"""
import cv2
import numpy as np

CONDITIONS = {
    "clean": None,
    "blur_9": ("blur", 9),
    "blur_17": ("blur", 17),
    "blur_31": ("blur", 31),
    "dark_50": ("dark", 0.50),
    "dark_25": ("dark", 0.25),
    "bright_180": ("bright", 1.80),
}

NAMES = {"clean": "Clean", "blur_9": "Motion blur, 9 px", "blur_17": "Motion blur, 17 px",
         "blur_31": "Motion blur, 31 px", "dark_50": "Low light, 50%", "dark_25": "Low light, 25%",
         "bright_180": "Glare"}


def motion_blur(frame: np.ndarray, length: int) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    kernel[:, length // 2] = 1.0 / length
    return cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REFLECT)


def dark(frame: np.ndarray, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Lower exposure, then sensor noise that grows as the gain drops."""
    x = frame.astype(np.float32) * gain
    noise = rng.normal(0.0, 2.0 + 6.0 * (1.0 - gain), x.shape)
    return np.clip(x + noise, 0, 255).astype(np.uint8)


def bright(frame: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def apply(frame: np.ndarray, condition: str, seed: int = 0) -> np.ndarray:
    spec = CONDITIONS[condition]
    if spec is None:
        return frame
    kind, level = spec
    if kind == "blur":
        return motion_blur(frame, level)
    if kind == "dark":
        return dark(frame, level, np.random.default_rng(seed))
    return bright(frame, level)
