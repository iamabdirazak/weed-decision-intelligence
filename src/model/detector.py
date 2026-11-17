"""Detection: every plant in a frame, with both its crop and its weed confidence.

YOLO11n, fine-tuned on crops and weeds and run through ONNX Runtime on the
CPU. Two choices differ from the usual setup. Detections are kept down to a
confidence of 0.01 instead of 0.25, so faint plants reach the decision, and
each detection keeps both class confidences instead of only the winner.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

FAINT = 0.01        # lowest confidence kept; the usual default is 0.25


class Detector:
    def __init__(self, path: Path, size: int = 640, threads: int = 4, min_confidence: float = FAINT,
                 overlap: float = 0.5):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size, self.min_confidence, self.overlap = size, min_confidence, overlap

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.size / w, self.size / h)
        nw, nh = round(w * r), round(h * r)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, dx, dy

    def __call__(self, frame_bgr: np.ndarray):
        """Boxes (n, 4) as left, top, right, bottom in frame pixels, and confidences (n, 2) as crop, weed."""
        image, r, dx, dy = self._letterbox(frame_bgr)
        x = image[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input: np.ascontiguousarray(x)})[0][0].T   # (anchors, 4 + classes)
        confidence = out[:, 4:]
        keep = confidence.max(1) >= self.min_confidence
        box, confidence = out[keep, :4], confidence[keep]
        if not len(box):
            return np.zeros((0, 4)), np.zeros((0, 2))
        cx, cy, w, h = box.T
        xywh = np.column_stack([cx - w / 2, cy - h / 2, w, h])
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), confidence.max(1).tolist(), self.min_confidence,
                                        self.overlap)).reshape(-1)
        xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])[idx]
        return (xyxy - [dx, dy, dx, dy]) / r, confidence[idx]
