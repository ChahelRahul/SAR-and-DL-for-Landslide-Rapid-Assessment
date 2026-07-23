from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from app.config import InferenceConfig
from app.inference.windows import Window, non_max_suppression, sliding_windows
from app.preprocessing.sentinel1 import validate_stack

@dataclass(slots=True)
class Prediction:
    scores: np.ndarray
    candidate_windows: list[Window]
    selected_windows: list[Window]
    mask: np.ndarray


def predict_stack(model, stack: np.ndarray, config: InferenceConfig = InferenceConfig()) -> Prediction:
    stack = validate_stack(stack)
    extracted = list(sliding_windows(stack, size=64, step=config.window_step))
    if not extracted:
        return Prediction(np.empty(0), [], [], np.zeros(stack.shape[:2], dtype=np.uint8))
    windows, patches = zip(*extracted)
    scores = np.asarray(model.predict(np.asarray(patches, dtype=np.float32), batch_size=config.batch_size, verbose=0)).reshape(-1)
    candidates = [window for window, score in zip(windows, scores) if score > config.probability_threshold]
    boxes = np.asarray([[w.x0,w.y0,w.x1,w.y1] for w in candidates], dtype=np.int32)
    kept = non_max_suppression(boxes, config.nms_overlap)
    selected = [Window(*map(int, box)) for box in kept]
    mask = np.zeros(stack.shape[:2], dtype=np.uint8)
    for window in selected:
        mask[window.y0:window.y1, window.x0:window.x1] = 1
    return Prediction(scores, candidates, selected, mask)
