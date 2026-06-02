from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LinearModel:
    weights: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        w = np.array(self.weights, dtype=np.float64, copy=True)
        w.setflags(write=False)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "bias", float(self.bias))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("x must be 2D")
        if arr.shape[1] != self.weights.shape[0]:
            raise ValueError("x feature width mismatch")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must contain only finite values")
        if arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        logits = arr @ self.weights + float(self.bias)
        logits = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))


def load_linear_model(path, n_features: int, expected_feature_columns: list[str] | tuple[str, ...] | None = None) -> LinearModel:
    model_path = Path(path)
    if not model_path.exists():
        return LinearModel(weights=np.zeros((n_features,), dtype=np.float64), bias=0.0)

    with np.load(model_path, allow_pickle=False) as data:
        if "weights" not in data or "bias" not in data:
            raise ValueError("model file missing required keys")
        if expected_feature_columns is not None and "feature_columns" not in data:
            raise ValueError("model file missing feature_columns")

        w = np.asarray(data["weights"], dtype=np.float64)
        if w.shape != (int(n_features),):
            raise ValueError("weights shape mismatch")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")

        b_arr = np.asarray(data["bias"], dtype=np.float64)
        if b_arr.ndim != 0:
            raise ValueError("bias must be a scalar")
        b = float(b_arr)
        if not np.isfinite(b):
            raise ValueError("bias must be finite")

        if expected_feature_columns is not None:
            cols = np.asarray(data["feature_columns"])
            loaded = [str(x) for x in cols.tolist()]
            expected = [str(x) for x in expected_feature_columns]
            if loaded != expected:
                raise ValueError("feature schema mismatch")
    return LinearModel(weights=w, bias=b)
