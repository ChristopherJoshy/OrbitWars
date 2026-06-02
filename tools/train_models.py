from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ow.features import FEATURE_COLUMNS


def _validate_training_inputs(x, y, epochs: int, lr: float) -> tuple[np.ndarray, np.ndarray, float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError("x must be 2D")
    if x_arr.shape[0] == 0:
        raise ValueError("x must not be empty")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be finite")
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D")
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("x and y length mismatch")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must be finite")
    unique = np.unique(y_arr)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError("y must be binary 0/1")
    if unique.size != 2:
        raise ValueError("y must contain both classes 0 and 1")
    if not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be nonnegative int")
    lr_f = float(lr)
    if not np.isfinite(lr_f) or lr_f <= 0:
        raise ValueError("lr must be finite positive")
    return x_arr, y_arr, lr_f


def train_logistic_numpy(x, y, epochs: int = 50, lr: float = 0.05) -> tuple[np.ndarray, float]:
    x_arr, y_arr, lr_f = _validate_training_inputs(x, y, epochs=epochs, lr=lr)
    n_samples, n_features = x_arr.shape
    w = np.zeros((n_features,), dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        logits = np.clip(x_arr @ w + b, -60.0, 60.0)
        if not np.all(np.isfinite(logits)):
            raise ValueError("non-finite logits during training")
        probs = 1.0 / (1.0 + np.exp(-logits))
        err = probs - y_arr
        try:
            with np.errstate(over="raise", invalid="raise"):
                grad_w = (x_arr.T @ err) / n_samples
                grad_b = float(np.mean(err))
        except FloatingPointError as exc:
            raise ValueError("non-finite gradient during training") from exc
        if not np.all(np.isfinite(grad_w)) or not np.isfinite(grad_b):
            raise ValueError("non-finite gradient during training")
        try:
            with np.errstate(over="raise", invalid="raise"):
                update_w = lr_f * grad_w
                update_b = lr_f * grad_b
        except FloatingPointError as exc:
            raise ValueError("non-finite update during training") from exc
        if not np.all(np.isfinite(update_w)) or not np.isfinite(update_b):
            raise ValueError("non-finite update during training")
        w -= update_w
        b -= update_b
        if not np.all(np.isfinite(w)) or not np.isfinite(b):
            raise ValueError("non-finite parameters during training")
    return w, float(b)


def save_npz_model(path, weights, bias, feature_columns=None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    b = float(bias)
    if not np.isfinite(b):
        raise ValueError("bias must be finite scalar")
    bias_arr = np.asarray(b, dtype=np.float64)
    feature_columns_arr = None
    if feature_columns is not None:
        cols = list(feature_columns)
        if len(cols) != w.shape[0]:
            raise ValueError("feature_columns length mismatch")
        feature_columns_arr = np.asarray(cols, dtype=str)
    tmp = out.with_name(f".{out.stem}.{uuid4().hex}.tmp.npz")
    try:
        if feature_columns_arr is None:
            np.savez(tmp, weights=w, bias=bias_arr)
        else:
            np.savez(tmp, weights=w, bias=bias_arr, feature_columns=feature_columns_arr)
        os.replace(tmp, out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NumPy logistic replay model")
    parser.add_argument("--features", type=Path, default=Path("data/processed/shot_features.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/shot_validator.npz"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=False) as data:
        if "x" not in data or "y" not in data:
            raise ValueError("features npz must contain x and y")
        x = data["x"]
        y = data["y"]
        cols = data["feature_columns"].tolist() if "feature_columns" in data else FEATURE_COLUMNS
    w, b = train_logistic_numpy(x, y, epochs=args.epochs, lr=args.lr)
    save_npz_model(args.output, w, b, feature_columns=cols)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "rows": int(x.shape[0]),
            "features": int(x.shape[1]),
            "positives": int(np.sum(y == 1)),
            "negatives": int(np.sum(y == 0)),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "output": str(args.output),
            "weight_norm": float(np.linalg.norm(w)),
            "bias": float(b),
        }
        tmp = args.summary_output.with_name(f".{args.summary_output.name}.{uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            os.replace(tmp, args.summary_output)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    main()
