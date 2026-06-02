import numpy as np
import pytest

from src.ow.models_numpy import load_linear_model
from tools.train_models import main, save_npz_model, train_logistic_numpy


def test_train_and_save_model_roundtrip_predicts_higher_positive(tmp_path):
    x = np.array([[-2.0, -1.0], [-1.5, -0.5], [1.0, 1.0], [2.0, 1.5]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    w, b = train_logistic_numpy(x, y, epochs=200, lr=0.1)
    model_path = tmp_path / "model.npz"
    save_npz_model(model_path, w, b)

    model = load_linear_model(model_path, n_features=2)
    probs = model.predict_proba(x)
    assert probs[2] > probs[0]


def test_train_logistic_numpy_rejects_invalid_inputs():
    x = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    import pytest

    with pytest.raises(ValueError):
        train_logistic_numpy(np.array([], dtype=np.float64).reshape(0, 2), y[:0])
    with pytest.raises(ValueError):
        train_logistic_numpy(np.array([1.0, 2.0], dtype=np.float64), y)
    with pytest.raises(ValueError):
        train_logistic_numpy(x, np.array([[0.0], [1.0]], dtype=np.float64))
    with pytest.raises(ValueError):
        train_logistic_numpy(x, np.array([0.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError):
        train_logistic_numpy(x, y, epochs=-1)
    with pytest.raises(ValueError):
        train_logistic_numpy(x, y, lr=0.0)
    with pytest.raises(ValueError):
        train_logistic_numpy(x, np.array([0.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError):
        train_logistic_numpy(x, np.array([1.0, 1.0], dtype=np.float64))


def test_train_logistic_numpy_raises_on_divergent_nonfinite_updates():
    import warnings

    import pytest

    x = np.array([[1e300, -1e300], [1e300, 1e300]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            train_logistic_numpy(x, y, epochs=2, lr=1e100)


def test_save_npz_model_with_feature_columns(tmp_path):
    path = tmp_path / "m.npz"
    save_npz_model(path, np.array([1.0, 2.0]), 0.5, feature_columns=["a", "b"])
    model = load_linear_model(path, n_features=2, expected_feature_columns=["a", "b"])
    assert model.weights.shape == (2,)


def test_save_npz_model_rejects_invalid_schema_before_writing(tmp_path):
    import pytest

    path = tmp_path / "m.npz"
    with pytest.raises(ValueError):
        save_npz_model(path, np.array([[1.0, 2.0]]), 0.5)
    with pytest.raises(ValueError):
        save_npz_model(path, np.array([1.0, np.inf]), 0.5)
    with pytest.raises(ValueError):
        save_npz_model(path, np.array([1.0, 2.0]), np.nan)
    with pytest.raises(ValueError):
        save_npz_model(path, np.array([1.0, 2.0]), 0.5, feature_columns=["a"])
    assert not path.exists()


def test_save_npz_model_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    import os
    import pytest

    path = tmp_path / "m.npz"
    save_npz_model(path, np.array([1.0, 2.0]), 0.5)
    before = path.read_bytes()

    def fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        save_npz_model(path, np.array([3.0, 4.0]), 0.9)
    after = path.read_bytes()
    assert after == before


def test_save_npz_model_replace_failure_cleans_tmp_file(tmp_path, monkeypatch):
    import os

    path = tmp_path / "m.npz"
    created_tmp = {}
    real_replace = os.replace

    def capture_replace(src, dst):
        if str(dst) == str(path):
            created_tmp["path"] = src
            raise OSError("replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", capture_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_npz_model(path, np.array([1.0, 2.0]), 0.5)
    assert "path" in created_tmp
    assert not created_tmp["path"].exists()


def test_main_writes_summary_output(tmp_path, monkeypatch):
    feat = tmp_path / "f.npz"
    np.savez(feat, x=np.array([[0.0], [1.0]]), y=np.array([0, 1]), feature_columns=np.array(["f0"]))
    out = tmp_path / "m.npz"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", ["train_models.py", "--features", str(feat), "--output", str(out), "--summary-output", str(summary), "--epochs", "1"])
    main()
    payload = __import__("json").loads(summary.read_text(encoding="utf-8"))
    assert payload["rows"] == 2
    assert payload["features"] == 1


def test_main_rejects_missing_x_or_y(tmp_path, monkeypatch):
    feat = tmp_path / "f.npz"
    np.savez(feat, x=np.array([[0.0], [1.0]]))
    out = tmp_path / "m.npz"
    monkeypatch.setattr("sys.argv", ["train_models.py", "--features", str(feat), "--output", str(out)])
    with pytest.raises(ValueError, match="x and y"):
        main()
