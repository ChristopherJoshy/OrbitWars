from __future__ import annotations

from pathlib import Path

from src.ow.features import FEATURE_COLUMNS
from src.ow.models_numpy import LinearModel, load_linear_model
from src.ow.policy import PolicyConfig, choose_moves
from src.ow.state import parse_observation

_MODEL_CACHE: dict[str, LinearModel] = {}


def _get_model(path: str | None = None) -> LinearModel:
    p = str(path or Path("models") / "shot_validator.npz")
    if p not in _MODEL_CACHE:
        _MODEL_CACHE[p] = load_linear_model(p, n_features=len(FEATURE_COLUMNS), expected_feature_columns=FEATURE_COLUMNS)
    return _MODEL_CACHE[p]


def agent_impl(obs, config=None):
    state = parse_observation(obs)
    if config is None:
        cfg = PolicyConfig()
    else:
        cfg = PolicyConfig(
            max_moves=int(getattr(config, "max_moves", PolicyConfig().max_moves)),
            time_budget_ms=float(getattr(config, "time_budget_ms", PolicyConfig().time_budget_ms)),
        )
    model = _get_model(getattr(config, "model_path", None) if config is not None else None)
    return choose_moves(state, model=model, config=cfg)


def safe_agent(obs, config=None):
    try:
        return agent_impl(obs, config=config)
    except Exception:
        return []
