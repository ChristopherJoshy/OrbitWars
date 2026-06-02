from dataclasses import dataclass

import numpy as np

from src.ow.constants import FLOAT_DTYPE, PLANET_COL_OWNER, ROW_WIDTH


@dataclass(frozen=True)
class GameState:
    player: int
    step: int
    angular_velocity: float
    planets: np.ndarray
    fleets: np.ndarray
    comets: np.ndarray
    initial_planets: np.ndarray
    my_planet_mask: np.ndarray
    enemy_planet_mask: np.ndarray
    neutral_planet_mask: np.ndarray


def _value(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _array(data, field: str) -> np.ndarray:
    if data is None:
        return np.empty((0, ROW_WIDTH), dtype=FLOAT_DTYPE)
    if isinstance(data, list) and len(data) == 0:
        return np.empty((0, ROW_WIDTH), dtype=FLOAT_DTYPE)
    try:
        arr = np.asarray(data, dtype=FLOAT_DTYPE)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be numeric rows of width {ROW_WIDTH}") from exc
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError(f"{field}: expected width {ROW_WIDTH}, got 0")
        if arr.shape[0] != ROW_WIDTH:
            raise ValueError(f"{field}: expected width {ROW_WIDTH}, got {arr.shape[0]}")
        return arr.reshape(1, ROW_WIDTH)
    if arr.ndim != 2:
        raise ValueError(f"{field}: expected 2D array-like, got ndim={arr.ndim}")
    if arr.shape[1] != ROW_WIDTH:
        raise ValueError(f"{field}: expected width {ROW_WIDTH}, got {arr.shape[1]}")
    return arr.astype(FLOAT_DTYPE, copy=False)


def _comets_array(data) -> np.ndarray:
    try:
        return _array(data, "comets")
    except ValueError:
        return np.empty((0, ROW_WIDTH), dtype=FLOAT_DTYPE)


def parse_observation(obs) -> GameState:
    player = int(_value(obs, "player", 0))
    step_value = _value(obs, "step", None)
    if step_value is None:
        step_value = _value(obs, "turn", 0)
    step = int(step_value)
    angular_velocity = float(_value(obs, "angular_velocity", 0.0))

    planets = _array(_value(obs, "planets", []), "planets")
    fleets = _array(_value(obs, "fleets", []), "fleets")
    comets = _comets_array(_value(obs, "comets", []))
    initial_planets = _array(_value(obs, "initial_planets", planets), "initial_planets")

    owners = planets[:, PLANET_COL_OWNER]
    finite_owners = np.isfinite(owners)
    my_planet_mask = finite_owners & (owners == player)
    enemy_planet_mask = finite_owners & (owners >= 0) & (owners != player)
    neutral_planet_mask = finite_owners & (owners < 0)

    return GameState(
        player=player,
        step=step,
        angular_velocity=angular_velocity,
        planets=planets,
        fleets=fleets,
        comets=comets,
        initial_planets=initial_planets,
        my_planet_mask=my_planet_mask,
        enemy_planet_mask=enemy_planet_mask,
        neutral_planet_mask=neutral_planet_mask,
    )
