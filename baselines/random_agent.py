from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from src.ow.constants import (
    CENTER_X,
    CENTER_Y,
    PLANET_COL_ID,
    PLANET_COL_OWNER,
    PLANET_COL_SHIPS,
    PLANET_COL_X,
    PLANET_COL_Y,
    SUN_RADIUS,
)
from src.ow.geometry import angle_between, segment_hits_circle
from src.ow.state import parse_observation


def _legal_targets(planets: np.ndarray, src_row: int, player: int) -> Iterable[int]:
    sx, sy = planets[src_row, PLANET_COL_X], planets[src_row, PLANET_COL_Y]
    for tr in range(planets.shape[0]):
        if tr == src_row:
            continue
        if not _row_is_finite_for_action(planets[tr]):
            continue
        if int(planets[tr, PLANET_COL_OWNER]) == player:
            continue
        tx, ty = planets[tr, PLANET_COL_X], planets[tr, PLANET_COL_Y]
        if segment_hits_circle([sx, sy], [tx, ty], [CENTER_X, CENTER_Y], SUN_RADIUS):
            continue
        yield tr


def _row_is_finite_for_action(row: np.ndarray) -> bool:
    return bool(
        np.isfinite(float(row[PLANET_COL_ID]))
        and np.isfinite(float(row[PLANET_COL_OWNER]))
        and np.isfinite(float(row[PLANET_COL_X]))
        and np.isfinite(float(row[PLANET_COL_Y]))
        and np.isfinite(float(row[PLANET_COL_SHIPS]))
    )


def agent(obs, config=None):
    try:
        state = parse_observation(obs)
    except Exception:
        return []
    cfg = config or {}
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    out = []
    available = {}
    for src_row in np.flatnonzero(state.my_planet_mask):
        if not _row_is_finite_for_action(state.planets[src_row]):
            continue
        src_id = int(state.planets[src_row, PLANET_COL_ID])
        available[src_id] = int(max(0, math.floor(float(state.planets[src_row, PLANET_COL_SHIPS]))))
    for src_row in np.flatnonzero(state.my_planet_mask):
        if not _row_is_finite_for_action(state.planets[src_row]):
            continue
        src_id = int(state.planets[src_row, PLANET_COL_ID])
        can = available.get(src_id, 0)
        if can <= 0 or rng.random() > 0.75:
            continue
        targets = list(_legal_targets(state.planets, int(src_row), state.player))
        if not targets:
            continue
        tr = int(rng.choice(np.asarray(targets, dtype=np.int64)))
        ships = int(rng.integers(1, can + 1))
        sx, sy = state.planets[src_row, PLANET_COL_X], state.planets[src_row, PLANET_COL_Y]
        tx, ty = state.planets[tr, PLANET_COL_X], state.planets[tr, PLANET_COL_Y]
        out.append([float(src_id), float(angle_between(float(sx), float(sy), float(tx), float(ty))), float(ships)])
        available[src_id] = can - ships
    return out
