from __future__ import annotations

import math

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

    planets = state.planets
    best = None
    for src_row in [int(i) for i in state.my_planet_mask.nonzero()[0]]:
        if not _row_is_finite_for_action(planets[src_row]):
            continue
        sx, sy = planets[src_row, PLANET_COL_X], planets[src_row, PLANET_COL_Y]
        can = int(max(0, math.floor(float(planets[src_row, PLANET_COL_SHIPS]))))
        if can < 1:
            continue
        for tr in range(planets.shape[0]):
            if not _row_is_finite_for_action(planets[tr]):
                continue
            if tr == src_row or int(planets[tr, PLANET_COL_OWNER]) == state.player:
                continue
            tx, ty = planets[tr, PLANET_COL_X], planets[tr, PLANET_COL_Y]
            if segment_hits_circle([sx, sy], [tx, ty], [CENTER_X, CENTER_Y], SUN_RADIUS):
                continue
            dist = (float(tx - sx) ** 2 + float(ty - sy) ** 2) ** 0.5
            need = int(max(1, math.floor(float(planets[tr, PLANET_COL_SHIPS])) + 1))
            if can < need:
                continue
            cand = (dist, src_row, tr, need)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        return []
    _d, src_row, tr, ships = best
    sx, sy = planets[src_row, PLANET_COL_X], planets[src_row, PLANET_COL_Y]
    tx, ty = planets[tr, PLANET_COL_X], planets[tr, PLANET_COL_Y]
    src_id = int(planets[src_row, PLANET_COL_ID])
    return [[float(src_id), float(angle_between(float(sx), float(sy), float(tx), float(ty))), float(ships)]]
