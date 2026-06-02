from __future__ import annotations

import math

import numpy as np

from baselines.greedy_sample import agent as greedy_agent
from baselines.random_agent import agent as random_agent
from src.ow.constants import PLANET_COL_ID, PLANET_COL_SHIPS
from src.ow.state import parse_observation


def rush_agent(obs, config=None):
    try:
        state = parse_observation(obs)
    except Exception:
        return []
    moves = []
    first = greedy_agent(obs, config=config)
    if first:
        moves.extend(first[:1])
    second = random_agent(obs, config={"seed": int((config or {}).get("seed", 17)) + 1})
    for mv in second:
        if len(moves) >= 2:
            break
        if int(mv[0]) != int(moves[0][0]) if moves else True:
            moves.append(mv)
    return moves[:2]


def turtle_agent(obs, config=None):
    cfg = config or {}
    delay = int(cfg.get("delay", 25))
    try:
        state = parse_observation(obs)
    except Exception:
        return []
    if state.step < delay:
        return []
    moves = greedy_agent(obs, config=config)
    return moves[:1]


def greedy_expand_agent(obs, config=None):
    return greedy_agent(obs, config=config)


def swarm_agent(obs, config=None):
    try:
        state = parse_observation(obs)
    except Exception:
        return []
    cfg = config or {}
    seed = int(cfg.get("seed", 9))
    rnd = random_agent(obs, config={"seed": seed})
    # cap to three and enforce aggregate spend per source
    ships_by_src = {}
    id_to_ships = {}
    for row in state.planets:
        pid = float(row[PLANET_COL_ID])
        ships = float(row[PLANET_COL_SHIPS])
        if not np.isfinite(pid) or not np.isfinite(ships):
            continue
        id_to_ships[int(pid)] = int(max(0, math.floor(ships)))
    out = []
    for src, ang, ships in rnd:
        s = int(src)
        n = int(ships)
        if ships_by_src.get(s, 0) + n > id_to_ships.get(s, 0):
            continue
        ships_by_src[s] = ships_by_src.get(s, 0) + n
        out.append([src, ang, float(n)])
        if len(out) >= 3:
            break
    return out


def idle_heavy_agent(obs, config=None):
    if int((config or {}).get("wake_mod", 4)) <= 1:
        return turtle_agent(obs, config=config)
    try:
        state = parse_observation(obs)
    except Exception:
        return []
    if state.step % int((config or {}).get("wake_mod", 4)) != 0:
        return []
    return turtle_agent(obs, config=config)
