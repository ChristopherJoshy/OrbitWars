import math

import numpy as np

from src.ow.constants import FLEET_COL_ANGLE, FLEET_COL_OWNER, FLEET_COL_SHIPS, FLEET_COL_X, FLEET_COL_Y, PLANET_CLEARANCE, PLANET_COL_OWNER, PLANET_COL_PROD, PLANET_COL_RADIUS, PLANET_COL_SHIPS, PLANET_COL_X, PLANET_COL_Y, ROW_WIDTH
from src.ow.geometry import fleet_speed, predict_planet_position, segment_hits_circle


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def estimate_required_capture_ships(target_ships, target_prod, eta, margin: float = 2) -> int:
    target_ships = _finite_nonnegative(target_ships, "target_ships")
    target_prod = _finite_nonnegative(target_prod, "target_prod")
    eta = _finite_nonnegative(eta, "eta")
    margin = _finite_nonnegative(margin, "margin")

    eta_turns = int(math.ceil(eta))
    defenders = target_ships + target_prod * eta_turns
    return int(math.ceil(defenders + margin + 1.0))


def estimate_required_capture_ships_fixed_point(target_ships, target_prod, distance, margin: float = 2, max_iter=6) -> int:
    target_ships = _finite_nonnegative(target_ships, "target_ships")
    target_prod = _finite_nonnegative(target_prod, "target_prod")
    distance = _finite_nonnegative(distance, "distance")
    margin = _finite_nonnegative(margin, "margin")
    _ = max_iter  # Kept for backwards-compatible call sites; binary search is deterministic.

    def required_for(ship_count: int) -> int:
        eta = distance / fleet_speed(max(1, int(ship_count)))
        return estimate_required_capture_ships(target_ships, target_prod, eta, margin=margin)

    low = 1
    upper = max(1, required_for(1))
    while low < upper:
        mid = (low + upper) // 2
        if mid >= required_for(mid):
            upper = mid
        else:
            low = mid + 1
    return low


def keep_needed_by_planet(planets, player, horizon) -> np.ndarray:
    horizon = _finite_nonnegative(horizon, "horizon")
    try:
        arr = np.asarray(planets, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("planets schema invalid: expected numeric array with width ROW_WIDTH") from exc
    if arr.ndim != 2:
        raise ValueError(f"planets schema invalid: expected 2D array with width {ROW_WIDTH}")
    if arr.shape[1] != ROW_WIDTH:
        raise ValueError(f"planets schema invalid: expected width {ROW_WIDTH}, got {arr.shape[1]}")
    if arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    ships = arr[:, PLANET_COL_SHIPS]
    prods = arr[:, PLANET_COL_PROD]
    if not np.all(np.isfinite(ships)) or np.any(ships < 0.0):
        raise ValueError("planets ships must be finite and nonnegative")
    if not np.all(np.isfinite(prods)) or np.any(prods < 0.0):
        raise ValueError("planets prod must be finite and nonnegative")

    owners = arr[:, PLANET_COL_OWNER].astype(np.int64, copy=False)

    reserves = np.zeros(arr.shape[0], dtype=np.int64)
    owned = owners == int(player)
    if np.any(owned):
        available = np.floor(ships[owned]).astype(np.int64)
        # bounded opening reserve: scales with production but never consumes all ships
        horizon_factor = min(1.0, float(horizon) / 500.0)
        base = np.floor(prods[owned] * 0.5 * horizon_factor).astype(np.int64)
        base = np.maximum(0, base)
        cap = np.maximum(0, available - 1)
        reserves[owned] = np.minimum(base, cap)
    return reserves


def threat_keep_needed_by_planet(state, horizon, threat_lookahead: float = 28.0, safety: float = 1.12) -> np.ndarray:
    reserves = keep_needed_by_planet(state.planets, player=state.player, horizon=horizon)
    fleets = np.asarray(state.fleets, dtype=np.float64)
    if fleets.ndim != 2 or fleets.shape[0] == 0:
        return reserves

    planets = np.asarray(state.planets, dtype=np.float64)
    owned_idx = np.flatnonzero(state.my_planet_mask)
    if owned_idx.size == 0:
        return reserves

    sample_step = 1.0
    for fleet in fleets:
        owner = fleet[FLEET_COL_OWNER]
        ships = fleet[FLEET_COL_SHIPS]
        x0 = fleet[FLEET_COL_X]
        y0 = fleet[FLEET_COL_Y]
        angle = fleet[FLEET_COL_ANGLE]
        if (not np.isfinite(owner)) or owner < 0.0 or int(owner) == int(state.player):
            continue
        if (not np.isfinite(ships)) or ships <= 0.0:
            continue
        if (not np.isfinite(x0)) or (not np.isfinite(y0)) or (not np.isfinite(angle)):
            continue

        dist = fleet_speed(float(ships)) * float(threat_lookahead)
        x1 = float(x0) + dist * math.cos(float(angle))
        y1 = float(y0) + dist * math.sin(float(angle))
        reserve_delta = int(math.ceil(float(ships) * float(safety))) + 1

        speed = fleet_speed(float(ships))
        for pidx in owned_idx.tolist():
            pr = planets[pidx, PLANET_COL_RADIUS]
            pships = planets[pidx, PLANET_COL_SHIPS]
            px = planets[pidx, PLANET_COL_X]
            py = planets[pidx, PLANET_COL_Y]
            if (not np.isfinite(pr)) or (not np.isfinite(pships)):
                continue
            radius = float(pr) + PLANET_CLEARANCE

            threat_hit = False
            impact_t = None
            for t in np.arange(0.0, float(threat_lookahead) + sample_step, sample_step):
                fx = float(x0) + speed * float(t) * math.cos(float(angle))
                fy = float(y0) + speed * float(t) * math.sin(float(angle))
                px_t, py_t = predict_planet_position(planets[pidx], turns=float(t), angular_velocity=state.angular_velocity)
                if (fx - px_t) * (fx - px_t) + (fy - py_t) * (fy - py_t) <= radius * radius:
                    threat_hit = True
                    impact_t = float(t)
                    break
                if t <= 0.0:
                    continue
                t_prev = float(t - sample_step)
                fx_prev = float(x0) + speed * t_prev * math.cos(float(angle))
                fy_prev = float(y0) + speed * t_prev * math.sin(float(angle))
                if segment_hits_circle([fx_prev, fy_prev], [fx, fy], [px_t, py_t], radius):
                    threat_hit = True
                    impact_t = float(t)
                    break

            if not threat_hit:
                if (not np.isfinite(px)) or (not np.isfinite(py)):
                    continue
                threat_hit = segment_hits_circle([x0, y0], [x1, y1], [px, py], radius)
            if not threat_hit:
                continue

            if impact_t is None:
                dx = float(px) - float(x0)
                dy = float(py) - float(y0)
                impact_t = min(float(threat_lookahead), math.hypot(dx, dy) / max(speed, 1e-9))

            pprod = planets[pidx, PLANET_COL_PROD]
            if not np.isfinite(pprod):
                continue
            plausible_defenders = int(math.floor(float(pships))) + int(math.floor(float(pprod) * float(math.ceil(impact_t))))
            if plausible_defenders < reserve_delta:
                continue

            cap = max(0, int(np.floor(pships)) - 1)
            reserves[pidx] = min(cap, reserves[pidx] + reserve_delta)

    return reserves
