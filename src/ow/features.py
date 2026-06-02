import math

import numpy as np

from src.ow.constants import (
    CENTER_X,
    CENTER_Y,
    EPISODE_STEPS,
    PLANET_COL_ID,
    PLANET_COL_OWNER,
    PLANET_COL_PROD,
    PLANET_COL_SHIPS,
    PLANET_COL_X,
    PLANET_COL_Y,
    SUN_RADIUS,
    FLEET_COL_OWNER,
    FLEET_COL_SHIPS,
)
from src.ow.geometry import distance_matrix, fleet_speed, segment_hits_circle
from src.ow.state import GameState
from src.ow.timeline import estimate_required_capture_ships, threat_keep_needed_by_planet

FEATURE_COLUMNS = [
    "turn_norm",
    "turns_left_norm",
    "src_ships",
    "src_prod",
    "target_ships",
    "target_prod",
    "target_owner_rel",
    "distance",
    "eta",
    "required",
    "surplus_after_send",
    "sun_blocked",
    "target_is_neutral",
    "target_is_enemy",
    "source_keep",
    "score_pressure",
]


def build_candidate_features(state: GameState, *, keeps: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    planets = state.planets
    my_idx = np.flatnonzero(state.my_planet_mask)
    target_idx = np.flatnonzero(~state.my_planet_mask)
    if my_idx.size == 0 or target_idx.size == 0:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float64), np.zeros((0, 2), dtype=np.int64)

    coords = planets[:, [PLANET_COL_X, PLANET_COL_Y]]
    dmat = distance_matrix(coords[my_idx], coords[target_idx])
    if keeps is None:
        keeps = threat_keep_needed_by_planet(state, horizon=max(0, EPISODE_STEPS - state.step))
    else:
        keeps = np.asarray(keeps, dtype=np.float64)
        if keeps.ndim != 1 or keeps.shape[0] != planets.shape[0]:
            raise ValueError("keeps must be a 1D array matching number of planets")

    rows = []
    pairs = []

    def _validate_nonnegative_finite(value: float, name: str) -> float:
        val = float(value)
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return val

    for i_src, src_row in enumerate(my_idx):
        src = planets[src_row]
        src_id = int(src[PLANET_COL_ID])
        src_ships = _validate_nonnegative_finite(src[PLANET_COL_SHIPS], "source ships")
        src_prod = _validate_nonnegative_finite(src[PLANET_COL_PROD], "source prod")
        keep = float(keeps[src_row])
        sendable = max(0.0, src_ships - keep)
        speed = fleet_speed(max(1.0, sendable))

        for i_tgt, tgt_row in enumerate(target_idx):
            tgt = planets[tgt_row]
            tgt_id = int(tgt[PLANET_COL_ID])
            tgt_ships = _validate_nonnegative_finite(tgt[PLANET_COL_SHIPS], "target ships")
            tgt_prod = _validate_nonnegative_finite(tgt[PLANET_COL_PROD], "target prod")
            tgt_owner = int(tgt[PLANET_COL_OWNER])
            eff_tgt_prod = 0.0 if tgt_owner < 0 else tgt_prod
            dist = float(dmat[i_src, i_tgt])
            eta = dist / speed if speed > 0 else math.inf
            required = float(estimate_required_capture_ships(tgt_ships, eff_tgt_prod, eta, margin=2))
            surplus = sendable - required
            sun_blocked = float(
                segment_hits_circle(
                    [src[PLANET_COL_X], src[PLANET_COL_Y]],
                    [tgt[PLANET_COL_X], tgt[PLANET_COL_Y]],
                    [CENTER_X, CENTER_Y],
                    SUN_RADIUS,
                )
            )
            target_is_neutral = float(tgt_owner < 0)
            target_is_enemy = float(tgt_owner >= 0 and tgt_owner != state.player)
            owner_rel = float(tgt_owner - state.player)
            score_pressure = (tgt_prod + 1.0) / (required + 1.0)

            rows.append(
                [
                    state.step / EPISODE_STEPS,
                    max(0.0, EPISODE_STEPS - state.step) / EPISODE_STEPS,
                    src_ships,
                    src_prod,
                    tgt_ships,
                    tgt_prod,
                    owner_rel,
                    dist,
                    eta,
                    required,
                    surplus,
                    sun_blocked,
                    target_is_neutral,
                    target_is_enemy,
                    keep,
                    score_pressure,
                ]
            )
            pairs.append([src_id, tgt_id])

    feature_arr = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(feature_arr)):
        raise ValueError("feature generation produced non-finite values")
    return feature_arr, np.asarray(pairs, dtype=np.int64)


def opponent_profile_features(state: GameState) -> dict[str, float]:
    fleets = state.fleets
    if fleets.shape[0] == 0:
        return {
            "enemy_fleet_count": 0.0,
            "enemy_mean_launch_ships": 0.0,
            "swarm_score": 0.0,
            "heavy_score": 0.0,
        }

    owners = fleets[:, FLEET_COL_OWNER]
    ships = fleets[:, FLEET_COL_SHIPS]
    valid = np.isfinite(owners) & np.isfinite(ships) & (ships > 0.0)
    enemy_mask = valid & (owners >= 0.0) & (owners != float(state.player))
    enemy_ships = ships[enemy_mask]
    if enemy_ships.size == 0:
        return {
            "enemy_fleet_count": 0.0,
            "enemy_mean_launch_ships": 0.0,
            "swarm_score": 0.0,
            "heavy_score": 0.0,
        }

    enemy_count = float(enemy_ships.size)
    # Use overflow-safe mean (scaling by max) to avoid inf/warnings
    max_ship = float(np.max(enemy_ships))
    if not np.isfinite(max_ship) or max_ship <= 0.0:
        mean_ships = 0.0
    else:
        scaled = enemy_ships / max_ship
        mean_ships = float(np.mean(scaled)) * max_ship
    if not np.isfinite(mean_ships):
        mean_ships = float(np.finfo(np.float64).max)
    small_ratio = float(np.mean(enemy_ships <= 8.0))
    count_factor = min(1.0, enemy_count / 6.0)
    swarm_score = min(1.0, max(0.0, small_ratio * count_factor))
    heavy_score = min(1.0, max(0.0, mean_ships / 40.0))
    # Guarantee finite bounded outputs
    if not np.isfinite(swarm_score):
        swarm_score = 0.0
    if not np.isfinite(heavy_score):
        heavy_score = 1.0
    return {
        "enemy_fleet_count": enemy_count,
        "enemy_mean_launch_ships": mean_ships,
        "swarm_score": swarm_score,
        "heavy_score": heavy_score,
    }
