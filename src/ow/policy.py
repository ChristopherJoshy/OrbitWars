from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.ow.constants import CENTER_X, CENTER_Y, EPISODE_STEPS, PLANET_COL_ID, PLANET_COL_OWNER, PLANET_COL_PROD, PLANET_COL_SHIPS, PLANET_COL_X, PLANET_COL_Y, SUN_RADIUS
from src.ow.features import FEATURE_COLUMNS, build_candidate_features, opponent_profile_features
from src.ow.geometry import aim_with_prediction, predict_planet_position, segment_hits_circle
from src.ow.models_numpy import LinearModel
from src.ow.timeline import estimate_required_capture_ships_fixed_point, threat_keep_needed_by_planet


@dataclass(frozen=True)
class PolicyConfig:
    max_moves: int = 8
    min_send: int = 1
    time_budget_ms: float = 20.0


def score_candidates(features: np.ndarray, model_scores: np.ndarray | None = None) -> np.ndarray:
    probs = np.full((features.shape[0],), 0.5, dtype=np.float64) if model_scores is None else model_scores
    score_pressure_idx = FEATURE_COLUMNS.index("score_pressure")
    sun_blocked_idx = FEATURE_COLUMNS.index("sun_blocked")
    distance_idx = FEATURE_COLUMNS.index("distance")
    turn_idx = FEATURE_COLUMNS.index("turn_norm")
    tgt_prod_idx = FEATURE_COLUMNS.index("target_prod")
    neutral_idx = FEATURE_COLUMNS.index("target_is_neutral")
    enemy_idx = FEATURE_COLUMNS.index("target_is_enemy")
    surplus_idx = FEATURE_COLUMNS.index("surplus_after_send")
    base = probs + 0.25 * features[:, score_pressure_idx] - 0.75 * features[:, sun_blocked_idx] - 0.006 * features[:, distance_idx]
    early = np.clip(1.0 - (features[:, turn_idx] / 0.25), 0.0, 1.0)
    mid = np.clip(1.0 - np.abs(features[:, turn_idx] - 0.55) / 0.30, 0.0, 1.0)
    affordable = (features[:, surplus_idx] >= 0.0).astype(np.float64)
    base += early * affordable * features[:, neutral_idx] * (0.08 * features[:, tgt_prod_idx])
    base += mid * affordable * features[:, enemy_idx] * (0.07 * features[:, tgt_prod_idx])
    low_surplus_enemy = (features[:, surplus_idx] <= 5.0).astype(np.float64) * features[:, enemy_idx]
    low_surplus_neutral = (features[:, surplus_idx] <= 2.0).astype(np.float64) * features[:, neutral_idx]
    base -= 0.25 * low_surplus_enemy
    base -= 0.08 * low_surplus_neutral
    return base


def choose_moves(state, model: LinearModel | None = None, config: PolicyConfig | None = None) -> list[list[float]]:
    raw_cfg = config or PolicyConfig()
    safe_time_budget_ms = 20.0 if (not np.isfinite(raw_cfg.time_budget_ms) or raw_cfg.time_budget_ms < 0.0) else float(raw_cfg.time_budget_ms)
    cfg = PolicyConfig(max_moves=max(0, min(10, int(raw_cfg.max_moves))), min_send=max(1, int(raw_cfg.min_send)), time_budget_ms=safe_time_budget_ms)
    t0 = time.perf_counter()
    keeps = threat_keep_needed_by_planet(state, horizon=max(0, EPISODE_STEPS - state.step))
    features, pairs = build_candidate_features(state, keeps=keeps)
    if pairs.shape[0] == 0:
        return []

    probs = np.full((features.shape[0],), 0.5, dtype=np.float64) if model is None else model.predict_proba(features)
    scores = score_candidates(features, probs)
    surplus_idx = FEATURE_COLUMNS.index("surplus_after_send")
    low_surplus_mask = features[:, surplus_idx] <= 5.0
    if np.any(low_surplus_mask):
        profile = opponent_profile_features(state)
        swarm_score = float(profile["swarm_score"])
        if np.any(low_surplus_mask):
            if swarm_score > 0.0:
                scores[low_surplus_mask] -= 0.6 * swarm_score
    order = np.argsort(-scores)

    planets = state.planets
    id_to_row = {int(planets[i, PLANET_COL_ID]): i for i in range(planets.shape[0])}
    available = {int(planets[i, PLANET_COL_ID]): max(0, int(np.floor(planets[i, PLANET_COL_SHIPS])) - int(keeps[i])) for i in np.flatnonzero(state.my_planet_mask)}

    out: list[list[float]] = []
    claimed_targets: set[int] = set()
    for idx in order.tolist():
        if len(out) >= cfg.max_moves:
            break
        if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms:
            break

        src_id, tgt_id = int(pairs[idx, 0]), int(pairs[idx, 1])
        if tgt_id in claimed_targets:
            continue
        src_row = id_to_row.get(src_id)
        tgt_row = id_to_row.get(tgt_id)
        if src_row is None or tgt_row is None:
            continue

        sx, sy = planets[src_row, PLANET_COL_X], planets[src_row, PLANET_COL_Y]

        can_send = available.get(src_id, 0)
        target_ships = float(planets[tgt_row, PLANET_COL_SHIPS])
        target_prod = float(planets[tgt_row, PLANET_COL_PROD])
        target_owner = int(planets[tgt_row, PLANET_COL_OWNER])
        eff_prod = 0.0 if target_owner < 0 else target_prod
        dist = float(features[idx, FEATURE_COLUMNS.index("distance")])
        need = int(max(cfg.min_send, estimate_required_capture_ships_fixed_point(target_ships, eff_prod, dist, margin=2)))
        if can_send < need:
            continue
        ships = need
        if ships < cfg.min_send:
            continue
        ang, eta, predicted_dist = aim_with_prediction(float(sx), float(sy), planets[tgt_row], ships=ships, angular_velocity=state.angular_velocity)
        if state.step + int(np.ceil(eta)) >= EPISODE_STEPS:
            continue

        ptx, pty = predict_planet_position(planets[tgt_row], turns=eta, angular_velocity=state.angular_velocity)
        if segment_hits_circle([sx, sy], [ptx, pty], [CENTER_X, CENTER_Y], SUN_RADIUS):
            continue
        if predicted_dist <= 0.0 or not np.isfinite(predicted_dist):
            continue

        out.append([float(src_id), ang, float(ships)])
        available[src_id] = can_send - ships
        claimed_targets.add(tgt_id)
    return out
