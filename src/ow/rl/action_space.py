from __future__ import annotations

import math

from src.ow.geometry import aim_with_prediction, predict_planet_position
from src.ow.timeline import estimate_required_capture_ships_fixed_point

PLANET_ID = 0
PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
PLANET_SHIPS = 5
PLANET_PRODUCTION = 6
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD_MIN = 0.0
BOARD_MAX = 100.0


def angle_to_target(src_xy: tuple[float, float], dst_xy: tuple[float, float]) -> float:
    sx, sy = src_xy
    dx, dy = dst_xy
    return float(math.atan2(dy - sy, dx - sx))


def build_action_candidates(
    obs: dict,
    max_targets_per_source: int = 8,
    send_fraction: float | None = None,
    avoid_waste: bool = True,
    send_fractions: tuple[float, ...] | None = None,
    include_owned_targets: bool = True,
) -> list[dict]:
    player = int(obs.get("player", 0))
    planets = obs.get("planets", []) or []
    owned = [p for p in planets if int(p[PLANET_OWNER]) == player and int(p[PLANET_SHIPS]) > 1]
    targets = [p for p in planets if bool(include_owned_targets) or int(p[PLANET_OWNER]) != player]
    player_planets = [p for p in planets if int(p[PLANET_OWNER]) == player]
    enemy_planets = [p for p in planets if int(p[PLANET_OWNER]) not in {player, -1}]
    neutral_planets = [p for p in planets if int(p[PLANET_OWNER]) == -1]
    context = {
        "my_planet_count": len(player_planets),
        "enemy_planet_count": len(enemy_planets),
        "neutral_planet_count": len(neutral_planets),
        "my_total_ships": int(sum(int(p[PLANET_SHIPS]) for p in player_planets)),
        "enemy_total_ships": int(sum(int(p[PLANET_SHIPS]) for p in enemy_planets)),
        "neutral_total_ships": int(sum(int(p[PLANET_SHIPS]) for p in neutral_planets)),
        "my_production": int(sum(int(p[PLANET_PRODUCTION]) for p in player_planets if len(p) > PLANET_PRODUCTION)),
        "enemy_production": int(sum(int(p[PLANET_PRODUCTION]) for p in enemy_planets if len(p) > PLANET_PRODUCTION)),
    }
    fraction_values = tuple(
        float(x)
        for x in (
            send_fractions
            if send_fractions is not None
            else ((send_fraction,) if send_fraction is not None else (0.35, 0.55, 0.75))
        )
    )
    expand_ship_counts = send_fraction is None and send_fractions is None
    angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
    candidates = [{"kind": "noop", "source_id": -1, "target_id": -1, "angle": 0.0, "ships": 0, **context}]

    def make_launch_candidate(src: list, target: list, ships: int) -> dict:
        source_id = int(src[PLANET_ID])
        ships = max(1, min(int(ships), int(src[PLANET_SHIPS]) - 1))
        angle, eta, predicted_distance = aim_with_prediction(
            float(src[PLANET_X]),
            float(src[PLANET_Y]),
            target,
            ships=ships,
            angular_velocity=angular_velocity,
        )
        intercept_x, intercept_y = predict_planet_position(target, turns=eta, angular_velocity=angular_velocity)
        target_owner = int(target[PLANET_OWNER])
        target_ships = int(target[PLANET_SHIPS])
        target_production = int(target[PLANET_PRODUCTION]) if len(target) > PLANET_PRODUCTION else 0
        required_capture_ships = estimate_required_capture_ships_fixed_point(
            target_ships,
            target_production if target_owner not in {player, -1} else 0,
            predicted_distance,
        )
        return {
            "kind": "launch",
            "source_id": source_id,
            "target_id": int(target[PLANET_ID]),
            "source_x": float(src[PLANET_X]),
            "source_y": float(src[PLANET_Y]),
            "target_x": float(target[PLANET_X]),
            "target_y": float(target[PLANET_Y]),
            "intercept_x": float(intercept_x),
            "intercept_y": float(intercept_y),
            "target_radius": float(target[PLANET_RADIUS]),
            "angle": angle,
            "eta": float(eta),
            "ships": ships,
            "source_ships": int(src[PLANET_SHIPS]),
            "target_owner": target_owner,
            "target_is_own": target_owner == player,
            "target_is_enemy": target_owner not in {player, -1},
            "target_is_neutral": target_owner == -1,
            "target_ships": target_ships,
            "target_production": target_production,
            "required_capture_ships": int(required_capture_ships),
            "distance": float(predicted_distance),
            **context,
        }

    for src in owned:
        source_id = int(src[PLANET_ID])
        ranked = sorted(
            [target for target in targets if int(target[PLANET_ID]) != source_id],
            key=lambda p: (float(p[PLANET_X]) - float(src[PLANET_X])) ** 2 + (float(p[PLANET_Y]) - float(src[PLANET_Y])) ** 2,
        )[:max_targets_per_source]
        for target in ranked:
            used_ship_counts: set[int] = set()
            candidate_ship_counts = [max(1, int(int(src[PLANET_SHIPS]) * max(0.0, fraction))) for fraction in fraction_values]
            target_owner = int(target[PLANET_OWNER])
            if expand_ship_counts and target_owner != player:
                probe = make_launch_candidate(src, target, int(src[PLANET_SHIPS]) - 1)
                required = int(probe.get("required_capture_ships", int(target[PLANET_SHIPS]) + 1))
                candidate_ship_counts.extend([required, required + 2, int(int(src[PLANET_SHIPS]) * 0.9), int(src[PLANET_SHIPS]) - 1])
            for ships in candidate_ship_counts:
                ships = max(1, min(int(ships), int(src[PLANET_SHIPS]) - 1))
                if ships in used_ship_counts:
                    continue
                used_ship_counts.add(ships)
                candidate = make_launch_candidate(src, target, ships)
                if avoid_waste and estimate_candidate_mistakes(obs, candidate)["wasted_ships"] > 0:
                    continue
                if not candidate_is_strategically_useful(candidate):
                    continue
                candidates.append(candidate)
    if expand_ship_counts:
        candidates.extend(build_macro_candidates(candidates, max_macros=24))
    return candidates


def build_macro_candidates(candidates: list[dict], max_macros: int = 24) -> list[dict]:
    launches = [
        c
        for c in candidates
        if c.get("kind") == "launch" and not bool(c.get("target_is_own", False)) and int(c.get("ships", 0)) > 0
    ]
    macros: list[dict] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for i, first in enumerate(launches):
        for second in launches[i + 1 :]:
            if int(first["source_id"]) == int(second["source_id"]):
                continue
            if int(first["target_id"]) == int(second["target_id"]):
                continue
            bundle = [first, second]
            key = tuple(sorted((int(c["source_id"]), int(c["target_id"]), int(c["ships"])) for c in bundle))
            if key in seen:
                continue
            seen.add(key)
            total_ships = sum(int(c["ships"]) for c in bundle)
            macro = {
                **first,
                "kind": "macro",
                "macro_size": len(bundle),
                "ships": total_ships,
                "launches": [
                    {
                        "source_id": int(c["source_id"]),
                        "angle": float(c["angle"]),
                        "ships": int(c["ships"]),
                        "target_id": int(c["target_id"]),
                    }
                    for c in bundle
                ],
            }
            macros.append(macro)
            if len(macros) >= int(max_macros):
                return macros
    return macros


def candidate_is_strategically_useful(candidate: dict) -> bool:
    """Reject dominated non-owned launches before policy scoring.

    Own-planet reinforcement remains unrestricted. Neutral launches must be
    able to capture immediately; enemy launches must either capture or send a
    material attrition fleet. This keeps the RL policy from spending deterministic
    argmax capacity on tiny fleets into impossible high-ship targets.
    """
    if candidate.get("kind") == "noop":
        return True
    if candidate.get("kind") == "macro":
        return True
    ships = int(candidate.get("ships", 0))
    target_ships = int(candidate.get("target_ships", 0))
    if bool(candidate.get("target_is_own", False)):
        return True
    if bool(candidate.get("target_is_neutral", False)):
        required_capture_ships = int(candidate.get("required_capture_ships", target_ships + 1))
        return ships >= required_capture_ships
    if bool(candidate.get("target_is_enemy", False)):
        required_capture_ships = int(candidate.get("required_capture_ships", target_ships + 1))
        captures_after_eta = ships >= required_capture_ships
        material_attrition = ships <= target_ships and float(ships) >= max(25.0, 0.25 * float(target_ships))
        return captures_after_eta or material_attrition
    return True


def candidate_to_env_action(candidate: dict) -> list:
    if candidate["kind"] == "noop":
        return []
    if candidate["kind"] == "macro":
        return [
            [int(launch["source_id"]), float(launch["angle"]), int(launch["ships"])]
            for launch in candidate.get("launches", [])
        ]
    return [[int(candidate["source_id"]), float(candidate["angle"]), int(candidate["ships"])]]


def _ray_circle_t(origin: tuple[float, float], angle: float, center: tuple[float, float], radius: float) -> float | None:
    ox, oy = origin
    cx, cy = center
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    candidates = [(-b - root) / 2.0, (-b + root) / 2.0]
    positive = [t for t in candidates if t > 1e-6]
    return min(positive) if positive else None


def _ray_board_exit_t(origin: tuple[float, float], angle: float) -> float:
    ox, oy = origin
    dx, dy = math.cos(angle), math.sin(angle)
    ts = []
    if abs(dx) > 1e-9:
        ts.extend([(BOARD_MIN - ox) / dx, (BOARD_MAX - ox) / dx])
    if abs(dy) > 1e-9:
        ts.extend([(BOARD_MIN - oy) / dy, (BOARD_MAX - oy) / dy])
    positive = [t for t in ts if t > 1e-6]
    return min(positive) if positive else float("inf")


def _segment_hits_circle_fast(
    p0: tuple[float, float],
    p1: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    """Scalar hot-path segment/circle test without NumPy validation overhead."""

    x0, y0 = p0
    x1, y1 = p1
    cx, cy = center
    dx = x1 - x0
    dy = y1 - y0
    dd = dx * dx + dy * dy
    if dd <= 0.0:
        return (x0 - cx) * (x0 - cx) + (y0 - cy) * (y0 - cy) <= radius * radius
    t = ((cx - x0) * dx + (cy - y0) * dy) / dd
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest_x = x0 + t * dx
    closest_y = y0 + t * dy
    return (closest_x - cx) * (closest_x - cx) + (closest_y - cy) * (closest_y - cy) <= radius * radius


def estimate_candidate_mistakes(obs: dict, candidate: dict) -> dict[str, int]:
    """Estimate cheap launch mistake components before sending an action.

    This intentionally uses a conservative straight-ray approximation: it flags
    launches whose path hits the sun, leaves the board, or hits a different
    planet before the intended target. These components are logged and shaped
    as penalties by the trainer.
    """
    ships = int(candidate.get("ships", 0))
    out = {"sun_loss_ships": 0, "wasted_ships": 0, "out_of_bounds_ships": 0, "wrong_target_ships": 0, "invalid_actions": 0}
    if candidate.get("kind") == "macro":
        by_source: dict[int, int] = {}
        planets = obs.get("planets", []) or []
        source_ships = {int(p[PLANET_ID]): int(p[PLANET_SHIPS]) for p in planets if len(p) > PLANET_SHIPS}
        for launch in candidate.get("launches", []) or []:
            source_id = int(launch.get("source_id", -999))
            by_source[source_id] = by_source.get(source_id, 0) + int(launch.get("ships", 0))
            launch_candidate = dict(candidate)
            launch_candidate.update(launch)
            launch_candidate["kind"] = "launch"
            for key, value in estimate_candidate_mistakes(obs, launch_candidate).items():
                out[key] = out.get(key, 0) + int(value)
        for source_id, total in by_source.items():
            if total >= source_ships.get(source_id, 0):
                out["invalid_actions"] += 1
                out["wasted_ships"] += int(total)
        return out
    if candidate.get("kind") == "noop" or ships <= 0:
        return out
    source_id = int(candidate.get("source_id", -999))
    target_id = int(candidate.get("target_id", -999))
    planets = obs.get("planets", []) or []
    by_id = {int(p[PLANET_ID]): p for p in planets if len(p) > PLANET_RADIUS}
    source = by_id.get(source_id)
    if source is None:
        out["invalid_actions"] = 1
        out["wasted_ships"] = ships
        return out
    origin = (float(source[PLANET_X]), float(source[PLANET_Y]))
    angle = float(candidate.get("angle", 0.0))

    if "intercept_x" in candidate and "intercept_y" in candidate:
        endpoint = (float(candidate["intercept_x"]), float(candidate["intercept_y"]))
        if not (BOARD_MIN <= endpoint[0] <= BOARD_MAX and BOARD_MIN <= endpoint[1] <= BOARD_MAX):
            out["out_of_bounds_ships"] = ships
            out["wasted_ships"] = ships
            return out
        if _segment_hits_circle_fast(origin, endpoint, SUN_CENTER, SUN_RADIUS):
            out["sun_loss_ships"] = ships
            out["wasted_ships"] = ships
            return out
        for planet in planets:
            if len(planet) <= PLANET_RADIUS:
                continue
            pid = int(planet[PLANET_ID])
            if pid in {source_id, target_id}:
                continue
            if _segment_hits_circle_fast(origin, endpoint, (float(planet[PLANET_X]), float(planet[PLANET_Y])), float(planet[PLANET_RADIUS])):
                out["wrong_target_ships"] = ships
                out["wasted_ships"] = ships
                return out
        return out

    events: list[tuple[float, str, int | None]] = []
    sun_t = _ray_circle_t(origin, angle, SUN_CENTER, SUN_RADIUS)
    if sun_t is not None:
        events.append((sun_t, "sun", None))
    events.append((_ray_board_exit_t(origin, angle), "out", None))
    for planet in planets:
        if len(planet) <= PLANET_RADIUS:
            continue
        pid = int(planet[PLANET_ID])
        if pid == source_id:
            continue
        t = _ray_circle_t(origin, angle, (float(planet[PLANET_X]), float(planet[PLANET_Y])), float(planet[PLANET_RADIUS]))
        if t is not None:
            events.append((t, "planet", pid))
    first = min(events, key=lambda item: item[0]) if events else (float("inf"), "out", None)
    if first[1] == "sun":
        out["sun_loss_ships"] = ships
        out["wasted_ships"] = ships
    elif first[1] == "out":
        out["out_of_bounds_ships"] = ships
        out["wasted_ships"] = ships
    elif first[1] == "planet" and first[2] != target_id:
        out["wrong_target_ships"] = ships
        out["wasted_ships"] = ships
    return out
