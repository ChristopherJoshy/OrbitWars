import math

import numpy as np

from src.ow.constants import CENTER_X, CENTER_Y, PLANET_COL_RADIUS, PLANET_COL_X, PLANET_COL_Y, ROTATION_RADIUS_LIMIT, SHIP_SPEED_MAX


def fleet_speed(ships: float) -> float:
    ships = float(ships)
    if not math.isfinite(ships) or ships <= 0.0:
        raise ValueError("ships must be positive and finite")
    if ships <= 1.0:
        return 1.0
    speed = 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(SHIP_SPEED_MAX, max(1.0, speed))


def angle_between(x0, y0, x1, y1) -> float:
    return math.atan2(y1 - y0, x1 - x0)


def distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(delta, axis=2)


def segment_hits_circle(p0, p1, center, radius: float) -> bool:
    def _point2(value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 1 or arr.shape != (2,) or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite point of shape (2,)")
        return arr

    p0 = _point2(p0, "start")
    p1 = _point2(p1, "end")
    c = _point2(center, "center")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and >= 0")
    r2 = radius * radius

    d = p1 - p0
    dd = float(np.dot(d, d))
    if dd == 0.0:
        return float(np.dot(p0 - c, p0 - c)) <= r2

    t = float(np.dot(c - p0, d) / dd)
    t = min(1.0, max(0.0, t))
    closest = p0 + t * d
    return float(np.dot(closest - c, closest - c)) <= r2


def predict_planet_position(planet_row, turns, angular_velocity) -> tuple[float, float]:
    row = np.asarray(planet_row, dtype=np.float64)
    if row.ndim != 1 or row.shape[0] <= PLANET_COL_RADIUS or not np.isfinite(row).all():
        raise ValueError("planet_row must be a finite 1D row with x/y/radius")
    turns = float(turns)
    angular_velocity = float(angular_velocity)
    if not math.isfinite(turns) or turns < 0.0:
        raise ValueError("turns must be finite and >= 0")
    if not math.isfinite(angular_velocity):
        raise ValueError("angular_velocity must be finite")

    x = float(row[PLANET_COL_X])
    y = float(row[PLANET_COL_Y])
    radius = float(row[PLANET_COL_RADIUS])
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("planet radius must be finite and >= 0")
    dx = x - CENTER_X
    dy = y - CENTER_Y
    orbital_r = math.hypot(dx, dy)
    if orbital_r + radius >= ROTATION_RADIUS_LIMIT:
        return (x, y)
    theta = angular_velocity * turns
    ct = math.cos(theta)
    st = math.sin(theta)
    return (CENTER_X + dx * ct - dy * st, CENTER_Y + dx * st + dy * ct)


def aim_with_prediction(source_x, source_y, target_row, ships, angular_velocity, max_iter=6) -> tuple[float, float, float]:
    source_x = float(source_x)
    source_y = float(source_y)
    ships = float(ships)
    max_iter = int(max_iter)
    if not math.isfinite(source_x) or not math.isfinite(source_y):
        raise ValueError("source position must be finite")
    if not math.isfinite(ships) or ships <= 0.0:
        raise ValueError("ships must be positive and finite")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not math.isfinite(float(angular_velocity)):
        raise ValueError("angular_velocity must be finite")

    tx0, ty0 = predict_planet_position(target_row, turns=0.0, angular_velocity=angular_velocity)
    dist = math.hypot(tx0 - source_x, ty0 - source_y)
    speed = fleet_speed(ships)
    eta = dist / speed
    tx, ty = tx0, ty0
    for _ in range(max_iter):
        tx, ty = predict_planet_position(target_row, turns=eta, angular_velocity=angular_velocity)
        new_dist = math.hypot(tx - source_x, ty - source_y)
        new_eta = new_dist / speed
        if abs(new_eta - eta) <= 1e-6:
            dist, eta = new_dist, new_eta
            break
        dist, eta = new_dist, new_eta
    angle = angle_between(source_x, source_y, tx, ty)
    return (float(angle), float(eta), float(dist))
