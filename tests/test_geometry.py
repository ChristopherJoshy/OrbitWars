import math

import numpy as np

from src.ow.constants import (
    BOARD_SIZE,
    CENTER_X,
    CENTER_Y,
    COMET_SPEED,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    EPISODE_STEPS,
    FLEET_COL_ANGLE,
    FLEET_COL_FROM,
    FLEET_COL_ID,
    FLEET_COL_OWNER,
    FLEET_COL_SHIPS,
    FLEET_COL_X,
    FLEET_COL_Y,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS,
    PLANET_COL_ID,
    PLANET_COL_OWNER,
    PLANET_COL_PROD,
    PLANET_COL_RADIUS,
    PLANET_COL_SHIPS,
    PLANET_COL_X,
    PLANET_COL_Y,
    SHIP_SPEED_MAX,
    SUN_RADIUS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
)
from src.ow.geometry import aim_with_prediction, angle_between, distance_matrix, fleet_speed, predict_planet_position, segment_hits_circle


def test_fleet_speed_values():
    assert fleet_speed(1) == 1.0
    assert np.isclose(fleet_speed(1000), 6.0, atol=1e-6)
    expected_500 = 1 + 5 * (math.log(500) / math.log(1000)) ** 1.5
    assert math.isclose(fleet_speed(500), expected_500, rel_tol=1e-12, abs_tol=1e-12)


def test_constants_match_official_values_and_column_indexes():
    assert BOARD_SIZE == 100.0
    assert CENTER_X == 50.0
    assert CENTER_Y == 50.0
    assert SUN_RADIUS == 10.0
    assert SHIP_SPEED_MAX == 6.0
    assert EPISODE_STEPS == 500
    assert COMET_SPEED == 4.0
    assert ROTATION_RADIUS_LIMIT == 50.0
    assert COMET_RADIUS == 1.0
    assert COMET_PRODUCTION == 1
    assert PLANET_CLEARANCE == 0.5
    assert MIN_PLANET_GROUPS == 5
    assert MAX_PLANET_GROUPS == 10
    assert MIN_STATIC_GROUPS == 2
    assert COMET_SPAWN_STEPS == (50, 150, 250, 350, 450)
    assert (PLANET_COL_ID, PLANET_COL_OWNER, PLANET_COL_X, PLANET_COL_Y, PLANET_COL_RADIUS, PLANET_COL_SHIPS, PLANET_COL_PROD) == (0, 1, 2, 3, 4, 5, 6)
    assert (FLEET_COL_ID, FLEET_COL_OWNER, FLEET_COL_X, FLEET_COL_Y, FLEET_COL_ANGLE, FLEET_COL_FROM, FLEET_COL_SHIPS) == (0, 1, 2, 3, 4, 5, 6)


def test_segment_hits_circle_true_and_false():
    assert segment_hits_circle((-10.0, 0.0), (10.0, 0.0), (0.0, 0.0), 2.0)
    assert not segment_hits_circle((10.0, 10.0), (12.0, 12.0), (0.0, 0.0), 2.0)


def test_distance_matrix_shape_and_values():
    a = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[0.0, 0.0], [6.0, 8.0]], dtype=np.float64)
    dm = distance_matrix(a, b)
    assert dm.shape == (2, 2)
    assert np.isclose(dm[1, 0], 5.0)
    assert np.isclose(dm[1, 1], 5.0)


def test_angle_between_uses_atan2_semantics():
    assert np.isclose(angle_between(0, 0, 1, 0), 0.0)
    assert np.isclose(angle_between(0, 0, 0, 1), np.pi / 2)


def test_fleet_speed_rejects_nan_and_non_positive():
    for ships in (float("nan"), 0, -1):
        try:
            fleet_speed(ships)
            assert False, f"Expected ValueError for ships={ships}"
        except ValueError:
            pass


def test_segment_hits_circle_rejects_negative_radius():
    try:
        segment_hits_circle((0.0, 0.0), (1.0, 1.0), (0.0, 0.0), -1.0)
        assert False, "Expected ValueError for negative radius"
    except ValueError:
        pass


def test_segment_hits_circle_rejects_non_finite_or_malformed_points():
    bad_cases = [
        ((float("nan"), 0.0), (1.0, 1.0), (0.0, 0.0), 1.0),
        ((0.0, 0.0, 1.0), (1.0, 1.0), (0.0, 0.0), 1.0),
        ((0.0, 0.0), (1.0, float("inf")), (0.0, 0.0), 1.0),
        ((0.0, 0.0), (1.0, 1.0), (0.0,), 1.0),
    ]
    for args in bad_cases:
        try:
            segment_hits_circle(*args)
            assert False, f"Expected ValueError for args={args}"
        except ValueError:
            pass


def test_predict_planet_position_rotating_and_static_cases():
    rotating = np.array([1, 0, 60.0, 50.0, 1.0, 5.0, 1.0], dtype=np.float64)
    x1, y1 = predict_planet_position(rotating, turns=5.0, angular_velocity=0.2)
    assert not np.isclose(x1, rotating[PLANET_COL_X])
    assert not np.isclose(y1, rotating[PLANET_COL_Y])

    static = np.array([2, 0, 120.0, 120.0, 1.0, 5.0, 1.0], dtype=np.float64)
    x2, y2 = predict_planet_position(static, turns=5.0, angular_velocity=0.2)
    assert np.isclose(x2, static[PLANET_COL_X])
    assert np.isclose(y2, static[PLANET_COL_Y])


def test_aim_with_prediction_rotating_target_changes_angle_and_returns_finite_positive_values():
    source_x, source_y = 20.0, 50.0
    target = np.array([3, -1, 60.0, 50.0, 1.0, 5.0, 1.0], dtype=np.float64)
    angle_now = angle_between(source_x, source_y, target[PLANET_COL_X], target[PLANET_COL_Y])
    angle_pred, eta, distance = aim_with_prediction(source_x, source_y, target, ships=50, angular_velocity=0.3)
    assert np.isfinite(angle_pred)
    assert np.isfinite(eta) and eta > 0.0
    assert np.isfinite(distance) and distance > 0.0
    assert abs(angle_pred - angle_now) > 1e-3
