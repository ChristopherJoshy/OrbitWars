import numpy as np
import pytest

from src.ow.constants import PLANET_COL_PROD, PLANET_COL_SHIPS, ROW_WIDTH
from src.ow.geometry import fleet_speed, predict_planet_position
from src.ow.state import parse_observation
from src.ow.timeline import estimate_required_capture_ships, estimate_required_capture_ships_fixed_point, keep_needed_by_planet, threat_keep_needed_by_planet


def test_estimate_required_capture_ships_example_case():
    assert estimate_required_capture_ships(10, 2, 3, margin=2) == 19


def test_estimate_required_capture_ships_fixed_point_oscillation_regression_safe_result():
    s = estimate_required_capture_ships_fixed_point(0, 1, 1.5, margin=2)
    assert s >= 5
    eta = 1.5 / fleet_speed(s)
    required = estimate_required_capture_ships(0, 1, eta, margin=2)
    assert s >= required


def test_estimate_required_capture_ships_fixed_point_high_prod_long_distance_safe_result():
    s = estimate_required_capture_ships_fixed_point(0, 9, 128.8, margin=2)
    eta = 128.8 / fleet_speed(s)
    required = estimate_required_capture_ships(0, 9, eta, margin=2)
    assert s >= required


@pytest.mark.parametrize(
    "target_ships,target_prod,eta,margin",
    [
        (np.nan, 1, 1, 2),
        (1, np.nan, 1, 2),
        (1, 1, np.nan, 2),
        (1, 1, 1, np.nan),
        (-1, 1, 1, 2),
        (1, -1, 1, 2),
        (1, 1, -1, 2),
        (1, 1, 1, -1),
    ],
)
def test_estimate_required_capture_ships_rejects_invalid_inputs(target_ships, target_prod, eta, margin):
    with pytest.raises(ValueError):
        estimate_required_capture_ships(target_ships, target_prod, eta, margin)


def test_keep_needed_by_planet_owned_planets_only_and_bounded():
    planets = np.array(
        [
            [0, 0, 10, 10, 2, 30, 3],
            [1, 1, 20, 10, 2, 40, 4],
            [2, -1, 30, 10, 2, 5, 1],
        ],
        dtype=np.float64,
    )
    reserves = keep_needed_by_planet(planets, player=0, horizon=20)
    assert reserves.shape == (3,)
    assert reserves.dtype == np.int64
    assert reserves[0] >= 0
    assert reserves[1] == 0
    assert reserves[2] == 0
    assert reserves[0] <= 30


def test_keep_needed_by_planet_accepts_true_empty_schema():
    planets = np.zeros((0, ROW_WIDTH), dtype=np.float64)
    reserves = keep_needed_by_planet(planets, player=0, horizon=20)
    assert reserves.shape == (0,)
    assert reserves.dtype == np.int64


@pytest.mark.parametrize(
    "planets",
    [
        np.zeros((0, ROW_WIDTH - 1), dtype=np.float64),
        np.zeros((1, ROW_WIDTH - 1), dtype=np.float64),
        np.zeros((ROW_WIDTH,), dtype=np.float64),
        [[0, 0, 10, 10, 2, 10]],
    ],
)
def test_keep_needed_by_planet_rejects_bad_schema_shapes(planets):
    with pytest.raises(ValueError, match="planets|width|schema"):
        keep_needed_by_planet(planets, player=0, horizon=20)


@pytest.mark.parametrize(
    "ships,prod",
    [
        (np.nan, 1.0),
        (1.0, np.nan),
        (np.inf, 1.0),
        (1.0, np.inf),
        (-1.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_keep_needed_by_planet_rejects_invalid_ships_or_prod(ships, prod):
    planets = np.array([[0, 0, 10, 10, 2, ships, prod]], dtype=np.float64)
    with pytest.raises(ValueError, match="ships|prod|finite|nonnegative"):
        keep_needed_by_planet(planets, player=0, horizon=20)


def test_keep_needed_opening_reserve_allows_send_and_scales_with_production():
    planets = np.array(
        [
            [0, 0, 10, 10, 2, 10, 3],
            [1, 0, 20, 10, 2, 10, 1],
            [2, 0, 30, 10, 2, 50, 6],
        ],
        dtype=np.float64,
    )
    reserves = keep_needed_by_planet(planets, player=0, horizon=500)
    assert reserves[0] <= 2
    assert reserves[0] < 10
    assert reserves[0] > reserves[1]
    assert reserves[2] > reserves[0]


def test_threat_keep_needed_reserves_for_intersecting_enemy_only():
    obs = {
        "player": 0,
        "step": 0,
        "planets": [[1, 0, 20, 20, 2, 50, 3], [2, 1, 90, 90, 2, 10, 2]],
        "fleets": [
            [10, 1, 10, 20, 0.0, 0, 20],   # enemy, intersects p1 path
            [11, 0, 10, 20, 0.0, 0, 50],   # friendly, ignored
            [12, 1, 10, 90, 0.0, 0, 30],   # enemy, does not hit p1
        ],
    }
    state = parse_observation(obs)
    baseline = keep_needed_by_planet(state.planets, player=state.player, horizon=300)
    threat = threat_keep_needed_by_planet(state, horizon=300)
    assert threat[0] > baseline[0]
    assert threat[1] == baseline[1]


def test_threat_keep_needed_accounts_for_rotating_future_planet_position():
    angular_velocity = np.pi / 20.0
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": angular_velocity,
        "planets": [[1, 0, 60, 50, 2, 50, 3]],
        "fleets": [[10, 1, 40, 60, 0.0, 0, 1]],
    }
    state = parse_observation(obs)

    eta = 10.0 / fleet_speed(1)
    px_now, py_now = state.planets[0, 2], state.planets[0, 3]
    px_eta, py_eta = predict_planet_position(state.planets[0], turns=eta, angular_velocity=state.angular_velocity)
    assert abs(py_now - 60.0) > 5.0
    assert abs(py_eta - 60.0) < 1e-6
    assert abs(px_eta - 50.0) < 1e-6

    baseline = keep_needed_by_planet(state.planets, player=state.player, horizon=300)
    threat = threat_keep_needed_by_planet(state, horizon=300)
    assert threat[0] > baseline[0]


def test_threat_keep_needed_does_not_freeze_unsavable_planet():
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "planets": [[1, 0, 50, 50, 2, 8, 1]],
        "fleets": [[10, 1, 30, 50, 0.0, 0, 20]],
    }
    state = parse_observation(obs)
    baseline = keep_needed_by_planet(state.planets, player=state.player, horizon=300)
    threat = threat_keep_needed_by_planet(state, horizon=300)
    assert threat[0] == baseline[0]
