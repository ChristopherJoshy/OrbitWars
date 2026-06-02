from types import SimpleNamespace
import warnings

import numpy as np

from src.ow.state import GameState, parse_observation


def test_parse_observation_from_dict_and_object():
    obs_dict = {
        "player": 1,
        "step": 7,
        "angular_velocity": 0.25,
        "planets": [
            [0, -1, 0.0, 0.0, 3, 20, 5],
            [1, 1, 1.0, 0.0, 2, 10, 3],
        ],
        "fleets": [[1, 0, 1, 5, 4, 0, 0]],
        "comets": [],
        "initial_planets": [
            [0, -1, 0.0, 0.0, 3, 20, 5],
            [1, 1, 1.0, 0.0, 2, 10, 3],
        ],
    }

    state = parse_observation(obs_dict)
    assert isinstance(state, GameState)
    assert state.player == 1
    assert state.step == 7
    assert state.angular_velocity == 0.25
    assert state.planets.shape == (2, 7)
    assert state.fleets.shape == (1, 7)
    assert state.fleets.dtype == np.float64
    assert state.comets.shape == (0, 7)
    assert state.my_planet_mask.tolist() == [False, True]
    assert state.enemy_planet_mask.tolist() == [False, False]
    assert state.neutral_planet_mask.tolist() == [True, False]

    obj = SimpleNamespace(**{**obs_dict, "turn": 9, "step": None})
    state2 = parse_observation(obj)
    assert state2.step == 9
    assert state2.angular_velocity == 0.25
    assert state2.planets.shape == (2, 7)


def test_parse_observation_raises_for_wrong_width_non_empty_array():
    obs = {
        "player": 0,
        "step": 0,
        "planets": [[0, 1, 2, 3, 4, 5]],
    }
    try:
        parse_observation(obs)
        assert False, "Expected ValueError for malformed planets width"
    except ValueError as exc:
        msg = str(exc)
        assert "planets" in msg
        assert "expected width 7" in msg


def test_parse_observation_rejects_explicit_empty_planets_wrong_width():
    obs = {
        "player": 0,
        "step": 0,
        "planets": np.empty((0, 6)),
    }

    try:
        parse_observation(obs)
        assert False, "Expected ValueError for malformed empty planets width"
    except ValueError as exc:
        msg = str(exc)
        assert "planets" in msg
        assert "7" in msg


def test_parse_observation_rejects_ragged_or_non_numeric_planets():
    obs = {
        "player": 0,
        "step": 0,
        "planets": [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2]],
    }

    try:
        parse_observation(obs)
        assert False, "Expected ValueError for ragged/non-numeric planets"
    except ValueError as exc:
        msg = str(exc)
        assert "planets" in msg
        assert ("7" in msg) or ("numeric" in msg)


def test_parse_observation_non_finite_owner_has_no_runtime_warning_and_masks_false():
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [1, float("nan"), 10, 10, 2, 30, 3],
            [2, float("inf"), 20, 20, 2, 5, 1],
            [3, float("-inf"), 30, 30, 2, 5, 1],
        ],
    }

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        state = parse_observation(obs)

    assert len(record) == 0
    assert state.my_planet_mask.tolist() == [False, False, False]
    assert state.enemy_planet_mask.tolist() == [False, False, False]
    assert state.neutral_planet_mask.tolist() == [False, False, False]


def test_parse_observation_tolerates_struct_like_comets_as_empty():
    comet_struct = SimpleNamespace(
        planet_ids=[0, 1],
        paths=[[0.0, 0.0], [1.0, 1.0]],
        path_index=0,
    )
    obs = {
        "player": 1,
        "step": 3,
        "planets": [
            [0, -1, 0.0, 0.0, 3, 20, 5],
            [1, 1, 1.0, 0.0, 2, 10, 3],
        ],
        "comets": comet_struct,
    }

    state = parse_observation(obs)
    assert state.planets.shape == (2, 7)
    assert state.comets.shape == (0, 7)
