import numpy as np
import pytest

from src.ow.features import FEATURE_COLUMNS, build_candidate_features, opponent_profile_features
from src.ow.state import parse_observation


def test_feature_columns_stable_order():
    assert FEATURE_COLUMNS == [
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


def test_build_candidate_features_empty_shapes():
    state = parse_observation({"player": 0, "step": 0, "planets": []})
    features, pairs = build_candidate_features(state)
    assert features.shape == (0, 16)
    assert pairs.shape == (0, 2)
    assert features.dtype == np.float64
    assert pairs.dtype == np.int64


def test_build_candidate_features_uses_planet_ids_and_sun_blocked():
    planets = [
        [101, 0, 20, 50, 2, 25, 3],
        [205, 1, 80, 50, 2, 10, 2],
        [309, -1, 20, 80, 2, 5, 1],
    ]
    state = parse_observation({"player": 0, "step": 10, "planets": planets})

    features, pairs = build_candidate_features(state)

    assert pairs.shape == (2, 2)
    assert set(map(tuple, pairs.tolist())) == {(101, 205), (101, 309)}

    blocked_idx = FEATURE_COLUMNS.index("sun_blocked")
    pair_rows = {tuple(pair): i for i, pair in enumerate(pairs.tolist())}
    assert features[pair_rows[(101, 205)], blocked_idx] == 1.0
    assert features[pair_rows[(101, 309)], blocked_idx] == 0.0


@pytest.mark.parametrize(
    "src_ships,src_prod,tgt_ships,tgt_prod",
    [
        (np.nan, 3, 10, 2),
        (25, np.nan, 10, 2),
        (25, 3, np.nan, 2),
        (25, 3, 10, np.nan),
        (-1, 3, 10, 2),
        (25, -1, 10, 2),
        (25, 3, -1, 2),
        (25, 3, 10, -1),
        (np.inf, 3, 10, 2),
        (25, 3, np.inf, 2),
    ],
)
def test_build_candidate_features_rejects_invalid_ship_prod_inputs(src_ships, src_prod, tgt_ships, tgt_prod):
    planets = [
        [101, 0, 20, 50, 2, src_ships, src_prod],
        [205, 1, 80, 50, 2, tgt_ships, tgt_prod],
    ]
    state = parse_observation({"player": 0, "step": 10, "planets": planets})
    with pytest.raises(ValueError, match="ships|prod|finite|nonnegative"):
        build_candidate_features(state)


def test_build_candidate_features_uses_min_send_guess_when_sendable_zero():
    planets = [
        [101, 0, 20, 50, 2, 1, 100],
        [205, 1, 80, 50, 2, 10, 2],
    ]
    state = parse_observation({"player": 0, "step": 10, "planets": planets})
    features, _ = build_candidate_features(state)
    assert features.shape[0] == 1
    assert np.all(np.isfinite(features))


def test_opponent_profile_detects_swarm_from_many_fleets():
    obs = {
        "player": 0,
        "step": 50,
        "angular_velocity": 0.03,
        "planets": [[0, 0, 10, 10, 2, 50, 3], [1, 1, 80, 80, 2, 50, 3]],
        "fleets": [[i, 1, 70 + i, 70, 0, 1, 2] for i in range(6)],
        "comets": [],
        "initial_planets": [[0, 0, 10, 10, 2, 10, 3], [1, 1, 80, 80, 2, 10, 3]],
    }
    state = parse_observation(obs)
    prof = opponent_profile_features(state)
    assert prof["enemy_fleet_count"] == 6
    assert prof["swarm_score"] > 0.5


def test_opponent_profile_zero_enemy_fleets_and_friendly_only_fleets():
    empty_state = parse_observation({"player": 0, "step": 0, "planets": [[1, 0, 10, 10, 2, 10, 2]], "fleets": []})
    prof_empty = opponent_profile_features(empty_state)
    assert prof_empty["enemy_fleet_count"] == 0
    assert prof_empty["enemy_mean_launch_ships"] == 0.0
    assert prof_empty["swarm_score"] == 0.0
    assert prof_empty["heavy_score"] == 0.0

    friendly_only = parse_observation(
        {
            "player": 0,
            "step": 0,
            "planets": [[1, 0, 10, 10, 2, 10, 2], [2, 1, 80, 80, 2, 10, 2]],
            "fleets": [[10, 0, 15, 15, 0, 5, 2], [11, 0, 16, 16, 0, 6, 2]],
        }
    )
    prof_friendly = opponent_profile_features(friendly_only)
    assert prof_friendly["enemy_fleet_count"] == 0
    assert prof_friendly["swarm_score"] == 0.0


def test_opponent_profile_heavy_enemy_fleets_and_malformed_rows_ignored():
    state = parse_observation(
        {
            "player": 0,
            "step": 10,
            "planets": [[1, 0, 10, 10, 2, 10, 2], [2, 1, 80, 80, 2, 10, 2]],
            "fleets": [
                [1, 1, 70, 70, 0, 2, 45],
                [2, 1, 71, 70, 0, 2, 55],
                [3, np.nan, 72, 70, 0, 2, 5],
                [4, 1, 73, 70, 0, 2, np.nan],
                [5, np.inf, 74, 70, 0, 2, 10],
            ],
        }
    )
    prof = opponent_profile_features(state)
    assert prof["enemy_fleet_count"] == 2
    assert prof["enemy_mean_launch_ships"] >= 50.0
    assert 0.0 <= prof["swarm_score"] <= 1.0
    assert prof["heavy_score"] > 0.7


def test_opponent_profile_huge_finite_ship_counts_stays_finite_and_bounded():
    state = parse_observation(
        {
            "player": 0,
            "step": 10,
            "planets": [[1, 0, 10, 10, 2, 10, 2], [2, 1, 80, 80, 2, 10, 2]],
            "fleets": [
                [1, 1, 70, 70, 0, 2, 1e308],
                [2, 1, 71, 70, 0, 2, 1e308],
            ],
        }
    )
    with np.errstate(all="raise"):
        prof = opponent_profile_features(state)
    assert np.isfinite(prof["enemy_fleet_count"])
    assert np.isfinite(prof["enemy_mean_launch_ships"])
    assert np.isfinite(prof["swarm_score"])
    assert np.isfinite(prof["heavy_score"])
    assert 0.0 <= prof["swarm_score"] <= 1.0
    assert 0.0 <= prof["heavy_score"] <= 1.0


def test_build_candidate_features_neutral_required_ignores_target_production():
    planets = [
        [101, 0, 10, 10, 2, 30, 3],
        [205, -1, 90, 10, 2, 10, 5],
    ]
    state = parse_observation({"player": 0, "step": 0, "planets": planets})
    features, _ = build_candidate_features(state)
    req_idx = FEATURE_COLUMNS.index("required")
    assert features.shape[0] == 1
    assert int(features[0, req_idx]) == 13


def test_build_candidate_features_threat_increases_source_keep_and_reduces_surplus():
    base_obs = {
        "player": 0,
        "step": 10,
        "planets": [[1, 0, 20, 20, 2, 50, 3], [2, 1, 85, 20, 2, 8, 2]],
        "fleets": [],
    }
    threat_obs = {
        **base_obs,
        "fleets": [[10, 1, 10, 20, 0.0, 0, 35]],
    }
    base_features, _ = build_candidate_features(parse_observation(base_obs))
    threat_features, _ = build_candidate_features(parse_observation(threat_obs))
    keep_idx = FEATURE_COLUMNS.index("source_keep")
    surplus_idx = FEATURE_COLUMNS.index("surplus_after_send")
    assert threat_features[0, keep_idx] > base_features[0, keep_idx]
    assert threat_features[0, surplus_idx] < base_features[0, surplus_idx]


def test_build_candidate_features_uses_precomputed_keeps(monkeypatch):
    import src.ow.features as features_mod

    planets = [
        [101, 0, 20, 50, 2, 25, 3],
        [205, 1, 80, 50, 2, 10, 2],
    ]
    state = parse_observation({"player": 0, "step": 10, "planets": planets})
    provided_keeps = np.array([7.0, 0.0], dtype=np.float64)

    def _should_not_call(*_args, **_kwargs):
        raise AssertionError("threat_keep_needed_by_planet should not be called when keeps is provided")

    monkeypatch.setattr(features_mod, "threat_keep_needed_by_planet", _should_not_call)
    features, _ = build_candidate_features(state, keeps=provided_keeps)
    keep_idx = FEATURE_COLUMNS.index("source_keep")
    surplus_idx = FEATURE_COLUMNS.index("surplus_after_send")
    required_idx = FEATURE_COLUMNS.index("required")

    assert features[0, keep_idx] == 7.0
    expected_sendable = 25.0 - 7.0
    assert np.isclose(features[0, surplus_idx], expected_sendable - features[0, required_idx])
