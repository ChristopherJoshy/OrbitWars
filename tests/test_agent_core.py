import time
from pathlib import Path

import numpy as np
import pytest

from main import agent
from src.ow.models_numpy import LinearModel, load_linear_model
from src.ow.features import FEATURE_COLUMNS
from src.ow.policy import choose_moves, score_candidates
from src.ow.state import parse_observation


def _obs(planets, player=0, step=0):
    return {"player": player, "step": step, "planets": planets, "fleets": [], "comets": []}


def test_agent_returns_legal_action_shape_and_bounds():
    obs = _obs(
        [
            [1, 0, 20, 20, 2, 40, 4],
            [2, 1, 80, 80, 2, 8, 2],
            [3, -1, 20, 80, 2, 5, 1],
        ]
    )
    actions = agent(obs)
    assert isinstance(actions, list)
    assert len(actions) <= 10

    planets = {int(p[0]): p for p in obs["planets"]}
    for move in actions:
        assert isinstance(move, list)
        assert len(move) == 3
        src, angle, ships = move
        src = int(src)
        assert src in planets
        assert int(planets[src][1]) == obs["player"]
        assert -2 * np.pi <= float(angle) <= 2 * np.pi
        assert 1 <= int(ships) <= int(planets[src][5])


def test_policy_returns_no_moves_without_owned_or_targets():
    no_owned = parse_observation(_obs([[1, 1, 20, 20, 2, 20, 2], [2, -1, 80, 80, 2, 10, 1]]))
    assert choose_moves(no_owned) == []

    no_targets = parse_observation(_obs([[1, 0, 20, 20, 2, 20, 2], [2, 0, 80, 80, 2, 10, 1]]))
    assert choose_moves(no_targets) == []


def test_policy_skips_sun_blocked_direct_target():
    state = parse_observation(_obs([[1, 0, 20, 50, 2, 50, 4], [2, 1, 80, 50, 2, 1, 1]]))
    moves = choose_moves(state)
    assert moves == []


def test_policy_skips_late_game_arrival_using_official_fleet_speed():
    # Reviewer edge case: short distance, send=3, step=498.
    # Official fleet_speed(3) ~= 1.317 so ceil(2 / 1.317) = 2 and arrival is at step 500.
    state = parse_observation(
        _obs(
            [
                [1, 0, 10, 10, 2, 10, 3],
                [2, -1, 12, 10, 2, 0, 1],
            ],
            step=498,
        )
    )
    moves = choose_moves(state)
    assert moves == []


def test_agent_impl_wrapper_handles_malformed_observation():
    assert agent({"player": 0, "planets": [[1, 0, 20]]}) == []


def test_policy_helpers_still_raise_on_bad_state_inputs():
    with pytest.raises(ValueError):
        parse_observation({"player": 0, "planets": [[1, 0, 20]]})


def test_linear_model_predict_empty_and_finite_clipped_outputs():
    model = LinearModel(weights=np.ones(16, dtype=np.float64) * 1000.0, bias=1000.0)
    out_empty = model.predict_proba(np.zeros((0, 16), dtype=np.float64))
    assert out_empty.shape == (0,)

    x = np.ones((2, 16), dtype=np.float64)
    out = model.predict_proba(x)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_load_linear_model_missing_file_fallback(tmp_path):
    model = load_linear_model(tmp_path / "missing_model.npz", n_features=16)
    x = np.zeros((3, 16), dtype=np.float64)
    out = model.predict_proba(x)
    assert np.allclose(out, 0.5)


def test_bootstrap_model_artifact_exists_and_has_nonzero_weights():
    model_path = Path("models") / "shot_validator.npz"
    assert model_path.exists()
    model = load_linear_model(model_path, n_features=len(FEATURE_COLUMNS), expected_feature_columns=FEATURE_COLUMNS)
    assert np.any(np.abs(model.weights) > 0.0)


@pytest.mark.parametrize(
    "payload",
    [
        {"bias": 0.0, "feature_columns": np.asarray(FEATURE_COLUMNS)},
        {"weights": np.zeros(len(FEATURE_COLUMNS)), "feature_columns": np.asarray(FEATURE_COLUMNS)},
        {"weights": np.zeros(len(FEATURE_COLUMNS)), "bias": 0.0},
        {"weights": np.zeros((len(FEATURE_COLUMNS), 1)), "bias": 0.0, "feature_columns": np.asarray(FEATURE_COLUMNS)},
        {"weights": np.concatenate([np.zeros(len(FEATURE_COLUMNS) - 1), np.asarray([np.nan])]), "bias": 0.0, "feature_columns": np.asarray(FEATURE_COLUMNS)},
        {"weights": np.zeros(len(FEATURE_COLUMNS)), "bias": np.asarray(np.nan), "feature_columns": np.asarray(FEATURE_COLUMNS)},
        {"weights": np.zeros(len(FEATURE_COLUMNS)), "bias": np.asarray([0.0]), "feature_columns": np.asarray(FEATURE_COLUMNS)},
        {"weights": np.zeros(len(FEATURE_COLUMNS)), "bias": 0.0, "feature_columns": np.asarray(FEATURE_COLUMNS[:-1])},
        {"weights": np.zeros(len(FEATURE_COLUMNS)), "bias": 0.0, "feature_columns": np.asarray(list(reversed(FEATURE_COLUMNS)))},
    ],
)
def test_load_linear_model_rejects_invalid_npz_payloads(tmp_path, payload):
    p = tmp_path / "bad_model.npz"
    np.savez(p, **payload)
    with pytest.raises(ValueError):
        load_linear_model(p, n_features=len(FEATURE_COLUMNS), expected_feature_columns=FEATURE_COLUMNS)


def test_loaded_model_weights_are_immutable_and_cached_default_is_immutable():
    from src.ow.agent_core import _MODEL_CACHE, _get_model

    model = load_linear_model(Path("models") / "shot_validator.npz", n_features=len(FEATURE_COLUMNS), expected_feature_columns=FEATURE_COLUMNS)
    assert model.weights.flags.writeable is False
    with pytest.raises(ValueError):
        model.weights[0] = 123.0

    _MODEL_CACHE.clear()
    cached = _get_model(None)
    assert cached.weights.flags.writeable is False


def test_choose_moves_does_not_aggregate_overspend_per_source():
    obs = _obs(
        [
            [1, 0, 10, 10, 3, 12, 3],
            [2, 1, 90, 10, 1, 5, 2],
            [3, -1, 90, 20, 1, 5, 2],
            [4, -1, 90, 30, 1, 5, 2],
        ]
    )
    state = parse_observation(obs)
    moves = choose_moves(state, config=__import__("src.ow.policy", fromlist=["PolicyConfig"]).PolicyConfig(max_moves=3, min_send=1, time_budget_ms=20.0))
    spent_by_src = {}
    for src, _ang, ships in moves:
        spent_by_src[int(src)] = spent_by_src.get(int(src), 0) + int(ships)
    assert spent_by_src.get(1, 0) <= 12


def test_policy_config_clamping_enforces_legal_max_and_min_send():
    obs = _obs(
        [
            [1, 0, 10, 10, 2, 30, 3],
            [2, -1, 90, 10, 1, 2, 1],
            [3, -1, 90, 20, 1, 2, 1],
            [4, -1, 90, 30, 1, 2, 1],
            [5, -1, 90, 40, 1, 2, 1],
        ]
    )
    state = parse_observation(obs)
    from src.ow.policy import PolicyConfig
    moves = choose_moves(state, config=PolicyConfig(max_moves=99, min_send=0, time_budget_ms=float("nan")))
    assert len(moves) <= 10
    if moves:
        assert all(int(m[2]) >= 1 for m in moves)


def test_agent_speed_100_calls_under_1s():
    obs = _obs(
        [
            [1, 0, 20, 20, 2, 30, 3],
            [2, 1, 80, 80, 2, 20, 3],
            [3, -1, 20, 80, 2, 8, 1],
            [4, 0, 80, 20, 2, 28, 2],
        ]
    )
    t0 = time.perf_counter()
    for _ in range(100):
        _ = agent(obs)
    dt = time.perf_counter() - t0
    assert dt < 1.0


def test_score_candidates_rewards_pressure_and_penalizes_sun_blocked_and_distance():
    f = np.zeros((2, len(FEATURE_COLUMNS)), dtype=np.float64)
    pressure_idx = FEATURE_COLUMNS.index("score_pressure")
    blocked_idx = FEATURE_COLUMNS.index("sun_blocked")
    distance_idx = FEATURE_COLUMNS.index("distance")

    # Strong candidate: high pressure, no sun block, short distance
    f[0, pressure_idx] = 5.0
    f[0, blocked_idx] = 0.0
    f[0, distance_idx] = 10.0

    # Weak candidate: low pressure, sun blocked, long distance
    f[1, pressure_idx] = 1.0
    f[1, blocked_idx] = 1.0
    f[1, distance_idx] = 90.0

    scores = score_candidates(f)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_choose_moves_skips_profile_call_when_no_low_surplus_candidates(monkeypatch):
    import src.ow.policy as policy_mod

    called = {"count": 0}

    def _fake_profile(_state):
        called["count"] += 1
        return {
            "enemy_fleet_count": 6.0,
            "enemy_mean_launch_ships": 2.0,
            "swarm_score": 1.0,
            "heavy_score": 0.0,
        }

    monkeypatch.setattr(policy_mod, "opponent_profile_features", _fake_profile)

    state = parse_observation(_obs([[1, 0, 10, 10, 2, 25, 3], [2, 1, 85, 85, 2, 6, 2]], step=40))
    features = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float64)
    features[0, FEATURE_COLUMNS.index("surplus_after_send")] = 20.0
    features[0, FEATURE_COLUMNS.index("eta")] = 1.0
    pairs = np.asarray([[1, 2]], dtype=np.int64)
    monkeypatch.setattr(policy_mod, "build_candidate_features", lambda _state, *, keeps=None: (features, pairs))

    _ = choose_moves(state)
    assert called["count"] == 0


def test_choose_moves_swarm_profile_changes_scores_passed_to_selection(monkeypatch):
    import src.ow.policy as policy_mod

    scores_seen = {}

    def _score_candidates(features, model_scores=None):
        return np.array([0.6, 0.4], dtype=np.float64)

    real_argsort = np.argsort

    def _argsort_spy(values):
        scores_seen["values"] = np.asarray(values, dtype=np.float64).copy()
        return real_argsort(values)

    monkeypatch.setattr(policy_mod, "score_candidates", _score_candidates)
    monkeypatch.setattr(policy_mod.np, "argsort", _argsort_spy)

    state = parse_observation(_obs([[1, 0, 10, 10, 2, 50, 3], [2, 1, 40, 10, 2, 20, 2], [3, 1, 50, 10, 2, 30, 2]], step=40))
    features = np.zeros((2, len(FEATURE_COLUMNS)), dtype=np.float64)
    features[:, FEATURE_COLUMNS.index("eta")] = 1.0
    features[:, FEATURE_COLUMNS.index("surplus_after_send")] = np.array([2.0, 8.0])
    pairs = np.asarray([[1, 2], [1, 3]], dtype=np.int64)
    monkeypatch.setattr(policy_mod, "build_candidate_features", lambda _state, *, keeps=None: (features, pairs))

    monkeypatch.setattr(
        policy_mod,
        "opponent_profile_features",
        lambda _state: {"enemy_fleet_count": 6.0, "enemy_mean_launch_ships": 2.0, "swarm_score": 1.0, "heavy_score": 0.0},
    )

    moves = choose_moves(state)
    assert "values" in scores_seen
    # choose_moves sorts by -scores, so captured values are negated
    assert np.allclose(scores_seen["values"], np.array([0.0, -0.4], dtype=np.float64))
    assert len(moves) <= 10
    total_by_src = {}
    planets = {int(p[0]): p for p in state.planets.tolist()}
    for src, angle, ships in moves:
        src_i = int(src)
        ships_i = int(ships)
        assert int(planets[src_i][1]) == state.player
        assert np.isfinite(angle)
        assert -np.pi <= float(angle) <= np.pi
        assert ships_i > 0
        total_by_src[src_i] = total_by_src.get(src_i, 0) + ships_i
    for src_i, spent in total_by_src.items():
        assert spent <= int(planets[src_i][5])


def test_opening_state_produces_at_least_one_legal_move():
    state = parse_observation(
        _obs(
            [
                [1, 0, 20, 20, 2, 10, 3],
                [2, -1, 35, 20, 2, 5, 1],
                [3, 1, 80, 80, 2, 30, 3],
            ],
            step=0,
        )
    )
    moves = choose_moves(state)
    assert len(moves) >= 1


def test_choose_moves_computes_threat_keep_once(monkeypatch):
    import src.ow.policy as policy_mod
    import src.ow.features as features_mod
    from src.ow.policy import PolicyConfig

    calls = {"count": 0}

    def _counting_keep(state, horizon):
        calls["count"] += 1
        return np.zeros((state.planets.shape[0],), dtype=np.float64)

    monkeypatch.setattr(policy_mod, "threat_keep_needed_by_planet", _counting_keep)
    monkeypatch.setattr(features_mod, "threat_keep_needed_by_planet", _counting_keep)

    state = parse_observation(
        _obs(
            [
                [1, 0, 10, 10, 2, 20, 3],
                [2, -1, 14, 10, 2, 1, 1],
            ],
            step=0,
        )
    )
    _ = choose_moves(state, config=PolicyConfig(time_budget_ms=1000.0))
    assert calls["count"] == 1


def test_policy_skips_unaffordable_target_no_partial_send(monkeypatch):
    import src.ow.policy as policy_mod

    state = parse_observation(_obs([[1, 0, 10, 10, 2, 10, 2], [2, 1, 20, 10, 2, 30, 3]], step=10))
    features = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float64)
    features[0, FEATURE_COLUMNS.index("eta")] = 2.0
    pairs = np.asarray([[1, 2]], dtype=np.int64)
    monkeypatch.setattr(policy_mod, "build_candidate_features", lambda _state, *, keeps=None: (features, pairs))
    moves = choose_moves(state)
    assert moves == []


def test_policy_does_not_emit_partial_move_for_fixed_point_unsafe_lower_regression(monkeypatch):
    import src.ow.policy as policy_mod

    state = parse_observation(_obs([[1, 0, 10, 10, 2, 4, 2], [2, 1, 11.5, 10, 2, 0, 1]], step=10))
    features = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float64)
    features[0, FEATURE_COLUMNS.index("distance")] = 1.5
    pairs = np.asarray([[1, 2]], dtype=np.int64)
    monkeypatch.setattr(policy_mod, "build_candidate_features", lambda _state, *, keeps=None: (features, pairs))

    moves = choose_moves(state)
    assert moves == []


def test_choose_moves_avoids_spending_threatened_source():
    planets = [[1, 0, 20, 20, 2, 50, 3], [2, -1, 20, 85, 2, 5, 1], [3, 1, 85, 20, 2, 20, 2]]
    no_threat = parse_observation(_obs(planets, step=10))
    with_threat = parse_observation(
        {
            "player": 0,
            "step": 10,
            "planets": planets,
            "fleets": [[10, 1, 10, 20, 0.0, 0, 45]],
            "comets": [],
        }
    )
    assert choose_moves(no_threat) != []
    assert choose_moves(with_threat) == []


def test_policy_suppresses_duplicate_target_selection(monkeypatch):
    import src.ow.policy as policy_mod

    state = parse_observation(
        _obs(
            [
                [1, 0, 10, 10, 2, 40, 3],
                [2, 0, 10, 20, 2, 40, 3],
                [3, -1, 60, 10, 2, 5, 1],
            ],
            step=20,
        )
    )
    features = np.zeros((2, len(FEATURE_COLUMNS)), dtype=np.float64)
    features[:, FEATURE_COLUMNS.index("eta")] = 1.0
    pairs = np.asarray([[1, 3], [2, 3]], dtype=np.int64)
    monkeypatch.setattr(policy_mod, "build_candidate_features", lambda _state, *, keeps=None: (features, pairs))
    monkeypatch.setattr(policy_mod, "score_candidates", lambda _f, _m=None: np.array([0.9, 0.8], dtype=np.float64))
    moves = choose_moves(state)
    assert len(moves) == 1
    assert int(moves[0][0]) == 1


def test_score_candidates_early_boosts_affordable_high_prod_neutral_over_closer_low_prod():
    f = np.zeros((2, len(FEATURE_COLUMNS)), dtype=np.float64)
    idx = FEATURE_COLUMNS.index
    f[:, idx("turn_norm")] = 0.02
    f[:, idx("turns_left_norm")] = 0.98
    f[:, idx("target_is_neutral")] = 1.0
    f[:, idx("target_is_enemy")] = 0.0
    f[:, idx("sun_blocked")] = 0.0
    f[:, idx("surplus_after_send")] = 6.0
    f[:, idx("score_pressure")] = 0.8
    f[0, idx("target_prod")] = 4.0
    f[0, idx("distance")] = 40.0
    f[1, idx("target_prod")] = 1.0
    f[1, idx("distance")] = 10.0
    s = score_candidates(f)
    assert s[0] > s[1]


def test_late_game_far_target_skipped_if_cannot_arrive_before_end():
    state = parse_observation(_obs([[1, 0, 10, 10, 2, 30, 3], [2, -1, 95, 95, 2, 1, 1]], step=499))
    moves = choose_moves(state)
    assert moves == []


def test_choose_moves_uses_predictive_angle_for_rotating_target(monkeypatch):
    import src.ow.policy as policy_mod

    state = parse_observation({"player": 0, "step": 10, "angular_velocity": 0.25, "planets": [[1, 0, 20, 20, 2, 80, 3], [2, -1, 60, 20, 1, 1, 1]], "fleets": [], "comets": []})
    features = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float64)
    features[0, FEATURE_COLUMNS.index("distance")] = 40.0
    pairs = np.asarray([[1, 2]], dtype=np.int64)
    monkeypatch.setattr(policy_mod, "build_candidate_features", lambda _state, *, keeps=None: (features, pairs))

    moves = choose_moves(state)
    assert len(moves) == 1
    current_center_angle = np.arctan2(0.0, 40.0)
    assert abs(float(moves[0][1]) - float(current_center_angle)) > 1e-3


def test_default_policy_can_emit_more_than_three_moves_and_never_above_ten():
    planets = [[1, 0, 10, 10, 2, 200, 3]]
    for i in range(2, 14):
        planets.append([i, -1, 70 + (i % 3), 10 + (i * 3 % 70), 1, 1, 1])
    state = parse_observation(_obs(planets, step=20))
    moves = choose_moves(state)
    assert len(moves) > 3
    assert len(moves) <= 10


def test_agent_impl_default_config_uses_policy_defaults_for_move_cap():
    from src.ow.agent_core import agent_impl

    planets = [[1, 0, 10, 10, 2, 200, 3]]
    for i in range(2, 14):
        planets.append([i, -1, 70 + (i % 3), 10 + (i * 3 % 70), 1, 1, 1])
    obs = _obs(planets, step=20)

    moves = agent_impl(obs, config=None)
    assert len(moves) > 3
    assert len(moves) <= 10

    owned = {int(p[0]): int(p[5]) for p in planets if int(p[1]) == 0}
    for src, _ang, ships in moves:
        assert int(src) in owned
        assert 1 <= int(ships) <= owned[int(src)]
