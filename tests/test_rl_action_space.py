import math

from src.ow.geometry import aim_with_prediction
from src.ow.rl.action_space import angle_to_target, build_action_candidates, candidate_to_env_action, estimate_candidate_mistakes


def test_build_action_candidates_includes_noop_and_owned_source_targets():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 30, 5],
            [1, -1, 20.0, 10.0, 2.0, 8, 3],
            [2, 1, 10.0, 20.0, 2.0, 12, 0],
        ],
    }
    candidates = build_action_candidates(obs, max_targets_per_source=4)
    assert candidates[0]["kind"] == "noop"
    launches = [c for c in candidates if c["kind"] == "launch"]
    assert {c["target_id"] for c in launches} == {1, 2}
    assert all(c["source_id"] == 0 for c in launches)
    assert all(c["ships"] >= 1 for c in launches)


def test_build_action_candidates_includes_own_planet_reinforcement_targets_by_default():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 50, 5],
            [1, 0, 30.0, 10.0, 2.0, 8, 4],
            [2, -1, 50.0, 10.0, 2.0, 12, 3],
        ],
    }

    launches = [c for c in build_action_candidates(obs, max_targets_per_source=4) if c["kind"] == "launch"]
    own_reinforcements = [c for c in launches if c["target_id"] == 1]

    assert own_reinforcements
    assert all(c["target_owner"] == 0 for c in own_reinforcements)
    assert all(c["target_is_own"] is True for c in own_reinforcements)
    assert all(c["source_id"] != c["target_id"] for c in own_reinforcements)


def test_angle_to_target_is_deterministic():
    assert angle_to_target((10.0, 10.0), (20.0, 10.0)) == 0.0


def test_candidate_to_env_action_maps_launch():
    c = {"kind": "launch", "source_id": 4, "angle": 1.25, "ships": 9}
    assert candidate_to_env_action(c) == [[4, 1.25, 9]]


def test_candidate_to_env_action_maps_macro_launch_bundle():
    c = {
        "kind": "macro",
        "launches": [
            {"source_id": 4, "angle": 1.25, "ships": 9},
            {"source_id": 6, "angle": -0.25, "ships": 12},
        ],
    }

    assert candidate_to_env_action(c) == [[4, 1.25, 9], [6, -0.25, 12]]


def test_estimate_candidate_mistakes_penalizes_sun_hit_before_target():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 50.0, 20.0, 3.0, 40, 2],
            [1, -1, 50.0, 85.0, 3.0, 10, 2],
        ],
    }
    candidate = {"kind": "launch", "source_id": 0, "target_id": 1, "angle": math.pi / 2, "ships": 12}
    mistakes = estimate_candidate_mistakes(obs, candidate)
    assert mistakes["sun_loss_ships"] == 12
    assert mistakes["wasted_ships"] == 12


def test_estimate_candidate_mistakes_allows_safe_target_hit():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 20.0, 20.0, 3.0, 40, 2],
            [1, -1, 35.0, 20.0, 3.0, 10, 2],
        ],
    }
    candidate = {"kind": "launch", "source_id": 0, "target_id": 1, "angle": 0.0, "ships": 9}
    mistakes = estimate_candidate_mistakes(obs, candidate)
    assert mistakes["sun_loss_ships"] == 0
    assert mistakes["wasted_ships"] == 0


def test_build_action_candidates_filters_known_sun_and_wrong_target_waste_by_default():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 50.0, 20.0, 3.0, 40, 2],
            [1, -1, 50.0, 85.0, 3.0, 10, 2],  # line from source crosses sun
            [2, -1, 80.0, 20.0, 3.0, 10, 2],  # safe horizontal launch
        ],
    }

    candidates = build_action_candidates(obs, max_targets_per_source=4)
    launches = [c for c in candidates if c["kind"] == "launch"]

    assert {c["target_id"] for c in launches} == {2}
    assert all(estimate_candidate_mistakes(obs, c)["wasted_ships"] == 0 for c in launches)


def test_build_action_candidates_can_keep_unsafe_launches_for_diagnostics():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 50.0, 20.0, 3.0, 40, 2],
            [1, -1, 50.0, 85.0, 3.0, 10, 2],
        ],
    }

    candidates = build_action_candidates(obs, max_targets_per_source=4, avoid_waste=False)

    assert any(c.get("target_id") == 1 for c in candidates if c["kind"] == "launch")


def test_build_action_candidates_adds_target_metadata_for_rl_features():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, -1, 40.0, 10.0, 2.0, 12, 6],
        ],
    }

    launch = next(c for c in build_action_candidates(obs) if c["kind"] == "launch")

    assert launch["source_ships"] == 50
    assert launch["target_owner"] == -1
    assert launch["target_ships"] == 12
    assert launch["target_production"] == 6
    assert launch["distance"] == 30.0


def test_build_action_candidates_offers_multiple_safe_send_fractions_for_rl_by_default():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, -1, 40.0, 10.0, 2.0, 12, 6],
        ],
    }

    launches = [c for c in build_action_candidates(obs) if c["kind"] == "launch"]

    assert {17, 27, 37}.issubset({c["ships"] for c in launches})
    assert {15, 45, 49}.issubset({c["ships"] for c in launches})
    assert all(estimate_candidate_mistakes(obs, c)["wasted_ships"] == 0 for c in launches)


def test_build_action_candidates_can_keep_single_fraction_for_diagnostics():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, -1, 40.0, 10.0, 2.0, 12, 6],
        ],
    }

    launches = [c for c in build_action_candidates(obs, send_fraction=0.55) if c["kind"] == "launch"]

    assert {c["ships"] for c in launches} == {27}


def test_build_action_candidates_adds_exact_capture_count_when_fractions_miss_it():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, -1, 40.0, 10.0, 2.0, 18, 3],
        ],
    }

    launches = [c for c in build_action_candidates(obs, max_targets_per_source=4, avoid_waste=False) if c["kind"] == "launch"]
    required = {c["required_capture_ships"] for c in launches if c["target_id"] == 1}
    ship_counts = {c["ships"] for c in launches if c["target_id"] == 1}

    assert required <= ship_counts
    assert required.isdisjoint({17, 27, 37})


def test_build_action_candidates_adds_safe_macro_actions_without_overspending_sources():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 60, 4],
            [1, 0, 10.0, 35.0, 3.0, 60, 4],
            [2, -1, 35.0, 10.0, 2.0, 12, 2],
            [3, -1, 35.0, 35.0, 2.0, 12, 2],
        ],
    }

    macros = [c for c in build_action_candidates(obs, max_targets_per_source=3, avoid_waste=False) if c["kind"] == "macro"]

    assert macros
    for macro in macros:
        action = candidate_to_env_action(macro)
        assert len(action) >= 2
        by_source = {}
        for source_id, _angle, ships in action:
            by_source[source_id] = by_source.get(source_id, 0) + ships
        assert all(ships < 60 for ships in by_source.values())


def test_build_action_candidates_adds_global_state_metadata_for_rl_context():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, 1, 40.0, 10.0, 2.0, 30, 3],
            [2, -1, 80.0, 10.0, 2.0, 12, 6],
        ],
    }

    launch = next(c for c in build_action_candidates(obs) if c["kind"] == "launch")

    assert launch["my_planet_count"] == 1
    assert launch["enemy_planet_count"] == 1
    assert launch["neutral_planet_count"] == 1
    assert launch["my_total_ships"] == 50
    assert launch["enemy_total_ships"] == 30
    assert launch["my_production"] == 4
    assert launch["enemy_production"] == 3


def test_build_action_candidates_marks_target_relation_metadata_for_features():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, 0, 40.0, 10.0, 2.0, 12, 6],
            [2, 1, 70.0, 10.0, 2.0, 20, 0],
            [3, -1, 90.0, 10.0, 2.0, 8, 2],
        ],
    }

    launches = [c for c in build_action_candidates(obs, avoid_waste=False) if c["kind"] == "launch"]
    by_target = {c["target_id"]: c for c in launches}

    assert by_target[1]["target_is_own"] is True
    assert by_target[1]["target_is_enemy"] is False
    assert by_target[1]["target_is_neutral"] is False
    assert by_target[2]["target_is_enemy"] is True
    assert by_target[3]["target_is_neutral"] is True


def test_build_action_candidates_filters_futile_enemy_attacks_but_keeps_reinforcement():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 40, 4],
            [1, 0, 10.0, 25.0, 2.0, 12, 3],
            [2, 1, 40.0, 12.0, 2.0, 1600, 5],
        ],
    }

    launches = [
        c
        for c in build_action_candidates(obs, max_targets_per_source=4, avoid_waste=False)
        if c["kind"] == "launch"
    ]

    assert {c["ships"] for c in launches if c["target_id"] == 2} == set()
    assert {c["ships"] for c in launches if c["target_id"] == 1} == {14, 22, 30}


def test_build_action_candidates_filters_neutral_undercapture_launches():
    weak_source_obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 20, 4],
            [1, -1, 35.0, 10.0, 2.0, 21, 3],
        ],
    }
    strong_source_obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 40, 4],
            [1, -1, 35.0, 10.0, 2.0, 21, 3],
        ],
    }

    weak_launches = [
        c
        for c in build_action_candidates(weak_source_obs, max_targets_per_source=4, avoid_waste=False)
        if c["kind"] == "launch"
    ]
    strong_launches = [
        c
        for c in build_action_candidates(strong_source_obs, max_targets_per_source=4, avoid_waste=False)
        if c["kind"] == "launch"
    ]

    assert weak_launches == []
    strong_neutral_ship_counts = {c["ships"] for c in strong_launches if c["target_id"] == 1}
    assert 30 in strong_neutral_ship_counts
    assert min(strong_neutral_ship_counts) >= 24


def test_build_action_candidates_keeps_material_enemy_attrition_only():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 40, 4],
            [1, 1, 35.0, 10.0, 2.0, 100, 3],
        ],
    }

    launches = [
        c
        for c in build_action_candidates(
            obs,
            max_targets_per_source=4,
            avoid_waste=False,
            send_fractions=(0.25, 0.75),
        )
        if c["kind"] == "launch"
    ]

    assert {c["ships"] for c in launches if c["target_id"] == 1} == {30}


def test_build_action_candidates_leads_rotating_targets_with_prediction():
    obs = {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 10.0, 80.0, 3.0, 80, 4],
            [1, -1, 40.0, 80.0, 2.0, 10, 0],
        ],
    }

    launch = next(
        c
        for c in build_action_candidates(obs, max_targets_per_source=2, avoid_waste=False, send_fraction=0.75)
        if c["kind"] == "launch"
    )
    expected_angle, expected_eta, expected_distance = aim_with_prediction(10.0, 80.0, obs["planets"][1], ships=60, angular_velocity=0.2)

    assert not math.isclose(expected_angle, angle_to_target((10.0, 80.0), (40.0, 80.0)))
    assert math.isclose(launch["angle"], expected_angle, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(launch["eta"], expected_eta, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(launch["distance"], expected_distance, rel_tol=1e-12, abs_tol=1e-12)


def test_estimate_candidate_mistakes_uses_predicted_intercept_segment_not_current_target_center():
    obs = {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 10.0, 80.0, 3.0, 80, 4],
            [1, -1, 40.0, 80.0, 2.0, 10, 0],
        ],
    }
    launch = next(
        c
        for c in build_action_candidates(obs, max_targets_per_source=2, avoid_waste=False, send_fraction=0.75)
        if c["kind"] == "launch"
    )

    assert estimate_candidate_mistakes(obs, launch)["wasted_ships"] == 0


def test_build_action_candidates_requires_eta_capture_ships_for_enemy_production():
    weak_obs = {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 30, 4],
            [1, 1, 60.0, 50.0, 2.0, 20, 10],
        ],
    }
    strong_obs = {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 200, 4],
            [1, 1, 60.0, 50.0, 2.0, 20, 10],
        ],
    }

    weak_launches = [
        c
        for c in build_action_candidates(weak_obs, max_targets_per_source=2, avoid_waste=False, send_fraction=0.75)
        if c["kind"] == "launch"
    ]
    strong_launches = [
        c
        for c in build_action_candidates(strong_obs, max_targets_per_source=2, avoid_waste=False, send_fraction=0.75)
        if c["kind"] == "launch"
    ]

    assert weak_launches == []
    assert {c["ships"] for c in strong_launches if c["target_id"] == 1} == {150}


def test_build_action_candidates_requires_eta_capture_margin_for_neutral_targets():
    weak_obs = {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 16, 4],
            [1, -1, 60.0, 50.0, 2.0, 10, 0],
        ],
    }
    strong_obs = {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 18, 4],
            [1, -1, 60.0, 50.0, 2.0, 10, 0],
        ],
    }

    weak_launches = [
        c
        for c in build_action_candidates(
            weak_obs,
            max_targets_per_source=2,
            avoid_waste=False,
            send_fraction=0.75,
        )
        if c["kind"] == "launch"
    ]
    strong_launches = [
        c
        for c in build_action_candidates(
            strong_obs,
            max_targets_per_source=2,
            avoid_waste=False,
            send_fraction=0.75,
        )
        if c["kind"] == "launch"
    ]

    assert weak_launches == []
    assert {c["ships"] for c in strong_launches if c["target_id"] == 1} == {13}
