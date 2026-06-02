import random

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from kaggle_environments.envs.orbit_wars import orbit_wars as official
from kaggle_environments.utils import Struct

from src.ow.geometry import aim_with_prediction, distance_matrix, fleet_speed, predict_planet_position, segment_hits_circle
from src.ow.rl.action_space import build_action_candidates, candidate_to_env_action
from src.ow.state import parse_observation
from src.ow.timeline import estimate_required_capture_ships, estimate_required_capture_ships_fixed_point, keep_needed_by_planet
from src.ow.rl import jax_env, jax_policy
from tools.train_rl_phase1 import featurize


_JAX_FEATURE_DIM = 32


def _jax_feature_vector(values):
    vector = jnp.asarray(values, dtype=jnp.float64)
    if vector.shape[0] == _JAX_FEATURE_DIM:
        return vector
    return jnp.pad(vector, (0, _JAX_FEATURE_DIM - vector.shape[0]))


def _jax_actor_w(first_weight=5.0):
    return jnp.zeros((_JAX_FEATURE_DIM,), dtype=jnp.float64).at[0].set(first_weight)


def _jax_value_w(start=-0.05, stop=0.05):
    return jnp.linspace(start, stop, _JAX_FEATURE_DIM, dtype=jnp.float64)


def _struct(**kwargs):
    obj = Struct()
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj


def _run_official_tick(
    planets,
    fleets,
    actions=None,
    *,
    players=2,
    step=1,
    episode_steps=100,
    angular_velocity=0.0,
    initial_planets=None,
    comets=None,
    comet_planet_ids=None,
    seed=0,
    comet_speed=1.0,
):
    if actions is None:
        actions = [[] for _ in range(players)]
    if initial_planets is None:
        initial_planets = planets
    if comets is None:
        comets = []
    if comet_planet_ids is None:
        comet_planet_ids = []
    state = [
        _struct(
            observation=_struct(
                remainingOverageTime=60.0,
                step=step,
                planets=[p.copy() for p in planets],
                initial_planets=[p.copy() for p in initial_planets],
                fleets=[f.copy() for f in fleets],
                player=player,
                angular_velocity=angular_velocity,
                next_fleet_id=len(fleets),
                comets=[{"planet_ids": g["planet_ids"].copy(), "paths": [[xy.copy() for xy in path] for path in g["paths"]], "path_index": g["path_index"]} for g in comets],
                comet_planet_ids=comet_planet_ids.copy(),
            ),
            action=actions[player],
            reward=0,
            status="ACTIVE",
        )
        for player in range(players)
    ]
    env = _struct(
        done=False,
        configuration=_struct(shipSpeed=6.0, episodeSteps=episode_steps, cometSpeed=comet_speed),
        info={"seed": seed},
    )
    official.interpreter(state, env)
    return state[0].observation, [s.reward for s in state], [s.status for s in state]


def _active_fleets(result):
    return np.asarray(result.fleets)[np.asarray(result.fleet_active, dtype=bool)]


def _active_planets(result):
    return np.asarray(result.planets)[np.asarray(result.planet_active, dtype=bool)]


def _run_official_reset(*, players=2, seed=0, episode_steps=500):
    state = [
        _struct(
            observation=_struct(remainingOverageTime=60.0, step=0),
            action=[],
            reward=0,
            status="INACTIVE",
        )
        for _ in range(players)
    ]
    env = _struct(
        done=True,
        configuration=_struct(seed=seed, shipSpeed=6.0, episodeSteps=episode_steps, cometSpeed=4.0),
        info={},
    )
    official.interpreter(state, env)
    return state[0].observation, env.info["seed"]


def test_jax_parse_observation_matches_numpy_state_masks_and_arrays():
    obs = {
        "player": 1,
        "step": 7,
        "angular_velocity": 0.25,
        "planets": [
            [0, -1, 0.0, 0.0, 3, 20, 5],
            [1, 1, 1.0, 0.0, 2, 10, 3],
            [2, 0, 2.0, 0.0, 2, 11, 4],
        ],
        "fleets": [[1, 0, 1, 5, 4, 0, 9]],
        "comets": [],
    }
    expected = parse_observation(obs)

    actual = jax_env.parse_observation(obs)

    assert int(actual.player) == expected.player
    assert int(actual.step) == expected.step
    assert float(actual.angular_velocity) == expected.angular_velocity
    assert np.allclose(np.asarray(actual.planets), expected.planets)
    assert np.allclose(np.asarray(actual.fleets), expected.fleets)
    assert np.asarray(actual.comets).shape == expected.comets.shape
    assert np.asarray(actual.my_planet_mask).tolist() == expected.my_planet_mask.tolist()
    assert np.asarray(actual.enemy_planet_mask).tolist() == expected.enemy_planet_mask.tolist()
    assert np.asarray(actual.neutral_planet_mask).tolist() == expected.neutral_planet_mask.tolist()


def test_jax_geometry_matches_numpy_reference_and_vectorizes():
    ships = np.array([1.0, 10.0, 500.0, 1000.0], dtype=np.float64)
    expected_speeds = np.array([fleet_speed(x) for x in ships], dtype=np.float64)

    actual_speeds = jax.vmap(jax_env.fleet_speed)(jnp.asarray(ships))

    assert np.allclose(np.asarray(actual_speeds), expected_speeds, rtol=1e-12, atol=1e-12)

    a = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[0.0, 0.0], [6.0, 8.0]], dtype=np.float64)
    assert np.allclose(np.asarray(jax_env.distance_matrix(a, b)), distance_matrix(a, b))

    assert bool(jax_env.segment_hits_circle(jnp.array([-10.0, 0.0]), jnp.array([10.0, 0.0]), jnp.array([0.0, 0.0]), 2.0)) is True
    assert bool(jax_env.segment_hits_circle(jnp.array([10.0, 10.0]), jnp.array([12.0, 12.0]), jnp.array([0.0, 0.0]), 2.0)) is False

    planet = np.array([1, 0, 60.0, 50.0, 1.0, 5.0, 1.0], dtype=np.float64)
    expected_x, expected_y = predict_planet_position(planet, turns=5.0, angular_velocity=0.2)
    actual_x, actual_y = jax_env.predict_planet_position(jnp.asarray(planet), turns=5.0, angular_velocity=0.2)
    assert np.allclose([float(actual_x), float(actual_y)], [expected_x, expected_y])


def test_jax_timeline_capture_and_reserve_helpers_match_numpy_reference():
    assert int(jax_env.estimate_required_capture_ships(10, 2, 3, margin=2)) == estimate_required_capture_ships(10, 2, 3, margin=2)

    planets = np.array(
        [
            [0, 0, 10, 10, 2, 30, 3],
            [1, 1, 20, 10, 2, 40, 4],
            [2, -1, 30, 10, 2, 5, 1],
        ],
        dtype=np.float64,
    )

    expected = keep_needed_by_planet(planets, player=0, horizon=20)
    actual = jax_env.keep_needed_by_planet(jnp.asarray(planets), player=0, horizon=20)
    assert np.array_equal(np.asarray(actual), expected)


def test_jax_core_helpers_are_jittable_for_tpu_hot_path_foundation():
    planet = jnp.array([1, 0, 60.0, 50.0, 1.0, 5.0, 1.0], dtype=jnp.float64)

    @jax.jit
    def _compiled(planet_row):
        x, y = jax_env.predict_planet_position(planet_row, turns=5.0, angular_velocity=0.2)
        speed = jax_env.fleet_speed(500.0)
        required = jax_env.estimate_required_capture_ships(10.0, 2.0, 3.0, margin=2.0)
        return x + y + speed + required

    out = _compiled(planet)

    assert np.isfinite(float(out))


def test_jax_aim_prediction_and_fixed_point_capture_match_numpy_reference():
    target = np.array([1, -1, 60.0, 50.0, 2.0, 12.0, 3.0], dtype=np.float64)
    expected_angle, expected_eta, expected_distance = aim_with_prediction(
        10.0,
        10.0,
        target,
        ships=75.0,
        angular_velocity=0.03,
    )

    actual_angle, actual_eta, actual_distance = jax_env.aim_with_prediction(
        10.0,
        10.0,
        jnp.asarray(target),
        ships=75.0,
        angular_velocity=0.03,
    )

    assert np.allclose(
        [float(actual_angle), float(actual_eta), float(actual_distance)],
        [expected_angle, expected_eta, expected_distance],
        rtol=1e-10,
        atol=1e-10,
    )

    assert int(jax_env.estimate_required_capture_ships_fixed_point(20.0, 4.0, 35.0, margin=2.0)) == estimate_required_capture_ships_fixed_point(20.0, 4.0, 35.0, margin=2.0)


def test_jax_official_collision_helpers_match_orbit_wars_engine():
    p = (50.0, 50.0)
    v = (40.0, 50.0)
    w = (60.0, 50.0)
    assert np.isclose(float(jax_env.point_to_segment_distance(jnp.asarray(p), jnp.asarray(v), jnp.asarray(w))), official.point_to_segment_distance(p, v, w))

    moving_fleet_start = (40.0, 50.0)
    moving_fleet_end = (60.0, 50.0)
    planet_start = (50.0, 48.0)
    planet_end = (50.0, 52.0)
    assert bool(
        jax_env.swept_pair_hit(
            jnp.asarray(moving_fleet_start),
            jnp.asarray(moving_fleet_end),
            jnp.asarray(planet_start),
            jnp.asarray(planet_end),
            1.0,
        )
    ) is official.swept_pair_hit(moving_fleet_start, moving_fleet_end, planet_start, planet_end, 1.0)

    miss_fleet_start = (10.0, 10.0)
    miss_fleet_end = (20.0, 10.0)
    assert bool(
        jax_env.swept_pair_hit(
            jnp.asarray(miss_fleet_start),
            jnp.asarray(miss_fleet_end),
            jnp.asarray(planet_start),
            jnp.asarray(planet_end),
            1.0,
        )
    ) is official.swept_pair_hit(miss_fleet_start, miss_fleet_end, planet_start, planet_end, 1.0)


def test_jax_official_collision_helpers_are_jittable():
    @jax.jit
    def _compiled():
        d = jax_env.point_to_segment_distance(
            jnp.array([50.0, 50.0]),
            jnp.array([40.0, 50.0]),
            jnp.array([60.0, 50.0]),
        )
        hit = jax_env.swept_pair_hit(
            jnp.array([40.0, 50.0]),
            jnp.array([60.0, 50.0]),
            jnp.array([50.0, 48.0]),
            jnp.array([50.0, 52.0]),
            1.0,
        )
        return d, hit

    distance_out, hit_out = _compiled()
    assert np.isfinite(float(distance_out))
    assert bool(hit_out) is True


def test_jax_step_no_comets_matches_official_launch_production_and_movement():
    planets = [[0, 0, 10.0, 10.0, 2.0, 50.0, 3.0], [1, -1, 95.0, 95.0, 2.0, 0.0, 0.0]]
    fleets = []
    actions = [[[0, 0.0, 10.0]], []]
    official_obs, official_rewards, official_statuses = _run_official_tick(planets, fleets, actions=actions)

    state = jax_env.JaxTickState.from_arrays(
        planets=jnp.asarray(planets, dtype=jnp.float64),
        fleets=jnp.zeros((4, 7), dtype=jnp.float64),
        fleet_active=jnp.zeros((4,), dtype=jnp.bool_),
        initial_planets=jnp.asarray(planets, dtype=jnp.float64),
        next_fleet_id=0,
        step=1,
        angular_velocity=0.0,
    )
    action_array = jnp.asarray([[[0.0, 0.0, 10.0]], [[0.0, 0.0, 0.0]]], dtype=jnp.float64)
    action_active = jnp.asarray([[True], [False]])

    actual = jax_env.step_no_comets(state, action_array, action_active, episode_steps=100, ship_speed=6.0)

    assert np.allclose(np.asarray(actual.planets), np.asarray(official_obs.planets))
    assert np.allclose(_active_fleets(actual), np.asarray(official_obs.fleets))
    assert int(actual.next_fleet_id) == official_obs.next_fleet_id
    assert np.asarray(actual.rewards).tolist() == official_rewards
    assert bool(actual.done) == all(status == "DONE" for status in official_statuses)


def test_jax_step_no_comets_matches_official_combat_capture_resolution():
    planets = [[0, 0, 50.0, 50.0, 1.0, 3.0, 2.0]]
    fleets = [[0, 1, 49.0, 50.0, 0.0, 1.0, 6.0]]
    official_obs, _official_rewards, _official_statuses = _run_official_tick(planets, fleets)

    state = jax_env.JaxTickState.from_arrays(
        planets=jnp.asarray(planets, dtype=jnp.float64),
        fleets=jnp.asarray(fleets + [[0, 0, 0, 0, 0, 0, 0]], dtype=jnp.float64),
        fleet_active=jnp.asarray([True, False]),
        initial_planets=jnp.asarray(planets, dtype=jnp.float64),
        next_fleet_id=1,
        step=1,
        angular_velocity=0.0,
    )
    actual = jax_env.step_no_comets(
        state,
        jnp.zeros((2, 1, 3), dtype=jnp.float64),
        jnp.zeros((2, 1), dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    assert np.allclose(np.asarray(actual.planets), np.asarray(official_obs.planets))
    assert len(_active_fleets(actual)) == len(official_obs.fleets)


def test_jax_step_with_comets_matches_official_first_placement_collision_skip():
    planets = [[0, -1, 90.0, 90.0, 2.0, 0.0, 0.0], [99, -1, -99.0, -99.0, 1.0, 5.0, 1.0]]
    fleets = [[0, 0, 49.0, 70.0, 0.0, 0.0, 1.0]]
    comet_paths = [[[50.0, 70.0], [51.0, 70.0]]]
    comets = [{"planet_ids": [99], "paths": comet_paths, "path_index": -1}]
    official_obs, _official_rewards, _official_statuses = _run_official_tick(
        planets,
        fleets,
        comets=comets,
        comet_planet_ids=[99],
    )

    state = jax_env.JaxCometTickState.from_arrays(
        planets=jnp.asarray(planets, dtype=jnp.float64),
        fleets=jnp.asarray(fleets, dtype=jnp.float64),
        fleet_active=jnp.asarray([True]),
        initial_planets=jnp.asarray(planets, dtype=jnp.float64),
        next_fleet_id=1,
        step=1,
        angular_velocity=0.0,
        comet_planet_ids=jnp.asarray([99], dtype=jnp.int64),
        comet_paths=jnp.asarray(comet_paths, dtype=jnp.float64),
        comet_path_index=-1,
        comet_path_len=2,
    )

    actual = jax_env.step_with_comets(
        state,
        jnp.zeros((2, 1, 3), dtype=jnp.float64),
        jnp.zeros((2, 1), dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    assert np.allclose(np.asarray(actual.planets), np.asarray(official_obs.planets))
    assert np.allclose(_active_fleets(actual), np.asarray(official_obs.fleets))


def test_jax_step_with_comets_matches_official_moving_comet_collision():
    planets = [[0, -1, 90.0, 90.0, 2.0, 0.0, 0.0], [99, -1, 50.0, 70.0, 1.0, 5.0, 1.0]]
    fleets = [[0, 0, 50.0, 70.0, 0.0, 0.0, 1.0]]
    comet_paths = [[[50.0, 70.0], [51.0, 70.0]]]
    comets = [{"planet_ids": [99], "paths": comet_paths, "path_index": 0}]
    official_obs, _official_rewards, _official_statuses = _run_official_tick(
        planets,
        fleets,
        comets=comets,
        comet_planet_ids=[99],
    )

    state = jax_env.JaxCometTickState.from_arrays(
        planets=jnp.asarray(planets, dtype=jnp.float64),
        fleets=jnp.asarray(fleets, dtype=jnp.float64),
        fleet_active=jnp.asarray([True]),
        initial_planets=jnp.asarray(planets, dtype=jnp.float64),
        next_fleet_id=1,
        step=1,
        angular_velocity=0.0,
        comet_planet_ids=jnp.asarray([99], dtype=jnp.int64),
        comet_paths=jnp.asarray(comet_paths, dtype=jnp.float64),
        comet_path_index=0,
        comet_path_len=2,
    )

    actual = jax_env.step_with_comets(
        state,
        jnp.zeros((2, 1, 3), dtype=jnp.float64),
        jnp.zeros((2, 1), dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    assert np.allclose(np.asarray(actual.planets), np.asarray(official_obs.planets))
    assert len(_active_fleets(actual)) == len(official_obs.fleets)


def test_jax_step_with_comets_matches_official_expiry_before_combat():
    planets = [[0, -1, 90.0, 90.0, 2.0, 0.0, 0.0], [99, -1, 51.0, 70.0, 1.0, 5.0, 1.0]]
    fleets = [[0, 0, 50.0, 70.0, 0.0, 0.0, 1.0]]
    comet_paths = [[[50.0, 70.0], [51.0, 70.0]]]
    comets = [{"planet_ids": [99], "paths": comet_paths, "path_index": 1}]
    official_obs, _official_rewards, _official_statuses = _run_official_tick(
        planets,
        fleets,
        comets=comets,
        comet_planet_ids=[99],
    )

    state = jax_env.JaxCometTickState.from_arrays(
        planets=jnp.asarray(planets, dtype=jnp.float64),
        planet_active=jnp.asarray([True, True]),
        fleets=jnp.asarray(fleets, dtype=jnp.float64),
        fleet_active=jnp.asarray([True]),
        initial_planets=jnp.asarray(planets, dtype=jnp.float64),
        next_fleet_id=1,
        step=1,
        angular_velocity=0.0,
        comet_planet_ids=jnp.asarray([99], dtype=jnp.int64),
        comet_paths=jnp.asarray(comet_paths, dtype=jnp.float64),
        comet_path_index=1,
        comet_path_len=2,
    )
    actual = jax_env.step_with_comets(
        state,
        jnp.zeros((2, 1, 3), dtype=jnp.float64),
        jnp.zeros((2, 1), dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(actual), np.asarray(official_obs.planets))
    assert len(_active_fleets(actual)) == len(official_obs.fleets)


def test_jax_step_with_comets_matches_official_precomputed_spawn_tick():
    planets = [[0, -1, 90.0, 90.0, 2.0, 0.0, 0.0]]
    comet_paths = [
        [[20.0, 70.0], [21.0, 70.0]],
        [[80.0, 70.0], [79.0, 70.0]],
        [[20.0, 30.0], [21.0, 30.0]],
        [[80.0, 30.0], [79.0, 30.0]],
    ]
    original_generate = official.generate_comet_paths
    official.generate_comet_paths = lambda *args, **kwargs: comet_paths
    try:
        official_obs, _official_rewards, _official_statuses = _run_official_tick(planets, [], step=49)
    finally:
        official.generate_comet_paths = original_generate
    rng = random.Random("orbit_wars-comet-0-50")
    spawn_ships = min(rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99))

    state = jax_env.JaxCometTickState.from_arrays(
        planets=jnp.asarray(planets + [[0, 0, 0, 0, 0, 0, 0] for _ in range(4)], dtype=jnp.float64),
        planet_active=jnp.asarray([True, False, False, False, False]),
        fleets=jnp.zeros((1, 7), dtype=jnp.float64),
        fleet_active=jnp.asarray([False]),
        initial_planets=jnp.asarray(planets + [[0, 0, 0, 0, 0, 0, 0] for _ in range(4)], dtype=jnp.float64),
        next_fleet_id=0,
        step=49,
        angular_velocity=0.0,
        comet_planet_ids=jnp.zeros((4,), dtype=jnp.int64),
        comet_paths=jnp.zeros((4, 2, 2), dtype=jnp.float64),
        comet_path_index=-1,
        comet_path_len=0,
        spawn_active=True,
        spawn_step=50,
        spawn_paths=jnp.asarray(comet_paths, dtype=jnp.float64),
        spawn_path_len=2,
        spawn_ships=spawn_ships,
    )
    actual = jax_env.step_with_comets(
        state,
        jnp.zeros((2, 1, 3), dtype=jnp.float64),
        jnp.zeros((2, 1), dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(actual), np.asarray(official_obs.planets))


def test_jax_env_state_from_official_reset_pads_arrays_and_precomputes_comet_schedule():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)

    assert np.allclose(_active_planets(state), np.asarray(obs.planets))
    assert np.allclose(np.asarray(state.initial_planets)[np.asarray(state.planet_active, dtype=bool)], np.asarray(obs.initial_planets))
    assert np.asarray(state.fleet_active).tolist() == [False] * 8
    assert int(state.next_fleet_id) == obs.next_fleet_id
    assert int(state.step) == obs.step
    assert np.isclose(float(state.angular_velocity), obs.angular_velocity)

    rng = random.Random(f"orbit_wars-comet-{episode_seed}-50")
    expected_paths = official.generate_comet_paths(obs.initial_planets, obs.angular_velocity, 50, [], 4.0, rng=rng)
    expected_ships = min(rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99))
    assert bool(np.asarray(state.spawn_available)[0]) == (expected_paths is not None)
    if expected_paths is not None:
        expected_len = len(expected_paths[0])
        assert int(np.asarray(state.spawn_steps)[0]) == 50
        assert int(np.asarray(state.spawn_path_lens)[0]) == expected_len
        assert np.allclose(np.asarray(state.spawn_paths)[0, :, :expected_len, :], np.asarray(expected_paths))
        assert float(np.asarray(state.spawn_ships)[0]) == expected_ships


def test_jax_observation_from_env_state_matches_official_reset_for_policy_candidates():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)

    actual = jax_env.observation_from_env_state(state, player=0)
    expected = {
        "player": 0,
        "step": obs.step,
        "angular_velocity": obs.angular_velocity,
        "planets": obs.planets,
        "fleets": obs.fleets,
        "initial_planets": obs.initial_planets,
        "next_fleet_id": obs.next_fleet_id,
        "comets": obs.comets,
        "comet_planet_ids": obs.comet_planet_ids,
    }

    assert actual["player"] == expected["player"]
    assert actual["step"] == expected["step"]
    assert np.isclose(actual["angular_velocity"], expected["angular_velocity"])
    assert np.allclose(np.asarray(actual["planets"]), np.asarray(expected["planets"]))
    assert np.allclose(np.asarray(actual["initial_planets"]), np.asarray(expected["initial_planets"]))
    assert actual["fleets"] == expected["fleets"]
    assert actual["next_fleet_id"] == expected["next_fleet_id"]
    assert actual["comets"] == expected["comets"]
    assert actual["comet_planet_ids"] == expected["comet_planet_ids"]
    assert build_action_candidates(actual) == build_action_candidates(expected)


def test_jax_action_candidates_from_env_state_match_python_candidates_and_features():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    py_obs = jax_env.observation_from_env_state(state, player=0)
    expected_candidates = build_action_candidates(
        py_obs,
        max_targets_per_source=3,
        avoid_waste=False,
        send_fractions=(0.35, 0.55, 0.75),
    )

    actual = jax_env.action_candidates_from_env_state(
        state,
        player=0,
        max_targets_per_source=3,
        send_fractions=jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64),
    )
    active = np.asarray(actual.active, dtype=bool)

    expected_actions = np.asarray(
        [(candidate_to_env_action(c) or [[-1.0, 0.0, 0.0]])[0] for c in expected_candidates],
        dtype=np.float64,
    )
    expected_features = np.stack([featurize(c) for c in expected_candidates], axis=0)

    assert np.allclose(np.asarray(actual.env_actions)[active], expected_actions)
    assert np.allclose(np.asarray(actual.features)[active], expected_features)


def test_jax_action_candidates_match_python_futile_non_owned_target_filter():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 40.0, 4.0],
            [1, -1, 35.0, 10.0, 2.0, 21.0, 0.0],
            [2, 1, 60.0, 10.0, 2.0, 1600.0, 5.0],
            [3, 1, 10.0, 35.0, 2.0, 100.0, 3.0],
            [4, 0, 15.0, 35.0, 2.0, 12.0, 3.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 40.0, 4.0],
            [1, -1, 35.0, 10.0, 2.0, 21.0, 0.0],
            [2, 1, 60.0, 10.0, 2.0, 1600.0, 5.0],
            [3, 1, 10.0, 35.0, 2.0, 100.0, 3.0],
            [4, 0, 15.0, 35.0, 2.0, 12.0, 3.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    py_obs = jax_env.observation_from_env_state(state, player=0)
    fractions = (0.25, 0.35, 0.55, 0.75)
    expected_candidates = build_action_candidates(
        py_obs,
        max_targets_per_source=4,
        avoid_waste=False,
        send_fractions=fractions,
    )

    actual = jax_env.action_candidates_from_env_state(
        state,
        player=0,
        max_targets_per_source=4,
        send_fractions=jnp.asarray(fractions, dtype=jnp.float64),
    )
    active = np.asarray(actual.active, dtype=bool)
    actual_actions = np.asarray(actual.env_actions)[active]
    expected_actions = np.asarray(
        [(candidate_to_env_action(c) or [[-1.0, 0.0, 0.0]])[0] for c in expected_candidates],
        dtype=np.float64,
    )

    assert np.allclose(actual_actions, expected_actions)
    active_by_target = {}
    for candidate in expected_candidates:
        if candidate["kind"] != "launch":
            continue
        active_by_target.setdefault(candidate["target_id"], set()).add(candidate["ships"])
    assert active_by_target[1] == {30}
    assert active_by_target.get(2, set()) == set()
    assert active_by_target[3] == {30}


def test_jax_action_candidates_match_python_predicted_lead_and_eta_capture_filter():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.2,
        planets=[
            [0, 0, 20.0, 50.0, 3.0, 160.0, 4.0],
            [1, -1, 60.0, 50.0, 2.0, 10.0, 0.0],
            [2, 1, 60.0, 60.0, 2.0, 20.0, 10.0],
            [3, 1, 60.0, 40.0, 2.0, 20.0, 25.0],
        ],
        initial_planets=[
            [0, 0, 20.0, 50.0, 3.0, 160.0, 4.0],
            [1, -1, 60.0, 50.0, 2.0, 10.0, 0.0],
            [2, 1, 60.0, 60.0, 2.0, 20.0, 10.0],
            [3, 1, 60.0, 40.0, 2.0, 20.0, 25.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    py_obs = jax_env.observation_from_env_state(state, player=0)
    fractions = (0.25, 0.75)
    expected_candidates = build_action_candidates(
        py_obs,
        max_targets_per_source=4,
        avoid_waste=False,
        send_fractions=fractions,
    )

    actual = jax_env.action_candidates_from_env_state(
        state,
        player=0,
        max_targets_per_source=4,
        send_fractions=jnp.asarray(fractions, dtype=jnp.float64),
    )
    active = np.asarray(actual.active, dtype=bool)
    actual_actions = np.asarray(actual.env_actions)[active]
    expected_actions = np.asarray(
        [(candidate_to_env_action(c) or [[-1.0, 0.0, 0.0]])[0] for c in expected_candidates],
        dtype=np.float64,
    )
    expected_features = np.stack([featurize(c) for c in expected_candidates], axis=0)

    assert np.allclose(actual_actions, expected_actions)
    assert np.allclose(np.asarray(actual.features)[active], expected_features)

    lead_candidate = next(c for c in expected_candidates if c.get("target_id") == 1 and c.get("ships") == 120)
    expected_angle, expected_eta, expected_distance = aim_with_prediction(20.0, 50.0, py_obs["planets"][1], ships=120, angular_velocity=0.2)
    assert np.isclose(lead_candidate["angle"], expected_angle)
    assert np.isclose(lead_candidate["eta"], expected_eta)
    assert np.isclose(lead_candidate["distance"], expected_distance)
    assert not np.isclose(lead_candidate["angle"], 0.0)
    assert {c["ships"] for c in expected_candidates if c.get("target_id") == 1} == {40, 120}
    assert {c["ships"] for c in expected_candidates if c.get("target_id") == 2} == {120}
    assert {c["ships"] for c in expected_candidates if c.get("target_id") == 3} == set()


def test_jitted_jax_select_action_from_env_state_uses_masked_fixed_candidates():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = _jax_feature_vector([0.2, 0.05, -0.03, 0.01, -0.04, 3.0, -0.2, -0.1, 0.5, -0.25, 0.1, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.02, 0.1])
    value_w = _jax_value_w()

    candidates = jax_env.action_candidates_from_env_state(
        state,
        player=0,
        max_targets_per_source=3,
        send_fractions=send_fractions,
    )
    logits = np.asarray(candidates.features @ actor_w)
    active = np.asarray(candidates.active, dtype=bool)
    masked_logits = np.where(active, logits, -1.0e30)
    expected_idx = int(np.argmax(masked_logits))

    @jax.jit
    def select(state):
        return jax_env.select_action_from_env_state(
            state,
            player=0,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=3,
            send_fractions=send_fractions,
        )

    selected = select(state)

    assert int(selected.chosen_index) == expected_idx
    assert np.allclose(np.asarray(selected.env_action), np.asarray(candidates.env_actions)[expected_idx])
    assert bool(np.asarray(candidates.active)[int(selected.chosen_index)]) is True


def test_jitted_jax_sample_action_from_env_state_threads_key_and_uses_fixed_candidates():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = _jax_feature_vector([0.2, 0.05, -0.03, 0.01, -0.04, 3.0, -0.2, -0.1, 0.5, -0.25, 0.1, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.02, 0.1])
    value_w = _jax_value_w()
    key = jax.random.PRNGKey(11)

    candidates = jax_env.action_candidates_from_env_state(
        state,
        player=0,
        max_targets_per_source=3,
        send_fractions=send_fractions,
    )
    expected_idx, expected_logp, expected_value, expected_entropy, expected_key = jax_policy.sample_masked_candidate(
        key,
        candidates.features,
        candidates.active,
        actor_w,
        value_w,
    )

    @jax.jit
    def sample(state, key):
        return jax_env.sample_action_from_env_state(
            key,
            state,
            player=0,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=3,
            send_fractions=send_fractions,
        )

    actual, next_key = sample(state, key)

    assert int(actual.chosen_index) == int(expected_idx)
    assert np.allclose(np.asarray(actual.env_action), np.asarray(candidates.env_actions)[int(expected_idx)])
    assert bool(np.asarray(candidates.active)[int(actual.chosen_index)]) is True
    assert float(actual.logp) == pytest.approx(float(expected_logp), rel=1e-9, abs=1e-9)
    assert float(actual.value) == pytest.approx(float(expected_value), rel=1e-9, abs=1e-9)
    assert float(actual.entropy) == pytest.approx(float(expected_entropy), rel=1e-9, abs=1e-9)
    np.testing.assert_array_equal(np.asarray(next_key), np.asarray(expected_key))


def test_jitted_jax_sample_action_and_step_env_threads_key_and_steps_sampled_action():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = _jax_actor_w(5.0)
    value_w = _jax_value_w()
    key = jax.random.PRNGKey(17)

    selected, expected_key = jax_env.sample_action_from_env_state(
        key,
        state,
        player=0,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=3,
        send_fractions=send_fractions,
    )
    expected = jax_env.step_env(
        state,
        jnp.asarray([[selected.env_action], [[0.0, 0.0, 0.0]]], dtype=jnp.float64),
        jnp.asarray([[selected.env_action[0] >= 0.0], [False]], dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    @jax.jit
    def sample_and_step(state, key):
        return jax_env.sample_action_and_step_env(
            key,
            state,
            player=0,
            actor_w=actor_w,
            value_w=value_w,
            num_players=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    actual, actual_selected, next_key = sample_and_step(state, key)

    assert np.allclose(np.asarray(actual_selected.env_action), np.asarray(selected.env_action))
    assert int(actual_selected.chosen_index) == int(selected.chosen_index)
    np.testing.assert_array_equal(np.asarray(next_key), np.asarray(expected_key))
    assert np.allclose(np.asarray(actual.planets), np.asarray(expected.planets))
    assert np.allclose(np.asarray(actual.fleets), np.asarray(expected.fleets))
    assert np.asarray(actual.fleet_active).tolist() == np.asarray(expected.fleet_active).tolist()
    assert int(actual.next_fleet_id) == int(expected.next_fleet_id)
    assert int(actual.step) == int(expected.step)


def test_jitted_jax_select_action_and_step_env_launches_selected_fixed_candidate():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = _jax_actor_w(5.0)
    value_w = _jax_value_w()
    selected = jax_env.select_action_from_env_state(
        state,
        player=0,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=3,
        send_fractions=send_fractions,
    )
    assert float(selected.env_action[0]) >= 0.0
    expected = jax_env.step_env(
        state,
        jnp.asarray([[selected.env_action], [[0.0, 0.0, 0.0]]], dtype=jnp.float64),
        jnp.asarray([[True], [False]], dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    @jax.jit
    def select_and_step(state):
        return jax_env.select_action_and_step_env(
            state,
            player=0,
            actor_w=actor_w,
            value_w=value_w,
            num_players=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    actual, actual_selected = select_and_step(state)

    assert np.allclose(np.asarray(actual_selected.env_action), np.asarray(selected.env_action))
    assert int(actual_selected.chosen_index) == int(selected.chosen_index)
    assert np.allclose(np.asarray(actual.planets), np.asarray(expected.planets))
    assert np.allclose(np.asarray(actual.fleets), np.asarray(expected.fleets))
    assert np.asarray(actual.fleet_active).tolist() == np.asarray(expected.fleet_active).tolist()
    assert int(actual.next_fleet_id) == int(expected.next_fleet_id)
    assert int(actual.step) == int(expected.step)


def test_jitted_jax_rollout_env_with_policy_matches_repeated_selected_steps():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = _jax_actor_w(5.0)
    value_w = _jax_value_w()

    expected_1, selected_1 = jax_env.select_action_and_step_env(
        state,
        player=0,
        actor_w=actor_w,
        value_w=value_w,
        num_players=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )
    expected_2, selected_2 = jax_env.select_action_and_step_env(
        expected_1,
        player=0,
        actor_w=actor_w,
        value_w=value_w,
        num_players=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )

    @jax.jit
    def rollout(state):
        return jax_env.rollout_env_with_policy(
            state,
            player=0,
            actor_w=actor_w,
            value_w=value_w,
            num_players=2,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    final_state, rewards, dones, chosen_indices, env_actions = rollout(state)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected_2.planets))
    assert np.allclose(np.asarray(final_state.fleets), np.asarray(expected_2.fleets))
    assert np.asarray(final_state.fleet_active).tolist() == np.asarray(expected_2.fleet_active).tolist()
    assert int(final_state.next_fleet_id) == int(expected_2.next_fleet_id)
    assert int(final_state.step) == int(expected_2.step)
    assert np.asarray(rewards).shape == (2, 2)
    assert np.asarray(dones).shape == (2,)
    assert np.asarray(chosen_indices).tolist() == [int(selected_1.chosen_index), int(selected_2.chosen_index)]
    assert np.allclose(np.asarray(env_actions), np.stack([np.asarray(selected_1.env_action), np.asarray(selected_2.env_action)]))


def test_jitted_jax_rollout_env_with_sampling_threads_key_and_matches_repeated_sampled_steps():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 30.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[] ,
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=4)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = _jax_actor_w(5.0)
    value_w = _jax_value_w()
    key = jax.random.PRNGKey(23)

    expected_1, selected_1, key_1 = jax_env.sample_action_and_step_env(
        key,
        state,
        player=0,
        actor_w=actor_w,
        value_w=value_w,
        num_players=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )
    expected_2, selected_2, expected_key = jax_env.sample_action_and_step_env(
        key_1,
        expected_1,
        player=0,
        actor_w=actor_w,
        value_w=value_w,
        num_players=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )

    @jax.jit
    def rollout(state, key):
        return jax_env.rollout_env_with_sampling(
            key,
            state,
            player=0,
            actor_w=actor_w,
            value_w=value_w,
            num_players=2,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    final_state, rewards, dones, chosen_indices, env_actions, next_key = rollout(state, key)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected_2.planets))
    assert np.allclose(np.asarray(final_state.fleets), np.asarray(expected_2.fleets))
    assert np.asarray(final_state.fleet_active).tolist() == np.asarray(expected_2.fleet_active).tolist()
    assert int(final_state.next_fleet_id) == int(expected_2.next_fleet_id)
    assert int(final_state.step) == int(expected_2.step)
    assert np.asarray(rewards).shape == (2, 2)
    assert np.asarray(dones).shape == (2,)
    assert np.asarray(chosen_indices).tolist() == [int(selected_1.chosen_index), int(selected_2.chosen_index)]
    assert np.allclose(np.asarray(env_actions), np.stack([np.asarray(selected_1.env_action), np.asarray(selected_2.env_action)]))
    np.testing.assert_array_equal(np.asarray(next_key), np.asarray(expected_key))


def test_jitted_jax_rollout_env_with_multiagent_sampling_matches_manual_all_player_steps():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(31), 2)

    selected_0, key_0 = jax_env.sample_action_from_env_state(
        keys[0], state, player=0, actor_w=actor_w[0], value_w=value_w[0], max_targets_per_source=3, send_fractions=send_fractions
    )
    selected_1, key_1 = jax_env.sample_action_from_env_state(
        keys[1], state, player=1, actor_w=actor_w[1], value_w=value_w[1], max_targets_per_source=3, send_fractions=send_fractions
    )
    expected_1 = jax_env.step_env(
        state,
        jnp.stack([selected_0.env_action, selected_1.env_action])[:, None, :],
        jnp.asarray([[selected_0.env_action[0] >= 0.0], [selected_1.env_action[0] >= 0.0]], dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )
    selected_2, key_2 = jax_env.sample_action_from_env_state(
        key_0, expected_1, player=0, actor_w=actor_w[0], value_w=value_w[0], max_targets_per_source=3, send_fractions=send_fractions
    )
    selected_3, expected_keys_2 = jax_env.sample_action_from_env_state(
        key_1, expected_1, player=1, actor_w=actor_w[1], value_w=value_w[1], max_targets_per_source=3, send_fractions=send_fractions
    )
    expected_2 = jax_env.step_env(
        expected_1,
        jnp.stack([selected_2.env_action, selected_3.env_action])[:, None, :],
        jnp.asarray([[selected_2.env_action[0] >= 0.0], [selected_3.env_action[0] >= 0.0]], dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )
    expected_keys = jnp.stack([key_2, expected_keys_2])

    @jax.jit
    def rollout(state, keys):
        return jax_env.rollout_env_with_multiagent_sampling(
            keys,
            state,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    final_state, rewards, dones, chosen_indices, env_actions, next_keys = rollout(state, keys)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected_2.planets))
    assert np.allclose(np.asarray(final_state.fleets), np.asarray(expected_2.fleets))
    assert np.asarray(final_state.fleet_active).tolist() == np.asarray(expected_2.fleet_active).tolist()
    assert int(final_state.next_fleet_id) == int(expected_2.next_fleet_id)
    assert int(final_state.step) == int(expected_2.step)
    assert np.asarray(rewards).shape == (2, 2)
    assert np.asarray(dones).shape == (2,)
    assert np.asarray(chosen_indices).tolist() == [
        [int(selected_0.chosen_index), int(selected_1.chosen_index)],
        [int(selected_2.chosen_index), int(selected_3.chosen_index)],
    ]
    assert np.allclose(
        np.asarray(env_actions),
        np.asarray(
            [
                [np.asarray(selected_0.env_action), np.asarray(selected_1.env_action)],
                [np.asarray(selected_2.env_action), np.asarray(selected_3.env_action)],
            ]
        ),
    )
    np.testing.assert_array_equal(np.asarray(next_keys), np.asarray(expected_keys))


def test_jitted_jax_rollout_env_with_multiagent_sampling_stats_collects_policy_outputs():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(37), 2)

    selected_0, key_0 = jax_env.sample_action_from_env_state(
        keys[0], state, player=0, actor_w=actor_w[0], value_w=value_w[0], max_targets_per_source=3, send_fractions=send_fractions
    )
    selected_1, key_1 = jax_env.sample_action_from_env_state(
        keys[1], state, player=1, actor_w=actor_w[1], value_w=value_w[1], max_targets_per_source=3, send_fractions=send_fractions
    )
    expected = jax_env.step_env(
        state,
        jnp.stack([selected_0.env_action, selected_1.env_action])[:, None, :],
        jnp.asarray([[selected_0.env_action[0] >= 0.0], [selected_1.env_action[0] >= 0.0]], dtype=jnp.bool_),
        episode_steps=100,
        ship_speed=6.0,
    )

    @jax.jit
    def rollout(state, keys):
        return jax_env.rollout_env_with_multiagent_sampling_stats(
            keys,
            state,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=1,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    final_state, rewards, dones, chosen_indices, env_actions, logps, values, entropies, next_keys = rollout(state, keys)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected.planets))
    assert np.asarray(rewards).shape == (1, 2)
    assert np.asarray(dones).shape == (1,)
    assert np.asarray(chosen_indices).tolist() == [[int(selected_0.chosen_index), int(selected_1.chosen_index)]]
    assert np.allclose(np.asarray(env_actions[0]), np.stack([np.asarray(selected_0.env_action), np.asarray(selected_1.env_action)]))
    assert np.allclose(np.asarray(logps), np.asarray([[float(selected_0.logp), float(selected_1.logp)]]))
    assert np.allclose(np.asarray(values), np.asarray([[float(selected_0.value), float(selected_1.value)]]))
    assert np.allclose(np.asarray(entropies), np.asarray([[float(selected_0.entropy), float(selected_1.entropy)]]))
    np.testing.assert_array_equal(np.asarray(next_keys), np.asarray(jnp.stack([key_0, key_1])))


def test_jitted_jax_sample_all_actions_training_data_collects_fixed_candidates_for_update():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(41), 2)
    expected_candidates_0 = jax_env.action_candidates_from_env_state(
        state, player=0, max_targets_per_source=3, send_fractions=send_fractions
    )
    expected_candidates_1 = jax_env.action_candidates_from_env_state(
        state, player=1, max_targets_per_source=3, send_fractions=send_fractions
    )
    selected_0, key_0 = jax_env.sample_action_from_env_state(
        keys[0], state, player=0, actor_w=actor_w[0], value_w=value_w[0], max_targets_per_source=3, send_fractions=send_fractions
    )
    selected_1, key_1 = jax_env.sample_action_from_env_state(
        keys[1], state, player=1, actor_w=actor_w[1], value_w=value_w[1], max_targets_per_source=3, send_fractions=send_fractions
    )

    @jax.jit
    def sample_step(state, keys):
        return jax_env.sample_all_actions_and_step_env_with_training_data(
            keys,
            state,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    (
        _next_state,
        chosen_indices,
        env_actions,
        candidate_features,
        candidate_active,
        logps,
        values,
        entropies,
        next_keys,
    ) = sample_step(state, keys)

    assert np.asarray(chosen_indices).tolist() == [int(selected_0.chosen_index), int(selected_1.chosen_index)]
    assert np.allclose(np.asarray(env_actions), np.stack([np.asarray(selected_0.env_action), np.asarray(selected_1.env_action)]))
    assert np.allclose(
        np.asarray(candidate_features),
        np.stack([np.asarray(expected_candidates_0.features), np.asarray(expected_candidates_1.features)]),
    )
    np.testing.assert_array_equal(
        np.asarray(candidate_active),
        np.stack([np.asarray(expected_candidates_0.active), np.asarray(expected_candidates_1.active)]),
    )
    assert np.allclose(np.asarray(logps), np.asarray([float(selected_0.logp), float(selected_1.logp)]))
    assert np.allclose(np.asarray(values), np.asarray([float(selected_0.value), float(selected_1.value)]))
    assert np.allclose(np.asarray(entropies), np.asarray([float(selected_0.entropy), float(selected_1.entropy)]))
    np.testing.assert_array_equal(np.asarray(next_keys), np.asarray(jnp.stack([key_0, key_1])))


def test_jitted_jax_rollout_env_with_multiagent_training_data_collects_scan_batches():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(43), 2)

    (
        expected_1,
        chosen_1,
        actions_1,
        features_1,
        active_1,
        logps_1,
        values_1,
        entropies_1,
        keys_1,
    ) = jax_env.sample_all_actions_and_step_env_with_training_data(
        keys,
        state,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )
    (
        expected_2,
        chosen_2,
        actions_2,
        features_2,
        active_2,
        logps_2,
        values_2,
        entropies_2,
        expected_keys,
    ) = jax_env.sample_all_actions_and_step_env_with_training_data(
        keys_1,
        expected_1,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )

    @jax.jit
    def rollout(state, keys):
        return jax_env.rollout_env_with_multiagent_training_data(
            keys,
            state,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    (
        final_state,
        rewards,
        dones,
        chosen_indices,
        env_actions,
        candidate_features,
        candidate_active,
        logps,
        values,
        entropies,
        next_keys,
    ) = rollout(state, keys)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected_2.planets))
    assert np.allclose(np.asarray(final_state.fleets), np.asarray(expected_2.fleets))
    assert np.asarray(final_state.fleet_active).tolist() == np.asarray(expected_2.fleet_active).tolist()
    assert int(final_state.next_fleet_id) == int(expected_2.next_fleet_id)
    assert int(final_state.step) == int(expected_2.step)
    assert np.asarray(rewards).shape == (2, 2)
    assert np.asarray(dones).shape == (2,)
    np.testing.assert_array_equal(np.asarray(chosen_indices), np.asarray(jnp.stack([chosen_1, chosen_2])))
    assert np.allclose(np.asarray(env_actions), np.asarray(jnp.stack([actions_1, actions_2])))
    assert np.allclose(np.asarray(candidate_features), np.asarray(jnp.stack([features_1, features_2])))
    np.testing.assert_array_equal(np.asarray(candidate_active), np.asarray(jnp.stack([active_1, active_2])))
    assert np.allclose(np.asarray(logps), np.asarray(jnp.stack([logps_1, logps_2])))
    assert np.allclose(np.asarray(values), np.asarray(jnp.stack([values_1, values_2])))
    assert np.allclose(np.asarray(entropies), np.asarray(jnp.stack([entropies_1, entropies_2])))
    np.testing.assert_array_equal(np.asarray(next_keys), np.asarray(expected_keys))


def test_jitted_jax_multiagent_ppo_train_step_matches_explicit_rollout_bootstrap_and_update():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(47), 2)

    (
        expected_final_state,
        rewards,
        dones,
        chosen_indices,
        _env_actions,
        candidate_features,
        candidate_active,
        logps,
        values,
        _entropies,
        expected_keys,
    ) = jax_env.rollout_env_with_multiagent_training_data(
        keys,
        state,
        actor_w=actor_w,
        value_w=value_w,
        num_steps=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )
    bootstrap_values = []
    for player in range(2):
        candidates = jax_env.action_candidates_from_env_state(
            expected_final_state,
            player=player,
            max_targets_per_source=3,
            send_fractions=send_fractions,
        )
        _probs, value, _entropy = jax_policy.masked_policy_outputs(candidates.features, candidates.active, actor_w[player], value_w[player])
        bootstrap_values.append(value)
    bootstrap_values = jnp.stack(bootstrap_values)
    expected_actor_w, expected_value_w, expected_returns, expected_advantages, expected_metrics = jax_policy.multiagent_ppo_update_from_rollout(
        candidate_features,
        candidate_active,
        chosen_indices,
        logps,
        rewards,
        dones,
        values,
        bootstrap_values,
        actor_w,
        value_w,
        gamma=0.9,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    @jax.jit
    def train_step(state, keys):
        return jax_env.multiagent_ppo_train_step(
            keys,
            state,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
            gamma=0.9,
            lr=0.01,
            clip_epsilon=0.2,
            entropy_coef=0.003,
            value_coef=0.5,
        )

    final_state, next_actor_w, next_value_w, returns, advantages, metrics, next_keys = train_step(state, keys)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected_final_state.planets))
    assert np.allclose(np.asarray(final_state.fleets), np.asarray(expected_final_state.fleets))
    assert np.asarray(final_state.fleet_active).tolist() == np.asarray(expected_final_state.fleet_active).tolist()
    assert int(final_state.next_fleet_id) == int(expected_final_state.next_fleet_id)
    assert int(final_state.step) == int(expected_final_state.step)
    assert np.allclose(np.asarray(next_actor_w), np.asarray(expected_actor_w))
    assert np.allclose(np.asarray(next_value_w), np.asarray(expected_value_w))
    assert np.allclose(np.asarray(returns), np.asarray(expected_returns))
    assert np.allclose(np.asarray(advantages), np.asarray(expected_advantages))
    for key in expected_metrics:
        assert np.allclose(np.asarray(metrics[key]), np.asarray(expected_metrics[key]))
    np.testing.assert_array_equal(np.asarray(next_keys), np.asarray(expected_keys))


def test_jitted_jax_batched_multiagent_training_data_rollout_matches_single_env_rollouts():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(53), 2)
    expected = jax_env.rollout_env_with_multiagent_training_data(
        keys,
        state,
        actor_w=actor_w,
        value_w=value_w,
        num_steps=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )
    state_batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
    keys_batch = jnp.stack([keys, keys])

    @jax.jit
    def batched_rollout(state_batch, keys_batch):
        return jax_env.batched_rollout_env_with_multiagent_training_data(
            keys_batch,
            state_batch,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
        )

    actual = batched_rollout(state_batch, keys_batch)
    expected_final_state = expected[0]
    actual_final_state = actual[0]
    for batch_index in range(2):
        assert np.allclose(np.asarray(actual_final_state.planets[batch_index]), np.asarray(expected_final_state.planets))
        assert np.allclose(np.asarray(actual_final_state.fleets[batch_index]), np.asarray(expected_final_state.fleets))
        np.testing.assert_array_equal(np.asarray(actual_final_state.fleet_active[batch_index]), np.asarray(expected_final_state.fleet_active))
        assert int(actual_final_state.next_fleet_id[batch_index]) == int(expected_final_state.next_fleet_id)
        assert int(actual_final_state.step[batch_index]) == int(expected_final_state.step)
        for actual_item, expected_item in zip(actual[1:], expected[1:]):
            assert np.allclose(np.asarray(actual_item[batch_index]), np.asarray(expected_item))


def test_jitted_jax_batched_multiagent_ppo_train_step_matches_explicit_batched_rollout_and_update():
    obs = _struct(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        initial_planets=[
            [0, 0, 10.0, 10.0, 3.0, 80.0, 5.0],
            [1, 0, 40.0, 10.0, 2.0, 20.0, 4.0],
            [2, 1, 70.0, 10.0, 2.0, 50.0, 3.0],
            [3, -1, 90.0, 10.0, 2.0, 10.0, 2.0],
        ],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=0, max_planets=8, max_fleets=8)
    state_batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
    send_fractions = jnp.asarray([0.35, 0.55, 0.75], dtype=jnp.float64)
    actor_w = jnp.stack(
        [
            _jax_actor_w(5.0),
            _jax_actor_w(4.0),
        ]
    )
    value_w = jnp.stack([_jax_value_w(-0.05, 0.05), _jax_value_w(0.03, -0.03)])
    keys = jax.random.split(jax.random.PRNGKey(59), 2)
    keys_batch = jnp.stack([keys, keys])
    expected_rollout = jax_env.batched_rollout_env_with_multiagent_training_data(
        keys_batch,
        state_batch,
        actor_w=actor_w,
        value_w=value_w,
        num_steps=2,
        max_targets_per_source=3,
        send_fractions=send_fractions,
        episode_steps=100,
        ship_speed=6.0,
    )
    (
        expected_final_state,
        rewards,
        dones,
        chosen_indices,
        _env_actions,
        candidate_features,
        candidate_active,
        logps,
        values,
        _entropies,
        expected_keys,
    ) = expected_rollout
    bootstrap_values = []
    for env_index in range(2):
        player_values = []
        env_state = jax.tree_util.tree_map(lambda x: x[env_index], expected_final_state)
        for player in range(2):
            candidates = jax_env.action_candidates_from_env_state(
                env_state,
                player=player,
                max_targets_per_source=3,
                send_fractions=send_fractions,
            )
            _probs, value, _entropy = jax_policy.masked_policy_outputs(candidates.features, candidates.active, actor_w[player], value_w[player])
            player_values.append(value)
        bootstrap_values.append(jnp.stack(player_values))
    bootstrap_values = jnp.stack(bootstrap_values)
    expected_actor_w, expected_value_w, expected_returns, expected_advantages, expected_metrics = jax_policy.batched_multiagent_ppo_update_from_rollout(
        candidate_features,
        candidate_active,
        chosen_indices,
        logps,
        rewards,
        dones,
        values,
        bootstrap_values,
        actor_w,
        value_w,
        gamma=0.9,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    @jax.jit
    def train_step(state_batch, keys_batch):
        return jax_env.batched_multiagent_ppo_train_step(
            keys_batch,
            state_batch,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=2,
            max_targets_per_source=3,
            send_fractions=send_fractions,
            episode_steps=100,
            ship_speed=6.0,
            gamma=0.9,
            lr=0.01,
            clip_epsilon=0.2,
            entropy_coef=0.003,
            value_coef=0.5,
        )

    final_state, next_actor_w, next_value_w, returns, advantages, metrics, next_keys = train_step(state_batch, keys_batch)

    assert np.allclose(np.asarray(final_state.planets), np.asarray(expected_final_state.planets))
    assert np.allclose(np.asarray(final_state.fleets), np.asarray(expected_final_state.fleets))
    np.testing.assert_array_equal(np.asarray(final_state.fleet_active), np.asarray(expected_final_state.fleet_active))
    assert np.allclose(np.asarray(next_actor_w), np.asarray(expected_actor_w))
    assert np.allclose(np.asarray(next_value_w), np.asarray(expected_value_w))
    assert np.allclose(np.asarray(returns), np.asarray(expected_returns))
    assert np.allclose(np.asarray(advantages), np.asarray(expected_advantages))
    for key in expected_metrics:
        assert np.allclose(np.asarray(metrics[key]), np.asarray(expected_metrics[key]))
    np.testing.assert_array_equal(np.asarray(next_keys), np.asarray(expected_keys))


def test_jax_env_step_matches_official_one_noop_tick_from_reset_state():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    official_obs, official_rewards, official_statuses = _run_official_tick(
        obs.planets,
        obs.fleets,
        players=2,
        step=obs.step,
        episode_steps=500,
        angular_velocity=obs.angular_velocity,
        initial_planets=obs.initial_planets,
        comet_planet_ids=obs.comet_planet_ids,
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)

    actual = jax_env.step_env(
        state,
        jnp.zeros((2, 1, 3), dtype=jnp.float64),
        jnp.zeros((2, 1), dtype=jnp.bool_),
        episode_steps=500,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(actual), np.asarray(official_obs.planets))
    assert len(_active_fleets(actual)) == len(official_obs.fleets)
    assert int(actual.next_fleet_id) == official_obs.next_fleet_id
    assert int(actual.step) == obs.step + 1
    assert np.asarray(actual.rewards).tolist() == official_rewards
    assert bool(actual.done) == all(status == "DONE" for status in official_statuses)


def test_jax_env_step_matches_official_launched_action_from_reset_state():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    source = next(p for p in obs.planets if int(p[1]) == 0 and int(p[5]) > 5)
    action = [[int(source[0]), 0.0, 5.0]]
    official_obs, official_rewards, official_statuses = _run_official_tick(
        obs.planets,
        obs.fleets,
        actions=[action, []],
        players=2,
        step=obs.step,
        episode_steps=500,
        angular_velocity=obs.angular_velocity,
        initial_planets=obs.initial_planets,
        comet_planet_ids=obs.comet_planet_ids,
        seed=episode_seed,
        comet_speed=4.0,
    )
    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)

    actual = jax_env.step_env(
        state,
        jnp.asarray([action, [[0.0, 0.0, 0.0]]], dtype=jnp.float64),
        jnp.asarray([[True], [False]]),
        episode_steps=500,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(actual), np.asarray(official_obs.planets))
    assert np.allclose(_active_fleets(actual), np.asarray(official_obs.fleets))
    assert int(actual.next_fleet_id) == official_obs.next_fleet_id
    assert int(actual.step) == obs.step + 1
    assert np.asarray(actual.rewards).tolist() == official_rewards
    assert bool(actual.done) == all(status == "DONE" for status in official_statuses)


def test_jax_rollout_env_matches_official_three_noop_ticks_from_reset_state():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    planets = obs.planets
    fleets = obs.fleets
    for step in range(obs.step, obs.step + 3):
        official_obs, _official_rewards, _official_statuses = _run_official_tick(
            planets,
            fleets,
            players=2,
            step=step,
            episode_steps=500,
            angular_velocity=obs.angular_velocity,
            initial_planets=obs.initial_planets,
            comet_planet_ids=obs.comet_planet_ids,
        )
        planets = official_obs.planets
        fleets = official_obs.fleets

    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)
    final_state, rewards, dones = jax_env.rollout_env(
        state,
        jnp.zeros((3, 2, 1, 3), dtype=jnp.float64),
        jnp.zeros((3, 2, 1), dtype=jnp.bool_),
        episode_steps=500,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(final_state), np.asarray(planets))
    assert len(_active_fleets(final_state)) == len(fleets)
    assert int(final_state.step) == obs.step + 3
    assert np.asarray(rewards).shape == (3, 2)
    assert np.asarray(dones).shape == (3,)


def test_jax_batched_rollout_env_matches_single_rollout_for_identical_noop_envs():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)
    action_arrays = jnp.zeros((3, 2, 1, 3), dtype=jnp.float64)
    action_active = jnp.zeros((3, 2, 1), dtype=jnp.bool_)
    expected_final, expected_rewards, expected_dones = jax_env.rollout_env(
        state,
        action_arrays,
        action_active,
        episode_steps=500,
        ship_speed=6.0,
    )
    batched_state = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)

    final_batch, reward_batch, done_batch = jax_env.batched_rollout_env(
        batched_state,
        jnp.stack([action_arrays, action_arrays]),
        jnp.stack([action_active, action_active]),
        episode_steps=500,
        ship_speed=6.0,
    )

    assert np.allclose(np.asarray(final_batch.planets[0]), np.asarray(expected_final.planets))
    assert np.allclose(np.asarray(final_batch.planets[1]), np.asarray(expected_final.planets))
    assert np.allclose(np.asarray(reward_batch[0]), np.asarray(expected_rewards))
    assert np.allclose(np.asarray(reward_batch[1]), np.asarray(expected_rewards))
    assert np.asarray(done_batch).shape == (2, 3)
    assert np.asarray(done_batch[0]).tolist() == np.asarray(expected_dones).tolist()


def test_jax_rollout_env_matches_official_real_spawn_tick_from_reset_state():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    planets = obs.planets
    fleets = obs.fleets
    comets = obs.comets
    comet_planet_ids = obs.comet_planet_ids
    final_official_obs = None
    for step in range(obs.step, obs.step + 50):
        final_official_obs, _official_rewards, _official_statuses = _run_official_tick(
            planets,
            fleets,
            players=2,
            step=step,
            episode_steps=500,
            angular_velocity=obs.angular_velocity,
            initial_planets=obs.initial_planets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            seed=episode_seed,
            comet_speed=4.0,
        )
        planets = final_official_obs.planets
        fleets = final_official_obs.fleets
        comets = final_official_obs.comets
        comet_planet_ids = final_official_obs.comet_planet_ids
    assert final_official_obs is not None
    assert len(comets) == 1
    assert len(comet_planet_ids) == 4

    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)
    final_state, rewards, dones = jax_env.rollout_env(
        state,
        jnp.zeros((50, 2, 1, 3), dtype=jnp.float64),
        jnp.zeros((50, 2, 1), dtype=jnp.bool_),
        episode_steps=500,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(final_state), np.asarray(final_official_obs.planets))
    assert len(_active_fleets(final_state)) == len(final_official_obs.fleets)
    assert int(final_state.step) == obs.step + 50
    assert np.asarray(final_state.comet_planet_ids).tolist() == final_official_obs.comet_planet_ids
    assert int(final_state.comet_path_index) == final_official_obs.comets[0]["path_index"]
    assert np.asarray(rewards).shape == (50, 2)
    assert np.asarray(dones).shape == (50,)


def test_jax_rollout_env_matches_official_real_comet_expiry_metadata_cleanup():
    obs, episode_seed = _run_official_reset(players=2, seed=123)
    state = jax_env.env_state_from_observation(obs, episode_seed=episode_seed, max_fleets=8)
    first_path_len = int(np.asarray(state.spawn_path_lens)[0])
    assert first_path_len > 0

    planets = obs.planets
    fleets = obs.fleets
    comets = obs.comets
    comet_planet_ids = obs.comet_planet_ids
    final_official_obs = None
    rollout_steps = 50 + first_path_len
    for step in range(obs.step, obs.step + rollout_steps):
        final_official_obs, _official_rewards, _official_statuses = _run_official_tick(
            planets,
            fleets,
            players=2,
            step=step,
            episode_steps=500,
            angular_velocity=obs.angular_velocity,
            initial_planets=obs.initial_planets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            seed=episode_seed,
            comet_speed=4.0,
        )
        planets = final_official_obs.planets
        fleets = final_official_obs.fleets
        comets = final_official_obs.comets
        comet_planet_ids = final_official_obs.comet_planet_ids
    assert final_official_obs is not None
    assert final_official_obs.comets == []
    assert final_official_obs.comet_planet_ids == []

    final_state, _rewards, _dones = jax_env.rollout_env(
        state,
        jnp.zeros((rollout_steps, 2, 1, 3), dtype=jnp.float64),
        jnp.zeros((rollout_steps, 2, 1), dtype=jnp.bool_),
        episode_steps=500,
        ship_speed=6.0,
    )

    assert np.allclose(_active_planets(final_state), np.asarray(final_official_obs.planets))
    assert len(_active_fleets(final_state)) == len(final_official_obs.fleets)
    assert np.asarray(final_state.comet_planet_ids).tolist() == [-1, -1, -1, -1]
    assert int(final_state.comet_path_len) == 0
