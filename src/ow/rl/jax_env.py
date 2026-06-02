"""JAX parity foundation for Orbit Wars environment helpers.

This module intentionally starts with small, testable mechanics that mirror the
existing NumPy/Python helpers. Full reset/step rollout support should build on
these parity-checked primitives rather than bypassing them.
"""

import random
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from kaggle_environments.envs.orbit_wars import orbit_wars as _official_orbit_wars

from src.ow.constants import (
    BOARD_SIZE,
    CENTER_X,
    CENTER_Y,
    FLEET_COL_ANGLE,
    FLEET_COL_FROM,
    FLEET_COL_ID,
    FLEET_COL_OWNER,
    FLEET_COL_SHIPS,
    FLEET_COL_X,
    FLEET_COL_Y,
    PLANET_COL_ID,
    PLANET_COL_OWNER,
    PLANET_COL_PROD,
    PLANET_COL_RADIUS,
    PLANET_COL_SHIPS,
    PLANET_COL_X,
    PLANET_COL_Y,
    ROTATION_RADIUS_LIMIT,
    SHIP_SPEED_MAX,
    SUN_RADIUS,
)
from src.ow.rl import jax_policy
from src.ow.state import parse_observation as _parse_numpy_observation


DEFAULT_MAX_BASE_PLANETS = 40
DEFAULT_MAX_COMET_PLANETS = 4
DEFAULT_MAX_PLANETS = DEFAULT_MAX_BASE_PLANETS + DEFAULT_MAX_COMET_PLANETS
DEFAULT_MAX_COMET_PATH = 40
DEFAULT_COMET_SPEED = 4.0


class JaxGameState(NamedTuple):
    player: jax.Array
    step: jax.Array
    angular_velocity: jax.Array
    planets: jax.Array
    fleets: jax.Array
    comets: jax.Array
    initial_planets: jax.Array
    my_planet_mask: jax.Array
    enemy_planet_mask: jax.Array
    neutral_planet_mask: jax.Array


class JaxTickState(NamedTuple):
    planets: jax.Array
    fleets: jax.Array
    fleet_active: jax.Array
    initial_planets: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    angular_velocity: jax.Array

    @classmethod
    def from_arrays(
        cls,
        *,
        planets,
        fleets,
        fleet_active,
        initial_planets,
        next_fleet_id,
        step,
        angular_velocity,
    ) -> "JaxTickState":
        return cls(
            planets=jnp.asarray(planets, dtype=jnp.float64),
            fleets=jnp.asarray(fleets, dtype=jnp.float64),
            fleet_active=jnp.asarray(fleet_active, dtype=jnp.bool_),
            initial_planets=jnp.asarray(initial_planets, dtype=jnp.float64),
            next_fleet_id=jnp.asarray(next_fleet_id, dtype=jnp.int64),
            step=jnp.asarray(step, dtype=jnp.int64),
            angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float64),
        )


class JaxTickResult(NamedTuple):
    planets: jax.Array
    fleets: jax.Array
    fleet_active: jax.Array
    initial_planets: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    rewards: jax.Array
    done: jax.Array


class JaxCometTickState(NamedTuple):
    planets: jax.Array
    planet_active: jax.Array
    fleets: jax.Array
    fleet_active: jax.Array
    initial_planets: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    angular_velocity: jax.Array
    comet_planet_ids: jax.Array
    comet_paths: jax.Array
    comet_path_index: jax.Array
    comet_path_len: jax.Array
    spawn_active: jax.Array
    spawn_step: jax.Array
    spawn_paths: jax.Array
    spawn_path_len: jax.Array
    spawn_ships: jax.Array

    @classmethod
    def from_arrays(
        cls,
        *,
        planets,
        planet_active=None,
        fleets,
        fleet_active,
        initial_planets,
        next_fleet_id,
        step,
        angular_velocity,
        comet_planet_ids,
        comet_paths,
        comet_path_index,
        comet_path_len,
        spawn_active=False,
        spawn_step=0,
        spawn_paths=None,
        spawn_path_len=0,
        spawn_ships=0.0,
    ) -> "JaxCometTickState":
        comet_paths_arr = jnp.asarray(comet_paths, dtype=jnp.float64)
        return cls(
            planets=jnp.asarray(planets, dtype=jnp.float64),
            planet_active=jnp.ones((jnp.asarray(planets).shape[0],), dtype=jnp.bool_)
            if planet_active is None
            else jnp.asarray(planet_active, dtype=jnp.bool_),
            fleets=jnp.asarray(fleets, dtype=jnp.float64),
            fleet_active=jnp.asarray(fleet_active, dtype=jnp.bool_),
            initial_planets=jnp.asarray(initial_planets, dtype=jnp.float64),
            next_fleet_id=jnp.asarray(next_fleet_id, dtype=jnp.int64),
            step=jnp.asarray(step, dtype=jnp.int64),
            angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float64),
            comet_planet_ids=jnp.asarray(comet_planet_ids, dtype=jnp.int64),
            comet_paths=comet_paths_arr,
            comet_path_index=jnp.asarray(comet_path_index, dtype=jnp.int64),
            comet_path_len=jnp.asarray(comet_path_len, dtype=jnp.int64),
            spawn_active=jnp.asarray(spawn_active, dtype=jnp.bool_),
            spawn_step=jnp.asarray(spawn_step, dtype=jnp.int64),
            spawn_paths=comet_paths_arr if spawn_paths is None else jnp.asarray(spawn_paths, dtype=jnp.float64),
            spawn_path_len=jnp.asarray(spawn_path_len, dtype=jnp.int64),
            spawn_ships=jnp.asarray(spawn_ships, dtype=jnp.float64),
        )


class JaxCometTickResult(NamedTuple):
    planets: jax.Array
    planet_active: jax.Array
    fleets: jax.Array
    fleet_active: jax.Array
    initial_planets: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    rewards: jax.Array
    done: jax.Array
    comet_planet_ids: jax.Array
    comet_paths: jax.Array
    comet_path_index: jax.Array
    comet_path_len: jax.Array


class JaxEnvState(NamedTuple):
    planets: jax.Array
    planet_active: jax.Array
    fleets: jax.Array
    fleet_active: jax.Array
    initial_planets: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    angular_velocity: jax.Array
    comet_planet_ids: jax.Array
    comet_paths: jax.Array
    comet_path_index: jax.Array
    comet_path_len: jax.Array
    spawn_steps: jax.Array
    spawn_available: jax.Array
    spawn_paths: jax.Array
    spawn_path_lens: jax.Array
    spawn_ships: jax.Array


class JaxEnvStepResult(NamedTuple):
    planets: jax.Array
    planet_active: jax.Array
    fleets: jax.Array
    fleet_active: jax.Array
    initial_planets: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    angular_velocity: jax.Array
    comet_planet_ids: jax.Array
    comet_paths: jax.Array
    comet_path_index: jax.Array
    comet_path_len: jax.Array
    spawn_steps: jax.Array
    spawn_available: jax.Array
    spawn_paths: jax.Array
    spawn_path_lens: jax.Array
    spawn_ships: jax.Array
    rewards: jax.Array
    done: jax.Array


class JaxActionCandidates(NamedTuple):
    env_actions: jax.Array
    active: jax.Array
    features: jax.Array


class JaxSelectedAction(NamedTuple):
    env_action: jax.Array
    chosen_index: jax.Array
    logp: jax.Array
    value: jax.Array
    entropy: jax.Array


def parse_observation(obs) -> JaxGameState:
    """Parse an Orbit Wars observation into a JAX state, matching NumPy parser."""

    state = _parse_numpy_observation(obs)
    return JaxGameState(
        player=jnp.asarray(state.player, dtype=jnp.int64),
        step=jnp.asarray(state.step, dtype=jnp.int64),
        angular_velocity=jnp.asarray(state.angular_velocity, dtype=jnp.float64),
        planets=jnp.asarray(state.planets, dtype=jnp.float64),
        fleets=jnp.asarray(state.fleets, dtype=jnp.float64),
        comets=jnp.asarray(state.comets, dtype=jnp.float64),
        initial_planets=jnp.asarray(state.initial_planets, dtype=jnp.float64),
        my_planet_mask=jnp.asarray(state.my_planet_mask, dtype=jnp.bool_),
        enemy_planet_mask=jnp.asarray(state.enemy_planet_mask, dtype=jnp.bool_),
        neutral_planet_mask=jnp.asarray(state.neutral_planet_mask, dtype=jnp.bool_),
    )


def _padded_rows(rows, *, shape, dtype=float):
    arr = jnp.zeros(shape, dtype=jnp.float64 if dtype is float else jnp.int64)
    rows_arr = jnp.asarray(rows, dtype=arr.dtype)
    if rows_arr.size == 0:
        return arr, 0
    if len(rows_arr.shape) == 1:
        rows_arr = rows_arr.reshape((0, shape[1]))
    n = min(int(rows_arr.shape[0]), shape[0])
    return arr.at[:n].set(rows_arr[:n]), n


def _current_comet_arrays(obs, *, max_comet_planets, max_comet_path):
    comet_planet_ids = jnp.full((max_comet_planets,), -1, dtype=jnp.int64)
    comet_paths = jnp.zeros((max_comet_planets, max_comet_path, 2), dtype=jnp.float64)
    comet_path_index = jnp.asarray(-1, dtype=jnp.int64)
    comet_path_len = jnp.asarray(0, dtype=jnp.int64)

    comets = getattr(obs, "comets", []) or []
    if not comets:
        return comet_planet_ids, comet_paths, comet_path_index, comet_path_len

    comet = comets[0]
    ids = list(comet.get("planet_ids", []))[:max_comet_planets]
    paths = comet.get("paths", [])[:max_comet_planets]
    comet_planet_ids = comet_planet_ids.at[: len(ids)].set(jnp.asarray(ids, dtype=jnp.int64))
    for i, path in enumerate(paths):
        path_arr = jnp.asarray(path, dtype=jnp.float64)
        path_len = min(int(path_arr.shape[0]), max_comet_path)
        comet_paths = comet_paths.at[i, :path_len].set(path_arr[:path_len])
    if paths:
        comet_path_len = jnp.asarray(min(len(paths[0]), max_comet_path), dtype=jnp.int64)
    comet_path_index = jnp.asarray(comet.get("path_index", -1), dtype=jnp.int64)
    return comet_planet_ids, comet_paths, comet_path_index, comet_path_len


def _precompute_comet_schedule(initial_planets, angular_velocity, episode_seed, *, max_comet_planets, max_comet_path, comet_speed):
    spawn_steps = jnp.asarray(_official_orbit_wars.COMET_SPAWN_STEPS, dtype=jnp.int64)
    spawn_available = jnp.zeros((len(_official_orbit_wars.COMET_SPAWN_STEPS),), dtype=jnp.bool_)
    spawn_paths = jnp.zeros((len(_official_orbit_wars.COMET_SPAWN_STEPS), max_comet_planets, max_comet_path, 2), dtype=jnp.float64)
    spawn_path_lens = jnp.zeros((len(_official_orbit_wars.COMET_SPAWN_STEPS),), dtype=jnp.int64)
    spawn_ships = jnp.zeros((len(_official_orbit_wars.COMET_SPAWN_STEPS),), dtype=jnp.float64)

    for schedule_i, spawn_step in enumerate(_official_orbit_wars.COMET_SPAWN_STEPS):
        rng = random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")
        paths = _official_orbit_wars.generate_comet_paths(
            initial_planets,
            angular_velocity,
            spawn_step,
            [],
            comet_speed,
            rng=rng,
        )
        ships = min(rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99))
        spawn_ships = spawn_ships.at[schedule_i].set(float(ships))
        if paths is None:
            continue
        path_len = min(len(paths[0]), max_comet_path)
        spawn_available = spawn_available.at[schedule_i].set(True)
        spawn_path_lens = spawn_path_lens.at[schedule_i].set(path_len)
        path_arr = jnp.asarray(paths[:max_comet_planets], dtype=jnp.float64)
        spawn_paths = spawn_paths.at[schedule_i, : path_arr.shape[0], :path_len].set(path_arr[:, :path_len])
    return spawn_steps, spawn_available, spawn_paths, spawn_path_lens, spawn_ships


def env_state_from_observation(
    obs,
    *,
    episode_seed,
    max_planets=DEFAULT_MAX_PLANETS,
    max_fleets=256,
    max_comet_planets=DEFAULT_MAX_COMET_PLANETS,
    max_comet_path=DEFAULT_MAX_COMET_PATH,
    comet_speed=DEFAULT_COMET_SPEED,
) -> JaxEnvState:
    """Build a padded static-shape JAX env state from an official observation.

    This reset/observation wrapper intentionally uses the official Python comet
    path generator at reset time. The resulting schedule is static JAX data for
    later compiled stepping; Python RNG is not part of the hot path.
    """

    planets, active_planets = _padded_rows(obs.planets, shape=(max_planets, 7), dtype=float)
    initial_planets, _active_initial = _padded_rows(obs.initial_planets, shape=(max_planets, 7), dtype=float)
    planet_active = jnp.arange(max_planets) < active_planets

    fleets, active_fleets = _padded_rows(getattr(obs, "fleets", []), shape=(max_fleets, 7), dtype=float)
    fleet_active = jnp.arange(max_fleets) < active_fleets

    comet_planet_ids, comet_paths, comet_path_index, comet_path_len = _current_comet_arrays(
        obs,
        max_comet_planets=max_comet_planets,
        max_comet_path=max_comet_path,
    )
    spawn_steps, spawn_available, spawn_paths, spawn_path_lens, spawn_ships = _precompute_comet_schedule(
        obs.initial_planets,
        obs.angular_velocity,
        episode_seed,
        max_comet_planets=max_comet_planets,
        max_comet_path=max_comet_path,
        comet_speed=comet_speed,
    )

    return JaxEnvState(
        planets=planets,
        planet_active=planet_active,
        fleets=fleets,
        fleet_active=fleet_active,
        initial_planets=initial_planets,
        next_fleet_id=jnp.asarray(obs.next_fleet_id, dtype=jnp.int64),
        step=jnp.asarray(obs.step, dtype=jnp.int64),
        angular_velocity=jnp.asarray(obs.angular_velocity, dtype=jnp.float64),
        comet_planet_ids=comet_planet_ids,
        comet_paths=comet_paths,
        comet_path_index=comet_path_index,
        comet_path_len=comet_path_len,
        spawn_steps=spawn_steps,
        spawn_available=spawn_available,
        spawn_paths=spawn_paths,
        spawn_path_lens=spawn_path_lens,
        spawn_ships=spawn_ships,
    )


def observation_from_env_state(state, *, player: int) -> dict:
    """Convert a padded JAX env state to an official-like Python observation.

    This is a host-side bridge for parity tests and existing Python policy glue;
    it is intentionally not part of the compiled rollout hot path.
    """

    planet_active = jax.device_get(state.planet_active).astype(bool)
    fleet_active = jax.device_get(state.fleet_active).astype(bool)
    planets = jax.device_get(state.planets)[planet_active].tolist()
    fleets = jax.device_get(state.fleets)[fleet_active].tolist()
    initial_planets = jax.device_get(state.initial_planets)[planet_active].tolist()
    comet_ids_arr = jax.device_get(state.comet_planet_ids).astype(int)
    comet_path_len = int(jax.device_get(state.comet_path_len))
    active_comet_ids = [int(pid) for pid in comet_ids_arr.tolist() if int(pid) >= 0]
    comet_planet_ids = active_comet_ids if comet_path_len > 0 else []
    comets = []
    if comet_planet_ids:
        paths_arr = jax.device_get(state.comet_paths)[: len(comet_planet_ids), :comet_path_len, :]
        comets.append(
            {
                "planet_ids": comet_planet_ids,
                "paths": paths_arr.tolist(),
                "path_index": int(jax.device_get(state.comet_path_index)),
            }
        )
    return {
        "player": int(player),
        "step": int(jax.device_get(state.step)),
        "angular_velocity": float(jax.device_get(state.angular_velocity)),
        "planets": planets,
        "fleets": fleets,
        "initial_planets": initial_planets,
        "next_fleet_id": int(jax.device_get(state.next_fleet_id)),
        "comets": comets,
        "comet_planet_ids": comet_planet_ids,
    }


def _candidate_feature_vector(
    *,
    is_launch,
    angle,
    ships,
    distance,
    target_ships,
    target_production,
    target_is_neutral,
    target_is_own,
    target_is_enemy,
    source_x,
    source_y,
    target_x,
    target_y,
    target_radius,
    my_planet_count,
    enemy_planet_count,
    neutral_planet_count,
    my_total_ships,
    enemy_total_ships,
    neutral_total_ships,
    my_production,
    enemy_production,
    source_ships,
):
    sent_fraction = ships / jnp.maximum(1.0, source_ships)
    target_value_per_distance = (target_ships + 8.0 * target_production + 25.0) / (distance + 1.0) / 10.0
    return jnp.asarray(
        [
            is_launch,
            jnp.sin(angle),
            jnp.cos(angle),
            ships / 100.0,
            distance / 100.0,
            target_ships / 100.0,
            target_production / 10.0,
            target_is_neutral,
            target_is_own,
            target_is_enemy,
            my_planet_count / 10.0,
            enemy_planet_count / 10.0,
            neutral_planet_count / 10.0,
            my_total_ships / 500.0,
            enemy_total_ships / 500.0,
            neutral_total_ships / 500.0,
            my_production / 50.0,
            enemy_production / 50.0,
            source_ships / 100.0,
            source_x / 100.0,
            source_y / 100.0,
            target_x / 100.0,
            target_y / 100.0,
            target_radius / 10.0,
            sent_fraction,
            (ships - target_ships) / 100.0,
            source_ships / jnp.maximum(1.0, my_total_ships),
            target_value_per_distance,
            target_production / (distance + 1.0),
            (my_total_ships - enemy_total_ships) / 500.0,
            (my_production - enemy_production) / 50.0,
            1.0 / (1.0 + distance / 100.0),
        ],
        dtype=jnp.float64,
    )


def action_candidates_from_env_state(
    state,
    *,
    player: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
) -> JaxActionCandidates:
    """Build fixed-shape JAX action rows and 32-D policy features.

    This mirrors the safe-action candidate ordering for the no-waste-filter case:
    noop first, then owned source planets in table order, nearest targets, then
    configured send fractions. The output is padded with an `active` mask so it
    can be consumed by compiled policy/training code without variable-length
    Python candidate lists.
    """

    planets = jnp.asarray(state.planets, dtype=jnp.float64)
    planet_active = jnp.asarray(state.planet_active, dtype=jnp.bool_)
    player_id = jnp.asarray(player, dtype=jnp.int64)
    fractions = jnp.asarray((0.35, 0.55, 0.75) if send_fractions is None else send_fractions, dtype=jnp.float64)

    n_planets = planets.shape[0]
    n_frac = fractions.shape[0]
    max_candidates = 1 + n_planets * int(max_targets_per_source) * n_frac

    env_actions = jnp.zeros((max_candidates, 3), dtype=jnp.float64).at[0].set(jnp.asarray([-1.0, 0.0, 0.0]))
    active = jnp.zeros((max_candidates,), dtype=jnp.bool_).at[0].set(True)
    features = jnp.zeros((max_candidates, 32), dtype=jnp.float64)

    owners = planets[:, PLANET_COL_OWNER].astype(jnp.int64)
    ships = jnp.floor(planets[:, PLANET_COL_SHIPS]).astype(jnp.int64)
    prods = jnp.floor(planets[:, PLANET_COL_PROD]).astype(jnp.int64)
    owned_mask = planet_active & (owners == player_id)
    enemy_mask = planet_active & (owners != player_id) & (owners != -1)
    neutral_mask = planet_active & (owners == -1)

    my_planet_count = jnp.sum(owned_mask).astype(jnp.float64)
    enemy_planet_count = jnp.sum(enemy_mask).astype(jnp.float64)
    neutral_planet_count = jnp.sum(neutral_mask).astype(jnp.float64)
    my_total_ships = jnp.sum(jnp.where(owned_mask, ships, 0)).astype(jnp.float64)
    enemy_total_ships = jnp.sum(jnp.where(enemy_mask, ships, 0)).astype(jnp.float64)
    neutral_total_ships = jnp.sum(jnp.where(neutral_mask, ships, 0)).astype(jnp.float64)
    my_production = jnp.sum(jnp.where(owned_mask, prods, 0)).astype(jnp.float64)
    enemy_production = jnp.sum(jnp.where(enemy_mask, prods, 0)).astype(jnp.float64)

    noop_features = _candidate_feature_vector(
        is_launch=0.0,
        angle=0.0,
        ships=0.0,
        distance=0.0,
        target_ships=0.0,
        target_production=0.0,
        target_is_neutral=0.0,
        target_is_own=0.0,
        target_is_enemy=0.0,
        source_x=0.0,
        source_y=0.0,
        target_x=0.0,
        target_y=0.0,
        target_radius=0.0,
        my_planet_count=my_planet_count,
        enemy_planet_count=enemy_planet_count,
        neutral_planet_count=neutral_planet_count,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        neutral_total_ships=neutral_total_ships,
        my_production=my_production,
        enemy_production=enemy_production,
        source_ships=0.0,
    )
    features = features.at[0].set(noop_features)

    ids = planets[:, PLANET_COL_ID].astype(jnp.int64)
    xs = planets[:, PLANET_COL_X]
    ys = planets[:, PLANET_COL_Y]
    radii = planets[:, PLANET_COL_RADIUS]

    row = 1
    for source_i in range(n_planets):
        source_id = ids[source_i]
        source_ships_int = ships[source_i]
        source_ok = owned_mask[source_i] & (source_ships_int > 1)
        dx_all = xs - xs[source_i]
        dy_all = ys - ys[source_i]
        dist2 = dx_all * dx_all + dy_all * dy_all
        target_mask = planet_active & (ids != source_id)
        masked_dist = jnp.where(target_mask, dist2, jnp.inf)
        target_order = jnp.argsort(masked_dist)[: int(max_targets_per_source)]
        ship_counts = jnp.maximum(1, jnp.floor(source_ships_int.astype(jnp.float64) * jnp.maximum(0.0, fractions)).astype(jnp.int64))

        for rank_i in range(int(max_targets_per_source)):
            target_i = target_order[rank_i]
            target_ok = source_ok & jnp.isfinite(masked_dist[target_i])
            target_owner = owners[target_i]
            target_is_own = target_owner == player_id
            target_is_enemy = (target_owner != player_id) & (target_owner != -1)
            target_is_neutral = target_owner == -1

            for frac_i in range(n_frac):
                count = ship_counts[frac_i]
                unique_count = jnp.all(jnp.where(jnp.arange(n_frac) < frac_i, ship_counts != count, True))
                target_ships_int = ships[target_i]
                angle, _eta, distance = aim_with_prediction(
                    xs[source_i],
                    ys[source_i],
                    planets[target_i],
                    count.astype(jnp.float64),
                    state.angular_velocity,
                )
                required_capture = estimate_required_capture_ships_fixed_point(
                    target_ships_int.astype(jnp.float64),
                    jnp.where(target_is_enemy, prods[target_i].astype(jnp.float64), 0.0),
                    distance,
                )
                captures_target = jnp.where(target_is_own, count > target_ships_int, count >= required_capture)
                material_enemy_attrition = (count <= target_ships_int) & (count.astype(jnp.float64) >= jnp.maximum(
                    25.0,
                    0.25 * target_ships_int.astype(jnp.float64),
                ))
                strategic_ok = (
                    target_is_own
                    | (target_is_neutral & captures_target)
                    | (target_is_enemy & (captures_target | material_enemy_attrition))
                )
                is_active = target_ok & unique_count & strategic_ok
                row_features = _candidate_feature_vector(
                    is_launch=1.0,
                    angle=angle,
                    ships=count.astype(jnp.float64),
                    distance=distance,
                    target_ships=ships[target_i].astype(jnp.float64),
                    target_production=prods[target_i].astype(jnp.float64),
                    target_is_neutral=target_is_neutral.astype(jnp.float64),
                    target_is_own=target_is_own.astype(jnp.float64),
                    target_is_enemy=target_is_enemy.astype(jnp.float64),
                    source_x=xs[source_i],
                    source_y=ys[source_i],
                    target_x=xs[target_i],
                    target_y=ys[target_i],
                    target_radius=radii[target_i],
                    my_planet_count=my_planet_count,
                    enemy_planet_count=enemy_planet_count,
                    neutral_planet_count=neutral_planet_count,
                    my_total_ships=my_total_ships,
                    enemy_total_ships=enemy_total_ships,
                    neutral_total_ships=neutral_total_ships,
                    my_production=my_production,
                    enemy_production=enemy_production,
                    source_ships=source_ships_int.astype(jnp.float64),
                )
                env_actions = env_actions.at[row].set(jnp.asarray([source_id.astype(jnp.float64), angle, count.astype(jnp.float64)]))
                active = active.at[row].set(is_active)
                features = features.at[row].set(row_features)
                row += 1

    return JaxActionCandidates(env_actions=env_actions, active=active, features=features)


def select_action_from_env_state(
    state: JaxEnvState,
    *,
    player: int,
    actor_w,
    value_w,
    max_targets_per_source: int = 8,
    send_fractions=None,
) -> JaxSelectedAction:
    candidates = action_candidates_from_env_state(
        state,
        player=player,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
    )
    idx, logp, value, entropy = jax_policy.select_masked_argmax_candidate(
        candidates.features,
        candidates.active,
        actor_w,
        value_w,
    )
    return JaxSelectedAction(
        env_action=candidates.env_actions[idx],
        chosen_index=idx,
        logp=logp,
        value=value,
        entropy=entropy,
    )


def sample_action_from_env_state(
    key,
    state: JaxEnvState,
    *,
    player: int,
    actor_w,
    value_w,
    max_targets_per_source: int = 8,
    send_fractions=None,
) -> tuple[JaxSelectedAction, jax.Array]:
    candidates = action_candidates_from_env_state(
        state,
        player=player,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
    )
    idx, logp, value, entropy, next_key = jax_policy.sample_masked_candidate(
        key,
        candidates.features,
        candidates.active,
        actor_w,
        value_w,
    )
    return (
        JaxSelectedAction(
            env_action=candidates.env_actions[idx],
            chosen_index=idx,
            logp=logp,
            value=value,
            entropy=entropy,
        ),
        next_key,
    )


def select_action_and_step_env(
    state: JaxEnvState,
    *,
    player: int,
    actor_w,
    value_w,
    num_players: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
) -> tuple[JaxEnvStepResult, JaxSelectedAction]:
    """Select a deterministic fixed-candidate action for one player and step.

    This helper keeps the policy-selection-to-env-action bridge in JAX space for
    smoke testing compiled rollout slices. Other players receive no-op actions.
    """

    selected = select_action_from_env_state(
        state,
        player=player,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
    )
    action_array = jnp.zeros((int(num_players), 1, 3), dtype=jnp.float64).at[int(player), 0].set(selected.env_action)
    action_active = jnp.zeros((int(num_players), 1), dtype=jnp.bool_).at[int(player), 0].set(selected.env_action[0] >= 0.0)
    next_state = step_env(
        state,
        action_array,
        action_active,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    return next_state, selected


def sample_action_and_step_env(
    key,
    state: JaxEnvState,
    *,
    player: int,
    actor_w,
    value_w,
    num_players: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
) -> tuple[JaxEnvStepResult, JaxSelectedAction, jax.Array]:
    selected, next_key = sample_action_from_env_state(
        key,
        state,
        player=player,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
    )
    action_array = jnp.zeros((int(num_players), 1, 3), dtype=jnp.float64).at[int(player), 0].set(selected.env_action)
    action_active = jnp.zeros((int(num_players), 1), dtype=jnp.bool_).at[int(player), 0].set(selected.env_action[0] >= 0.0)
    next_state = step_env(
        state,
        action_array,
        action_active,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    return next_state, selected, next_key


def fleet_speed(ships) -> jax.Array:
    """JAX equivalent of :func:`src.ow.geometry.fleet_speed` for valid inputs."""

    ships = jnp.asarray(ships, dtype=jnp.float64)
    raw = 1.0 + (SHIP_SPEED_MAX - 1.0) * (jnp.log(ships) / jnp.log(1000.0)) ** 1.5
    clipped = jnp.clip(raw, 1.0, SHIP_SPEED_MAX)
    return jnp.where(ships <= 1.0, 1.0, clipped)


def distance_matrix(a, b) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    delta = a[:, None, :] - b[None, :, :]
    return jnp.linalg.norm(delta, axis=2)


def point_to_segment_distance(p, v, w) -> jax.Array:
    """JAX equivalent of official Orbit Wars point-to-segment distance."""

    p = jnp.asarray(p, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    segment = w - v
    l2 = jnp.dot(segment, segment)
    t = jnp.dot(p - v, segment) / jnp.where(l2 == 0.0, 1.0, l2)
    t = jnp.clip(t, 0.0, 1.0)
    projection = v + t * segment
    projected_distance = jnp.linalg.norm(p - projection)
    point_distance = jnp.linalg.norm(p - v)
    return jnp.where(l2 == 0.0, point_distance, projected_distance)


def swept_pair_hit(a_start, a_end, p_start, p_end, radius) -> jax.Array:
    """JAX equivalent of official swept moving-fleet/moving-planet hit test."""

    a_start = jnp.asarray(a_start, dtype=jnp.float64)
    a_end = jnp.asarray(a_end, dtype=jnp.float64)
    p_start = jnp.asarray(p_start, dtype=jnp.float64)
    p_end = jnp.asarray(p_end, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)

    d0 = a_start - p_start
    dv = (a_end - a_start) - (p_end - p_start)
    qa = jnp.dot(dv, dv)
    qb = 2.0 * jnp.dot(d0, dv)
    qc = jnp.dot(d0, d0) - radius * radius
    near_static = qa < 1e-12
    disc = qb * qb - 4.0 * qa * qc
    safe_disc = jnp.maximum(disc, 0.0)
    safe_den = jnp.where(near_static, 1.0, 2.0 * qa)
    root = jnp.sqrt(safe_disc)
    t1 = (-qb - root) / safe_den
    t2 = (-qb + root) / safe_den
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    static_hit = qc <= 0.0
    return jnp.where(near_static, static_hit, moving_hit)


def segment_hits_circle(p0, p1, center, radius) -> jax.Array:
    """Return whether segment p0-p1 intersects a circle, branchlessly in JAX."""

    p0 = jnp.asarray(p0, dtype=jnp.float64)
    p1 = jnp.asarray(p1, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)

    d = p1 - p0
    dd = jnp.dot(d, d)
    t = jnp.where(dd == 0.0, 0.0, jnp.dot(center - p0, d) / dd)
    t = jnp.clip(t, 0.0, 1.0)
    closest = p0 + t * d
    dist2 = jnp.dot(closest - center, closest - center)
    return dist2 <= radius * radius


def predict_planet_position(planet_row, turns, angular_velocity) -> tuple[jax.Array, jax.Array]:
    """JAX equivalent of rotating/static planet prediction for valid rows."""

    row = jnp.asarray(planet_row, dtype=jnp.float64)
    turns = jnp.asarray(turns, dtype=jnp.float64)
    angular_velocity = jnp.asarray(angular_velocity, dtype=jnp.float64)

    x = row[PLANET_COL_X]
    y = row[PLANET_COL_Y]
    radius = row[PLANET_COL_RADIUS]
    dx = x - CENTER_X
    dy = y - CENTER_Y
    orbital_r = jnp.hypot(dx, dy)
    theta = angular_velocity * turns
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    rotated_x = CENTER_X + dx * ct - dy * st
    rotated_y = CENTER_Y + dx * st + dy * ct
    is_static = orbital_r + radius >= ROTATION_RADIUS_LIMIT
    return jnp.where(is_static, x, rotated_x), jnp.where(is_static, y, rotated_y)


def estimate_required_capture_ships(target_ships, target_prod, eta, margin=2) -> jax.Array:
    target_ships = jnp.asarray(target_ships, dtype=jnp.float64)
    target_prod = jnp.asarray(target_prod, dtype=jnp.float64)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    margin = jnp.asarray(margin, dtype=jnp.float64)
    defenders = target_ships + target_prod * jnp.ceil(eta)
    return jnp.ceil(defenders + margin + 1.0).astype(jnp.int64)


def estimate_required_capture_ships_fixed_point(target_ships, target_prod, distance, margin=2, max_iter=6) -> jax.Array:
    """JAX binary-search equivalent of fixed-point capture requirement."""

    del max_iter  # Python reference keeps this for backwards compatibility too.
    target_ships = jnp.asarray(target_ships, dtype=jnp.float64)
    target_prod = jnp.asarray(target_prod, dtype=jnp.float64)
    distance = jnp.asarray(distance, dtype=jnp.float64)
    margin = jnp.asarray(margin, dtype=jnp.float64)

    def required_for(ship_count):
        ship_count = jnp.maximum(1, ship_count).astype(jnp.float64)
        eta = distance / fleet_speed(ship_count)
        return estimate_required_capture_ships(target_ships, target_prod, eta, margin=margin)

    low0 = jnp.asarray(1, dtype=jnp.int64)
    upper0 = jnp.maximum(low0, required_for(low0))

    def cond(carry):
        low, upper = carry
        return low < upper

    def body(carry):
        low, upper = carry
        mid = (low + upper) // 2
        req = required_for(mid)
        return jnp.where(mid >= req, low, mid + 1), jnp.where(mid >= req, mid, upper)

    low, _upper = lax.while_loop(cond, body, (low0, upper0))
    return low


def aim_with_prediction(source_x, source_y, target_row, ships, angular_velocity, max_iter=6) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX equivalent of iterative lead-angle prediction for valid inputs."""

    source_x = jnp.asarray(source_x, dtype=jnp.float64)
    source_y = jnp.asarray(source_y, dtype=jnp.float64)
    target_row = jnp.asarray(target_row, dtype=jnp.float64)
    ships = jnp.asarray(ships, dtype=jnp.float64)
    angular_velocity = jnp.asarray(angular_velocity, dtype=jnp.float64)

    tx0, ty0 = predict_planet_position(target_row, turns=0.0, angular_velocity=angular_velocity)
    dist0 = jnp.hypot(tx0 - source_x, ty0 - source_y)
    speed = fleet_speed(ships)
    eta0 = dist0 / speed

    def cond(carry):
        i, _tx, _ty, _dist, eta, prev_eta, converged = carry
        return (i < int(max_iter)) & (~converged) & (jnp.abs(eta - prev_eta) > 1e-6)

    def body(carry):
        i, _tx, _ty, _dist, eta, _prev_eta, _converged = carry
        tx, ty = predict_planet_position(target_row, turns=eta, angular_velocity=angular_velocity)
        new_dist = jnp.hypot(tx - source_x, ty - source_y)
        new_eta = new_dist / speed
        converged = jnp.abs(new_eta - eta) <= 1e-6
        return i + 1, tx, ty, new_dist, new_eta, eta, converged

    init = (
        jnp.asarray(0, dtype=jnp.int64),
        tx0,
        ty0,
        dist0,
        eta0,
        eta0 + 2e-6,
        jnp.asarray(False),
    )
    _i, tx, ty, dist, eta, _prev_eta, _converged = lax.while_loop(cond, body, init)
    angle = jnp.atan2(ty - source_y, tx - source_x)
    return angle, eta, dist


def keep_needed_by_planet(planets, player, horizon) -> jax.Array:
    planets = jnp.asarray(planets, dtype=jnp.float64)
    player = jnp.asarray(player, dtype=jnp.int64)
    horizon = jnp.asarray(horizon, dtype=jnp.float64)

    owners = planets[:, PLANET_COL_OWNER].astype(jnp.int64)
    owned = owners == player
    available = jnp.floor(planets[:, PLANET_COL_SHIPS]).astype(jnp.int64)
    horizon_factor = jnp.minimum(1.0, horizon / 500.0)
    base = jnp.floor(planets[:, PLANET_COL_PROD] * 0.5 * horizon_factor).astype(jnp.int64)
    base = jnp.maximum(0, base)
    cap = jnp.maximum(0, available - 1)
    return jnp.where(owned, jnp.minimum(base, cap), 0).astype(jnp.int64)


def _planet_positions_after_tick(planets, initial_planets, step, angular_velocity):
    """Official no-comet planet movement for one interpreter tick."""

    planet_ids = planets[:, PLANET_COL_ID]
    initial_ids = initial_planets[:, PLANET_COL_ID]

    def moved_position(planet):
        pid = planet[PLANET_COL_ID]
        matches = initial_ids == pid
        init_idx = jnp.argmax(matches)
        init = initial_planets[init_idx]
        has_initial = jnp.any(matches)
        dx = init[PLANET_COL_X] - CENTER_X
        dy = init[PLANET_COL_Y] - CENTER_Y
        orbital_r = jnp.hypot(dx, dy)
        initial_angle = jnp.atan2(dy, dx)
        current_angle = initial_angle + angular_velocity * step.astype(jnp.float64)
        rotated_x = CENTER_X + orbital_r * jnp.cos(current_angle)
        rotated_y = CENTER_Y + orbital_r * jnp.sin(current_angle)
        rotates = has_initial & (orbital_r + planet[PLANET_COL_RADIUS] < ROTATION_RADIUS_LIMIT)
        return jnp.asarray(
            [
                jnp.where(rotates, rotated_x, planet[PLANET_COL_X]),
                jnp.where(rotates, rotated_y, planet[PLANET_COL_Y]),
            ],
            dtype=jnp.float64,
        )

    del planet_ids
    return jax.vmap(moved_position)(planets)


def _launch_fleets(planets, fleets, fleet_active, next_fleet_id, action_array, action_active):
    num_players, max_actions, _ = action_array.shape

    def launch_one(carry, flat_i):
        planets, fleets, fleet_active, next_fleet_id = carry
        player = flat_i // max_actions
        action_i = flat_i % max_actions
        action = action_array[player, action_i]
        from_id = action[0]
        angle = action[1]
        ships = jnp.floor(action[2])

        source_mask = planets[:, PLANET_COL_ID] == from_id
        source_idx = jnp.argmax(source_mask)
        source = planets[source_idx]
        source_valid = jnp.any(source_mask) & (source[PLANET_COL_OWNER].astype(jnp.int64) == player)
        ships_valid = (ships > 0.0) & (source[PLANET_COL_SHIPS] >= ships)
        free_mask = ~fleet_active
        slot = jnp.argmax(free_mask)
        valid = action_active[player, action_i] & source_valid & ships_valid & jnp.any(free_mask)

        updated_source = source.at[PLANET_COL_SHIPS].add(jnp.where(valid, -ships, 0.0))
        planets = planets.at[source_idx].set(jnp.where(valid, updated_source, source))

        start_x = source[PLANET_COL_X] + jnp.cos(angle) * (source[PLANET_COL_RADIUS] + 0.1)
        start_y = source[PLANET_COL_Y] + jnp.sin(angle) * (source[PLANET_COL_RADIUS] + 0.1)
        new_fleet = jnp.asarray(
            [
                next_fleet_id.astype(jnp.float64),
                player.astype(jnp.float64),
                start_x,
                start_y,
                angle,
                from_id,
                ships,
            ],
            dtype=jnp.float64,
        )
        fleets = fleets.at[slot].set(jnp.where(valid, new_fleet, fleets[slot]))
        fleet_active = fleet_active.at[slot].set(jnp.where(valid, True, fleet_active[slot]))
        next_fleet_id = next_fleet_id + valid.astype(jnp.int64)
        return (planets, fleets, fleet_active, next_fleet_id), None

    (planets, fleets, fleet_active, next_fleet_id), _ = lax.scan(
        launch_one,
        (planets, fleets, fleet_active, next_fleet_id),
        jnp.arange(num_players * max_actions, dtype=jnp.int64),
    )
    return planets, fleets, fleet_active, next_fleet_id


def _move_fleets_and_collect_hits(planets, fleets, fleet_active, planet_new_xy, ship_speed, planet_collision_enabled=None):
    if planet_collision_enabled is None:
        planet_collision_enabled = jnp.ones((planets.shape[0],), dtype=jnp.bool_)
    old_xy = fleets[:, [FLEET_COL_X, FLEET_COL_Y]]
    speeds = fleet_speed(fleets[:, FLEET_COL_SHIPS])
    speeds = jnp.minimum(speeds, jnp.asarray(ship_speed, dtype=jnp.float64))
    new_x = fleets[:, FLEET_COL_X] + jnp.cos(fleets[:, FLEET_COL_ANGLE]) * speeds
    new_y = fleets[:, FLEET_COL_Y] + jnp.sin(fleets[:, FLEET_COL_ANGLE]) * speeds
    new_xy = jnp.stack([new_x, new_y], axis=1)

    def fleet_hits_planets(fleet_old_xy, fleet_new_xy):
        def hit_planet(planet, planet_xy1):
            planet_xy0 = jnp.stack([planet[PLANET_COL_X], planet[PLANET_COL_Y]])
            return swept_pair_hit(fleet_old_xy, fleet_new_xy, planet_xy0, planet_xy1, planet[PLANET_COL_RADIUS])

        hits = jax.vmap(hit_planet)(planets, planet_new_xy) & planet_collision_enabled
        return jnp.any(hits), jnp.argmax(hits)

    hit_any, hit_idx = jax.vmap(fleet_hits_planets)(old_xy, new_xy)
    in_bounds = (new_x >= 0.0) & (new_x <= BOARD_SIZE) & (new_y >= 0.0) & (new_y <= BOARD_SIZE)
    sun_dist = jax.vmap(lambda p0, p1: point_to_segment_distance(jnp.asarray([CENTER_X, CENTER_Y]), p0, p1))(old_xy, new_xy)
    crosses_sun = sun_dist < SUN_RADIUS
    remove = fleet_active & (hit_any | (~in_bounds) | crosses_sun)
    hit_active = fleet_active & hit_any

    fleets = fleets.at[:, FLEET_COL_X].set(jnp.where(fleet_active, new_x, fleets[:, FLEET_COL_X]))
    fleets = fleets.at[:, FLEET_COL_Y].set(jnp.where(fleet_active, new_y, fleets[:, FLEET_COL_Y]))
    fleet_active = fleet_active & (~remove)
    return fleets, fleet_active, hit_active, hit_idx


def _comet_planet_mask(planets, comet_planet_ids):
    planet_ids = planets[:, PLANET_COL_ID].astype(jnp.int64)
    return jnp.any(planet_ids[:, None] == comet_planet_ids[None, :], axis=1)


def _planet_positions_with_existing_comets(planets, initial_planets, step, angular_velocity, comet_planet_ids, comet_paths, comet_path_index, comet_path_len):
    base_new_xy = _planet_positions_after_tick(planets, initial_planets, step, angular_velocity)
    comet_mask = _comet_planet_mask(planets, comet_planet_ids)
    next_idx = comet_path_index + jnp.asarray(1, dtype=jnp.int64)
    path_idx = jnp.clip(next_idx, 0, comet_paths.shape[1] - 1)

    def comet_xy_for_planet(planet):
        pid = planet[PLANET_COL_ID].astype(jnp.int64)
        matches = comet_planet_ids == pid
        comet_i = jnp.argmax(matches)
        path_xy = comet_paths[comet_i, path_idx]
        old_xy = jnp.stack([planet[PLANET_COL_X], planet[PLANET_COL_Y]])
        in_path = next_idx < comet_path_len
        return jnp.where(in_path & jnp.any(matches), path_xy, old_xy)

    comet_new_xy = jax.vmap(comet_xy_for_planet)(planets)
    planet_new_xy = jnp.where(comet_mask[:, None], comet_new_xy, base_new_xy)
    first_comet_placement = comet_mask & (planets[:, PLANET_COL_X] < 0.0)
    collision_enabled = ~first_comet_placement
    return planet_new_xy, collision_enabled, next_idx


def _spawn_precomputed_comets(
    planets,
    planet_active,
    initial_planets,
    comet_planet_ids,
    comet_paths,
    comet_path_index,
    comet_path_len,
    state_step,
    spawn_active,
    spawn_step,
    spawn_paths,
    spawn_path_len,
    spawn_ships,
):
    should_spawn = spawn_active & ((state_step + jnp.asarray(1, dtype=jnp.int64)) == spawn_step)
    max_id = jnp.max(jnp.where(planet_active, planets[:, PLANET_COL_ID], -1.0)).astype(jnp.int64)

    def spawn_one(carry, i):
        planets, planet_active, initial_planets, comet_planet_ids = carry
        slot = jnp.argmax(~planet_active)
        pid = max_id + i + jnp.asarray(1, dtype=jnp.int64)
        planet = jnp.asarray(
            [
                pid.astype(jnp.float64),
                -1.0,
                -99.0,
                -99.0,
                1.0,
                spawn_ships,
                1.0,
            ],
            dtype=jnp.float64,
        )
        planets = planets.at[slot].set(jnp.where(should_spawn, planet, planets[slot]))
        initial_planets = initial_planets.at[slot].set(jnp.where(should_spawn, planet, initial_planets[slot]))
        planet_active = planet_active.at[slot].set(jnp.where(should_spawn, True, planet_active[slot]))
        comet_planet_ids = comet_planet_ids.at[i].set(jnp.where(should_spawn, pid, comet_planet_ids[i]))
        return (planets, planet_active, initial_planets, comet_planet_ids), None

    (planets, planet_active, initial_planets, comet_planet_ids), _ = lax.scan(
        spawn_one,
        (planets, planet_active, initial_planets, comet_planet_ids),
        jnp.arange(spawn_paths.shape[0], dtype=jnp.int64),
    )
    comet_paths = jnp.where(should_spawn, spawn_paths, comet_paths)
    comet_path_index = jnp.where(should_spawn, jnp.asarray(-1, dtype=jnp.int64), comet_path_index)
    comet_path_len = jnp.where(should_spawn, spawn_path_len, comet_path_len)
    return planets, planet_active, initial_planets, comet_planet_ids, comet_paths, comet_path_index, comet_path_len


def _resolve_combat(planets, fleets, hit_active, hit_idx, num_players):
    fleet_owners = fleets[:, FLEET_COL_OWNER].astype(jnp.int64)
    fleet_ships = fleets[:, FLEET_COL_SHIPS]

    def resolve_planet(i, planet):
        player_sums = jnp.asarray(
            [jnp.sum(jnp.where(hit_active & (hit_idx == i) & (fleet_owners == player), fleet_ships, 0.0)) for player in range(num_players)],
            dtype=jnp.float64,
        )
        has_ships = jnp.any(player_sums > 0.0)
        top_player = jnp.argmax(player_sums)
        top_ships = player_sums[top_player]
        second_ships = jnp.max(jnp.where(jnp.arange(num_players) == top_player, -1.0, player_sums))
        tied_top = jnp.sum(player_sums == top_ships) > 1
        survivor_ships = jnp.where(tied_top, 0.0, top_ships - jnp.maximum(second_ships, 0.0))
        survivor_ships = jnp.where(has_ships, survivor_ships, 0.0)
        same_owner = planet[PLANET_COL_OWNER].astype(jnp.int64) == top_player

        reinforced_ships = planet[PLANET_COL_SHIPS] + survivor_ships
        attacked_ships = planet[PLANET_COL_SHIPS] - survivor_ships
        captured = attacked_ships < 0.0
        new_owner = jnp.where(captured, top_player.astype(jnp.float64), planet[PLANET_COL_OWNER])
        new_ships = jnp.where(captured, jnp.abs(attacked_ships), attacked_ships)
        resolved = planet.at[PLANET_COL_OWNER].set(jnp.where(same_owner, planet[PLANET_COL_OWNER], new_owner))
        resolved = resolved.at[PLANET_COL_SHIPS].set(jnp.where(same_owner, reinforced_ships, new_ships))
        apply = survivor_ships > 0.0
        return jnp.where(apply, resolved, planet)

    return jax.vmap(resolve_planet)(jnp.arange(planets.shape[0], dtype=jnp.int64), planets)


def _termination_and_rewards(planets, fleets, fleet_active, step, episode_steps, num_players, planet_active=None):
    if planet_active is None:
        planet_active = jnp.ones((planets.shape[0],), dtype=jnp.bool_)
    planet_owners = planets[:, PLANET_COL_OWNER].astype(jnp.int64)
    fleet_owners = fleets[:, FLEET_COL_OWNER].astype(jnp.int64)
    alive = jnp.asarray(
        [
            jnp.any(planet_active & (planet_owners == player)) | jnp.any(fleet_active & (fleet_owners == player))
            for player in range(num_players)
        ],
        dtype=jnp.bool_,
    )
    done = (jnp.sum(alive.astype(jnp.int64)) <= 1) | (step >= (jnp.asarray(episode_steps, dtype=jnp.int64) - 2))
    scores = jnp.asarray(
        [
            jnp.sum(jnp.where(planet_active & (planet_owners == player), planets[:, PLANET_COL_SHIPS], 0.0))
            + jnp.sum(jnp.where(fleet_active & (fleet_owners == player), fleets[:, FLEET_COL_SHIPS], 0.0))
            for player in range(num_players)
        ],
        dtype=jnp.float64,
    )
    max_score = jnp.max(scores)
    terminal_rewards = jnp.where((scores == max_score) & (max_score > 0.0), 1, -1).astype(jnp.int64)
    return done, jnp.where(done, terminal_rewards, jnp.zeros((num_players,), dtype=jnp.int64))


def step_no_comets(state: JaxTickState, action_array, action_active, *, episode_steps, ship_speed=SHIP_SPEED_MAX) -> JaxTickResult:
    """One official Orbit Wars interpreter tick excluding comet spawn/path logic.

    The inputs use fixed-size fleet arrays plus a mask so this can evolve into a
    jitted/vmapped rollout path. Comet behavior is intentionally out of scope for
    this helper and must be parity-tested separately before full-env claims.
    """

    action_array = jnp.asarray(action_array, dtype=jnp.float64)
    action_active = jnp.asarray(action_active, dtype=jnp.bool_)
    num_players = action_array.shape[0]

    planets, fleets, fleet_active, next_fleet_id = _launch_fleets(
        state.planets,
        state.fleets,
        state.fleet_active,
        state.next_fleet_id,
        action_array,
        action_active,
    )
    planets = planets.at[:, PLANET_COL_SHIPS].set(
        jnp.where(planets[:, PLANET_COL_OWNER] != -1.0, planets[:, PLANET_COL_SHIPS] + planets[:, PLANET_COL_PROD], planets[:, PLANET_COL_SHIPS])
    )
    planet_new_xy = _planet_positions_after_tick(planets, state.initial_planets, state.step, state.angular_velocity)
    fleets, fleet_active, hit_active, hit_idx = _move_fleets_and_collect_hits(planets, fleets, fleet_active, planet_new_xy, ship_speed)
    planets = planets.at[:, PLANET_COL_X].set(planet_new_xy[:, 0])
    planets = planets.at[:, PLANET_COL_Y].set(planet_new_xy[:, 1])
    planets = _resolve_combat(planets, fleets, hit_active, hit_idx, num_players)
    done, rewards = _termination_and_rewards(planets, fleets, fleet_active, state.step, episode_steps, num_players)
    return JaxTickResult(
        planets=planets,
        fleets=fleets,
        fleet_active=fleet_active,
        initial_planets=state.initial_planets,
        next_fleet_id=next_fleet_id,
        step=state.step + 1,
        rewards=rewards,
        done=done,
    )


def step_with_comets(state: JaxCometTickState, action_array, action_active, *, episode_steps, ship_speed=SHIP_SPEED_MAX) -> JaxCometTickResult:
    """One interpreter tick for an already-active comet group.

    This is still an incremental parity slice: spawn scheduling and fixed-slot
    expiry masks are tested separately before full environment claims. Existing
    comet movement/collision ordering matches the official interpreter.
    """

    action_array = jnp.asarray(action_array, dtype=jnp.float64)
    action_active = jnp.asarray(action_active, dtype=jnp.bool_)
    num_players = action_array.shape[0]
    (
        planets,
        planet_active,
        initial_planets,
        comet_planet_ids,
        comet_paths,
        comet_path_index,
        comet_path_len,
    ) = _spawn_precomputed_comets(
        state.planets,
        state.planet_active,
        state.initial_planets,
        state.comet_planet_ids,
        state.comet_paths,
        state.comet_path_index,
        state.comet_path_len,
        state.step,
        state.spawn_active,
        state.spawn_step,
        state.spawn_paths,
        state.spawn_path_len,
        state.spawn_ships,
    )

    planets, fleets, fleet_active, next_fleet_id = _launch_fleets(
        planets,
        state.fleets,
        state.fleet_active,
        state.next_fleet_id,
        action_array,
        action_active,
    )
    planets = planets.at[:, PLANET_COL_SHIPS].set(
        jnp.where(planets[:, PLANET_COL_OWNER] != -1.0, planets[:, PLANET_COL_SHIPS] + planets[:, PLANET_COL_PROD], planets[:, PLANET_COL_SHIPS])
    )
    planet_new_xy, collision_enabled, next_comet_path_index = _planet_positions_with_existing_comets(
        planets,
        initial_planets,
        state.step,
        state.angular_velocity,
        comet_planet_ids,
        comet_paths,
        comet_path_index,
        comet_path_len,
    )
    pre_expiry_planet_active = planet_active
    fleets, fleet_active, hit_active, hit_idx = _move_fleets_and_collect_hits(
        planets,
        fleets,
        fleet_active,
        planet_new_xy,
        ship_speed,
        collision_enabled & pre_expiry_planet_active,
    )
    planets = planets.at[:, PLANET_COL_X].set(planet_new_xy[:, 0])
    planets = planets.at[:, PLANET_COL_Y].set(planet_new_xy[:, 1])
    expired = _comet_planet_mask(planets, comet_planet_ids) & (next_comet_path_index >= comet_path_len)
    planet_active = pre_expiry_planet_active & (~expired)
    comet_group_expired = (comet_path_len > 0) & (next_comet_path_index >= comet_path_len)
    comet_planet_ids = jnp.where(comet_group_expired, jnp.full_like(comet_planet_ids, -1), comet_planet_ids)
    comet_paths = jnp.where(comet_group_expired, jnp.zeros_like(comet_paths), comet_paths)
    next_comet_path_index = jnp.where(comet_group_expired, jnp.asarray(-1, dtype=jnp.int64), next_comet_path_index)
    comet_path_len = jnp.where(comet_group_expired, jnp.asarray(0, dtype=jnp.int64), comet_path_len)
    hit_active_for_combat = hit_active & planet_active[hit_idx]
    planets = _resolve_combat(planets, fleets, hit_active_for_combat, hit_idx, num_players)
    done, rewards = _termination_and_rewards(planets, fleets, fleet_active, state.step, episode_steps, num_players, planet_active)
    return JaxCometTickResult(
        planets=planets,
        planet_active=planet_active,
        fleets=fleets,
        fleet_active=fleet_active,
        initial_planets=initial_planets,
        next_fleet_id=next_fleet_id,
        step=state.step + 1,
        rewards=rewards,
        done=done,
        comet_planet_ids=comet_planet_ids,
        comet_paths=comet_paths,
        comet_path_index=next_comet_path_index,
        comet_path_len=comet_path_len,
    )


def step_env(state: JaxEnvState, action_array, action_active, *, episode_steps, ship_speed=SHIP_SPEED_MAX) -> JaxEnvStepResult:
    """Step a padded env state using its precomputed comet spawn schedule."""

    spawn_match = state.spawn_available & (state.spawn_steps == (state.step + jnp.asarray(1, dtype=jnp.int64)))
    spawn_i = jnp.argmax(spawn_match)
    tick_state = JaxCometTickState.from_arrays(
        planets=state.planets,
        planet_active=state.planet_active,
        fleets=state.fleets,
        fleet_active=state.fleet_active,
        initial_planets=state.initial_planets,
        next_fleet_id=state.next_fleet_id,
        step=state.step,
        angular_velocity=state.angular_velocity,
        comet_planet_ids=state.comet_planet_ids,
        comet_paths=state.comet_paths,
        comet_path_index=state.comet_path_index,
        comet_path_len=state.comet_path_len,
        spawn_active=jnp.any(spawn_match),
        spawn_step=state.spawn_steps[spawn_i],
        spawn_paths=state.spawn_paths[spawn_i],
        spawn_path_len=state.spawn_path_lens[spawn_i],
        spawn_ships=state.spawn_ships[spawn_i],
    )
    result = step_with_comets(
        tick_state,
        action_array,
        action_active,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    return JaxEnvStepResult(
        planets=result.planets,
        planet_active=result.planet_active,
        fleets=result.fleets,
        fleet_active=result.fleet_active,
        initial_planets=result.initial_planets,
        next_fleet_id=result.next_fleet_id,
        step=result.step,
        angular_velocity=state.angular_velocity,
        comet_planet_ids=result.comet_planet_ids,
        comet_paths=result.comet_paths,
        comet_path_index=result.comet_path_index,
        comet_path_len=result.comet_path_len,
        spawn_steps=state.spawn_steps,
        spawn_available=state.spawn_available,
        spawn_paths=state.spawn_paths,
        spawn_path_lens=state.spawn_path_lens,
        spawn_ships=state.spawn_ships,
        rewards=result.rewards,
        done=result.done,
    )


def _env_state_as_step_result(state, num_players):
    return JaxEnvStepResult(
        planets=state.planets,
        planet_active=state.planet_active,
        fleets=state.fleets,
        fleet_active=state.fleet_active,
        initial_planets=state.initial_planets,
        next_fleet_id=state.next_fleet_id,
        step=state.step,
        angular_velocity=state.angular_velocity,
        comet_planet_ids=state.comet_planet_ids,
        comet_paths=state.comet_paths,
        comet_path_index=state.comet_path_index,
        comet_path_len=state.comet_path_len,
        spawn_steps=state.spawn_steps,
        spawn_available=state.spawn_available,
        spawn_paths=state.spawn_paths,
        spawn_path_lens=state.spawn_path_lens,
        spawn_ships=state.spawn_ships,
        rewards=jnp.zeros((num_players,), dtype=jnp.int64),
        done=jnp.asarray(False),
    )


def rollout_env(state: JaxEnvState, action_arrays, action_active, *, episode_steps, ship_speed=SHIP_SPEED_MAX):
    """Scan a fixed action stream through the padded JAX env state."""

    action_arrays = jnp.asarray(action_arrays, dtype=jnp.float64)
    action_active = jnp.asarray(action_active, dtype=jnp.bool_)
    num_players = action_arrays.shape[1]
    init = _env_state_as_step_result(state, num_players)

    def body(carry, xs):
        actions_t, active_t = xs
        next_state = step_env(carry, actions_t, active_t, episode_steps=episode_steps, ship_speed=ship_speed)
        return next_state, (next_state.rewards, next_state.done)

    final_state, (rewards, dones) = lax.scan(body, init, (action_arrays, action_active))
    return final_state, rewards, dones


def rollout_env_with_policy(
    state: JaxEnvState,
    *,
    player: int,
    actor_w,
    value_w,
    num_players: int,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Scan deterministic JAX policy selection and env stepping for one player."""

    init = _env_state_as_step_result(state, int(num_players))

    def body(carry, _t):
        next_state, selected = select_action_and_step_env(
            carry,
            player=player,
            actor_w=actor_w,
            value_w=value_w,
            num_players=num_players,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
        return next_state, (
            next_state.rewards,
            next_state.done,
            selected.chosen_index,
            selected.env_action,
        )

    final_state, (rewards, dones, chosen_indices, env_actions) = lax.scan(
        body,
        init,
        jnp.arange(int(num_steps), dtype=jnp.int64),
    )
    return final_state, rewards, dones, chosen_indices, env_actions


def rollout_env_with_sampling(
    key,
    state: JaxEnvState,
    *,
    player: int,
    actor_w,
    value_w,
    num_players: int,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Scan stochastic JAX policy selection and env stepping for one player."""

    init_state = _env_state_as_step_result(state, int(num_players))

    def body(carry, _t):
        state_carry, key_carry = carry
        next_state, selected, next_key = sample_action_and_step_env(
            key_carry,
            state_carry,
            player=player,
            actor_w=actor_w,
            value_w=value_w,
            num_players=num_players,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
        return (next_state, next_key), (
            next_state.rewards,
            next_state.done,
            selected.chosen_index,
            selected.env_action,
        )

    (final_state, next_key), (rewards, dones, chosen_indices, env_actions) = lax.scan(
        body,
        (init_state, key),
        jnp.arange(int(num_steps), dtype=jnp.int64),
    )
    return final_state, rewards, dones, chosen_indices, env_actions, next_key


def sample_all_actions_and_step_env(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
) -> tuple[JaxEnvStepResult, jax.Array, jax.Array, jax.Array]:
    """Sample one fixed-candidate action per player from a shared state, then step."""

    keys = jnp.asarray(keys)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)
    num_players = int(keys.shape[0])

    actions = []
    active = []
    chosen_indices = []
    next_keys = []
    for player in range(num_players):
        selected, next_key = sample_action_from_env_state(
            keys[player],
            state,
            player=player,
            actor_w=actor_w[player],
            value_w=value_w[player],
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
        )
        actions.append(selected.env_action)
        active.append(selected.env_action[0] >= 0.0)
        chosen_indices.append(selected.chosen_index)
        next_keys.append(next_key)

    action_array = jnp.stack(actions)[:, None, :]
    action_active = jnp.stack(active)[:, None]
    next_state = step_env(
        state,
        action_array,
        action_active,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    return next_state, jnp.stack(chosen_indices), jnp.stack(actions), jnp.stack(next_keys)


def rollout_env_with_multiagent_sampling(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Scan stochastic policy selection for every player and step the env."""

    keys = jnp.asarray(keys)
    init_state = _env_state_as_step_result(state, int(keys.shape[0]))

    def body(carry, _t):
        state_carry, keys_carry = carry
        next_state, chosen_indices, env_actions, next_keys = sample_all_actions_and_step_env(
            keys_carry,
            state_carry,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
        return (next_state, next_keys), (
            next_state.rewards,
            next_state.done,
            chosen_indices,
            env_actions,
        )

    (final_state, next_keys), (rewards, dones, chosen_indices, env_actions) = lax.scan(
        body,
        (init_state, keys),
        jnp.arange(int(num_steps), dtype=jnp.int64),
    )
    return final_state, rewards, dones, chosen_indices, env_actions, next_keys


def sample_all_actions_and_step_env_with_stats(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Sample all players, step, and keep policy outputs for RL batches."""

    keys = jnp.asarray(keys)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)
    num_players = int(keys.shape[0])

    actions = []
    active = []
    chosen_indices = []
    logps = []
    values = []
    entropies = []
    next_keys = []
    for player in range(num_players):
        selected, next_key = sample_action_from_env_state(
            keys[player],
            state,
            player=player,
            actor_w=actor_w[player],
            value_w=value_w[player],
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
        )
        actions.append(selected.env_action)
        active.append(selected.env_action[0] >= 0.0)
        chosen_indices.append(selected.chosen_index)
        logps.append(selected.logp)
        values.append(selected.value)
        entropies.append(selected.entropy)
        next_keys.append(next_key)

    next_state = step_env(
        state,
        jnp.stack(actions)[:, None, :],
        jnp.stack(active)[:, None],
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    return (
        next_state,
        jnp.stack(chosen_indices),
        jnp.stack(actions),
        jnp.stack(logps),
        jnp.stack(values),
        jnp.stack(entropies),
        jnp.stack(next_keys),
    )


def sample_all_actions_and_step_env_with_training_data(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Sample all players, step, and keep fixed candidate tensors for updates."""

    keys = jnp.asarray(keys)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)
    num_players = int(keys.shape[0])

    actions = []
    active = []
    candidate_features = []
    candidate_active = []
    chosen_indices = []
    logps = []
    values = []
    entropies = []
    next_keys = []
    for player in range(num_players):
        candidates = action_candidates_from_env_state(
            state,
            player=player,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
        )
        idx, logp, value, entropy, next_key = jax_policy.sample_masked_candidate(
            keys[player], candidates.features, candidates.active, actor_w[player], value_w[player]
        )
        env_action = candidates.env_actions[idx]
        actions.append(env_action)
        active.append(env_action[0] >= 0.0)
        candidate_features.append(candidates.features)
        candidate_active.append(candidates.active)
        chosen_indices.append(idx)
        logps.append(logp)
        values.append(value)
        entropies.append(entropy)
        next_keys.append(next_key)

    next_state = step_env(
        state,
        jnp.stack(actions)[:, None, :],
        jnp.stack(active)[:, None],
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    return (
        next_state,
        jnp.stack(chosen_indices),
        jnp.stack(actions),
        jnp.stack(candidate_features),
        jnp.stack(candidate_active),
        jnp.stack(logps),
        jnp.stack(values),
        jnp.stack(entropies),
        jnp.stack(next_keys),
    )


def rollout_env_with_multiagent_sampling_stats(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Scan all-player stochastic rollout and collect PPO-ready policy stats."""

    keys = jnp.asarray(keys)
    init_state = _env_state_as_step_result(state, int(keys.shape[0]))

    def body(carry, _t):
        state_carry, keys_carry = carry
        next_state, chosen_indices, env_actions, logps, values, entropies, next_keys = sample_all_actions_and_step_env_with_stats(
            keys_carry,
            state_carry,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
        return (next_state, next_keys), (
            next_state.rewards,
            next_state.done,
            chosen_indices,
            env_actions,
            logps,
            values,
            entropies,
        )

    (final_state, next_keys), (rewards, dones, chosen_indices, env_actions, logps, values, entropies) = lax.scan(
        body,
        (init_state, keys),
        jnp.arange(int(num_steps), dtype=jnp.int64),
    )
    return final_state, rewards, dones, chosen_indices, env_actions, logps, values, entropies, next_keys


def rollout_env_with_multiagent_training_data(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Scan all-player stochastic rollout and collect fixed candidate update tensors."""

    keys = jnp.asarray(keys)
    init_state = _env_state_as_step_result(state, int(keys.shape[0]))

    def body(carry, _t):
        state_carry, keys_carry = carry
        (
            next_state,
            chosen_indices,
            env_actions,
            candidate_features,
            candidate_active,
            logps,
            values,
            entropies,
            next_keys,
        ) = sample_all_actions_and_step_env_with_training_data(
            keys_carry,
            state_carry,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
        return (next_state, next_keys), (
            next_state.rewards,
            next_state.done,
            chosen_indices,
            env_actions,
            candidate_features,
            candidate_active,
            logps,
            values,
            entropies,
        )

    (
        (final_state, next_keys),
        (rewards, dones, chosen_indices, env_actions, candidate_features, candidate_active, logps, values, entropies),
    ) = lax.scan(
        body,
        (init_state, keys),
        jnp.arange(int(num_steps), dtype=jnp.int64),
    )
    return (
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
    )


def _multiagent_bootstrap_values(
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    max_targets_per_source: int = 8,
    send_fractions=None,
):
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)
    bootstrap_values = []
    for player in range(int(actor_w.shape[0])):
        candidates = action_candidates_from_env_state(
            state,
            player=player,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
        )
        _probs, value, _entropy = jax_policy.masked_policy_outputs(
            candidates.features,
            candidates.active,
            actor_w[player],
            value_w[player],
        )
        bootstrap_values.append(value)
    return jnp.stack(bootstrap_values)


def multiagent_ppo_train_step(
    keys,
    state: JaxEnvState,
    *,
    actor_w,
    value_w,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
    gamma=0.99,
    lr=0.01,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    """Run an all-player sampled rollout and one masked PPO update."""

    (
        final_state,
        rewards,
        dones,
        chosen_indices,
        _env_actions,
        candidate_features,
        candidate_active,
        logps,
        values,
        _entropies,
        next_keys,
    ) = rollout_env_with_multiagent_training_data(
        keys,
        state,
        actor_w=actor_w,
        value_w=value_w,
        num_steps=num_steps,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    bootstrap_values = _multiagent_bootstrap_values(
        final_state,
        actor_w=actor_w,
        value_w=value_w,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
    )
    next_actor_w, next_value_w, returns, advantages, metrics = jax_policy.multiagent_ppo_update_from_rollout(
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
        gamma=gamma,
        lr=lr,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
    )
    return final_state, next_actor_w, next_value_w, returns, advantages, metrics, next_keys


def batched_rollout_env(state_batch, action_arrays, action_active, *, episode_steps, ship_speed=SHIP_SPEED_MAX):
    """Vectorize fixed-stream rollouts across a leading environment batch."""

    return jax.vmap(
        lambda state, actions, active: rollout_env(
            state,
            actions,
            active,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
    )(state_batch, action_arrays, action_active)


def batched_rollout_env_with_multiagent_training_data(
    keys_batch,
    state_batch,
    *,
    actor_w,
    value_w,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
):
    """Vectorize all-player sampled training-data rollouts across envs."""

    return jax.vmap(
        lambda keys, state: rollout_env_with_multiagent_training_data(
            keys,
            state,
            actor_w=actor_w,
            value_w=value_w,
            num_steps=num_steps,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
        )
    )(keys_batch, state_batch)


def batched_multiagent_ppo_train_step(
    keys_batch,
    state_batch,
    *,
    actor_w,
    value_w,
    num_steps: int,
    max_targets_per_source: int = 8,
    send_fractions=None,
    episode_steps,
    ship_speed=SHIP_SPEED_MAX,
    gamma=0.99,
    lr=0.01,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    """Run batched all-player rollouts and one shared masked PPO update."""

    (
        final_state,
        rewards,
        dones,
        chosen_indices,
        _env_actions,
        candidate_features,
        candidate_active,
        logps,
        values,
        _entropies,
        next_keys,
    ) = batched_rollout_env_with_multiagent_training_data(
        keys_batch,
        state_batch,
        actor_w=actor_w,
        value_w=value_w,
        num_steps=num_steps,
        max_targets_per_source=max_targets_per_source,
        send_fractions=send_fractions,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
    )
    bootstrap_values = jax.vmap(
        lambda state: _multiagent_bootstrap_values(
            state,
            actor_w=actor_w,
            value_w=value_w,
            max_targets_per_source=max_targets_per_source,
            send_fractions=send_fractions,
        )
    )(final_state)
    next_actor_w, next_value_w, returns, advantages, metrics = jax_policy.batched_multiagent_ppo_update_from_rollout(
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
        gamma=gamma,
        lr=lr,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
    )
    return final_state, next_actor_w, next_value_w, returns, advantages, metrics, next_keys
