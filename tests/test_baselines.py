import math

import pytest

from baselines.greedy_sample import agent as greedy_agent
from baselines.random_agent import agent as random_agent
from baselines.style_bots import (
    greedy_expand_agent,
    idle_heavy_agent,
    rush_agent,
    swarm_agent,
    turtle_agent,
)


def _obs(planets, player=0, step=0):
    return {"player": player, "step": step, "planets": planets, "fleets": [], "comets": []}


def _assert_legal_moves(obs, moves):
    assert isinstance(moves, list)
    planets = {int(p[0]): p for p in obs["planets"]}
    owned = {int(p[0]) for p in obs["planets"] if int(p[1]) == int(obs["player"])}
    spent = {}
    for mv in moves:
        assert isinstance(mv, list)
        assert len(mv) == 3
        src, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
        assert src in owned
        assert math.isfinite(ang)
        assert -2 * math.pi <= ang <= 2 * math.pi
        assert 1 <= ships <= int(planets[src][5])
        spent[src] = spent.get(src, 0) + ships
    for src, total in spent.items():
        assert total <= int(planets[src][5])


@pytest.mark.parametrize(
    "fn",
    [random_agent, greedy_agent, rush_agent, turtle_agent, greedy_expand_agent, swarm_agent, idle_heavy_agent],
)
def test_baselines_return_legal_list(fn):
    obs = _obs(
        [
            [1, 0, 20, 20, 2, 30, 3],
            [2, 1, 80, 80, 2, 20, 3],
            [3, -1, 25, 70, 2, 8, 1],
            [4, -1, 70, 25, 2, 6, 2],
        ]
    )
    _assert_legal_moves(obs, fn(obs))


@pytest.mark.parametrize("fn", [random_agent, greedy_agent, rush_agent, turtle_agent])
def test_malformed_observation_is_safe(fn):
    assert fn({"player": 0, "planets": [[1, 0, 20]]}) == []


@pytest.mark.parametrize(
    "fn",
    [random_agent, greedy_agent, rush_agent, turtle_agent, greedy_expand_agent, swarm_agent, idle_heavy_agent],
)
def test_non_finite_planet_rows_are_safe(fn):
    bad_cases = [
        _obs([[1, 0, float("nan"), 10, 2, 30, 3], [2, -1, 20, 20, 2, 5, 1]]),
        _obs([[1, 0, 10, float("inf"), 2, 30, 3], [2, -1, 20, 20, 2, 5, 1]]),
        _obs([[1, 0, 10, 10, 2, float("nan"), 3], [2, -1, 20, 20, 2, 5, 1]]),
        _obs([[1, 0, 10, 10, 2, 30, float("inf")], [2, -1, 20, 20, 2, 5, 1]]),
    ]
    for obs in bad_cases:
        out = fn(obs)
        assert isinstance(out, list)
        if out:
            _assert_legal_moves(obs, out)


@pytest.mark.parametrize(
    "fn",
    [random_agent, greedy_agent, rush_agent, turtle_agent, greedy_expand_agent, swarm_agent, idle_heavy_agent],
)
def test_non_finite_id_or_owner_are_safe_for_correct_width_rows(fn):
    bad_cases = [
        _obs([[float("nan"), 0, 10, 10, 2, 30, 3], [2, -1, 20, 20, 2, 5, 1]]),
        _obs([[float("inf"), 0, 10, 10, 2, 30, 3], [2, -1, 20, 20, 2, 5, 1]]),
        _obs([[1, float("nan"), 10, 10, 2, 30, 3], [2, -1, 20, 20, 2, 5, 1]]),
        _obs([[1, float("-inf"), 10, 10, 2, 30, 3], [2, -1, 20, 20, 2, 5, 1]]),
    ]
    for obs in bad_cases:
        out = fn(obs)
        assert isinstance(out, list)
        if out:
            _assert_legal_moves(obs, out)


def test_greedy_attacks_nearest_non_owned_when_affordable():
    obs = _obs(
        [
            [1, 0, 10, 10, 2, 50, 3],
            [2, -1, 15, 10, 2, 2, 1],
            [3, -1, 80, 80, 2, 2, 1],
        ]
    )
    moves = greedy_agent(obs)
    assert moves
    assert int(moves[0][0]) == 1
    assert abs(float(moves[0][1])) < 0.2


@pytest.mark.parametrize("fn", [greedy_agent, rush_agent, turtle_agent])
def test_style_agents_avoid_sun_blocked_direct_paths(fn):
    obs = _obs([[1, 0, 20, 50, 2, 50, 3], [2, 1, 80, 50, 2, 1, 1], [3, -1, 80, 70, 2, 1, 1]])
    moves = fn(obs)
    for mv in moves:
        assert abs(float(mv[1])) > 1e-3


def test_random_agent_is_deterministic_and_nonempty_when_legal():
    obs = _obs([[1, 0, 10, 10, 2, 30, 3], [2, -1, 80, 10, 2, 10, 1], [3, -1, 10, 80, 2, 10, 1]])
    a1 = random_agent(obs, config={"seed": 123})
    a2 = random_agent(obs, config={"seed": 123})
    assert a1 == a2
    assert a1


def test_turtle_idles_before_delay_then_one_move_max():
    obs_early = _obs([[1, 0, 10, 10, 2, 40, 3], [2, -1, 20, 10, 2, 2, 1]], step=5)
    assert turtle_agent(obs_early, config={"delay": 10}) == []

    obs_late = _obs([[1, 0, 10, 10, 2, 40, 3], [2, -1, 20, 10, 2, 2, 1]], step=20)
    out = turtle_agent(obs_late, config={"delay": 10})
    assert len(out) <= 1
    _assert_legal_moves(obs_late, out)


def test_rush_emits_at_most_two_moves():
    obs = _obs([[1, 0, 10, 10, 2, 40, 3], [2, 0, 20, 20, 2, 40, 3], [3, -1, 80, 10, 2, 3, 1], [4, 1, 90, 20, 2, 3, 1]])
    out = rush_agent(obs)
    assert len(out) <= 2
    _assert_legal_moves(obs, out)
