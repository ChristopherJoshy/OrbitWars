import math

from kaggle_environments.envs.orbit_wars import orbit_wars as official
from kaggle_environments.utils import Struct
from src.ow import chadman


def _struct(**kwargs):
    obj = Struct()
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj


def _run_official_tick(planets, fleets, *, players=2, step=1):
    state = [
        _struct(
            observation=_struct(
                remainingOverageTime=60.0,
                step=step,
                planets=planets,
                initial_planets=planets,
                fleets=fleets,
                player=player,
                angular_velocity=0.0,
                next_fleet_id=len(fleets),
                comets=[],
                comet_planet_ids=[],
            ),
            action={},
            reward=0,
            status="ACTIVE",
        )
        for player in range(players)
    ]
    env = _struct(
        done=False,
        configuration=_struct(shipSpeed=6.0, episodeSteps=100, cometSpeed=1.0),
        info={"seed": 0},
    )
    official.interpreter(state, env)
    return state[0].observation


def test_schema_and_speed_match_official():
    assert chadman.Planet._fields == official.Planet._fields
    assert chadman.Fleet._fields == official.Fleet._fields
    for ships in [1, 2, 5, 10, 50, 100, 500, 1000]:
        expected = 1.0 if ships <= 1 else 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
        expected = min(expected, 6.0)
        assert chadman.fleet_speed(ships) == expected


def test_chad_resolve_arrival_tie_preserves_defender_when_attackers_tie():
    owner, ships = chadman.resolve_arrival_event(0, 10, [(1, 1, 5), (1, 2, 5)])
    assert owner == 0
    assert ships == 10


def test_chad_resolve_arrival_captures_when_attacker_survives_garrison():
    owner, ships = chadman.resolve_arrival_event(0, 3, [(1, 1, 8)])
    assert owner == 1
    assert ships == 5


def test_official_combat_scenarios_match_chad_resolver():
    planets = [[0, 0, 50.0, 50.0, 1.0, 10, 0]]
    fleets = [
        [0, 1, 49.0, 50.0, 0.0, 1, 5],
        [1, 2, 49.0, 50.0, 0.0, 1, 5],
    ]
    obs = _run_official_tick(planets, fleets, players=3)
    assert obs.planets[0][1] == 0
    assert obs.planets[0][5] == 10
    owner, ships = chadman.resolve_arrival_event(0, 10, [(1, 1, 5), (1, 2, 5)])
    assert (owner, ships) == (obs.planets[0][1], obs.planets[0][5])

    planets = [[0, 0, 50.0, 50.0, 1.0, 3, 0]]
    fleets = [[0, 1, 49.0, 50.0, 0.0, 1, 8]]
    obs = _run_official_tick(planets, fleets)
    assert obs.planets[0][1] == 1
    assert obs.planets[0][5] == 5
    owner, ships = chadman.resolve_arrival_event(0, 3, [(1, 1, 8)])
    assert (owner, ships) == (obs.planets[0][1], obs.planets[0][5])


def test_official_production_before_arrival_matches_chad_timeline():
    planets = [[0, 0, 50.0, 50.0, 1.0, 3, 2]]
    fleets = [[0, 1, 49.0, 50.0, 0.0, 1, 6]]
    obs = _run_official_tick(planets, fleets)
    assert obs.planets[0][1] == 1
    assert obs.planets[0][5] == 1

    timeline = chadman.simulate_planet_timeline(
        chadman.Planet(0, 0, 50.0, 50.0, 1.0, 3, 2),
        [(1, 1, 6)],
        player=0,
        horizon=1,
    )
    assert timeline["owner_at"][1] == obs.planets[0][1]
    assert timeline["ships_at"][1] == obs.planets[0][5]


def test_official_fleet_movement_matches_chad_speed():
    planets = [[0, -1, 95.0, 95.0, 1.0, 0, 0]]
    fleets = [[0, 0, 10.0, 10.0, 0.0, 0, 100]]
    obs = _run_official_tick(planets, fleets)
    moved = obs.fleets[0]
    assert math.isclose(moved[3], 10.0)
    assert math.isclose(moved[2] - 10.0, chadman.fleet_speed(100))
