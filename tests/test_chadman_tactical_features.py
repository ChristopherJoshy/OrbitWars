from types import SimpleNamespace

from src.ow import chadman


def _planet(pid, owner, x, y, ships, production):
    return chadman.Planet(pid, owner, float(x), float(y), 1.0, int(ships), int(production))


def _midgame_staging_world(*, remaining_steps=435, src_to_dst_turns=4, dst_to_target_turns=6, src_x=0, dst_x=20, src_attack_budget=120, dst_attack_budget=2):
    src = _planet(1, 0, src_x, 10, 180, 2)
    dst = _planet(2, 0, dst_x, 10, 20, 5)
    enemy = _planet(3, 1, 24, 10, 70, 4)
    neutral = _planet(4, -1, 23, 14, 18, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[neutral],
        remaining_steps=remaining_steps,
        is_early=False,
        is_late=False,
        my_total=320,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy, 4: neutral},
    )

    def plan_shot(src_id, dst_id, ships):
        if src_id == 1 and dst_id == 2:
            return (0.0, src_to_dst_turns, 10.0, None)
        if src_id == 2 and dst_id in (3, 4):
            return (0.0, dst_to_target_turns, 10.0, None)
        return None

    world.plan_shot = plan_shot
    policy = {"source_keep": {1: 30, 2: 8}, "attack_budget": {1: src_attack_budget, 2: dst_attack_budget}}
    return world, policy


def test_enemy_waste_score_rewards_attacks_that_die_into_reinforcement():
    weak = chadman.enemy_waste_score(enemy_ships=20, defender_after_rescue=8, production_saved=2, arrival_turn=4)
    waste = chadman.enemy_waste_score(enemy_ships=20, defender_after_rescue=28, production_saved=2, arrival_turn=4)

    assert waste > weak
    assert waste > 20


def test_friendly_transfer_value_prefers_frontier_high_prod_destination():
    src = _planet(1, 0, 10, 10, 80, 1)
    rear = _planet(2, 0, 12, 12, 15, 1)
    frontier = _planet(3, 0, 70, 70, 12, 5)
    enemy = _planet(4, 1, 74, 72, 40, 4)
    world = SimpleNamespace(player=0, enemy_planets=[enemy], remaining_steps=120, fall_turn_map={3: 9})
    policy = {"source_keep": {1: 20}, "attack_budget": {1: 50}}

    rear_value = chadman.friendly_transfer_value(src, rear, world, policy)
    frontier_value = chadman.friendly_transfer_value(src, frontier, world, policy)

    assert frontier_value > rear_value
    assert frontier_value > 0


def test_rank_pressure_value_penalizes_helping_weakest_when_it_does_not_improve_rank():
    world = SimpleNamespace(
        player=0,
        is_four_player=True,
        owner_production={0: 20, 1: 26, 2: 6, 3: 18},
        owner_strength={0: 120, 1: 150, 2: 25, 3: 90},
        _weakest_enemy=2,
    )

    leader_value = chadman.rank_pressure_value(world, target_owner=1, target_production=5)
    weakest_value = chadman.rank_pressure_value(world, target_owner=2, target_production=5)

    assert leader_value > 1.0
    assert weakest_value < leader_value


def test_enemy_waste_score_rewards_exact_defensive_trade():
    weak = chadman.enemy_waste_score(20, 19, 2, 4)
    exact_trade = chadman.enemy_waste_score(20, 20, 2, 4)

    assert exact_trade > weak
    assert exact_trade > 20


def test_rank_pressure_value_uses_target_production_to_relax_weakest_tax():
    world = SimpleNamespace(
        player=0,
        is_four_player=True,
        owner_production={0: 20, 1: 26, 2: 6, 3: 18},
        owner_strength={0: 120, 1: 150, 2: 25, 3: 90},
        _weakest_enemy=2,
    )

    low_prod_value = chadman.rank_pressure_value(world, target_owner=2, target_production=1)
    high_prod_value = chadman.rank_pressure_value(world, target_owner=2, target_production=8)

    assert high_prod_value > low_prod_value


def test_build_transfer_missions_creates_owned_to_owned_option(monkeypatch):
    src = _planet(1, 0, 10, 10, 80, 1)
    dst = _planet(2, 0, 20, 10, 10, 5)
    enemy = _planet(3, 1, 90, 90, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        remaining_steps=100,
        fall_turn_map={2: 8},
        arrivals_by_planet={2: [(8, 1, 200)]},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 3, 10.0, None)
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 50, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 60 if source_id == 1 else 2)

    assert missions
    option = missions[0].options[0]
    assert missions[0].kind == "transfer"
    assert option.src_id == 1
    assert option.target_id == 2
    assert option.mission == "transfer"
    assert option.send_cap >= chadman.TRANSFER_MIN_SURPLUS


def test_build_transfer_missions_skips_destination_that_falls_before_arrival():
    src = _planet(1, 0, 10, 10, 80, 1)
    dst = _planet(2, 0, 20, 10, 10, 5)
    enemy = _planet(3, 1, 23, 10, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        remaining_steps=100,
        fall_turn_map={2: 5},
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 8, 10.0, None)
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 50, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 60 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_skips_when_final_demand_send_arrives_after_fall(monkeypatch):
    src = _planet(1, 0, 10, 10, 100, 2)
    dst = _planet(2, 0, 20, 10, 12, 5)
    enemy = _planet(3, 1, 90, 90, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=300,
        is_early=False,
        is_late=False,
        my_total=220,
        fall_turn_map={2: 6},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 80, 2: 1}}

    calls = []

    def fake_plan_shot(src_id, dst_id, ships):
        calls.append((src_id, dst_id, ships))
        if ships <= 40:
            return (0.0, 5, 10.0, None)
        return (0.0, 8, 10.0, None)

    def fake_demand(*_args, **_kwargs):
        return {"kind": "emergency", "ships": 60, "value": 10.0}

    world.plan_shot = fake_plan_shot
    monkeypatch.setattr(chadman, "transfer_destination_demand", fake_demand)

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 80 if source_id == 1 else 2)

    assert missions == []
    assert len(calls) >= 2
    assert calls[0][2] <= 40
    assert calls[1][2] == 60


def test_build_transfer_missions_caps_routine_fanout_from_one_source(monkeypatch):
    src = _planet(1, 0, 10, 10, 500, 2)
    destinations = [
        _planet(2, 0, 30, 10, 30, 5),
        _planet(3, 0, 34, 10, 30, 5),
        _planet(4, 0, 38, 10, 30, 5),
        _planet(5, 0, 42, 10, 30, 5),
    ]
    enemy = _planet(6, 1, 90, 90, 50, 4)
    my_planets = [src, *destinations]
    world = SimpleNamespace(
        player=0,
        my_planets=my_planets,
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=240,
        is_early=False,
        is_late=True,
        my_total=650,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={planet.id: planet for planet in [*my_planets, enemy]},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 6, 10.0, None) if src_id == 1 else None
    policy = {
        "source_keep": {planet.id: (60 if planet.id == 1 else 30) for planet in my_planets},
        "attack_budget": {planet.id: (240 if planet.id == 1 else 0) for planet in my_planets},
    }

    def routine_demand(*_args, **_kwargs):
        return {"kind": "late_consolidation", "ships": 120, "value": 100.0}

    monkeypatch.setattr(chadman, "transfer_destination_demand", routine_demand)

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 240 if source_id == 1 else 0)

    assert len(missions) == 2
    assert {mission.options[0].src_id for mission in missions} == {1}


def test_bait_reinforce_value_requires_enemy_threat_and_friendly_rescue_window():
    target = _planet(5, 0, 50, 50, 6, 5)
    world = SimpleNamespace(player=0, remaining_steps=100, fall_turn_map={5: 6})
    no_rescue = chadman.bait_reinforce_value(target, enemy_ships=18, enemy_eta=6, friendly_rescue_ships=0, friendly_eta=4, world=world)
    rescue = chadman.bait_reinforce_value(target, enemy_ships=18, enemy_eta=6, friendly_rescue_ships=20, friendly_eta=4, world=world)

    assert no_rescue == 0
    assert rescue > 0


def test_bait_reinforce_value_rewards_exact_defensive_hold():
    target = _planet(6, 0, 40, 40, 8, 2)
    world = SimpleNamespace(player=0, remaining_steps=80)

    exact_hold = chadman.bait_reinforce_value(
        target,
        enemy_ships=20,
        enemy_eta=4,
        friendly_rescue_ships=4,
        friendly_eta=3,
        world=world,
    )
    weak_or_late = chadman.bait_reinforce_value(
        target,
        enemy_ships=20,
        enemy_eta=4,
        friendly_rescue_ships=2,
        friendly_eta=6,
        world=world,
    )

    assert exact_hold > 0
    assert exact_hold > weak_or_late


def test_enemy_threat_ships_at_turn_uses_actual_hostile_arrivals():
    world = SimpleNamespace(
        player=0,
        arrivals_by_planet={5: [(6, 1, 100), (6, 0, 20), (5, 2, 7), (7, 1, 50)]},
    )

    threat = chadman.enemy_threat_ships_at_turn(world, target_id=5, threat_turn=6)

    assert threat == 100


def test_select_transfer_hub_prefers_safe_high_connectivity_planet():
    a = _planet(1, 0, 10, 10, 50, 1)
    hub = _planet(2, 0, 45, 45, 30, 3)
    frontier = _planet(3, 0, 85, 85, 12, 5)
    enemy = _planet(4, 1, 90, 90, 50, 4)
    world = SimpleNamespace(player=0, my_planets=[a, hub, frontier], enemy_planets=[enemy], fall_turn_map={3: 8})

    assert chadman.select_transfer_hub(world).id == hub.id


def test_select_transfer_hub_ignores_falling_high_production_planet():
    safe = _planet(1, 0, 10, 10, 30, 1)
    falling_high_prod = _planet(2, 0, 70, 70, 45, 9)
    world = SimpleNamespace(
        player=0,
        my_planets=[safe, falling_high_prod],
        enemy_planets=[],
        fall_turn_map={2: 10},
    )

    selected = chadman.select_transfer_hub(world)

    assert selected is not None
    assert selected.id == safe.id


def test_transfer_dynamic_minimum_rejects_small_routine_packet():
    src = _planet(1, 0, 10, 10, 80, 2)
    dst = _planet(2, 0, 30, 10, 20, 3)
    world = SimpleNamespace(player=0, my_total=220, remaining_steps=420, is_early=True, is_late=False)
    policy = {"source_keep": {1: 20}, "attack_budget": {1: 60}}

    min_send = chadman.transfer_dynamic_min_send(src, dst, world, policy, available=60, emergency=False)

    assert min_send > chadman.TRANSFER_MIN_SURPLUS
    assert min_send >= 40


def test_transfer_destination_demand_allows_emergency_before_fall():
    src = _planet(1, 0, 10, 10, 80, 2)
    dst = _planet(2, 0, 30, 10, 8, 3)
    enemy = _planet(3, 1, 33, 10, 40, 4)
    world = SimpleNamespace(
        player=0,
        my_total=120,
        remaining_steps=420,
        is_early=True,
        is_late=False,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        fall_turn_map={2: 6},
        arrivals_by_planet={2: [(6, 1, 35)]},
        fleets=[],
    )
    policy = {"source_keep": {1: 20}, "attack_budget": {1: 60}}

    demand = chadman.transfer_destination_demand(src, dst, world, policy, {}, turns=4, available=60, transfer_hub_id=None)

    assert demand is not None
    assert demand["kind"] == "emergency"
    assert demand["ships"] >= chadman.TRANSFER_MIN_SURPLUS
    assert demand["value"] > 0


def test_transfer_destination_demand_rejects_early_routine_churn():
    src = _planet(1, 0, 10, 10, 80, 2)
    dst = _planet(2, 0, 30, 10, 20, 3)
    enemy = _planet(3, 1, 90, 90, 40, 4)
    world = SimpleNamespace(
        player=0,
        my_total=220,
        remaining_steps=440,
        is_early=True,
        is_late=False,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
    )
    policy = {"source_keep": {1: 20}, "attack_budget": {1: 60}}

    demand = chadman.transfer_destination_demand(src, dst, world, policy, {}, turns=4, available=60, transfer_hub_id=None)

    assert demand is None


def test_transfer_destination_demand_allows_late_large_hub_consolidation():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 35, 10, 40, 5)
    enemy = _planet(3, 1, 45, 10, 80, 4)
    world = SimpleNamespace(
        player=0,
        my_total=520,
        remaining_steps=250,
        is_early=False,
        is_late=True,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
    )
    policy = {"source_keep": {1: 80}, "attack_budget": {1: 170}}

    demand = chadman.transfer_destination_demand(src, dst, world, policy, {}, turns=5, available=170, transfer_hub_id=dst.id)

    assert demand is not None
    assert demand["kind"] == "late_consolidation"
    assert demand["ships"] >= 80
    assert demand["value"] > 0


def test_build_transfer_missions_rejects_early_small_churn_without_demand():
    src = _planet(1, 0, 10, 10, 80, 1)
    dst = _planet(2, 0, 20, 10, 10, 5)
    enemy = _planet(3, 1, 90, 90, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=450,
        is_early=True,
        is_late=False,
        my_total=160,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 3, 10.0, None)
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 50, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 60 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_accepts_late_large_consolidation():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 20, 10, 40, 5)
    enemy = _planet(3, 1, 30, 10, 60, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 5, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions
    assert missions[0].options[0].send_cap >= 80


def test_build_transfer_missions_blocks_saturated_destination():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 20, 10, 40, 5)
    enemy = _planet(3, 1, 30, 10, 60, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={2: [(4, 0, 180)]},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 5, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions == []


def test_transfer_dynamic_minimum_not_capped_by_available_for_late_routine():
    src = _planet(1, 0, 10, 10, 140, 2)
    dst = _planet(2, 0, 20, 10, 40, 5)
    world = SimpleNamespace(player=0, my_total=520, remaining_steps=250, is_early=False, is_late=True)
    policy = {"source_keep": {1: 80}, "attack_budget": {1: 170}}
    available = 30

    routine_min = chadman.transfer_dynamic_min_send(src, dst, world, policy, available=available, emergency=False)
    emergency_min = chadman.transfer_dynamic_min_send(src, dst, world, policy, available=available, emergency=True)

    assert routine_min >= chadman.TRANSFER_LATE_ROUTINE_MIN_SEND
    assert routine_min > available
    assert emergency_min == chadman.TRANSFER_MIN_SURPLUS


def test_build_transfer_missions_rejects_late_small_available_consolidation():
    src = _planet(1, 0, 10, 10, 60, 2)
    dst = _planet(2, 0, 20, 10, 40, 5)
    enemy = _planet(3, 1, 30, 10, 60, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 5, 10.0, None)
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 30, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 30 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_computes_transfer_hub_once(monkeypatch):
    p1 = _planet(1, 0, 10, 10, 90, 3)
    p2 = _planet(2, 0, 20, 10, 70, 4)
    p3 = _planet(3, 0, 30, 10, 65, 5)
    enemy = _planet(4, 1, 90, 90, 40, 2)
    world = SimpleNamespace(
        player=0,
        my_planets=[p1, p2, p3],
        enemy_planets=[enemy],
        remaining_steps=120,
        fall_turn_map={},
        planet_by_id={1: p1, 2: p2, 3: p3, 4: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 3, 10.0, None)
    policy = {
        "source_keep": {1: 10, 2: 10, 3: 10},
        "attack_budget": {1: 80, 2: 60, 3: 50},
    }

    calls = {"n": 0}
    original = chadman.select_transfer_hub

    def counted_select_transfer_hub(local_world):
        calls["n"] += 1
        return original(local_world)

    monkeypatch.setattr(chadman, "select_transfer_hub", counted_select_transfer_hub)

    _ = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 80)

    assert calls["n"] == 1


def test_build_transfer_missions_computes_transfer_hub_once_when_no_safe_hub(monkeypatch):
    p1 = _planet(1, 0, 10, 10, 90, 3)
    p2 = _planet(2, 0, 20, 10, 70, 4)
    p3 = _planet(3, 0, 30, 10, 65, 5)
    enemy = _planet(4, 1, 90, 90, 40, 2)
    world = SimpleNamespace(
        player=0,
        my_planets=[p1, p2, p3],
        enemy_planets=[enemy],
        remaining_steps=120,
        fall_turn_map={1: 7, 2: 8, 3: 9},
        planet_by_id={1: p1, 2: p2, 3: p3, 4: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 3, 10.0, None)
    policy = {
        "source_keep": {1: 10, 2: 10, 3: 10},
        "attack_budget": {1: 80, 2: 60, 3: 50},
    }

    calls = {"n": 0}
    original = chadman.select_transfer_hub

    def counted_select_transfer_hub(local_world):
        calls["n"] += 1
        return original(local_world)

    monkeypatch.setattr(chadman, "select_transfer_hub", counted_select_transfer_hub)

    _ = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 80)

    assert calls["n"] == 1


def test_build_transfer_missions_blocks_source_with_near_attack_opportunity():
    src = _planet(1, 0, 0, 0, 260, 2)
    dst = _planet(2, 0, 40, 0, 40, 5)
    enemy = _planet(3, 1, 42, 0, 400, 4)
    neutral = _planet(4, -1, 14, 10, 40, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[neutral],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy, 4: neutral},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 4 if dst_id == 4 else 5, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_blocks_reverse_friendly_flow():
    src = _planet(1, 0, 0, 0, 260, 2)
    dst = _planet(2, 0, 40, 0, 40, 5)
    enemy = _planet(3, 1, 42, 0, 400, 4)
    reverse = chadman.Fleet(9, 0, 30.0, 10.0, 0.0, 2, 80)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[reverse],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.fleet_target = lambda fleet, horizon: (src, 4)
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 5, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_accepts_midgame_attack_staging_transfer():
    world, policy = _midgame_staging_world(remaining_steps=435, src_to_dst_turns=4, dst_to_target_turns=6)
    world.enemy_planets = []
    world.neutral_planets = [_planet(4, -1, 23, 14, 40, 4)]
    world.planet_by_id[4] = world.neutral_planets[0]

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    assert [m for m in missions if m.kind == "transfer"] == []


def test_build_transfer_missions_blocks_pre60_attack_staging():
    world, policy = _midgame_staging_world(remaining_steps=450, src_to_dst_turns=4, dst_to_target_turns=6)

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    assert [m for m in missions if m.kind == "transfer"] == []


def test_build_transfer_missions_blocks_front_to_rear_staging():
    world, policy = _midgame_staging_world(remaining_steps=435, src_to_dst_turns=4, dst_to_target_turns=6, src_x=17, dst_x=20)

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    assert [m for m in missions if m.kind == "transfer"] == []


def test_build_transfer_missions_blocks_staging_when_destination_can_already_attack():
    world, policy = _midgame_staging_world(
        remaining_steps=435,
        src_to_dst_turns=4,
        dst_to_target_turns=6,
        dst_attack_budget=80,
    )
    world.enemy_planets = []
    world.neutral_planets = [_planet(7, -1, 23, 14, 18, 4)]
    world.planet_by_id[7] = world.neutral_planets[0]

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    assert [m for m in missions if m.kind == "transfer"] == []


def test_build_transfer_missions_blocks_staging_when_destination_can_capture_neutral_without_transfer():
    world, policy = _midgame_staging_world(
        remaining_steps=435,
        src_to_dst_turns=4,
        dst_to_target_turns=6,
        dst_attack_budget=2,
    )

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    assert [m for m in missions if m.kind == "transfer"] == []


def test_build_transfer_missions_requires_staging_to_cover_enemy_growth_until_attack():
    world, policy = _midgame_staging_world(
        remaining_steps=435,
        src_to_dst_turns=4,
        dst_to_target_turns=6,
        dst_attack_budget=2,
    )
    world.neutral_planets = []
    world.enemy_planets = [_planet(3, 1, 23, 14, 18, 3)]
    world.planet_by_id[3] = world.enemy_planets[0]

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    transfers = [m for m in missions if m.kind == "transfer"]
    assert transfers
    assert transfers[0].options[0].needed >= 42


def test_transfer_destination_demand_uses_min_ships_to_own_by_eval_turn_signature():
    world, policy = _midgame_staging_world(
        remaining_steps=435,
        src_to_dst_turns=4,
        dst_to_target_turns=6,
        dst_attack_budget=2,
    )
    calls = []

    def min_ships_to_own_by(target_id, eval_turn, attacker_owner, arrival_turn=None, planned_commitments=None, **_kwargs):
        calls.append(
            {
                "target_id": target_id,
                "eval_turn": eval_turn,
                "attacker_owner": attacker_owner,
                "arrival_turn": arrival_turn,
                "planned_commitments": planned_commitments,
            }
        )
        return 42

    world.min_ships_to_own_by = min_ships_to_own_by
    planned_commitments = {2: [(4, 0, 6)]}

    demand = chadman.transfer_destination_demand(
        world.my_planets[0],
        world.my_planets[1],
        world,
        policy,
        planned_commitments,
        turns=4,
        available=120,
        transfer_hub_id=None,
    )

    assert demand is not None
    assert demand["kind"] == "midgame_attack_staging"
    assert calls
    call = calls[0]
    assert call["target_id"] == 3
    assert call["eval_turn"] == 10
    assert call["arrival_turn"] == 10
    assert call["attacker_owner"] == world.player
    assert call["planned_commitments"] is planned_commitments


def test_build_transfer_missions_blocks_slow_midgame_staging():
    world, policy = _midgame_staging_world(remaining_steps=435, src_to_dst_turns=10, dst_to_target_turns=6)

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 120 if source_id == 1 else 2)

    assert [m for m in missions if m.kind == "transfer"] == []


def test_build_transfer_missions_requires_safe_interior_source():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 30, 10, 40, 5)
    enemy = _planet(3, 1, 18, 10, 60, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 5, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_blocks_unsafe_source_for_non_emergency_late_consolidation_with_fall_turn():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 30, 10, 40, 5)
    enemy = _planet(3, 1, 18, 10, 60, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={2: 20},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 4, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions == []


def test_build_transfer_missions_reaims_with_demand_backed_send():
    src = _planet(1, 0, 0, 0, 260, 2)
    dst = _planet(2, 0, 40, 0, 40, 5)
    enemy = _planet(3, 1, 42, 0, 400, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=220,
        is_early=False,
        is_late=True,
        my_total=520,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    calls = []

    def plan_shot(src_id, dst_id, ships):
        calls.append(int(ships))
        if ships < 80:
            return (0.1, 9, 10.0, None)
        return (0.9, 3, 10.0, None)

    world.plan_shot = plan_shot
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 2)

    assert missions
    option = missions[0].options[0]
    assert option.needed >= 80
    assert option.angle == 0.9
    assert option.turns == 3
    assert any(ship >= 80 for ship in calls)


def test_target_value_uses_rank_pressure_for_four_player_enemy_targets(monkeypatch):
    target = _planet(9, 1, 60, 60, 20, 5)
    world = SimpleNamespace(
        player=0,
        is_four_player=True,
        is_very_late=False,
        remaining_steps=100,
        _weakest_enemy=2,
        owner_production={0: 20, 1: 30, 2: 5, 3: 15},
        owner_strength={0: 100, 1: 150, 2: 25, 3: 80},
        leader_owner=1,
        comet_ids=set(),
        exposed_planet_ids=set(),
        my_total=100,
        my_prod=20,
        is_opening=False,
        is_early=False,
        is_late=False,
    )
    world.is_static = lambda _pid: False
    modes = {"is_finishing": False, "is_behind": False, "is_dominating": False}
    policy = {
        "meta": {},
        "phase_prior": {"enemy": 1.0, "neutral": 1.0, "capture": 1.0},
        "indirect_wealth_map": {target.id: 0.0},
    }
    monkeypatch.setattr(chadman, "rank_pressure_value", lambda _world, target_owner, target_production=0: 1.5)

    boosted = chadman.target_value(target, 5, "capture", world, modes, policy)
    monkeypatch.setattr(chadman, "rank_pressure_value", lambda _world, target_owner, target_production=0: 1.0)
    base = chadman.target_value(target, 5, "capture", world, modes, policy)

    assert boosted > base


def test_likely_enemy_targets_prioritizes_vulnerable_high_prod_frontier():
    mine = _planet(1, 0, 50, 50, 8, 5)
    safe = _planet(2, 0, 10, 10, 50, 1)
    enemy = _planet(3, 1, 55, 50, 60, 4)
    world = SimpleNamespace(player=0, my_planets=[mine, safe], enemy_planets=[enemy], remaining_steps=100)

    targets = chadman.likely_enemy_targets(world, top_k=1)

    assert targets[0]["target_id"] == mine.id
    assert targets[0]["score"] > 0


def test_likely_enemy_targets_handles_invalid_top_k_and_incomplete_planets():
    valid_target = _planet(1, 0, 30, 30, 10, 3)
    missing_coords = SimpleNamespace(id=2, owner=0, ships=5, production=2)
    missing_id = SimpleNamespace(owner=0, x=31.0, y=30.0, ships=5, production=2)
    missing_prod_ships = SimpleNamespace(id=3, owner=0, x=32.0, y=30.0)
    enemy_valid = _planet(9, 1, 35, 30, 20, 2)
    enemy_missing_coords = SimpleNamespace(id=10, owner=1, ships=10, production=2)
    world = SimpleNamespace(
        player=0,
        my_planets=[valid_target, missing_coords, missing_id, missing_prod_ships],
        enemy_planets=[enemy_valid, enemy_missing_coords],
        remaining_steps=100,
    )

    assert isinstance(chadman.likely_enemy_targets(world, top_k=None), list)
    assert isinstance(chadman.likely_enemy_targets(world, top_k="bad"), list)

    targets = chadman.likely_enemy_targets(world, top_k=5)
    assert isinstance(targets, list)
    assert any(row["target_id"] == valid_target.id for row in targets)


def test_likely_enemy_targets_bounds_extreme_scores():
    extreme = _planet(1, 0, 50, 50, -10_000, 1_000_000)
    enemy = _planet(2, 1, 50, 50, 10, 1)
    world = SimpleNamespace(player=0, my_planets=[extreme], enemy_planets=[enemy], remaining_steps=100)

    targets = chadman.likely_enemy_targets(world, top_k=1)

    assert targets
    assert targets[0]["score"] > 0
    assert targets[0]["score"] <= 200.0


def test_logistic_packet_send_preserves_seeded_margin_for_two_player_attack():
    target = _planet(9, 1, 40, 40, 20, 3)
    world = SimpleNamespace(
        is_four_player=False,
        is_total_war=False,
        remaining_steps=100,
        player=0,
        comet_ids=set(),
        is_opening=False,
    )
    policy = {"meta": {}}

    send = chadman._logistic_packet_send(
        target,
        needed=10,
        seeded_send=15,
        arrival_turns=10,
        src_available=60,
        world=world,
        policy=policy,
    )

    assert send >= 15


def test_build_elimination_missions_is_four_player_only():
    src = _planet(1, 0, 10, 10, 120, 3)
    enemy = _planet(2, 1, 20, 10, 20, 2)
    world = SimpleNamespace(
        player=0,
        is_four_player=False,
        _weakest_enemy=1,
        _weakest_enemy_strength=20,
        my_total=200,
        my_planets=[src],
        enemy_planets=[enemy],
        owner_strength={0: 200, 1: 20},
        remaining_steps=100,
    )
    world.nearest_my_sources = lambda _target_id, _k: [src]
    world.best_probe_aim = lambda *_args, **_kwargs: (None, (0.0, 5, 1.0, None))
    world.min_ships_to_own_at = lambda *_args, **_kwargs: 10
    world.comet_ids = set()
    world.comet_life = lambda _pid: 0
    world.is_static = lambda _pid: False
    world.exposed_planet_ids = set()
    world.is_opening = False
    world.is_late = False
    world.owner_production = {0: 6, 1: 2}
    world.my_prod = 6
    policy = {"attack_budget": {1: 100}, "indirect_wealth_map": {2: 0.0}, "meta": {}, "phase_prior": {}}

    assert chadman.build_elimination_missions(world, policy, {}, {}) == []


def test_two_player_target_value_does_not_apply_weakest_multiplier():
    target = _planet(2, 1, 20, 10, 20, 2)
    world = SimpleNamespace(
        player=0,
        is_four_player=False,
        remaining_steps=100,
        comet_ids=set(),
        is_static=lambda _pid: False,
        is_opening=False,
        is_early=False,
        is_late=False,
        exposed_planet_ids=set(),
        _weakest_enemy=1,
        owner_strength={0: 200, 1: 20},
        owner_production={0: 6, 1: 2},
        my_total=200,
        my_prod=6,
    )
    policy = {"indirect_wealth_map": {2: 0.0}, "reaction_time_map": {2: (2, 9)}, "meta": {}, "phase_prior": {}}
    modes = {"is_finishing": False, "is_behind": False, "is_dominating": False}

    with_weakest = chadman.target_value(target, 5, "capture", world, modes, policy)
    world._weakest_enemy = None
    without_weakest = chadman.target_value(target, 5, "capture", world, modes, policy)

    assert with_weakest == without_weakest


def test_apply_score_modifiers_weakest_bonus_is_four_player_only():
    target = _planet(2, 1, 20, 10, 20, 2)
    world = SimpleNamespace(
        player=0,
        is_four_player=False,
        is_static=lambda _pid: False,
        is_early=False,
        static_neutral_planets=[],
        exposed_planet_ids=set(),
        _weakest_enemy=1,
    )

    score = chadman.apply_score_modifiers(100.0, target, "capture", world)

    assert score == 100.0


def test_primary_enemy_targeting_enabled_in_two_player_when_weakest_present():
    world = SimpleNamespace(is_four_player=False, _weakest_enemy=1)
    assert chadman.primary_enemy_targeting_enabled(world) is False

    world.is_four_player = True
    assert chadman.primary_enemy_targeting_enabled(world) is True

    world._weakest_enemy = None
    assert chadman.primary_enemy_targeting_enabled(world) is False


def test_candidate_targets_weakest_boost_is_four_player_only():
    src = _planet(1, 0, 100, 100, 50, 1)
    non_weak_enemy = _planet(2, 2, 90, 100, 20, 4)
    weak_enemy = _planet(9, 1, 110, 100, 20, 4)
    world = SimpleNamespace(
        player=0,
        planets=[src, non_weak_enemy, weak_enemy],
        remaining_steps=100,
        is_four_player=False,
        exposed_planet_ids=set(),
        comet_ids=set(),
        _weakest_enemy=1,
        is_static=lambda _pid: False,
    )

    targets = chadman.candidate_targets_for_source(src, world, {}, {"is_finishing": False}, top_k=2)

    assert targets[0].id == non_weak_enemy.id


def test_adaptive_target_top_k_uses_full_two_player_budget():
    world = SimpleNamespace(is_four_player=False, planets=[object()] * 28)

    assert chadman.adaptive_target_top_k(world) == chadman.MAIN_TARGET_TOP_K_TWO_PLAYER


def test_adaptive_target_top_k_preserves_four_player_density_caps():
    dense_world = SimpleNamespace(is_four_player=True, planets=[object()] * 32)
    sparse_world = SimpleNamespace(is_four_player=True, planets=[object()] * 24)

    assert chadman.adaptive_target_top_k(dense_world) == 7
    assert chadman.adaptive_target_top_k(sparse_world) == 8

