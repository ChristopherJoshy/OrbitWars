from types import SimpleNamespace

from src.ow import chadman


def test_plan_moves_includes_transfer_mission_when_frontier_needs_ships(monkeypatch):
    src = chadman.Planet(1, 0, 10.0, 10.0, 1.0, 260, 2)
    dst = chadman.Planet(2, 0, 20.0, 10.0, 1.0, 40, 5)
    enemy = chadman.Planet(3, 1, 30.0, 10.0, 1.0, 80, 4)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        planets=[src, dst, enemy],
        remaining_steps=220,
        is_very_late=False,
        is_four_player=False,
        fall_turn_map={},
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, target_id, ships: (0.0, 3, 10.0, None)
    world.source_inventory_left = lambda source_id, spent: int(world.planet_by_id[source_id].ships) - spent[source_id]
    world.hold_status = lambda *args, **kwargs: {"holds_full": True, "fall_turn": None}
    world.comet_ids = set()
    world.is_late = True
    world.neutral_planets = []
    world.static_neutral_planets = []
    world.is_total_war = False
    policy = {"source_keep": {1: 80, 2: 8}, "attack_budget": {1: 170, 2: 1}}

    monkeypatch.setattr(chadman, "build_modes", lambda _world: {"is_finishing": False})
    monkeypatch.setattr(chadman, "build_policy_state", lambda _world, deadline=None: policy)
    monkeypatch.setattr(chadman, "build_reinforce_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_stabilize_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_rescue_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_recapture_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_elimination_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_gang_up_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "candidate_targets_for_source", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_snipe_mission", lambda *args, **kwargs: None)
    monkeypatch.setattr(chadman, "build_crash_exploit_missions", lambda *args, **kwargs: [])

    moves = chadman.plan_moves(world, deadline=None)

    assert moves
    assert moves[0][0] == 1
    assert moves[0][2] >= 80


def test_plan_moves_rejects_early_small_transfer_churn(monkeypatch):
    src = chadman.Planet(1, 0, 10.0, 10.0, 1.0, 80, 1)
    dst = chadman.Planet(2, 0, 20.0, 10.0, 1.0, 10, 5)
    enemy = chadman.Planet(3, 1, 90.0, 90.0, 1.0, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        planets=[src, dst, enemy],
        remaining_steps=460,
        is_very_late=False,
        is_four_player=False,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        my_total=160,
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, target_id, ships: (0.0, 3, 10.0, None)
    world.source_inventory_left = lambda source_id, spent: int(world.planet_by_id[source_id].ships) - spent[source_id]
    world.hold_status = lambda *args, **kwargs: {"holds_full": True, "fall_turn": None}
    world.comet_ids = set()
    world.is_late = False
    world.neutral_planets = []
    world.static_neutral_planets = []
    world.is_total_war = False
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 55, 2: 1}}

    monkeypatch.setattr(chadman, "build_modes", lambda _world: {"is_finishing": False})
    monkeypatch.setattr(chadman, "build_policy_state", lambda _world, deadline=None: policy)
    monkeypatch.setattr(chadman, "build_reinforce_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_stabilize_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_rescue_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_recapture_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_elimination_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_gang_up_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "candidate_targets_for_source", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_snipe_mission", lambda *args, **kwargs: None)
    monkeypatch.setattr(chadman, "build_crash_exploit_missions", lambda *args, **kwargs: [])

    moves = chadman.plan_moves(world, deadline=None)

    assert moves == []


def test_transfer_does_not_preempt_or_overspend_urgent_rescue(monkeypatch):
    src = chadman.Planet(1, 0, 10.0, 10.0, 1.0, 80, 1)
    dst = chadman.Planet(2, 0, 20.0, 10.0, 1.0, 10, 5)
    enemy = chadman.Planet(3, 1, 23.0, 10.0, 1.0, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        planets=[src, dst, enemy],
        remaining_steps=100,
        is_very_late=False,
        is_late=False,
        is_four_player=False,
        is_total_war=False,
        fall_turn_map={2: 8},
        planet_by_id={1: src, 2: dst, 3: enemy},
        comet_ids=set(),
        neutral_planets=[],
        static_neutral_planets=[],
    )
    world.plan_shot = lambda src_id, target_id, ships: (0.0, 3, 10.0, None)
    world.source_inventory_left = lambda source_id, spent: int(world.planet_by_id[source_id].ships) - spent[source_id]
    world.hold_status = lambda *args, **kwargs: {"holds_full": True, "fall_turn": None}
    policy = {"source_keep": {1: 50, 2: 8}, "attack_budget": {1: 30, 2: 1}}
    transfer = chadman.Mission(
        kind="transfer",
        score=999.0,
        target_id=2,
        turns=3,
        options=[chadman.ShotOption(score=999.0, src_id=1, target_id=2, angle=0.0, turns=3, needed=30, send_cap=30, mission="transfer")],
    )
    rescue = chadman.Mission(
        kind="rescue",
        score=-999.0,
        target_id=2,
        turns=2,
        options=[chadman.ShotOption(score=-999.0, src_id=1, target_id=2, angle=1.25, turns=2, needed=20, send_cap=20, mission="rescue")],
    )

    monkeypatch.setattr(chadman, "build_modes", lambda _world: {"is_finishing": False})
    monkeypatch.setattr(chadman, "build_policy_state", lambda _world, deadline=None: policy)
    monkeypatch.setattr(chadman, "build_reinforce_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_stabilize_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_rescue_missions", lambda *args, **kwargs: [rescue])
    monkeypatch.setattr(chadman, "build_recapture_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_transfer_missions", lambda *args, **kwargs: [transfer])
    monkeypatch.setattr(chadman, "build_elimination_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_gang_up_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "candidate_targets_for_source", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "build_snipe_mission", lambda *args, **kwargs: None)
    monkeypatch.setattr(chadman, "build_crash_exploit_missions", lambda *args, **kwargs: [])
    monkeypatch.setattr(chadman, "settle_plan", lambda *args, **kwargs: (1.25, 2, 10.0, 20, 20))

    moves = chadman.plan_moves(world, deadline=None)

    assert moves == [[1, 1.25, 20]]
