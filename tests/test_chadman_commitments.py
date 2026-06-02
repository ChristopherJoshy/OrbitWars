from src.ow import chadman

Planet = chadman.Planet


class FakeWorld:
    player = 0

    def __init__(self):
        self.planet_by_id = {
            1: Planet(1, 0, 10.0, 10.0, 1.0, 50, 1),
            2: Planet(2, -1, 20.0, 10.0, 1.0, 10, 1),
        }
        self.seen_commitments = None

    def plan_shot(self, src_id, target_id, send):
        return (0.0, 3)

    def min_ships_to_own_by(self, target_id, horizon, player, arrival_turn, planned_commitments, upper_bound):
        self.seen_commitments = planned_commitments
        return 5


def test_execute_actions_uses_passed_planned_commitments_for_repair():
    world = FakeWorld()
    commitments = {2: [(3, 0, 7)]}
    actions = [{"src_id": 1, "target_id": 2, "effective_send": 6, "send_cap": 6, "angle": 0.0, "turns": 3, "mission": "capture"}]

    moves = chadman.execute_actions(world, actions, policy=None, planned_commitments=commitments)

    assert moves == [[1, 0.0, 6]]
    assert world.seen_commitments is commitments


def test_execute_actions_preserves_empty_planned_commitments_identity():
    world = FakeWorld()
    commitments = {}
    actions = [{"src_id": 1, "target_id": 2, "effective_send": 6, "send_cap": 6, "angle": 0.0, "turns": 3, "mission": "capture"}]

    moves = chadman.execute_actions(world, actions, policy=None, planned_commitments=commitments)

    assert moves == [[1, 0.0, 6]]
    assert world.seen_commitments is commitments


def test_plan_moves_v21_does_not_pass_beam_commitments_to_final_repair(monkeypatch):
    world = FakeWorld()
    captured = {"planned_commitments": object()}

    monkeypatch.setattr(chadman, "adaptive_horizon", lambda _world: 10)
    monkeypatch.setattr(chadman, "build_modes", lambda _world: {"is_finishing": False})
    monkeypatch.setattr(chadman, "build_policy_state", lambda _world, deadline=None: {"attack_budget": {1: 20}})

    def fake_rollout_beam_select(world, policy, modes, planned_commitments, source_inventory_left, deadline=None):
        planned_commitments[2].append((3, 0, 100))
        return [{"src_id": 1, "target_id": 2, "effective_send": 6, "send_cap": 6, "angle": 0.0, "turns": 3, "mission": "capture"}]

    def fake_execute_actions(world, actions, policy=None, planned_commitments=None):
        captured["planned_commitments"] = planned_commitments
        return [[1, 0.0, 6]]

    monkeypatch.setattr(chadman, "rollout_beam_select", fake_rollout_beam_select)
    monkeypatch.setattr(chadman, "execute_actions", fake_execute_actions)

    moves = chadman.plan_moves_v21(world)

    assert moves == [[1, 0.0, 6]]
    assert captured["planned_commitments"] is None
