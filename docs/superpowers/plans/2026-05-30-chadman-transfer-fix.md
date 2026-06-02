# Chadman Transfer Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Chadman transfer missions enabled while preventing low-value early/mid transfer churn from stealing offensive tempo.

**Architecture:** Add demand and anti-churn helpers around the existing transfer mission builder in `src/ow/chadman.py`. `friendly_transfer_value` stays compatible for direct callers, while `build_transfer_missions` uses demand-backed ship amounts, dynamic minimums, source opportunity checks, in-flight saturation, and reverse-flow rejection before emitting missions. Verification is deterministic: unit tests, focused suites, and short head-to-head calibration only; no RL/self-play training/submissions.

**Tech Stack:** Python, pytest, existing Orbit Wars `chadman.py` tactical planner, existing `tools/evaluate_headtohead.py` for deterministic calibration.

---

## File Structure

- Modify `src/ow/chadman.py`
  - Add transfer constants near current tactical constants.
  - Add pure helpers near existing `select_transfer_hub` / `friendly_transfer_value` helpers.
  - Update `build_transfer_missions(...)` to use the new demand model.
  - Keep transfer execution safety checks intact.
- Modify `tests/test_chadman_tactical_features.py`
  - Add focused unit tests for demand gates and helper behavior.
- Modify `tests/test_chadman_feature_probes.py`
  - Update old transfer integration probe so it creates an allowed late/demand-backed transfer instead of depending on early routine transfer.
  - Add a regression probe for early churn rejection through `plan_moves`.
- Create optional report after implementation: `reports/chadman_transfer_fix_calibration.json` only if deterministic calibration is run.
- Do not modify `src/ow/chadman_baseline.py`.

---

### Task 1: Add Transfer Demand Helper Tests

**Files:**
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing helper tests**

Append these tests after the existing transfer hub tests and before rank/intent tests:

```python
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
```

- [ ] **Step 2: Run helper tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -k "transfer_dynamic_minimum or transfer_destination_demand" -q
```

Expected: FAIL because `transfer_dynamic_min_send` and `transfer_destination_demand` do not exist yet.

- [ ] **Step 3: Do not implement in this task**

This task only adds failing tests. Implementation happens in Task 2.

---

### Task 2: Implement Demand Helpers

**Files:**
- Modify: `src/ow/chadman.py`
- Test: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Add transfer constants near existing transfer constants**

Add these constants near `TRANSFER_MIN_SURPLUS`:

```python
TRANSFER_EARLY_ROUTINE_MIN_STEP = 60
TRANSFER_LATE_CONSOLIDATION_STEP = 180
TRANSFER_MIDGAME_ROUTINE_MIN_SEND = 40
TRANSFER_LATE_ROUTINE_MIN_SEND = 80
TRANSFER_SOURCE_SURPLUS_FRACTION = 0.35
TRANSFER_TOTAL_STRENGTH_FRACTION = 0.025
TRANSFER_REVERSE_FLOW_HORIZON = 24
TRANSFER_INCOMING_SATURATION_MARGIN = 2
TRANSFER_ATTACK_OPPORTUNITY_TURN_LIMIT = 10
TRANSFER_ATTACK_OPPORTUNITY_SHIP_MARGIN = 8
```

- [ ] **Step 2: Add helper functions near `select_transfer_hub`**

Add these helpers before `friendly_transfer_value`:

```python
def transfer_current_step(world):
    return max(0, TOTAL_STEPS - int(getattr(world, "remaining_steps", TOTAL_STEPS)))


def transfer_dynamic_min_send(src, dst, world, policy, available, emergency=False):
    available = max(0, int(available))
    if available <= 0:
        return TRANSFER_MIN_SURPLUS
    if emergency:
        return TRANSFER_MIN_SURPLUS
    step = transfer_current_step(world)
    surplus_floor = int(math.ceil(available * TRANSFER_SOURCE_SURPLUS_FRACTION))
    total_floor = int(math.ceil(max(0, int(getattr(world, "my_total", 0))) * TRANSFER_TOTAL_STRENGTH_FRACTION))
    phase_floor = TRANSFER_LATE_ROUTINE_MIN_SEND if step >= TRANSFER_LATE_CONSOLIDATION_STEP or getattr(world, "is_late", False) else TRANSFER_MIDGAME_ROUTINE_MIN_SEND
    return min(available, max(TRANSFER_MIN_SURPLUS, phase_floor, surplus_floor, total_floor))


def incoming_friendly_transfer_ships(world, target_id, horizon):
    total = 0
    horizon = max(1, int(math.ceil(float(horizon))))
    for eta, owner, ships in getattr(world, "arrivals_by_planet", {}).get(target_id, []):
        if owner == getattr(world, "player", None) and max(1, int(math.ceil(float(eta)))) <= horizon:
            total += max(0, int(ships))
    return total


def has_reverse_friendly_flow(world, src_id, dst_id, horizon=TRANSFER_REVERSE_FLOW_HORIZON):
    player = getattr(world, "player", None)
    planets = getattr(world, "planet_by_id", {})
    src = planets.get(src_id)
    if src is None:
        return False
    horizon = max(1, int(horizon))
    for fleet in getattr(world, "fleets", []):
        if getattr(fleet, "owner", None) != player or getattr(fleet, "from_planet_id", None) != dst_id:
            continue
        try:
            target, eta = world.fleet_target(fleet, horizon)
        except Exception:
            continue
        if getattr(target, "id", None) == src_id and eta <= horizon:
            return True
    return False


def source_has_near_attack_opportunity(src, world, policy, available):
    available = max(0, int(available))
    if available < TRANSFER_MIN_SURPLUS:
        return False
    targets = list(getattr(world, "enemy_planets", [])) + list(getattr(world, "neutral_planets", []))
    for target in targets:
        if getattr(target, "id", None) == getattr(src, "id", None):
            continue
        needed = int(getattr(target, "ships", 0)) + TRANSFER_ATTACK_OPPORTUNITY_SHIP_MARGIN
        if available < needed:
            continue
        try:
            aim = world.plan_shot(src.id, target.id, min(available, needed))
        except Exception:
            aim = None
        if aim is None:
            continue
        _, turns, _, _ = aim
        if turns <= TRANSFER_ATTACK_OPPORTUNITY_TURN_LIMIT:
            return True
    return False


def transfer_destination_demand(src, dst, world, policy, planned_commitments, turns, available, transfer_hub_id):
    if src.id == dst.id or src.owner != world.player or dst.owner != world.player:
        return None
    available = max(0, int(available))
    if available < TRANSFER_MIN_SURPLUS:
        return None
    turns = max(1, int(math.ceil(float(turns))))
    fall_turn = getattr(world, "fall_turn_map", {}).get(dst.id)
    incoming = incoming_friendly_transfer_ships(world, dst.id, turns + TRANSFER_INCOMING_SATURATION_MARGIN)
    planned = sum(max(0, int(ships)) for eta, owner, ships in planned_commitments.get(dst.id, []) if owner == world.player and eta <= turns + TRANSFER_INCOMING_SATURATION_MARGIN)
    existing_friendly = incoming + planned

    if fall_turn is not None and turns <= fall_turn:
        threat = enemy_threat_ships_at_turn(world, dst.id, fall_turn)
        projected = int(dst.ships) + int(dst.production) * max(0, int(fall_turn)) + existing_friendly
        deficit = max(0, threat - projected)
        if deficit > 0:
            ships = min(available, max(TRANSFER_MIN_SURPLUS, deficit + TRANSFER_INCOMING_SATURATION_MARGIN))
            return {"kind": "emergency", "ships": ships, "value": float(threat + max(1, dst.production) * 8)}

    step = transfer_current_step(world)
    late = step >= TRANSFER_LATE_CONSOLIDATION_STEP or getattr(world, "is_late", False)
    if not late:
        return None
    if has_reverse_friendly_flow(world, src.id, dst.id):
        return None
    if source_has_near_attack_opportunity(src, world, policy, available):
        return None
    dynamic_min = transfer_dynamic_min_send(src, dst, world, policy, available, emergency=False)
    if available < dynamic_min:
        return None
    nearest_enemy = nearest_distance_to_set(dst.x, dst.y, getattr(world, "enemy_planets", []))
    is_frontier = nearest_enemy < TRANSFER_FRONTIER_DISTANCE
    is_hub = transfer_hub_id == dst.id
    if not (is_frontier or is_hub):
        return None
    demand_cap = max(dynamic_min, int(dst.production) * 20 + 20 - existing_friendly)
    ships = min(available, max(dynamic_min, demand_cap))
    if ships < dynamic_min:
        return None
    value = float(max(1, dst.production) * (24 if is_frontier else 16))
    if is_hub:
        value *= 1.15
    return {"kind": "late_consolidation", "ships": ships, "value": value}
```

- [ ] **Step 3: Run helper tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -k "transfer_dynamic_minimum or transfer_destination_demand" -q
```

Expected: PASS for the helper tests.

---

### Task 3: Wire Demand Helpers into Transfer Builder

**Files:**
- Modify: `src/ow/chadman.py`
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Add failing builder tests**

Append these tests after the Task 1 helper tests:

```python
def test_build_transfer_missions_rejects_early_small_churn_without_demand():
    src = _planet(1, 0, 10, 10, 80, 2)
    dst = _planet(2, 0, 22, 10, 15, 3)
    enemy = _planet(3, 1, 90, 90, 50, 4)
    world = SimpleNamespace(
        player=0,
        my_total=160,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=440,
        is_early=True,
        is_late=False,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 3, 10.0, None)
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 30, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 30 if source_id == 1 else 1)

    assert missions == []


def test_build_transfer_missions_accepts_late_large_consolidation():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 24, 10, 40, 5)
    enemy = _planet(3, 1, 32, 10, 70, 4)
    world = SimpleNamespace(
        player=0,
        my_total=520,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=250,
        is_early=False,
        is_late=True,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 4, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 20}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 1)

    assert missions
    option = missions[0].options[0]
    assert option.src_id == 1
    assert option.target_id == 2
    assert option.send_cap >= 80


def test_build_transfer_missions_blocks_saturated_destination():
    src = _planet(1, 0, 10, 10, 260, 2)
    dst = _planet(2, 0, 24, 10, 40, 5)
    enemy = _planet(3, 1, 32, 10, 70, 4)
    world = SimpleNamespace(
        player=0,
        my_total=520,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        neutral_planets=[],
        remaining_steps=250,
        is_early=False,
        is_late=True,
        fall_turn_map={},
        arrivals_by_planet={2: [(4, 0, 200)]},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, dst_id, ships: (0.0, 4, 10.0, None)
    policy = {"source_keep": {1: 80, 2: 20}, "attack_budget": {1: 170, 2: 1}}

    missions = chadman.build_transfer_missions(world, policy, {}, {}, lambda source_id: 170 if source_id == 1 else 1)

    assert missions == []
```

- [ ] **Step 2: Run builder tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -k "build_transfer_missions_rejects_early_small_churn or accepts_late_large_consolidation or blocks_saturated_destination" -q
```

Expected: at least the early-churn and saturation tests fail against the current production builder.

- [ ] **Step 3: Update `build_transfer_missions`**

Replace the current value/send block inside the nested `for src in my_planets` loop with demand-backed logic:

```python
            aim_probe = world.plan_shot(src.id, dst.id, TRANSFER_MIN_SURPLUS)
            if aim_probe is None:
                continue
            _, probe_turns, _, _ = aim_probe
            demand = transfer_destination_demand(
                src,
                dst,
                world,
                policy,
                planned_commitments,
                probe_turns,
                available,
                transfer_hub_id,
            )
            if demand is None:
                continue
            send = min(available, max(TRANSFER_MIN_SURPLUS, int(demand["ships"])))
            if send < transfer_dynamic_min_send(src, dst, world, policy, available, emergency=demand["kind"] == "emergency"):
                continue
            aim = world.plan_shot(src.id, dst.id, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            fall_turn = getattr(world, "fall_turn_map", {}).get(dst.id)
            if fall_turn is not None and turns > fall_turn:
                continue
            score = float(demand["value"]) / (send + max(1, turns) * DEFENSE_COST_TURN_WEIGHT + 1.0)
```

Keep the existing `ShotOption`, `Mission`, and best-per-destination structure.

- [ ] **Step 4: Run builder tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -k "transfer_dynamic_minimum or transfer_destination_demand or build_transfer_missions_rejects_early_small_churn or accepts_late_large_consolidation or blocks_saturated_destination" -q
```

Expected: PASS.

---

### Task 4: Update Plan-Move Transfer Probes

**Files:**
- Modify: `tests/test_chadman_feature_probes.py`

- [ ] **Step 1: Update existing transfer-present probe to a late allowed transfer**

In `test_plan_moves_includes_transfer_mission_when_frontier_needs_ships`, change the fake world/policy setup so the transfer is late and large:

```python
    src = chadman.Planet(1, 0, 10.0, 10.0, 1.0, 260, 2)
    dst = chadman.Planet(2, 0, 20.0, 10.0, 1.0, 40, 5)
    enemy = chadman.Planet(3, 1, 23.0, 10.0, 1.0, 80, 3)
```

Set `remaining_steps=250`, `is_early=False`, `is_late=True`, `fall_turn_map={}`, `arrivals_by_planet={}`, and policy `source_keep={1: 80, 2: 20}`, `attack_budget={1: 170, 2: 1}`. Update assertions to require `moves[0][2] >= 80`.

- [ ] **Step 2: Add failing plan-move early-churn rejection probe**

Append:

```python
def test_plan_moves_rejects_early_small_transfer_churn(monkeypatch):
    src = chadman.Planet(1, 0, 10.0, 10.0, 1.0, 80, 2)
    dst = chadman.Planet(2, 0, 20.0, 10.0, 1.0, 15, 3)
    enemy = chadman.Planet(3, 1, 90.0, 90.0, 1.0, 50, 4)
    world = SimpleNamespace(
        player=0,
        my_total=160,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        planets=[src, dst, enemy],
        remaining_steps=440,
        is_early=True,
        is_late=False,
        is_very_late=False,
        is_four_player=False,
        is_total_war=False,
        fall_turn_map={},
        arrivals_by_planet={},
        fleets=[],
        planet_by_id={1: src, 2: dst, 3: enemy},
        comet_ids=set(),
        neutral_planets=[],
        static_neutral_planets=[],
    )
    world.plan_shot = lambda src_id, target_id, ships: (0.0, 3, 10.0, None)
    world.source_inventory_left = lambda source_id, spent: int(world.planet_by_id[source_id].ships) - spent[source_id]
    world.hold_status = lambda *args, **kwargs: {"holds_full": True, "fall_turn": None}
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 30, 2: 1}}

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
```

- [ ] **Step 3: Run feature probes**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_feature_probes.py -q
```

Expected: PASS after Task 3 implementation. If it fails, adjust tests or implementation only to preserve the intended behavior: late large transfers can happen; early routine churn cannot.

---

### Task 5: Verification and Short Calibration

**Files:**
- Read-only/optional report: `reports/chadman_transfer_fix_calibration.json`

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py tests/test_chadman_feature_probes.py tests/test_feature_probe_runner.py -q
```

Expected: PASS.

- [ ] **Step 2: Run conformance tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_conformance.py -q
```

Expected: PASS with only existing unrelated warnings.

- [ ] **Step 3: Run deterministic 8-seed calibration only after tests pass**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 8 --output reports/chadman_transfer_fix_calibration.json
```

Expected: command completes with `error_rate = 0.0`. This is deterministic evaluation, not RL/self-play training.

- [ ] **Step 4: Interpret calibration**

Open `reports/chadman_transfer_fix_calibration.json` and compare to prior tactical result:

- Prior tactical pass: 2 wins, 14 losses, win rate 0.125.
- Required improvement target for this iteration: zero errors and fewer transfer-churn losses than before. A tie-like result on known bad seeds is acceptable as progress; a full win-rate improvement is better.
- If still strategically bad, do not start RL/training. Return to systematic debugging and inspect transfer logs/ablation again.

---

## Plan Self-Review

- Spec coverage: covers demand gates, dynamic minimum, in-flight saturation, reverse-flow, source opportunity, tests, and deterministic calibration.
- Placeholder scan: no placeholders remain; each task includes exact file paths, test code, implementation snippets, commands, and expected outcomes.
- Type consistency: helper names and signatures are consistent across tests and implementation snippets.
- Scope: focused only on transfer behavior; does not alter baseline, RL, submissions, or unrelated tactical features.
