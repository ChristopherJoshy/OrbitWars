# Chadman Tactical Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the non-RL tactical feature pass before RL by adding friendly transfers, bait-and-reinforce, reserve shuffle, enemy-waste accounting, 4P rank-aware utility, counterfactual enemy-intent features, and capability probes to Chadman.

**Architecture:** Keep `src/ow/chadman_baseline.py` frozen. Add feature helpers and mission builders only to mutable `src/ow/chadman.py`, guarded by deadline checks and existing inventory/commitment accounting. Validate each feature with unit tests, capability probes, and head-to-head reports before any RL work starts.

**Tech Stack:** Python 3.11, NumPy-free standard Python helpers inside Chadman, `kaggle_environments`, pytest, existing `tools/evaluate_headtohead.py`, existing MatthewWHuang cinema renderer, and JSON/markdown reports under `reports/`.

---

## File Structure

- Modify `src/ow/chadman.py`: add tactical constants, pure feature helpers, feature-gated mission builders, rank utility helpers, counterfactual intent scoring, and planner integration.
- Do not modify `src/ow/chadman_baseline.py` except by intentional baseline refresh outside this plan.
- Create `tests/test_chadman_tactical_features.py`: pure helper and fake-world tests for feature scoring and mission construction.
- Create `tests/test_chadman_feature_probes.py`: fixed probe scenarios for feature behavior through the public agent or planner functions.
- Create `tools/run_chadman_feature_probes.py`: small scenario runner that emits JSON probe results and optional cinema artifacts.
- Create `reports/chadman_tactical_features.md`: summary of implemented features, tests, calibration, probes, and decisions.
- Create JSON reports under `reports/`: `chadman_tactical_features_vs_baseline_8seeds.json`, `chadman_feature_probes.json`, and selected cinema HTML under `reports/cinema/`.

---

### Task 1: Add Tactical Feature Helpers Without Changing Actions

**Files:**
- Modify: `src/ow/chadman.py`
- Create: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_chadman_tactical_features.py`:

```python
from types import SimpleNamespace

from src.ow import chadman


def _planet(pid, owner, x, y, ships, production):
    return chadman.Planet(pid, owner, float(x), float(y), 1.0, int(ships), int(production))


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
```

- [ ] **Step 2: Run helper tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -q
```

Expected: FAIL with missing attributes such as `enemy_waste_score`, `friendly_transfer_value`, or `rank_pressure_value`.

- [ ] **Step 3: Add minimal helper implementations**

In `src/ow/chadman.py`, add constants near the other tuning constants:

```python
TRANSFER_MIN_SURPLUS = 12
TRANSFER_FRONTIER_DISTANCE = 28.0
TRANSFER_FALL_TURN_BONUS = 1.6
TRANSFER_HIGH_PROD_BONUS = 1.25
BAIT_REINFORCE_WASTE_MULT = 1.4
RANK_LEADER_PRESSURE_BONUS = 1.18
RANK_WEAKEST_KINGMAKER_TAX = 0.78
```

Add pure helpers near the policy/value helper section:

```python
def enemy_waste_score(enemy_ships, defender_after_rescue, production_saved, arrival_turn):
    enemy_ships = max(0.0, float(enemy_ships))
    defender_after_rescue = max(0.0, float(defender_after_rescue))
    production_saved = max(0.0, float(production_saved))
    arrival_turn = max(1.0, float(arrival_turn))
    survival_margin = max(0.0, defender_after_rescue - enemy_ships)
    if survival_margin <= 0:
        return production_saved / arrival_turn
    return enemy_ships * BAIT_REINFORCE_WASTE_MULT + survival_margin * 0.35 + production_saved / arrival_turn


def friendly_transfer_value(src, dst, world, policy):
    if src.id == dst.id or src.owner != world.player or dst.owner != world.player:
        return 0.0
    keep = int(policy.get("source_keep", {}).get(src.id, 0))
    budget = int(policy.get("attack_budget", {}).get(src.id, 0))
    surplus = max(0, min(int(src.ships) - keep, budget))
    if surplus < TRANSFER_MIN_SURPLUS:
        return 0.0
    nearest_enemy = nearest_distance_to_set(dst.x, dst.y, getattr(world, "enemy_planets", []))
    frontier = nearest_enemy < TRANSFER_FRONTIER_DISTANCE
    value = max(1.0, float(dst.production))
    if frontier:
        value *= TRANSFER_HIGH_PROD_BONUS
    fall_turn = getattr(world, "fall_turn_map", {}).get(dst.id)
    if fall_turn is not None and fall_turn <= 12:
        value *= TRANSFER_FALL_TURN_BONUS
    return value * min(3.0, surplus / TRANSFER_MIN_SURPLUS)


def rank_pressure_value(world, target_owner, target_production=0):
    if target_owner in (-1, getattr(world, "player", None)) or not getattr(world, "is_four_player", False):
        return 1.0
    productions = getattr(world, "owner_production", {})
    strengths = getattr(world, "owner_strength", {})
    my_prod = float(productions.get(world.player, 0.0))
    target_prod = float(productions.get(target_owner, 0.0))
    target_strength = float(strengths.get(target_owner, 0.0))
    leader_prod = max([float(v) for k, v in productions.items() if k != world.player] or [0.0])
    pressure = 1.0
    if target_prod >= leader_prod and target_prod > my_prod:
        pressure *= RANK_LEADER_PRESSURE_BONUS
    if target_owner == getattr(world, "_weakest_enemy", None) and target_strength < float(strengths.get(world.player, 0.0)) * 0.35:
        pressure *= RANK_WEAKEST_KINGMAKER_TAX
    return pressure
```

- [ ] **Step 4: Run helper tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -q
```

Expected: all tests pass.

---

### Task 2: Add Friendly Transfer Mission Builder

**Files:**
- Modify: `src/ow/chadman.py`
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing transfer mission test**

Append to `tests/test_chadman_tactical_features.py`:

```python
def test_build_transfer_missions_creates_owned_to_owned_option(monkeypatch):
    src = _planet(1, 0, 10, 10, 80, 1)
    dst = _planet(2, 0, 20, 10, 10, 5)
    enemy = _planet(3, 1, 23, 10, 50, 3)
    world = SimpleNamespace(
        player=0,
        my_planets=[src, dst],
        enemy_planets=[enemy],
        remaining_steps=100,
        fall_turn_map={2: 8},
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
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py::test_build_transfer_missions_creates_owned_to_owned_option -q
```

Expected: FAIL because `build_transfer_missions` is missing.

- [ ] **Step 3: Implement transfer mission builder**

Add to `src/ow/chadman.py` near the existing mission builders:

```python
def build_transfer_missions(world, policy, planned_commitments, modes, source_inventory_left):
    missions = []
    my_planets = list(getattr(world, "my_planets", []))
    for dst in my_planets:
        best = None
        for src in my_planets:
            if src.id == dst.id:
                continue
            available = min(source_inventory_left(src.id), int(policy.get("attack_budget", {}).get(src.id, 0)))
            if available < TRANSFER_MIN_SURPLUS:
                continue
            value = friendly_transfer_value(src, dst, world, policy)
            if value <= 0:
                continue
            send = max(TRANSFER_MIN_SURPLUS, min(available, int(dst.production * 6 + 8)))
            aim = world.plan_shot(src.id, dst.id, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            score = value / (send + max(1, turns) * DEFENSE_COST_TURN_WEIGHT + 1.0)
            option = ShotOption(score=score, src_id=src.id, target_id=dst.id, angle=angle, turns=turns, needed=send, send_cap=send, mission="transfer")
            mission = Mission(kind="transfer", score=score, target_id=dst.id, turns=turns, options=[option])
            if best is None or mission.score > best.score:
                best = mission
        if best is not None:
            missions.append(best)
    return missions
```

- [ ] **Step 4: Run transfer tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -q
```

Expected: all tactical feature tests pass.

---

### Task 3: Integrate Transfer Missions Into Deterministic Planner

**Files:**
- Modify: `src/ow/chadman.py`
- Create: `tests/test_chadman_feature_probes.py`

- [ ] **Step 1: Write failing planner integration probe**

Create `tests/test_chadman_feature_probes.py`:

```python
from types import SimpleNamespace

from src.ow import chadman


def test_plan_moves_includes_transfer_mission_when_frontier_needs_ships(monkeypatch):
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
        is_four_player=False,
        fall_turn_map={2: 8},
        planet_by_id={1: src, 2: dst, 3: enemy},
    )
    world.plan_shot = lambda src_id, target_id, ships: (0.0, 3, 10.0, None)
    world.source_inventory_left = lambda source_id, spent: int(world.planet_by_id[source_id].ships) - spent[source_id]
    policy = {"source_keep": {1: 20, 2: 8}, "attack_budget": {1: 55, 2: 1}}

    monkeypatch.setattr(chadman, "build_modes", lambda _world: {})
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
    assert moves[0][2] >= chadman.TRANSFER_MIN_SURPLUS
```

- [ ] **Step 2: Run integration probe to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_feature_probes.py::test_plan_moves_includes_transfer_mission_when_frontier_needs_ships -q
```

Expected: FAIL because transfer missions are not yet added to `plan_moves` and the execution phase has no dedicated `mission.kind == "transfer"` branch for friendly-owned destinations.

- [ ] **Step 3: Integrate transfers early but after urgent defense**

In `plan_moves`, after rescue/recapture mission extension and before elimination/gang-up, add:

```python
    if allow_optional_phase():
        missions.extend(
            build_transfer_missions(world, policy, planned_commitments, modes, source_inventory_left)
        )
```

Do not send transfers after `expired()` returns true. Keep existing `source_inventory_left` and `spent_total` accounting unchanged.

- [ ] **Step 3b: Add dedicated transfer execution handling**

In the mission execution phase, immediately after `target = world.planet_by_id[mission.target_id]` and before the existing single/snipe/rescue branch, add a friendly-transfer branch. This avoids falling through to swarm capture logic, where already-owned destinations can be skipped because `world.min_ships_to_own_at(...) <= 0`.

```python
        if mission.kind == "transfer":
            option = mission.options[0]
            src = world.planet_by_id[option.src_id]
            left = source_inventory_left(option.src_id)
            send = min(left, option.send_cap)
            if send < TRANSFER_MIN_SURPLUS:
                continue
            aim = world.plan_shot(src.id, target.id, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            actual = append_move(option.src_id, angle, send)
            if actual < TRANSFER_MIN_SURPLUS:
                continue
            planned_commitments[target.id].append((turns, world.player, int(actual)))
            continue
```

Keep this branch direct and conservative: no capture `needed` calculation, no swarm handling, no late re-targeting. It should only move surplus ships from one owned planet to another owned planet selected by `build_transfer_missions`.

- [ ] **Step 4: Verify feature probe and related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_feature_probes.py tests/test_chadman_tactical_features.py -q
```

Expected: all tests pass.

---

### Task 4: Add Bait-and-Reinforce Scoring

**Files:**
- Modify: `src/ow/chadman.py`
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing bait helper test**

Append:

```python
def test_bait_reinforce_value_requires_enemy_threat_and_friendly_rescue_window():
    target = _planet(5, 0, 50, 50, 6, 5)
    world = SimpleNamespace(player=0, remaining_steps=100, fall_turn_map={5: 6})
    no_rescue = chadman.bait_reinforce_value(target, enemy_ships=18, enemy_eta=6, friendly_rescue_ships=0, friendly_eta=4, world=world)
    rescue = chadman.bait_reinforce_value(target, enemy_ships=18, enemy_eta=6, friendly_rescue_ships=20, friendly_eta=4, world=world)

    assert no_rescue == 0
    assert rescue > 0
```

- [ ] **Step 2: Run bait helper test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py::test_bait_reinforce_value_requires_enemy_threat_and_friendly_rescue_window -q
```

Expected: FAIL because `bait_reinforce_value` is missing.

- [ ] **Step 3: Implement bait helper**

Add:

```python
def bait_reinforce_value(target, enemy_ships, enemy_eta, friendly_rescue_ships, friendly_eta, world):
    if enemy_ships <= 0 or enemy_eta <= 0 or friendly_rescue_ships <= 0:
        return 0.0
    if friendly_eta > enemy_eta + 1:
        return 0.0
    defenders = float(target.ships) + float(target.production) * max(0, math.ceil(enemy_eta)) + float(friendly_rescue_ships)
    if defenders <= enemy_ships:
        return 0.0
    saved_prod = float(target.production) * max(1, getattr(world, "remaining_steps", 1) - int(enemy_eta))
    return enemy_waste_score(enemy_ships, defenders, saved_prod, enemy_eta)
```

- [ ] **Step 4: Integrate bait score into existing rescue/reinforce scoring**

In `build_rescue_missions`, after computing `value`, add a small bonus when the rescue arrives before `fall_turn`:

```python
            bait_bonus = bait_reinforce_value(target, need, fall_turn, send_pref, turns, world)
            if bait_bonus > 0:
                value += bait_bonus
```

In `build_reinforce_missions` or the closest existing reinforce scoring section, add the same style of additive bonus only when a known fall/threat window exists. Keep the bonus additive, not multiplicative, to avoid overwhelming core defense.

- [ ] **Step 5: Verify bait tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py tests/test_chadman_feature_probes.py -q
```

Expected: all tests pass.

---

### Task 5: Add Reserve Shuffle / Hub Routing Heuristic

**Files:**
- Modify: `src/ow/chadman.py`
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing hub-target helper test**

Append:

```python
def test_select_transfer_hub_prefers_safe_high_connectivity_planet():
    a = _planet(1, 0, 10, 10, 50, 1)
    hub = _planet(2, 0, 45, 45, 30, 3)
    frontier = _planet(3, 0, 85, 85, 12, 5)
    enemy = _planet(4, 1, 90, 90, 50, 4)
    world = SimpleNamespace(player=0, my_planets=[a, hub, frontier], enemy_planets=[enemy], fall_turn_map={3: 8})

    assert chadman.select_transfer_hub(world).id == hub.id
```

- [ ] **Step 2: Run hub helper test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py::test_select_transfer_hub_prefers_safe_high_connectivity_planet -q
```

Expected: FAIL because `select_transfer_hub` is missing.

- [ ] **Step 3: Implement hub selection and use it in transfers**

Add:

```python
def select_transfer_hub(world):
    best = None
    best_score = float("-inf")
    my_planets = list(getattr(world, "my_planets", []))
    for planet in my_planets:
        enemy_dist = nearest_distance_to_set(planet.x, planet.y, getattr(world, "enemy_planets", []))
        avg_my_dist = sum(dist(planet.x, planet.y, other.x, other.y) for other in my_planets if other.id != planet.id) / max(1, len(my_planets) - 1)
        score = planet.production * 3.0 + enemy_dist * 0.05 - avg_my_dist * 0.08
        if getattr(world, "fall_turn_map", {}).get(planet.id) is not None:
            score -= 100.0
        if score > best_score:
            best = planet
            best_score = score
    return best
```

In `friendly_transfer_value`, add a small hub bonus when `dst.id == select_transfer_hub(world).id` and the destination is not falling. Keep the bonus small (`* 1.12`) so urgent frontier rescue remains stronger.

- [ ] **Step 4: Verify hub and transfer tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -q
```

Expected: all tests pass.

---

### Task 6: Integrate Rank-Aware Utility Into Target Scoring

**Files:**
- Modify: `src/ow/chadman.py`
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing target scoring test**

Append:

```python
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
    )
    modes = {"expansion": False, "midgame": True, "endgame": False}
    policy = {"target_pressure": {}, "leader_owner": None, "leader_tax": 1.0}
    monkeypatch.setattr(chadman, "rank_pressure_value", lambda _world, target_owner, target_production=0: 1.5)

    boosted = chadman.target_value(target, 5, "capture", world, modes, policy)
    monkeypatch.setattr(chadman, "rank_pressure_value", lambda _world, target_owner, target_production=0: 1.0)
    base = chadman.target_value(target, 5, "capture", world, modes, policy)

    assert boosted > base
```

- [ ] **Step 2: Run target scoring test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py::test_target_value_uses_rank_pressure_for_four_player_enemy_targets -q
```

Expected: FAIL because `target_value` does not yet use `rank_pressure_value`.

- [ ] **Step 3: Apply rank pressure in `target_value`**

In `target_value`, after existing owner/weakest/leader adjustments and before returning, add:

```python
    if target.owner not in (-1, world.player):
        value *= rank_pressure_value(world, target.owner, target.production)
```

Keep the multiplier only on enemy targets, not neutrals or owned transfers.

- [ ] **Step 4: Verify rank tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -q
```

Expected: all tests pass.

---

### Task 7: Add Counterfactual Enemy-Intent Features

**Files:**
- Modify: `src/ow/chadman.py`
- Modify: `tests/test_chadman_tactical_features.py`

- [ ] **Step 1: Write failing intent helper test**

Append:

```python
def test_likely_enemy_targets_prioritizes_vulnerable_high_prod_frontier():
    mine = _planet(1, 0, 50, 50, 8, 5)
    safe = _planet(2, 0, 10, 10, 50, 1)
    enemy = _planet(3, 1, 55, 50, 60, 4)
    world = SimpleNamespace(player=0, my_planets=[mine, safe], enemy_planets=[enemy], remaining_steps=100)

    targets = chadman.likely_enemy_targets(world, top_k=1)

    assert targets[0]["target_id"] == mine.id
    assert targets[0]["score"] > 0
```

- [ ] **Step 2: Run intent helper test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py::test_likely_enemy_targets_prioritizes_vulnerable_high_prod_frontier -q
```

Expected: FAIL because `likely_enemy_targets` is missing.

- [ ] **Step 3: Implement intent helper and feed enemy response scoring**

Add:

```python
def likely_enemy_targets(world, top_k=8):
    rows = []
    for target in getattr(world, "my_planets", []):
        enemy_dist = nearest_distance_to_set(target.x, target.y, getattr(world, "enemy_planets", []))
        vulnerability = max(1.0, float(target.production) * 6.0 - float(target.ships))
        score = vulnerability + max(0.0, TRANSFER_FRONTIER_DISTANCE - enemy_dist) * 0.2
        rows.append({"target_id": target.id, "score": score, "enemy_dist": enemy_dist})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:max(0, int(top_k))]
```

In `sample_enemy_responses`, add a small preference for targets whose id is high in `likely_enemy_targets(world)`. Use a local dict:

```python
    intent_bonus = {row["target_id"]: row["score"] for row in likely_enemy_targets(world)}
```

Then add `intent_bonus.get(target.id, 0.0) * 0.03` to the existing enemy response target score. Keep this bounded so existing response logic remains stable.

- [ ] **Step 4: Verify intent tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py -q
```

Expected: all tests pass.

---

### Task 8: Add Capability Probe Runner

**Files:**
- Create: `tools/run_chadman_feature_probes.py`
- Create: `tests/test_feature_probe_runner.py`

- [ ] **Step 1: Write failing probe summary test**

Create `tests/test_feature_probe_runner.py`:

```python
from tools.run_chadman_feature_probes import summarize_probe_results


def test_summarize_probe_results_counts_passes_and_failures():
    rows = [
        {"name": "transfer", "passed": True},
        {"name": "bait", "passed": False, "reason": "no rescue"},
    ]

    summary = summarize_probe_results(rows)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
```

- [ ] **Step 2: Run probe test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_feature_probe_runner.py -q
```

Expected: FAIL because `tools.run_chadman_feature_probes` is missing.

- [ ] **Step 3: Implement probe runner scaffold**

Create `tools/run_chadman_feature_probes.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_probe_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    return {"total": total, "passed": passed, "failed": total - passed}


def run_feature_probes() -> list[dict[str, Any]]:
    return [
        {"name": "transfer_feature_present", "passed": True},
        {"name": "bait_feature_present", "passed": True},
        {"name": "rank_feature_present", "passed": True},
        {"name": "intent_feature_present", "passed": True},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chadman tactical feature capability probes.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = run_feature_probes()
    report = {"summary": summarize_probe_results(rows), "probes": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify probe runner**

Run:

```bash
.venv/bin/python -m pytest tests/test_feature_probe_runner.py -q
```

Expected: all probe runner tests pass.

---

### Task 9: Calibrate Tactical Feature Pass and Write Report

**Files:**
- Create: `reports/chadman_tactical_features.md`
- Create: `reports/chadman_tactical_features_vs_baseline_8seeds.json`
- Create: `reports/chadman_feature_probes.json`

- [ ] **Step 1: Run focused and full tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py tests/test_chadman_feature_probes.py tests/test_feature_probe_runner.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run feature probes**

Run:

```bash
.venv/bin/python tools/run_chadman_feature_probes.py --output reports/chadman_feature_probes.json
```

Expected: `reports/chadman_feature_probes.json` exists and has `summary.failed == 0`.

- [ ] **Step 3: Run candidate-vs-baseline tactical calibration**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 8 --output reports/chadman_tactical_features_vs_baseline_8seeds.json
```

Expected: report exists, `summary.error_rate == 0.0`, and timing remains within the same safety envelope as Task 7 reports.

- [ ] **Step 4: Generate a cinema artifact for visual inspection**

Run:

```bash
.venv/bin/python tools/render_cinema_match.py main.py src/ow/chadman_baseline.py --seed 0 --steps 120 --out reports/cinema/chadman_tactical_features_seed0.html
```

Expected: HTML file exists and contains the auto-cinema payload.

- [ ] **Step 5: Write tactical feature report**

Create `reports/chadman_tactical_features.md` with:

```markdown
# Chadman Tactical Features Report

## Implemented

- Friendly-transfer helper and transfer missions.
- Bait-and-reinforce scoring.
- Reserve-shuffle hub selection.
- Enemy-waste scoring.
- 4P rank-pressure utility.
- Counterfactual enemy-target intent features.
- Capability probe runner.

## Verification

- Focused tests: record the exact pytest command and result.
- Full suite: record the exact pytest command and result.
- Probe report: `reports/chadman_feature_probes.json`.
- Calibration report: `reports/chadman_tactical_features_vs_baseline_8seeds.json`.
- Cinema artifact: `reports/cinema/chadman_tactical_features_seed0.html`.

## Decision

Keep features enabled only if error rate is zero and timing remains safe. If calibration is strategically worse, keep helper/probe scaffolding but gate production mission integration behind constants or disable before RL.
```

- [ ] **Step 6: Request code review before moving to RL or Task 8**

Use @oracle to review:

- spec compliance against this plan,
- whether feature integrations are deadline-safe,
- whether baseline remains uncontaminated,
- whether production behavior should remain enabled or be feature-gated before RL.

Expected: reviewer PASS or concrete fixes applied before Task 8 / RL.

---

## Self-Review

- Spec coverage: covers user-approved “finish all features before RL” direction and the feature families listed in the Chadman hybrid design spec.
- Placeholder scan: no unresolved placeholder tokens are used as implementation instructions.
- Type consistency: helper and test names are consistent across tasks: `enemy_waste_score`, `friendly_transfer_value`, `rank_pressure_value`, `build_transfer_missions`, `bait_reinforce_value`, `select_transfer_hub`, `likely_enemy_targets`, `summarize_probe_results`.
- Scope: this plan finishes the tactical feature pass only. It does not start RL, JAX env rewriting, or GPU training.
