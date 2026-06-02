# Chadman Diagnosis Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transplant Chad v24 as the production baseline, build fast Chad-vs-candidate calibration and replay-analysis tooling, verify simulator/spec conformance, and implement the first measurable planner repairs before any ML work.

**Architecture:** Keep a frozen Chad baseline module and a mutable Chadman candidate module. Root `main.py` delegates to the candidate; calibration tools compare candidate against the frozen baseline. Diagnosis tools and tests prove official-environment compatibility before training a value/rerank model.

**Tech Stack:** Python 3.11, NumPy, `kaggle_environments`, pytest, Kaggle episode JSON replays, existing repo proof/submission tooling. Training is intentionally deferred until this diagnosis phase passes.

**Deferred Phase 2 training requirements captured:** after this diagnosis/repair phase, train 2-player and 4-player policies/value/rerankers separately from top-player replays plus self-play, combine them with a game-state switcher, and add a Fedora Pygame/SDL2 live match viewer, cinema-mode rollout snapshots, plus structured training logs/metrics. First training pass is 12 hours for 2P followed by 12 hours for 4P. The evidence-backed default is PPO/MAPPO-style self-play with an opponent snapshot pool, imitation bootstrap from top-player replays where clean, OpenAI-style autocurriculum/capability-probe evaluation for emergent strategies, arXiv:2509.15042-style BC-before-RL stability ablations, arXiv:2304.13004-style centralized training/critic framing for RTS-like control, arXiv:2408.01072-style population/opponent-sampling matrix tracking, and PSRO/league/PBT only after baseline stability. If a JAX environment is built for >=10k SPS, it must prove 1:1 parity with official Orbit Wars before any RL results are trusted. Use GPU for model training and CPU workers for environment simulation/viewer rendering. Final inference must be CPU/NumPy-safe with large headroom for Kaggle host CPU/memory variance. Do not start that training until Phase 1 conformance, calibration, and repair gates are complete.

---

## File Structure

- Create `src/ow/chadman_baseline.py`: frozen copy of `/home/chris/Downloads/experiments/chadmain_v24.py` used only as comparison baseline.
- Create `src/ow/chadman.py`: mutable candidate copy of Chad v24 used by production `main.py`.
- Modify `main.py`: delegate `agent(obs, config=None)` to `src.ow.chadman.agent` with a safe fallback to `[]`.
- Create `tests/test_chadman_transplant.py`: import, schema, smoke, and `main.py` delegation tests.
- Create `tools/evaluate_headtohead.py`: general head-to-head runner for candidate vs baseline Chad, both orders, timing/error reports.
- Create `tools/render_cinema_match.py`: visible single-match runner that uses MatthewWHuang/orbit-wars `visualize.py` to write auto-cinema HTML for human inspection.
- Create `tests/test_evaluate_headtohead.py`: smoke tests for report aggregation helpers.
- Create `tests/test_cinema_match.py`: tests for Matthew visualizer loading and auto-cinema HTML output.
- Create `tools/analyze_orbit_replays.py`: local replay corpus inventory and sampled action/outcome summaries.
- Create `tests/test_replay_analysis.py`: JSON parsing and aggregation tests using tiny fixture replay objects.
- Create `tests/test_chadman_conformance.py`: official schema/speed/turn-order/collision/combat conformance tests.
- Modify `src/ow/chadman.py`: planner fixes/ablations only after failing tests prove each issue.
- Create `tests/test_chadman_commitments.py`: failing tests for planned-commitment propagation into final execution.
- Create `tests/test_chadman_budgeting.py`: failing tests for `_META_BANDIT` ablation and `remainingOverageTime` deadline behavior.
- Create report outputs under `reports/`: calibration JSON, replay inventory JSON, and diagnostic markdown.

---

### Task 1: Transplant Chad v24 as Frozen Baseline and Candidate

**Files:**
- Create: `src/ow/chadman_baseline.py`
- Create: `src/ow/chadman.py`
- Modify: `main.py`
- Test: `tests/test_chadman_transplant.py`

- [ ] **Step 1: Write failing transplant tests**

Create `tests/test_chadman_transplant.py`:

```python
import importlib


def test_chadman_modules_export_agent_and_world_builder():
    baseline = importlib.import_module("src.ow.chadman_baseline")
    candidate = importlib.import_module("src.ow.chadman")

    assert callable(baseline.agent)
    assert callable(candidate.agent)
    assert callable(baseline.build_world)
    assert callable(candidate.build_world)


def test_chadman_planet_and_fleet_schema_match_official():
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
    from src.ow import chadman

    assert chadman.Planet._fields == Planet._fields
    assert chadman.Fleet._fields == Fleet._fields


def test_root_main_exports_chadman_agent():
    import main
    from src.ow import chadman

    assert callable(main.agent)
    assert main.agent.__name__ == "agent"
    assert main._CHADMAN_AGENT is chadman.agent
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_transplant.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.ow.chadman'` or missing `_CHADMAN_AGENT`.

- [ ] **Step 3: Copy Chad source into baseline and candidate modules**

Run:

```bash
cp /home/chris/Downloads/experiments/chadmain_v24.py src/ow/chadman_baseline.py
cp /home/chris/Downloads/experiments/chadmain_v24.py src/ow/chadman.py
```

Then add this comment at the top of `src/ow/chadman_baseline.py`:

```python
# Frozen baseline copy of /home/chris/Downloads/experiments/chadmain_v24.py.
# Do not edit except to refresh the baseline intentionally.
```

Add this comment at the top of `src/ow/chadman.py`:

```python
# Mutable Chadman candidate derived from /home/chris/Downloads/experiments/chadmain_v24.py.
```

- [ ] **Step 4: Modify root entrypoint to delegate to Chadman**

Replace `main.py` with:

```python
from src.ow.chadman import agent as _CHADMAN_AGENT


def agent(obs, config=None):
    try:
        return _CHADMAN_AGENT(obs, config)
    except Exception:
        return []
```

- [ ] **Step 5: Run transplant tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_transplant.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Run full suite and record expected old-policy failures**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: if tests tied to old greedy `main.py` behavior fail, list exact tests and defer retargeting to Task 2/3. Do not modify Chad logic in this task.

---

### Task 2: Build Chad-vs-Chadman Calibration Runner

**Files:**
- Create: `tools/evaluate_headtohead.py`
- Test: `tests/test_evaluate_headtohead.py`

- [ ] **Step 1: Write failing aggregation tests**

Create `tests/test_evaluate_headtohead.py`:

```python
from tools.evaluate_headtohead import summarize_results


def test_summarize_results_counts_candidate_wins_and_errors():
    rows = [
        {"candidate_reward": 1, "baseline_reward": -1, "candidate_status": "DONE", "baseline_status": "DONE", "candidate_time_mean": 0.01},
        {"candidate_reward": -1, "baseline_reward": 1, "candidate_status": "DONE", "baseline_status": "DONE", "candidate_time_mean": 0.02},
        {"candidate_reward": None, "baseline_reward": 1, "candidate_status": "ERROR", "baseline_status": "DONE", "candidate_time_mean": 0.03},
    ]

    summary = summarize_results(rows)

    assert summary["matches"] == 3
    assert summary["candidate_wins"] == 1
    assert summary["win_rate"] == 1 / 3
    assert summary["errors"] == 1
    assert summary["candidate_time_mean"] == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate_headtohead.py -q
```

Expected: FAIL with missing `tools.evaluate_headtohead`.

- [ ] **Step 3: Implement calibration runner**

Create `tools/evaluate_headtohead.py` with these public functions and CLI:

```python
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from kaggle_environments import make


def _load_agent(ref: str):
    if ref.endswith(".py") or "/" in ref:
        return ref
    module_name, func_name = ref.rsplit(":", 1) if ":" in ref else (ref, "agent")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = len(rows)
    wins = sum(1 for r in rows if r.get("candidate_reward") == 1)
    errors = sum(1 for r in rows if r.get("candidate_status") != "DONE" or r.get("baseline_status") != "DONE")
    times = [float(r.get("candidate_time_mean", 0.0) or 0.0) for r in rows]
    return {
        "matches": matches,
        "candidate_wins": wins,
        "win_rate": wins / matches if matches else 0.0,
        "errors": errors,
        "error_rate": errors / matches if matches else 0.0,
        "candidate_time_mean": statistics.mean(times) if times else 0.0,
        "candidate_time_p95": sorted(times)[int(0.95 * (len(times) - 1))] if times else 0.0,
        "candidate_time_max": max(times) if times else 0.0,
    }


def timed_agent(agent_fn, timings: list[float]):
    def wrapped(obs, config=None):
        start = time.perf_counter()
        try:
            return agent_fn(obs, config)
        finally:
            timings.append(time.perf_counter() - start)
    return wrapped


def run_match(candidate_ref: str, baseline_ref: str, seed: int, order: str) -> dict[str, Any]:
    candidate = _load_agent(candidate_ref)
    baseline = _load_agent(baseline_ref)
    candidate_times: list[float] = []
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    if order == "candidate_first":
        agents = [timed_agent(candidate, candidate_times), baseline]
        candidate_idx, baseline_idx = 0, 1
    else:
        agents = [baseline, timed_agent(candidate, candidate_times)]
        candidate_idx, baseline_idx = 1, 0
    env.run(agents)
    state = env.state
    return {
        "seed": seed,
        "order": order,
        "candidate_reward": state[candidate_idx].reward,
        "baseline_reward": state[baseline_idx].reward,
        "candidate_status": str(state[candidate_idx].status),
        "baseline_status": str(state[baseline_idx].status),
        "candidate_time_mean": statistics.mean(candidate_times) if candidate_times else 0.0,
        "candidate_time_max": max(candidate_times) if candidate_times else 0.0,
    }


def run_headtohead(candidate_ref: str, baseline_ref: str, seeds: int) -> dict[str, Any]:
    rows = []
    for seed in range(seeds):
        rows.append(run_match(candidate_ref, baseline_ref, seed, "candidate_first"))
        rows.append(run_match(candidate_ref, baseline_ref, seed, "candidate_second"))
    return {"summary": summarize_results(rows), "matches": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="src.ow.chadman:agent")
    parser.add_argument("--baseline", default="src.ow.chadman_baseline:agent")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_headtohead(args.candidate, args.baseline, args.seeds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate_headtohead.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run first baseline calibration smoke**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 2 --output reports/chadman_baseline_smoke.json
```

Expected: report exists, `matches=4`, no runtime errors. Since candidate and baseline are identical copies, either side may win depending on order/seed, but error rate must be `0.0`.

---

### Task 2A: Add MatthewWHuang Cinema-Mode Test Renderer

**Files:**
- Create: `tools/render_cinema_match.py`
- Test: `tests/test_cinema_match.py`

- [ ] **Step 1: Write failing cinema renderer tests**

Create `tests/test_cinema_match.py`:

```python
from pathlib import Path

from tools.render_cinema_match import render_env_json_to_cinema_html


def test_render_env_json_to_cinema_html_uses_auto_cinema(tmp_path):
    env_json = {
        "steps": [
            [
                {"observation": {"planets": [], "fleets": [], "comets": [], "angular_velocity": 0}, "reward": 0, "status": "ACTIVE", "action": []},
                {"observation": {"planets": [], "fleets": [], "comets": [], "angular_velocity": 0}, "reward": 0, "status": "ACTIVE", "action": []},
            ]
        ],
        "info": {"logs": []},
    }
    out = tmp_path / "cinema.html"

    rendered = render_env_json_to_cinema_html(env_json, out, bot_names=["candidate", "baseline"])

    html = Path(rendered).read_text(encoding="utf-8")
    assert "Orbit Wars Viewer" in html
    assert '"autoCinema": true' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_cinema_match.py -q
```

Expected: FAIL with missing `tools.render_cinema_match`.

- [ ] **Step 3: Implement renderer using cloned Matthew repo visualizer**

Create `tools/render_cinema_match.py`:

```python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make


MATTHEW_REPO = Path(__file__).resolve().parents[1] / ".slim" / "clonedeps" / "repos" / "MatthewWHuang__orbit-wars"
VISUALIZE_PATH = MATTHEW_REPO / "visualize.py"


def _load_visualize_module():
    if not VISUALIZE_PATH.exists():
        raise FileNotFoundError(f"MatthewWHuang visualizer not found at {VISUALIZE_PATH}")
    spec = importlib.util.spec_from_file_location("matthew_orbit_wars_visualize", VISUALIZE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load visualizer spec from {VISUALIZE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_env_json_to_cinema_html(env_json: dict[str, Any], output: str | Path, bot_names: list[str] | None = None) -> str:
    visualize = _load_visualize_module()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return visualize.write_html(env_json, str(out), bot_names=bot_names, auto_cinema=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Orbit Wars match and render MatthewWHuang cinema-mode HTML.")
    parser.add_argument("candidate")
    parser.add_argument("baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default="reports/cinema/chadman_match.html")
    args = parser.parse_args()

    cfg: dict[str, Any] = {"seed": args.seed}
    if args.steps is not None:
        cfg["episodeSteps"] = args.steps
    env = make("orbit_wars", configuration=cfg, debug=True)
    env.run([args.candidate, args.baseline])
    rendered = render_env_json_to_cinema_html(env.toJSON(), args.out, bot_names=[args.candidate, args.baseline])
    print(rendered)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cinema_match.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Generate first visible cinema artifact**

Run:

```bash
.venv/bin/python tools/render_cinema_match.py main.py src/ow/chadman_baseline.py --seed 0 --steps 80 --out reports/cinema/chadman_vs_baseline_seed0.html
```

Expected: prints an absolute path to an HTML file whose embedded payload has `autoCinema: true`. Share that path in the phase report so the user can open it.

---

### Task 3: Build Replay Corpus Inventory and Sample Analyzer

**Files:**
- Create: `tools/analyze_orbit_replays.py`
- Test: `tests/test_replay_analysis.py`

- [ ] **Step 1: Write failing replay parser tests**

Create `tests/test_replay_analysis.py`:

```python
from tools.analyze_orbit_replays import summarize_replay


def test_summarize_replay_extracts_teams_rewards_and_actions():
    replay = {
        "module_version": "1.29.3",
        "info": {"EpisodeId": 123, "TeamNames": ["A", "B"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {"observation": {"step": 0, "remainingOverageTime": 60, "planets": [[0, 0, 10, 10, 1, 10, 1]], "fleets": []}, "action": []},
                {"observation": {"step": 0, "remainingOverageTime": 60, "planets": [[0, 0, 10, 10, 1, 10, 1]], "fleets": []}, "action": []},
            ],
            [
                {"observation": {"step": 1, "remainingOverageTime": 59.9, "planets": [], "fleets": []}, "action": [[0, 0.5, 3]]},
                {"observation": {"step": 1, "remainingOverageTime": 59.8, "planets": [], "fleets": []}, "action": []},
            ],
        ],
    }

    summary = summarize_replay(replay)

    assert summary["episode_id"] == 123
    assert summary["teams"] == ["A", "B"]
    assert summary["rewards"] == [1, -1]
    assert summary["steps"] == 2
    assert summary["actions_by_player"] == [1, 0]
    assert summary["ships_launched_by_player"] == [3, 0]
    assert summary["min_remaining_overage_by_player"] == [59.9, 59.8]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_replay_analysis.py -q
```

Expected: FAIL with missing `tools.analyze_orbit_replays`.

- [ ] **Step 3: Implement replay analyzer**

Create `tools/analyze_orbit_replays.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_replay(replay: dict[str, Any]) -> dict[str, Any]:
    teams = list(replay.get("info", {}).get("TeamNames", []))
    n_players = len(teams) or len(replay.get("rewards", []))
    actions = [0 for _ in range(n_players)]
    ships = [0 for _ in range(n_players)]
    min_overage = [None for _ in range(n_players)]
    steps = replay.get("steps", []) or []
    for step_agents in steps:
        for pid, agent_row in enumerate(step_agents[:n_players]):
            obs = agent_row.get("observation", {}) or {}
            over = obs.get("remainingOverageTime")
            if over is not None:
                over = float(over)
                min_overage[pid] = over if min_overage[pid] is None else min(min_overage[pid], over)
            action = agent_row.get("action") or []
            if isinstance(action, list):
                for move in action:
                    if isinstance(move, list) and len(move) == 3:
                        actions[pid] += 1
                        ships[pid] += int(move[2])
    return {
        "episode_id": replay.get("info", {}).get("EpisodeId"),
        "module_version": replay.get("module_version"),
        "teams": teams,
        "rewards": replay.get("rewards", []),
        "statuses": replay.get("statuses", []),
        "steps": len(steps),
        "actions_by_player": actions,
        "ships_launched_by_player": ships,
        "min_remaining_overage_by_player": [0.0 if v is None else v for v in min_overage],
    }


def iter_replay_paths(root: Path, limit: int | None = None):
    count = 0
    for path in sorted(root.rglob("*.json")):
        yield path
        count += 1
        if limit is not None and count >= limit:
            return


def analyze_corpus(root: Path, limit: int | None = None) -> dict[str, Any]:
    rows = []
    by_date: dict[str, dict[str, Any]] = {}
    for path in iter_replay_paths(root, limit=limit):
        try:
            replay = json.loads(path.read_text())
            summary = summarize_replay(replay)
        except Exception as exc:
            summary = {"path": str(path), "error": str(exc)}
        summary["path"] = str(path)
        rows.append(summary)
        date = path.parent.name
        bucket = by_date.setdefault(date, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += path.stat().st_size
    return {"root": str(root), "count": len(rows), "by_date": by_date, "sample": rows[:200]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/chris/projects/leaderboard_replays")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze_corpus(Path(args.root), limit=args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_replay_analysis.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Generate first replay inventory**

Run:

```bash
.venv/bin/python tools/analyze_orbit_replays.py --root /home/chris/projects/leaderboard_replays --limit 2000 --output reports/replay_inventory_sample.json
```

Expected: report exists with `count=2000` unless fewer files are present.

---

### Task 4: Add Official Spec / Simulator Conformance Tests

**Files:**
- Create: `tests/test_chadman_conformance.py`

- [ ] **Step 1: Write conformance tests**

Create `tests/test_chadman_conformance.py`:

```python
import math

from kaggle_environments.envs.orbit_wars import orbit_wars as official
from src.ow import chadman


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

```

- [ ] **Step 2: Run conformance tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_conformance.py -q
```

Expected: PASS for schema/speed/combat. Any failure means stop and fix Chad conformance before planner repairs.

- [ ] **Step 3: Add explicit diagnostic note to report**

Create `reports/chadman_conformance_notes.md` containing:

```markdown
# Chadman Conformance Notes

- Planet schema: verified against official environment.
- Fleet schema: verified against official environment.
- Fleet speed: verified against official formula.
- Combat tie behavior: verified by unit tests.
- Remaining work: official-env scenario tests for swept moving-planet collisions, sun crossing, out-of-bounds, and production/arrival order.
```

---

### Task 5: Make Final Execution Commitment-Aware

**Files:**
- Modify: `src/ow/chadman.py`
- Test: `tests/test_chadman_commitments.py`

- [ ] **Step 1: Write failing commitment propagation test**

Create `tests/test_chadman_commitments.py`:

```python
from collections import namedtuple

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_commitments.py -q
```

Expected: FAIL with `TypeError: execute_actions() got an unexpected keyword argument 'planned_commitments'`.

- [ ] **Step 3: Modify `execute_actions` signature and call**

In `src/ow/chadman.py`, change:

```python
def execute_actions(world, actions, policy=None):
```

to:

```python
def execute_actions(world, actions, policy=None, planned_commitments=None):
    planned_commitments = planned_commitments or {}
```

Then change the `world.min_ships_to_own_by` call inside `execute_actions` from:

```python
planned_commitments={},
```

to:

```python
planned_commitments=planned_commitments,
```

Finally change `plan_moves_v21` from:

```python
moves = execute_actions(world, chosen, policy=policy)
```

to:

```python
moves = execute_actions(world, chosen, policy=policy, planned_commitments=planned_commitments)
```

- [ ] **Step 4: Run commitment tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_commitments.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run calibration smoke**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 4 --output reports/chadman_commitments_vs_baseline_4seeds.json
```

Expected: report exists. Candidate must have `error_rate=0.0`. Win rate may be noisy at 4 seeds.

---

### Task 6: Add Bandit Disable Ablation Switch

**Files:**
- Modify: `src/ow/chadman.py`
- Test: `tests/test_chadman_budgeting.py`

- [ ] **Step 1: Write failing bandit-disable test**

Create or extend `tests/test_chadman_budgeting.py`:

```python
import os

from src.ow import chadman


def test_bandit_disabled_env_helper(monkeypatch):
    monkeypatch.delenv("OW_DISABLE_META_BANDIT", raising=False)
    assert chadman.meta_bandit_enabled() is True

    monkeypatch.setenv("OW_DISABLE_META_BANDIT", "1")
    assert chadman.meta_bandit_enabled() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_budgeting.py -q
```

Expected: FAIL with missing `meta_bandit_enabled`.

- [ ] **Step 3: Implement helper and guard bandit calls**

In `src/ow/chadman.py`, add near utility helpers:

```python
def meta_bandit_enabled():
    return str(os.environ.get("OW_DISABLE_META_BANDIT", "0")).lower() not in ("1", "true", "yes", "on")
```

Ensure `os` is imported.

In `agent`, guard reset/observe calls:

```python
if meta_bandit_enabled():
    _META_BANDIT.reset()
```

and:

```python
if meta_bandit_enabled():
    _META_BANDIT.observe(world)
```

If Chad reads `_META_BANDIT` elsewhere for weights, keep default neutral behavior when disabled by making the bandit observe/update no-op only; do not remove policy construction.

- [ ] **Step 4: Run tests and ablation smoke**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_budgeting.py -q
OW_DISABLE_META_BANDIT=1 .venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 4 --output reports/chadman_bandit_disabled_vs_baseline_4seeds.json
```

Expected: tests pass, report exists, `error_rate=0.0`.

---

### Task 7: Add Remaining-Overage-Aware Deadline Policy

**Files:**
- Modify: `src/ow/chadman.py`
- Test: `tests/test_chadman_budgeting.py`

- [ ] **Step 1: Write failing soft-budget helper test**

Extend `tests/test_chadman_budgeting.py`:

```python
from src.ow import chadman


def test_compute_soft_budget_tightens_when_overage_low():
    normal = chadman.compute_soft_budget(act_timeout=1.0, remaining_overage=60.0)
    tight = chadman.compute_soft_budget(act_timeout=1.0, remaining_overage=2.0)

    assert 0.55 <= normal <= 0.90
    assert tight < normal
    assert tight >= 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_budgeting.py -q
```

Expected: FAIL with missing `compute_soft_budget`.

- [ ] **Step 3: Implement budget helper and use it in agent**

In `src/ow/chadman.py`, add:

```python
def compute_soft_budget(act_timeout=1.0, remaining_overage=None):
    base = min(SOFT_ACT_DEADLINE + 0.05, max(0.60, float(act_timeout) * 0.88))
    if remaining_overage is None:
        return base
    remaining = max(0.0, float(remaining_overage))
    if remaining < 1.0:
        return min(base, 0.08)
    if remaining < 3.0:
        return min(base, 0.18)
    if remaining < 8.0:
        return min(base, 0.35)
    return base
```

Then in `agent`, replace:

```python
soft_budget = min(SOFT_ACT_DEADLINE + 0.05, max(0.60, act_timeout * 0.88))
```

with:

```python
remaining_overage = _read(obs, "remainingOverageTime", None)
soft_budget = compute_soft_budget(act_timeout=act_timeout, remaining_overage=remaining_overage)
```

- [ ] **Step 4: Run tests and timing calibration**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_budgeting.py -q
.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 8 --output reports/chadman_overage_vs_baseline_8seeds.json
```

Expected: tests pass, report exists, `error_rate=0.0`, candidate timing max does not exceed baseline smoke timing by more than 20%.

---

### Task 8: Run First Broad Diagnosis and Decide ML Readiness

**Files:**
- Create: `reports/chadman_phase1_diagnosis.md`
- Read/Use: `reports/chadman_*_vs_baseline_*.json`, `reports/replay_inventory_sample.json`

- [ ] **Step 1: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass. If old greedy-policy tests fail only because production `main.py` now delegates to Chadman, retarget those tests to `src.ow.agent_core.safe_agent` or `src.ow.policy.choose_moves` without weakening Chadman tests.

- [ ] **Step 2: Run 16-match Chadman calibration**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 8 --output reports/chadman_phase1_vs_baseline_8seeds.json
```

Expected: `matches=16`, `error_rate=0.0`.

- [ ] **Step 3: Run local ladder smoke**

Run:

```bash
.venv/bin/python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 32 --mode 2 --output reports/chadman_phase1_ladder_2p_32.json
.venv/bin/python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 32 --mode 4 --output reports/chadman_phase1_ladder_4p_32.json
```

Expected: reports exist with `error_rate=0.0`.

- [ ] **Step 4: Write diagnosis report**

Create `reports/chadman_phase1_diagnosis.md`:

```markdown
# Chadman Phase 1 Diagnosis

## Verification

- Unit tests: record exact pytest command and result.
- Chadman vs frozen baseline: summarize report path, matches, win rate, error rate, timing.
- Local ladder 2P/4P: summarize report paths, win rates, avg ranks, error rates, p99 timing.

## Replay Corpus

- Summarize `reports/replay_inventory_sample.json` count, dates, teams sampled, and module versions.

## Repairs Tested

- Commitment-aware final execution: summarize before/after test and calibration result.
- Bandit ablation: summarize enabled/disabled result.
- Overage-aware deadline: summarize timing result.

## Decision

- If candidate beats frozen Chad baseline and has zero errors, proceed to deeper replay analysis and supervised reranker plan.
- If candidate does not beat frozen Chad baseline, continue diagnosis before ML.
```

Replace the bullet text with actual observed values before marking this task complete.

---

## Self-Review Checklist

- Spec coverage:
  - Chad transplant: Task 1.
  - Baseline calibration: Task 2 and Task 8.
  - Replay analysis: Task 3 and Task 8.
  - Official conformance: Task 4.
  - Commitment context: Task 5.
  - Bandit ablation: Task 6.
  - Overage budgeting: Task 7.
  - ML deferred until repairs: Task 8 decision gate.
- Placeholder scan: no placeholder tokens or undefined task references.
- Type consistency: public functions planned are `summarize_results`, `run_headtohead`, `summarize_replay`, `analyze_corpus`, `meta_bandit_enabled`, and `compute_soft_budget`.
- Execution rule: use TDD for every production behavior change and run calibration after each Chadman change.
