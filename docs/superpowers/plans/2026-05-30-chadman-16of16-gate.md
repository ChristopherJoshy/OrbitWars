# Chadman 16/16 Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get `src.ow.chadman:agent` to 16/16 wins against frozen `src.ow.chadman_baseline:agent` over seeds 0..7 in both seat orders before any RL/self-play training starts.

**Architecture:** First make the head-to-head gate faster and more precise by adding parallel/selected-match support to the existing evaluator. Then apply one root-cause-backed Chadman fix: stop passing beam-level `planned_commitments` into final action repair from `plan_moves_v21`, because diagnostics show losses persist even with tactical feature-family ablations and the candidate differs from baseline at this repair boundary. Verify with tests, selected loss seeds, and the full 16-match gate.

**Tech Stack:** Python, pytest, `kaggle_environments`, existing Chadman agent, existing `tools/evaluate_headtohead.py`.

**Commit policy:** Do not commit. The user has not requested commits.

**Post-gate iteration-1 RL constraint update:** After this 16/16 gate passes, local RL iteration 1 uses an 18-hour training budget split into three target phases: 6 hours 2P-only, 6 hours 4P-only, and 6 hours merged 2P+4P self-play. A phase may be extended if the model has not started winning all games or is still clearly improving. A live Pygame game viewer backed by NumPy state is mandatory for local training only. Kaggle training uses logs/JSONL/CSV/checkpoint summaries only, with no Pygame, GUI, browser visualization, or live renderer.

---

## File Structure

- Modify `tools/evaluate_headtohead.py`
  - Add selected match parsing and `--jobs` parallel execution.
  - Keep the existing default behavior unchanged when `--matches` is not supplied.
- Modify `tests/test_evaluate_headtohead.py`
  - Add tests for selected match parsing and dependency-injected selected runs.
- Modify `tests/test_chadman_commitments.py`
  - Keep existing direct `execute_actions` tests.
  - Add a `plan_moves_v21` regression test proving beam commitments are not passed into final repair.
- Modify `src/ow/chadman.py`
  - Change only the `plan_moves_v21` call to `execute_actions` so final repair uses baseline-style local repair context.
- Write reports under `reports/`
  - `reports/chadman_commitment_boundary_selected_220steps.json`
  - `reports/chadman_16of16_gate_full.json`

---

### Task 1: Add Parallel Selected-Match Evaluation

**Files:**
- Modify: `tools/evaluate_headtohead.py`
- Modify: `tests/test_evaluate_headtohead.py`

- [ ] **Step 1: Write failing evaluator tests**

Append these tests to `tests/test_evaluate_headtohead.py`:

```python
def test_parse_match_specs_accepts_seed_order_pairs():
    from tools import evaluate_headtohead

    specs = evaluate_headtohead.parse_match_specs("2:candidate_first,4:candidate_second,7:candidate_first")

    assert specs == [(2, "candidate_first"), (4, "candidate_second"), (7, "candidate_first")]


def test_parse_match_specs_rejects_bad_order():
    from tools import evaluate_headtohead

    import pytest

    with pytest.raises(ValueError, match="order must be"):
        evaluate_headtohead.parse_match_specs("2:middle")


def test_run_selected_matches_uses_injected_runner_and_summary():
    from tools import evaluate_headtohead


    calls = []

    def fake_runner(candidate_ref, baseline_ref, seed, order, episode_steps=80):
        calls.append((candidate_ref, baseline_ref, seed, order, episode_steps))
        return {
            "seed": seed,
            "order": order,
            "candidate_reward": 1,
            "baseline_reward": -1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.01,
            "candidate_time_p95": 0.02,
            "candidate_time_max": 0.03,
        }

    report = evaluate_headtohead.run_selected_matches(
        "candidate",
        "baseline",
        [(2, "candidate_first"), (4, "candidate_second")],
        episode_steps=220,
        jobs=4,
        runner=fake_runner,
    )

    assert calls == [
        ("candidate", "baseline", 2, "candidate_first", 220),
        ("candidate", "baseline", 4, "candidate_second", 220),
    ]
    assert report["episode_steps"] == 220
    assert report["summary"]["matches"] == 2
    assert report["summary"]["candidate_wins"] == 2
```

- [ ] **Step 2: Run evaluator tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate_headtohead.py -q
```

Expected: FAIL because `parse_match_specs` and `run_selected_matches` do not exist yet.

- [ ] **Step 3: Implement selected/parallel helpers**

In `tools/evaluate_headtohead.py`, add these imports near the existing imports:

```python
from concurrent.futures import ProcessPoolExecutor
```

Add these helpers after `run_match(...)` and before `run_headtohead(...)`:

```python
def parse_match_specs(raw: str) -> list[tuple[int, str]]:
    specs: list[tuple[int, str]] = []
    if not raw.strip():
        return specs
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            seed_text, order = item.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"match spec must be seed:order, got {item!r}") from exc
        order = order.strip()
        if order not in {"candidate_first", "candidate_second"}:
            raise ValueError(f"order must be candidate_first or candidate_second, got {order!r}")
        specs.append((int(seed_text), order))
    return specs


def _run_match_from_tuple(args: tuple[str, str, int, str, int | None]) -> dict[str, Any]:
    candidate_ref, baseline_ref, seed, order, episode_steps = args
    return run_match(candidate_ref, baseline_ref, seed, order, episode_steps=episode_steps)


def run_selected_matches(
    candidate_ref: str,
    baseline_ref: str,
    specs: list[tuple[int, str]],
    episode_steps: int | None = 80,
    jobs: int = 1,
    runner=run_match,
) -> dict[str, Any]:
    jobs = max(1, int(jobs))
    if jobs == 1 or runner is not run_match:
        rows = [runner(candidate_ref, baseline_ref, seed, order, episode_steps=episode_steps) for seed, order in specs]
    else:
        tasks = [(candidate_ref, baseline_ref, seed, order, episode_steps) for seed, order in specs]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_run_match_from_tuple, tasks))
    return {"episode_steps": episode_steps, "summary": summarize_results(rows), "matches": rows}
```

Replace `run_headtohead(...)` with this implementation:

```python
def run_headtohead(candidate_ref: str, baseline_ref: str, seeds: int, episode_steps: int | None = 80, jobs: int = 1) -> dict[str, Any]:
    specs: list[tuple[int, str]] = []
    for seed in range(seeds):
        specs.append((seed, "candidate_first"))
        specs.append((seed, "candidate_second"))
    return run_selected_matches(candidate_ref, baseline_ref, specs, episode_steps=episode_steps, jobs=jobs)
```

Add these CLI arguments in `main()` after `--steps`:

```python
    parser.add_argument("--jobs", type=int, default=1, help="Parallel match workers; use >1 for independent seeded matches")
    parser.add_argument(
        "--matches",
        default="",
        help="Optional comma-separated seed:order list, e.g. '2:candidate_first,4:candidate_second'",
    )
```

Replace the `report = ...` line in `main()` with:

```python
    if args.matches:
        report = run_selected_matches(
            args.candidate,
            args.baseline,
            parse_match_specs(args.matches),
            episode_steps=episode_steps,
            jobs=args.jobs,
        )
    else:
        report = run_headtohead(args.candidate, args.baseline, args.seeds, episode_steps=episode_steps, jobs=args.jobs)
```

- [ ] **Step 4: Run evaluator tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate_headtohead.py -q
```

Expected: PASS.

---

### Task 2: Add Commitment Boundary Regression Test

**Files:**
- Modify: `tests/test_chadman_commitments.py`

- [ ] **Step 1: Write failing `plan_moves_v21` boundary test**

Append this test to `tests/test_chadman_commitments.py`:

```python
def test_plan_moves_v21_does_not_pass_beam_commitments_to_final_repair(monkeypatch):
    world = FakeWorld()
    captured = {}

    def fake_rollout_beam_select(world, policy, modes, planned_commitments, source_inventory_left, deadline=None):
        planned_commitments[2].append((3, 0, 100))
        return [
            {
                "src_id": 1,
                "target_id": 2,
                "effective_send": 6,
                "send_cap": 6,
                "angle": 0.0,
                "turns": 3,
                "mission": "capture",
            }
        ]

    def fake_execute_actions(world_arg, actions, policy=None, planned_commitments=None):
        captured["planned_commitments"] = planned_commitments
        return [[1, 0.0, 6]]

    monkeypatch.setattr(chadman, "adaptive_horizon", lambda _world: 8)
    monkeypatch.setattr(chadman, "build_modes", lambda _world: {"is_finishing": False})
    monkeypatch.setattr(chadman, "build_policy_state", lambda _world, deadline=None: {"attack_budget": {1: 50}})
    monkeypatch.setattr(chadman, "rollout_beam_select", fake_rollout_beam_select)
    monkeypatch.setattr(chadman, "execute_actions", fake_execute_actions)

    moves = chadman.plan_moves_v21(world, deadline=None)

    assert moves == [[1, 0.0, 6]]
    assert captured["planned_commitments"] is None
```

- [ ] **Step 2: Run commitment test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_commitments.py::test_plan_moves_v21_does_not_pass_beam_commitments_to_final_repair -q
```

Expected: FAIL because current `plan_moves_v21` passes the live `defaultdict(list)` into `execute_actions`.

---

### Task 3: Implement Minimal Commitment Boundary Fix

**Files:**
- Modify: `src/ow/chadman.py:5830`
- Test: `tests/test_chadman_commitments.py`

- [ ] **Step 1: Change only the final repair call**

In `src/ow/chadman.py`, replace:

```python
    moves = execute_actions(world, chosen, policy=policy, planned_commitments=planned_commitments)
```

with:

```python
    moves = execute_actions(world, chosen, policy=policy)
```

Do not change `execute_actions(...)` itself in this task. Direct callers may still pass explicit `planned_commitments`; only `plan_moves_v21` should stop sending beam-level branch commitments into final repair.

- [ ] **Step 2: Run commitment tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_chadman_commitments.py -q
```

Expected: PASS. The existing direct `execute_actions` tests should still pass, and the new `plan_moves_v21` boundary test should pass.

---

### Task 4: Verify Targeted Loss Seeds in Parallel

**Files:**
- Read/write report: `reports/chadman_commitment_boundary_selected_220steps.json`

- [ ] **Step 1: Run selected known-loss seeds with parallel jobs**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py \
  --candidate src.ow.chadman:agent \
  --baseline src.ow.chadman_baseline:agent \
  --steps 220 \
  --jobs 4 \
  --matches '2:candidate_first,4:candidate_second,5:candidate_first,7:candidate_first' \
  --output reports/chadman_commitment_boundary_selected_220steps.json
```

Expected: command completes with `error_rate == 0.0`. The target is fewer than 4 losses on these four known-loss rows. If all four still lose, stop implementation and return to systematic debugging before applying another production change.

- [ ] **Step 2: Inspect the report summary**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('reports/chadman_commitment_boundary_selected_220steps.json')
data = json.loads(path.read_text())
print(json.dumps(data['summary'], indent=2, sort_keys=True))
for row in data['matches']:
    print(row['seed'], row['order'], row['candidate_reward'], row['baseline_reward'], row['candidate_status'], row['baseline_status'])
PY
```

Expected: printed summary and four rows. Record whether the commitment boundary fix improved the selected loss seeds.

---

### Task 5: Run Safety Tests in Parallel Batches

**Files:**
- Test-only verification

- [ ] **Step 1: Launch independent test batches concurrently**

Run these independent commands in parallel tool calls, not sequentially:

```bash
.venv/bin/python -m pytest tests/test_chadman_commitments.py tests/test_chadman_budgeting.py -q
```

```bash
.venv/bin/python -m pytest tests/test_chadman_tactical_features.py tests/test_chadman_feature_probes.py tests/test_feature_probe_runner.py -q
```

```bash
.venv/bin/python -m pytest tests/test_chadman_conformance.py tests/test_chadman_transplant.py -q
```

```bash
.venv/bin/python -m pytest tests/test_evaluate_headtohead.py tests/test_proof_gate.py -q
```

Expected: all batches PASS. If any batch fails, fix that failure before running the full gate.

---

### Task 6: Run the Full Local 16/16 Gate

**Files:**
- Write report: `reports/chadman_16of16_gate_full.json`

- [ ] **Step 1: Run full-env gate with parallel jobs**

Run:

```bash
.venv/bin/python tools/evaluate_headtohead.py \
  --candidate src.ow.chadman:agent \
  --baseline src.ow.chadman_baseline:agent \
  --seeds 8 \
  --steps 0 \
  --jobs 4 \
  --output reports/chadman_16of16_gate_full.json
```

Expected: command completes with `error_rate == 0.0`.

- [ ] **Step 2: Check gate criteria**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('reports/chadman_16of16_gate_full.json')
data = json.loads(path.read_text())
summary = data['summary']
print(json.dumps(summary, indent=2, sort_keys=True))
losses = [row for row in data['matches'] if row['candidate_reward'] <= row['baseline_reward']]
print('non_wins:', [(row['seed'], row['order'], row['candidate_reward'], row['baseline_reward']) for row in losses])
if summary['candidate_wins'] != 16 or summary['candidate_losses'] != 0 or summary['ties'] != 0 or summary['errors'] != 0:
    raise SystemExit(1)
PY
```

Expected: exits `0` only if the gate is exactly 16 wins, 0 losses, 0 ties, 0 errors. If this fails, do not start RL; continue root-cause debugging using the remaining non-win rows.

---

### Task 7: Update Readiness Report

**Files:**
- Create or modify: `reports/chadman_16of16_readiness.md`

- [ ] **Step 1: Write the readiness report**

Create `reports/chadman_16of16_readiness.md` with this structure filled from the actual reports:

```markdown
# Chadman 16/16 Readiness Report

## Gate definition

- Candidate: `src.ow.chadman:agent`
- Baseline: `src.ow.chadman_baseline:agent`
- Seeds: 0..7
- Orders: candidate_first and candidate_second
- Final run: full environment (`--steps 0`)

## Changes in this iteration

- Added selected/parallel head-to-head evaluation support.
- Added a regression test for the `plan_moves_v21` to `execute_actions` commitment boundary.
- Stopped passing beam-level `planned_commitments` into final repair from `plan_moves_v21`.

## Selected 220-step diagnostic

- Report: `reports/chadman_commitment_boundary_selected_220steps.json`
- Summary: paste the JSON summary here.
- Non-wins: list seed/order rows here.

## Full 16/16 gate

- Report: `reports/chadman_16of16_gate_full.json`
- Summary: paste the JSON summary here.
- Non-wins: list seed/order rows here.

## Decision

- If 16/16 passed: RL implementation planning may proceed.
- If 16/16 failed: RL is blocked; continue systematic debugging on listed non-win rows.
```

Replace the instruction text with actual summary values. Do not leave prose like “paste here” in the final report.

- [ ] **Step 2: Verify report has no placeholders**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
text = Path('reports/chadman_16of16_readiness.md').read_text()
bad = ['paste here', 'list seed/order rows here', 'TBD', 'TODO']
found = [item for item in bad if item in text]
if found:
    raise SystemExit(f'placeholder text remains: {found}')
print('readiness report placeholders: none')
PY
```

Expected: `readiness report placeholders: none`.

---

## Plan Self-Review

- Spec coverage: This plan implements the immediate Phase A gate required before RL. It does not implement PPO/JAX/viewer yet because the approved spec blocks RL until 16/16 passes.
- User constraints: Features are not removed; baseline is not edited; no Kaggle submission; no commits; independent tests/calibrations are run in parallel.
- Placeholder scan: No task uses unspecified file paths or vague test instructions. Task 7 includes placeholder-like template text only inside a required replacement step and has a verification command to reject leaving it in the final report.
- Type consistency: `parse_match_specs`, `run_selected_matches`, and `run_headtohead(..., jobs=...)` signatures are consistent across tests, implementation, and CLI usage.
