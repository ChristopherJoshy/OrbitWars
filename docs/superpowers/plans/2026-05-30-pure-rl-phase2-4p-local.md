# Pure RL Phase 2 4P Local Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not commit changes unless the user explicitly asks for a commit.

**Goal:** Extend the local pure-RL self-play trainer from 2-player-only episodes to 4-player pure self-play episodes so Phase 2 can train for the 6-hour budget after Phase 1.

**Architecture:** Keep the current NumPy PPO/IPPO fallback: one shared `LinearActorCritic` samples actions for every player, and every player decision is recorded with player-relative rewards. Generalize the trainer mode, player count, terminal outcome calculation, metrics, and viewer labels while preserving existing 2P behavior and the local-only Pygame viewer.

**Tech Stack:** Python, NumPy, kaggle-environments Orbit Wars, pytest, optional Pygame, JSONL/CSV metrics.

---

## File Structure

- Modify: `tools/train_rl_phase1.py` — add `--mode 4p`, player-count helpers, rank/outcome helpers, 4-agent environment runs, and 4P metrics.
- Modify: `src/ow/rl/pygame_viewer.py` — make player identity/status helpers handle all visible players in 4P while keeping player 0 labeled `OUR AGENT`.
- Modify: `tests/test_rl_train_phase1.py` — TDD coverage for 4P parsing, player counts, rank rewards, and 4P metric payloads.
- Modify: `tests/test_rl_viewer.py` — TDD coverage for 4P identity labels and 4-player summaries.

## Task 1: Mode and Player Count Helpers

**Files:**
- Modify: `tools/train_rl_phase1.py`
- Modify: `tests/test_rl_train_phase1.py`

- [ ] **Step 1: Write failing tests for 4P mode parsing and player count**

Add to `tests/test_rl_train_phase1.py`:

```python
from tools.train_rl_phase1 import mode_player_count


def test_parse_args_accepts_4p_mode():
    args = parse_args(["--mode", "4p", "--no-viewer"])
    assert args.mode == "4p"
    assert mode_player_count(args.mode) == 4


def test_mode_player_count_keeps_2p_behavior():
    assert mode_player_count("2p") == 2
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py::test_parse_args_accepts_4p_mode tests/test_rl_train_phase1.py::test_mode_player_count_keeps_2p_behavior -q`

Expected: FAIL because `parse_args` rejects `4p` or `mode_player_count` is missing.

- [ ] **Step 3: Implement minimal helper and parser change**

In `tools/train_rl_phase1.py`, change:

```python
p.add_argument("--mode", choices=["2p"], default="2p")
```

to:

```python
p.add_argument("--mode", choices=["2p", "4p"], default="2p")
```

Add near `build_run_dir`:

```python
def mode_player_count(mode: str) -> int:
    if mode == "2p":
        return 2
    if mode == "4p":
        return 4
    raise ValueError(f"unsupported mode: {mode}")
```

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py -q`

Expected: PASS.

## Task 2: Player-Relative 4P Outcome Rewards

**Files:**
- Modify: `tools/train_rl_phase1.py`
- Modify: `tests/test_rl_train_phase1.py`

- [ ] **Step 1: Write failing tests for rank-based fallback rewards**

Add to `tests/test_rl_train_phase1.py`:

```python
from types import SimpleNamespace
from tools.train_rl_phase1 import player_outcome_rewards


def _state(player, reward, planets):
    return SimpleNamespace(reward=reward, observation={"player": player, "planets": planets, "fleets": []})


def test_player_outcome_rewards_uses_asset_rank_for_4p_equal_env_rewards():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 40, 5],
        [1, 1, 30.0, 10.0, 2.0, 30, 4],
        [2, 2, 50.0, 10.0, 2.0, 20, 3],
        [3, 3, 70.0, 10.0, 2.0, 10, 2],
    ]
    state = [_state(pid, 1.0, planets) for pid in range(4)]
    rewards, ranks = player_outcome_rewards(state, n_players=4)
    assert rewards == [1.5, 0.5, -0.5, -1.5]
    assert ranks == [1, 2, 3, 4]


def test_player_outcome_rewards_preserves_distinct_env_rewards_for_2p():
    state = [_state(0, 3.0, []), _state(1, 1.0, [])]
    rewards, ranks = player_outcome_rewards(state, n_players=2)
    assert rewards == [3.0, 1.0]
    assert ranks == [1, 2]
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py::test_player_outcome_rewards_uses_asset_rank_for_4p_equal_env_rewards tests/test_rl_train_phase1.py::test_player_outcome_rewards_preserves_distinct_env_rewards_for_2p -q`

Expected: FAIL because `player_outcome_rewards` is missing.

- [ ] **Step 3: Implement asset score and outcome helpers**

Add to `tools/train_rl_phase1.py`:

```python
def _asset_score(observation: dict, player_id: int) -> float:
    score = 0.0
    for planet in observation.get("planets", []) or []:
        if len(planet) > 6 and int(planet[1]) == int(player_id):
            score += float(planet[5]) + 8.0 * float(planet[6]) + 25.0
    for fleet in observation.get("fleets", []) or []:
        if len(fleet) > 6 and int(fleet[1]) == int(player_id):
            score += 0.5 * float(fleet[6])
    return score


def _dense_ranks(scores: list[float]) -> list[int]:
    ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(ordered, start=1):
        ranks[idx] = rank
    return ranks


def player_outcome_rewards(state: list, n_players: int) -> tuple[list[float], list[int]]:
    env_rewards = [float(getattr(state[i], "reward", 0.0) or 0.0) for i in range(int(n_players))]
    ranks = _dense_ranks(env_rewards)
    if len(set(env_rewards)) > 1:
        return env_rewards, ranks
    obs = getattr(state[0], "observation", {}) or {}
    scores = [_asset_score(obs, pid) for pid in range(int(n_players))]
    ranks = _dense_ranks(scores)
    center = (int(n_players) + 1) / 2.0
    rewards = [float(center - rank) for rank in ranks]
    return rewards, ranks
```

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py -q`

Expected: PASS.

## Task 3: Generalize Episode Finalization and Main Loop to N Players

**Files:**
- Modify: `tools/train_rl_phase1.py`
- Modify: `tests/test_rl_train_phase1.py`

- [ ] **Step 1: Write failing tests for player-relative N-player returns**

Add to `tests/test_rl_train_phase1.py`:

```python
def test_finalize_episode_records_accepts_outcome_reward_list_for_4p():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    for pid in range(4):
        rec.record(pid, {"player": pid}, feats, 0, -0.1, 0.0, mistake_penalty=-0.25)
    batch = finalize_episode_records(rec.decisions, outcome_rewards=[1.5, 0.5, -0.5, -1.5])
    assert np.allclose(batch["returns"], np.array([1.25, 0.25, -0.75, -1.75]))
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py::test_finalize_episode_records_accepts_outcome_reward_list_for_4p -q`

Expected: FAIL because `finalize_episode_records` does not accept `outcome_rewards`.

- [ ] **Step 3: Implement backwards-compatible finalization**

Change `finalize_episode_records` signature to:

```python
def finalize_episode_records(decisions: list[dict], r0: float | None = None, r1: float | None = None, outcome_rewards: list[float] | None = None) -> dict:
```

Inside the loop, replace the terminal reward line with:

```python
if outcome_rewards is None:
    if r0 is None or r1 is None:
        raise ValueError("r0/r1 or outcome_rewards required")
    terminal_reward = float(r0 - r1) if player_id == 0 else float(r1 - r0)
else:
    terminal_reward = float(outcome_rewards[player_id])
```

- [ ] **Step 4: Generalize the main loop**

In `main`, compute `n_players = mode_player_count(args.mode)` after parsing. Change the environment run from:

```python
env.run([mk_agent(0), mk_agent(1)])
```

to:

```python
env.run([mk_agent(pid) for pid in range(n_players)])
```

Replace `r0/r1/won/ep_reward` with:

```python
outcome_rewards, ranks = player_outcome_rewards(state, n_players=n_players)
won = ranks[0] == 1
ep_reward = outcome_rewards[0]
ep_batch = finalize_episode_records(recorder.decisions, outcome_rewards=outcome_rewards)
```

Add metrics fields in both `viewer.draw` and `writer.write` payloads:

```python
"mode": args.mode,
"n_players": n_players,
"player0_rank": ranks[0],
"player0_outcome_reward": outcome_rewards[0],
```

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py -q`

Expected: PASS with existing 2P tests unchanged.

## Task 4: 4P Viewer Identity and Summaries

**Files:**
- Modify: `src/ow/rl/pygame_viewer.py`
- Modify: `tests/test_rl_viewer.py`

- [ ] **Step 1: Write failing tests for 4-player labels**

Add to `tests/test_rl_viewer.py`:

```python
def test_agent_identity_lines_can_show_four_players_with_our_agent_first():
    lines = agent_identity_lines(our_player=0, n_players=4)
    assert lines[0]["label"] == "OUR AGENT / Player 0"
    assert [line["color"] for line in lines] == PLAYER_COLORS[:4]
    assert lines[1]["label"] == "SELF-PLAY OPPONENT / Player 1"
    assert lines[3]["label"] == "SELF-PLAY OPPONENT / Player 3"
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_rl_viewer.py::test_agent_identity_lines_can_show_four_players_with_our_agent_first -q`

Expected: FAIL because `agent_identity_lines` has no `n_players` argument.

- [ ] **Step 3: Implement 4P identity helper**

Change `agent_identity_lines` signature to:

```python
def agent_identity_lines(our_player: int = 0, n_players: int = 2) -> list[dict[str, object]]:
```

Return one label per player:

```python
out = []
for pid in range(max(2, int(n_players))):
    label = f"OUR AGENT / Player {pid}" if pid == int(our_player) else f"SELF-PLAY OPPONENT / Player {pid}"
    out.append({"label": label, "color": PLAYER_COLORS[pid % len(PLAYER_COLORS)]})
return out
```

In `_draw_status_panels`, set `n_players = int(metrics.get("n_players", 2))` and pass it to `agent_identity_lines` and `summarize_observation`.

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_viewer.py -q`

Expected: PASS.

## Task 5: 4P Smoke Verification and Phase 2 Readiness

**Files:**
- Modify: `tools/train_rl_phase1.py` only if smoke reveals a tested defect.
- Test command outputs under `reports/rl_phase2_4p_smoke/`.

- [ ] **Step 1: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_rl_action_space.py tests/test_rl_policy.py tests/test_rl_rewards_metrics.py tests/test_rl_viewer.py tests/test_rl_train_phase1.py -q`

Expected: PASS.

- [ ] **Step 2: Run 4P no-viewer smoke**

Run: `rm -rf reports/rl_phase2_4p_smoke && .venv/bin/python tools/train_rl_phase1.py --mode 4p --episodes 2 --updates 1 --no-viewer --run-dir reports/rl_phase2_4p_smoke`

Expected: command exits 0, `reports/rl_phase2_4p_smoke/metrics.jsonl` has 2 rows, `checkpoints/policy_update_1.npz` exists, and each row includes `mode="4p"`, `n_players=4`, and `player0_rank`.

- [ ] **Step 3: Run 4P windowed viewer smoke**

Run: `rm -rf reports/rl_phase2_4p_viewer_smoke && .venv/bin/python tools/train_rl_phase1.py --mode 4p --episodes 1 --updates 1 --viewer --windowed --resizable-window --minimal-viewer --run-dir reports/rl_phase2_4p_viewer_smoke`

Expected: command exits 0, `metrics.jsonl` has 1 row, checkpoint exists, and the viewer does not interrupt training if minimized/resized.

- [ ] **Step 4: Do not stop active Phase 1 unless necessary**

Check the active Phase 1 PID in `reports/rl_phase1_window_controls_live_run/pid.txt`. If it is alive and no test failure indicates a correctness bug in already-running 2P training, leave it running until the 6-hour target or until the user asks to stop.

## Self-Review

- Spec coverage: the plan covers Phase 2 local 4P pure self-play, mode-specific metrics, viewer support, and smoke verification. It does not implement Phase 3 merged training; that will require a separate plan after Phase 2 is verified.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: helper names are consistent across tasks: `mode_player_count`, `player_outcome_rewards`, `finalize_episode_records(..., outcome_rewards=...)`, and `agent_identity_lines(..., n_players=...)`.
