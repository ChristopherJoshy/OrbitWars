# Pure RL Phase 1 Local Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and start a local pure-RL 2P self-play training loop with mandatory original-style Pygame visualization that displays iteration/update count and win statistics.

**Architecture:** Add a small, testable NumPy RL package under `src/ow/rl/`. The first trainer uses Orbit Wars via `kaggle_environments`, generates legal source-target action candidates from observations, trains a compact NumPy actor-critic/PPO-style policy with clipped updates and dense rewards, writes metrics/checkpoints, and streams sampled episodes to an original Orbit-Wars-style Pygame viewer. Chadman remains an evaluation opponent/diagnostic only, not part of the learned policy.

**Tech Stack:** Python, NumPy, kaggle-environments, pytest, optional Pygame for local viewer, JSONL/CSV metrics.

---

## File Structure

- Modify: `requirements.txt` — add `pygame>=2.5` for local training visualization.
- Create: `src/ow/rl/__init__.py` — package exports for RL modules.
- Create: `src/ow/rl/action_space.py` — legal candidate action construction and angle/target helpers.
- Create: `src/ow/rl/policy.py` — NumPy linear actor-critic policy, softmax sampling, clipped policy update, checkpoint save/load.
- Create: `src/ow/rl/rewards.py` — dense reward components from consecutive Orbit Wars observations.
- Create: `src/ow/rl/metrics.py` — JSONL/CSV metric writer and rolling win-rate tracker.
- Create: `src/ow/rl/pygame_viewer.py` — local viewer drawing planets/fleets and overlaying iteration, episode, wins/losses/win-rate.
- Create: `tools/train_rl_phase1.py` — CLI entrypoint for 2P self-play training and smoke runs.
- Create: `tests/test_rl_action_space.py` — action generation tests.
- Create: `tests/test_rl_policy.py` — policy sampling/update/checkpoint tests.
- Create: `tests/test_rl_rewards_metrics.py` — reward shaping and metrics tests.
- Create: `tests/test_rl_viewer.py` — viewer overlay/state tests with import guard.

## Viewer Redesign Addendum: Original-Style Local Pygame Viewer

The first Pygame viewer was too sparse to understand. Stop any active training before changing it. Replace the simple circles-only board with a local-only renderer inspired by `.slim/clonedeps/repos/MatthewWHuang__orbit-wars/visualize.py`:

- Dark navy board and original player colors (`#3aa6ff`, `#ff5a5a`, `#3ad17a`, `#ffd23a`) plus neutral gray.
- Glowing central sun and orbit boundary circle.
- Planets rendered as scaled circles with centered ship counts and production labels.
- Fleets rendered as triangular arrows with ship counts.
- Training overlay/status panel showing update, episode, wins, losses, recent win rate, reward, checkpoint/status where available.
- Bottom mini-panels or text summary for player ships, production, and planets.
- Viewer must draw only real latest env observations captured during training; no synthetic board state.
- The viewer top panel must label player 0 as `OUR AGENT` with the same color used on the board, so the user can identify which color is ours.
- The trainer uses a win-stay seed curriculum: player-0 win -> next deterministic random seed; player-0 loss -> repeat same seed.
- Default viewer mode is minimal for older screens: compact top HUD, no cluttered side panels, fullscreen support, and burn-in protection that shifts static HUD pixels every 30 minutes by default.
- Viewer status includes seed, update, episode, wins/losses, recent win rate, reward, sun-loss ships, wasted ships, invalid actions, repeated-seed count, and checkpoint/update id when available.

Required TDD before restart:

- `tests/test_rl_viewer.py` must assert original colors/constants are exposed.
- `tests/test_rl_viewer.py` must assert `TrainingOverlay.lines` includes readable labels for update, episode, wins, losses, win rate, and reward.
- `tests/test_rl_viewer.py` must assert `summarize_observation(obs)` computes per-player ships, production, planet counts, and fleet ships from a real observation-shaped dict.
- Smoke run with `--viewer` must complete before restarting a longer run.

## Task 1: Action Candidate Builder

**Files:**
- Create: `src/ow/rl/action_space.py`
- Create: `tests/test_rl_action_space.py`

- [ ] **Step 1: Write failing tests for candidate generation**

```python
from src.ow.rl.action_space import build_action_candidates, angle_to_target


def test_build_action_candidates_includes_noop_and_owned_source_targets():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 30, 5],
            [1, -1, 20.0, 10.0, 2.0, 8, 3],
            [2, 1, 10.0, 20.0, 2.0, 12, 4],
        ],
    }
    candidates = build_action_candidates(obs, max_targets_per_source=4)
    assert candidates[0]["kind"] == "noop"
    launches = [c for c in candidates if c["kind"] == "launch"]
    assert {c["target_id"] for c in launches} == {1, 2}
    assert all(c["source_id"] == 0 for c in launches)
    assert all(c["ships"] >= 1 for c in launches)


def test_angle_to_target_is_deterministic():
    assert angle_to_target((10.0, 10.0), (20.0, 10.0)) == 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_rl_action_space.py -q`

Expected: FAIL with import error for missing `src.ow.rl.action_space`.

- [ ] **Step 3: Implement minimal action builder**

```python
import math


PLANET_ID = 0
PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_SHIPS = 5


def angle_to_target(src_xy, dst_xy):
    sx, sy = src_xy
    dx, dy = dst_xy
    return float(math.atan2(dy - sy, dx - sx))


def build_action_candidates(obs, max_targets_per_source=8, send_fraction=0.55):
    player = int(obs.get("player", 0))
    planets = obs.get("planets", []) or []
    owned = [p for p in planets if int(p[PLANET_OWNER]) == player and int(p[PLANET_SHIPS]) > 1]
    targets = [p for p in planets if int(p[PLANET_OWNER]) != player]
    candidates = [{"kind": "noop", "source_id": -1, "target_id": -1, "angle": 0.0, "ships": 0}]
    for src in owned:
        ranked = sorted(
            targets,
            key=lambda p: (float(p[PLANET_X]) - float(src[PLANET_X])) ** 2 + (float(p[PLANET_Y]) - float(src[PLANET_Y])) ** 2,
        )[:max_targets_per_source]
        for target in ranked:
            ships = max(1, int(int(src[PLANET_SHIPS]) * send_fraction))
            candidates.append({
                "kind": "launch",
                "source_id": int(src[PLANET_ID]),
                "target_id": int(target[PLANET_ID]),
                "angle": angle_to_target((float(src[PLANET_X]), float(src[PLANET_Y])), (float(target[PLANET_X]), float(target[PLANET_Y]))),
                "ships": ships,
            })
    return candidates


def candidate_to_env_action(candidate):
    if candidate["kind"] == "noop":
        return []
    return [[int(candidate["source_id"]), float(candidate["angle"]), int(candidate["ships"])]]
```

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_action_space.py -q`

Expected: PASS.

## Task 2: NumPy PPO-Style Policy

**Files:**
- Create: `src/ow/rl/policy.py`
- Create: `tests/test_rl_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
import numpy as np
from src.ow.rl.policy import LinearActorCritic


def test_policy_sample_returns_valid_index_and_logprob():
    policy = LinearActorCritic(feature_dim=3, seed=7)
    features = np.array([[0.0, 0.0, 1.0], [1.0, 0.5, 1.0]], dtype=np.float64)
    idx, logp, value, entropy = policy.sample(features)
    assert idx in (0, 1)
    assert np.isfinite(logp)
    assert np.isfinite(value)
    assert entropy > 0.0


def test_policy_checkpoint_roundtrip(tmp_path):
    policy = LinearActorCritic(feature_dim=2, seed=1)
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = LinearActorCritic.load(path)
    assert np.allclose(policy.actor_w, loaded.actor_w)
    assert np.allclose(policy.value_w, loaded.value_w)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_rl_policy.py -q`

Expected: FAIL with missing module/class.

- [ ] **Step 3: Implement minimal actor-critic**

Implement `LinearActorCritic` with methods `score(features)`, `probs(features)`, `sample(features)`, `update_batch(batch, lr=3e-3, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5)`, `save(path)`, and `load(path)`. `update_batch` must normalize advantages and clip ratios.

- [ ] **Step 4: Verify GREEN and validation rules**

Run: `.venv/bin/python -m pytest tests/test_rl_policy.py -q`

Expected: PASS. Inspect `update_batch` to confirm PPO clipping, advantage normalization, entropy bonus, and gradient clipping are present.

## Task 3: Dense Reward and Metrics

**Files:**
- Create: `src/ow/rl/rewards.py`
- Create: `src/ow/rl/metrics.py`
- Create: `tests/test_rl_rewards_metrics.py`

- [ ] **Step 1: Write failing reward/metrics tests**

```python
import json
from src.ow.rl.rewards import shaped_reward
from src.ow.rl.metrics import RollingStats, MetricsWriter


def test_shaped_reward_rewards_production_gain_and_penalizes_idle():
    prev = {"player": 0, "planets": [[0, 0, 0, 0, 1, 10, 2], [1, 1, 1, 0, 1, 10, 3]]}
    curr = {"player": 0, "planets": [[0, 0, 0, 0, 1, 10, 5], [1, 1, 1, 0, 1, 10, 3]]}
    assert shaped_reward(prev, curr, action_count=1, terminal_reward=0.0) > shaped_reward(prev, curr, action_count=0, terminal_reward=0.0)


def test_metrics_writer_writes_jsonl(tmp_path):
    writer = MetricsWriter(tmp_path)
    writer.write({"update": 1, "wins": 2})
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows == [{"update": 1, "wins": 2}]
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_rl_rewards_metrics.py -q`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement reward components and metrics writer**

Implement reward from player production delta, planet-count delta, opponent-production penalty, action/noop penalty, invalid-action penalty hook, direct mistake penalties (sun-loss ships, out-of-bounds/no-hit ships, wasted ships, failed attacks, repeated failed attacks, oversend), and terminal reward. Implement `RollingStats` with wins/losses/episodes/recent win rate and `MetricsWriter` writing JSONL and CSV including mistake components.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_rewards_metrics.py -q`

Expected: PASS.

## Task 4: Pygame Viewer

**Files:**
- Create: `src/ow/rl/pygame_viewer.py`
- Create: `tests/test_rl_viewer.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing viewer overlay test**

```python
from src.ow.rl.pygame_viewer import TrainingOverlay


def test_training_overlay_text_contains_iterations_and_wins():
    overlay = TrainingOverlay()
    text = overlay.lines({"update": 12, "episode": 34, "wins": 9, "losses": 3, "recent_win_rate": 0.75})
    joined = "\n".join(text)
    assert "update=12" in joined
    assert "episode=34" in joined
    assert "wins=9" in joined
    assert "losses=3" in joined
    assert "win_rate=0.750" in joined
```

- [ ] **Step 2: Run test to verify RED**

Run: `.venv/bin/python -m pytest tests/test_rl_viewer.py -q`

Expected: FAIL with missing viewer module.

- [ ] **Step 3: Add dependency and implement viewer**

Add `pygame>=2.5` to `requirements.txt`. Implement `TrainingOverlay.lines(metrics)` and `PygameTrainingViewer` with `enabled`, `draw(obs, metrics)`, and `close()`. The viewer must render planets/fleets from observation dictionaries and overlay update/episode/wins/losses/recent win-rate.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_rl_viewer.py -q`

Expected: PASS.

## Task 5: Phase-1 Training CLI

**Files:**
- Create: `tools/train_rl_phase1.py`
- Create: `tests/test_rl_train_phase1.py`

- [ ] **Step 1: Write failing CLI smoke tests**

```python
from tools.train_rl_phase1 import parse_args, build_run_dir


def test_parse_args_defaults_to_2p_smoke():
    args = parse_args(["--episodes", "2", "--no-viewer"])
    assert args.mode == "2p"
    assert args.episodes == 2
    assert args.no_viewer is True


def test_build_run_dir_uses_mode(tmp_path):
    path = build_run_dir(tmp_path, "2p")
    assert path.name.startswith("2p_")
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py -q`

Expected: FAIL with missing CLI.

- [ ] **Step 3: Implement CLI**

Implement arguments: `--mode {2p}`, `--episodes`, `--updates`, `--run-dir`, `--seed`, `--viewer/--no-viewer`, `--fullscreen`, `--minimal-viewer/--rich-viewer`, `--burn-in-shift-seconds`, `--checkpoint-every`, `--eval-every`. Training loop must create Orbit Wars env, run self-play episodes with the same `LinearActorCritic` on both sides initially, collect trajectories, call clipped update, write metrics including mistake components, save checkpoints, and update Pygame viewer on sampled steps.

- [ ] **Step 4: Verify CLI smoke**

Run: `.venv/bin/python -m pytest tests/test_rl_train_phase1.py -q`

Expected: PASS.

Run: `.venv/bin/python tools/train_rl_phase1.py --episodes 1 --updates 1 --no-viewer --run-dir reports/rl_smoke`

Expected: exits 0 and writes metrics/checkpoint files under `reports/rl_smoke/`.

## Task 6: Start Local Training Smoke Run

**Files:**
- Output: `reports/rl_phase1_*/metrics.jsonl`
- Output: `reports/rl_phase1_*/metrics.csv`
- Output: `reports/rl_phase1_*/checkpoints/*.npz`

- [ ] **Step 1: Run test suite in parallel-safe batches**

Run independent commands in parallel where possible:

```bash
.venv/bin/python -m pytest tests/test_rl_action_space.py tests/test_rl_policy.py tests/test_rl_rewards_metrics.py -q
.venv/bin/python -m pytest tests/test_rl_viewer.py tests/test_rl_train_phase1.py -q
.venv/bin/python -m pytest tests/test_evaluate_headtohead.py tests/test_proof_gate.py -q
```

Expected: all PASS, allowing existing Kaggle/Pydantic warnings.

- [ ] **Step 2: Start no-viewer smoke run**

Run: `.venv/bin/python tools/train_rl_phase1.py --mode 2p --episodes 2 --updates 1 --no-viewer --run-dir reports/rl_phase1_smoke`

Expected: exits 0, prints visible metrics, writes JSONL/CSV/checkpoint.

- [ ] **Step 3: Start viewer run locally**

Run: `.venv/bin/python tools/train_rl_phase1.py --mode 2p --episodes 200 --updates 50 --viewer --run-dir reports/rl_phase1_live`

Expected: Pygame window shows the game plus overlay with update, episode, wins, losses, and recent win-rate.

## Self-Review

- Spec coverage: gate is diagnostic, pure RL starts now, local Pygame viewer is mandatory, visible iteration and win counters are included, Kaggle remains logs-only.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation tasks remain.
- Type consistency: module names and function names are consistent across tasks.
