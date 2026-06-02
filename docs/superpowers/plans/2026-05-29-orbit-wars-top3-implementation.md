# Orbit Wars Top-3-Ready Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, optimize, and locally prove a non-RL, NumPy-inference Orbit Wars agent that is strong enough to be considered top-3-ready before any manual Kaggle submission.

**Architecture:** The project is a reproducible local competition stack: official assets and public baselines feed a ladder harness; replay datasets feed supervised feature/model pipelines; a feature-heavy NumPy agent combines physics, timelines, defense, attack, opponent profiling, and small exported models. Submission is guarded by a proof gate that requires local ladder evidence, timing safety, ablation evidence, and explicit user approval.

**Tech Stack:** Python 3.11+, NumPy, pytest, kaggle-environments, Kaggle API/CLI/MCP, optional PyTorch or scikit-learn for training only, tar/json streaming, shell scripts for repeatable runs.

---

## File Structure

- Create: `.gitignore` — prevents credential, replay, model, and report leaks.
- Create: `requirements.txt` — local runtime and test dependencies.
- Create: `README.md` — exact local workflow and no-submission rule.
- Create: `docs/research/orbit_wars_research.md` — preserved mechanics, discussions, and public-agent findings.
- Create: `configs/default.json` — seeds, opponents, replay quota, timing budgets, proof thresholds.
- Create: `src/ow/__init__.py` — package marker.
- Create: `src/ow/constants.py` — official game constants.
- Create: `src/ow/types.py` — small dataclasses for match results and replay metadata.
- Create: `src/ow/state.py` — robust observation-to-NumPy conversion.
- Create: `src/ow/geometry.py` — distances, fleet speed, sun collision, moving positions, intercepts.
- Create: `src/ow/timeline.py` — per-planet arrival and threat projections.
- Create: `src/ow/features.py` — engineered feature matrices for candidate moves and supervised training.
- Create: `src/ow/policy.py` — deterministic heuristic modules and candidate generation.
- Create: `src/ow/models_numpy.py` — NumPy inference for exported supervised models.
- Create: `src/ow/agent_core.py` — orchestrates state, features, policy, model filters, time budget.
- Create: `main.py` — Kaggle root submission entrypoint using `src/ow` code during development.
- Create: `tools/build_submission.py` — creates a single-file or archive submission only after proof gate passes.
- Create: `tools/download_competition_assets.py` — downloads official files.
- Create: `tools/download_replays.py` — downloads replay datasets while enforcing the 100 GB cap.
- Create: `tools/parse_replays.py` — streams replay JSON into compact local training records.
- Create: `tools/train_models.py` — trains supervised models and exports `.npz` weights.
- Create: `tools/evaluate_ladder.py` — runs local matches and writes reports.
- Create: `tools/proof_gate.py` — blocks submission until evidence files satisfy thresholds.
- Create: `baselines/README.md` — attribution and source links for imported public agents.
- Create: `baselines/random_agent.py` — deterministic simple legal baseline.
- Create: `baselines/greedy_sample.py` — official sample-style baseline.
- Create: `baselines/style_bots.py` — rush, greedy, turtle, swarm, and idle-heavy local opponents.
- Create: `data/.gitkeep` — local data root marker.
- Create: `models/.gitkeep` — exported model root marker.
- Create: `reports/.gitkeep` — evaluation report root marker.
- Create: `tests/test_geometry.py` — physics tests.
- Create: `tests/test_state.py` — observation parser tests.
- Create: `tests/test_timeline.py` — combat/timeline tests.
- Create: `tests/test_features.py` — feature-shape and feature-value tests.
- Create: `tests/test_agent_core.py` — legal action and time-budget tests.
- Create: `tests/test_replay_quota.py` — 100 GB replay quota tests.
- Create: `tests/test_proof_gate.py` — no-submission guard tests.

## Global Execution Rules

- Keep all project files inside `/home/chris/orbitwars`.
- Do not store Kaggle credentials in the repo.
- Do not submit to Kaggle from implementation tasks.
- Run unit tests after each task that changes code.
- Run the ladder only after the agent can produce legal moves locally.
- Use small smoke datasets before large replay downloads.
- Keep final `main.py` NumPy-compatible; training-only dependencies must not be required by Kaggle inference.
- Record every benchmark in `reports/` with command, seed range, opponents, timing, and git commit hash when git is available.

---

### Task 1: Initialize Project Foundation

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `configs/default.json`
- Create: `src/ow/__init__.py`
- Create: `data/.gitkeep`
- Create: `models/.gitkeep`
- Create: `reports/.gitkeep`

- [ ] **Step 1: Initialize git if absent**

Run from `/home/chris/orbitwars`:

```bash
git rev-parse --is-inside-work-tree || git init
```

Expected: either `true` or `Initialized empty Git repository`.

- [ ] **Step 2: Write project ignore rules**

Create `.gitignore` with:

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
.env
kaggle.json
*.log
data/raw/
data/replays/
data/episodes/
data/processed/
data/cache/
models/*.pt
models/*.pth
models/*.ckpt
models/*.joblib
reports/*.jsonl
reports/*.csv
reports/*.html
submission.tar.gz
submission.zip
```

- [ ] **Step 3: Write dependencies**

Create `requirements.txt` with:

```text
numpy>=1.26
pytest>=8.0
kaggle-environments>=1.28.0
kaggle>=1.6.17
tqdm>=4.66
orjson>=3.10
```

- [ ] **Step 4: Write default configuration**

Create `configs/default.json` with:

```json
{
  "competition": "orbit-wars",
  "replay_download_cap_gb": 100,
  "episode_steps": 500,
  "turn_time_budget_seconds": 0.80,
  "hard_turn_timeout_seconds": 1.00,
  "training": {
    "max_vram_gb": 6,
    "batch_size": 8192,
    "epochs": 20,
    "validation_split_by_episode": true
  },
  "ladder": {
    "seed_start": 1,
    "seed_count_smoke": 16,
    "seed_count_full": 256,
    "modes": [2, 4],
    "proof_min_win_rate_vs_public": 0.58,
    "proof_min_avg_rank_4p": 1.85,
    "proof_max_p99_turn_seconds": 0.80,
    "proof_max_error_rate": 0.0
  }
}
```

- [ ] **Step 5: Write README workflow**

Create `README.md` with:

```markdown
# Orbit Wars Top-3-Ready Agent

Local non-RL competition stack for Kaggle Orbit Wars.

## Hard rule

No Kaggle submission is allowed until `tools/proof_gate.py` passes and the user explicitly approves the submission.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Test

```bash
python -m pytest -q
```

## Local evaluation

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 16 --mode 2
```

## Submission guard

```bash
python tools/proof_gate.py --config configs/default.json --report reports/latest_summary.json
```
```

- [ ] **Step 6: Create package and marker files**

Create `src/ow/__init__.py`, `data/.gitkeep`, `models/.gitkeep`, `reports/.gitkeep` as empty files.

- [ ] **Step 7: Verify foundation files exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
required = ['.gitignore','requirements.txt','README.md','configs/default.json','src/ow/__init__.py','data/.gitkeep','models/.gitkeep','reports/.gitkeep']
missing = [p for p in required if not Path(p).exists()]
assert not missing, missing
print('foundation ok')
PY
```

Expected: `foundation ok`.

- [ ] **Step 8: Commit foundation**

```bash
git add .gitignore requirements.txt README.md configs/default.json src/ow/__init__.py data/.gitkeep models/.gitkeep reports/.gitkeep
git commit -m "chore: initialize orbit wars project foundation"
```

Expected: commit succeeds. If git identity is unavailable, record the failure text in `reports/git_commit_notes.md` and continue without configuring global git.

---

### Task 2: Preserve Competition Research and Official Mechanics

**Files:**
- Create: `docs/research/orbit_wars_research.md`
- Create: `tools/download_competition_assets.py`
- Test: `tests/test_replay_quota.py`

- [ ] **Step 1: Write official asset downloader test**

Create `tests/test_replay_quota.py` with:

```python
from tools.download_replays import bytes_to_gb, would_exceed_cap


def test_bytes_to_gb_uses_binary_gib():
    assert bytes_to_gb(1024 ** 3) == 1.0


def test_would_exceed_cap_blocks_over_100_gb():
    assert would_exceed_cap(current_bytes=99 * 1024 ** 3, next_bytes=2 * 1024 ** 3, cap_gb=100)
    assert not would_exceed_cap(current_bytes=90 * 1024 ** 3, next_bytes=5 * 1024 ** 3, cap_gb=100)
```

- [ ] **Step 2: Run quota test and verify import failure**

Run:

```bash
python -m pytest tests/test_replay_quota.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.download_replays'`.

- [ ] **Step 3: Write replay quota helpers**

Create `tools/download_replays.py` with:

```python
from __future__ import annotations


def bytes_to_gb(num_bytes: int) -> float:
    return round(num_bytes / float(1024 ** 3), 6)


def would_exceed_cap(current_bytes: int, next_bytes: int, cap_gb: float) -> bool:
    return current_bytes + next_bytes > int(cap_gb * (1024 ** 3))
```

- [ ] **Step 4: Write competition asset downloader**

Create `tools/download_competition_assets.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path


def write_known_competition_summary(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'official_summary.md'
    path.write_text(
        '# Official Orbit Wars Summary\n\n'
        '- Board: continuous 100x100.\n'
        '- Sun: center (50,50), radius 10, path crossing destroys fleets.\n'
        '- Players: 2 or 4.\n'
        '- Episode length: 500 turns.\n'
        '- Fleet speed: 1 + 5 * (log(ships)/log(1000)) ** 1.5.\n'
        '- Final score: ships on owned planets plus ships in owned fleets.\n'
        '- Submission: root-level main.py exposing agent(obs, config=None).\n',
        encoding='utf-8',
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/raw/competition')
    args = parser.parse_args()
    path = write_known_competition_summary(Path(args.output_dir))
    print(path)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Write research log**

Create `docs/research/orbit_wars_research.md` with:

```markdown
# Orbit Wars Research Log

## Required discussions

- 697725: RL lessons. Main takeaways: fast environment matters, RL is unstable and costly, reward shaping and inductive bias matter, and generated strategy advice must be audited.
- 701894: Daily episode datasets. Use daily top-rated episode datasets for replay mining and opponent-style analysis.
- 697413: Top-10% daily episode replay datasets. Use tar streaming and manifests to avoid wasteful extraction.

## Official mechanics to preserve

- Continuous 100x100 board.
- Sun at (50,50), radius 10.
- Planets: `[id, owner, x, y, radius, ships, production]`.
- Fleets: `[id, owner, x, y, angle, from_planet_id, ships]`.
- Comets spawn at fixed intervals and can be captured as planets.
- Production happens after launch and before movement.
- Combat groups arrivals by owner; top attacker fights second attacker; ties destroy attackers.

## Public-agent techniques to implement and test

- Hellburner-style geometry, future timelines, early expansion optimizer, and reinforcement graph.
- Roman-style phase weights and heavily tuned thresholds.
- Emanuel-style formal simulation layers and FFA anti-kingmaking.
- Yash-style shallow sim/value search within a strict time budget.
- Konbu-style supervised shot validator exported to NumPy.
```

- [ ] **Step 6: Verify tests pass**

Run:

```bash
python -m pytest tests/test_replay_quota.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit research assets**

```bash
git add docs/research/orbit_wars_research.md tools/download_competition_assets.py tools/download_replays.py tests/test_replay_quota.py
git commit -m "docs: preserve orbit wars research inputs"
```

Expected: commit succeeds or a local commit note is recorded if git identity is unavailable.

---

### Task 3: Implement Core Constants, State Parsing, and Geometry

**Files:**
- Create: `src/ow/constants.py`
- Create: `src/ow/state.py`
- Create: `src/ow/geometry.py`
- Test: `tests/test_state.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write state parser tests**

Create `tests/test_state.py` with:

```python
import numpy as np

from src.ow.state import GameState, parse_observation


def test_parse_dict_observation_to_numpy_arrays():
    obs = {
        'player': 0,
        'step': 7,
        'angular_velocity': 0.03,
        'planets': [[0, 0, 10.0, 10.0, 2.0, 20, 3], [1, -1, 20.0, 20.0, 2.6, 5, 5]],
        'fleets': [[0, 1, 15.0, 15.0, 0.0, 2, 12]],
        'comets': [],
        'initial_planets': [[0, 0, 10.0, 10.0, 2.0, 10, 3], [1, -1, 20.0, 20.0, 2.6, 5, 5]],
    }
    state = parse_observation(obs)
    assert isinstance(state, GameState)
    assert state.player == 0
    assert state.step == 7
    assert state.planets.shape == (2, 7)
    assert state.fleets.shape == (1, 7)
    assert np.array_equal(state.my_planet_mask, np.array([True, False]))


def test_parse_object_observation_to_numpy_arrays():
    class Obs:
        player = 1
        step = 3
        angular_velocity = 0.04
        planets = [[0, 1, 80.0, 80.0, 2.0, 10, 2]]
        fleets = []
        comets = []
        initial_planets = [[0, 1, 80.0, 80.0, 2.0, 10, 2]]

    state = parse_observation(Obs())
    assert state.player == 1
    assert state.planets.dtype == np.float64
    assert state.fleets.shape == (0, 7)
```

- [ ] **Step 2: Write geometry tests**

Create `tests/test_geometry.py` with:

```python
import math

import numpy as np

from src.ow.geometry import distance_matrix, fleet_speed, segment_hits_circle, angle_between


def test_fleet_speed_matches_known_bounds():
    assert fleet_speed(1) == 1.0
    assert 5.9 < fleet_speed(1000) <= 6.0
    assert 4.5 < fleet_speed(500) < 5.5


def test_segment_hits_sun_center_crossing():
    assert segment_hits_circle(np.array([0.0, 50.0]), np.array([100.0, 50.0]), np.array([50.0, 50.0]), 10.0)
    assert not segment_hits_circle(np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([50.0, 50.0]), 10.0)


def test_distance_matrix_shape_and_values():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [6.0, 8.0]])
    d = distance_matrix(a, b)
    assert d.shape == (2, 2)
    assert d[1, 0] == 5.0
    assert d[0, 1] == 10.0


def test_angle_between_uses_atan2():
    assert math.isclose(angle_between(0, 0, 0, 1), math.pi / 2)
```

- [ ] **Step 3: Run tests and verify failures**

Run:

```bash
python -m pytest tests/test_state.py tests/test_geometry.py -q
```

Expected: FAIL with missing `src.ow.state` and `src.ow.geometry` modules.

- [ ] **Step 4: Implement constants**

Create `src/ow/constants.py` with:

```python
BOARD_SIZE = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SHIP_SPEED_MAX = 6.0
EPISODE_STEPS = 500
COMET_SPEED = 4.0
PLANET_COL_ID = 0
PLANET_COL_OWNER = 1
PLANET_COL_X = 2
PLANET_COL_Y = 3
PLANET_COL_RADIUS = 4
PLANET_COL_SHIPS = 5
PLANET_COL_PROD = 6
FLEET_COL_ID = 0
FLEET_COL_OWNER = 1
FLEET_COL_X = 2
FLEET_COL_Y = 3
FLEET_COL_ANGLE = 4
FLEET_COL_FROM = 5
FLEET_COL_SHIPS = 6
```

- [ ] **Step 5: Implement state parser**

Create `src/ow/state.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import PLANET_COL_OWNER


def _get(obs: Any, name: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _array(rows: Any, width: int) -> np.ndarray:
    if rows is None or len(rows) == 0:
        return np.zeros((0, width), dtype=np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, width)
    return arr


@dataclass(frozen=True)
class GameState:
    player: int
    step: int
    angular_velocity: float
    planets: np.ndarray
    fleets: np.ndarray
    comets: np.ndarray
    initial_planets: np.ndarray
    my_planet_mask: np.ndarray
    enemy_planet_mask: np.ndarray
    neutral_planet_mask: np.ndarray


def parse_observation(obs: Any) -> GameState:
    player = int(_get(obs, 'player', 0))
    step = int(_get(obs, 'step', _get(obs, 'turn', 0)))
    angular_velocity = float(_get(obs, 'angular_velocity', 0.0))
    planets = _array(_get(obs, 'planets', []), 7)
    fleets = _array(_get(obs, 'fleets', []), 7)
    comets = _array(_get(obs, 'comets', []), 7)
    initial_planets = _array(_get(obs, 'initial_planets', planets.tolist()), 7)
    owners = planets[:, PLANET_COL_OWNER].astype(np.int64) if len(planets) else np.zeros(0, dtype=np.int64)
    my_planet_mask = owners == player
    enemy_planet_mask = (owners >= 0) & (owners != player)
    neutral_planet_mask = owners < 0
    return GameState(
        player=player,
        step=step,
        angular_velocity=angular_velocity,
        planets=planets,
        fleets=fleets,
        comets=comets,
        initial_planets=initial_planets,
        my_planet_mask=my_planet_mask,
        enemy_planet_mask=enemy_planet_mask,
        neutral_planet_mask=neutral_planet_mask,
    )
```

- [ ] **Step 6: Implement geometry**

Create `src/ow/geometry.py` with:

```python
from __future__ import annotations

import math
import numpy as np

from .constants import SHIP_SPEED_MAX


def fleet_speed(ships: float) -> float:
    ships = max(float(ships), 1.0)
    if ships <= 1.0:
        return 1.0
    speed = 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(SHIP_SPEED_MAX, max(1.0, speed))


def angle_between(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(y1 - y0, x1 - x0)


def distance_matrix(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if len(a_xy) == 0 or len(b_xy) == 0:
        return np.zeros((len(a_xy), len(b_xy)), dtype=np.float64)
    delta = a_xy[:, None, :] - b_xy[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def segment_hits_circle(start: np.ndarray, end: np.ndarray, center: np.ndarray, radius: float) -> bool:
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 == 0.0:
        return float(np.linalg.norm(start - center)) <= radius
    t = float(np.dot(center - start, seg) / seg_len2)
    t = max(0.0, min(1.0, t))
    closest = start + t * seg
    return float(np.linalg.norm(closest - center)) <= radius
```

- [ ] **Step 7: Verify tests pass**

Run:

```bash
python -m pytest tests/test_state.py tests/test_geometry.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit core parsing and geometry**

```bash
git add src/ow/constants.py src/ow/state.py src/ow/geometry.py tests/test_state.py tests/test_geometry.py
git commit -m "feat: add orbit wars state parsing and geometry"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 4: Implement Timeline, Threat, and Feature Extraction

**Files:**
- Create: `src/ow/timeline.py`
- Create: `src/ow/features.py`
- Test: `tests/test_timeline.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write timeline tests**

Create `tests/test_timeline.py` with:

```python
import numpy as np

from src.ow.timeline import estimate_required_capture_ships, keep_needed_by_planet


def test_capture_ship_estimate_includes_margin_and_production():
    assert estimate_required_capture_ships(target_ships=10, target_prod=2, eta=3, margin=2) == 19


def test_keep_needed_reserves_frontline_planet():
    planets = np.array([
        [0, 0, 10, 10, 2, 30, 3],
        [1, 1, 20, 10, 2, 40, 4],
    ], dtype=float)
    reserves = keep_needed_by_planet(planets, player=0, horizon=20)
    assert reserves.shape == (2,)
    assert reserves[0] >= 6
    assert reserves[1] == 0
```

- [ ] **Step 2: Write feature tests**

Create `tests/test_features.py` with:

```python
import numpy as np

from src.ow.state import parse_observation
from src.ow.features import build_candidate_features


def test_candidate_features_have_stable_columns():
    obs = {
        'player': 0,
        'step': 10,
        'angular_velocity': 0.03,
        'planets': [[0, 0, 10, 10, 2, 50, 3], [1, -1, 30, 10, 2, 5, 5], [2, 1, 80, 80, 2, 20, 4]],
        'fleets': [],
        'comets': [],
        'initial_planets': [[0, 0, 10, 10, 2, 10, 3], [1, -1, 30, 10, 2, 5, 5], [2, 1, 80, 80, 2, 10, 4]],
    }
    state = parse_observation(obs)
    features, pairs = build_candidate_features(state)
    assert features.shape[1] == 16
    assert pairs.shape[1] == 2
    assert len(features) >= 2
```

- [ ] **Step 3: Run tests and verify failures**

Run:

```bash
python -m pytest tests/test_timeline.py tests/test_features.py -q
```

Expected: FAIL with missing timeline/features modules.

- [ ] **Step 4: Implement timeline helpers**

Create `src/ow/timeline.py` with:

```python
from __future__ import annotations

import numpy as np

from .constants import PLANET_COL_OWNER, PLANET_COL_PROD, PLANET_COL_SHIPS


def estimate_required_capture_ships(target_ships: float, target_prod: float, eta: float, margin: int = 2) -> int:
    eta_i = max(0, int(np.ceil(float(eta))))
    return int(np.floor(float(target_ships) + float(target_prod) * eta_i + margin + 1))


def keep_needed_by_planet(planets: np.ndarray, player: int, horizon: int) -> np.ndarray:
    reserves = np.zeros(len(planets), dtype=np.int64)
    if len(planets) == 0:
        return reserves
    owners = planets[:, PLANET_COL_OWNER].astype(np.int64)
    my_idx = np.where(owners == player)[0]
    enemy_idx = np.where((owners >= 0) & (owners != player))[0]
    for i in my_idx:
        prod = int(planets[i, PLANET_COL_PROD])
        base = max(4, prod * 2)
        if len(enemy_idx):
            d = np.sqrt(np.sum((planets[enemy_idx, 2:4] - planets[i, 2:4]) ** 2, axis=1))
            pressure = max(0, int(30 - float(np.min(d))))
        else:
            pressure = 0
        reserves[i] = min(int(planets[i, PLANET_COL_SHIPS]), base + pressure // 3 + max(0, horizon // 25))
    return reserves
```

- [ ] **Step 5: Implement candidate feature matrix**

Create `src/ow/features.py` with:

```python
from __future__ import annotations

import numpy as np

from .constants import CENTER_X, CENTER_Y, SUN_RADIUS
from .geometry import distance_matrix, fleet_speed, segment_hits_circle
from .state import GameState
from .timeline import estimate_required_capture_ships, keep_needed_by_planet


FEATURE_COLUMNS = [
    'turn_norm', 'turns_left_norm', 'src_ships', 'src_prod', 'target_ships', 'target_prod',
    'target_owner_rel', 'distance', 'eta', 'required', 'surplus_after_send', 'sun_blocked',
    'target_is_neutral', 'target_is_enemy', 'source_keep', 'score_pressure'
]


def build_candidate_features(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    planets = state.planets
    if len(planets) == 0:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
    my_idx = np.where(state.my_planet_mask)[0]
    target_idx = np.where(~state.my_planet_mask)[0]
    if len(my_idx) == 0 or len(target_idx) == 0:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
    src_xy = planets[my_idx, 2:4]
    tgt_xy = planets[target_idx, 2:4]
    dmat = distance_matrix(src_xy, tgt_xy)
    keep = keep_needed_by_planet(planets, state.player, horizon=40)
    rows: list[list[float]] = []
    pairs: list[tuple[int, int]] = []
    center = np.array([CENTER_X, CENTER_Y], dtype=np.float64)
    for a, src_i in enumerate(my_idx):
        src = planets[src_i]
        src_ships = float(src[5])
        source_keep = float(keep[src_i])
        for b, tgt_i in enumerate(target_idx):
            tgt = planets[tgt_i]
            dist = float(dmat[a, b])
            send_guess = max(1.0, src_ships - source_keep)
            eta = dist / fleet_speed(send_guess)
            required = float(estimate_required_capture_ships(tgt[5], tgt[6], eta, margin=2))
            sun_blocked = float(segment_hits_circle(src[2:4], tgt[2:4], center, SUN_RADIUS + 0.25))
            owner = int(tgt[1])
            rel = 0.0 if owner < 0 else (1.0 if owner == state.player else 2.0)
            rows.append([
                state.step / 500.0,
                max(0, 500 - state.step) / 500.0,
                src_ships / 200.0,
                float(src[6]) / 5.0,
                float(tgt[5]) / 200.0,
                float(tgt[6]) / 5.0,
                rel,
                dist / 141.5,
                eta / 120.0,
                required / 200.0,
                (src_ships - source_keep - required) / 200.0,
                sun_blocked,
                float(owner < 0),
                float(owner >= 0 and owner != state.player),
                source_keep / 100.0,
                float(np.sum(planets[state.enemy_planet_mask, 6]) - np.sum(planets[state.my_planet_mask, 6])) / 50.0,
            ])
            pairs.append((int(src[0]), int(tgt[0])))
    return np.asarray(rows, dtype=np.float64), np.asarray(pairs, dtype=np.int64)
```

- [ ] **Step 6: Verify tests pass**

Run:

```bash
python -m pytest tests/test_timeline.py tests/test_features.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit timelines and features**

```bash
git add src/ow/timeline.py src/ow/features.py tests/test_timeline.py tests/test_features.py
git commit -m "feat: add timeline reserves and candidate features"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 5: Implement Legal Heuristic Agent Core

**Files:**
- Create: `src/ow/policy.py`
- Create: `src/ow/models_numpy.py`
- Create: `src/ow/agent_core.py`
- Create: `main.py`
- Test: `tests/test_agent_core.py`

- [ ] **Step 1: Write agent legality tests**

Create `tests/test_agent_core.py` with:

```python
import math
import time

from main import agent


def make_obs():
    return {
        'player': 0,
        'step': 12,
        'angular_velocity': 0.03,
        'planets': [[0, 0, 10, 10, 2, 60, 3], [1, -1, 25, 10, 2, 5, 5], [2, 1, 85, 85, 2, 30, 4]],
        'fleets': [],
        'comets': [],
        'initial_planets': [[0, 0, 10, 10, 2, 10, 3], [1, -1, 25, 10, 2, 5, 5], [2, 1, 85, 85, 2, 10, 4]],
    }


def test_agent_returns_legal_action_shape():
    moves = agent(make_obs())
    assert isinstance(moves, list)
    assert len(moves) <= 3
    for move in moves:
        assert len(move) == 3
        assert move[0] == 0
        assert -math.pi * 2 <= float(move[1]) <= math.pi * 2
        assert 1 <= int(move[2]) <= 60


def test_agent_smoke_time_budget():
    obs = make_obs()
    start = time.perf_counter()
    for _ in range(100):
        agent(obs)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
python -m pytest tests/test_agent_core.py -q
```

Expected: FAIL because `main.py` does not exist.

- [ ] **Step 3: Implement NumPy model fallback**

Create `src/ow/models_numpy.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LinearModel:
    weights: np.ndarray
    bias: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float64)
        logits = x @ self.weights + self.bias
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def load_linear_model(path: str | Path, n_features: int) -> LinearModel:
    p = Path(path)
    if not p.exists():
        return LinearModel(weights=np.zeros(n_features, dtype=np.float64), bias=0.0)
    data = np.load(p)
    return LinearModel(weights=data['weights'].astype(np.float64), bias=float(data['bias']))
```

- [ ] **Step 4: Implement heuristic policy**

Create `src/ow/policy.py` with:

```python
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .constants import CENTER_X, CENTER_Y, SUN_RADIUS
from .features import build_candidate_features
from .geometry import angle_between, segment_hits_circle
from .state import GameState
from .timeline import estimate_required_capture_ships, keep_needed_by_planet


@dataclass(frozen=True)
class PolicyConfig:
    max_moves: int = 3
    min_model_score: float = 0.45
    turn_budget_seconds: float = 0.80


def score_candidates(features: np.ndarray, model_scores: np.ndarray | None) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(0, dtype=np.float64)
    target_prod = features[:, 5]
    eta = features[:, 8]
    surplus = features[:, 10]
    sun_blocked = features[:, 11]
    neutral = features[:, 12]
    enemy = features[:, 13]
    pressure = features[:, 15]
    base = 2.5 * target_prod + 1.2 * enemy + 0.7 * neutral + 0.8 * surplus - 1.4 * eta - 4.0 * sun_blocked + 0.3 * pressure
    if model_scores is not None and len(model_scores) == len(base):
        base += 1.5 * (model_scores - 0.5)
    return base


def choose_moves(state: GameState, model=None, config: PolicyConfig | None = None) -> list[list[float]]:
    cfg = config or PolicyConfig()
    deadline = time.perf_counter() + cfg.turn_budget_seconds
    features, pairs = build_candidate_features(state)
    if len(features) == 0:
        return []
    model_scores = model.predict_proba(features) if model is not None else None
    scores = score_candidates(features, model_scores)
    order = np.argsort(-scores)
    planets_by_id = {int(p[0]): p for p in state.planets}
    keep = keep_needed_by_planet(state.planets, state.player, horizon=45)
    keep_by_id = {int(state.planets[i, 0]): int(keep[i]) for i in range(len(state.planets))}
    spent: dict[int, int] = {}
    targeted: set[int] = set()
    moves: list[list[float]] = []
    center = np.array([CENTER_X, CENTER_Y], dtype=np.float64)
    for idx in order:
        if time.perf_counter() > deadline or len(moves) >= cfg.max_moves:
            break
        if scores[idx] <= -1.0:
            continue
        src_id, tgt_id = map(int, pairs[idx])
        if tgt_id in targeted:
            continue
        src = planets_by_id[src_id]
        tgt = planets_by_id[tgt_id]
        if segment_hits_circle(src[2:4], tgt[2:4], center, SUN_RADIUS + 0.25):
            continue
        available = int(src[5]) - keep_by_id.get(src_id, 0) - spent.get(src_id, 0)
        if available <= 0:
            continue
        dist = float(np.linalg.norm(src[2:4] - tgt[2:4]))
        eta = max(1.0, dist / 4.0)
        required = estimate_required_capture_ships(tgt[5], tgt[6], eta, margin=2)
        send = min(available, required)
        if send <= 0 or send < required * 0.7:
            continue
        angle = angle_between(float(src[2]), float(src[3]), float(tgt[2]), float(tgt[3]))
        moves.append([src_id, float(angle), int(send)])
        spent[src_id] = spent.get(src_id, 0) + int(send)
        targeted.add(tgt_id)
    return moves
```

- [ ] **Step 5: Implement agent core**

Create `src/ow/agent_core.py` with:

```python
from __future__ import annotations

from .features import FEATURE_COLUMNS
from .models_numpy import load_linear_model
from .policy import PolicyConfig, choose_moves
from .state import parse_observation


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_linear_model('models/shot_validator.npz', n_features=len(FEATURE_COLUMNS))
    return _MODEL


def agent_impl(obs, config=None):
    try:
        state = parse_observation(obs)
        model = get_model()
        return choose_moves(state, model=model, config=PolicyConfig())
    except Exception:
        return []
```

- [ ] **Step 6: Implement Kaggle entrypoint**

Create `main.py` with:

```python
from src.ow.agent_core import agent_impl


def agent(obs, config=None):
    return agent_impl(obs, config)
```

- [ ] **Step 7: Verify agent tests pass**

Run:

```bash
python -m pytest tests/test_agent_core.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run all current tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit agent core**

```bash
git add src/ow/policy.py src/ow/models_numpy.py src/ow/agent_core.py main.py tests/test_agent_core.py
git commit -m "feat: add legal heuristic agent core"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 6: Add Public Baselines and Style Opponents

**Files:**
- Create: `baselines/README.md`
- Create: `baselines/random_agent.py`
- Create: `baselines/greedy_sample.py`
- Create: `baselines/style_bots.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write baseline tests**

Create `tests/test_baselines.py` with:

```python
from baselines.random_agent import agent as random_agent
from baselines.greedy_sample import agent as greedy_agent
from baselines.style_bots import rush_agent, turtle_agent


def obs():
    return {
        'player': 0,
        'step': 1,
        'angular_velocity': 0.03,
        'planets': [[0, 0, 10, 10, 2, 30, 3], [1, -1, 20, 10, 2, 5, 5], [2, 1, 80, 80, 2, 30, 4]],
        'fleets': [],
        'comets': [],
        'initial_planets': [[0, 0, 10, 10, 2, 10, 3], [1, -1, 20, 10, 2, 5, 5], [2, 1, 80, 80, 2, 10, 4]],
    }


def test_baselines_return_lists():
    for fn in [random_agent, greedy_agent, rush_agent, turtle_agent]:
        assert isinstance(fn(obs()), list)
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
python -m pytest tests/test_baselines.py -q
```

Expected: FAIL because baseline modules do not exist.

- [ ] **Step 3: Write baseline attribution**

Create `baselines/README.md` with:

```markdown
# Baseline Agents

This folder contains local baseline and style opponents used for evaluation.

Imported public agents must include source URL, author, notebook slug, access date, and license/competition sharing status when known.

Local style bots are not intended as submissions. They stress-test rush, greedy, turtle, swarm, and idle-heavy behaviors.
```

- [ ] **Step 4: Implement random legal baseline**

Create `baselines/random_agent.py` with:

```python
def agent(obs, config=None):
    return []
```

- [ ] **Step 5: Implement greedy sample baseline**

Create `baselines/greedy_sample.py` with:

```python
import math


def _get(obs, key, default):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def agent(obs, config=None):
    player = int(_get(obs, 'player', 0))
    planets = _get(obs, 'planets', [])
    moves = []
    for src in planets:
        if int(src[1]) != player:
            continue
        best = None
        best_dist = 1e9
        for tgt in planets:
            if int(tgt[1]) == player:
                continue
            dist = math.hypot(float(src[2]) - float(tgt[2]), float(src[3]) - float(tgt[3]))
            if dist < best_dist:
                best = tgt
                best_dist = dist
        if best is None:
            continue
        need = int(best[5]) + 1
        if int(src[5]) > need:
            moves.append([int(src[0]), math.atan2(float(best[3]) - float(src[3]), float(best[2]) - float(src[2])), need])
    return moves
```

- [ ] **Step 6: Implement style bots**

Create `baselines/style_bots.py` with:

```python
from baselines.greedy_sample import agent as greedy_agent


def rush_agent(obs, config=None):
    moves = greedy_agent(obs, config)
    return moves[:2]


def turtle_agent(obs, config=None):
    step = obs.get('step', 0) if isinstance(obs, dict) else getattr(obs, 'step', 0)
    if step < 80:
        return []
    return greedy_agent(obs, config)[:1]


def greedy_expand_agent(obs, config=None):
    return greedy_agent(obs, config)[:4]


def swarm_agent(obs, config=None):
    moves = greedy_agent(obs, config)
    return [[m[0], m[1], max(1, min(3, int(m[2])))] for m in moves[:5]]


def idle_heavy_agent(obs, config=None):
    step = obs.get('step', 0) if isinstance(obs, dict) else getattr(obs, 'step', 0)
    return [] if step % 25 else greedy_agent(obs, config)[:1]
```

- [ ] **Step 7: Verify baseline tests pass**

Run:

```bash
python -m pytest tests/test_baselines.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit baselines**

```bash
git add baselines tests/test_baselines.py
git commit -m "feat: add baseline and style opponent agents"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 7: Implement Local Ladder Evaluation Harness

**Files:**
- Create: `src/ow/types.py`
- Create: `tools/evaluate_ladder.py`
- Test: `tests/test_evaluate_ladder.py`

- [ ] **Step 1: Write ladder helper tests**

Create `tests/test_evaluate_ladder.py` with:

```python
from tools.evaluate_ladder import summarize_results
from src.ow.types import MatchResult


def test_summarize_results_computes_win_rate_and_timing():
    results = [
        MatchResult(seed=1, mode=2, rank=1, score=100.0, turns=500, error=False, p99_turn_seconds=0.2),
        MatchResult(seed=2, mode=2, rank=2, score=80.0, turns=500, error=False, p99_turn_seconds=0.4),
    ]
    summary = summarize_results(results)
    assert summary['matches'] == 2
    assert summary['win_rate'] == 0.5
    assert summary['avg_rank'] == 1.5
    assert summary['max_p99_turn_seconds'] == 0.4
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_evaluate_ladder.py -q
```

Expected: FAIL because `tools.evaluate_ladder` does not exist.

- [ ] **Step 3: Implement match result type**

Create `src/ow/types.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchResult:
    seed: int
    mode: int
    rank: int
    score: float
    turns: int
    error: bool
    p99_turn_seconds: float
```

- [ ] **Step 4: Implement ladder summary and runner shell**

Create `tools/evaluate_ladder.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.ow.types import MatchResult


def summarize_results(results: list[MatchResult]) -> dict:
    if not results:
        return {'matches': 0, 'win_rate': 0.0, 'avg_rank': 0.0, 'max_p99_turn_seconds': 0.0, 'error_rate': 0.0}
    return {
        'matches': len(results),
        'win_rate': sum(1 for r in results if r.rank == 1) / len(results),
        'avg_rank': mean(r.rank for r in results),
        'avg_score': mean(r.score for r in results),
        'max_p99_turn_seconds': max(r.p99_turn_seconds for r in results),
        'error_rate': sum(1 for r in results if r.error) / len(results),
    }


def write_summary(results: list[MatchResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize_results(results), indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--candidate', default='main.py')
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--mode', type=int, choices=[2, 4], default=2)
    parser.add_argument('--output', default='reports/latest_summary.json')
    args = parser.parse_args()
    results = [MatchResult(seed=i + 1, mode=args.mode, rank=1, score=0.0, turns=0, error=False, p99_turn_seconds=0.0) for i in range(args.seeds)]
    write_summary(results, Path(args.output))
    print(args.output)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Verify ladder helper tests pass**

Run:

```bash
python -m pytest tests/test_evaluate_ladder.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Replace shell runner with real kaggle-environments execution**

Modify `tools/evaluate_ladder.py` so `main()` imports `kaggle_environments.make`, runs `orbit_wars` with candidate path plus baseline paths, extracts final ranks/scores/errors, and writes a JSON summary. The implementation must preserve `summarize_results()` unchanged because tests depend on it.

Use this command for a smoke run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 2 --mode 2 --output reports/smoke_summary.json
```

Expected: `reports/smoke_summary.json` exists and contains `matches: 2`.

- [ ] **Step 7: Commit ladder harness**

```bash
git add src/ow/types.py tools/evaluate_ladder.py tests/test_evaluate_ladder.py reports/.gitkeep
git commit -m "feat: add local ladder evaluation harness"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 8: Implement Replay Download and Streaming Parser

**Files:**
- Modify: `tools/download_replays.py`
- Create: `tools/parse_replays.py`
- Test: `tests/test_parse_replays.py`

- [ ] **Step 1: Write replay parser test with tiny tar**

Create `tests/test_parse_replays.py` with:

```python
import io
import json
import tarfile
from pathlib import Path

from tools.parse_replays import iter_replay_json_records


def test_iter_replay_json_records_streams_tar(tmp_path: Path):
    tar_path = tmp_path / 'episodes.tar.gz'
    payload = {'id': 123, 'steps': [[{'observation': {'player': 0}, 'action': []}]]}
    with tarfile.open(tar_path, 'w:gz') as tar:
        raw = json.dumps(payload).encode('utf-8')
        info = tarfile.TarInfo('episode_123.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    records = list(iter_replay_json_records(tar_path))
    assert records == [payload]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_parse_replays.py -q
```

Expected: FAIL because `tools.parse_replays` does not exist.

- [ ] **Step 3: Implement streaming parser**

Create `tools/parse_replays.py` with:

```python
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterator


def iter_replay_json_records(tar_path: Path) -> Iterator[dict]:
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.json'):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            yield json.loads(fh.read().decode('utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('tar_path')
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    count = 0
    for _record in iter_replay_json_records(Path(args.tar_path)):
        count += 1
        if count >= args.limit:
            break
    print(count)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Extend replay downloader with manifest entries**

Modify `tools/download_replays.py` to include:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


def bytes_to_gb(num_bytes: int) -> float:
    return round(num_bytes / float(1024 ** 3), 6)


def would_exceed_cap(current_bytes: int, next_bytes: int, cap_gb: float) -> bool:
    return current_bytes + next_bytes > int(cap_gb * (1024 ** 3))


@dataclass(frozen=True)
class ReplayDatasetEntry:
    source: str
    local_path: str
    size_bytes: int
    discussion_id: str


def append_manifest_entry(manifest_path: Path, entry: ReplayDatasetEntry, cap_gb: float = 100.0) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
    current_bytes = sum(int(row['size_bytes']) for row in existing)
    if would_exceed_cap(current_bytes, entry.size_bytes, cap_gb):
        raise ValueError(f'replay cap exceeded: current={bytes_to_gb(current_bytes)} next={bytes_to_gb(entry.size_bytes)} cap={cap_gb}')
    existing.append(asdict(entry))
    manifest_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
```

- [ ] **Step 5: Verify replay tests pass**

Run:

```bash
python -m pytest tests/test_replay_quota.py tests/test_parse_replays.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit replay pipeline foundation**

```bash
git add tools/download_replays.py tools/parse_replays.py tests/test_replay_quota.py tests/test_parse_replays.py
git commit -m "feat: add replay quota and streaming parser"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 9: Implement Training Dataset Builder and Supervised Model Export

**Files:**
- Modify: `tools/parse_replays.py`
- Create: `tools/train_models.py`
- Test: `tests/test_train_models.py`

- [ ] **Step 1: Write model export test**

Create `tests/test_train_models.py` with:

```python
import numpy as np
from pathlib import Path

from tools.train_models import train_logistic_numpy, save_npz_model
from src.ow.models_numpy import load_linear_model


def test_train_logistic_numpy_exports_loadable_model(tmp_path: Path):
    x = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.8], [0.1, 0.2]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    weights, bias = train_logistic_numpy(x, y, epochs=200, lr=0.5)
    path = tmp_path / 'model.npz'
    save_npz_model(path, weights, bias)
    model = load_linear_model(path, n_features=2)
    probs = model.predict_proba(x)
    assert probs[1] > probs[0]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_train_models.py -q
```

Expected: FAIL because `tools.train_models` does not exist.

- [ ] **Step 3: Implement NumPy logistic trainer**

Create `tools/train_models.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def train_logistic_numpy(x: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.05) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = np.clip(x @ weights + bias, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        err = probs - y
        weights -= lr * (x.T @ err) / max(1, len(x))
        bias -= lr * float(np.mean(err))
    return weights, bias


def save_npz_model(path: Path, weights: np.ndarray, bias: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=np.asarray(weights, dtype=np.float64), bias=np.asarray(float(bias)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='data/processed/shot_features.npz')
    parser.add_argument('--output', default='models/shot_validator.npz')
    args = parser.parse_args()
    data = np.load(args.features)
    weights, bias = train_logistic_numpy(data['x'], data['y'], epochs=50, lr=0.05)
    save_npz_model(Path(args.output), weights, bias)
    print(args.output)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verify model export test passes**

Run:

```bash
python -m pytest tests/test_train_models.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Add replay-to-feature extraction**

Modify `tools/parse_replays.py` to add a function `extract_training_rows(record: dict) -> tuple[list[list[float]], list[int]]` that reads each replay step, parses observations using `parse_observation`, calls `build_candidate_features`, labels candidate pairs as positive when the recorded action launches from the same source planet, and returns feature rows plus binary labels.

Run:

```bash
python -m pytest tests/test_parse_replays.py tests/test_train_models.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit training export pipeline**

```bash
git add tools/train_models.py tools/parse_replays.py tests/test_train_models.py
git commit -m "feat: add supervised model export pipeline"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 10: Integrate Model Scores and Opponent Profiling

**Files:**
- Modify: `src/ow/state.py`
- Modify: `src/ow/features.py`
- Modify: `src/ow/policy.py`
- Test: `tests/test_features.py`
- Test: `tests/test_agent_core.py`

- [ ] **Step 1: Add opponent profile test**

Append to `tests/test_features.py`:

```python
from src.ow.features import opponent_profile_features


def test_opponent_profile_detects_swarm_from_many_fleets():
    obs = {
        'player': 0,
        'step': 50,
        'angular_velocity': 0.03,
        'planets': [[0, 0, 10, 10, 2, 50, 3], [1, 1, 80, 80, 2, 50, 3]],
        'fleets': [[i, 1, 70 + i, 70, 0, 1, 2] for i in range(6)],
        'comets': [],
        'initial_planets': [[0, 0, 10, 10, 2, 10, 3], [1, 1, 80, 80, 2, 10, 3]],
    }
    state = parse_observation(obs)
    prof = opponent_profile_features(state)
    assert prof['enemy_fleet_count'] == 6
    assert prof['swarm_score'] > 0.5
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_features.py -q
```

Expected: FAIL because `opponent_profile_features` does not exist.

- [ ] **Step 3: Implement opponent profile features**

Modify `src/ow/features.py` to add:

```python
def opponent_profile_features(state: GameState) -> dict[str, float]:
    if len(state.fleets) == 0:
        return {'enemy_fleet_count': 0.0, 'enemy_mean_launch_ships': 0.0, 'swarm_score': 0.0, 'heavy_score': 0.0}
    enemy = state.fleets[state.fleets[:, 1].astype(int) != state.player]
    if len(enemy) == 0:
        return {'enemy_fleet_count': 0.0, 'enemy_mean_launch_ships': 0.0, 'swarm_score': 0.0, 'heavy_score': 0.0}
    mean_ships = float(np.mean(enemy[:, 6]))
    count = float(len(enemy))
    return {
        'enemy_fleet_count': count,
        'enemy_mean_launch_ships': mean_ships,
        'swarm_score': min(1.0, count / 8.0) * (1.0 if mean_ships <= 5.0 else 0.5),
        'heavy_score': min(1.0, mean_ships / 80.0),
    }
```

- [ ] **Step 4: Increase policy defense margins using profile**

Modify `src/ow/policy.py` to import `opponent_profile_features`, compute `profile = opponent_profile_features(state)`, and reduce attack score by `0.6 * profile['swarm_score']` when source surplus is low. Keep the existing legal-move tests passing.

- [ ] **Step 5: Verify tests pass**

Run:

```bash
python -m pytest tests/test_features.py tests/test_agent_core.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit adaptive profile integration**

```bash
git add src/ow/features.py src/ow/policy.py tests/test_features.py tests/test_agent_core.py
git commit -m "feat: add opponent profile features to policy"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 11: Improve Tactical Modules Through Iterative Ladder Runs

**Files:**
- Modify: `src/ow/policy.py`
- Modify: `src/ow/timeline.py`
- Modify: `src/ow/features.py`
- Modify: `reports/` outputs

- [ ] **Step 1: Run 2-player smoke ladder**

Run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 16 --mode 2 --output reports/ladder_2p_smoke.json
```

Expected: report exists, zero errors, no timeout failures.

- [ ] **Step 2: Run 4-player smoke ladder**

Run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 16 --mode 4 --output reports/ladder_4p_smoke.json
```

Expected: report exists, zero errors, no timeout failures.

- [ ] **Step 3: Add one tactical feature at a time**

For each feature below, modify the relevant module and run the two smoke ladder commands after the change:

- duplicate-target suppression in `src/ow/policy.py`
- source starvation risk penalty in `src/ow/features.py`
- high-production defense boost in `src/ow/timeline.py`
- late-game arrival-before-end filter in `src/ow/policy.py`
- neutral opening production ROI boost in `src/ow/policy.py`
- enemy-frontline pressure boost in `src/ow/features.py`
- comet target ROI feature when comets are present in `src/ow/features.py`

Expected for each feature: smoke reports have zero errors and no p99 turn time above `0.80` seconds.

- [ ] **Step 4: Commit tactical feature batch only if smoke improves or remains neutral**

```bash
git add src/ow/policy.py src/ow/timeline.py src/ow/features.py reports/.gitkeep
git commit -m "feat: improve tactical policy features"
```

Expected: commit succeeds only for features that do not regress smoke ladders. If a feature regresses both 2p and 4p smoke ladders, revert that feature before committing.

---

### Task 12: Mine Replays and Train First Validator

**Files:**
- Modify: `tools/download_replays.py`
- Modify: `tools/parse_replays.py`
- Modify: `tools/train_models.py`
- Create/Modify: `models/shot_validator.npz`
- Create/Modify: `reports/replay_training_summary.json`

- [ ] **Step 1: Download a small replay sample**

Use Kaggle dataset sources from discussions `701894` and `697413`. Start below 5 GB before any larger run:

```bash
python tools/download_replays.py --cap-gb 100 --max-new-gb 5 --output-dir data/replays --manifest data/replays/manifest.json
```

Expected: manifest exists, total recorded size is at most 5 GiB for the first sample.

- [ ] **Step 2: Parse sample replays to features**

Run:

```bash
python tools/parse_replays.py data/replays/episodes.tar.gz --limit 1000
```

Expected: printed count is greater than zero for a non-empty downloaded tar.

- [ ] **Step 3: Build shot feature table**

Run:

```bash
python tools/parse_replays.py data/replays/episodes.tar.gz --limit 5000 --features-output data/processed/shot_features.npz
```

Expected: `data/processed/shot_features.npz` exists with arrays `x` and `y`; `x.shape[1]` equals `len(FEATURE_COLUMNS)`.

- [ ] **Step 4: Train first validator**

Run:

```bash
python tools/train_models.py --features data/processed/shot_features.npz --output models/shot_validator.npz
```

Expected: `models/shot_validator.npz` exists and loads through `src.ow.models_numpy.load_linear_model`.

- [ ] **Step 5: Compare heuristic-only and model-filtered smoke ladders**

Run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 32 --mode 2 --output reports/model_2p_smoke.json
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 32 --mode 4 --output reports/model_4p_smoke.json
```

Expected: zero errors; compare summaries against pre-model smoke reports.

- [ ] **Step 6: Commit validator only if smoke evidence is non-regressive**

```bash
git add tools/download_replays.py tools/parse_replays.py tools/train_models.py models/shot_validator.npz reports/.gitkeep
git commit -m "feat: train first replay-derived shot validator"
```

Expected: commit succeeds if the model does not regress both smoke ladders. If it regresses, keep the model file untracked and adjust thresholds in `src/ow/policy.py` before retesting.

---

### Task 13: Full Local Optimization Loop

**Files:**
- Modify: `src/ow/policy.py`
- Modify: `src/ow/features.py`
- Modify: `src/ow/timeline.py`
- Modify: `models/shot_validator.npz`
- Create/Modify: `reports/optimization_log.md`

- [ ] **Step 1: Create optimization log**

Create `reports/optimization_log.md` with:

```markdown
# Optimization Log

Each row records one candidate change, command, result, and decision.

| Candidate | Change | 2p win rate | 4p avg rank | p99 turn | Decision |
|---|---|---:|---:|---:|---|
```

- [ ] **Step 2: Run broader public/style ladder**

Run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 128 --mode 2 --output reports/ladder_2p_128.json
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 128 --mode 4 --output reports/ladder_4p_128.json
```

Expected: zero errors, p99 below `0.80` seconds.

- [ ] **Step 3: Tune policy weights with ablations**

For each candidate weight group, run 64-seed 2p and 64-seed 4p ladders and append a row to `reports/optimization_log.md`:

- production ROI weight
- ETA penalty
- sun-blocked penalty
- neutral-vs-enemy phase weights
- source reserve multiplier
- swarm defense penalty
- heavy top-k opponent penalty
- late-game launch filter
- synchronized attack threshold

Expected: keep a change only when it improves one mode without unacceptable regression in the other mode. Unacceptable regression is a 4p average-rank decline above `0.05` or a 2p win-rate decline above `0.03` unless the other mode improves by at least double the loss.

- [ ] **Step 4: Expand replay data if model helps**

If the first replay-derived model is non-regressive, download more datasets up to 100 GB total:

```bash
python tools/download_replays.py --cap-gb 100 --max-new-gb 95 --output-dir data/replays --manifest data/replays/manifest.json
```

Expected: manifest total size is at most 100 GiB.

- [ ] **Step 5: Retrain with larger replay feature table**

Run:

```bash
python tools/parse_replays.py data/replays/episodes.tar.gz --limit 200000 --features-output data/processed/shot_features_large.npz
python tools/train_models.py --features data/processed/shot_features_large.npz --output models/shot_validator.npz
```

Expected: exported model loads and smoke ladders have zero errors.

- [ ] **Step 6: Commit best optimization candidate**

```bash
git add src/ow/policy.py src/ow/features.py src/ow/timeline.py models/shot_validator.npz reports/optimization_log.md
git commit -m "perf: tune orbit wars policy and validator"
```

Expected: commit succeeds for the best non-regressive candidate.

---

### Task 14: Implement Proof Gate and Submission Builder

**Files:**
- Create: `tools/proof_gate.py`
- Create: `tools/build_submission.py`
- Test: `tests/test_proof_gate.py`

- [ ] **Step 1: Write proof gate tests**

Create `tests/test_proof_gate.py` with:

```python
import json
from pathlib import Path

from tools.proof_gate import proof_gate_passes


def test_proof_gate_rejects_missing_manual_approval(tmp_path: Path):
    report = tmp_path / 'summary.json'
    report.write_text(json.dumps({'win_rate': 0.9, 'avg_rank': 1.5, 'max_p99_turn_seconds': 0.5, 'error_rate': 0.0}), encoding='utf-8')
    assert not proof_gate_passes(report, user_approved=False)


def test_proof_gate_accepts_strong_report_with_approval(tmp_path: Path):
    report = tmp_path / 'summary.json'
    report.write_text(json.dumps({'win_rate': 0.9, 'avg_rank': 1.5, 'max_p99_turn_seconds': 0.5, 'error_rate': 0.0}), encoding='utf-8')
    assert proof_gate_passes(report, user_approved=True)
```

- [ ] **Step 2: Run proof tests and verify failure**

Run:

```bash
python -m pytest tests/test_proof_gate.py -q
```

Expected: FAIL because `tools.proof_gate` does not exist.

- [ ] **Step 3: Implement proof gate**

Create `tools/proof_gate.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def proof_gate_passes(report_path: Path, user_approved: bool) -> bool:
    if not user_approved or not report_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding='utf-8'))
    return (
        float(report.get('win_rate', 0.0)) >= 0.58
        and float(report.get('avg_rank', 99.0)) <= 1.85
        and float(report.get('max_p99_turn_seconds', 99.0)) <= 0.80
        and float(report.get('error_rate', 1.0)) == 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='reports/latest_summary.json')
    parser.add_argument('--user-approved', action='store_true')
    args = parser.parse_args()
    ok = proof_gate_passes(Path(args.report), user_approved=args.user_approved)
    if not ok:
        raise SystemExit('proof gate failed: no submission allowed')
    print('proof gate passed')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Implement guarded submission builder**

Create `tools/build_submission.py` with:

```python
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from tools.proof_gate import proof_gate_passes


def build_submission(output: Path, report: Path, user_approved: bool) -> Path:
    if not proof_gate_passes(report, user_approved=user_approved):
        raise SystemExit('proof gate failed: refusing to build submission')
    output.unlink(missing_ok=True)
    with tarfile.open(output, 'w:gz') as tar:
        tar.add('main.py', arcname='main.py')
        tar.add('src', arcname='src')
        model_path = Path('models/shot_validator.npz')
        if model_path.exists():
            tar.add(model_path, arcname='models/shot_validator.npz')
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='submission.tar.gz')
    parser.add_argument('--report', default='reports/latest_summary.json')
    parser.add_argument('--user-approved', action='store_true')
    args = parser.parse_args()
    print(build_submission(Path(args.output), Path(args.report), args.user_approved))


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Verify proof gate tests pass**

Run:

```bash
python -m pytest tests/test_proof_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit proof gate**

```bash
git add tools/proof_gate.py tools/build_submission.py tests/test_proof_gate.py
git commit -m "feat: add submission proof gate"
```

Expected: commit succeeds or a local commit note is recorded.

---

### Task 15: Final Verification Before Any Submission Request

**Files:**
- Modify: `reports/latest_summary.json`
- Modify: `reports/final_proof_report.md`

- [ ] **Step 1: Run all unit tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run full 2-player ladder**

Run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 256 --mode 2 --output reports/final_2p_256.json
```

Expected: zero errors and p99 below `0.80` seconds.

- [ ] **Step 3: Run full 4-player ladder**

Run:

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 256 --mode 4 --output reports/final_4p_256.json
```

Expected: zero errors and p99 below `0.80` seconds.

- [ ] **Step 4: Write final proof report**

Create `reports/final_proof_report.md` with:

```markdown
# Final Proof Report

## Unit tests

Command: `python -m pytest -q`

## 2-player ladder

Report: `reports/final_2p_256.json`

## 4-player ladder

Report: `reports/final_4p_256.json`

## Timing

Maximum accepted p99 turn time: 0.80 seconds.

## Submission decision

Submission requires explicit user approval after reviewing this file.
```

- [ ] **Step 5: Run proof gate without approval to verify it blocks**

Run:

```bash
python tools/proof_gate.py --report reports/latest_summary.json
```

Expected: exits nonzero with `proof gate failed: no submission allowed`.

- [ ] **Step 6: Ask user before submission**

Show `reports/final_proof_report.md`, `reports/final_2p_256.json`, and `reports/final_4p_256.json` summary values to the user. Do not run `kaggle competitions submit` until the user gives explicit submission approval.

---

## Self-Review

- Spec coverage: bootstrap, research preservation, replay quota, state/geometry, feature-heavy agent, supervised model export, local ladder, optimization, proof gate, and no-submission rule are all represented by tasks.
- Placeholder scan: this plan avoids unresolved placeholders and names exact files, commands, and expected outcomes.
- Type consistency: `GameState`, `FEATURE_COLUMNS`, `LinearModel`, `MatchResult`, and `proof_gate_passes` are defined before later tasks reference them.
- Scope control: the first implementation pass builds a real legal agent and evaluation pipeline before larger replay downloads and training; expensive optimization starts only after smoke tests pass.
