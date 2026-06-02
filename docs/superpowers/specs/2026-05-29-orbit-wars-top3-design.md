# Orbit Wars Top-3-Ready Non-RL Agent Design

## Goal

Build a serious non-RL Orbit Wars competition stack inside `/home/chris/orbitwars` that can research public strategy, download up to 100 GB of replay data, train small supervised models on a 6 GB VRAM GPU, export submission-safe NumPy inference, and produce strict local evidence before any Kaggle submission.

This project must not submit automatically. Submission is blocked until local proof is strong and the user manually approves.

## Hard Constraints

- Use only files under `/home/chris/orbitwars` for project work.
- Final Kaggle agent must expose `agent(obs, config=None)` in root-level `main.py`.
- Final submission logic must be real, not mock/scaffold code.
- Final inference must be NumPy-compatible and submission-safe.
- RL is out of scope because cost/risk is too high.
- Supervised learning from replays is allowed.
- Replay download budget: maximum 100 GB.
- Hardware assumptions: 6 GB VRAM GPU, 32 GB RAM CPU.
- Do not submit until local evidence supports a top-tier candidate and the user explicitly approves.

## Research Inputs To Include

The project will preserve notes, citations, and extracted techniques from:

- Official Orbit Wars files: `README.md`, `agents.md`, sample `main.py`.
- Kaggle discussion `697725` — RL lessons: fast environment/simulator matters, reward/inductive bias matters, RL is expensive and unstable, audit AI-generated strategy.
- Kaggle discussion `701894` — daily episode datasets: use official/public top episode index datasets for replay mining.
- Kaggle discussion `697413` — top-10% daily replay datasets from Bovard.
- Public strong notebooks/agents including Hellburner, Roman LB-max, Emanuel heuristic, Yash sim/value/search, Konbu ML validator, and simpler greedy baselines.
- Similar non-RL game competitions: Planet Wars, Kore 2022, Hungry Geese, Google Research Football.

## Recommended Strategy

Use a hybrid non-RL stack:

1. A robust hand-built heuristic/search agent with strong physics and defensive planning.
2. Feature-heavy adaptive decision modules inspired by public top agents.
3. Small supervised models trained from replay-derived decisions and outcomes.
4. Pure NumPy model inference embedded in the final submission.
5. A strict local ladder, timing, and ablation proof gate.

This is more realistic than pure RL and has a higher ceiling than copying public baselines.

## Project Components

### 1. Competition Bootstrap

Create a reproducible local project with:

- Downloaded official competition files.
- Dependency manifest.
- Local runner using `kaggle_environments.make("orbit_wars")`.
- Baseline agent folder with public agents and attribution metadata.
- Submission builder that packages root-level `main.py` only after proof gate passes.

### 2. Replay Data Pipeline

Download and process replay datasets up to the 100 GB cap:

- Prefer daily top-rated/top-10% replay datasets from discussions `701894` and `697413`.
- Keep a manifest of dataset source, date, size, checksum if available, and local path.
- Stream parse `.tar.gz` replay JSON where possible instead of extracting everything.
- Extract per-turn observations, actions, winners, scores, player ratings, and episode metadata.
- Build compact feature tables for training and analysis.
- Cache derived datasets in compressed NumPy/parquet-style files, staying inside the project folder.

### 3. Local Evaluation Harness

Build a ladder runner that supports:

- 2-player and 4-player matches.
- Many deterministic seeds.
- Multiple opponent pools: official random/sample, public baselines, previous local candidates, and style bots.
- Score summaries: win rate, average rank, final score margin, planets owned, production owned, ships wasted, timeout/error rate.
- Timing summaries: mean, p95, p99, and worst turn time.
- Regression reports comparing each candidate to the current best local baseline and the user’s known ~996 public-score anchor.

The ladder is the primary proof mechanism. Public leaderboard claims from notebooks are not enough.

### 4. Core NumPy Agent

The final agent will be feature-heavy and adaptive. Core modules:

#### Observation and State Features

- Convert planets/fleets/comets into NumPy arrays.
- Track owner masks, ship counts, production, radius, orbit/static status, and distance matrices.
- Infer game phase: opening, early expansion, midgame, late, very late, final score race.
- Detect 2-player vs 4-player mode.

#### Geometry and Physics

- Fleet speed function matching official rules.
- Sun/path collision rejection using segment-circle checks.
- Moving planet prediction from `initial_planets` and `angular_velocity`.
- Iterative intercept solver for orbiting targets.
- First-hit detection to avoid accidentally colliding with the wrong planet.
- Out-of-bounds and comet path handling.

#### Planet Timeline Simulation

- Per-planet future timelines for ownership, production, incoming fleets, and combat.
- Fleet destination inference for visible fleets.
- Threat windows for enemy arrivals.
- Reinforcement arrival feasibility.
- Estimate minimum keep-needed ships over the next horizon.

#### Defense First

- Reserve ships on valuable/frontline planets.
- Reinforce threatened high-production planets before expansion.
- Avoid launching ships that expose a planet to near-term capture.
- Model enemy fast-fleet counterattack risk.

#### Opening Expansion

- Rank neutral captures by production, travel time, risk, sun safety, and future frontier value.
- Use multi-turn opening planning inspired by Hellburner-style early expansion.
- Avoid over-sending; send just enough plus production/contest margin.

#### Midgame Pressure and Attacks

- Target enemy planets with ROI scoring: production swing, ship cost, arrival time, defense risk, and tactical value.
- Multi-source synchronized attacks when one source cannot safely capture.
- Delay/third-party snipe logic: arrive after enemies weaken each other.
- Avoid attacking into obvious enemy reinforcement traps.

#### 4-Player FFA Adaptation

- Anti-kingmaking pressure: prefer leader pressure when safe, harvest exposed weak targets, avoid mutual destruction.
- Detect leader, weakest player, local neighbor, and isolated production clusters.
- Reduce commitment when two enemies can punish overextension.
- Exploit fights between other players without donating ships.

#### Comets

- Detect comet spawn windows and comet planet IDs.
- Evaluate comet captures by remaining lifetime, travel time, ship cost, production value, and tactical position.
- Avoid chasing comets that expire too soon or pull ships away from critical defense.

#### Late-Game Score Maximization

- Switch from long-term production ROI to final ship-score ROI near the end.
- Avoid launches that cannot arrive before game end unless fleet score preservation is beneficial.
- Capture/deny planets only when final score swing is positive.

#### Opponent Profiling

Track each opponent’s observed behavior:

- Rushy: frequent early attacks, low reserves.
- Greedy: many neutral captures, weak defense.
- Turtle: high reserves, low launch rate.
- Top-k/idle-heavy: waits then sends large decisive attacks.
- Swarm: many small launches.
- Opportunistic/sniper: targets weakened planets.

Use profiles to adjust defense margins, attack thresholds, snipe timing, and overcommit risk.

#### Time-Budgeted Search

- Generate a small candidate set from heuristics.
- Run shallow forward simulation/value scoring only within a conservative time budget.
- Fall back to safe heuristic moves when time is low.
- Maintain p99 turn time well below the 1 second limit.

### 5. Supervised Models

Train small models from replay-derived data. No RL loop.

Candidate models:

- Shot validator: should this proposed launch happen?
- Target value model: how valuable/risky is this source-target pair?
- Defense urgency model: how much reserve is needed for this planet?
- Opening target ranker: which neutral capture is strongest from this state?
- Endgame score swing model: whether a late launch improves final score.

Training approach:

- Use GPU for PyTorch/LightGBM-style experimentation if available.
- Keep models small enough for 6 GB VRAM.
- Validate by game split, not random row split, to reduce leakage.
- Compare against heuristic-only decisions.
- Export final selected models to `.npz` weights or compact tree tables.
- Implement inference in pure NumPy for submission.

Feature examples:

- Source/target ships, production, radius, owner relation.
- Distance, angle, ETA, fleet speed, sun safety, moving target intercept error.
- Incoming friendly/enemy ships by time bucket.
- Production and ship totals by player.
- Planet centrality/frontier score.
- Phase, turns remaining, 2p/4p flag.
- Opponent profile features.
- Local neighborhood pressure and reinforcement availability.
- Whether target is comet/static/orbiting/home/frontline.

### 6. Feature Expansion List

The implementation should favor many real, measurable features. Candidate feature groups:

- Distance matrix features.
- ETA matrix features for multiple send sizes.
- Sun-blocked path masks.
- Orbit prediction features for 5/10/20/50 turns.
- Target first-hit identity.
- Planet safety margin over future horizons.
- Production ROI by arrival time.
- Enemy contest probability.
- Friendly reinforcement slack.
- Fleet congestion and duplicate-target detection.
- Launch-size speed tradeoff features.
- Source starvation risk.
- Score leader pressure.
- Weak-player harvest opportunity.
- Third-party battle cleanup windows.
- Comet expiration/capture feasibility.
- Endgame arrival-before-deadline flag.
- Historical opponent launch rate.
- Historical opponent mean ships per launch.
- Idle-heavy opponent detector.
- Swarm detector.
- Greedy expansion detector.
- Rush detector.
- Tactical local cluster ownership.
- Home-planet vulnerability.
- Frontier depth from owned graph.
- Synchronized multi-source arrival feasibility.
- Overkill/waste estimate.
- Under-send failure probability.
- Confidence score for each proposed move.

### 7. Proof Gate

No Kaggle submission until all are satisfied:

1. Local candidate beats official/sample/random baselines decisively.
2. Local candidate beats imported public baselines across broad seeds.
3. Local candidate beats prior local best/current anchor in both 2p and 4p pools.
4. Timing is safe: p99 turn time comfortably below 1 second, with no overage depletion risk.
5. No crashes/exceptions/timeouts across long ladder runs.
6. Ablations show the main added modules improve performance or are removed.
7. Replay validation shows model-filtered decisions are closer to high-rated agents than heuristic-only decisions.
8. A report is generated with seeds, opponents, win rates, rank distributions, timings, and ablation tables.
9. User manually approves submission.

The proof gate can show “strong local evidence,” but it cannot mathematically prove hidden leaderboard top 3. The final decision remains manual.

## Risks and Mitigations

- Public replay overfit: use game-level validation, multiple dates, and opponent diversity.
- Runtime blowups: enforce conservative time budgets and fallback actions.
- Wrong physics: write local tests against official environment behavior.
- Overcomplex feature bloat: require ablation wins; remove features that do not help.
- Weak baselines: include multiple public agents and style bots.
- Submission packaging errors: test root-level `main.py` locally before packaging.
- Hidden meta mismatch: tune for robustness across broad styles, not one public notebook.

## Acceptance Criteria

- Project contains reproducible local tooling for downloads, training, evaluation, and packaging.
- Final `main.py` is a real NumPy-compatible adaptive agent.
- Replay pipeline can ingest public daily/top replay datasets within the 100 GB cap.
- Supervised models can be trained/exported and used from NumPy inference.
- Local reports provide evidence before any submission decision.
- No submission occurs without explicit user approval.
