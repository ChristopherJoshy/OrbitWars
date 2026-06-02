# Chadman Diagnosis-First Hybrid Design

## Goal

Turn Chad v24 (`/home/chris/Downloads/experiments/chadmain_v24.py`) into a stronger Orbit Wars production agent by first proving the simulator/planner matches the official environment, then repairing measurable defects, then adding replay-supervised value/rerank training and self-play tuning.

The target is a top-tier candidate. Top-1 remains a stretch goal; a realistic staged target is first to recover and exceed baseline Chad, then reach top-10/top-3-ready local evidence before any submission. Public leaderboard rank is not guaranteed.

## Updated User Requirements

- Chad v24 is the base agent; do not continue investing in the old greedy production policy.
- Use the user’s local GPU for training experiments.
- Run calibration/evaluation directly against baseline Chadman v24 before considering any candidate.
- Research and analyze replays, including the local replay corpus under `/home/chris/projects/leaderboard_replays`.
- Treat recent score drop evidence seriously: the latest CLI check showed `chadmain_v24.py` at public score `869.7`, lower than the earlier `929.7`, so diagnosis comes before ML.
- Use Kaggle CLI/local files, not Kaggle MCP.
- Do not submit to Kaggle unless proof gates pass and the user explicitly approves.
- For ML/RL phase, train 2-player and 4-player policies separately from top-player replay data plus self-play, then combine them with an automatic game-state switcher rather than forcing one shared policy to cover both modes.
- During self-play/training on Fedora, open a lightweight live local Pygame/SDL2 window that shows sampled matches being played, produce cinema-mode snapshots, and write richer logs so training behavior, collapse, action volume, reward/clip/KL/entropy trends, and replay examples can be inspected while runs are active.
- Treat the 24-hour first training pass as fixed wall-clock budget: 12 hours for 2P, then 12 hours for 4P, with checkpoints and stop conditions rather than blind uninterrupted training.
- If rewriting the environment in JAX for training throughput, prove it is 1:1 against the official Orbit Wars engine before trusting any learning results; target at least 10k samples/sec only after parity is demonstrated.
- Build a feature-rich tactical agent, including friendly transfer / bait-and-reinforce behavior that can make low-garrison planets look attackable, then punish enemy attacks with hidden reserves or timed reinforcements.
- Account for Kaggle host CPU/memory variance by designing final inference for worst-case hosts, not average local timing.

## Verified Facts From Source and Official Environment

These findings were checked against the local official environment and Chad source:

- Official `Planet` fields are `('id', 'owner', 'x', 'y', 'radius', 'ships', 'production')`.
- Chad v24 defines the same `Planet` field order at lines `365-366`, so planet field order is not currently a confirmed bug.
- Official `Fleet` fields are `('id', 'owner', 'x', 'y', 'angle', 'from_planet_id', 'ships')`; Chad matches at lines `368-369`.
- Official fleet speed uses `1 + (maxSpeed - 1) * (log(ships) / log(1000)) ** 1.5`; Chad matches at lines `410-415`.
- Official planet radius is carried in row column 4 and generated as `1 + log(production)` for normal planets; Chad uses `planet.radius` in orbit/static/collision planning, so radius is not a simple confirmed constant bug.
- Official replays are Kaggle episode JSON with `steps`, per-agent `observation`, actions, `remainingOverageTime`, planets, fleets, comets, team names, rewards, and statuses. One inspected local replay used module version `1.29.3` and had normal Kaggle episode structure.
- Local replay corpus exists at `/home/chris/projects/leaderboard_replays`: 20 daily folders, about 97,450 JSON files, about 400 GB extracted.

## Confirmed / Suspected Critical Chad Issues

### 1. Beam commitment context is still suspect

The exact user-reported line `5196` is not empty in the local file: it passes `planned_commitments` into `empire_state_score`. However:

- `empire_state_score` does not actually use the `planned_commitments` argument in its body.
- `rollout_beam_select` tracks beam actions in node-local structures but does not append selected beam actions into `planned_commitments`.
- `execute_actions` definitely calls `world.min_ships_to_own_by(... planned_commitments={})` at line `5240`.

So the mechanism remains real enough to prioritize: final scoring/repair can evaluate or repair a selected beam branch without the action commitments that made the branch feasible.

### 2. Fast-forward simulation is approximate

Chad’s `fast_forward_sim` schedules actions directly to target IDs by ETA and applies production before scheduled arrivals. That production-before-arrival order matches the official tick order for arriving fleets, but the simulation does not fully model:

- continuous swept collision with moving planets/comets,
- sun/out-of-bounds destruction during planned paths,
- wrong-first-planet hits,
- re-aim drift for moving targets after send-size repair,
- exact 4-player rank utility.

This can cause search leaf values to be systematically wrong even if the tactical shot planner is correct.

### 3. Overage-time budgeting is incomplete

Official observations include `remainingOverageTime`. Chad uses `config.actTimeout` to choose a soft per-turn deadline and does not read `remainingOverageTime`. If early turns burn excess time, Chad may need to tighten later-turn search budgets.

### 4. Online bandit/replay tracker may be harmful

Chad includes `_META_BANDIT` and replay tracker machinery, but several replay-prior paths were previously found no-op or weakly wired. The bandit must be ablated before adding another learned layer.

### 5. Orbit/intercept accuracy must be audited from real replays

Even though Chad’s speed formula and field order match the official spec, capture failures can still come from aim/intercept or collision modeling errors. Replay analysis should compare launched fleets against actual first-hit targets and arrival outcomes.

## Revised Priority Order

Training is not first. The corrected order is:

1. **Transplant baseline Chad without behavior changes** into the repo and reproduce its local behavior.
2. **Build Chad-vs-Chad calibration harness**: baseline Chad v24 vs candidate Chadman, both orders, fixed seed sets, plus timing/error reports.
3. **Replay format and score-drop analysis**: parse local leaderboard replay JSON, identify Chad/user episodes where available, summarize scores, actions, timeouts, launch volume, capture failures, and top-agent action patterns.
4. **Simulator/spec conformance tests**: assert official field order, speed formula, radius use, turn order, combat tie behavior, swept collision behavior, and `remainingOverageTime` availability.
5. **Planner bug fixes and ablations before ML**:
   - commitment-aware beam scoring/execution,
   - `_META_BANDIT` disabled-vs-enabled ablation,
   - overage-aware deadline policy,
   - `plan_shot` and collision/arrival audit against official replays.
6. **Only after measurable repair gains**, train replay/self-play value or rerank models.
7. **Self-play variant search** over repaired Chadman constants and model-blend weights.

## Replay Analysis Design

Use replay analysis before training and before major planner rewrites.

Inputs:

- Local replay corpus: `/home/chris/projects/leaderboard_replays`.
- Official episode-index metadata already downloaded under the project.
- Public/Hugging Face preprocessed transition datasets if useful.
- The current user submission metadata from Kaggle CLI.

Required outputs under `/home/chris/orbitwars/reports`:

- Replay corpus inventory by date, file count, size, module version, and teams.
- Chad/user episode finder where team/submission names can be matched.
- Top-player sample report: launch volume, mission timing, ships per launch, capture success, production curves.
- Chad failure report: missed captures, wrong-first-hit events, sun/out-of-bounds losses, timeouts/errors, overage usage if observable.
- Dataset selection report for supervised training that explains which games are included and why.

## Calibration Design Against Baseline Chadman

Every candidate must be measured against frozen baseline Chad v24.

Minimum reports:

- Baseline Chad v24 versus current local agent for continuity.
- Candidate Chadman versus baseline Chad v24, seeds `0..N`, both orders.
- Candidate Chadman versus public/baseline pool in 2P and 4P.
- Timing report with mean, p95, p99, max, and error rate.

Initial gates:

- Candidate must beat baseline Chad v24 head-to-head.
- Candidate must not regress timing or legal-move reliability.
- If a fix does not improve calibration, revert or quarantine it before moving on.

## Repair Work Before ML

### Commitment-aware beam/execution

Thread candidate action commitments into leaf scoring and final repair. Do not score or repair a beam branch as if its prior selected actions do not exist.

### Bandit ablation

Run baseline Chad with `_META_BANDIT` enabled and disabled over the same seed sets. Keep the bandit only if it improves robust calibration.

### Overage-aware deadline

Read `remainingOverageTime` from obs and tighten search when total overage is low. Verify no timeout/error regression.

### Official simulator audit

Create small official-env scenarios to compare Chad predictions against real next-step outcomes for:

- speed/ETA,
- production-before-arrival,
- moving planet swept collision,
- sun crossing,
- out-of-bounds loss,
- combat tie and multi-owner arrivals.

## Hybrid ML Design After Repairs

The learned model should rank Chadman search leaves/actions, not replace the planner.

The training path is split by mode:

- **2P policy/value/reranker**: train and evaluate only on 2-player states, 2-player top-player replays, and 2-player self-play leagues.
- **4P policy/value/reranker**: train and evaluate only on 4-player states, 4-player top-player replays, and 4-player self-play leagues.
- **Game-state switcher**: at inference, select the 2P or 4P model from the observed active player count / game state, with fallback to Chadman heuristic search if model loading or timing fails.

This follows the leaderboard discussion evidence that mixed 2P/4P training can destabilize RL and that separate policies plus an automatic selector are a safer engineering path.

Evidence-backed self-play defaults for the first implementation:

- **Primary algorithm**: PPO for 2P and MAPPO/IPPO-style PPO for 4P, because clipped policy-gradient updates are the most practical stable baseline for this kind of game.
- **Opponent curriculum**: snapshot pool / league-style opponent sampling, with recent and older frozen policies. Use PFSP-like or Elo-gap sampling once enough checkpoints exist; avoid single latest-self-only training.
- **Autocurriculum principle from OpenAI emergent tool-use work**: competitive self-play can create staged strategy/counter-strategy learning without hand-coding every tactic, but it must be paired with opponent diversity, capability probes, and replay inspection; do not treat reward/Elo alone as proof of useful skill.
- **Hybrid imitation evidence from arXiv:2509.15042**: use behavior cloning from rule-based/top-player demonstrations as a stability bootstrap before RL, but verify with a no-BC ablation because the paper is BC+DQN in a 2D shooter rather than PPO/MAPPO in Orbit Wars.
- **Centralized-training evidence from arXiv:2304.13004**: keep centralized critic / centralized training as the default for RTS-like multi-agent coordination, especially for 4P MAPPO-style training, but do not treat centralized control alone as a recipe for leaderboard strength.
- **Self-play survey evidence from arXiv:2408.01072**: treat the policy population, opponent-sampling matrix, and meta-strategy/rating updates as first-class training objects; add matchup-matrix and approximate exploitability/NashConv-style diagnostics where practical because Elo alone can mislead in non-transitive games.
- **Bootstrap**: imitation / behavior-cloning pretraining from top-player replay data when labels are clean enough, then self-play fine-tuning.
- **Optional after baseline**: PSRO / league exploiters / PBT-style hyperparameter search. These are robustness upgrades, not the first 24-hour run.
- **Do not trust** any RL result unless it improves held-out head-to-head tests against scripted baselines, frozen historical policies, baseline Chadman, and held-out replay-derived states.

First 24-hour training schedule:

- **Hours 0-12: 2P** — validate JAX/original env parity, action masks, dense rewards, then train PPO against a snapshot queue. Stop early to fix env/reward if win rate vs simple baselines or baseline Chadman does not improve.
- **Hours 12-24: 4P** — reuse validated engine and observation stack, add 4P rewards/rank utility and opponent/team permutations, train MAPPO/IPPO-style self-play against a snapshot/lineup pool. Stop early if entropy collapses, clip fraction rises, KL spikes, or role/order sensitivity appears.
- Save checkpoints frequently enough to build the opponent pool and make postmortems possible.
- Add targeted capability probes before scaling long runs: expansion, target denial, focus-fire/anti-focus, bait-and-reinforce, survival under pressure, and endgame conversion. This converts emergent self-play behavior into measurable Orbit Wars skills instead of trusting aggregate reward alone.
- Add ablations from the added papers: no imitation vs BC warm-start vs BC+RL; latest-self only vs snapshot pool vs league/PFSP sampling; 2P-only transfer vs explicit 4P MAPPO/IPPO phase; centralized critic vs IPPO fallback if 4P destabilizes.

Training features should include explicit leaf/state deltas:

- production by player before/after simulated horizon,
- ships by player before/after horizon,
- owned planet counts and high-production planet counts,
- home/frontier vulnerability,
- planned action count and total committed ships,
- expected capture/retention outcomes,
- enemy response exposure,
- phase, player count, remaining turns,
- timing/collision confidence.

Additional tactical features / behaviors to expose to Chadman and future learned rerankers:

- friendly transfers between owned planets;
- bait planets with intentionally low visible garrison plus timed friendly reinforcement;
- reserve-shuffle patterns that move ships across many owned planets to misdirect enemy targeting;
- enemy waste accounting: enemy ships committed to doomed attacks, overkills, sun/comet losses, and attacks into reinforced planets;
- threat-window features: time until enemy impact, time until friendly rescue, production before impact, and whether the planet is deliberately baited;
- multi-source and delayed-wave commitments by target;
- leader/weakest-player pressure and 4P rank-risk features;
- host-latency-safe feature subsets for degraded inference mode.

Labels should be game-level or trajectory-level, not row-random:

- winner/rank outcome from replays,
- future production/ship advantage after a horizon,
- self-play outcome for candidate action bundles,
- pairwise preference between candidate leaves from the same state.

Data split must be by episode/date/team bucket to avoid leakage.

Training may use the local GPU, but final inference must export to NumPy-compatible arrays or compact tables.

JAX environment rewrite requirements:

- Implement only as a training-speed engine; final Kaggle inference must not depend on JAX/GPU.
- Keep the JAX engine pure and static-shape friendly (`jit`/`vmap`/`scan`) so throughput claims are benchmarkable.
- Prove 1:1 parity against the official engine with fixed action-sequence rollouts, edge-case tests for swept fleet/planet collision, sun crossing, out-of-bounds, comet lifecycle/collisions, production-before-arrival, combat ties, simultaneous launches, and terminal/rank rewards.
- Benchmark only after parity; report warmup, compile time, steady-state SPS, batch size, device, and host-device transfer costs separately.
- A JAX engine that reaches 10k SPS but fails parity is rejected for training.

Training observability is a required deliverable, not an afterthought:

- log PPO/self-play metrics such as reward, win rate vs frozen opponents, value loss, policy loss, entropy, KL, clip fraction, explained variance, SPS, and checkpoint Elo/head-to-head summaries;
- run a lightweight local Pygame/SDL2 live-viewer process on Fedora that streams sampled 2P/4P self-play matches while training continues, keeping the high-frequency display off the GPU training path;
- show important live values in the Pygame overlay: current mode (2P/4P), checkpoint/opponent IDs, reward components, action mask validity, top action probabilities, chosen action, value estimate, advantage, entropy, KL, clip fraction, win rate vs pool, Elo trend, SPS, env/render queue lag, GPU/CPU utilization, invalid-action count, timeout/desync warnings, and per-player production/ship/planet totals;
- use the GPU for model forward/backward/update work and CPU workers for environment simulation, replay sampling, logging, and viewer rendering;
- generate periodic cinema-mode HTML rollouts for 2P and 4P snapshots using the MatthewWHuang visualizer integration;
- write structured JSONL/CSV logs that can be analyzed without reading terminal scrollback;
- keep separate run directories for 2P, 4P, switcher, and ablations.

Kaggle host-variance safeguards for final inference:

- Assume per-turn CPU time can vary by 3-4x across hosts and memory can differ materially between episodes.
- Keep a fast-safe inference path that can drop model/search depth if timing budget or overage is low.
- Limit threads, preallocate arrays, avoid dynamic allocation and heavy logging in the hot path, warm/load all artifacts at startup, and keep final runtime CPU-only NumPy-compatible.
- Validate turn-time p99 under artificially reduced budgets before submission.

## Self-Play and Variant Search

After simulator repairs and first value/rerank experiments, run local tournaments over Chadman variants.

Tune:

- defense margins,
- send-size ratios,
- leader/weakest-player pressure,
- beam width/depth/horizon,
- candidate caps,
- value-model blend weight,
- bandit on/off or constrained arms,
- bait/reinforcement thresholds and friendly-transfer volume.

Use held-out seeds and mixed opponent pools to avoid overfitting to baseline Chad.

## Feature-First Tactical Upgrade Design

Before starting RL, finish a non-ML tactical feature pass in Chadman. The goal is to give both the hand-coded planner and future learned rerankers the missing strategic primitives that the user requested.

Priority order:

1. **Friendly-transfer economy**: owned-to-owned ship transfers, rear-to-front reinforcement chains, hub-and-spoke routing, and idle-ship pressure accounting.
2. **Bait-and-reinforce**: deliberately keep selected frontier planets apparently weak only when timed friendly reinforcement can arrive before or shortly after likely enemy impact; score enemy ships wasted into reinforced planets.
3. **Reserve-shuffle / deception**: redistribute ships among owned planets so enemy heuristic targeting sees changing weak points, while our true defense remains timeline-safe.
4. **Enemy-waste accounting**: score overkills, doomed attacks, attacks into reinforced/bait planets, sun/comet/path losses, and enemy ships delayed away from high-value targets.
5. **4P rank-aware utility**: favor actions that improve expected rank, contain the leader without kingmaking, eliminate the weakest only when rank-positive, and avoid becoming an easy coalition target.
6. **Counterfactual enemy-intent features**: infer likely enemy target windows from enemy source surplus, angle/timing, and projected value; penalize our plans that collapse under obvious next-turn responses.
7. **Capability probes**: add fixed local scenarios for expansion, transfer logistics, bait/reinforce, target denial, focus-fire/anti-focus, survival under pressure, and endgame conversion.

This feature pass must stay CPU/NumPy-safe, deadline-aware, and measurable against frozen Chad v24 before any RL training starts. If a feature cannot improve calibration or capability probes without timing regression, quarantine it behind a config/feature flag rather than relying on it in the production path.

## Verification Gates

No Kaggle submission until all required gates pass:

1. Unit/smoke tests pass.
2. Official simulator conformance tests pass or known approximations are documented.
3. Replay analysis report exists for selected training/evaluation data.
4. Candidate beats frozen baseline Chad v24 in calibration runs.
5. Candidate improves broad 2P/4P local ladders.
6. Turn-time p99 and error rate satisfy proof thresholds.
7. Submission package is built only by guarded tooling.
8. User explicitly approves the submission.

## Implementation Stages

1. Transplant Chad v24 as frozen baseline and candidate path.
2. Build Chad-vs-Chadman calibration runner.
3. Build replay inventory/analyzer for local leaderboard replay JSON.
4. Add official simulator conformance tests.
5. Fix/ablate commitment context, bandit, overage timing, and shot/collision prediction.
6. Finish the feature-first tactical upgrade pass: friendly transfers, bait-and-reinforce, reserve shuffle, enemy-waste accounting, rank-aware utility, counterfactual enemy-intent features, and capability probes.
7. Recalibrate against baseline Chad and public/baseline pools.
8. Build supervised dataset only from verified replay/self-play sources.
9. Train separate local-GPU 2P and 4P value/rerank/RL models and export NumPy inference artifacts.
10. If needed for throughput, build and validate the 1:1 JAX training engine before trusting self-play results.
11. Run the first 24-hour training pass: 12 hours 2P, then 12 hours 4P, with live Pygame viewer, structured logs, periodic cinema snapshots, and checkpoint gates.
12. Integrate the 2P/4P model switcher into Chadman search with timing caps and heuristic fallback.
13. Run self-play variant search, training-observability checks, and final proof gates.

## Risks

- The score gap is large; top-1 is a stretch even after major repairs.
- Large replay corpus requires careful sampling and metadata filtering.
- A learned reranker trained on top of an incorrect simulator will learn brittle corrections; simulator audit must come first.
- Public score can move or decay; local calibration is necessary but not sufficient.
- Timing regressions can erase strategic gains.
- Kaggle host variance means an agent with acceptable mean local timing can still fail validation; final timing gates must use large safety margins.
- RL can learn reward hacks or brittle self-play conventions; held-out opponents, replay states, and rule-conformance checks are mandatory.

## Approval Check

This revised spec makes diagnosis, replay analysis, and baseline Chad calibration prerequisites for ML. Implementation should not begin until the user approves this revised written spec.
