# Pure RL Self-Play Design

## Status

Approved by the user in chat on 2026-05-30. This document records the agreed design before implementation planning.

Do not commit this file unless the user explicitly asks for a commit.

## Goal

Build toward a pure reinforcement-learning self-play Orbit Wars agent. The current Chadman deterministic baseline gate is now diagnostic, not a hard blocker for starting local RL.

The immediate objective is iteration 1 today: start the first local pure-RL self-play training iteration with live local visibility and strong logs. A later iteration 2 will move to Kaggle 2xT4 training with a verified JAX environment.

## Non-goals and constraints

- Do not remove Chadman tactical features as the final solution. Diagnostic ablations are allowed only to isolate root causes.
- Do not edit `src/ow/chadman_baseline.py`.
- Do not submit to Kaggle unless proof gates pass and the user explicitly approves.
- The local 16/16 Chadman gate is no longer a hard blocker for starting pure RL. It remains a diagnostic baseline/evaluation report so blockers are not hidden.
- Do not rely on JAX for iteration 1 local training. JAX is for Kaggle-scale iteration 2.
- Do not treat the iteration-1 wall-clock target as a guarantee of model strength; it is a phase budget for producing and reporting local trained checkpoints, and phases may be extended if the model has not started winning all games.
- Run independent tests/checks/calibrations in parallel whenever possible.

## Current project evidence

Repository research found the current project is useful for deterministic Chadman repair and evaluation, but not yet a complete RL stack.

Existing useful assets:

- Frozen baseline: `src/ow/chadman_baseline.py`.
- Current candidate: `src/ow/chadman.py`.
- Local gate harness: `tools/evaluate_headtohead.py` runs both seat orders per seed.
- Structured evaluation tools: `tools/evaluate_headtohead.py`, `tools/evaluate_ladder.py`, `tools/proof_gate.py`.
- Tests for Chadman tactical features, feature probes, conformance, budgeting, commitments, transplant, cinema rendering, and evaluation.
- Replay analysis and parsing tools: `tools/analyze_orbit_replays.py`, `tools/parse_replays.py`.
- Existing NumPy supervised/replay model pipeline: `tools/train_models.py`, `src/ow/agent_core.py`, `src/ow/features.py`.
- Post-run visualization: `tools/render_cinema_match.py` plus the cloned `MatthewWHuang/orbit-wars` visualizer under `.slim/clonedeps/repos/MatthewWHuang__orbit-wars/`.

Missing assets that must be added before serious pure-RL self-play:

- PPO/IPPO/MAPPO training loop.
- Self-play opponent checkpoint pool and sampling policy.
- RL checkpoint format and model export path.
- Mandatory local live Pygame game viewer backed by NumPy state for sampled training games.
- Visible training log printer for terminal/Kaggle logs section.
- JSONL/CSV training metrics files.
- JAX Orbit Wars environment and parity tests for iteration 2.
- RL/JAX dependencies in project requirements.

## Phase A: local Chadman diagnostic gate

The Chadman gate remains a diagnostic target for measuring deterministic-agent strength:

```text
candidate: src.ow.chadman:agent
baseline:  src.ow.chadman_baseline:agent
seeds:     0..7
orders:    candidate_first and candidate_second
target:    16 wins / 16 matches
errors:    0
timing:    safe enough for local and future Kaggle-host variance
```

The likely immediate root-cause path is still systematic debugging of the remaining Chadman regressions. Existing diagnostics indicate transfer-only disablement does not explain all losses; a strong current hypothesis is shared planning/execution behavior around live `planned_commitments` passed into `execute_actions`, plus remaining transfer/tactical tuning. Fixes must be root-cause-backed and test-first.

Features must remain enabled in the deterministic agent. Temporary monkeypatch/ablation diagnostics may be used to locate root causes, but not as final feature removal. Because the user approved moving to RL now, this gate no longer blocks iteration-1 RL implementation or local training start.

## Iteration 1: local pure RL self-play

Run local pure RL self-play now, using the Chadman/frozen-baseline gate as a diagnostic evaluation rather than a prerequisite.

Training order:

1. Train **2P solo self-play RL** first.
2. Gate the 2P checkpoint against the frozen baseline using the exact same 16/16 local gate.
3. Train **4P solo self-play RL** second.
4. Merge 2P and 4P into one agent or mode-switching agent.
5. Gate the merged 2P/4P agent against the frozen baseline using the exact same 16/16 local gate.
6. Train the merged agent against versions of itself with alternating 2P and 4P batches so it learns strategy changes and coordination/adaptation across modes.

The current frozen-baseline 16/16 gate is a 2-player gate. The 4P solo phase should also get mode-specific 4P diagnostics once the 4P self-play/evaluation harness exists, but those diagnostics do not replace the exact 16/16 frozen-baseline gate requested by the user.

Algorithm defaults:

- 2P: PPO self-play with clipped updates, value baseline, entropy bonus, gradient clipping, and checkpoint opponent pool.
- 4P: MAPPO-style PPO with centralized critic if implementation can be kept clean and verified; otherwise IPPO/PPO fallback.
- Merged alternating loop: same PPO-family update discipline with checkpoint pools and mode-conditioned observations/actions.

PPO is the first default because clipped policy updates, entropy control, value baselines, and opponent checkpoints reduce collapse risk compared with unconstrained policy gradients. If tests or implementation constraints make MAPPO too slow or risky for today, IPPO/PPO is the fallback for the first iteration.

Iteration-1 local observability:

- Mandatory live Pygame game viewer, backed by NumPy game state, showing sampled training games like the actual Orbit Wars game.
- The Pygame viewer must show overlay counters for iteration/update number, episode count, wins, losses, and recent win rate.
- The local Pygame viewer must be visually understandable and close to the original Orbit Wars viewer: dark board, glowing central sun, orbit boundary, original player colors, neutral gray planets, centered ship counts, production labels, fleet arrows with ship counts, and readable training/status panels.
- The top of the local viewer must identify our agent by color, labeling player 0 as `OUR AGENT` and showing the matching player color swatch.
- The default local viewer should be minimal/readable on older screens: compact top HUD, fullscreen option, minimal panel clutter, and burn-in protection that shifts static HUD elements every 30 minutes during long runs.
- The viewer status must include seed, update, episode, wins/losses, recent win rate, reward, sun-loss ships, wasted ships, invalid actions, repeated-seed count, and checkpoint/update id when available.
- If the viewer is confusing or does not look like the original game, stop training and fix the viewer before continuing local visual training.
- Visible terminal/log-section summaries every fixed number of updates.
- JSONL and CSV metrics written to run directories.
- Frequent checkpoints and best-checkpoint tracking.
- Periodic replay/cinema snapshots when useful and not disruptive.
- 2P phase-1 seed curriculum: start from a deterministic random seed stream; when our player-0 agent wins an episode, sample a new random seed for the next episode; when it loses, repeat the same seed until it wins.

Iteration-1 18-hour phase budget:

- Phase 1: 2P-only pure self-play RL, target 6 hours.
- Phase 2: 4P-only pure self-play RL, target 6 hours.
- Phase 3: merged 2P+4P agent self-play RL, target 6 hours.
- Treat 6 hours per phase as a training budget, not a hard stop when learning is still improving.
- If a phase has not started winning all games or the win-rate curve is still materially improving, the phase may be extended before moving on.
- Do not skip gates or verification to fill the budget.
- If the budget expires or is extended, save the best checkpoint and report metrics honestly.

## Iteration 2: Kaggle pure RL self-play

Only after local RL produces successful and good results, move to Kaggle training.

Kaggle structure:

- Two separate Kaggle sessions.
- Each session uses 2xT4 GPU for 24 hours.
- Pure self-play only.
- No GUI on Kaggle: no Pygame, no browser visualization, and no live renderer.
- Compare checkpoints locally before trusting or integrating them.

Kaggle observability:

- Logs only in Kaggle runtime; visual game playback is local-only.
- Human-readable summaries streamed to the visible Kaggle notebook/logs section, not just JSONL files.
- JSONL and CSV files saved for detailed analysis.
- Frequent checkpoint summaries printed visibly.
- Loud warnings for collapse, desync, invalid-action spikes, SPS drops, parity warnings, and bad reward/win-rate divergence.

Example visible log format:

```text
[2P PPO step=12800 ckpt=003 opp=001] win=0.62 reward=1.38 entropy=0.91 kl=0.014 clip=0.18 invalid=0 sps=8421 best=ckpt_003
```

## JAX environment for Kaggle iteration

JAX is required for Kaggle-scale iteration 2, not for local iteration 1.

Targets:

- Minimum acceptable throughput: 10k samples/sec.
- Stretch target: 20k samples/sec.
- Must be as close to a 1:1 clone of the original Orbit Wars environment as possible before trusting Kaggle training.

Required parity tests:

- Fixed seeded rollouts.
- Fixed action-sequence rollouts.
- Planet orbit movement.
- Fleet speed and ETA.
- Production timing.
- Combat and tie behavior.
- Simultaneous arrivals.
- Collision and first-hit behavior.
- Sun and out-of-bounds loss.
- Comet lifecycle/collisions if relevant to the official environment state.
- Terminal rewards and 2P/4P ranks.

Benchmarking requirements:

- Report compile time separately from steady-state runtime.
- Report device, batch size, warmup, host-device transfer costs, and true samples/sec.
- Reject high SPS results if parity tests fail.

## Reward shaping

Reward shaping must be dense enough to avoid sparse-reward failure, while preserving final win/rank pressure.

Positive rewards:

- Winning.
- Improving rank in 4P.
- Gaining planets.
- Gaining production.
- Successful captures.
- Defending valuable planets.
- Efficient use of transfers, reinforcement, and bait behavior.
- Efficient ship use and favorable exchange ratios.

Penalties:

- Per-step existence/idling tax for simply existing without useful action.
- Penalty for every production point opponents gain.
- Losing planets.
- Missing clear capture opportunities.
- Invalid actions.
- Wasting ships.
- Repeated failed attacks.
- Timeouts or slow decisions.
- Brittle repeated mistakes.

Reward safeguards:

- Penalties should be heavy enough to discourage idling and repeated mistakes.
- Penalties must not be so dominant that the model panic-sends ships or becomes afraid to act.
- Monitor entropy, action volume, invalid-action count, reward components, and held-out win rate to detect collapse or reward hacking.

## Training logs and metrics

Both local and Kaggle runs must log:

- mode: 2P, 4P, or alternating merged mode;
- global step/update;
- checkpoint ID and opponent checkpoint ID;
- win rate vs sampled opponent pool;
- reward total and reward components;
- policy loss, value loss, entropy, KL, clip fraction, explained variance;
- invalid action count/rate;
- action volume and no-op/idling rate;
- planet count/production/ship totals by player;
- opponent production gained;
- planets lost/gained;
- missed capture opportunity counts where detectable;
- SPS and environment/render queue lag;
- timing/timeout/desync/parity warnings;
- best checkpoint and latest gate result.

Logs must be both machine-readable and visible to the operator:

- JSONL/CSV for analysis.
- Readable periodic summaries printed to terminal or Kaggle logs section.

## Stop conditions

Stop or pause for diagnosis if:

- Unit or conformance tests fail.
- Diagnostic gate metrics regress unexpectedly; this pauses for review but does not automatically block all RL work.
- RL entropy collapses toward zero too early.
- Invalid actions spike.
- Reward improves while held-out win rate worsens.
- Opponent-production penalties induce panic-sending or suicidal attacks.
- Direct mistake penalties (sun loss, wasted ships, invalids, oversend, repeated failures) dominate terminal win-rate or collapse action volume.
- SPS drops below viable thresholds.
- JAX parity fails in iteration 2.
- Training reaches wall-clock budget; save best checkpoint and report status.

## Implementation sequence after this spec

1. Write a concrete implementation plan.
2. Treat the latest Chadman 16/16 result as a diagnostic baseline, not a blocker.
3. Run relevant tests and calibrations in parallel where independent.
4. Implement local pure-RL infrastructure immediately:
   - PPO/IPPO/MAPPO components,
   - self-play checkpoint pool,
   - reward shaping,
   - direct logged mistake penalties for sun hits, out-of-bounds/no-hit shots, wasted attacks, invalid actions, oversend, idling, lost planets, and opponent production gains,
   - mandatory local live Pygame viewer backed by NumPy state,
   - visible logs and files,
   - checkpoint/gate integration.
5. Run local 2P first for a 6-hour target budget, then local 4P for a 6-hour target budget, then merged alternating self-play for a 6-hour target budget; extend a phase if the model has not started winning all games or is still clearly improving.
6. After local success, design and implement the JAX parity environment for Kaggle iteration 2.

## Approval record

The user approved the design direction in chat and asked to continue using subagents for speed. The user emphasized:

- pure RL self-play may start before the 16/16 Chadman gate, with that gate retained as diagnostic evidence;
- 2P solo first, then 4P solo, then merged alternating self-play;
- mandatory local live Pygame game view backed by NumPy state;
- Kaggle uses logs only, with no Pygame or GUI;
- Kaggle visible logs, not just JSONL;
- JAX only for Kaggle-scale training;
- 10k-20k SPS JAX env target with 1:1 parity checks;
- explicit penalties for idling/existence and for every opponent production point gained.

## Important iteration-1 update

For now, focus only on iteration 1. The iteration-1 local training budget is 18 hours total: 6 hours for 2P-only training, 6 hours for 4P-only training, and 6 hours for merged 2P+4P self-play. A phase may be extended if the model has not started winning all games or if metrics show it is still learning. Do not reduce or remove the requested features.

## Gate-skip approval update

The user approved starting pure RL now instead of continuing to treat the Chadman 16/16 gate as a hard blocker. The current deterministic gate failures must still be reported honestly, but RL implementation and local training can proceed. The local Pygame viewer is mandatory and must visibly show iteration/update number and win counts while training.
