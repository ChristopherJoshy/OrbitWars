# Orbit Wars Agent Notes

## Non-negotiable workflow

- Use `.venv/bin/python` for repo commands; dependencies are in `requirements.txt`.
- Do not touch Kaggle/cloud until all local gates pass; protect Kaggle quota.
- Current goal is pure RL only: no Chadman imitation, no hybrid, no reranker, no supervised transplant.
- Do not edit `src/ow/chadman_baseline.py`; it is a protected frozen baseline.
- Keep Pygame off for throughput unless the user explicitly asks: `--no-viewer` is the training default.
- Do not delete old `reports/` runs; use fresh run dirs and `reports/latest_rl_run.txt` to find the active run.
- No commits/submissions unless explicitly requested.

## High-value commands

- Install/update deps: `.venv/bin/python -m pip install -r requirements.txt`
- Full tests: `.venv/bin/python -m pytest -q`
- Focused RL/JAX suite used as the local iteration gate:
  `.venv/bin/python -m pytest tests/test_rl_action_space.py tests/test_rl_policy.py tests/test_rl_rewards_metrics.py tests/test_rl_viewer.py tests/test_rl_train_phase1.py tests/test_rl_phase_gates.py tests/test_rl_jax_policy.py -q`
- Submission packaging is guarded by `tools/proof_gate.py`; `tools/build_submission.py` refuses to build unless proof gate passes with `--user-approved`.

## Current local RL pipeline

- Main trainer: `tools/train_rl_phase1.py` supports `--mode 2p`, `--mode 4p`, and `--mode mixed`.
- Active/last run pointer: `reports/latest_rl_run.txt`; PID is in `<run>/pid.txt`; metrics source of truth is `<run>/metrics.jsonl`.
- Phase gate implementation: `src/ow/rl/phase_gates.py` uses latest contiguous `gate_eval` rows, `player0_rank == 1.0`, duration `>=1800s`, and `ship_waste_loss=max(wasted_ships,sun_loss_ships)` currently `<20`.
- User’s latest wording is “20 or less ships loss”; current code is stricter (`<20`). Change only with TDD if needed.
- Ties are losses. Phase 2 must not start until Phase 1 passes the 30-minute gate; Phase 3 must not start until Phase 2 passes.
- Latest Phase 1 strategy includes deterministic strict gate, latest-opponent alignment, deterministic stream weighting, gate streak curriculum, streak-sensitive LR, action-margin/argmax-flip diagnostics, and clean-gate rollback.

## Local-before-cloud gates

1. Phase 1: 2-player self-play reaches 100% gate win rate for 30 minutes with ship loss within threshold.
2. Phase 2: 4-player self-play reaches the same gate.
3. Phase 3: merge 2P+4P models and mixed-train until both modes satisfy the same gate.
4. JAX env must be a 1:1 clone of the original Orbit Wars engine; parity tests must pass before JAX training.
5. Run local mock JAX training using data/checkpoints from local NumPy training; only then start Kaggle/cloud.

## Kaggle/cloud plan to preserve

- Cloud starts only after all local gates and JAX/local mock gates pass.
- Week 1: S1 TPU v5e-8 planned as a 9h allocation but stop at 8h40m to avoid session cutoff; warm-start from Phase 3, bulk rollouts, checkpoint every 30min. S2 same 8h40m safety limit, continue from S1 with league opponents. TPU quota used: ~18h.
- Week 2: S3 GPU T4×2 for 12h continue from TPU checkpoint; S4 GPU T4×2 for 12h final push/eval vs diverse opponents; S5 GPU T4×2 for 2h export, submission, leaderboard check.
- Total planned compute: 18h TPU + 26h GPU = 44h across 2 weeks.
- TPU↔GPU checkpoint handoff must go through `np.savez`/`.npz` so weights are hardware-agnostic.
- Test between every cloud session before continuing. If anything fails, fix locally before spending more Kaggle time.
- TPU throughput target is 100k–200k env steps/s only if the JAX env/train loop is fully vectorized/on-device and benchmarked; at 8h40m this is ~3.12B–6.24B steps/session, so two TPU sessions can exceed 1B steps comfortably if the benchmark holds.
- During cloud self-play, when the agent holds 100% win rate for 10 straight minutes, strengthen opponents by refreshing/adding the learned policy to the opponent league; do this as self-play snapshot/league curriculum, not imitation.

## Repo quirks

- `README.md` is partly stale/non-RL; trust executable code/tests over README prose when they conflict.
- `src/ow/rl/metrics.py` writes JSONL and dynamically rewrites CSV when late gate fields appear; JSONL is still the safest monitoring source.
- Read-only dependency clone: `.slim/clonedeps/repos/MatthewWHuang__orbit-wars/`; inspect only, never edit.
