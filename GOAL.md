# Orbit Wars Recovery Goal

## Primary objective

Build a Kaggle-ready Orbit Wars agent that can realistically compete for a top-10 position, but do not spend Kaggle/cloud quota or submit until local proof gates pass.

## Current truth

- The codebase is recoverable and backed by tests, RL tooling, phase gates, JAX work, metrics, and evaluation scripts.
- The latest saved model is not leaderboard-ready yet: the final Phase 3 checkpoint passed local self-play gates but failed validation versus the frozen Chadman baseline.
- The most recent repair direction showed a better learning signal with dense rewards, exact/macro actions, and alternating 2P train seats, but strict Chadman gate proof was not completed before processes were stopped.

## What to do next

1. Restore the environment and install dependencies with `.venv/bin/python -m pip install -r requirements.txt`.
2. Run the focused local gate suite before changing training code:
   `.venv/bin/python -m pytest tests/test_rl_action_space.py tests/test_rl_policy.py tests/test_rl_rewards_metrics.py tests/test_rl_viewer.py tests/test_rl_train_phase1.py tests/test_rl_phase_gates.py tests/test_rl_jax_policy.py -q`
3. Resume from the latest promising pure-RL repair idea, not the old dead long runs:
   - dense per-decision reward-to-go,
   - exact capture and safe macro action candidates,
   - `--train-seat alternating` for 2P frozen-opponent canaries,
   - short fixed-seed canaries before any long run.
4. Validate against `src.ow.chadman_baseline:agent` as a frozen black-box opponent.
5. Only after 2P and 4P local Chadman validation are healthy, continue JAX parity/mock training.
6. Only after all local and JAX proof gates pass, use Kaggle/cloud for larger self-play training.

## What not to do

- Do not edit `src/ow/chadman_baseline.py`; it is the frozen protected baseline.
- Do not use Chadman imitation, distillation, supervised transplant, reranking, or hybrid handoff unless the project goal is explicitly changed.
- Do not submit to Kaggle or spend cloud quota before local proof gates pass.
- Do not keep dead long runs alive when metrics show 0 wins, flat asset margin, tiny action margins, and low throughput.
- Do not delete old `reports/` runs; create fresh run directories and use `reports/latest_rl_run.txt` for the active pointer.

## Fast recovery command pattern

Use short canaries first. Example shape, adjusted to the checkpoint being tested:

```bash
.venv/bin/python tools/train_rl_phase1.py \
  --mode 2p \
  --initial-policy <checkpoint.npz> \
  --frozen-opponent-agent src.ow.chadman_baseline:agent \
  --run-dir reports/<fresh-run-name> \
  --updates 200 \
  --episodes 8 \
  --episode-steps 80 \
  --eval-every 5 \
  --checkpoint-every 10 \
  --policy-lr 0.01 \
  --entropy-coef 0.03 \
  --ppo-epochs 8 \
  --terminal-reward-scale 5.0 \
  --asset-margin-reward-scale 0.75 \
  --dense-reward-scale 2.0 \
  --reward-gamma 0.99 \
  --losing-sample-weight 4 \
  --winning-sample-weight 8 \
  --gate-advance-win-streak 1 \
  --train-seat alternating \
  --no-viewer \
  --seed 7
```

## Success gates before cloud/submission

- 2P frozen-Chadman canary: clean deterministic wins from both seats, no ties, no invalid actions, ship waste within threshold.
- 2P local gate: 100% strict clean wins for 30 minutes.
- 4P local gate: same strict clean 30-minute gate.
- Mixed Phase 3: both 2P and 4P satisfy the strict gate.
- JAX environment parity and local mock JAX training pass.

## Source of truth

Read `AGENTS.md` first. If `README.md` conflicts with executable code, tests, or `AGENTS.md`, trust executable code and `AGENTS.md`.
