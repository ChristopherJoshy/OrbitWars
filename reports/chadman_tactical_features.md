# Chadman Tactical Features Report

## Implemented

- Friendly-transfer helper and transfer missions.
- Bait-and-reinforce scoring.
- Reserve-shuffle hub selection.
- Enemy-waste scoring.
- 4P rank-pressure utility.
- Counterfactual enemy-target intent features.
- Capability probe runner.

## Verification

- Focused tests:
  - Command: `.venv/bin/python -m pytest tests/test_chadman_tactical_features.py tests/test_chadman_feature_probes.py tests/test_feature_probe_runner.py -q`
  - Result: `21 passed in 0.13s`
- Full suite:
  - Command: `.venv/bin/python -m pytest -q`
  - Result: `285 passed, 3 warnings in 28.45s`
- Probe report:
  - Command: `.venv/bin/python tools/run_chadman_feature_probes.py --output reports/chadman_feature_probes.json`
  - Output file: `reports/chadman_feature_probes.json` (exists)
  - Summary: `failed=0, passed=4, total=4`
- Calibration report:
  - Command: `.venv/bin/python tools/evaluate_headtohead.py --candidate src.ow.chadman:agent --baseline src.ow.chadman_baseline:agent --seeds 8 --output reports/chadman_tactical_features_vs_baseline_8seeds.json`
  - Output file: `reports/chadman_tactical_features_vs_baseline_8seeds.json` (exists)
  - Summary: `matches=16, candidate_wins=2, candidate_losses=14, ties=0, win_rate=0.125, error_rate=0.0, errors=0`
  - Timing: `candidate_time_mean=0.1363131217784662, candidate_time_p95=0.2847593409996989, candidate_time_max=0.44866435899984936`
- Cinema artifact:
  - Command: `.venv/bin/python tools/render_cinema_match.py main.py src/ow/chadman_baseline.py --seed 0 --steps 120 --out reports/cinema/chadman_tactical_features_seed0.html`
  - Output file: `reports/cinema/chadman_tactical_features_seed0.html` (exists)
  - Payload check: file contains cinema-mode payload markers (for example, `cinemaMode`, `cinemaBtn`, and `.cinema-hype`).

## Decision

Keep features enabled only if error rate is zero and timing remains safe. If calibration is strategically worse, keep helper/probe scaffolding but gate production mission integration behind constants or disable before RL.

Current calibration has `error_rate=0.0` and timing appears within a safe envelope, but strategic performance in this 8-seed calibration is worse than baseline (`win_rate=0.125`); retain helper/probe scaffolding and gate or disable production mission integration before RL.
