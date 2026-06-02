# Final Proof Report

## Unit tests

Command: `.venv/bin/python -m pytest -q`

Result: `219 passed in 5.15s`

## 2-player ladder

Command: `.venv/bin/python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 256 --mode 2 --output reports/final_2p_256.json`

Report: `reports/final_2p_256.json`

- Matches: 256
- Win rate: 0.8828125
- Average rank: 1.1171875
- Average score: 0.765625
- Error rate: 0.0
- Max p99 turn time: 0.017644962595950345 seconds

## 4-player ladder

Command: `.venv/bin/python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 256 --mode 4 --output reports/final_4p_256.json`

Report: `reports/final_4p_256.json`

- Matches: 256
- Win rate: 0.828125
- Average rank: 1.171875
- Average score: 0.65625
- Error rate: 0.0
- Max p99 turn time: 0.013387579298578202 seconds

## Conservative proof-gate summary

Report: `reports/latest_summary.json`

- Conservative win rate: 0.828125
- Conservative average rank: 1.171875
- Conservative max p99 turn time: 0.017644962595950345 seconds
- Conservative error rate: 0.0

## Timing

Maximum accepted p99 turn time: 0.80 seconds.

Observed maximum p99 turn time: 0.017644962595950345 seconds.

## Submission decision

No Kaggle submission has been made.

This proof report is local evidence against the current local baseline/style pool. It does not guarantee hidden leaderboard top-3 placement. Submission requires explicit user approval after reviewing this file and the final ladder JSON reports.
