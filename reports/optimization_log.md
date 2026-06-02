# Optimization Log

Each row records one candidate change, command, result, and decision.

| Candidate | Change | 2p win rate | 4p avg rank | p99 turn | Decision |
|---|---|---:|---:|---:|---|
| baseline-model-64 | Task12 trained validator + Task11 policy, 64-seed reference | 0.8750 | 1.1250 | 0.0158 | Keep as reference |
| baseline-model-128 | Broader 128-seed reference | 0.8906 | 1.1484 | 0.0228 | Keep as reference |
| roi-distance-v1 | Distance penalty -0.006→-0.004, early neutral ROI 0.08→0.12, enemy prod 0.07→0.09 | 0.8438 | 1.1406 | 0.0197 | Rejected: 2p win-rate regression >0.03 without compensating 4p gain |
| replay-large-v1 | Trained validator from 10 replay records / 744,911 rows at lr=1e-6 | 0.8828 | 1.1563 | 0.0176 | Rejected: 128-seed 2p and 4p both slightly regressed vs baseline; restored Task12 single-replay validator |
