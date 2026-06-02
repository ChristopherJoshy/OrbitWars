# Orbit Wars local baselines

These baseline agents are simple, legal local opponents used for evaluation and regression checks.
They are intentionally lightweight and should not be presented as top-performing competition agents.

## Attribution policy

- Keep baseline code and notes attributable to this repository.
- Do not claim external leaderboard performance for these bots.
- If reused externally, preserve provenance and mark modifications clearly.

## Included bots

- `random_agent.py`: deterministic-seeded stochastic sender that issues legal random attacks when available.
- `greedy_sample.py`: nearest affordable non-owned target capture attempt, with sun-block avoidance.
- `style_bots.py`:
  - `rush_agent`: aggressive style, up to two quick attacks.
  - `turtle_agent`: delayed, conservative single-move behavior.
  - `greedy_expand_agent`: greedy single-target expansion wrapper.
  - `swarm_agent`: multi-source random swarm with legal spend caps.
  - `idle_heavy_agent`: mostly idle periodic turtle variant.
