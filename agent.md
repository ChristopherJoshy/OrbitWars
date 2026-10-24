# Agent Handoff Notes

Start with `AGENTS.md`. This lowercase file exists so agents that look for `agent.md` still find the important recovery instructions.

## Must follow

- Use `.venv/bin/python` for repo commands.
- Keep work pure RL unless the user explicitly changes the strategy.
- Do not edit `src/ow/chadman_baseline.py`.
- Do not use Kaggle/cloud/submission until local proof gates pass.
- Keep training headless with `--no-viewer` unless the user asks for the viewer.
- Do not delete old `reports/` runs.

## Current recovery priority

1. Re-run focused tests after transfer.
2. Continue from the dense-reward + exact/macro-action + alternating-seat repair path.
3. Use short frozen-Chadman canaries before long runs.
4. Promote to longer 2P/4P/mixed gates only after canaries show clean wins.
5. Preserve the project remote: `https://github.com/ChristopherJoshy/OrbitWars.git`.

## Important files

- `AGENTS.md` — detailed non-negotiable workflow.
- `GOAL.md` — current recovery goal and next steps.
- `tools/train_rl_phase1.py` — main RL trainer.
- `src/ow/rl/action_space.py` — exact/macro candidate action logic.
- `src/ow/rl/phase_gates.py` — strict local phase gate logic.
- `reports/latest_rl_run.txt` — active/latest run pointer.
