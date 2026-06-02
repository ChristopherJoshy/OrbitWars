# Orbit Wars Top-3-Ready Agent

Local non-RL competition stack for Kaggle Orbit Wars.

## Hard rule

No Kaggle submission is allowed until `tools/proof_gate.py` passes and the user explicitly approves the submission.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Test

_Status: planned command. Test suite files are not in this repo yet; use this once tests are implemented._

```bash
python -m pytest -q
```

## Local evaluation

_Status: planned command. Evaluation tooling is referenced for the target workflow and becomes available after `tools/evaluate_ladder.py` is implemented._

```bash
python tools/evaluate_ladder.py --config configs/default.json --candidate main.py --seeds 16 --mode 2
```

## Submission guard

_Status: planned command. Proof gate script path is reserved and should be used after `tools/proof_gate.py` is implemented._

```bash
python tools/proof_gate.py --config configs/default.json --report reports/latest_summary.json
```
