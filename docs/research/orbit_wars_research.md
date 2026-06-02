# Orbit Wars Research Preservation Log

## Required Kaggle discussions

### Discussion 697725
- Takeaway: RL can work but is expensive and unstable without a very fast simulator and careful reward shaping.
- Takeaway: inductive bias and reliable environment modeling matter more than brute-force training.
- Takeaway: AI-generated strategy should be audited against real match evidence.

### Discussion 701894
- Takeaway: daily episode datasets provide a practical replay stream for offline mining.
- Takeaway: maintain date/source manifests to track replay provenance and drift.

### Discussion 697413
- Takeaway: Bovard top-10% replay datasets are high-value supervised signal.
- Takeaway: prioritize higher-rated replay slices for action quality filtering.

## Official mechanics (README-preserved)
- Board: circular arena with a central sun hazard.
- Planet tuple fields: include ownership/ships/production and motion metadata.
- Fleet tuple fields: include source/target ownership context, ships, and travel state.
- Production ordering: production is handled in official turn resolution order before downstream outcomes.
- Fleet speed: official launch-size speed behavior must be matched exactly in local physics.
- Comets: temporary capturable objects with lifetime constraints.
- Combat: arrivals resolve by ship-count contest and ownership transition rules.
- Game length: fixed 500 turns.
- Score: final score is determined by official end-state scoring (ship-equivalent control/value).

## Public-agent technique themes to implement
- Hellburner: strong opening expansion sequencing and practical tactical pressure.
- Roman (LB-max): high-tempo exploit timing, pressure balancing, and robust fallback heuristics.
- Emanuel: compact heuristic prioritization with good defense/ROI discipline.
- Yash: simulation/value/search blending under strict compute budgets.
- Konbu: ML-assisted validation/filtering of candidate actions.

## Local operating constraints
- Replay storage cap: 100 GB maximum.
- Hardware envelope: 6 GB VRAM GPU and 32 GB RAM CPU.
- Final inference path: NumPy-only submission-safe inference.
- Submission governance: no Kaggle submission before proof gate passes and user approval is explicit.
