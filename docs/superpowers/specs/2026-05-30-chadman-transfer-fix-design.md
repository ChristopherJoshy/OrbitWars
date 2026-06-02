# Chadman Transfer Fix Design

## Context

The tactical feature pass passed tests but regressed strategically: `reports/chadman_tactical_features_vs_baseline_8seeds.json` recorded 2 wins and 14 losses against `src.ow.chadman_baseline:agent` with zero runtime errors. Ablations and transfer logging implicated `build_transfer_missions` as the main regression. The bad behavior was repeated small owned-to-owned transfers, often 12-30 ships, including back-and-forth shuffles that consumed attack budget and interrupted offense.

The user explicitly rejected disabling transfers and asked to keep the feature while fixing it. Top-1 replay research downloaded Isaiah @ Tufa Labs replays under `/tmp/opencode/orbitwars-top1-replays/isaiah_top1/`. Those replays show transfers can be strong, but successful top-1 transfer behavior is mostly late, large, and strategic; losing examples and Chadman’s regression are early/mid small-packet logistics.

## Goal

Keep transfer behavior enabled, but make it demand-driven and anti-churn so it no longer sends small routine shuffles that steal offensive tempo.

## Non-goals

- Do not run RL, self-play training, Kaggle submissions, or 100-game loops.
- Do not edit `src/ow/chadman_baseline.py`.
- Do not blindly copy Isaiah’s absolute packet sizes or global transfer rate.
- Do not remove transfer helpers/tests/probes.

## Design

### Transfer classes

Transfer missions should be allowed only when they satisfy one of three roles:

1. **Emergency reinforcement** — destination has a real defensive need before or near arrival. This can allow smaller packets, but only when the transfer fills a concrete deficit and arrives before the fall/threat window.
2. **Attack staging** — destination can use the received ships for a concrete near-term attack better than the source can. This is a future extension point and should be conservative in the first implementation.
3. **Late consolidation** — late-game or sufficiently mature empire state; source is safe/interior, destination is hub/frontier/high-reach, and packet size is large enough to matter.

Everything else should score zero.

### Hard gates

Before scoring a routine transfer:

- Reject opening/early routine transfers unless emergency demand exists.
- Reject tiny routine transfers by replacing fixed `TRANSFER_MIN_SURPLUS = 12` as the effective threshold with a dynamic minimum based on source surplus and total friendly strength.
- Reject destination demand already saturated by in-flight friendly commitments.
- Reject reverse-flow churn: if a friendly fleet is already traveling from destination back to source, do not send source to destination unless emergency.
- Reject sources with a clear attractive attack/neutral capture option, so transfer does not steal offensive tempo.

### Demand model

Add small pure helpers near existing transfer helpers in `src/ow/chadman.py`:

- `transfer_dynamic_min_send(src, dst, world, policy, available, emergency=False)` returns a minimum meaningful packet size.
- `incoming_friendly_transfer_ships(world, target_id, horizon)` estimates friendly ships already committed to a destination within a horizon.
- `has_reverse_friendly_flow(world, src_id, dst_id, horizon=24)` detects friendly in-flight fleets from `dst_id` back toward `src_id`.
- `source_has_near_attack_opportunity(src, world, policy, available)` conservatively detects when source should attack/capture instead of transfer.
- `transfer_destination_demand(src, dst, world, policy, planned_commitments, turns, available, transfer_hub_id)` returns a structured demand row such as `{"kind": "emergency"|"late_consolidation", "ships": int, "value": float}` or `None`.

The existing `friendly_transfer_value` should remain as a compatibility helper for direct tests, but `build_transfer_missions` should rely on the demand helper and dynamic minimum before emitting `Mission(kind="transfer")`.

### Scoring

Transfer scoring should tie value to marginal demand, not just destination production/frontier/hub. A simple first version is:

```text
score = demand_value / (send + eta_penalty + source_opportunity_cost + churn_penalty)
```

Where `send` is the demand-backed amount, capped by available budget/inventory, and cannot be below the dynamic minimum except for emergency demand.

### Verification

Unit and feature-probe tests must prove:

- Early 12-30 ship routine transfer is rejected without emergency demand.
- Emergency transfer before fall/threat remains allowed.
- Late large interior-to-hub/frontier consolidation is allowed.
- Reverse in-flight friendly flow blocks non-emergency transfer.
- Existing in-flight friendly commitments saturate destination demand and block over-transfer.
- Source with a nearby attractive attack/neutral capture option does not transfer.

After tests pass, run focused tests and a small deterministic calibration on the known bad seeds. The calibration target is not to start training; it is to verify that transfer-enabled behavior is no worse than the no-transfer ablation on regression seeds before any future broader tuning.

## User approval

The user replied: "try everything until wrks". Treat this as approval to proceed autonomously through this design, implementation plan, TDD implementation, and deterministic non-training verification while preserving the no-RL/no-submission and baseline-freeze constraints.
