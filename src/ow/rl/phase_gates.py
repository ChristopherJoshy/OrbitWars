from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class PhaseGateResult:
    satisfied: bool
    reason: str
    window_seconds: float = 0.0
    episodes: int = 0


def ship_waste_loss(row: Mapping[str, Any]) -> int:
    wasted = max(0, int(row.get("wasted_ships", 0) or 0))
    sun = max(0, int(row.get("sun_loss_ships", 0) or 0))
    return max(wasted, sun)


def _is_strict_clean_win(row: Mapping[str, Any], phase_mode: str, max_ship_waste: int) -> bool:
    return (
        str(row.get("phase_mode", "")) == str(phase_mode)
        and float(row.get("player0_rank", 0.0) or 0.0) == 1.0
        and ship_waste_loss(row) <= int(max_ship_waste)
    )


def phase_gate_satisfied(
    rows: Iterable[Mapping[str, Any]],
    phase_mode: str,
    min_duration_seconds: float = 1800,
    max_ship_waste: int = 20,
) -> PhaseGateResult:
    phase_rows = [row for row in rows if str(row.get("phase_mode", "")) == str(phase_mode)]
    gate_rows = [row for row in phase_rows if bool(row.get("gate_eval", False))]
    if gate_rows:
        phase_rows = gate_rows
    if phase_rows and "gate_opponent_version" in phase_rows[-1]:
        latest_gate_opponent_version = phase_rows[-1].get("gate_opponent_version")
        phase_rows = [row for row in phase_rows if row.get("gate_opponent_version") == latest_gate_opponent_version]
    if not phase_rows:
        return PhaseGateResult(False, "no_rows")
    if any("wall_time" not in row for row in phase_rows):
        return PhaseGateResult(False, "missing_wall_time")

    suffix: list[Mapping[str, Any]] = []
    for row in reversed(phase_rows):
        if not _is_strict_clean_win(row, phase_mode=phase_mode, max_ship_waste=max_ship_waste):
            break
        suffix.append(row)
    suffix.reverse()
    if not suffix:
        return PhaseGateResult(False, "latest_row_not_clean_strict_win")

    window_seconds = float(suffix[-1]["wall_time"]) - float(suffix[0]["wall_time"])
    if window_seconds >= float(min_duration_seconds):
        return PhaseGateResult(True, "satisfied", window_seconds=window_seconds, episodes=len(suffix))
    return PhaseGateResult(False, "duration_too_short", window_seconds=window_seconds, episodes=len(suffix))
