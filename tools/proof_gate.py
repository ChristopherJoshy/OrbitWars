from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def proof_gate_passes(report_path: Path, user_approved: bool) -> bool:
    if not user_approved or not report_path.exists():
        return False

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(report, dict):
        return False

    win_rate = _finite_float(report.get("win_rate"))
    avg_rank = _finite_float(report.get("avg_rank"))
    max_p99_turn_seconds = _finite_float(report.get("max_p99_turn_seconds"))
    error_rate = _finite_float(report.get("error_rate"))
    if any(v is None for v in (win_rate, avg_rank, max_p99_turn_seconds, error_rate)):
        return False

    assert win_rate is not None
    assert avg_rank is not None
    assert max_p99_turn_seconds is not None
    assert error_rate is not None

    return (
        win_rate >= 0.58
        and avg_rank <= 1.85
        and max_p99_turn_seconds <= 0.80
        and error_rate == 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/latest_summary.json")
    parser.add_argument("--user-approved", action="store_true")
    args = parser.parse_args()

    ok = proof_gate_passes(Path(args.report), user_approved=args.user_approved)
    if not ok:
        raise SystemExit("proof gate failed: no submission allowed")
    print("proof gate passed")


if __name__ == "__main__":
    main()
