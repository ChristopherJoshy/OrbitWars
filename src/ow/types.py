from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MatchResult:
    seed: int
    mode: int
    rank: int
    score: float
    turns: int
    error: bool
    p99_turn_seconds: float
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
