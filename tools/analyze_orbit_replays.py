from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterator


def summarize_replay(replay: dict[str, Any]) -> dict[str, Any]:
    info = replay.get("info") if isinstance(replay, dict) else {}
    info = info if isinstance(info, dict) else {}

    teams = info.get("TeamNames")
    if not isinstance(teams, list):
        teams = info.get("teamNames")
    if not isinstance(teams, list):
        teams = []

    rewards = replay.get("rewards") if isinstance(replay, dict) else []
    if not isinstance(rewards, list):
        rewards = []

    statuses = replay.get("statuses") if isinstance(replay, dict) else []
    if not isinstance(statuses, list):
        statuses = []

    steps = replay.get("steps") if isinstance(replay, dict) else []
    if not isinstance(steps, list):
        steps = []

    final_non_empty_step = None
    for step in reversed(steps):
        if isinstance(step, list) and step:
            final_non_empty_step = step
            break

    if not rewards and final_non_empty_step is not None:
        candidate_rewards = [row.get("reward") if isinstance(row, dict) else None for row in final_non_empty_step]
        if any(x is not None for x in candidate_rewards):
            rewards = candidate_rewards

    if not statuses and final_non_empty_step is not None:
        candidate_statuses = [row.get("status") if isinstance(row, dict) else None for row in final_non_empty_step]
        if any(x is not None for x in candidate_statuses):
            statuses = candidate_statuses

    max_step_players = 0
    for step in steps:
        if isinstance(step, list):
            max_step_players = max(max_step_players, len(step))

    n_players = max(len(teams), len(rewards), len(statuses), max_step_players)
    actions_by_player = [0] * n_players
    ships_launched_by_player = [0] * n_players
    min_overage: list[float | None] = [None] * n_players

    for step in steps:
        if not isinstance(step, list):
            continue
        for player_idx in range(min(n_players, len(step))):
            row = step[player_idx]
            if not isinstance(row, dict):
                continue

            obs = row.get("observation")
            if isinstance(obs, dict):
                overage = obs.get("remainingOverageTime")
                if isinstance(overage, (int, float)):
                    overage_value = float(overage)
                    current = min_overage[player_idx]
                    min_overage[player_idx] = (
                        overage_value if current is None else min(current, overage_value)
                    )

            action = row.get("action")
            if not isinstance(action, list):
                continue
            for move in action:
                if isinstance(move, list) and len(move) == 3:
                    actions_by_player[player_idx] += 1
                    try:
                        ships_launched_by_player[player_idx] += int(move[2])
                    except (TypeError, ValueError):
                        pass

    return {
        "episode_id": info.get("EpisodeId"),
        "module_version": replay.get("module_version") if isinstance(replay, dict) else None,
        "teams": teams,
        "rewards": rewards,
        "statuses": statuses,
        "steps": len(steps),
        "actions_by_player": actions_by_player,
        "ships_launched_by_player": ships_launched_by_player,
        "min_remaining_overage_by_player": [x if x is not None else 0.0 for x in min_overage],
        "remaining_overage_observed_by_player": [x is not None for x in min_overage],
    }


def iter_replay_paths(root: Path, limit: int | None = None) -> Iterator[Path]:
    if hasattr(root, "rglob") and not isinstance(root, Path):
        paths = root.rglob("*.json")
        if limit is not None:
            paths = itertools.islice(paths, limit)
        for path in paths:
            yield path
        return

    count = 0
    root_path = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue
            if limit is not None and count >= limit:
                return
            yield Path(dirpath) / filename
            count += 1


def analyze_corpus(root: Path, limit: int | None = None) -> dict[str, Any]:
    count = 0
    sample: list[dict[str, Any]] = []
    by_date: dict[str, dict[str, Any]] = {}

    for path in iter_replay_paths(root, limit=limit):
        date_key = path.parent.name
        count += 1
        bucket = by_date.setdefault(date_key, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        try:
            size = path.stat().st_size
            bucket["bytes"] += size

            if len(sample) >= 200:
                continue

            with path.open("r", encoding="utf-8") as f:
                replay = json.load(f)
            row = summarize_replay(replay)
            row["path"] = str(path)
        except Exception as exc:  # noqa: BLE001
            row = {"path": str(path), "error": str(exc)}
        if len(sample) < 200:
            sample.append(row)

    return {
        "root": str(root),
        "count": count,
        "by_date": by_date,
        "sample": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze orbit replay corpus")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/chris/projects/leaderboard_replays"),
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_corpus(args.root, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
