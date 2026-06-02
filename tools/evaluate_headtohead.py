from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any

from kaggle_environments import make


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_agent(ref: str):
    if ref.endswith(".py") or "/" in ref:
        path = Path(ref).resolve()
        module_name = f"evaluate_headtohead_agent_{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for agent file: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ImportError(f"Failed to import agent file: {path}") from exc
        if not hasattr(module, "agent"):
            raise AttributeError(f"Agent file does not define 'agent': {path}")
        agent_fn = getattr(module, "agent")
        if not callable(agent_fn):
            raise AttributeError(f"'agent' is not callable in: {path}")
        return agent_fn
    module_name, func_name = ref.rsplit(":", 1) if ":" in ref else (ref, "agent")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = len(rows)
    errors = sum(1 for r in rows if r.get("candidate_status") != "DONE" or r.get("baseline_status") != "DONE")
    numeric_rows: list[tuple[float, float]] = []
    for row in rows:
        try:
            cand = _to_float(row.get("candidate_reward"), default=float("nan"))
            base = _to_float(row.get("baseline_reward"), default=float("nan"))
        except (TypeError, ValueError):
            continue
        if cand != cand or base != base:  # NaN check
            continue
        numeric_rows.append((cand, base))

    wins = sum(1 for cand, base in numeric_rows if cand > base)
    losses = sum(1 for cand, base in numeric_rows if cand < base)
    ties = sum(1 for cand, base in numeric_rows if cand == base)

    mean_times = [_to_float(r.get("candidate_time_mean", 0.0) or 0.0) for r in rows]
    p95_times = [_to_float(r.get("candidate_time_p95", 0.0) or 0.0) for r in rows]
    max_times = [_to_float(r.get("candidate_time_max", 0.0) or 0.0) for r in rows]
    return {
        "matches": matches,
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": ties,
        "win_rate": wins / matches if matches else 0.0,
        "errors": errors,
        "error_rate": errors / matches if matches else 0.0,
        "candidate_time_mean": statistics.mean(mean_times) if mean_times else 0.0,
        "candidate_time_p95": sorted(p95_times)[int(0.95 * (len(p95_times) - 1))] if p95_times else 0.0,
        "candidate_time_max": max(max_times) if max_times else 0.0,
    }


def timed_agent(agent_fn, timings: list[float]):
    def wrapped(obs, config=None):
        start = time.perf_counter()
        try:
            return agent_fn(obs, config)
        finally:
            timings.append(time.perf_counter() - start)

    return wrapped


def run_match(candidate_ref: str, baseline_ref: str, seed: int, order: str, episode_steps: int | None = 80) -> dict[str, Any]:
    candidate = _load_agent(candidate_ref)
    baseline = _load_agent(baseline_ref)
    candidate_times: list[float] = []
    normalized_episode_steps = episode_steps if (episode_steps is None or episode_steps > 0) else None
    configuration: dict[str, Any] = {"seed": seed}
    if normalized_episode_steps is not None:
        configuration["episodeSteps"] = normalized_episode_steps
    env = make("orbit_wars", configuration=configuration, debug=False)
    if order == "candidate_first":
        agents = [timed_agent(candidate, candidate_times), baseline]
        candidate_idx, baseline_idx = 0, 1
    else:
        agents = [baseline, timed_agent(candidate, candidate_times)]
        candidate_idx, baseline_idx = 1, 0
    env.run(agents)
    state = env.state
    return {
        "seed": seed,
        "order": order,
        "candidate_reward": state[candidate_idx].reward,
        "baseline_reward": state[baseline_idx].reward,
        "candidate_status": str(state[candidate_idx].status),
        "baseline_status": str(state[baseline_idx].status),
        "candidate_time_mean": statistics.mean(candidate_times) if candidate_times else 0.0,
        "candidate_time_p95": sorted(candidate_times)[int(0.95 * (len(candidate_times) - 1))] if candidate_times else 0.0,
        "candidate_time_max": max(candidate_times) if candidate_times else 0.0,
    }


def parse_match_specs(raw: str) -> list[tuple[int, str]]:
    specs: list[tuple[int, str]] = []
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("match spec must be '<seed>:<order>'")
        seed_raw, order = chunk.split(":", 1)
        order = order.strip()
        try:
            seed = int(seed_raw)
        except ValueError as exc:
            raise ValueError("seed must be an integer") from exc
        if order not in {"candidate_first", "candidate_second"}:
            raise ValueError("order must be 'candidate_first' or 'candidate_second'")
        specs.append((seed, order))
    return specs


def _run_match_from_tuple(args: tuple[str, str, int, str, int | None]) -> dict[str, Any]:
    candidate_ref, baseline_ref, seed, order, episode_steps = args
    return run_match(candidate_ref, baseline_ref, seed, order, episode_steps=episode_steps)


def run_selected_matches(
    candidate_ref: str,
    baseline_ref: str,
    specs: list[tuple[int, str]],
    episode_steps: int | None = 80,
    jobs: int = 1,
    runner=None,
) -> dict[str, Any]:
    if runner is None:
        runner = run_match
    if jobs <= 1 or runner is not run_match:
        rows = [runner(candidate_ref, baseline_ref, seed, order, episode_steps=episode_steps) for seed, order in specs]
    else:
        args_list = [(candidate_ref, baseline_ref, seed, order, episode_steps) for seed, order in specs]
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            rows = list(executor.map(_run_match_from_tuple, args_list))
    return {"episode_steps": episode_steps, "summary": summarize_results(rows), "matches": rows}


def run_headtohead(
    candidate_ref: str,
    baseline_ref: str,
    seeds: int,
    episode_steps: int | None = 80,
    jobs: int = 1,
) -> dict[str, Any]:
    specs = []
    for seed in range(seeds):
        specs.append((seed, "candidate_first"))
        specs.append((seed, "candidate_second"))
    return run_selected_matches(candidate_ref, baseline_ref, specs, episode_steps=episode_steps, jobs=jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="src.ow.chadman:agent")
    parser.add_argument("--baseline", default="src.ow.chadman_baseline:agent")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--matches",
        default=None,
        help="Comma-separated '<seed>:<order>' specs, where order is candidate_first or candidate_second",
    )
    parser.add_argument(
        "--steps",
        dest="episode_steps",
        type=int,
        default=80,
        help="Episode step cap per game; use 0 or negative for full environment default",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    episode_steps = args.episode_steps if args.episode_steps > 0 else None
    if args.matches:
        report = run_selected_matches(
            args.candidate,
            args.baseline,
            parse_match_specs(args.matches),
            episode_steps=episode_steps,
            jobs=args.jobs,
        )
    else:
        if args.jobs == 1:
            report = run_headtohead(args.candidate, args.baseline, args.seeds, episode_steps=episode_steps)
        else:
            report = run_headtohead(args.candidate, args.baseline, args.seeds, episode_steps=episode_steps, jobs=args.jobs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
