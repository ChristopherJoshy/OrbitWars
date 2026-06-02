from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
import uuid
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_root_on_path(project_root: Path = ROOT) -> None:
    root_str = str(Path(project_root).resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import numpy as np

if __package__ in (None, ""):
    _ensure_project_root_on_path()
from src.ow.types import MatchResult


def percentile_99(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(np.asarray(samples, dtype=np.float64), 99))


def summarize_results(results: list[MatchResult]) -> dict:
    if not results:
        return {
            "matches": 0,
            "win_rate": 0.0,
            "avg_rank": 0.0,
            "avg_score": 0.0,
            "max_p99_turn_seconds": 0.0,
            "error_rate": 0.0,
        }
    n = len(results)
    return {
        "matches": n,
        "win_rate": sum(1 for r in results if r.rank == 1) / n,
        "avg_rank": sum(r.rank for r in results) / n,
        "avg_score": sum(r.score for r in results) / n,
        "max_p99_turn_seconds": max(r.p99_turn_seconds for r in results),
        "error_rate": sum(1 for r in results if r.error) / n,
    }


def _resolve_under_root(path: str | Path, project_root: Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = (project_root / target).resolve()
    else:
        target = target.resolve()
    try:
        target.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Path must be under project root {project_root}: {target}") from exc
    return target


def write_summary(results: list[MatchResult], output: str | Path, project_root: Path = ROOT) -> dict:
    summary = summarize_results(results)
    output_path = _resolve_under_root(output, Path(project_root).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(output_path)
    return summary


def _load_agent_from_py(candidate: str | Path, project_root: Path = ROOT):
    root = Path(project_root).resolve()
    path = _resolve_under_root(candidate, root)
    if not path.exists():
        raise ValueError(f"Candidate path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Candidate path must be a regular file: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Candidate must be a .py file: {path}")
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load candidate file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise ValueError(f"Candidate file must expose callable agent(obs, config=None): {path}")
    return fn


def build_lineup(candidate_agent, mode: int):
    from baselines.greedy_sample import agent as greedy_agent
    from baselines.random_agent import agent as random_agent
    from baselines.style_bots import rush_agent

    if mode == 2:
        return [candidate_agent, greedy_agent]
    if mode == 4:
        return [candidate_agent, greedy_agent, random_agent, rush_agent]
    raise ValueError(f"Unsupported mode {mode}; expected 2 or 4")


def _timed_agent(agent_fn, timings: list[float]):
    def _wrapped(obs, config=None):
        start = time.perf_counter()
        try:
            return agent_fn(obs, config=config)
        finally:
            timings.append(time.perf_counter() - start)

    return _wrapped


def _state_value(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _extract_candidate_result(env, seed: int, mode: int, p99: float, error: bool) -> MatchResult:
    if error:
        return MatchResult(seed=seed, mode=mode, rank=mode, score=0.0, turns=0, error=True, p99_turn_seconds=p99, error_message="match marked as error")
    steps = getattr(env, "steps", None)
    turns = len(steps or [])
    if not steps:
        return MatchResult(seed=seed, mode=mode, rank=mode, score=0.0, turns=turns, error=True, p99_turn_seconds=p99, error_message="missing or empty env.steps")
    final = steps[-1] or []
    if not final:
        return MatchResult(seed=seed, mode=mode, rank=mode, score=0.0, turns=turns, error=True, p99_turn_seconds=p99, error_message="missing final step state")
    cand = final[0]
    reward = _state_value(cand, "reward", None)
    status = str(_state_value(cand, "status", "")).upper()
    if reward is None:
        return MatchResult(seed=seed, mode=mode, rank=mode, score=0.0, turns=turns, error=True, p99_turn_seconds=p99, error_message="candidate reward missing")
    cand_score = float(reward)
    if not isfinite(cand_score):
        return MatchResult(seed=seed, mode=mode, rank=mode, score=0.0, turns=turns, error=True, p99_turn_seconds=p99, error_message="candidate reward not finite")
    rewards = [_state_value(a, "reward", None) for a in final]
    clean_rewards = [float(r) if r is not None and isfinite(float(r)) else float("-inf") for r in rewards]
    rank = 1 + sum(1 for r in clean_rewards[1:] if r > cand_score)
    err = status in {"ERROR", "INVALID", "TIMEOUT", "CRASHED"}
    err_msg = f"failure status: {status}" if err else ""
    return MatchResult(seed=seed, mode=mode, rank=rank, score=cand_score, turns=turns, error=err, p99_turn_seconds=p99, error_message=err_msg)


def run_match(candidate_agent, seed: int, mode: int) -> MatchResult:
    try:
        kaggle_env = importlib.import_module("kaggle_environments")
    except ModuleNotFoundError as exc:
        raise RuntimeError("kaggle_environments is required to run ladder evaluation") from exc
    timings: list[float] = []
    wrapped = _timed_agent(candidate_agent, timings)
    lineup = build_lineup(wrapped, mode=mode)
    env = kaggle_env.make("orbit_wars", configuration={"seed": int(seed)})
    try:
        env.run(lineup)
        return _extract_candidate_result(env, seed=seed, mode=mode, p99=percentile_99(timings), error=False)
    except Exception as exc:
        return MatchResult(seed=seed, mode=mode, rank=mode, score=0.0, turns=0, error=True, p99_turn_seconds=percentile_99(timings), error_message=str(exc))


def run_ladder(candidate: str | Path, seeds: int, mode: int, output: str | Path, config: str | Path | None = None):
    _ = config
    candidate_agent = _load_agent_from_py(candidate)
    results = [run_match(candidate_agent, seed=s, mode=mode) for s in range(int(seeds))]
    write_summary(results, output)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--mode", type=int, choices=[2, 4], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        run_ladder(args.candidate, args.seeds, args.mode, args.output, config=args.config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(args.output)


if __name__ == "__main__":
    main()
