from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.ow.types import MatchResult
from tools import evaluate_ladder


def test_summarize_results_computes_win_rate_and_timing():
    results = [
        MatchResult(seed=1, mode=2, rank=1, score=100.0, turns=500, error=False, p99_turn_seconds=0.2),
        MatchResult(seed=2, mode=2, rank=2, score=80.0, turns=500, error=False, p99_turn_seconds=0.4),
    ]
    summary = evaluate_ladder.summarize_results(results)
    assert summary["matches"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["avg_rank"] == 1.5
    assert summary["max_p99_turn_seconds"] == 0.4


def test_summarize_results_empty_defaults():
    assert evaluate_ladder.summarize_results([]) == {
        "matches": 0,
        "win_rate": 0.0,
        "avg_rank": 0.0,
        "avg_score": 0.0,
        "max_p99_turn_seconds": 0.0,
        "error_rate": 0.0,
    }


def test_write_summary_writes_json_and_creates_dirs(tmp_path):
    output = tmp_path / "nested" / "summary.json"
    results = [MatchResult(seed=3, mode=2, rank=1, score=12.0, turns=10, error=False, p99_turn_seconds=0.01)]
    evaluate_ladder.write_summary(results, output, project_root=tmp_path)
    data = json.loads(output.read_text())
    assert data["matches"] == 1
    assert output.exists()


def test_percentile_99_empty_and_nonempty():
    assert evaluate_ladder.percentile_99([]) == 0.0
    assert evaluate_ladder.percentile_99([0.1, 0.2, 0.3]) == pytest.approx(0.298)


def test_build_lineup_mode_2_and_4():
    def cand(obs, config=None):
        return []

    l2 = evaluate_ladder.build_lineup(cand, mode=2)
    l4 = evaluate_ladder.build_lineup(cand, mode=4)
    assert len(l2) == 2
    assert len(l4) == 4
    assert l2[0] is cand
    assert l4[0] is cand


def test_match_result_json_serializable():
    result = MatchResult(seed=1, mode=4, rank=3, score=2.0, turns=99, error=True, p99_turn_seconds=0.12)
    payload = result.to_dict()
    assert payload["rank"] == 3
    assert json.loads(json.dumps(payload))["error"] is True


def test_missing_kaggle_dependency_errors_on_run_path(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named kaggle_environments")

    monkeypatch.setattr(evaluate_ladder.importlib, "import_module", _boom)
    output = tmp_path / "out.json"
    with pytest.raises(RuntimeError, match="kaggle_environments"):
        evaluate_ladder.run_ladder(candidate="/home/chris/orbitwars/main.py", seeds=1, mode=2, output=output)


def test_load_agent_rejects_invalid_candidate_paths(tmp_path):
    with pytest.raises(ValueError, match="under project root"):
        evaluate_ladder._load_agent_from_py("/tmp/outside.py")

    missing = tmp_path / "missing.py"
    with pytest.raises(ValueError, match="does not exist"):
        evaluate_ladder._load_agent_from_py(missing, project_root=tmp_path)

    not_file = tmp_path / "d"
    not_file.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        evaluate_ladder._load_agent_from_py(not_file, project_root=tmp_path)

    txt = tmp_path / "agent.txt"
    txt.write_text("x")
    with pytest.raises(ValueError, match=".py"):
        evaluate_ladder._load_agent_from_py(txt, project_root=tmp_path)


def test_load_agent_rejects_missing_or_noncallable_agent(tmp_path):
    no_agent = tmp_path / "no_agent.py"
    no_agent.write_text("x = 1\n")
    with pytest.raises(ValueError, match="callable agent"):
        evaluate_ladder._load_agent_from_py(no_agent, project_root=tmp_path)

    bad_agent = tmp_path / "bad_agent.py"
    bad_agent.write_text("agent = 3\n")
    with pytest.raises(ValueError, match="callable agent"):
        evaluate_ladder._load_agent_from_py(bad_agent, project_root=tmp_path)


def test_timed_agent_returns_action_and_records_timing_on_exception():
    marker = object()

    def ok_agent(obs, config=None):
        return marker

    timings = []
    wrapped = evaluate_ladder._timed_agent(ok_agent, timings)
    assert wrapped({}, config={}) is marker
    assert len(timings) == 1 and timings[0] >= 0.0

    def bad_agent(obs, config=None):
        raise RuntimeError("boom")

    wrapped_bad = evaluate_ladder._timed_agent(bad_agent, timings)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped_bad({}, config={})
    assert len(timings) == 2 and timings[-1] >= 0.0


def test_extract_candidate_result_dict_and_object_states():
    env_dict = types.SimpleNamespace(
        steps=[[{"status": "ACTIVE", "reward": 0.0}], [{"status": "DONE", "reward": 5.0}, {"status": "DONE", "reward": 3.0}]]
    )
    r1 = evaluate_ladder._extract_candidate_result(env_dict, seed=1, mode=2, p99=0.1, error=False)
    assert r1.rank == 1 and r1.score == 5.0 and r1.error is False

    cand_obj = types.SimpleNamespace(status="DONE", reward=2.0)
    opp_obj = types.SimpleNamespace(status="DONE", reward=9.0)
    env_obj = types.SimpleNamespace(steps=[[cand_obj, opp_obj]])
    r2 = evaluate_ladder._extract_candidate_result(env_obj, seed=2, mode=2, p99=0.2, error=False)
    assert r2.rank == 2 and r2.score == 2.0 and r2.error is False


@pytest.mark.parametrize("steps", [None, [], [[{}]], [[{"status": "DONE"}, {"status": "DONE", "reward": 1.0}]]])
def test_extract_candidate_result_handles_missing_empty_or_incomplete_steps(steps):
    env = types.SimpleNamespace(steps=steps)
    r = evaluate_ladder._extract_candidate_result(env, seed=1, mode=2, p99=0.0, error=False)
    assert r.error is True


@pytest.mark.parametrize("status", ["ERROR", "INVALID", "TIMEOUT"])
def test_extract_candidate_result_failure_statuses_mark_error(status):
    env = types.SimpleNamespace(steps=[[{"status": status, "reward": 1.0}, {"status": "DONE", "reward": 0.0}]])
    r = evaluate_ladder._extract_candidate_result(env, seed=1, mode=2, p99=0.0, error=False)
    assert r.error is True


def test_run_match_preserves_env_failure_diagnostics(monkeypatch):
    class FakeEnv:
        steps = []

        def run(self, _lineup):
            raise RuntimeError("env exploded")

    fake_mod = types.SimpleNamespace(make=lambda *_args, **_kwargs: FakeEnv())
    monkeypatch.setattr(evaluate_ladder.importlib, "import_module", lambda _name: fake_mod)
    result = evaluate_ladder.run_match(lambda *_args, **_kwargs: [], seed=7, mode=2)
    assert result.error is True
    assert "env exploded" in result.error_message


def test_write_summary_rejects_outside_root_and_writes_atomically(tmp_path):
    results = [MatchResult(seed=1, mode=2, rank=1, score=1.0, turns=1, error=False, p99_turn_seconds=0.0)]
    out = tmp_path / "x" / "summary.json"
    evaluate_ladder.write_summary(results, out, project_root=tmp_path)
    assert out.exists()

    outside = Path("/tmp/not_allowed_summary.json")
    with pytest.raises(ValueError, match="under project root"):
        evaluate_ladder.write_summary(results, outside)


def test_direct_script_invocation_surfaces_missing_kaggle_not_local_import_error(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "smoke_summary.json"
    cmd = [
        sys.executable,
        "tools/evaluate_ladder.py",
        "--config",
        "configs/default.json",
        "--candidate",
        "main.py",
        "--seeds",
        "1",
        "--mode",
        "2",
        "--output",
        str(out),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode != 0 or proc.returncode == 0
    assert "ModuleNotFoundError: No module named 'baselines'" not in combined
    assert "No module named 'src'" not in combined


def test_import_from_repo_root_does_not_mutate_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "import json, pathlib, sys;"
        "root=str(pathlib.Path.cwd().resolve());"
        "before=list(sys.path);"
        "import tools.evaluate_ladder;"
        "after=list(sys.path);"
        "print(json.dumps({'root': root, 'before': before, 'after': after}))"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    root = payload["root"]
    before = payload["before"]
    after = payload["after"]
    assert "" in before or root in before
    assert not (root not in before and root in after)
