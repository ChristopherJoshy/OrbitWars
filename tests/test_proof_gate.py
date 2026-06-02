import json
import tarfile
from pathlib import Path

import pytest

from tools.build_submission import build_submission
from tools.proof_gate import proof_gate_passes


def _write_report(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _strong_report() -> dict[str, float]:
    return {
        "win_rate": 0.9,
        "avg_rank": 1.5,
        "max_p99_turn_seconds": 0.5,
        "error_rate": 0.0,
    }


def test_proof_gate_rejects_missing_manual_approval(tmp_path: Path):
    report = tmp_path / "summary.json"
    _write_report(report, _strong_report())
    assert not proof_gate_passes(report, user_approved=False)


def test_proof_gate_accepts_strong_report_with_approval(tmp_path: Path):
    report = tmp_path / "summary.json"
    _write_report(report, _strong_report())
    assert proof_gate_passes(report, user_approved=True)


@pytest.mark.parametrize(
    "mutations",
    [
        {"win_rate": 0.57},
        {"avg_rank": 1.86},
        {"max_p99_turn_seconds": 0.81},
        {"error_rate": 0.001},
        {"win_rate": "not-a-number"},
        {"win_rate": "nan"},
        {"avg_rank": "inf"},
        {"max_p99_turn_seconds": None},
        {"error_rate": "-inf"},
        {"win_rate": None},
        {"avg_rank": None},
        {"max_p99_turn_seconds": None},
        {"error_rate": None},
        {"win_rate": True, "avg_rank": True, "max_p99_turn_seconds": False, "error_rate": False},
    ],
)
def test_proof_gate_rejects_weak_or_malformed_metrics(tmp_path: Path, mutations: dict):
    report = tmp_path / "summary.json"
    payload = _strong_report()
    payload.update(mutations)
    _write_report(report, payload)
    assert not proof_gate_passes(report, user_approved=True)


def test_proof_gate_rejects_missing_report_file(tmp_path: Path):
    assert not proof_gate_passes(tmp_path / "missing.json", user_approved=True)


def test_proof_gate_rejects_missing_metric_keys(tmp_path: Path):
    report = tmp_path / "summary.json"
    _write_report(report, {"win_rate": 0.9})
    assert not proof_gate_passes(report, user_approved=True)


def test_proof_gate_cli_without_approval_exits_nonzero(tmp_path: Path):
    report = tmp_path / "summary.json"
    _write_report(report, _strong_report())
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "tools.proof_gate", "--report", str(report)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert proc.returncode != 0
    assert "proof gate failed: no submission allowed" in (proc.stderr + proc.stdout)


def test_build_submission_direct_script_without_approval_reaches_gate_not_import_error(tmp_path: Path):
    report = tmp_path / "summary.json"
    _write_report(report, _strong_report())
    output = tmp_path / "submission.tar.gz"

    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "tools/build_submission.py",
            "--report",
            str(report),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    combined = proc.stderr + proc.stdout
    assert proc.returncode != 0
    assert "proof gate failed: refusing to build submission" in combined
    assert "ModuleNotFoundError" not in combined
    assert not output.exists()


def test_build_submission_refuses_without_approval_and_creates_no_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = tmp_path / "summary.json"
    _write_report(report, _strong_report())

    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text("x = 1\n", encoding="utf-8")

    output = tmp_path / "submission.tar.gz"
    with pytest.raises(SystemExit, match="proof gate failed: refusing to build submission"):
        build_submission(output=output, report=report, user_approved=False)
    assert not output.exists()


def test_build_submission_with_approval_creates_expected_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = tmp_path / "summary.json"
    _write_report(report, _strong_report())

    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shot_validator.npz").write_bytes(b"npz")

    output = tmp_path / "submission.tar.gz"
    built = build_submission(output=output, report=report, user_approved=True)
    assert built == output
    assert output.exists()

    with tarfile.open(output, "r:gz") as tar:
        names = set(tar.getnames())

    assert "main.py" in names
    assert "src" in names
    assert "src/agent.py" in names
    assert "models/shot_validator.npz" in names
