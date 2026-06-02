from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from tools.proof_gate import proof_gate_passes


def build_submission(output: Path, report: Path, user_approved: bool) -> Path:
    if not proof_gate_passes(report, user_approved=user_approved):
        raise SystemExit("proof gate failed: refusing to build submission")

    output.unlink(missing_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        tar.add("main.py", arcname="main.py")
        tar.add("src", arcname="src")
        model_path = Path("models/shot_validator.npz")
        if model_path.exists():
            tar.add(model_path, arcname="models/shot_validator.npz")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission.tar.gz")
    parser.add_argument("--report", default="reports/latest_summary.json")
    parser.add_argument("--user-approved", action="store_true")
    args = parser.parse_args()
    print(build_submission(Path(args.output), Path(args.report), args.user_approved))


if __name__ == "__main__":
    main()
