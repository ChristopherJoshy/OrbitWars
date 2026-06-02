"""Helpers for replay download quota checks."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable
import re


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to binary GiB rounded to 6 decimals."""
    gib = num_bytes / float(1024 ** 3)
    return round(gib, 6)


def would_exceed_cap(current_bytes: int, next_bytes: int, cap_gb: float) -> bool:
    """Return True when adding next_bytes crosses the cap in bytes."""
    cap_bytes = int(cap_gb * 1024 ** 3)
    total_bytes = current_bytes + next_bytes
    return total_bytes > cap_bytes


@dataclass(frozen=True)
class ReplayDatasetEntry:
    source: str
    local_path: str
    size_bytes: int
    discussion_id: str


DEFAULT_SOURCES = [
    ("bovard/orbit-wars-top10-episodes-2026-05-04", "697413"),
    ("kaggle/orbit-wars-episodes-index", "701894"),
]


def _parse_size_to_bytes(size_text: str) -> int:
    t = size_text.strip().upper().replace(" ", "")
    if re.fullmatch(r"\d+", t):
        return int(t)
    mult = (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1))
    for unit, m in mult:
        if t.endswith(unit):
            return int(float(t[: -len(unit)]) * m)
    raise ValueError(f"Unrecognized size string: {size_text}")


def parse_kaggle_files_table(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("Next Page Token =")]
    if not lines or len(lines) == 1:
        return []
    out = []
    reader = csv.DictReader(lines)
    for row in reader:
        if not row:
            continue
        name = (row.get("name") or "").strip()
        size_text = (row.get("size") or "").strip()
        if not name or not size_text:
            continue
        try:
            size_bytes = _parse_size_to_bytes(size_text)
        except ValueError:
            continue
        out.append({"name": name, "size_bytes": size_bytes})
    return out


def _extract_next_page_token(text: str) -> str | None:
    for line in text.splitlines():
        ln = line.strip()
        if ln.startswith("Next Page Token ="):
            token = ln.split("=", 1)[1].strip()
            return token or None
    return None


def _safe_local_artifact_path(output_dir: Path, artifact_name: str) -> Path | None:
    p = Path(artifact_name)
    if p.is_absolute():
        return None
    candidate = (output_dir / p).resolve()
    base = output_dir.resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _find_downloaded_artifact_path(output_dir: Path, artifact_name: str, preferred_path: Path) -> Path | None:
    if preferred_path.exists():
        return preferred_path
    flattened = _safe_local_artifact_path(output_dir, Path(artifact_name).name)
    if flattened is not None and flattened.exists():
        return flattened
    return None


def _run_or_raise(cmd: list[str], run_command: Callable | None = None):
    runner = run_command or subprocess.run
    try:
        return runner(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Missing kaggle executable in PATH") from exc


def download_replays(
    cap_gb: float,
    max_new_gb: float,
    output_dir: Path,
    manifest_path: Path,
    run_command: Callable | None = None,
    sources=DEFAULT_SOURCES,
    max_files: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_bytes = 0
    if manifest_path.exists():
        current_bytes = sum(int(x.get("size_bytes", 0)) for x in json.loads(manifest_path.read_text(encoding="utf-8")))
    new_bytes = 0
    downloaded_files = 0
    max_new_bytes = int(max_new_gb * 1024**3)
    for dataset, discussion_id in sources:
        if max_files is not None and downloaded_files >= max_files:
            break
        next_token = None
        while True:
            if max_files is not None and downloaded_files >= max_files:
                break
            cmd = ["kaggle", "datasets", "files", "-d", dataset, "--csv", "--page-size", "100"]
            if next_token:
                cmd.extend(["--page-token", next_token])
            res = _run_or_raise(cmd, run_command=run_command)
            for item in parse_kaggle_files_table(res.stdout):
                name = item["name"]
                size = item["size_bytes"]
                if not name.endswith((".json", ".tar", ".tar.gz", ".tgz", ".zip")):
                    continue
                local_path = _safe_local_artifact_path(output_dir, name)
                if local_path is None:
                    continue
                if new_bytes + size > max_new_bytes:
                    continue
                if would_exceed_cap(current_bytes=current_bytes + new_bytes, next_bytes=size, cap_gb=cap_gb):
                    continue
                _run_or_raise(["kaggle", "datasets", "download", "-d", dataset, "-f", name, "-p", str(output_dir), "--force", "--unzip"], run_command=run_command)
                actual_local_path = _find_downloaded_artifact_path(output_dir, name, local_path)
                if actual_local_path is None:
                    raise FileNotFoundError(f"Downloaded artifact missing: {local_path}")
                append_manifest_entry(manifest_path, ReplayDatasetEntry(source=dataset, local_path=str(actual_local_path), size_bytes=size, discussion_id=discussion_id), cap_gb=cap_gb)
                new_bytes += size
                downloaded_files += 1
                if max_files is not None and downloaded_files >= max_files:
                    break
            next_token = _extract_next_page_token(res.stdout)
            if not next_token:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Orbit Wars replay JSON files under quota")
    parser.add_argument("--cap-gb", type=float, default=100.0)
    parser.add_argument("--max-new-gb", type=float, default=5.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/replays"))
    parser.add_argument("--manifest", type=Path, default=Path("data/replays/manifest.json"))
    args = parser.parse_args()
    download_replays(cap_gb=args.cap_gb, max_new_gb=args.max_new_gb, output_dir=args.output_dir, manifest_path=args.manifest, max_files=args.max_files)


def append_manifest_entry(
    manifest_path: Path, entry: ReplayDatasetEntry, cap_gb: float = 100.0
) -> None:
    if entry.size_bytes < 0:
        raise ValueError("entry.size_bytes cannot be negative")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Manifest must be a JSON list")
    else:
        data = []

    current_bytes = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry at index {idx} must be an object")
        size_bytes = item.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"Manifest entry at index {idx} has invalid size_bytes")
        current_bytes += size_bytes

    if would_exceed_cap(current_bytes=current_bytes, next_bytes=entry.size_bytes, cap_gb=cap_gb):
        next_total_bytes = current_bytes + entry.size_bytes
        raise ValueError(
            "Replay manifest size cap exceeded: "
            f"current={bytes_to_gb(current_bytes)} GiB, "
            f"next={bytes_to_gb(next_total_bytes)} GiB, "
            f"cap={cap_gb} GiB"
        )

    data.append(asdict(entry))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        delete=False,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_path = Path(tmp_file.name)

    try:
        os.replace(tmp_path, manifest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
