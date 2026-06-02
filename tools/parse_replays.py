"""Replay archive parsing utilities."""

from __future__ import annotations

import argparse
import math
import json
import os
from pathlib import Path
import sys
import tarfile
from typing import Iterator
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ow.features import FEATURE_COLUMNS, build_candidate_features
from src.ow.constants import PLANET_COL_ID, PLANET_COL_X, PLANET_COL_Y
from src.ow.state import parse_observation


def iter_replay_json_records(tar_path: Path) -> Iterator[dict]:
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted as fh:
                yield json.load(fh)


def iter_replay_records(path: Path) -> Iterator[dict]:
    if path.is_dir():
        for p in sorted(path.rglob("*.json")):
            yield json.loads(p.read_text(encoding="utf-8"))
        return
    if path.suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8"))
        return
    yield from iter_replay_json_records(path)


def _step_entries(step):
    if isinstance(step, list):
        return [entry for entry in step if isinstance(entry, dict)]
    if isinstance(step, dict):
        return [step]
    return []


def _wrap_angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def _extract_positive_pairs(action, state, pairs) -> set[tuple[int, int]] | None:
    if action == []:
        return set()
    if not isinstance(action, list):
        return None
    pair_set = {(int(src), int(tgt)) for src, tgt in pairs.tolist()} if len(pairs) else set()
    if not pair_set and action:
        return set()

    pos: set[tuple[int, int]] = set()
    planet_by_id = {int(row[PLANET_COL_ID]): row for row in state.planets}
    for item in action:
        # Orbit Wars action item must include [source, angle, ships].
        # Any malformed item makes the whole non-empty action list unknown.
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            return None
        try:
            source = float(item[0])
            launch_angle = float(item[1])
            ships = float(item[2])
            src_id = int(source)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(source) and math.isfinite(launch_angle) and math.isfinite(ships) and ships >= 0.0):
            return None
        src_row = planet_by_id.get(src_id)
        if src_row is None:
            continue
        candidates = [tgt for (src, tgt) in pair_set if src == src_id]
        if not candidates:
            continue
        sx = float(src_row[PLANET_COL_X])
        sy = float(src_row[PLANET_COL_Y])
        best_tgt = None
        best_diff = math.inf
        for tgt_id in candidates:
            tgt_row = planet_by_id.get(tgt_id)
            if tgt_row is None:
                continue
            dx = float(tgt_row[PLANET_COL_X]) - sx
            dy = float(tgt_row[PLANET_COL_Y]) - sy
            bearing = math.atan2(dy, dx)
            diff = _wrap_angle_diff(launch_angle, bearing)
            if diff < best_diff:
                best_diff = diff
                best_tgt = tgt_id
        if best_tgt is not None:
            pos.add((src_id, int(best_tgt)))
    return pos


def extract_training_rows(record: dict) -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for step in record.get("steps", []):
        for entry in _step_entries(step):
            try:
                state = parse_observation(entry.get("observation", {}))
                features, pairs = build_candidate_features(state)
            except Exception:
                continue
            positive_pairs = _extract_positive_pairs(entry.get("action"), state, pairs)
            if positive_pairs is None:
                continue
            for feature_row, pair in zip(features, pairs):
                rows.append(feature_row.astype(float, copy=False).tolist())
                labels.append(1 if (int(pair[0]), int(pair[1])) in positive_pairs else 0)
    return rows, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse replay JSON records")
    parser.add_argument("path", type=Path, help="Path to replay tar/tar.gz archive, JSON, or directory")
    parser.add_argument("--limit", type=int, default=10, help="Max records to count")
    parser.add_argument("--features-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    count = 0
    all_rows: list[list[float]] = []
    all_labels: list[int] = []
    for rec in iter_replay_records(args.path):
        if not isinstance(rec, dict):
            continue
        count += 1
        if args.features_output is not None:
            rows, labels = extract_training_rows(rec)
            all_rows.extend(rows)
            all_labels.extend(labels)
        if count >= args.limit:
            break
    if args.features_output is None:
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "records_seen": count,
                "rows": 0,
                "positives": 0,
                "negatives": 0,
                "feature_columns": list(FEATURE_COLUMNS),
                "output": None,
            }
            tmp_json = args.summary_output.with_name(f".{args.summary_output.name}.{uuid4().hex}.tmp")
            try:
                tmp_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                os.replace(tmp_json, args.summary_output)
            except Exception:
                tmp_json.unlink(missing_ok=True)
                raise
        print(count)
        return
    if not all_rows:
        raise ValueError("No training rows extracted; refusing to save empty dataset")
    x = np.asarray(all_rows, dtype=np.float64)
    y = np.asarray(all_labels, dtype=np.int64)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.features_output.with_name(f".{args.features_output.stem}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(tmp, x=x, y=y, feature_columns=np.asarray(FEATURE_COLUMNS, dtype=str))
        os.replace(tmp, args.features_output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "records_seen": count,
            "rows": int(y.shape[0]),
            "positives": int((y == 1).sum()),
            "negatives": int((y == 0).sum()),
            "feature_columns": list(FEATURE_COLUMNS),
            "output": str(args.features_output),
        }
        tmp_json = args.summary_output.with_name(f".{args.summary_output.name}.{uuid4().hex}.tmp")
        try:
            tmp_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            os.replace(tmp_json, args.summary_output)
        except Exception:
            tmp_json.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    main()
