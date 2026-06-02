from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path


class RollingStats:
    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.episodes = 0
        self.wins = 0
        self.losses = 0
        self._recent = deque(maxlen=window)

    def add(self, won: bool) -> None:
        self.episodes += 1
        if won:
            self.wins += 1
            self._recent.append(1.0)
        else:
            self.losses += 1
            self._recent.append(0.0)

    @property
    def recent_win_rate(self) -> float:
        if not self._recent:
            return 0.0
        return float(sum(self._recent) / len(self._recent))


class MetricsWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self._csv_fields = None

    def write(self, row: dict) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
            write_header = True
        elif any(key not in self._csv_fields for key in row):
            self._csv_fields = self._merged_csv_fields()
            self._rewrite_csv_from_jsonl()
            return
        else:
            write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self._csv_fields})

    def _jsonl_rows(self) -> list[dict]:
        rows: list[dict] = []
        if not self.jsonl_path.exists():
            return rows
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _merged_csv_fields(self) -> list[str]:
        fields: list[str] = []
        seen: set[str] = set()
        for json_row in self._jsonl_rows():
            for key in json_row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        return fields

    def _rewrite_csv_from_jsonl(self) -> None:
        rows = self._jsonl_rows()
        assert self._csv_fields is not None
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            writer.writeheader()
            for json_row in rows:
                writer.writerow({k: json_row.get(k) for k in self._csv_fields})
