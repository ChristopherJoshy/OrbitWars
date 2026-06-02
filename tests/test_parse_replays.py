import io
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tools.parse_replays import extract_training_rows, iter_replay_json_records, iter_replay_records, main


def test_iter_replay_json_records_reads_json_from_tar_gz(tmp_path):
    tar_path = tmp_path / "episodes.tar.gz"
    payload = {
        "id": 123,
        "steps": [[{"observation": {"player": 0}, "action": []}]],
    }

    with tarfile.open(tar_path, "w:gz") as tar:
        content = json.dumps(payload).encode("utf-8")
        info = tarfile.TarInfo(name="episode_123.json")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))

    assert list(iter_replay_json_records(tar_path)) == [payload]


def test_iter_replay_json_records_skips_non_json_and_dirs(tmp_path):
    tar_path = tmp_path / "episodes.tar.gz"
    payload = {"id": 1}

    with tarfile.open(tar_path, "w:gz") as tar:
        dir_info = tarfile.TarInfo(name="episodes")
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)

        txt_content = b"ignore"
        txt_info = tarfile.TarInfo(name="episodes/readme.txt")
        txt_info.size = len(txt_content)
        tar.addfile(txt_info, io.BytesIO(txt_content))

        json_content = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name="episodes/episode_1.json")
        json_info.size = len(json_content)
        tar.addfile(json_info, io.BytesIO(json_content))

    assert list(iter_replay_json_records(tar_path)) == [payload]


def test_iter_replay_json_records_handles_hostile_member_name_without_disk_write(tmp_path):
    tar_path = tmp_path / "episodes.tar.gz"
    payload = {"id": 7, "ok": True}
    outside_target = tmp_path.parent / "x.json"

    with tarfile.open(tar_path, "w:gz") as tar:
        content = json.dumps(payload).encode("utf-8")
        info = tarfile.TarInfo(name="../../x.json")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))

    assert list(iter_replay_json_records(tar_path)) == [payload]
    assert not outside_target.exists()


def test_extract_training_rows_labels_action_source_and_no_action():
    record = {
        "steps": [
            [
                {
                    "observation": {
                        "player": 0,
                        "step": 1,
                        "planets": [
                            [1, 0, 0, 0, 1, 20, 3],
                            [2, -1, 10, 0, 1, 10, 2],
                        ],
                        "fleets": [],
                        "comets": [],
                        "initial_planets": [
                            [1, 0, 0, 0, 1, 20, 3],
                            [2, -1, 10, 0, 1, 10, 2],
                        ],
                    },
                    "action": [[1, 2, 5]],
                }
            ],
            [
                {
                    "observation": {
                        "player": 0,
                        "step": 2,
                        "planets": [
                            [1, 0, 0, 0, 1, 18, 3],
                            [2, -1, 10, 0, 1, 10, 2],
                        ],
                        "fleets": [],
                        "comets": [],
                        "initial_planets": [
                            [1, 0, 0, 0, 1, 20, 3],
                            [2, -1, 10, 0, 1, 10, 2],
                        ],
                    },
                    "action": [],
                }
            ],
        ]
    }

    rows, labels = extract_training_rows(record)
    assert len(rows) == 2
    assert labels == [1, 0]


def test_extract_training_rows_skips_malformed_states():
    record = {
        "steps": [
            [{"observation": {"player": 0, "step": 1, "planets": "bad"}, "action": []}],
            [{"observation": {"player": 0, "step": 2, "planets": [[1, 0, 0, 0, 1, 20, 3]], "fleets": [], "comets": [], "initial_planets": [[1, 0, 0, 0, 1, 20, 3]]}, "action": []}],
        ]
    }
    rows, labels = extract_training_rows(record)
    assert rows == []
    assert labels == []


def test_extract_training_rows_angle_heuristic_picks_closest_target():
    record = {
        "steps": [
            [
                {
                    "observation": {
                        "player": 0,
                        "step": 1,
                        "planets": [
                            [1, 0, 0, 0, 1, 20, 3],
                            [2, -1, 10, 0, 1, 10, 2],
                            [3, -1, 0, 10, 1, 10, 2],
                        ],
                        "fleets": [],
                        "comets": [],
                        "initial_planets": [
                            [1, 0, 0, 0, 1, 20, 3],
                            [2, -1, 10, 0, 1, 10, 2],
                            [3, -1, 0, 10, 1, 10, 2],
                        ],
                    },
                    "action": [[1, 0.0, 5]],
                }
            ]
        ]
    }
    _, labels = extract_training_rows(record)
    assert labels.count(1) == 1


def test_extract_training_rows_explicit_empty_action_labels_all_zero():
    record = {
        "steps": [
            [{"observation": {"player": 0, "step": 1, "planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]], "fleets": [], "comets": [], "initial_planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]]}, "action": []}]
        ]
    }
    _, labels = extract_training_rows(record)
    assert labels == [0]


def test_extract_training_rows_skips_none_missing_and_malformed_action_entries():
    base_obs = {"player": 0, "step": 1, "planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]], "fleets": [], "comets": [], "initial_planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]]}
    record = {
        "steps": [
            [{"observation": base_obs, "action": None}],
            [{"observation": base_obs}],
            [{"observation": base_obs, "action": {"bad": 1}}],
        ]
    }
    rows, labels = extract_training_rows(record)
    assert rows == []
    assert labels == []


def test_extract_training_rows_skips_malformed_nonempty_action_lists():
    obs = {
        "player": 0,
        "step": 1,
        "planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]],
        "fleets": [],
        "comets": [],
        "initial_planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]],
    }
    bad_actions = [
        [[1]],
        [[1, 0.0]],
        [["bad"]],
        ["bad"],
        [[math.inf, 0.0, 1]],
        [[1, math.inf, 1]],
        [[1, 0.0, math.inf]],
        [[1, 0.0, -1]],
    ]

    for bad_action in bad_actions:
        rows, labels = extract_training_rows({"steps": [[{"observation": obs, "action": bad_action}] ]})
        assert rows == []
        assert labels == []


def test_extract_training_rows_mixed_action_list_with_malformed_item_is_skipped():
    obs = {
        "player": 0,
        "step": 1,
        "planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]],
        "fleets": [],
        "comets": [],
        "initial_planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]],
    }
    record = {
        "steps": [
            [
                {"observation": obs, "action": [[1, 0.0, 5], [1]]},
                {"observation": obs, "action": []},
            ]
        ]
    }
    rows, labels = extract_training_rows(record)
    assert len(rows) == 1
    assert labels == [0]


def test_extract_training_rows_processes_multiple_agents_per_step():
    obs = {"player": 0, "step": 1, "planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]], "fleets": [], "comets": [], "initial_planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]]}
    record = {"steps": [[[{"observation": obs, "action": []}, {"observation": obs, "action": []}][0], {"observation": obs, "action": []}]]}
    rows, labels = extract_training_rows(record)
    assert len(rows) == 2
    assert labels == [0, 0]


def test_extract_training_rows_skips_malformed_first_keeps_valid_second_agent():
    obs = {"player": 0, "step": 1, "planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]], "fleets": [], "comets": [], "initial_planets": [[1, 0, 0, 0, 1, 20, 3], [2, -1, 10, 0, 1, 10, 2]]}
    record = {"steps": [[{"observation": obs, "action": {"bad": 1}}, {"observation": obs, "action": []}]]}
    rows, labels = extract_training_rows(record)
    assert len(rows) == 1
    assert labels == [0]


def test_iter_replay_records_reads_single_json(tmp_path):
    p = tmp_path / "episode.json"
    payload = {"id": 1, "steps": []}
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert list(iter_replay_records(p)) == [payload]


def test_iter_replay_records_reads_json_directory_recursively(tmp_path):
    d = tmp_path / "episodes" / "nested"
    d.mkdir(parents=True)
    a = {"id": "a", "steps": []}
    b = {"id": "b", "steps": []}
    (tmp_path / "episodes" / "a.json").write_text(json.dumps(a), encoding="utf-8")
    (d / "b.json").write_text(json.dumps(b), encoding="utf-8")
    got = list(iter_replay_records(tmp_path / "episodes"))
    assert got == [a, b]


def test_main_writes_features_and_summary(tmp_path, monkeypatch):
    replay = tmp_path / "episode.json"
    replay.write_text(json.dumps({"steps": []}), encoding="utf-8")
    out_npz = tmp_path / "processed" / "shot_features.npz"
    out_summary = tmp_path / "reports" / "summary.json"

    monkeypatch.setattr("sys.argv", ["parse_replays.py", str(replay), "--features-output", str(out_npz), "--summary-output", str(out_summary), "--limit", "1"])
    monkeypatch.setattr("tools.parse_replays.extract_training_rows", lambda rec: ([[1.0, 2.0]], [1]))
    main()

    with np.load(out_npz, allow_pickle=False) as data:
        assert data["x"].shape == (1, 2)
        assert data["y"].tolist() == [1]
        assert "feature_columns" in data
    summary = json.loads(out_summary.read_text(encoding="utf-8"))
    assert summary["rows"] == 1
    assert summary["positives"] == 1


def test_main_raises_on_no_rows(tmp_path, monkeypatch):
    replay = tmp_path / "episode.json"
    replay.write_text(json.dumps({"steps": []}), encoding="utf-8")
    out_npz = tmp_path / "features.npz"
    monkeypatch.setattr("sys.argv", ["parse_replays.py", str(replay), "--features-output", str(out_npz)])
    monkeypatch.setattr("tools.parse_replays.extract_training_rows", lambda rec: ([], []))
    with pytest.raises(ValueError, match="No training rows"):
        main()


def test_features_atomic_replace_failure_keeps_existing(tmp_path, monkeypatch):
    replay = tmp_path / "episode.json"
    replay.write_text(json.dumps({"steps": []}), encoding="utf-8")
    out_npz = tmp_path / "features.npz"
    np.savez(out_npz, x=np.array([[0.0]]), y=np.array([0]), feature_columns=np.array(["f0"]))
    before = out_npz.read_bytes()

    monkeypatch.setattr("sys.argv", ["parse_replays.py", str(replay), "--features-output", str(out_npz)])
    monkeypatch.setattr("tools.parse_replays.extract_training_rows", lambda rec: ([[1.0]], [1]))
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        main()
    assert out_npz.read_bytes() == before


def test_main_writes_count_only_summary_when_features_output_missing(tmp_path, monkeypatch):
    replay = tmp_path / "episode.json"
    replay.write_text(json.dumps({"steps": []}), encoding="utf-8")
    out_summary = tmp_path / "reports" / "summary.json"

    monkeypatch.setattr(
        "sys.argv",
        ["parse_replays.py", str(replay), "--summary-output", str(out_summary), "--limit", "1"],
    )
    main()

    summary = json.loads(out_summary.read_text(encoding="utf-8"))
    assert summary["records_seen"] == 1
    assert summary["rows"] == 0
    assert summary["positives"] == 0
    assert summary["negatives"] == 0
    assert summary["feature_columns"]
    assert summary["output"] is None


def test_main_summary_replace_failure_cleans_tmp(tmp_path, monkeypatch):
    replay = tmp_path / "episode.json"
    replay.write_text(json.dumps({"steps": []}), encoding="utf-8")
    out_npz = tmp_path / "processed" / "shot_features.npz"
    out_summary = tmp_path / "reports" / "summary.json"
    monkeypatch.setattr("sys.argv", ["parse_replays.py", str(replay), "--features-output", str(out_npz), "--summary-output", str(out_summary), "--limit", "1"])
    monkeypatch.setattr("tools.parse_replays.extract_training_rows", lambda rec: ([[1.0, 2.0]], [1]))

    tmp_path_holder = {}
    real_replace = os.replace

    def fail_summary_replace(src, dst):
        if Path(dst) == out_summary:
            tmp_path_holder["tmp"] = Path(src)
            raise OSError("summary replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_summary_replace)
    with pytest.raises(OSError, match="summary replace failed"):
        main()
    assert "tmp" in tmp_path_holder
    assert not tmp_path_holder["tmp"].exists()
