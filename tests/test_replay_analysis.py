import json
from pathlib import Path

from tools.analyze_orbit_replays import analyze_corpus, iter_replay_paths, summarize_replay


def test_summarize_replay_extracts_teams_rewards_and_actions():
    replay = {
        "module_version": "1.29.3",
        "info": {"EpisodeId": 123, "TeamNames": ["A", "B"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {
                    "observation": {
                        "step": 0,
                        "remainingOverageTime": 60,
                        "planets": [[0, 0, 10, 10, 1, 10, 1]],
                        "fleets": [],
                    },
                    "action": [],
                },
                {
                    "observation": {
                        "step": 0,
                        "remainingOverageTime": 60,
                        "planets": [[0, 0, 10, 10, 1, 10, 1]],
                        "fleets": [],
                    },
                    "action": [],
                },
            ],
            [
                {
                    "observation": {
                        "step": 1,
                        "remainingOverageTime": 59.9,
                        "planets": [],
                        "fleets": [],
                    },
                    "action": [[0, 0.5, 3]],
                },
                {
                    "observation": {
                        "step": 1,
                        "remainingOverageTime": 59.8,
                        "planets": [],
                        "fleets": [],
                    },
                    "action": [],
                },
            ],
        ],
    }

    summary = summarize_replay(replay)

    assert summary["episode_id"] == 123
    assert summary["teams"] == ["A", "B"]
    assert summary["rewards"] == [1, -1]
    assert summary["steps"] == 2
    assert summary["actions_by_player"] == [1, 0]
    assert summary["ships_launched_by_player"] == [3, 0]
    assert summary["min_remaining_overage_by_player"] == [59.9, 59.8]


def test_summarize_replay_accepts_lowercase_team_names_and_final_step_outcomes():
    replay = {
        "info": {"teamNames": ["A", "B"]},
        "steps": [
            [{"action": []}, {"action": []}],
            [
                {"reward": 1, "status": "DONE", "action": []},
                {"reward": -1, "status": "DONE", "action": []},
            ],
        ],
    }

    summary = summarize_replay(replay)

    assert summary["teams"] == ["A", "B"]
    assert summary["rewards"] == [1, -1]
    assert summary["statuses"] == ["DONE", "DONE"]


def test_summarize_replay_infers_players_from_steps_when_metadata_missing():
    replay = {
        "steps": [
            [
                {"action": [[0, 0.5, 1]]},
                {"action": [[0, 0.5, 2]]},
                {"action": [[0, 0.5, 3]]},
            ]
        ]
    }

    summary = summarize_replay(replay)

    assert len(summary["actions_by_player"]) == 3
    assert len(summary["ships_launched_by_player"]) == 3
    assert len(summary["min_remaining_overage_by_player"]) == 3


def test_summarize_replay_marks_unobserved_overage():
    replay = {
        "info": {"TeamNames": ["A", "B"]},
        "steps": [
            [
                {"observation": {"remainingOverageTime": 10}, "action": []},
                {"observation": {}, "action": []},
            ]
        ],
    }

    summary = summarize_replay(replay)

    assert summary["min_remaining_overage_by_player"] == [10.0, 0.0]
    assert summary["remaining_overage_observed_by_player"] == [True, False]


def test_iter_replay_paths_honors_limit_without_exhausting_generator():
    class FakeRoot:
        def rglob(self, pattern: str):
            assert pattern == "*.json"
            yield Path("a.json")
            yield Path("b.json")
            raise RuntimeError("should not be consumed past limit")

    paths = list(iter_replay_paths(FakeRoot(), limit=2))
    assert paths == [Path("a.json"), Path("b.json")]


def test_analyze_corpus_caps_sample_without_losing_count(tmp_path):
    date_dir = tmp_path / "2026-05-29"
    date_dir.mkdir()
    for i in range(205):
        replay_path = date_dir / f"r{i:03d}.json"
        replay_path.write_text(json.dumps({"info": {"TeamNames": ["A", "B"]}, "steps": []}), encoding="utf-8")

    report = analyze_corpus(tmp_path)

    assert report["count"] == 205
    assert len(report["sample"]) == 200
    assert report["by_date"]["2026-05-29"]["files"] == 205


def test_analyze_corpus_does_not_parse_files_beyond_sample_cap(tmp_path, monkeypatch):
    date_dir = tmp_path / "2026-05-29"
    date_dir.mkdir()
    for i in range(205):
        (date_dir / f"r{i:03d}.json").write_text("{}", encoding="utf-8")

    import tools.analyze_orbit_replays as replay_analysis

    load_calls = 0

    def fake_load(_file_obj):
        nonlocal load_calls
        load_calls += 1
        return {"info": {"TeamNames": ["A", "B"]}, "steps": []}

    monkeypatch.setattr(replay_analysis.json, "load", fake_load)

    report = analyze_corpus(tmp_path)

    assert report["count"] == 205
    assert len(report["sample"]) == 200
    assert load_calls == 200
