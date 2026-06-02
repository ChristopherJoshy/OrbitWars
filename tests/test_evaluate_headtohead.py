import pytest

from tools.evaluate_headtohead import _load_agent, main, run_headtohead, run_match, summarize_results


def test_summarize_results_counts_candidate_wins_and_errors():
    rows = [
        {
            "candidate_reward": 1,
            "baseline_reward": -1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.01,
        },
        {
            "candidate_reward": -1,
            "baseline_reward": 1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.02,
        },
        {
            "candidate_reward": None,
            "baseline_reward": 1,
            "candidate_status": "ERROR",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.03,
        },
    ]

    summary = summarize_results(rows)

    assert summary["matches"] == 3
    assert summary["candidate_wins"] == 1
    assert summary["win_rate"] == 1 / 3
    assert summary["errors"] == 1
    assert summary["candidate_time_mean"] == 0.02


def test_summarize_results_tie_does_not_count_as_candidate_win():
    rows = [
        {
            "candidate_reward": 1,
            "baseline_reward": 1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.01,
            "candidate_time_max": 0.02,
            "candidate_time_p95": 0.02,
        }
    ]

    summary = summarize_results(rows)

    assert summary["candidate_wins"] == 0
    assert summary["ties"] == 1
    assert summary["candidate_losses"] == 0
    assert summary["win_rate"] == 0.0


def test_summarize_results_uses_match_max_and_p95_columns_for_timing_aggregation():
    rows = [
        {
            "candidate_reward": 1,
            "baseline_reward": -1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.1,
            "candidate_time_max": 3.0,
            "candidate_time_p95": 2.0,
        },
        {
            "candidate_reward": -1,
            "baseline_reward": 1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.2,
            "candidate_time_max": 4.0,
            "candidate_time_p95": 2.5,
        },
    ]

    summary = summarize_results(rows)

    assert summary["candidate_time_mean"] == pytest.approx(0.15)
    assert summary["candidate_time_max"] == 4.0
    assert summary["candidate_time_p95"] == 2.0


def test_load_agent_from_python_file_path_returns_callable(tmp_path):
    agent_file = tmp_path / "temp_agent.py"
    agent_file.write_text(
        "def agent(obs, config=None):\n"
        "    return [42]\n"
    )

    loaded = _load_agent(str(agent_file))

    assert callable(loaded)
    assert loaded({}, None) == [42]


def test_run_headtohead_forwards_episode_steps_to_both_orders(monkeypatch):
    calls = []

    def fake_run_match(candidate_ref, baseline_ref, seed, order, episode_steps=80):
        calls.append(
            {
                "candidate_ref": candidate_ref,
                "baseline_ref": baseline_ref,
                "seed": seed,
                "order": order,
                "episode_steps": episode_steps,
            }
        )
        return {
            "seed": seed,
            "order": order,
            "candidate_reward": 1,
            "baseline_reward": -1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.01,
            "candidate_time_max": 0.01,
        }

    monkeypatch.setattr("tools.evaluate_headtohead.run_match", fake_run_match)

    run_headtohead("cand:agent", "base:agent", seeds=2, episode_steps=80)

    assert len(calls) == 4
    assert [c["order"] for c in calls] == [
        "candidate_first",
        "candidate_second",
        "candidate_first",
        "candidate_second",
    ]
    assert all(c["episode_steps"] == 80 for c in calls)


def test_run_match_passes_episode_steps_configuration(monkeypatch):
    captured = {}

    class DummyState:
        def __init__(self, reward, status):
            self.reward = reward
            self.status = status

    class DummyEnv:
        def __init__(self):
            self.state = [DummyState(1, "DONE"), DummyState(-1, "DONE")]

        def run(self, agents):
            agents[0]({}, None)
            return self.state

    def fake_make(name, configuration, debug):
        captured["name"] = name
        captured["configuration"] = configuration
        captured["debug"] = debug
        return DummyEnv()

    monkeypatch.setattr("tools.evaluate_headtohead.make", fake_make)
    monkeypatch.setattr("tools.evaluate_headtohead._load_agent", lambda _ref: (lambda _obs, _cfg=None: [0]))

    run_match("cand:agent", "base:agent", seed=7, order="candidate_first", episode_steps=80)

    assert captured["name"] == "orbit_wars"
    assert captured["configuration"]["seed"] == 7
    assert captured["configuration"]["episodeSteps"] == 80


def test_run_match_omits_episode_steps_when_none_or_non_positive(monkeypatch):
    captured_configs = []

    class DummyState:
        def __init__(self, reward, status):
            self.reward = reward
            self.status = status

    class DummyEnv:
        def __init__(self):
            self.state = [DummyState(1, "DONE"), DummyState(-1, "DONE")]

        def run(self, agents):
            agents[0]({}, None)
            return self.state

    def fake_make(name, configuration, debug):
        captured_configs.append(configuration)
        return DummyEnv()

    monkeypatch.setattr("tools.evaluate_headtohead.make", fake_make)
    monkeypatch.setattr("tools.evaluate_headtohead._load_agent", lambda _ref: (lambda _obs, _cfg=None: [0]))

    run_match("cand:agent", "base:agent", seed=1, order="candidate_first", episode_steps=None)
    run_match("cand:agent", "base:agent", seed=2, order="candidate_first", episode_steps=0)

    assert "episodeSteps" not in captured_configs[0]
    assert "episodeSteps" not in captured_configs[1]


def test_cli_uses_bounded_default_steps(monkeypatch, tmp_path):
    captured = {}
    output = tmp_path / "report.json"

    def fake_run_headtohead(candidate_ref, baseline_ref, seeds, episode_steps=80):
        captured["episode_steps"] = episode_steps
        return {"episode_steps": episode_steps, "summary": {}, "matches": []}

    monkeypatch.setattr("tools.evaluate_headtohead.run_headtohead", fake_run_headtohead)
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_headtohead.py", "--output", str(output)],
    )

    main()

    assert captured["episode_steps"] == 80


def test_cli_maps_non_positive_steps_to_full_default(monkeypatch, tmp_path):
    captured = {}
    output = tmp_path / "report.json"

    def fake_run_headtohead(candidate_ref, baseline_ref, seeds, episode_steps=80):
        captured["episode_steps"] = episode_steps
        return {"episode_steps": episode_steps, "summary": {}, "matches": []}

    monkeypatch.setattr("tools.evaluate_headtohead.run_headtohead", fake_run_headtohead)
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_headtohead.py", "--steps", "0", "--output", str(output)],
    )

    main()

    assert captured["episode_steps"] is None


def test_parse_match_specs_accepts_seed_order_pairs():
    from tools import evaluate_headtohead

    specs = evaluate_headtohead.parse_match_specs("2:candidate_first,4:candidate_second,7:candidate_first")

    assert specs == [(2, "candidate_first"), (4, "candidate_second"), (7, "candidate_first")]


def test_parse_match_specs_strips_seed_and_order_whitespace():
    from tools import evaluate_headtohead

    specs = evaluate_headtohead.parse_match_specs(" 2: candidate_first , 4:candidate_second ")

    assert specs == [(2, "candidate_first"), (4, "candidate_second")]


def test_parse_match_specs_rejects_bad_order():
    from tools import evaluate_headtohead

    import pytest

    with pytest.raises(ValueError, match="order must be"):
        evaluate_headtohead.parse_match_specs("2:middle")


def test_run_selected_matches_uses_injected_runner_and_summary():
    from tools import evaluate_headtohead

    calls = []

    def fake_runner(candidate_ref, baseline_ref, seed, order, episode_steps=80):
        calls.append((candidate_ref, baseline_ref, seed, order, episode_steps))
        return {
            "seed": seed,
            "order": order,
            "candidate_reward": 1,
            "baseline_reward": -1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.01,
            "candidate_time_p95": 0.02,
            "candidate_time_max": 0.03,
        }

    report = evaluate_headtohead.run_selected_matches(
        "candidate",
        "baseline",
        [(2, "candidate_first"), (4, "candidate_second")],
        episode_steps=220,
        jobs=4,
        runner=fake_runner,
    )

    assert calls == [
        ("candidate", "baseline", 2, "candidate_first", 220),
        ("candidate", "baseline", 4, "candidate_second", 220),
    ]
    assert report["episode_steps"] == 220
    assert report["summary"]["matches"] == 2
    assert report["summary"]["candidate_wins"] == 2


def test_run_selected_matches_uses_process_pool_for_default_runner(monkeypatch):
    from tools import evaluate_headtohead

    captured = {}

    class FakeProcessPoolExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, args_iterable):
            args_list = list(args_iterable)
            captured["args_list"] = args_list
            return [func(args) for args in args_list]

    def fake_run_match_from_tuple(args):
        candidate_ref, baseline_ref, seed, order, episode_steps = args
        return {
            "seed": seed,
            "order": order,
            "candidate_reward": 1 if order == "candidate_first" else -1,
            "baseline_reward": -1 if order == "candidate_first" else 1,
            "candidate_status": "DONE",
            "baseline_status": "DONE",
            "candidate_time_mean": 0.01,
            "candidate_time_p95": 0.02,
            "candidate_time_max": 0.03,
        }

    monkeypatch.setattr("tools.evaluate_headtohead.ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr("tools.evaluate_headtohead._run_match_from_tuple", fake_run_match_from_tuple)

    report = evaluate_headtohead.run_selected_matches(
        "candidate:agent",
        "baseline:agent",
        [(2, "candidate_first"), (4, "candidate_second")],
        episode_steps=220,
        jobs=3,
    )

    assert captured["max_workers"] == 3
    assert captured["args_list"] == [
        ("candidate:agent", "baseline:agent", 2, "candidate_first", 220),
        ("candidate:agent", "baseline:agent", 4, "candidate_second", 220),
    ]
    assert report["summary"]["matches"] == 2
    assert report["summary"]["candidate_wins"] == 1
