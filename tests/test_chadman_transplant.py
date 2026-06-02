import importlib


def test_chadman_modules_export_agent_and_world_builder():
    baseline = importlib.import_module("src.ow.chadman_baseline")
    candidate = importlib.import_module("src.ow.chadman")

    assert callable(baseline.agent)
    assert callable(candidate.agent)
    assert callable(baseline.build_world)
    assert callable(candidate.build_world)


def test_chadman_planet_and_fleet_schema_match_official():
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
    from src.ow import chadman

    assert chadman.Planet._fields == Planet._fields
    assert chadman.Fleet._fields == Fleet._fields


def test_root_main_exports_chadman_agent():
    import main
    from src.ow import chadman

    assert callable(main.agent)
    assert main.agent.__name__ == "agent"
    assert main._CHADMAN_AGENT is chadman.agent


def test_root_main_debug_fallback_reports_exception(monkeypatch, capsys):
    import main

    def boom(obs, config=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "_CHADMAN_AGENT", boom)
    monkeypatch.setenv("OW_AGENT_DEBUG", "1")

    assert main.agent({"player": 0}, None) == []
    assert "Chadman agent exception" in capsys.readouterr().err
