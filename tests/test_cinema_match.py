from __future__ import annotations

from tools.render_cinema_match import render_env_json_to_cinema_html


def test_render_env_json_to_cinema_html_uses_auto_cinema(tmp_path):
    env_json = {
        "steps": [
            [
                {
                    "observation": {
                        "planets": [],
                        "fleets": [],
                        "comets": [],
                        "comet_planet_ids": [],
                        "angular_velocity": 0,
                    },
                    "status": "ACTIVE",
                    "reward": 0,
                    "action": {},
                },
                {"observation": {}, "status": "ACTIVE", "reward": 0, "action": {}},
            ],
            [
                {
                    "observation": {
                        "planets": [],
                        "fleets": [],
                        "comets": [],
                        "comet_planet_ids": [],
                        "angular_velocity": 0,
                    },
                    "status": "DONE",
                    "reward": 1,
                    "action": {},
                },
                {"observation": {}, "status": "DONE", "reward": 0, "action": {}},
            ],
        ],
        "info": {"logs": []},
    }
    out = tmp_path / "cinema" / "match.html"

    render_env_json_to_cinema_html(env_json, out, bot_names=["candidate", "baseline"])

    html = out.read_text(encoding="utf-8")
    assert "Orbit Wars Viewer" in html
    assert '"autoCinema": true' in html
