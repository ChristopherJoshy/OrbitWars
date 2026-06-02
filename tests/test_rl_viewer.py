import pytest

from src.ow.rl.pygame_viewer import (
    NEUTRAL_COLOR,
    PLAYER_COLORS,
    SUN_CENTER,
    SUN_RADIUS,
    TrainingOverlay,
    agent_identity_lines,
    board_rect,
    burn_in_offset,
    minimal_status_line,
    summarize_observation,
    world_to_screen,
)


class _FakeScreen:
    def __init__(self, size):
        self._size = tuple(size)
        self.fill_calls = []

    def get_size(self):
        return self._size

    def fill(self, color):
        self.fill_calls.append(color)


class _CrashingScreen(_FakeScreen):
    def fill(self, color):
        raise RuntimeError("display died")


class _FakeDisplay:
    def __init__(self):
        self.calls = []
        self.iconify_calls = 0
        self.flip_calls = 0

    def set_mode(self, size, flags=0):
        self.calls.append((tuple(size), flags))
        return _FakeScreen(size)

    def set_caption(self, _caption):
        return None

    def iconify(self):
        self.iconify_calls += 1

    def flip(self):
        self.flip_calls += 1
        return None


class _FakeEvent:
    def __init__(self, type_, key=None, w=None, h=None):
        self.type = type_
        self.key = key
        self.w = w
        self.h = h


class _FakeEventQueue:
    def __init__(self, events):
        self._events = list(events)

    def get(self):
        events = list(self._events)
        self._events.clear()
        return events


class _FakePygame:
    FULLSCREEN = 1 << 0
    RESIZABLE = 1 << 1
    QUIT = 10
    VIDEORESIZE = 11
    KEYDOWN = 12
    K_F11 = 13
    K_m = 14

    def __init__(self, events=None):
        self.display = _FakeDisplay()
        self.event = _FakeEventQueue(events or [])
        self.font = type("_Font", (), {"SysFont": staticmethod(lambda *a, **k: object())})

    def init(self):
        return None


def test_original_style_colors_and_sun_constants_are_exposed():
    assert PLAYER_COLORS[0] == (58, 166, 255)
    assert NEUTRAL_COLOR == (136, 136, 136)
    assert SUN_CENTER == (50.0, 50.0)
    assert SUN_RADIUS == 10.0


def test_training_overlay_text_contains_iterations_and_wins():
    overlay = TrainingOverlay()
    text = overlay.lines(
        {"update": 12, "episode": 34, "wins": 9, "losses": 3, "recent_win_rate": 0.75, "episode_reward": 1.25}
    )
    joined = "\n".join(text)
    assert "update=12" in joined
    assert "episode=34" in joined
    assert "wins=9" in joined
    assert "losses=3" in joined
    assert "win_rate=0.750" in joined
    assert "reward=1.250" in joined


def test_training_overlay_text_contains_mistake_and_status_metrics():
    overlay = TrainingOverlay()
    text = overlay.lines(
        {
            "update": 3,
            "episode": 4,
            "wins": 1,
            "losses": 2,
            "recent_win_rate": 0.25,
            "seed": 99,
            "sun_loss_ships": 12,
            "wasted_ships": 15,
            "invalid_actions": 2,
            "repeated_seed_count": 5,
            "checkpoint": "policy_update_3.npz",
        }
    )
    joined = "\n".join(text)
    assert "seed=99" in joined
    assert "sun_loss=12" in joined
    assert "wasted=15" in joined
    assert "invalid=2" in joined
    assert "repeat_seed=5" in joined
    assert "ckpt=policy_update_3.npz" in joined


def test_minimal_status_line_shows_only_episode_wins_losses_and_win_rate():
    line = minimal_status_line(
        {
            "update": 1,
            "episode": 20,
            "wins": 11,
            "losses": 8,
            "recent_win_rate": 0.579,
            "seed": 1875941738,
            "sun_loss_ships": 1560,
            "wasted_ships": 4178,
            "invalid_actions": 0,
            "repeated_seed_count": 0,
            "episode_reward": 99,
        }
    )

    assert line == "episode=20  wins=11  losses=8  win_rate=0.579"
    assert "reward" not in line
    assert "OUR" not in line
    assert "seed" not in line
    assert "sun_loss" not in line
    assert "wasted" not in line
    assert "invalid" not in line


def test_minimal_status_line_hides_episode_rate_from_pygame_hud():
    line = minimal_status_line({"episode": 12, "episodes_per_second": 18.75, "recent_win_rate": 0.5})

    assert line == "episode=12  wins=0  losses=0  win_rate=0.500"
    assert "eps/s" not in line


def test_agent_identity_lines_show_our_agent_color_at_top():
    lines = agent_identity_lines(our_player=0)

    assert lines[0]["label"] == "OUR AGENT / Player 0"
    assert lines[0]["color"] == PLAYER_COLORS[0]
    assert lines[1]["label"] == "SELF-PLAY OPPONENT / Player 1"
    assert lines[1]["color"] == PLAYER_COLORS[1]


def test_summarize_observation_counts_player_assets():
    obs = {
        "planets": [
            [0, 0, 20.0, 30.0, 4.0, 15, 2],
            [1, 1, 80.0, 30.0, 5.0, 30, 4],
            [2, -1, 45.0, 70.0, 3.0, 9, 1],
            [3, 0, 15.0, 60.0, 2.0, 5, 3],
        ],
        "fleets": [
            [0, 0, 25.0, 35.0, 0.0, 0, 7],
            [1, 1, 75.0, 35.0, 3.14, 1, 11],
            [2, 0, 30.0, 40.0, 1.0, 0, 13],
        ],
    }

    summary = summarize_observation(obs, n_players=2)

    assert summary[0] == {"ships": 40, "production": 5, "planets": 2, "fleet_ships": 20}
    assert summary[1] == {"ships": 41, "production": 4, "planets": 1, "fleet_ships": 11}


@pytest.mark.parametrize(
    ("window", "expected"),
    [((900, 700), (100, 0, 700, 700)), ((700, 900), (0, 100, 700, 700)), ((800, 800), (0, 0, 800, 800))],
)
def test_board_rect_is_centered_square(window, expected):
    assert board_rect(*window) == expected


def test_board_rect_minimal_mode_keeps_top_status_room():
    x, y, w, h = board_rect(900, 700, top_reserved=42)
    assert y >= 42
    assert w == h
    assert y + h <= 700


def test_burn_in_offset_moves_after_interval_but_stays_small():
    assert burn_in_offset(now_seconds=0, interval_seconds=1800) == (0, 0)
    assert burn_in_offset(now_seconds=1800, interval_seconds=1800) != (0, 0)
    dx, dy = burn_in_offset(now_seconds=1800 * 7, interval_seconds=1800)
    assert -18 <= dx <= 18
    assert -18 <= dy <= 18


def test_world_to_screen_maps_board_coords_into_square_play_rect():
    rect = board_rect(900, 700)

    assert world_to_screen(0.0, 0.0, rect) == (100, 0)
    assert world_to_screen(100.0, 100.0, rect) == (800, 700)
    assert world_to_screen(50.0, 50.0, rect) == (450, 350)


def test_viewer_display_flags_prefers_resizable_windowed_and_fullscreen_modes():
    from src.ow.rl import pygame_viewer as viewer_module

    pg = _FakePygame()
    flags_fn = getattr(viewer_module, "viewer_display_flags")
    assert flags_fn(pg, fullscreen=False, resizable=True) == pg.RESIZABLE
    assert flags_fn(pg, fullscreen=True, resizable=True) == pg.FULLSCREEN


def test_pygame_training_viewer_requests_resizable_window(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame()
    monkeypatch.setattr(viewer_module, "pygame", fake_pg, raising=False)

    viewer_module.PygameTrainingViewer(enabled=True, width=640, height=480, fullscreen=False, **{"resizable": True})

    assert fake_pg.display.calls[-1] == ((640, 480), fake_pg.RESIZABLE)


def test_videoresize_event_updates_viewer_dimensions(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame(events=[_FakeEvent(_FakePygame.VIDEORESIZE, w=1234, h=777)])
    monkeypatch.setattr(viewer_module, "pygame", fake_pg, raising=False)
    viewer = viewer_module.PygameTrainingViewer(enabled=True, width=900, height=700, fullscreen=False)

    viewer.draw({"planets": [], "fleets": []}, {})

    assert (viewer.width, viewer.height) == (1234, 777)


def test_f11_toggles_from_fullscreen_to_resizable_window(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame(events=[_FakeEvent(_FakePygame.KEYDOWN, key=_FakePygame.K_F11)])
    monkeypatch.setattr(viewer_module, "pygame", fake_pg, raising=False)
    viewer = viewer_module.PygameTrainingViewer(enabled=True, width=800, height=600, fullscreen=True, **{"resizable": True})

    viewer.draw({"planets": [], "fleets": []}, {})

    assert viewer.fullscreen is False
    assert fake_pg.display.calls[-1] == ((800, 600), fake_pg.RESIZABLE)


def test_m_key_iconifies_window(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame(events=[_FakeEvent(_FakePygame.KEYDOWN, key=_FakePygame.K_m)])
    monkeypatch.setattr(viewer_module, "pygame", fake_pg, raising=False)
    viewer = viewer_module.PygameTrainingViewer(enabled=True, width=800, height=600, fullscreen=False)

    viewer.draw({"planets": [], "fleets": []}, {})

    assert fake_pg.display.iconify_calls == 1


def test_minimal_draw_skips_board_planets_fleets_and_identity_for_speed(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame()
    fake_pg.display.flip = lambda: None

    viewer = viewer_module.PygameTrainingViewer.__new__(viewer_module.PygameTrainingViewer)
    viewer.enabled = True
    viewer.minimal = True
    viewer.width = 800
    viewer.height = 80
    viewer._pygame = fake_pg
    viewer._screen = _FakeScreen((800, 80))
    viewer._planet_history = []

    calls = []
    viewer._draw_status_panels = lambda *args: calls.append("status")
    viewer._draw_board_shell = lambda *args: calls.append("board")
    viewer._draw_sun = lambda *args: calls.append("sun")
    viewer._draw_planet_trails = lambda *args: calls.append("trails")
    viewer._draw_planets = lambda *args: calls.append("planets")
    viewer._draw_fleets = lambda *args: calls.append("fleets")

    viewer.draw({"planets": [], "fleets": []}, {"update": 1})

    assert calls == ["status"]


def test_viewer_draw_disables_viewer_instead_of_raising_when_display_crashes(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame()
    viewer = viewer_module.PygameTrainingViewer.__new__(viewer_module.PygameTrainingViewer)
    viewer.enabled = True
    viewer.minimal = True
    viewer.width = 800
    viewer.height = 80
    viewer._pygame = fake_pg
    viewer._screen = _CrashingScreen((800, 80))
    viewer._planet_history = []

    viewer.draw({"planets": [], "fleets": []}, {"update": 1})

    assert viewer.enabled is False


def test_viewer_can_pump_events_without_redrawing_for_responsiveness(monkeypatch):
    from src.ow.rl import pygame_viewer as viewer_module

    fake_pg = _FakePygame(events=[_FakeEvent(_FakePygame.KEYDOWN, key=_FakePygame.K_m)])
    monkeypatch.setattr(viewer_module, "pygame", fake_pg, raising=False)
    viewer = viewer_module.PygameTrainingViewer(enabled=True, width=800, height=80, fullscreen=False)

    viewer.pump_events()

    assert fake_pg.display.iconify_calls == 1
    assert fake_pg.display.flip_calls == 0
