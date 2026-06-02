from __future__ import annotations

import math
from dataclasses import dataclass


pygame = None


PLAYER_COLORS = [(58, 166, 255), (255, 90, 90), (58, 209, 122), (255, 210, 58)]
NEUTRAL_COLOR = (136, 136, 136)
BACKGROUND_COLOR = (7, 11, 26)
PANEL_COLOR = (19, 26, 48)
GRID_COLOR = (31, 42, 74)
TEXT_COLOR = (232, 237, 247)
MUTED_TEXT_COLOR = (138, 150, 184)
ACCENT_COLOR = (255, 217, 102)
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
TRAIL_LEN = 30


PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
PLANET_SHIPS = 5
PLANET_PRODUCTION = 6
FLEET_OWNER = 1
FLEET_X = 2
FLEET_Y = 3
FLEET_ANGLE = 4
FLEET_SHIPS = 6


def _rgb(owner: int) -> tuple[int, int, int]:
    if owner < 0:
        return NEUTRAL_COLOR
    return PLAYER_COLORS[owner % len(PLAYER_COLORS)]


def _with_alpha(color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], alpha)


def board_rect(width: int, height: int, margin: int = 0, top_reserved: int = 0) -> tuple[int, int, int, int]:
    """Return a centered square play rectangle as ``(x, y, w, h)``."""
    usable_height = max(1, int(height) - int(top_reserved))
    side = max(1, min(int(width), usable_height) - int(margin) * 2)
    x = (int(width) - side) // 2
    y = int(top_reserved) + (usable_height - side) // 2
    return (x, y, side, side)


def burn_in_offset(now_seconds: float, interval_seconds: int = 1800, radius: int = 18) -> tuple[int, int]:
    if interval_seconds <= 0:
        return (0, 0)
    slot = int(float(now_seconds) // int(interval_seconds))
    if slot <= 0:
        return (0, 0)
    # Deterministic low-amplitude orbit so static HUD text does not sit on the
    # exact same pixels for hours on older displays.
    angle = (slot % 12) * (math.tau / 12.0)
    return (int(round(math.cos(angle) * radius)), int(round(math.sin(angle) * radius)))


def viewer_display_flags(pg, fullscreen: bool = False, resizable: bool = True) -> int:
    if fullscreen:
        return int(getattr(pg, "FULLSCREEN", 0))
    if resizable:
        return int(getattr(pg, "RESIZABLE", 0))
    return 0


def world_to_screen(x: float, y: float, rect: tuple[int, int, int, int]) -> tuple[int, int]:
    """Map Orbit Wars 0..100 board coordinates into a square play rectangle."""
    rx, ry, rw, rh = rect
    return (int(round(rx + (float(x) / BOARD_SIZE) * rw)), int(round(ry + (float(y) / BOARD_SIZE) * rh)))


def fleet_size_multiplier(ships: int | float) -> float:
    return min(1.7, 0.4 + math.log(max(1.0, float(ships))) * 0.17)


def summarize_observation(obs: dict, n_players: int = 2) -> list[dict[str, int]]:
    """Summarize visible ships, production, planet counts, and fleet ships per player."""
    summary = [dict(ships=0, production=0, planets=0, fleet_ships=0) for _ in range(max(0, int(n_players)))]
    for planet in obs.get("planets", []) or []:
        if len(planet) <= PLANET_SHIPS:
            continue
        owner = int(planet[PLANET_OWNER])
        if 0 <= owner < len(summary):
            ships = int(planet[PLANET_SHIPS])
            production = int(planet[PLANET_PRODUCTION]) if len(planet) > PLANET_PRODUCTION else 0
            summary[owner]["ships"] += ships
            summary[owner]["production"] += production
            summary[owner]["planets"] += 1
    for fleet in obs.get("fleets", []) or []:
        if len(fleet) <= FLEET_SHIPS:
            continue
        owner = int(fleet[FLEET_OWNER])
        if 0 <= owner < len(summary):
            ships = int(fleet[FLEET_SHIPS])
            summary[owner]["ships"] += ships
            summary[owner]["fleet_ships"] += ships
    return summary


def agent_identity_lines(our_player: int = 0) -> list[dict[str, object]]:
    opponent = 1 if int(our_player) == 0 else 0
    return [
        {"label": f"OUR AGENT / Player {int(our_player)}", "color": PLAYER_COLORS[int(our_player) % len(PLAYER_COLORS)]},
        {"label": f"SELF-PLAY OPPONENT / Player {opponent}", "color": PLAYER_COLORS[opponent % len(PLAYER_COLORS)]},
    ]


@dataclass
class TrainingOverlay:
    def lines(self, metrics: dict) -> list[str]:
        lines = [
            f"update={int(metrics.get('update', 0))}",
            f"episode={int(metrics.get('episode', 0))}",
            f"wins={int(metrics.get('wins', 0))}",
            f"losses={int(metrics.get('losses', 0))}",
            f"win_rate={float(metrics.get('recent_win_rate', 0.0)):.3f}",
        ]
        if "episode_reward" in metrics:
            lines.append(f"reward={float(metrics.get('episode_reward', 0.0)):.3f}")
        if "seed" in metrics:
            lines.append(f"seed={int(metrics.get('seed', 0))}")
        if "sun_loss_ships" in metrics:
            lines.append(f"sun_loss={int(metrics.get('sun_loss_ships', 0))}")
        if "wasted_ships" in metrics:
            lines.append(f"wasted={int(metrics.get('wasted_ships', 0))}")
        if "invalid_actions" in metrics:
            lines.append(f"invalid={int(metrics.get('invalid_actions', 0))}")
        if "repeated_seed_count" in metrics:
            lines.append(f"repeat_seed={int(metrics.get('repeated_seed_count', 0))}")
        if "checkpoint" in metrics:
            lines.append(f"ckpt={metrics.get('checkpoint')}")
        return lines


def minimal_status_line(metrics: dict) -> str:
    """Single ultra-light HUD line: only episode count and win-rate stats."""
    return "  ".join(
        [
            f"episode={int(metrics.get('episode', 0))}",
            f"wins={int(metrics.get('wins', 0))}",
            f"losses={int(metrics.get('losses', 0))}",
            f"win_rate={float(metrics.get('recent_win_rate', 0.0)):.3f}",
        ]
    )


class PygameTrainingViewer:
    def __init__(
        self,
        enabled: bool = True,
        width: int = 900,
        height: int = 700,
        fullscreen: bool = False,
        resizable: bool = True,
        minimal: bool = True,
        burn_in_shift_seconds: int = 1800,
    ):
        self.enabled = enabled
        self.overlay = TrainingOverlay()
        self.width = width
        self.height = height
        self.fullscreen = bool(fullscreen)
        self.resizable = bool(resizable)
        self._windowed_size = (int(width), int(height))
        self.minimal = bool(minimal)
        self.burn_in_shift_seconds = int(burn_in_shift_seconds)
        self._pygame = None
        self._screen = None
        self._font = None
        self._small_font = None
        self._title_font = None
        self._planet_history: list[list] = []
        if not enabled:
            return
        try:
            pg = self._load_pygame()

            pg.init()
            self._pygame = pg
            self._screen = self._set_display_mode((width, height))
            if self.fullscreen:
                self.width, self.height = self._screen.get_size()
            pg.display.set_caption("Orbit Wars RL Viewer")
            self._font = pg.font.SysFont("Verdana", 18)
            self._small_font = pg.font.SysFont("Verdana", 13)
            self._title_font = pg.font.SysFont("Georgia", 22, bold=True)
        except Exception:
            self.enabled = False

    def _load_pygame(self):
        global pygame
        if pygame is None:
            import pygame as imported_pygame

            pygame = imported_pygame
        return pygame

    def _display_flags(self) -> int:
        return viewer_display_flags(self._pygame, fullscreen=self.fullscreen, resizable=self.resizable)

    def _set_display_mode(self, size: tuple[int, int]):
        screen = self._pygame.display.set_mode((int(size[0]), int(size[1])), self._display_flags())
        self.width, self.height = screen.get_size()
        return screen

    def _toggle_fullscreen(self) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self._screen = self._set_display_mode(self._windowed_size)
        else:
            self._windowed_size = (int(self.width), int(self.height))
            self.fullscreen = True
            self._screen = self._set_display_mode((self.width, self.height))

    def _handle_event(self, event) -> bool:
        pg = self._pygame
        if event.type == getattr(pg, "QUIT", object()):
            self.close()
            return False
        if event.type == getattr(pg, "VIDEORESIZE", object()):
            new_size = (max(1, int(getattr(event, "w", self.width))), max(1, int(getattr(event, "h", self.height))))
            self.fullscreen = False
            self._windowed_size = new_size
            self._screen = self._set_display_mode(new_size)
        if event.type == getattr(pg, "KEYDOWN", object()):
            key = getattr(event, "key", None)
            if key == getattr(pg, "K_F11", None):
                self._toggle_fullscreen()
            elif key == getattr(pg, "K_m", None):
                iconify = getattr(pg.display, "iconify", None)
                if callable(iconify):
                    iconify()
        return True


    def pump_events(self) -> None:
        if not self.enabled or self._pygame is None:
            return
        try:
            for event in self._pygame.event.get():
                if not self._handle_event(event):
                    return
        except Exception:
            self.enabled = False

    def draw(self, obs: dict, metrics: dict) -> None:
        if not self.enabled or self._pygame is None or self._screen is None:
            return
        pg = self._pygame
        try:
            self.pump_events()
            if not self.enabled:
                return
            if self.minimal:
                self._screen.fill((0, 0, 0))
                rect = board_rect(self.width, self.height, margin=0, top_reserved=0)
                self._draw_status_panels(pg, obs, metrics, rect)
                pg.display.flip()
                return

            self._planet_history.append(list(obs.get("planets", []) or []))
            if len(self._planet_history) > TRAIL_LEN:
                self._planet_history = self._planet_history[-TRAIL_LEN:]

            self._screen.fill((0, 0, 0) if self.minimal else BACKGROUND_COLOR)
            rect = board_rect(self.width, self.height, margin=8 if self.minimal else 0, top_reserved=44 if self.minimal else 0)
            self._draw_board_shell(pg, rect)
            self._draw_sun(pg, rect)
            self._draw_planet_trails(pg, rect)
            self._draw_planets(pg, obs, rect)
            self._draw_fleets(pg, obs, rect)
            self._draw_status_panels(pg, obs, metrics, rect)
            pg.display.flip()
        except Exception:
            # A local viewer must never interrupt training if SDL/display support flakes out.
            self.enabled = False

    def _scale(self, rect: tuple[int, int, int, int]) -> float:
        return rect[2] / BOARD_SIZE

    def _draw_board_shell(self, pg, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        pg.draw.rect(self._screen, (10, 15, 34), rect)
        pg.draw.rect(self._screen, GRID_COLOR, rect, 1)
        center = world_to_screen(*SUN_CENTER, rect)
        pg.draw.circle(self._screen, GRID_COLOR, center, int(round(50 * self._scale(rect))), 1)

    def _draw_sun(self, pg, rect: tuple[int, int, int, int]) -> None:
        cx, cy = world_to_screen(*SUN_CENTER, rect)
        radius = max(2, int(round(SUN_RADIUS * self._scale(rect))))
        halo = pg.Surface((radius * 5, radius * 5), pg.SRCALPHA)
        hc = radius * 5 // 2
        for i in range(radius * 2, 0, -2):
            alpha = max(8, int(80 * (i / (radius * 2)) ** 2))
            pg.draw.circle(halo, (255, 122, 26, alpha), (hc, hc), i)
        self._screen.blit(halo, (cx - hc, cy - hc))
        pg.draw.circle(self._screen, (255, 122, 26), (cx, cy), radius)
        pg.draw.circle(self._screen, (255, 210, 138), (cx, cy), max(1, radius // 2))
        pg.draw.circle(self._screen, (255, 230, 160), (cx - radius // 4, cy - radius // 4), max(1, radius // 5))

    def _draw_planet_trails(self, pg, rect: tuple[int, int, int, int]) -> None:
        if len(self._planet_history) < 2:
            return
        scale = self._scale(rect)
        for idx, planets in enumerate(self._planet_history[:-1]):
            alpha = int(18 + 45 * (idx / max(1, len(self._planet_history) - 1)))
            for planet in planets:
                if len(planet) <= PLANET_RADIUS:
                    continue
                sx, sy = world_to_screen(float(planet[PLANET_X]), float(planet[PLANET_Y]), rect)
                r = max(1, int(round(float(planet[PLANET_RADIUS]) * scale * 0.4)))
                color = _with_alpha(_rgb(int(planet[PLANET_OWNER])), alpha)
                surf = pg.Surface((r * 2 + 2, r * 2 + 2), pg.SRCALPHA)
                pg.draw.circle(surf, color, (r + 1, r + 1), r)
                self._screen.blit(surf, (sx - r - 1, sy - r - 1))

    def _draw_planets(self, pg, obs: dict, rect: tuple[int, int, int, int]) -> None:
        scale = self._scale(rect)
        for planet in obs.get("planets", []) or []:
            if len(planet) <= PLANET_SHIPS:
                continue
            owner = int(planet[PLANET_OWNER])
            x, y = float(planet[PLANET_X]), float(planet[PLANET_Y])
            r_world = float(planet[PLANET_RADIUS])
            ships = int(planet[PLANET_SHIPS])
            production = int(planet[PLANET_PRODUCTION]) if len(planet) > PLANET_PRODUCTION else 0
            sx, sy = world_to_screen(x, y, rect)
            r = max(3, int(round(r_world * scale)))
            pg.draw.circle(self._screen, _rgb(owner), (sx, sy), r)
            pg.draw.circle(self._screen, (245, 250, 255), (sx, sy), r, 1)
            self._draw_centered_text(str(ships), (sx, sy), TEXT_COLOR, max(10, int(r * 0.7)))
            self._draw_centered_text(f"+{production}", (sx, sy + r + 9), ACCENT_COLOR, 12)

    def _draw_fleets(self, pg, obs: dict, rect: tuple[int, int, int, int]) -> None:
        scale = self._scale(rect)
        for fleet in obs.get("fleets", []) or []:
            if len(fleet) <= FLEET_SHIPS:
                continue
            owner = int(fleet[FLEET_OWNER])
            x, y = float(fleet[FLEET_X]), float(fleet[FLEET_Y])
            angle = float(fleet[FLEET_ANGLE]) if len(fleet) > FLEET_ANGLE else 0.0
            ships = int(fleet[FLEET_SHIPS])
            sx, sy = world_to_screen(x, y, rect)
            size = fleet_size_multiplier(ships) * scale
            points = [(1.2 * size, 0), (-0.8 * size, 0.7 * size), (-0.8 * size, -0.7 * size)]
            ca, sa = math.cos(angle), math.sin(angle)
            rotated = [(int(round(sx + px * ca - py * sa)), int(round(sy + px * sa + py * ca))) for px, py in points]
            pg.draw.polygon(self._screen, _rgb(owner), rotated)
            pg.draw.polygon(self._screen, (245, 250, 255), rotated, 1)
            label = self._small_font.render(str(ships), True, TEXT_COLOR)
            self._screen.blit(label, (sx + int(1.3 * size), sy - label.get_height() // 2))

    def _draw_status_panels(self, pg, obs: dict, metrics: dict, rect: tuple[int, int, int, int]) -> None:
        elapsed = pg.time.get_ticks() / 1000.0
        dx, dy = burn_in_offset(elapsed, self.burn_in_shift_seconds)
        if self.minimal:
            bar = (max(4, 10 + dx), max(4, 6 + dy), min(self.width - 8, self.width - 20), 34)
            self._draw_panel(pg, bar)
            text = self._small_font.render(minimal_status_line(metrics)[:180], True, ACCENT_COLOR)
            self._screen.blit(text, (bar[0] + 10, bar[1] + 7))
            return

        left_panel = (8, 8, max(86, rect[0] - 16), self.height - 16) if rect[0] > 24 else (8, 8, 180, 132)
        right_width = max(86, self.width - (rect[0] + rect[2]) - 16)
        right_panel = (rect[0] + rect[2] + 8, 8, right_width, self.height - 16) if right_width > 90 else None
        self._draw_panel(pg, left_panel)
        title = self._title_font.render("Orbit Wars", True, TEXT_COLOR)
        self._screen.blit(title, (left_panel[0] + 10, left_panel[1] + 10))
        y = left_panel[1] + 42
        for item in agent_identity_lines(our_player=0):
            pg.draw.rect(self._screen, item["color"], (left_panel[0] + 10, y + 3, 12, 12))
            text = self._small_font.render(str(item["label"]), True, TEXT_COLOR)
            self._screen.blit(text, (left_panel[0] + 28, y))
            y += 18
        y += 6
        for line in self.overlay.lines(metrics):
            text = self._small_font.render(line, True, ACCENT_COLOR)
            self._screen.blit(text, (left_panel[0] + 10, y))
            y += 18

        summary = summarize_observation(obs, n_players=max(2, len(PLAYER_COLORS)))
        target_panel = right_panel or (self.width - 188, self.height - 122, 180, 114)
        self._draw_panel(pg, target_panel)
        y = target_panel[1] + 10
        heading = self._small_font.render("VISIBLE ASSETS", True, MUTED_TEXT_COLOR)
        self._screen.blit(heading, (target_panel[0] + 10, y))
        y += 20
        for pid, stats in enumerate(summary[:2]):
            dot = (target_panel[0] + 14, y + 7)
            pg.draw.circle(self._screen, PLAYER_COLORS[pid], dot, 4)
            line = f"P{pid} ships {stats['ships']}  prod {stats['production']}  planets {stats['planets']}  fleets {stats['fleet_ships']}"
            text = self._small_font.render(line, True, TEXT_COLOR)
            self._screen.blit(text, (target_panel[0] + 24, y))
            y += 18

    def _draw_panel(self, pg, rect: tuple[int, int, int, int]) -> None:
        panel = pg.Surface((rect[2], rect[3]), pg.SRCALPHA)
        panel.fill((19, 26, 48, 218))
        self._screen.blit(panel, (rect[0], rect[1]))
        pg.draw.rect(self._screen, (42, 53, 96), rect, 1)

    def _draw_centered_text(self, text: str, center: tuple[int, int], color: tuple[int, int, int], size: int) -> None:
        font = self._font if 14 <= size <= 20 else self._small_font
        rendered = font.render(text, True, color)
        self._screen.blit(rendered, (center[0] - rendered.get_width() // 2, center[1] - rendered.get_height() // 2))

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self.enabled = False
