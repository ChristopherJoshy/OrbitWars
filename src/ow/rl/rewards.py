from __future__ import annotations

PLANET_OWNER = 1
PLANET_SHIPS = 5
PLANET_PROD = 6


def _prod_and_count(obs: dict, player: int) -> tuple[float, float, float]:
    my_prod = 0.0
    opp_prod = 0.0
    my_count = 0.0
    for p in obs.get("planets", []) or []:
        owner = int(p[PLANET_OWNER])
        if owner == player:
            my_prod += float(p[PLANET_PROD])
            my_count += 1.0
        elif owner >= 0:
            opp_prod += float(p[PLANET_PROD])
    return my_prod, opp_prod, my_count


def shaped_reward(
    prev_obs: dict,
    curr_obs: dict,
    action_count: int,
    terminal_reward: float,
    invalid_action: bool = False,
    sun_loss_ships: int = 0,
    wasted_ships: int = 0,
    failed_attack_ships: int = 0,
    oversend_ships: int = 0,
    repeated_failure: bool = False,
) -> float:
    player = int(curr_obs.get("player", prev_obs.get("player", 0)))
    prev_my, prev_opp, prev_count = _prod_and_count(prev_obs, player)
    cur_my, cur_opp, cur_count = _prod_and_count(curr_obs, player)
    reward = 0.1 * (cur_my - prev_my)
    reward += 0.05 * (cur_count - prev_count)
    reward += -0.05 * (cur_opp - prev_opp)
    reward += 0.01 if action_count > 0 else -0.01
    if invalid_action:
        reward -= 0.05
    reward -= 0.035 * float(max(0, sun_loss_ships))
    reward -= 0.02 * float(max(0, wasted_ships))
    reward -= 0.015 * float(max(0, failed_attack_ships))
    reward -= 0.005 * float(max(0, oversend_ships))
    if repeated_failure:
        reward -= 0.15
    reward += float(terminal_reward)
    return float(reward)


def mistake_penalty(components: dict, mode: str = "2p") -> float:
    multiplier = 1.5 if mode == "4p" else 1.0
    return -multiplier * (
        0.07 * float(max(0, int(components.get("sun_loss_ships", 0))))
        + 0.045 * float(max(0, int(components.get("wasted_ships", 0))))
        + 0.1 * float(max(0, int(components.get("invalid_actions", 0))))
        + 0.04 * float(max(0, int(components.get("wrong_target_ships", 0))))
        + 0.055 * float(max(0, int(components.get("out_of_bounds_ships", 0))))
        + 0.025 * float(max(0, int(components.get("failed_attack_ships", 0))))
        + 0.01 * float(max(0, int(components.get("oversend_ships", 0))))
    )
