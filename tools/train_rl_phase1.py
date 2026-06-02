from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from kaggle_environments import make

from src.ow.rl.action_space import build_action_candidates, candidate_to_env_action, estimate_candidate_mistakes
from src.ow.rl.metrics import MetricsWriter, RollingStats
from src.ow.rl.phase_gates import ship_waste_loss
from src.ow.rl.policy import LinearActorCritic
from src.ow.rl.pygame_viewer import PygameTrainingViewer
from src.ow.rl.rewards import mistake_penalty

RL_FEATURE_DIM = 32


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["2p", "4p", "mixed"], default="2p")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--updates", type=int, default=1)
    p.add_argument("--episode-steps", type=int, default=150)
    p.add_argument("--step-penalty-weight", type=float, default=0.001)
    p.add_argument("--terminal-reward-scale", type=float, default=2.5)
    p.add_argument("--max-mistake-penalty", type=float, default=3.0)
    p.add_argument("--fourp-rank-reward-scale", type=float, default=2.5)
    p.add_argument("--asset-margin-reward-scale", type=float, default=0.25)
    p.add_argument("--dense-reward-scale", type=float, default=0.0)
    p.add_argument("--reward-gamma", type=float, default=0.99)
    p.add_argument("--opponent-pool-size", type=int, default=8)
    p.add_argument("--opponent-snapshot-every", type=int, default=5)
    p.add_argument("--policy-lr", type=float, default=0.01)
    p.add_argument("--gate-streak-lr-multiplier", type=float, default=0.25)
    p.add_argument("--gate-streak-high-threshold", type=int, default=3)
    p.add_argument("--gate-streak-high-lr-multiplier", type=float, default=0.1)
    p.add_argument("--gate-streak-lr-min-streak", type=int, default=1)
    p.add_argument("--rollback-lr-cooldown-updates", type=int, default=2)
    p.add_argument("--rollback-lr-cooldown-enabled", dest="rollback_lr_cooldown_enabled", action="store_true")
    p.add_argument("--no-rollback-lr-cooldown", dest="rollback_lr_cooldown_enabled", action="store_false")
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--deterministic-stream-weight", type=int, default=1)
    p.add_argument("--winning-sample-weight", type=int, default=1)
    p.add_argument("--losing-sample-weight", type=int, default=1)
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--train-all-players", action="store_true")
    p.add_argument("--train-seat", choices=["player0", "alternating"], default="player0")
    p.add_argument("--strict-gate-opponents-all-streams", action="store_true")
    p.add_argument("--frozen-opponent-agent", default=None)
    p.add_argument("--diverse-training-seeds", action="store_true")
    p.add_argument("--gate-advance-win-streak", type=int, default=3)
    p.add_argument("--gate-policy-rollback", dest="gate_policy_rollback", action="store_true")
    p.add_argument("--no-gate-policy-rollback", dest="gate_policy_rollback", action="store_false")
    p.add_argument("--promote-gate-opponent-on-advance", dest="promote_gate_opponent_on_advance", action="store_true")
    p.add_argument("--no-promote-gate-opponent-on-advance", dest="promote_gate_opponent_on_advance", action="store_false")
    p.add_argument("--run-dir", type=Path, default=Path("reports"))
    p.add_argument("--allow-existing-run-dir", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--initial-policy", type=Path, default=None)
    p.add_argument("--initial-gate-opponent-policy", type=Path, default=None)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--viewer", dest="no_viewer", action="store_false")
    p.add_argument("--no-viewer", dest="no_viewer", action="store_true")
    p.add_argument("--fullscreen", dest="fullscreen", action="store_true")
    p.add_argument("--windowed", dest="fullscreen", action="store_false")
    p.add_argument("--resizable-window", dest="resizable_window", action="store_true", default=True)
    p.add_argument("--fixed-window", dest="resizable_window", action="store_false")
    p.add_argument("--minimal-viewer", action="store_true", default=True)
    p.add_argument("--rich-viewer", dest="minimal_viewer", action="store_false")
    p.add_argument("--viewer-draw-interval-seconds", type=float, default=30.0)
    p.add_argument("--viewer-event-poll-interval-seconds", type=float, default=1.0)
    p.add_argument("--burn-in-shift-seconds", type=int, default=1800)
    p.set_defaults(no_viewer=True)
    p.set_defaults(gate_policy_rollback=True)
    p.set_defaults(promote_gate_opponent_on_advance=False)
    p.set_defaults(rollback_lr_cooldown_enabled=True)
    args = p.parse_args(argv)
    if args.frozen_opponent_agent is not None and args.train_all_players:
        p.error("--frozen-opponent-agent cannot be combined with --train-all-players")
    if args.frozen_opponent_agent is not None and args.initial_gate_opponent_policy is not None:
        p.error("--frozen-opponent-agent cannot be combined with --initial-gate-opponent-policy")
    return args


def load_agent_ref(ref: str):
    if ref.endswith(".py") or "/" in ref:
        path = Path(ref).resolve()
        module_name = f"train_rl_frozen_agent_{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for agent file: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ImportError(f"Failed to import agent file: {path}") from exc
        if not hasattr(module, "agent"):
            raise AttributeError(f"Agent file does not define 'agent': {path}")
        agent_fn = getattr(module, "agent")
        if not callable(agent_fn):
            raise AttributeError(f"'agent' is not callable in: {path}")
        return agent_fn
    module_name, func_name = ref.rsplit(":", 1) if ":" in ref else (ref, "agent")
    module = importlib.import_module(module_name)
    agent_fn = getattr(module, func_name)
    if not callable(agent_fn):
        raise AttributeError(f"'{func_name}' is not callable in: {module_name}")
    return agent_fn


def build_run_dir(base: Path, mode: str, allow_existing: bool = False) -> Path:
    base = Path(base)
    if base.name.startswith(("rl_", "2p_", "4p_")):
        out = base
    else:
        out = base / f"{mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    if out.exists() and not allow_existing:
        raise FileExistsError(f"refusing to reuse existing run dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


def create_initial_policy(
    feature_dim: int = RL_FEATURE_DIM,
    seed: int = 0,
    initial_policy_path: Path | None = None,
) -> LinearActorCritic:
    if initial_policy_path is not None:
        return LinearActorCritic.load(initial_policy_path, feature_dim=feature_dim)
    return LinearActorCritic(feature_dim=feature_dim, seed=seed)


def create_initial_gate_opponent_policy(
    learner_policy: LinearActorCritic,
    feature_dim: int = RL_FEATURE_DIM,
    seed: int = 0,
    initial_gate_opponent_path: Path | None = None,
) -> LinearActorCritic:
    if initial_gate_opponent_path is not None:
        return LinearActorCritic.load(initial_gate_opponent_path, feature_dim=feature_dim)
    return learner_policy.clone()


def mode_player_count(mode: str) -> int:
    if mode == "2p":
        return 2
    if mode in {"4p", "mixed"}:
        return 4
    raise ValueError(f"unsupported mode: {mode}")


def episode_mode(mode: str, episode: int) -> str:
    if mode in {"2p", "4p"}:
        return mode
    if mode == "mixed":
        return "2p" if int(episode) % 2 == 1 else "4p"
    raise ValueError(f"unsupported mode: {mode}")


def episode_player_count(mode: str, episode: int) -> int:
    return mode_player_count(episode_mode(mode, episode))


def player_is_trainable(current_mode: str, player_id: int) -> bool:
    if current_mode in {"2p", "4p"}:
        return int(player_id) == 0
    raise ValueError(f"unsupported mode: {current_mode}")


def training_focus_player_id(current_mode: str, global_episode: int = 1, train_seat: str = "player0") -> int:
    if current_mode == "2p" and train_seat == "alternating":
        return 0 if int(global_episode) % 2 == 1 else 1
    if current_mode in {"2p", "4p"}:
        return 0
    raise ValueError(f"unsupported mode: {current_mode}")


def training_player_is_trainable(
    current_mode: str,
    player_id: int,
    train_all_players: bool = False,
    train_seat: str = "player0",
    global_episode: int = 1,
) -> bool:
    if current_mode == "4p" and bool(train_all_players):
        return True
    if current_mode == "2p" and train_seat == "alternating":
        return int(player_id) == training_focus_player_id(current_mode, global_episode=global_episode, train_seat=train_seat)
    return player_is_trainable(current_mode, player_id)


def gate_eval_player_is_trainable(current_mode: str, player_id: int) -> bool:
    return player_is_trainable(current_mode, player_id)


def global_episode_number(update: int, episode: int, episodes_per_update: int) -> int:
    return (max(1, int(update)) - 1) * max(1, int(episodes_per_update)) + max(1, int(episode))


def should_draw_viewer(now_seconds: float, last_draw_seconds: float | None, interval_seconds: float) -> bool:
    interval_seconds = float(interval_seconds)
    if interval_seconds <= 0:
        return True
    if last_draw_seconds is None:
        return True
    return float(now_seconds) - float(last_draw_seconds) >= interval_seconds


def should_poll_viewer_events(now_seconds: float, last_poll_seconds: float | None, interval_seconds: float) -> bool:
    interval_seconds = float(interval_seconds)
    if interval_seconds <= 0:
        return True
    if last_poll_seconds is None:
        return True
    return float(now_seconds) - float(last_poll_seconds) >= interval_seconds


def should_advance_training_seed_during_episode(eval_every: int) -> bool:
    return int(eval_every) <= 0


def training_actor_should_be_deterministic(trainable: bool, eval_every: int, rollout_stream_index: int = 0) -> bool:
    if int(eval_every) <= 0:
        return False
    if not bool(trainable):
        return True
    return int(rollout_stream_index) == 0


def gate_actor_should_be_deterministic(trainable: bool) -> bool:
    return True


def actor_rng_seed_for_episode(episode_seed: int, player_id: int, rollout_index: int = 0) -> int:
    return int(
        (
            int(episode_seed) * 1_000_003
            + int(player_id) * 9_176
            + int(rollout_index) * 104_729
            + 13_579
        )
        % (2**32)
    )


def training_episode_seed(
    base_seed: int,
    gate_seed: int,
    global_episode: int,
    rollout_stream_index: int,
    diverse_training_seeds: bool = False,
) -> int:
    gate_seed = int(gate_seed)
    if not bool(diverse_training_seeds):
        return gate_seed
    digest = hashlib.blake2s(
        (
            f"training-episode:{int(base_seed)}:{gate_seed}:"
            f"{int(global_episode)}:{int(rollout_stream_index)}"
        ).encode("utf-8"),
        digest_size=4,
    ).hexdigest()
    return int(digest, 16) % (2**31 - 2) + 1


def rollout_stream_index_for_episode(episode: int, episodes_per_update: int, eval_every: int) -> int:
    episode = max(1, int(episode))
    return episode - 1


def deterministic_stream_sample_weight(rollout_stream_index: int, eval_every: int, configured_weight: int) -> int:
    if int(eval_every) <= 0 or int(rollout_stream_index) != 0:
        return 1
    return max(1, int(configured_weight))


def outcome_sample_weight(
    base_weight: int,
    won: bool,
    winning_sample_weight: int = 1,
    losing_sample_weight: int = 1,
) -> int:
    base_weight = max(1, int(base_weight))
    if not bool(won):
        return base_weight * max(0, int(losing_sample_weight))
    return base_weight * max(1, int(winning_sample_weight))


def update_policy_epochs(
    policy,
    batch: dict,
    lr: float,
    entropy_coef: float,
    value_coef: float,
    ppo_epochs: int = 1,
) -> dict:
    metrics = {}
    for _ in range(max(1, int(ppo_epochs))):
        metrics = policy.update_batch(
            batch,
            lr=lr,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
    return {**metrics, "ppo_epochs": max(1, int(ppo_epochs))}


def rollout_signal_metrics(rollouts: list[dict]) -> dict:
    deterministic = [r for r in rollouts if bool(r.get("deterministic_rollout"))]
    stochastic = [r for r in rollouts if not bool(r.get("deterministic_rollout"))]
    stochastic_wins = sum(1 for r in stochastic if bool(r.get("won")))
    stochastic_losses = len(stochastic) - stochastic_wins
    weighted_winning_decisions = 0
    weighted_losing_decisions = 0
    for rollout in rollouts:
        if "weighted_winning_decisions" in rollout or "weighted_losing_decisions" in rollout:
            weighted_winning_decisions += int(rollout.get("weighted_winning_decisions", 0))
            weighted_losing_decisions += int(rollout.get("weighted_losing_decisions", 0))
            continue
        weighted_decisions = int(rollout.get("decision_count", 0)) * max(0, int(rollout.get("stream_weight", 1)))
        if bool(rollout.get("won")):
            weighted_winning_decisions += weighted_decisions
        else:
            weighted_losing_decisions += weighted_decisions
    deterministic_rank = float(deterministic[-1].get("rank", 0.0)) if deterministic else 0.0
    deterministic_won = bool(deterministic[-1].get("won")) if deterministic else False
    return {
        "train_deterministic_rank": deterministic_rank,
        "train_deterministic_won": deterministic_won,
        "train_stochastic_win_count": int(stochastic_wins),
        "train_stochastic_loss_count": int(stochastic_losses),
        "train_stochastic_win_rate": float(stochastic_wins) / max(1.0, float(len(stochastic))),
        "weighted_winning_decisions": int(weighted_winning_decisions),
        "weighted_losing_decisions": int(weighted_losing_decisions),
    }


def policy_lr_for_gate_streak(
    base_lr: float,
    clean_win_streak: int,
    streak_multiplier: float = 0.25,
    high_streak_threshold: int = 3,
    high_streak_multiplier: float = 0.1,
    min_streak_for_reduction: int = 1,
) -> float:
    if int(clean_win_streak) < max(1, int(min_streak_for_reduction)):
        return float(base_lr)
    if int(clean_win_streak) >= max(1, int(high_streak_threshold)):
        return float(base_lr) * max(0.0, float(high_streak_multiplier))
    if int(clean_win_streak) > 0:
        return float(base_lr) * max(0.0, float(streak_multiplier))
    return float(base_lr)


class RollbackLRCooldown:
    def __init__(self) -> None:
        self.remaining = 0
        self.source_streak = 0

    @property
    def active(self) -> bool:
        return self.remaining > 0 and self.source_streak > 0

    def arm(self, gate_policy_restored: bool, gate_streak_before: int, cooldown_updates: int) -> None:
        if not bool(gate_policy_restored):
            return
        gate_streak_before = max(0, int(gate_streak_before))
        cooldown_updates = max(0, int(cooldown_updates))
        if gate_streak_before <= 0 or cooldown_updates <= 0:
            return
        self.remaining = cooldown_updates
        self.source_streak = gate_streak_before

    def streak_for_update(self, clean_win_streak: int) -> int:
        if not self.active:
            return max(0, int(clean_win_streak))
        return max(max(0, int(clean_win_streak)), self.source_streak)

    def consume_update(self) -> None:
        if self.remaining <= 0:
            self.source_streak = 0
            return
        self.remaining -= 1
        if self.remaining <= 0:
            self.source_streak = 0


def action_margin_stats(policy: LinearActorCritic, decisions: list[dict]) -> dict:
    margins: list[float] = []
    for decision in decisions:
        features = np.asarray(decision.get("candidate_features"), dtype=np.float64)
        if features.ndim != 2 or features.shape[0] < 2:
            continue
        logits = np.asarray(policy.score(features), dtype=np.float64)
        top_two = np.partition(logits, -2)[-2:]
        margins.append(float(np.max(top_two) - np.min(top_two)))
    if not margins:
        return {
            "gate_action_margin_mean": 0.0,
            "gate_action_margin_p10": 0.0,
            "gate_action_margin_min": 0.0,
            "gate_low_margin_fraction_0p01": 0.0,
            "gate_low_margin_fraction_0p05": 0.0,
        }
    arr = np.asarray(margins, dtype=np.float64)
    return {
        "gate_action_margin_mean": float(np.mean(arr)),
        "gate_action_margin_p10": float(np.percentile(arr, 10)),
        "gate_action_margin_min": float(np.min(arr)),
        "gate_low_margin_fraction_0p01": float(np.mean(arr < 0.01)),
        "gate_low_margin_fraction_0p05": float(np.mean(arr < 0.05)),
    }


def argmax_flip_stats(reference_policy: LinearActorCritic, current_policy: LinearActorCritic, decisions: list[dict]) -> dict:
    total = 0
    flips = 0
    for decision in decisions:
        features = np.asarray(decision.get("candidate_features"), dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            continue
        before_idx, *_ = reference_policy.argmax(features)
        after_idx, *_ = current_policy.argmax(features)
        total += 1
        if int(before_idx) != int(after_idx):
            flips += 1
    rate = float(flips) / max(1.0, float(total))
    return {
        "gate_pre_update_argmax_flips": int(flips),
        "gate_pre_update_argmax_flip_rate": rate,
        "gate_pre_update_action_match_rate": 1.0 - rate,
    }


def remember_clean_gate_policy(
    clean_policies: dict[int, LinearActorCritic],
    seed: int,
    policy: LinearActorCritic,
    eval_clean_win: bool,
) -> None:
    if bool(eval_clean_win):
        clean_policies[int(seed)] = policy.clone()


def should_restore_clean_gate_policy(
    eval_clean_win: bool,
    gate_streak_before: int,
    clean_policy_available: bool,
    enabled: bool = True,
) -> bool:
    return bool(enabled) and (not bool(eval_clean_win)) and int(gate_streak_before) > 0 and bool(clean_policy_available)


def restore_clean_gate_policy(
    clean_policies: dict[int, LinearActorCritic],
    seed: int,
    policy: LinearActorCritic,
    eval_clean_win: bool,
    gate_streak_before: int,
    enabled: bool = True,
) -> tuple[LinearActorCritic, bool]:
    clean_policy = clean_policies.get(int(seed))
    if should_restore_clean_gate_policy(
        eval_clean_win=eval_clean_win,
        gate_streak_before=gate_streak_before,
        clean_policy_available=clean_policy is not None,
        enabled=enabled,
    ):
        assert clean_policy is not None
        return clean_policy.clone(), True
    return policy, False


def training_feature_flags(args, phase_mode: str) -> dict:
    return {
        "safe_action_candidates": True,
        "terminal_reward_scale": float(args.terminal_reward_scale),
        "max_mistake_penalty": float(args.max_mistake_penalty),
        "fourp_rank_reward_scale": float(args.fourp_rank_reward_scale),
        "asset_margin_reward_scale": float(args.asset_margin_reward_scale),
        "dense_reward_scale": float(args.dense_reward_scale),
        "reward_gamma": float(args.reward_gamma),
        "viewer_draw_interval_seconds": float(args.viewer_draw_interval_seconds),
        "viewer_event_poll_interval_seconds": float(args.viewer_event_poll_interval_seconds),
        "policy_lr": float(args.policy_lr),
        "gate_streak_lr_multiplier": float(args.gate_streak_lr_multiplier),
        "gate_streak_high_threshold": int(args.gate_streak_high_threshold),
        "gate_streak_high_lr_multiplier": float(args.gate_streak_high_lr_multiplier),
        "gate_streak_lr_min_streak": int(args.gate_streak_lr_min_streak),
        "rollback_lr_cooldown_enabled": bool(args.rollback_lr_cooldown_enabled),
        "rollback_lr_cooldown_updates": int(args.rollback_lr_cooldown_updates),
        "gate_policy_rollback": bool(args.gate_policy_rollback),
        "promote_gate_opponent_on_advance": bool(args.promote_gate_opponent_on_advance),
        "entropy_coef": float(args.entropy_coef),
        "value_coef": float(args.value_coef),
        "deterministic_stream_weight": int(args.deterministic_stream_weight),
        "winning_sample_weight": int(args.winning_sample_weight),
        "losing_sample_weight": int(args.losing_sample_weight),
        "ppo_epochs": int(args.ppo_epochs),
        "train_all_players": bool(args.train_all_players),
        "train_seat": str(args.train_seat),
        "strict_gate_opponents_all_streams": bool(args.strict_gate_opponents_all_streams),
        "frozen_opponent_agent": bool(args.frozen_opponent_agent),
        "diverse_training_seeds": bool(args.diverse_training_seeds),
        "gate_advance_win_streak": int(args.gate_advance_win_streak),
        "phase_mode_effective": str(phase_mode),
    }


def featurize(c: dict) -> np.ndarray:
    angle = float(c.get("angle", 0.0))
    ships = float(c.get("ships", 0.0))
    source_ships = float(c.get("source_ships", 0.0))
    target_ships = float(c.get("target_ships", 0.0))
    target_production = float(c.get("target_production", 0.0))
    distance = float(c.get("distance", 0.0))
    my_total_ships = float(c.get("my_total_ships", 0.0))
    enemy_total_ships = float(c.get("enemy_total_ships", 0.0))
    my_production = float(c.get("my_production", 0.0))
    enemy_production = float(c.get("enemy_production", 0.0))
    sent_fraction = ships / max(1.0, source_ships)
    target_value_per_distance = (target_ships + 8.0 * target_production + 25.0) / (distance + 1.0) / 10.0
    return np.array(
        [
            1.0 if c.get("kind") == "launch" else 0.0,
            float(np.sin(angle)),
            float(np.cos(angle)),
            ships / 100.0,
            distance / 100.0,
            target_ships / 100.0,
            target_production / 10.0,
            1.0 if bool(c.get("target_is_neutral", int(c.get("target_owner", -999)) == -1)) else 0.0,
            1.0 if bool(c.get("target_is_own", False)) else 0.0,
            1.0 if bool(c.get("target_is_enemy", False)) else 0.0,
            float(c.get("my_planet_count", 0.0)) / 10.0,
            float(c.get("enemy_planet_count", 0.0)) / 10.0,
            float(c.get("neutral_planet_count", 0.0)) / 10.0,
            my_total_ships / 500.0,
            enemy_total_ships / 500.0,
            float(c.get("neutral_total_ships", 0.0)) / 500.0,
            my_production / 50.0,
            enemy_production / 50.0,
            source_ships / 100.0,
            float(c.get("source_x", 0.0)) / 100.0,
            float(c.get("source_y", 0.0)) / 100.0,
            float(c.get("target_x", 0.0)) / 100.0,
            float(c.get("target_y", 0.0)) / 100.0,
            float(c.get("target_radius", 0.0)) / 10.0,
            sent_fraction,
            (ships - target_ships) / 100.0,
            source_ships / max(1.0, my_total_ships),
            target_value_per_distance,
            target_production / (distance + 1.0),
            (my_total_ships - enemy_total_ships) / 500.0,
            (my_production - enemy_production) / 50.0,
            1.0 / (1.0 + distance / 100.0),
        ],
        dtype=np.float64,
    )


def _asset_score(observation: dict, player_id: int) -> float:
    score = 0.0
    for planet in observation.get("planets") or []:
        if len(planet) > 6 and int(planet[1]) == int(player_id):
            score += float(planet[5]) + 8.0 * float(planet[6]) + 25.0
    for fleet in observation.get("fleets") or []:
        if len(fleet) > 6 and int(fleet[1]) == int(player_id):
            score += 0.5 * float(fleet[6])
    return score


def _owned_planet_count(observation: dict, player_id: int) -> int:
    return sum(1 for planet in observation.get("planets") or [] if len(planet) > 1 and int(planet[1]) == int(player_id))


def _owned_production(observation: dict, player_id: int) -> float:
    return sum(
        float(planet[6])
        for planet in observation.get("planets") or []
        if len(planet) > 6 and int(planet[1]) == int(player_id)
    )


def _known_player_ids(observation: dict, fallback_player_id: int) -> list[int]:
    ids = {int(fallback_player_id)}
    for planet in observation.get("planets") or []:
        if len(planet) > 1 and int(planet[1]) >= 0:
            ids.add(int(planet[1]))
    for fleet in observation.get("fleets") or []:
        if len(fleet) > 1 and int(fleet[1]) >= 0:
            ids.add(int(fleet[1]))
    return sorted(ids)


def dense_observation_delta_reward(
    before: dict | None,
    after: dict | None,
    player_id: int,
    *,
    asset_normalizer: float = 100.0,
) -> float:
    if before is None or after is None:
        return 0.0
    player_id = int(player_id)
    players = sorted(set(_known_player_ids(before, player_id)) | set(_known_player_ids(after, player_id)))
    opponents = [pid for pid in players if pid != player_id]
    own_asset_delta = _asset_score(after, player_id) - _asset_score(before, player_id)
    if opponents:
        before_enemy = max(_asset_score(before, pid) for pid in opponents)
        after_enemy = max(_asset_score(after, pid) for pid in opponents)
        enemy_asset_delta = after_enemy - before_enemy
    else:
        enemy_asset_delta = 0.0
    planet_delta = _owned_planet_count(after, player_id) - _owned_planet_count(before, player_id)
    production_delta = _owned_production(after, player_id) - _owned_production(before, player_id)
    return float(
        (own_asset_delta - enemy_asset_delta) / max(1e-9, float(asset_normalizer))
        + 0.5 * float(planet_delta)
        + 0.1 * float(production_delta)
    )


def _dense_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[order[k]] = avg_rank
        pos = end
    return ranks


def player_outcome_rewards(state: list, n_players: int, fourp_rank_reward_scale: float = 1.0) -> tuple[list[float], list[float]]:
    env_rewards = [float((state[i].reward or 0.0)) for i in range(n_players)]
    if n_players == 4:
        rank_values = env_rewards
        if len(set(env_rewards)) == 1:
            rank_values = [_asset_score(state[i].observation, i) for i in range(n_players)]
        ranks = _dense_ranks(rank_values)
        scale = float(fourp_rank_reward_scale)
        rewards = [scale if float(rank) == 1.0 else -scale for rank in ranks]
        return rewards, ranks
    ranks = _dense_ranks(env_rewards)
    return env_rewards, ranks


def player_asset_margin_rewards(state: list, n_players: int, scale: float = 0.25, normalizer: float = 100.0) -> list[float]:
    assets = [_asset_score(state[i].observation, i) for i in range(n_players)]
    rewards: list[float] = []
    for player_id, asset in enumerate(assets):
        other_assets = [other for idx, other in enumerate(assets) if idx != player_id]
        best_other = max(other_assets) if other_assets else asset
        margin = (float(asset) - float(best_other)) / max(1e-9, float(normalizer))
        rewards.append(float(scale) * float(np.clip(margin, -1.0, 1.0)))
    return rewards


def player0_won(ranks: list[float]) -> bool:
    return bool(ranks and float(ranks[0]) == 1.0)


class EpisodeTrajectoryRecorder:
    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.latest_observation = None
        self._last_decision_by_player: dict[int, dict] = {}

    def record(
        self,
        player_id: int,
        observation: dict,
        candidate_features: np.ndarray,
        chosen_idx: int,
        old_logp: float,
        value: float,
        mistake_penalty: float = 0.0,
        mistake_metrics: dict | None = None,
        action_kind: str = "unknown",
    ) -> None:
        self.latest_observation = observation
        player_id = int(player_id)
        if player_id in self._last_decision_by_player:
            self._last_decision_by_player[player_id]["next_observation"] = observation
        decision = {
            "player_id": player_id,
            "observation": observation,
            "next_observation": None,
            "candidate_features": np.asarray(candidate_features, dtype=np.float64),
            "chosen_idx": int(chosen_idx),
            "old_logp": float(old_logp),
            "value": float(value),
            "mistake_penalty": float(mistake_penalty),
            "mistake_metrics": dict(mistake_metrics or {}),
            "action_kind": str(action_kind),
        }
        self.decisions.append(decision)
        self._last_decision_by_player[player_id] = decision


class OpponentPool:
    def __init__(self, capacity: int = 8, seed: int = 0) -> None:
        self.capacity = max(0, int(capacity))
        self.rng = np.random.default_rng(int(seed))
        self._snapshots: list[LinearActorCritic] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    def add_snapshot(self, policy: LinearActorCritic) -> None:
        if self.capacity <= 0:
            return
        self._snapshots.append(policy.clone())
        if len(self._snapshots) > self.capacity:
            self._snapshots = self._snapshots[-self.capacity :]

    def maybe_add_snapshot(self, policy: LinearActorCritic, update: int, every: int) -> None:
        if int(every) > 0 and int(update) % int(every) == 0:
            self.add_snapshot(policy)

    def sample(self) -> LinearActorCritic | None:
        if not self._snapshots:
            return None
        return self._snapshots[int(self.rng.integers(0, len(self._snapshots)))]

    def latest(self) -> LinearActorCritic | None:
        if not self._snapshots:
            return None
        return self._snapshots[-1]


class AcceptedGateOpponent:
    def __init__(self) -> None:
        self.policy: LinearActorCritic | None = None
        self.version = 0
        self.accepted_update = 0

    def promote(self, policy: LinearActorCritic, update: int) -> None:
        self.policy = policy.clone()
        self.version += 1
        self.accepted_update = int(update)

    def current(self, fallback: LinearActorCritic) -> LinearActorCritic:
        return self.policy or fallback


def should_promote_gate_opponent(gate_seed_advanced: bool, enabled: bool = False) -> bool:
    return bool(enabled) and bool(gate_seed_advanced)


def strict_gate_opponent_policy(gate_opponent: AcceptedGateOpponent, fallback: LinearActorCritic) -> LinearActorCritic:
    return gate_opponent.current(fallback)


def opponent_policy_for_rollout(
    opponent_pool: OpponentPool,
    gate_opponent: AcceptedGateOpponent,
    fallback: LinearActorCritic,
    eval_every: int,
    rollout_stream_index: int,
    strict_gate_opponents_all_streams: bool = False,
) -> LinearActorCritic:
    if int(eval_every) > 0 and (int(rollout_stream_index) == 0 or bool(strict_gate_opponents_all_streams)):
        return strict_gate_opponent_policy(gate_opponent, fallback)
    return opponent_pool.sample() or fallback


def select_frozen_or_policy_opponent(frozen_opponent_agent, policy_opponent):
    return frozen_opponent_agent if frozen_opponent_agent is not None else policy_opponent


class WinStaySeedScheduler:
    """Repeat a seed until enough clean wins, then sample a new deterministic random seed."""

    def __init__(self, seed: int = 0, advance_win_streak: int = 1) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.advance_win_streak = max(1, int(advance_win_streak))
        self.current_seed = self._next_seed()
        self.repeated_seed_count = 0
        self.clean_win_streak = 0
        self.last_seed_advanced = False

    def _next_seed(self) -> int:
        return int(self.rng.integers(1, 2**31 - 1))

    def after_episode(self, won: bool) -> int:
        if bool(won):
            self.clean_win_streak += 1
            if self.clean_win_streak >= self.advance_win_streak:
                next_seed = self._next_seed()
                while next_seed == self.current_seed:
                    next_seed = self._next_seed()
                self.current_seed = next_seed
                self.repeated_seed_count = 0
                self.clean_win_streak = 0
                self.last_seed_advanced = True
            else:
                self.repeated_seed_count += 1
                self.last_seed_advanced = False
        else:
            self.repeated_seed_count += 1
            self.clean_win_streak = 0
            self.last_seed_advanced = False
        return self.current_seed


def summarize_decision_metrics(decisions: list[dict]) -> dict[str, int]:
    totals = {"sun_loss_ships": 0, "wasted_ships": 0, "invalid_actions": 0, "out_of_bounds_ships": 0, "wrong_target_ships": 0}
    actions = {"decisions": 0, "noop_actions": 0, "launch_actions": 0}
    for decision in decisions:
        for key, value in (decision.get("mistake_metrics") or {}).items():
            totals[key] = totals.get(key, 0) + int(value)
        action_kind = str(decision.get("action_kind", "unknown"))
        actions["decisions"] += 1
        if action_kind == "noop":
            actions["noop_actions"] += 1
        elif action_kind == "launch":
            actions["launch_actions"] += 1
    return {**totals, **actions}


def gate_strict_clean_win(
    ranks: list[float],
    mistake_metrics: dict[str, int],
    phase_mode: str,
    expected_phase_mode: str,
    max_ship_waste: int = 20,
) -> bool:
    return (
        str(phase_mode) == str(expected_phase_mode)
        and player0_won(ranks)
        and ship_waste_loss(mistake_metrics) <= int(max_ship_waste)
    )


def step_efficiency_penalty(episode_steps: int, weight: float = 0.001) -> float:
    return -float(max(0, int(episode_steps))) * float(max(0.0, weight))


def finalize_episode_records(
    decisions: list[dict],
    r0: float = 0.0,
    r1: float = 0.0,
    player_rewards: list[float] | None = None,
    episode_steps: int = 0,
    step_penalty_weight: float = 0.0,
    terminal_reward_scale: float = 1.0,
    max_mistake_penalty: float | None = None,
    tie_loss_penalty: float = 1.0,
    asset_margin_rewards: list[float] | None = None,
    dense_reward_scale: float = 0.0,
    reward_gamma: float = 0.99,
) -> dict:
    candidate_features = []
    chosen_indices = []
    old_logps = []
    returns = []
    advantages = []
    player_ids = []
    step_penalty = step_efficiency_penalty(episode_steps, step_penalty_weight)
    metric_summary = summarize_decision_metrics(decisions)
    dense_enabled = float(dense_reward_scale) > 0.0
    last_decision_index_by_player: dict[int, int] = {}
    if dense_enabled:
        for idx, d in enumerate(decisions):
            last_decision_index_by_player[int(d["player_id"])] = idx
    immediate_rewards: list[float] = []
    for idx, d in enumerate(decisions):
        player_id = int(d["player_id"])
        if player_rewards is None:
            if float(r0) == float(r1):
                terminal_reward = -float(max(0.0, tie_loss_penalty))
            else:
                terminal_reward = float(r0 - r1) if player_id == 0 else float(r1 - r0)
        else:
            terminal_reward = float(player_rewards[player_id])
        terminal_reward *= float(terminal_reward_scale)
        asset_margin_reward = 0.0 if asset_margin_rewards is None else float(asset_margin_rewards[player_id])
        decision_penalty = float(d.get("mistake_penalty", 0.0))
        if max_mistake_penalty is not None:
            decision_penalty = max(decision_penalty, -float(max(0.0, max_mistake_penalty)))
        if dense_enabled:
            dense_reward = float(dense_reward_scale) * dense_observation_delta_reward(
                d.get("observation"),
                d.get("next_observation"),
                player_id,
            )
            shaped_return = decision_penalty + dense_reward
            if idx == last_decision_index_by_player.get(player_id):
                shaped_return += terminal_reward + asset_margin_reward + step_penalty
        else:
            shaped_return = terminal_reward + asset_margin_reward + decision_penalty + step_penalty
        candidate_features.append(d["candidate_features"])
        player_ids.append(player_id)
        chosen_indices.append(int(d["chosen_idx"]))
        old_logps.append(float(d["old_logp"]))
        immediate_rewards.append(float(shaped_return))
    if dense_enabled:
        reward_gamma = float(np.clip(float(reward_gamma), 0.0, 1.0))
        running_by_player: dict[int, float] = {}
        returns = [0.0] * len(immediate_rewards)
        for idx in range(len(immediate_rewards) - 1, -1, -1):
            player_id = int(player_ids[idx])
            running = float(immediate_rewards[idx]) + reward_gamma * running_by_player.get(player_id, 0.0)
            running_by_player[player_id] = running
            returns[idx] = running
    else:
        returns = immediate_rewards
    for shaped_return, d in zip(returns, decisions, strict=True):
        advantages.append(float(shaped_return) - float(d["value"]))
    return {
        "candidate_features": candidate_features,
        "player_ids": np.asarray(player_ids, dtype=np.int64),
        "chosen_indices": np.asarray(chosen_indices, dtype=np.int64),
        "old_logps": np.asarray(old_logps, dtype=np.float64),
        "returns": np.asarray(returns, dtype=np.float64),
        "advantages": np.asarray(advantages, dtype=np.float64),
        "mistake_metrics": {
            key: metric_summary[key]
            for key in ["sun_loss_ships", "wasted_ships", "invalid_actions", "out_of_bounds_ships", "wrong_target_ships"]
        },
        "action_metrics": {key: metric_summary[key] for key in ["decisions", "noop_actions", "launch_actions"]},
        "step_penalty": step_penalty,
    }


def append_episode_batch(batch: dict, ep_batch: dict, repeat_count: int = 1) -> None:
    repeat_count = max(0, int(repeat_count))
    for _ in range(repeat_count):
        for key in batch:
            if key == "candidate_features":
                batch[key].extend(ep_batch[key])
            else:
                values = ep_batch[key]
                batch[key].extend(values.tolist() if hasattr(values, "tolist") else list(values))


def decision_outcome_sample_weights(
    player_ids,
    ranks: list[float],
    base_weight: int,
    winning_sample_weight: int = 1,
    losing_sample_weight: int = 1,
) -> np.ndarray:
    weights: list[int] = []
    for player_id in np.asarray(player_ids, dtype=np.int64).tolist():
        player_rank = float(ranks[int(player_id)]) if 0 <= int(player_id) < len(ranks) else 0.0
        weights.append(
            outcome_sample_weight(
                base_weight=base_weight,
                won=player_rank == 1.0,
                winning_sample_weight=winning_sample_weight,
                losing_sample_weight=losing_sample_weight,
            )
        )
    return np.asarray(weights, dtype=np.int64)


def append_episode_batch_by_decision_weight(batch: dict, ep_batch: dict, repeat_weights) -> None:
    weights = np.asarray(repeat_weights, dtype=np.int64)
    for idx, repeat_count in enumerate(weights.tolist()):
        for _ in range(max(0, int(repeat_count))):
            for key in batch:
                value = ep_batch[key][idx]
                batch[key].append(value)


def episode_step_count(env) -> int:
    steps = getattr(env, "steps", None)
    if steps is not None:
        return max(0, len(steps) - 1)
    return 0


def main(argv=None):
    args = parse_args(argv)
    train_start_time = time.time()
    run_dir = build_run_dir(args.run_dir, args.mode, allow_existing=args.allow_existing_run_dir)
    writer = MetricsWriter(run_dir)
    stats = RollingStats()
    viewer = PygameTrainingViewer(
        enabled=not args.no_viewer,
        fullscreen=args.fullscreen,
        resizable=args.resizable_window,
        minimal=args.minimal_viewer,
        burn_in_shift_seconds=args.burn_in_shift_seconds,
    )
    policy = create_initial_policy(feature_dim=RL_FEATURE_DIM, seed=args.seed, initial_policy_path=args.initial_policy)
    frozen_opponent_agent = load_agent_ref(args.frozen_opponent_agent) if args.frozen_opponent_agent else None
    last_viewer_draw_time: float | None = None
    last_viewer_event_poll_time: float | None = None
    opponent_pool = OpponentPool(capacity=args.opponent_pool_size, seed=args.seed + 17)
    opponent_pool.add_snapshot(policy)
    gate_opponent = AcceptedGateOpponent()
    initial_gate_policy = create_initial_gate_opponent_policy(
        learner_policy=policy,
        feature_dim=RL_FEATURE_DIM,
        seed=args.seed,
        initial_gate_opponent_path=args.initial_gate_opponent_policy,
    )
    gate_opponent.promote(initial_gate_policy, update=0)
    seed_scheduler = WinStaySeedScheduler(seed=args.seed, advance_win_streak=args.gate_advance_win_streak)
    clean_gate_policies: dict[int, LinearActorCritic] = {}
    rollback_lr_cooldown = RollbackLRCooldown()

    def mk_eval_agent(
        player_id: int,
        actor_policy: LinearActorCritic,
        deterministic: bool = True,
        recorder: EpisodeTrajectoryRecorder | None = None,
        phase_mode: str = "2p",
    ):
        def _agent(observation, configuration):
            candidates = build_action_candidates(observation)
            features = np.stack([featurize(c) for c in candidates], axis=0)
            if deterministic:
                idx, logp, value, _ = actor_policy.argmax(features)
            else:
                idx, logp, value, _ = actor_policy.sample(features)
            if recorder is not None and player_is_trainable(phase_mode, player_id):
                mistake_metrics = estimate_candidate_mistakes(observation, candidates[idx])
                recorder.record(
                    player_id=player_id,
                    observation=observation,
                    candidate_features=features,
                    chosen_idx=idx,
                    old_logp=logp,
                    value=value,
                    mistake_penalty=mistake_penalty(mistake_metrics, mode=phase_mode),
                    mistake_metrics=mistake_metrics,
                    action_kind=str(candidates[idx].get("kind", "unknown")),
                )
            return candidate_to_env_action(candidates[idx])

        return _agent

    for update in range(1, args.updates + 1):
        batch = {"candidate_features": [], "chosen_indices": [], "old_logps": [], "returns": [], "advantages": []}
        weighted_batch_decisions = 0
        weighted_deterministic_decisions = 0
        rollout_signal_rows: list[dict] = []
        for ep in range(1, args.episodes + 1):
            global_ep = global_episode_number(update, ep, args.episodes)
            current_mode = episode_mode(args.mode, global_ep)
            n_players = episode_player_count(args.mode, global_ep)
            gate_seed = int(seed_scheduler.current_seed)
            rollout_stream_index = rollout_stream_index_for_episode(
                ep,
                episodes_per_update=args.episodes,
                eval_every=args.eval_every,
            )
            episode_seed = training_episode_seed(
                base_seed=args.seed,
                gate_seed=gate_seed,
                global_episode=global_ep,
                rollout_stream_index=rollout_stream_index,
                diverse_training_seeds=args.diverse_training_seeds,
            )
            env = make("orbit_wars", configuration={"episodeSteps": args.episode_steps, "seed": episode_seed}, debug=False)
            recorder = EpisodeTrajectoryRecorder()
            train_focus_player = training_focus_player_id(
                current_mode,
                global_episode=global_ep,
                train_seat=args.train_seat,
            )

            def mk_agent(
                player_id: int,
                actor_policy: LinearActorCritic,
                trainable: bool = True,
                rollout_stream_index: int = 0,
            ):
                def _agent(observation, configuration):
                    candidates = build_action_candidates(observation)
                    features = np.stack([featurize(c) for c in candidates], axis=0)
                    if training_actor_should_be_deterministic(
                        trainable=trainable,
                        eval_every=args.eval_every,
                        rollout_stream_index=rollout_stream_index,
                    ):
                        idx, logp, value, _ = actor_policy.argmax(features)
                    else:
                        idx, logp, value, _ = actor_policy.sample(features)
                    if trainable:
                        mistake_metrics = estimate_candidate_mistakes(observation, candidates[idx])
                        recorder.record(
                            player_id=player_id,
                            observation=observation,
                            candidate_features=features,
                            chosen_idx=idx,
                            old_logp=logp,
                            value=value,
                            mistake_penalty=mistake_penalty(mistake_metrics, mode=current_mode),
                            mistake_metrics=mistake_metrics,
                            action_kind=str(candidates[idx].get("kind", "unknown")),
                        )
                    return candidate_to_env_action(candidates[idx])

                return _agent

            agents = []
            for player_id in range(n_players):
                trainable = training_player_is_trainable(
                    current_mode,
                    player_id,
                    train_all_players=args.train_all_players,
                    train_seat=args.train_seat,
                    global_episode=global_ep,
                )
                if trainable:
                    actor_policy = policy
                    if int(args.eval_every) > 0:
                        actor_policy = policy.clone(
                            seed=actor_rng_seed_for_episode(
                                episode_seed,
                                player_id,
                                rollout_index=rollout_stream_index,
                            )
                        )
                    agents.append(mk_agent(player_id, actor_policy, trainable=True, rollout_stream_index=rollout_stream_index))
                else:
                    opponent_policy = opponent_policy_for_rollout(
                        opponent_pool,
                        gate_opponent,
                        policy,
                        eval_every=args.eval_every,
                        rollout_stream_index=rollout_stream_index,
                        strict_gate_opponents_all_streams=args.strict_gate_opponents_all_streams,
                    )
                    opponent_agent = select_frozen_or_policy_opponent(
                        frozen_opponent_agent,
                        mk_agent(player_id, opponent_policy, trainable=False),
                    )
                    agents.append(opponent_agent)
            env.run(agents)
            state = env.state
            player_rewards, ranks = player_outcome_rewards(state, n_players=n_players, fourp_rank_reward_scale=args.fourp_rank_reward_scale)
            candidate_rank = float(ranks[train_focus_player])
            candidate_won = candidate_rank == 1.0
            if n_players == 2:
                won = candidate_won
                other_player = 1 - int(train_focus_player)
                ep_reward = player_rewards[train_focus_player] - player_rewards[other_player]
            else:
                won = candidate_won
                ep_reward = player_rewards[train_focus_player]
            asset_margin_rewards = player_asset_margin_rewards(
                state,
                n_players=n_players,
                scale=args.asset_margin_reward_scale,
            )
            ep_steps = episode_step_count(env)
            now = time.time()
            episodes_per_second = float(global_ep) / max(1e-9, now - train_start_time)
            ep_batch = finalize_episode_records(
                recorder.decisions,
                player_rewards=player_rewards if n_players > 2 else None,
                r0=float(state[0].reward or 0.0),
                r1=float(state[1].reward or 0.0) if n_players > 1 else 0.0,
                episode_steps=ep_steps,
                step_penalty_weight=args.step_penalty_weight,
                terminal_reward_scale=args.terminal_reward_scale,
                max_mistake_penalty=args.max_mistake_penalty,
                asset_margin_rewards=asset_margin_rewards,
                dense_reward_scale=args.dense_reward_scale,
                reward_gamma=args.reward_gamma,
            )
            deterministic_rollout = training_actor_should_be_deterministic(
                trainable=True,
                eval_every=args.eval_every,
                rollout_stream_index=rollout_stream_index,
            )
            stream_weight = deterministic_stream_sample_weight(
                rollout_stream_index=rollout_stream_index,
                eval_every=args.eval_every,
                configured_weight=args.deterministic_stream_weight,
            )
            sample_weight = outcome_sample_weight(
                base_weight=stream_weight,
                won=won,
                winning_sample_weight=args.winning_sample_weight,
                losing_sample_weight=args.losing_sample_weight,
            )
            decision_count = int(len(ep_batch.get("chosen_indices", [])))
            weighted_winning_decisions = decision_count * sample_weight if won else 0
            weighted_losing_decisions = 0 if won else decision_count * sample_weight
            if args.train_all_players and n_players > 2:
                repeat_weights = decision_outcome_sample_weights(
                    ep_batch.get("player_ids", []),
                    ranks=ranks,
                    base_weight=stream_weight,
                    winning_sample_weight=args.winning_sample_weight,
                    losing_sample_weight=args.losing_sample_weight,
                )
                append_episode_batch_by_decision_weight(batch, ep_batch, repeat_weights=repeat_weights)
                player_ids = np.asarray(ep_batch.get("player_ids", []), dtype=np.int64)
                player_wins = np.asarray([float(ranks[int(player_id)]) == 1.0 for player_id in player_ids], dtype=bool)
                weighted_winning_decisions = int(np.asarray(repeat_weights)[player_wins].sum())
                weighted_losing_decisions = int(np.asarray(repeat_weights)[~player_wins].sum())
            else:
                append_episode_batch(batch, ep_batch, repeat_count=sample_weight)
            weighted_decisions = int(weighted_winning_decisions + weighted_losing_decisions)
            weighted_batch_decisions += weighted_decisions
            if deterministic_rollout:
                weighted_deterministic_decisions += weighted_decisions
            rollout_signal_rows.append(
                {
                    "deterministic_rollout": deterministic_rollout,
                    "won": won,
                    "decision_count": decision_count,
                    "stream_weight": sample_weight,
                    "weighted_winning_decisions": weighted_winning_decisions,
                    "weighted_losing_decisions": weighted_losing_decisions,
                    "rank": candidate_rank,
                }
            )
            draw_obs = recorder.latest_observation
            feature_flags = training_feature_flags(args, current_mode)
            stats.add(won)
            if should_poll_viewer_events(
                now_seconds=now,
                last_poll_seconds=last_viewer_event_poll_time,
                interval_seconds=args.viewer_event_poll_interval_seconds,
            ):
                viewer.pump_events()
                last_viewer_event_poll_time = now
            if draw_obs is not None and should_draw_viewer(
                now_seconds=now,
                last_draw_seconds=last_viewer_draw_time,
                interval_seconds=args.viewer_draw_interval_seconds,
            ):
                viewer.draw(
                    draw_obs,
                    {
                        "update": update,
                        "episode": ep,
                        "wins": stats.wins,
                        "losses": stats.losses,
                        "recent_win_rate": stats.recent_win_rate,
                        "seed": episode_seed,
                        "gate_seed": gate_seed,
                        "phase_mode": current_mode,
                        "n_players": n_players,
                        "episode_steps": ep_steps,
                        "episodes_per_second": episodes_per_second,
                        "step_penalty": ep_batch.get("step_penalty", 0.0),
                        "rollout_stream_index": rollout_stream_index,
                        "deterministic_rollout": deterministic_rollout,
                        "deterministic_stream_sample_weight": stream_weight,
                        "repeated_seed_count": seed_scheduler.repeated_seed_count,
                        **feature_flags,
                        **ep_batch.get("mistake_metrics", {}),
                        **ep_batch.get("action_metrics", {}),
                    },
                )
                last_viewer_draw_time = now
                last_viewer_event_poll_time = now
            writer.write(
                {
                    "update": update,
                    "episode": ep,
                    "seed": episode_seed,
                    "gate_seed": gate_seed,
                    "phase_mode": current_mode,
                    "n_players": n_players,
                    "wall_time": now,
                    "episodes_per_second": episodes_per_second,
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "recent_win_rate": stats.recent_win_rate,
                    "episode_reward": ep_reward,
                    "player0_asset_margin_reward": asset_margin_rewards[0] if asset_margin_rewards else 0.0,
                    "episode_steps": ep_steps,
                    "step_penalty": ep_batch.get("step_penalty", 0.0),
                    "rollout_stream_index": rollout_stream_index,
                    "deterministic_rollout": deterministic_rollout,
                    "deterministic_stream_sample_weight": stream_weight,
                    "player0_rank": ranks[0],
                    "candidate_player_id": int(train_focus_player),
                    "candidate_rank": candidate_rank,
                    "candidate_won": bool(candidate_won),
                    "candidate_asset_margin_reward": asset_margin_rewards[train_focus_player] if asset_margin_rewards else 0.0,
                    "repeated_seed_count": seed_scheduler.repeated_seed_count,
                    **feature_flags,
                    **ep_batch.get("mistake_metrics", {}),
                    **ep_batch.get("action_metrics", {}),
                }
            )
            if should_advance_training_seed_during_episode(args.eval_every):
                seed_scheduler.after_episode(won=won)

        final_batch = {
            "candidate_features": batch["candidate_features"],
            "chosen_indices": np.asarray(batch["chosen_indices"], dtype=np.int64),
            "old_logps": np.asarray(batch["old_logps"], dtype=np.float64),
            "returns": np.asarray(batch["returns"], dtype=np.float64),
            "advantages": np.asarray(batch["advantages"], dtype=np.float64),
        }
        update_metrics = {}
        lr_streak_for_update = rollback_lr_cooldown.streak_for_update(seed_scheduler.clean_win_streak)
        lr_cooldown_active = rollback_lr_cooldown.active
        lr_cooldown_remaining = rollback_lr_cooldown.remaining
        lr_cooldown_source_streak = rollback_lr_cooldown.source_streak
        effective_policy_lr = policy_lr_for_gate_streak(
            args.policy_lr,
            clean_win_streak=lr_streak_for_update,
            streak_multiplier=args.gate_streak_lr_multiplier,
            high_streak_threshold=args.gate_streak_high_threshold,
            high_streak_multiplier=args.gate_streak_high_lr_multiplier,
            min_streak_for_reduction=args.gate_streak_lr_min_streak,
        )
        pre_update_policy = policy.clone()
        if final_batch["candidate_features"]:
            update_metrics = update_policy_epochs(
                policy,
                final_batch,
                lr=effective_policy_lr,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                ppo_epochs=args.ppo_epochs,
            )
        rollback_lr_cooldown.consume_update()
        if int(args.eval_every) > 0 and update % int(args.eval_every) == 0:
            eval_global_ep = global_episode_number(update, args.episodes, args.episodes)
            eval_mode = episode_mode(args.mode, eval_global_ep)
            eval_n_players = episode_player_count(args.mode, eval_global_ep)
            eval_seed = int(seed_scheduler.current_seed)
            eval_env = make("orbit_wars", configuration={"episodeSteps": args.episode_steps, "seed": eval_seed}, debug=False)
            eval_recorder = EpisodeTrajectoryRecorder()
            eval_agents = []
            for player_id in range(eval_n_players):
                trainable = gate_eval_player_is_trainable(eval_mode, player_id)
                if trainable:
                    eval_agents.append(
                        mk_eval_agent(
                            player_id,
                            policy.clone(seed=actor_rng_seed_for_episode(eval_seed, player_id)),
                            deterministic=gate_actor_should_be_deterministic(trainable),
                            recorder=eval_recorder,
                            phase_mode=eval_mode,
                        )
                    )
                else:
                    policy_eval_agent = mk_eval_agent(
                        player_id,
                        strict_gate_opponent_policy(gate_opponent, policy),
                        deterministic=gate_actor_should_be_deterministic(trainable),
                        phase_mode=eval_mode,
                    )
                    eval_agents.append(select_frozen_or_policy_opponent(frozen_opponent_agent, policy_eval_agent))
            eval_env.run(eval_agents)
            eval_state = eval_env.state
            eval_rewards, eval_ranks = player_outcome_rewards(
                eval_state,
                n_players=eval_n_players,
                fourp_rank_reward_scale=args.fourp_rank_reward_scale,
            )
            eval_asset_margin_rewards = player_asset_margin_rewards(
                eval_state,
                n_players=eval_n_players,
                scale=args.asset_margin_reward_scale,
            )
            eval_steps = episode_step_count(eval_env)
            eval_now = time.time()
            eval_metric_summary = summarize_decision_metrics(eval_recorder.decisions)
            eval_policy_stability_metrics = {
                **action_margin_stats(policy, eval_recorder.decisions),
                **argmax_flip_stats(pre_update_policy, policy, eval_recorder.decisions),
            }
            eval_clean_win = gate_strict_clean_win(
                eval_ranks,
                eval_metric_summary,
                phase_mode=eval_mode,
                expected_phase_mode=eval_mode,
            )
            gate_streak_before = seed_scheduler.clean_win_streak
            gate_opponent_version = gate_opponent.version
            gate_opponent_accepted_update = gate_opponent.accepted_update
            remember_clean_gate_policy(clean_gate_policies, eval_seed, policy, eval_clean_win)
            policy, gate_policy_restored = restore_clean_gate_policy(
                clean_gate_policies,
                eval_seed,
                policy,
                eval_clean_win=eval_clean_win,
                gate_streak_before=gate_streak_before,
                enabled=args.gate_policy_rollback,
            )
            rollback_lr_cooldown.arm(
                gate_policy_restored=gate_policy_restored and bool(args.rollback_lr_cooldown_enabled),
                gate_streak_before=gate_streak_before,
                cooldown_updates=args.rollback_lr_cooldown_updates,
            )
            seed_scheduler.after_episode(won=eval_clean_win)
            gate_opponent_promoted = should_promote_gate_opponent(
                gate_seed_advanced=seed_scheduler.last_seed_advanced,
                enabled=args.promote_gate_opponent_on_advance,
            )
            if gate_opponent_promoted:
                gate_opponent.promote(policy, update=update)
            writer.write(
                {
                    "update": update,
                    "episode": 0,
                    "seed": eval_seed,
                    "gate_seed": eval_seed,
                    "phase_mode": eval_mode,
                    "n_players": eval_n_players,
                    "wall_time": eval_now,
                    "episodes_per_second": float(eval_global_ep) / max(1e-9, eval_now - train_start_time),
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "recent_win_rate": stats.recent_win_rate,
                    "episode_reward": float(eval_rewards[0]) if eval_rewards else 0.0,
                    "player0_asset_margin_reward": eval_asset_margin_rewards[0] if eval_asset_margin_rewards else 0.0,
                    "loss_policy": float(update_metrics.get("loss_policy", 0.0)),
                    "loss_value": float(update_metrics.get("loss_value", 0.0)),
                    "ratio_mean": float(update_metrics.get("ratio_mean", 0.0)),
                    "policy_lr_effective": float(effective_policy_lr),
                    "gate_lr_streak_for_update": int(lr_streak_for_update),
                    "gate_lr_cooldown_active": bool(lr_cooldown_active),
                    "gate_lr_cooldown_remaining": int(lr_cooldown_remaining),
                    "gate_lr_cooldown_source_streak": int(lr_cooldown_source_streak),
                    "weighted_batch_decisions": int(weighted_batch_decisions),
                    "weighted_deterministic_decisions": int(weighted_deterministic_decisions),
                    "weighted_deterministic_fraction": float(weighted_deterministic_decisions)
                    / max(1.0, float(weighted_batch_decisions)),
                    **rollout_signal_metrics(rollout_signal_rows),
                    "episode_steps": eval_steps,
                    "step_penalty": 0.0,
                    "player0_rank": eval_ranks[0],
                    "repeated_seed_count": seed_scheduler.repeated_seed_count,
                    "gate_seed_clean_streak": seed_scheduler.clean_win_streak,
                    "gate_seed_clean_streak_before": gate_streak_before,
                    "gate_advance_win_streak": int(args.gate_advance_win_streak),
                    "gate_strict_clean_win": bool(eval_clean_win),
                    "gate_seed_advanced": bool(seed_scheduler.last_seed_advanced),
                    "gate_policy_restored": bool(gate_policy_restored),
                    "gate_opponent_version": int(gate_opponent_version),
                    "gate_opponent_accepted_update": int(gate_opponent_accepted_update),
                    "gate_opponent_promoted": bool(gate_opponent_promoted),
                    "gate_eval": True,
                    "training_episode": False,
                    **training_feature_flags(args, eval_mode),
                    **eval_metric_summary,
                    **eval_policy_stability_metrics,
                }
            )
        opponent_pool.maybe_add_snapshot(policy, update=update, every=args.opponent_snapshot_every)
        if update % args.checkpoint_every == 0:
            policy.save(run_dir / "checkpoints" / f"policy_update_{update}.npz")
    viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
