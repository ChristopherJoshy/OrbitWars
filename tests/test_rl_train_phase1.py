import numpy as np
from types import SimpleNamespace

from tools.train_rl_phase1 import (
    AcceptedGateOpponent,
    EpisodeTrajectoryRecorder,
    OpponentPool,
    RL_FEATURE_DIM,
    RollbackLRCooldown,
    rollout_signal_metrics,
    WinStaySeedScheduler,
    action_margin_stats,
    argmax_flip_stats,
    remember_clean_gate_policy,
    restore_clean_gate_policy,
    should_restore_clean_gate_policy,
    should_promote_gate_opponent,
    gate_strict_clean_win,
    build_run_dir,
    create_initial_gate_opponent_policy,
    create_initial_policy,
    episode_mode,
    episode_player_count,
    global_episode_number,
    featurize,
    finalize_episode_records,
    mode_player_count,
    parse_args,
    player0_won,
    player_asset_margin_rewards,
    training_focus_player_id,
    gate_eval_player_is_trainable,
    load_agent_ref,
    select_frozen_or_policy_opponent,
    player_is_trainable,
    player_outcome_rewards,
    training_player_is_trainable,
    opponent_policy_for_rollout,
    outcome_sample_weight,
    policy_lr_for_gate_streak,
    strict_gate_opponent_policy,
    actor_rng_seed_for_episode,
    rollout_stream_index_for_episode,
    should_draw_viewer,
    should_poll_viewer_events,
    should_advance_training_seed_during_episode,
    training_episode_seed,
    training_feature_flags,
    update_policy_epochs,
    gate_actor_should_be_deterministic,
    deterministic_stream_sample_weight,
    append_episode_batch,
    append_episode_batch_by_decision_weight,
    decision_outcome_sample_weights,
    summarize_decision_metrics,
    training_actor_should_be_deterministic,
    step_efficiency_penalty,
)
from src.ow.rl.policy import LinearActorCritic
from src.ow.rl.action_space import build_action_candidates


def _state(player, reward, planets):
    return SimpleNamespace(reward=reward, observation={"player": player, "planets": planets, "fleets": []})


def test_parse_args_defaults_to_2p_smoke():
    args = parse_args(["--episodes", "2"])
    assert args.mode == "2p"
    assert args.episodes == 2
    assert args.no_viewer is True


def test_parse_args_accepts_minimal_fullscreen_and_burnin_flags():
    args = parse_args(["--viewer", "--fullscreen", "--minimal-viewer", "--burn-in-shift-seconds", "12"])
    assert args.no_viewer is False
    assert args.fullscreen is True
    assert args.minimal_viewer is True
    assert args.burn_in_shift_seconds == 12


def test_parse_args_accepts_window_mode_and_resizable_switches():
    args_default = parse_args([])
    assert args_default.resizable_window is True

    args_windowed = parse_args(["--windowed", "--resizable-window"])
    assert args_windowed.fullscreen is False
    assert args_windowed.resizable_window is True

    args_fixed = parse_args(["--fixed-window"])
    assert args_fixed.resizable_window is False


def test_parse_args_defaults_to_stronger_rewards_with_capped_penalties():
    args = parse_args([])

    assert args.terminal_reward_scale == 2.5
    assert args.fourp_rank_reward_scale == 2.5
    assert args.max_mistake_penalty == 3.0


def test_parse_args_uses_faster_policy_learning_defaults_for_local_rl():
    args = parse_args([])

    assert args.policy_lr == 0.01
    assert args.entropy_coef == 0.003
    assert args.value_coef == 0.5


def test_parse_args_enables_dense_asset_margin_reward_by_default():
    args = parse_args([])

    assert args.asset_margin_reward_scale == 0.25


def test_parse_args_keeps_deterministic_stream_weight_configurable():
    args = parse_args(["--deterministic-stream-weight", "8"])

    assert args.deterministic_stream_weight == 8


def test_parse_args_keeps_winning_sample_weight_configurable():
    args = parse_args(["--winning-sample-weight", "3"])

    assert args.winning_sample_weight == 3


def test_parse_args_keeps_losing_sample_weight_configurable():
    default_args = parse_args([])
    args = parse_args(["--losing-sample-weight", "0"])

    assert default_args.losing_sample_weight == 1
    assert args.losing_sample_weight == 0


def test_parse_args_keeps_ppo_epochs_configurable():
    default_args = parse_args([])
    args = parse_args(["--ppo-epochs", "4"])

    assert default_args.ppo_epochs == 1
    assert args.ppo_epochs == 4


def test_parse_args_keeps_train_all_players_disabled_by_default():
    default_args = parse_args([])
    args = parse_args(["--train-all-players"])

    assert default_args.train_all_players is False
    assert args.train_all_players is True


def test_parse_args_keeps_train_seat_player0_by_default_and_allows_alternating():
    default_args = parse_args([])
    args = parse_args(["--train-seat", "alternating"])

    assert default_args.train_seat == "player0"
    assert args.train_seat == "alternating"


def test_parse_args_keeps_strict_gate_training_opponents_disabled_by_default():
    default_args = parse_args([])
    args = parse_args(["--strict-gate-opponents-all-streams"])

    assert default_args.strict_gate_opponents_all_streams is False
    assert args.strict_gate_opponents_all_streams is True


def test_parse_args_keeps_frozen_opponent_agent_configurable():
    default_args = parse_args([])
    args = parse_args(["--frozen-opponent-agent", "src.ow.chadman_baseline:agent"])

    assert default_args.frozen_opponent_agent is None
    assert args.frozen_opponent_agent == "src.ow.chadman_baseline:agent"


def test_parse_args_keeps_diverse_training_seeds_disabled_by_default():
    default_args = parse_args([])
    args = parse_args(["--diverse-training-seeds"])

    assert default_args.diverse_training_seeds is False
    assert args.diverse_training_seeds is True


def test_parse_args_rejects_frozen_opponent_with_train_all_players():
    try:
        parse_args(["--frozen-opponent-agent", "src.ow.chadman_baseline:agent", "--train-all-players"])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected argparse to reject frozen opponent with train-all")


def test_parse_args_rejects_frozen_opponent_with_initial_gate_policy(tmp_path):
    checkpoint = tmp_path / "gate.npz"
    LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=1).save(checkpoint)

    try:
        parse_args([
            "--frozen-opponent-agent",
            "src.ow.chadman_baseline:agent",
            "--initial-gate-opponent-policy",
            str(checkpoint),
        ])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected argparse to reject ambiguous frozen/gate opponents")


def test_parse_args_keeps_gate_advance_win_streak_configurable():
    default_args = parse_args([])
    args = parse_args(["--gate-advance-win-streak", "5"])

    assert default_args.gate_advance_win_streak == 3
    assert args.gate_advance_win_streak == 5


def test_parse_args_keeps_streak_lr_reduction_configurable():
    args = parse_args(
        [
            "--gate-streak-lr-multiplier",
            "0.2",
            "--gate-streak-high-threshold",
            "4",
            "--gate-streak-high-lr-multiplier",
            "0.05",
            "--gate-streak-lr-min-streak",
            "3",
        ]
    )

    assert args.gate_streak_lr_multiplier == 0.2
    assert args.gate_streak_high_threshold == 4
    assert args.gate_streak_high_lr_multiplier == 0.05
    assert args.gate_streak_lr_min_streak == 3


def test_parse_args_accepts_initial_policy_checkpoint(tmp_path):
    checkpoint = tmp_path / "policy.npz"

    args = parse_args(["--initial-policy", str(checkpoint)])

    assert args.initial_policy == checkpoint


def test_parse_args_accepts_initial_gate_opponent_checkpoint(tmp_path):
    checkpoint = tmp_path / "gate_opponent.npz"

    args = parse_args(["--initial-gate-opponent-policy", str(checkpoint)])

    assert args.initial_gate_opponent_policy == checkpoint


def test_create_initial_policy_loads_checkpoint_when_configured(tmp_path):
    checkpoint = tmp_path / "policy.npz"
    saved = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=123)
    saved.actor_w += 1.5
    saved.save(checkpoint)

    loaded = create_initial_policy(feature_dim=RL_FEATURE_DIM, seed=7, initial_policy_path=checkpoint)

    assert np.allclose(loaded.actor_w, saved.actor_w)
    assert np.allclose(loaded.value_w, saved.value_w)


def test_create_initial_policy_pads_legacy_checkpoint_to_current_feature_dim(tmp_path):
    checkpoint = tmp_path / "legacy_policy.npz"
    legacy = LinearActorCritic(feature_dim=19, seed=123)
    legacy.actor_w[:] = np.arange(19, dtype=np.float64)
    legacy.value_w[:] = np.arange(19, dtype=np.float64) + 100.0
    legacy.save(checkpoint)

    loaded = create_initial_policy(feature_dim=RL_FEATURE_DIM, seed=7, initial_policy_path=checkpoint)

    assert RL_FEATURE_DIM > 19
    assert loaded.actor_w.shape == (RL_FEATURE_DIM,)
    assert loaded.value_w.shape == (RL_FEATURE_DIM,)
    assert np.allclose(loaded.actor_w[:19], legacy.actor_w)
    assert np.allclose(loaded.value_w[:19], legacy.value_w)
    assert np.allclose(loaded.actor_w[19:], 0.0)
    assert np.allclose(loaded.value_w[19:], 0.0)


def test_create_initial_gate_opponent_policy_can_differ_from_loaded_policy(tmp_path):
    learner = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=1)
    gate_opponent = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=2)
    checkpoint = tmp_path / "gate_opponent.npz"
    gate_opponent.save(checkpoint)

    loaded = create_initial_gate_opponent_policy(
        learner_policy=learner,
        feature_dim=RL_FEATURE_DIM,
        seed=7,
        initial_gate_opponent_path=checkpoint,
    )

    assert np.allclose(loaded.actor_w, gate_opponent.actor_w)
    assert not np.allclose(loaded.actor_w, learner.actor_w)


def test_create_initial_gate_opponent_policy_defaults_to_learner_snapshot():
    learner = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=3)

    loaded = create_initial_gate_opponent_policy(
        learner_policy=learner,
        feature_dim=RL_FEATURE_DIM,
        seed=7,
        initial_gate_opponent_path=None,
    )

    assert np.allclose(loaded.actor_w, learner.actor_w)
    assert loaded is not learner


def test_parse_args_enables_rollback_lr_cooldown_by_default():
    args = parse_args([])

    assert args.rollback_lr_cooldown_enabled is True
    assert args.rollback_lr_cooldown_updates == 2


def test_parse_args_enables_gate_policy_rollback_by_default():
    args = parse_args([])
    disabled = parse_args(["--no-gate-policy-rollback"])

    assert args.gate_policy_rollback is True
    assert disabled.gate_policy_rollback is False


def test_parse_args_updates_pygame_dashboard_every_30_seconds_for_training_speed():
    args = parse_args([])

    assert args.viewer_draw_interval_seconds == 30.0
    assert args.viewer_event_poll_interval_seconds == 1.0


def test_should_draw_viewer_keeps_first_draw_and_then_throttles_by_seconds():
    assert should_draw_viewer(now_seconds=100.0, last_draw_seconds=None, interval_seconds=30.0) is True
    assert should_draw_viewer(now_seconds=129.9, last_draw_seconds=100.0, interval_seconds=30.0) is False
    assert should_draw_viewer(now_seconds=130.0, last_draw_seconds=100.0, interval_seconds=30.0) is True
    assert should_draw_viewer(now_seconds=101.0, last_draw_seconds=100.0, interval_seconds=0.0) is True


def test_should_poll_viewer_events_keeps_window_responsive_between_slow_draws():
    assert should_poll_viewer_events(now_seconds=100.0, last_poll_seconds=None, interval_seconds=1.0) is True
    assert should_poll_viewer_events(now_seconds=100.5, last_poll_seconds=100.0, interval_seconds=1.0) is False
    assert should_poll_viewer_events(now_seconds=101.0, last_poll_seconds=100.0, interval_seconds=1.0) is True
    assert should_poll_viewer_events(now_seconds=100.1, last_poll_seconds=100.0, interval_seconds=0.0) is True


def test_seed_advancement_waits_for_gate_eval_when_gate_eval_is_enabled():
    assert should_advance_training_seed_during_episode(eval_every=1) is False
    assert should_advance_training_seed_during_episode(eval_every=0) is True


def test_training_opponents_use_deterministic_policy_when_gate_eval_is_enabled():
    assert training_actor_should_be_deterministic(trainable=False, eval_every=1) is True
    assert training_actor_should_be_deterministic(trainable=True, eval_every=1, rollout_stream_index=0) is True
    assert training_actor_should_be_deterministic(trainable=True, eval_every=1, rollout_stream_index=1) is False
    assert training_actor_should_be_deterministic(trainable=False, eval_every=0) is False


def test_deterministic_stream_weight_only_applies_to_gate_aligned_argmax_rollout():
    assert deterministic_stream_sample_weight(rollout_stream_index=0, eval_every=1, configured_weight=8) == 8
    assert deterministic_stream_sample_weight(rollout_stream_index=1, eval_every=1, configured_weight=8) == 1
    assert deterministic_stream_sample_weight(rollout_stream_index=0, eval_every=0, configured_weight=8) == 1
    assert deterministic_stream_sample_weight(rollout_stream_index=0, eval_every=1, configured_weight=0) == 1


def test_outcome_sample_weight_upweights_winning_rollouts_only():
    assert outcome_sample_weight(base_weight=4, won=True, winning_sample_weight=3) == 12
    assert outcome_sample_weight(base_weight=4, won=False, winning_sample_weight=3) == 4
    assert outcome_sample_weight(base_weight=4, won=True, winning_sample_weight=0) == 4


def test_outcome_sample_weight_can_skip_losing_rollouts_for_elite_replay():
    assert outcome_sample_weight(base_weight=4, won=False, winning_sample_weight=8, losing_sample_weight=0) == 0
    assert outcome_sample_weight(base_weight=4, won=False, winning_sample_weight=8, losing_sample_weight=2) == 8
    assert outcome_sample_weight(base_weight=4, won=True, winning_sample_weight=8, losing_sample_weight=0) == 32


def test_update_policy_epochs_runs_configured_number_of_passes():
    class CountingPolicy:
        def __init__(self):
            self.calls = []

        def update_batch(self, batch, lr, entropy_coef, value_coef):
            self.calls.append((batch, lr, entropy_coef, value_coef))
            return {"loss_policy": float(len(self.calls)), "loss_value": 2.0, "ratio_mean": 1.0}

    policy = CountingPolicy()
    batch = {"candidate_features": [np.zeros((1, RL_FEATURE_DIM), dtype=np.float64)]}

    metrics = update_policy_epochs(policy, batch, lr=0.01, entropy_coef=0.0, value_coef=0.5, ppo_epochs=4)

    assert len(policy.calls) == 4
    assert metrics["loss_policy"] == 4.0
    assert metrics["ppo_epochs"] == 4


def test_weight4_keeps_deterministic_stream_from_dominating_16_stream_update():
    weights = [
        deterministic_stream_sample_weight(i, eval_every=1, configured_weight=4)
        for i in range(16)
    ]

    assert sum(weights) == 19
    assert weights[0] == 4
    assert sum(weights[1:]) == 15
    assert weights[0] / sum(weights) < 0.25


def test_rollout_signal_metrics_separates_deterministic_and_stochastic_winners():
    metrics = rollout_signal_metrics(
        [
            {"deterministic_rollout": True, "won": False, "decision_count": 10, "stream_weight": 4, "rank": 2.0},
            {"deterministic_rollout": False, "won": True, "decision_count": 8, "stream_weight": 1, "rank": 1.0},
            {"deterministic_rollout": False, "won": False, "decision_count": 6, "stream_weight": 1, "rank": 2.0},
        ]
    )

    assert metrics["train_deterministic_rank"] == 2.0
    assert metrics["train_deterministic_won"] is False
    assert metrics["train_stochastic_win_count"] == 1
    assert metrics["train_stochastic_loss_count"] == 1
    assert metrics["train_stochastic_win_rate"] == 0.5
    assert metrics["weighted_winning_decisions"] == 8
    assert metrics["weighted_losing_decisions"] == 46


def test_rollout_signal_metrics_allows_zero_weight_losing_rollouts():
    metrics = rollout_signal_metrics(
        [
            {"deterministic_rollout": True, "won": False, "decision_count": 10, "stream_weight": 0, "rank": 3.0},
            {"deterministic_rollout": False, "won": True, "decision_count": 8, "stream_weight": 4, "rank": 1.0},
        ]
    )

    assert metrics["weighted_winning_decisions"] == 32
    assert metrics["weighted_losing_decisions"] == 0


def test_action_margin_stats_reports_top1_top2_gap_for_gate_decisions():
    policy = LinearActorCritic(feature_dim=2, seed=0)
    policy.actor_w = np.array([1.0, 0.0], dtype=np.float64)
    decisions = [
        {"candidate_features": np.array([[1.0, 0.0], [0.75, 0.0], [0.0, 1.0]], dtype=np.float64)},
        {"candidate_features": np.array([[0.2, 0.0], [0.1, 1.0]], dtype=np.float64)},
    ]

    stats = action_margin_stats(policy, decisions)

    assert stats["gate_action_margin_mean"] == 0.175
    assert np.isclose(stats["gate_action_margin_p10"], 0.115)
    assert stats["gate_action_margin_min"] == 0.1
    assert stats["gate_low_margin_fraction_0p01"] == 0.0
    assert stats["gate_low_margin_fraction_0p05"] == 0.0


def test_argmax_flip_stats_compares_reference_policy_for_gate_decisions():
    before = LinearActorCritic(feature_dim=2, seed=0)
    after = LinearActorCritic(feature_dim=2, seed=0)
    before.actor_w = np.array([1.0, 0.0], dtype=np.float64)
    after.actor_w = np.array([0.0, 1.0], dtype=np.float64)
    decisions = [
        {"candidate_features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)},
        {"candidate_features": np.array([[0.2, 0.0], [0.1, 0.0]], dtype=np.float64)},
    ]

    stats = argmax_flip_stats(before, after, decisions)

    assert stats["gate_pre_update_argmax_flips"] == 1
    assert stats["gate_pre_update_argmax_flip_rate"] == 0.5
    assert stats["gate_pre_update_action_match_rate"] == 0.5


def test_gate_policy_rollback_restores_last_clean_policy_after_streak_break():
    clean = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=1)
    failed = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=2)
    clean.actor_w[:] = 3.0
    failed.actor_w[:] = -3.0
    clean_policies = {}

    remember_clean_gate_policy(clean_policies, seed=123, policy=clean, eval_clean_win=True)
    restored, did_restore = restore_clean_gate_policy(
        clean_policies,
        seed=123,
        policy=failed,
        eval_clean_win=False,
        gate_streak_before=2,
        enabled=True,
    )

    assert did_restore is True
    assert np.all(restored.actor_w == 3.0)
    assert restored is not clean


def test_gate_policy_rollback_only_when_clean_streak_breaks_and_is_enabled():
    assert should_restore_clean_gate_policy(
        eval_clean_win=False, gate_streak_before=1, clean_policy_available=True, enabled=True
    ) is True
    assert should_restore_clean_gate_policy(
        eval_clean_win=False, gate_streak_before=0, clean_policy_available=True, enabled=True
    ) is False
    assert should_restore_clean_gate_policy(
        eval_clean_win=True, gate_streak_before=1, clean_policy_available=True, enabled=True
    ) is False
    assert should_restore_clean_gate_policy(
        eval_clean_win=False, gate_streak_before=1, clean_policy_available=False, enabled=True
    ) is False
    assert should_restore_clean_gate_policy(
        eval_clean_win=False, gate_streak_before=1, clean_policy_available=True, enabled=False
    ) is False


def test_policy_lr_for_gate_streak_reduces_updates_after_clean_gate_wins():
    assert policy_lr_for_gate_streak(0.01, clean_win_streak=0) == 0.01
    assert policy_lr_for_gate_streak(0.01, clean_win_streak=1) == 0.0025
    assert policy_lr_for_gate_streak(0.01, clean_win_streak=2) == 0.0025
    assert policy_lr_for_gate_streak(0.01, clean_win_streak=3) == 0.001
    assert policy_lr_for_gate_streak(
        0.01,
        clean_win_streak=4,
        streak_multiplier=0.2,
        high_streak_threshold=4,
        high_streak_multiplier=0.05,
    ) == 0.0005


def test_policy_lr_for_gate_streak_can_wait_for_required_clean_streak_before_reducing():
    assert policy_lr_for_gate_streak(0.03, clean_win_streak=0, min_streak_for_reduction=3) == 0.03
    assert policy_lr_for_gate_streak(0.03, clean_win_streak=1, min_streak_for_reduction=3) == 0.03
    assert policy_lr_for_gate_streak(0.03, clean_win_streak=2, min_streak_for_reduction=3) == 0.03
    assert policy_lr_for_gate_streak(
        0.03,
        clean_win_streak=3,
        streak_multiplier=0.1,
        high_streak_threshold=3,
        high_streak_multiplier=0.03,
        min_streak_for_reduction=3,
    ) == 0.0009


def test_rollback_lr_cooldown_uses_broken_streak_without_changing_scheduler_streak():
    scheduler = WinStaySeedScheduler(seed=7, advance_win_streak=5)
    scheduler.clean_win_streak = 2
    scheduler.after_episode(won=False)
    cooldown = RollbackLRCooldown()

    cooldown.arm(gate_policy_restored=True, gate_streak_before=2, cooldown_updates=2)

    assert scheduler.clean_win_streak == 0
    assert cooldown.streak_for_update(clean_win_streak=scheduler.clean_win_streak) == 2
    assert policy_lr_for_gate_streak(0.01, cooldown.streak_for_update(0)) == 0.0025


def test_rollback_lr_cooldown_uses_high_streak_reduction_and_expires():
    cooldown = RollbackLRCooldown()

    cooldown.arm(gate_policy_restored=True, gate_streak_before=3, cooldown_updates=2)

    assert policy_lr_for_gate_streak(0.01, cooldown.streak_for_update(0)) == 0.001
    assert cooldown.active is True

    cooldown.consume_update()
    assert cooldown.streak_for_update(0) == 3
    cooldown.consume_update()
    assert cooldown.active is False
    assert cooldown.streak_for_update(0) == 0


def test_strict_gate_and_deterministic_training_stream_use_latest_opponent_snapshot():
    fallback = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=1)
    older = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=2)
    newer = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=3)
    older.actor_w[:] = 1.0
    newer.actor_w[:] = 2.0
    pool = OpponentPool(capacity=4, seed=0)
    pool.add_snapshot(older)
    pool.add_snapshot(newer)
    accepted = AcceptedGateOpponent()
    accepted.promote(older, update=0)

    gate_opponent = strict_gate_opponent_policy(accepted, fallback)
    stream0_opponent = opponent_policy_for_rollout(
        pool,
        accepted,
        fallback,
        eval_every=1,
        rollout_stream_index=0,
    )

    assert np.all(gate_opponent.actor_w == 1.0)
    assert np.all(stream0_opponent.actor_w == 1.0)
    assert np.all(pool.latest().actor_w == 2.0)


def test_strict_gate_opponents_can_be_used_for_all_training_streams():
    fallback = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=1)
    older = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=2)
    newer = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=3)
    older.actor_w[:] = 1.0
    newer.actor_w[:] = 2.0
    pool = OpponentPool(capacity=4, seed=0)
    pool.add_snapshot(newer)
    accepted = AcceptedGateOpponent()
    accepted.promote(older, update=0)

    stream1_opponent = opponent_policy_for_rollout(
        pool,
        accepted,
        fallback,
        eval_every=1,
        rollout_stream_index=1,
        strict_gate_opponents_all_streams=True,
    )

    assert np.all(stream1_opponent.actor_w == 1.0)


def test_load_agent_ref_loads_module_function_and_agent_file(tmp_path):
    module_agent = load_agent_ref("src.ow.chadman_baseline:agent")
    assert callable(module_agent)

    agent_file = tmp_path / "agent_file.py"
    agent_file.write_text("def agent(obs, config=None):\n    return []\n")
    file_agent = load_agent_ref(str(agent_file))

    assert callable(file_agent)
    assert file_agent({}, None) == []


def test_select_frozen_or_policy_opponent_prefers_black_box_frozen_agent():
    frozen = lambda obs, config=None: []
    policy = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=3)

    assert select_frozen_or_policy_opponent(frozen, policy) is frozen
    assert select_frozen_or_policy_opponent(None, policy) is policy


def test_opponent_policy_for_rollout_falls_back_when_pool_is_empty():
    fallback = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=1)
    pool = OpponentPool(capacity=4, seed=0)
    accepted = AcceptedGateOpponent()

    assert strict_gate_opponent_policy(accepted, fallback) is fallback
    assert opponent_policy_for_rollout(pool, accepted, fallback, eval_every=1, rollout_stream_index=0) is fallback


def test_gate_opponent_promotion_is_explicit_and_versioned():
    accepted = AcceptedGateOpponent()
    policy = LinearActorCritic(feature_dim=RL_FEATURE_DIM, seed=4)
    policy.actor_w[:] = 3.0

    assert accepted.version == 0
    assert should_promote_gate_opponent(gate_seed_advanced=True, enabled=False) is False
    assert should_promote_gate_opponent(gate_seed_advanced=False, enabled=True) is False
    assert should_promote_gate_opponent(gate_seed_advanced=True, enabled=True) is True

    accepted.promote(policy, update=9)
    policy.actor_w[:] = 4.0

    assert accepted.version == 1
    assert accepted.accepted_update == 9
    assert np.all(accepted.current(policy).actor_w == 3.0)


def test_append_episode_batch_can_weight_deterministic_stream_without_duplicate_metrics():
    batch = {"candidate_features": [], "chosen_indices": [], "old_logps": [], "returns": [], "advantages": []}
    ep_batch = {
        "candidate_features": [np.array([[1.0, 0.0]], dtype=np.float64), np.array([[0.0, 1.0]], dtype=np.float64)],
        "chosen_indices": np.array([0, 1], dtype=np.int64),
        "old_logps": np.array([-0.1, -0.2], dtype=np.float64),
        "returns": np.array([-2.0, -1.5], dtype=np.float64),
        "advantages": np.array([-2.1, -1.6], dtype=np.float64),
    }

    append_episode_batch(batch, ep_batch, repeat_count=3)

    assert len(batch["candidate_features"]) == 6
    assert batch["chosen_indices"] == [0, 1, 0, 1, 0, 1]
    assert batch["returns"] == [-2.0, -1.5, -2.0, -1.5, -2.0, -1.5]


def test_decision_outcome_sample_weights_use_each_decision_owner_not_player0():
    weights = decision_outcome_sample_weights(
        player_ids=np.array([0, 1, 2, 3], dtype=np.int64),
        ranks=[2.0, 3.0, 1.0, 4.0],
        base_weight=1,
        winning_sample_weight=4,
        losing_sample_weight=1,
    )

    assert weights.tolist() == [1, 1, 4, 1]


def test_append_episode_batch_by_decision_weight_can_keep_only_winner_seat_decisions():
    batch = {"candidate_features": [], "chosen_indices": [], "old_logps": [], "returns": [], "advantages": []}
    ep_batch = {
        "candidate_features": [
            np.array([[1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0]], dtype=np.float64),
            np.array([[1.0, 1.0]], dtype=np.float64),
        ],
        "chosen_indices": np.array([0, 1, 0], dtype=np.int64),
        "old_logps": np.array([-0.1, -0.2, -0.3], dtype=np.float64),
        "returns": np.array([-2.0, 2.5, -1.5], dtype=np.float64),
        "advantages": np.array([-2.1, 2.4, -1.6], dtype=np.float64),
        "player_ids": np.array([0, 2, 3], dtype=np.int64),
    }

    weights = decision_outcome_sample_weights(
        ep_batch["player_ids"],
        ranks=[2.0, 3.0, 1.0, 4.0],
        base_weight=1,
        winning_sample_weight=4,
        losing_sample_weight=0,
    )
    append_episode_batch_by_decision_weight(batch, ep_batch, repeat_weights=weights)

    assert len(batch["candidate_features"]) == 4
    assert batch["chosen_indices"] == [1, 1, 1, 1]
    assert batch["returns"] == [2.5, 2.5, 2.5, 2.5]


def test_rollout_signal_metrics_prefers_explicit_weighted_decision_counts():
    metrics = rollout_signal_metrics(
        [
            {
                "deterministic_rollout": False,
                "won": False,
                "decision_count": 12,
                "stream_weight": 1,
                "weighted_winning_decisions": 8,
                "weighted_losing_decisions": 4,
                "rank": 2.0,
            }
        ]
    )

    assert metrics["weighted_winning_decisions"] == 8
    assert metrics["weighted_losing_decisions"] == 4


def test_gate_strict_clean_win_requires_first_place_and_ship_loss_20_or_less():
    assert gate_strict_clean_win([1.0, 2.0], {"wasted_ships": 19}, phase_mode="2p", expected_phase_mode="2p") is True
    assert gate_strict_clean_win([1.0, 2.0], {"wasted_ships": 20}, phase_mode="2p", expected_phase_mode="2p") is True
    assert gate_strict_clean_win([1.5, 1.5], {"wasted_ships": 0}, phase_mode="2p", expected_phase_mode="2p") is False
    assert gate_strict_clean_win([1.0, 2.0], {"wasted_ships": 21}, phase_mode="2p", expected_phase_mode="2p") is False
    assert gate_strict_clean_win([1.0, 2.0], {"wasted_ships": 0}, phase_mode="4p", expected_phase_mode="2p") is False


def test_summarize_decision_metrics_reports_gate_eval_mistakes_instead_of_forced_zeroes():
    decisions = [
        {"mistake_metrics": {"wasted_ships": 12, "sun_loss_ships": 12, "invalid_actions": 0}, "action_kind": "launch"},
        {"mistake_metrics": {"wasted_ships": 5, "sun_loss_ships": 0, "invalid_actions": 1}, "action_kind": "noop"},
    ]

    summary = summarize_decision_metrics(decisions)

    assert summary["wasted_ships"] == 17
    assert summary["sun_loss_ships"] == 12
    assert summary["invalid_actions"] == 1
    assert summary["launch_actions"] == 1
    assert summary["noop_actions"] == 1


def test_strict_gate_eval_uses_deterministic_policy_for_all_actors():
    assert gate_actor_should_be_deterministic(trainable=True) is True
    assert gate_actor_should_be_deterministic(trainable=False) is True


def test_trainable_actor_rng_seed_is_stable_for_gate_aligned_sampling():
    seed_a = actor_rng_seed_for_episode(episode_seed=1790251936, player_id=0)

    assert seed_a == actor_rng_seed_for_episode(episode_seed=1790251936, player_id=0)
    assert seed_a != actor_rng_seed_for_episode(episode_seed=1790251937, player_id=0)
    assert seed_a != actor_rng_seed_for_episode(episode_seed=1790251936, player_id=1)


def test_trainable_actor_rng_seed_varies_by_rollout_stream_on_repeated_env_seed():
    gate_stream = actor_rng_seed_for_episode(episode_seed=1790251936, player_id=0, rollout_index=0)

    assert gate_stream == actor_rng_seed_for_episode(episode_seed=1790251936, player_id=0)
    assert gate_stream != actor_rng_seed_for_episode(episode_seed=1790251936, player_id=0, rollout_index=1)
    assert len({actor_rng_seed_for_episode(1790251936, 0, rollout_index=i) for i in range(16)}) == 16


def test_rollout_stream_schedule_keeps_exploration_diverse_on_repeated_seed():
    streams = [rollout_stream_index_for_episode(ep, episodes_per_update=16, eval_every=1) for ep in range(1, 17)]

    assert streams == list(range(16))
    assert rollout_stream_index_for_episode(episode=3, episodes_per_update=16, eval_every=0) == 2


def test_training_episode_seed_preserves_gate_seed_by_default():
    assert training_episode_seed(
        base_seed=7,
        gate_seed=2029167940,
        global_episode=11,
        rollout_stream_index=3,
        diverse_training_seeds=False,
    ) == 2029167940


def test_training_episode_seed_diversifies_rollouts_without_changing_gate_seed():
    gate_seed = 2029167940


    seeds = {
        training_episode_seed(
            base_seed=7,
            gate_seed=gate_seed,
            global_episode=11,
            rollout_stream_index=stream,
            diverse_training_seeds=True,
        )
        for stream in range(4)
    }

    assert len(seeds) == 4
    assert gate_seed not in seeds
    assert seeds == {
        training_episode_seed(
            base_seed=7,
            gate_seed=gate_seed,
            global_episode=11,
            rollout_stream_index=stream,
            diverse_training_seeds=True,
        )
        for stream in range(4)
    }


def test_training_feature_flags_prove_safety_rewards_and_viewer_settings_are_active():
    args = parse_args(["--viewer-draw-interval-seconds", "30"])

    flags = training_feature_flags(args, phase_mode="2p")

    assert flags == {
        "safe_action_candidates": True,
        "terminal_reward_scale": 2.5,
        "max_mistake_penalty": 3.0,
        "fourp_rank_reward_scale": 2.5,
        "asset_margin_reward_scale": 0.25,
        "dense_reward_scale": 0.0,
        "reward_gamma": 0.99,
        "viewer_draw_interval_seconds": 30.0,
        "viewer_event_poll_interval_seconds": 1.0,
        "policy_lr": 0.01,
        "entropy_coef": 0.003,
        "value_coef": 0.5,
        "deterministic_stream_weight": 1,
        "winning_sample_weight": 1,
        "losing_sample_weight": 1,
        "ppo_epochs": 1,
        "train_all_players": False,
        "train_seat": "player0",
        "strict_gate_opponents_all_streams": False,
        "frozen_opponent_agent": False,
        "diverse_training_seeds": False,
        "gate_advance_win_streak": 3,
        "gate_streak_lr_multiplier": 0.25,
        "gate_streak_high_threshold": 3,
        "gate_streak_high_lr_multiplier": 0.1,
        "gate_streak_lr_min_streak": 1,
        "rollback_lr_cooldown_enabled": True,
        "rollback_lr_cooldown_updates": 2,
        "gate_policy_rollback": True,
        "promote_gate_opponent_on_advance": False,
        "phase_mode_effective": "2p",
    }


def test_parse_args_accepts_4p_mode():
    args = parse_args(["--mode", "4p", "--no-viewer"])
    assert args.mode == "4p"
    assert mode_player_count(args.mode) == 4


def test_parse_args_accepts_mixed_mode():
    args = parse_args(["--mode", "mixed", "--no-viewer"])
    assert args.mode == "mixed"


def test_mode_player_count_keeps_2p_behavior():
    assert mode_player_count("2p") == 2


def test_player_training_roles_use_protagonist_vs_frozen_opponents():
    assert player_is_trainable("2p", 0) is True
    assert player_is_trainable("2p", 1) is False
    assert player_is_trainable("4p", 0) is True
    assert player_is_trainable("4p", 1) is False
    assert player_is_trainable("4p", 3) is False


def test_training_player_roles_can_enable_shared_policy_all_seats_for_4p_self_play():
    assert [training_player_is_trainable("4p", i, train_all_players=True) for i in range(4)] == [True, True, True, True]
    assert [training_player_is_trainable("4p", i, train_all_players=False) for i in range(4)] == [True, False, False, False]
    assert [training_player_is_trainable("2p", i, train_all_players=True) for i in range(2)] == [True, False]


def test_training_player_roles_can_alternate_2p_train_seat_by_global_episode():
    assert training_focus_player_id("2p", global_episode=1, train_seat="alternating") == 0
    assert training_focus_player_id("2p", global_episode=2, train_seat="alternating") == 1
    assert training_focus_player_id("2p", global_episode=3, train_seat="alternating") == 0

    assert [
        training_player_is_trainable("2p", i, train_all_players=False, train_seat="alternating", global_episode=1)
        for i in range(2)
    ] == [True, False]
    assert [
        training_player_is_trainable("2p", i, train_all_players=False, train_seat="alternating", global_episode=2)
        for i in range(2)
    ] == [False, True]


def test_training_player_roles_ignore_alternating_train_seat_for_4p():
    assert training_focus_player_id("4p", global_episode=2, train_seat="alternating") == 0
    assert [
        training_player_is_trainable("4p", i, train_all_players=False, train_seat="alternating", global_episode=2)
        for i in range(4)
    ] == [True, False, False, False]


def test_gate_eval_player_roles_keep_only_player0_trainable_for_strict_gate():
    assert [gate_eval_player_is_trainable("4p", i) for i in range(4)] == [True, False, False, False]
    assert [gate_eval_player_is_trainable("2p", i) for i in range(2)] == [True, False]


def test_episode_player_count_alternates_for_mixed_mode():
    assert episode_player_count("2p", episode=1) == 2
    assert episode_player_count("2p", episode=2) == 2
    assert episode_player_count("4p", episode=1) == 4
    assert episode_player_count("4p", episode=2) == 4
    assert episode_player_count("mixed", episode=1) == 2
    assert episode_player_count("mixed", episode=2) == 4
    assert episode_player_count("mixed", episode=3) == 2


def test_episode_mode_returns_effective_phase_mode():
    assert episode_mode("2p", episode=1) == "2p"
    assert episode_mode("4p", episode=1) == "4p"
    assert episode_mode("mixed", episode=1) == "2p"
    assert episode_mode("mixed", episode=2) == "4p"


def test_global_episode_number_keeps_mixed_alternating_across_updates():
    assert global_episode_number(update=1, episode=1, episodes_per_update=1) == 1
    assert global_episode_number(update=2, episode=1, episodes_per_update=1) == 2
    assert episode_mode("mixed", global_episode_number(update=2, episode=1, episodes_per_update=1)) == "4p"


def test_mode_player_count_reports_mixed_max_players_for_setup():
    assert mode_player_count("mixed") == 4


def test_build_run_dir_uses_mode(tmp_path):
    path = build_run_dir(tmp_path, "2p")
    assert path.name.startswith("2p_")


def test_build_run_dir_refuses_existing_explicit_run_dir_by_default(tmp_path):
    existing = tmp_path / "rl_existing"
    existing.mkdir()
    (existing / "metrics.jsonl").write_text("{}\n")

    try:
        build_run_dir(existing, "2p")
    except FileExistsError as exc:
        assert str(existing) in str(exc)
    else:
        raise AssertionError("expected existing explicit run dir to be refused")


def test_build_run_dir_can_explicitly_allow_existing_run_dir(tmp_path):
    existing = tmp_path / "rl_existing"
    existing.mkdir()

    path = build_run_dir(existing, "2p", allow_existing=True)

    assert path == existing
    assert (existing / "checkpoints").is_dir()


def test_build_run_dir_treats_4p_prefix_as_explicit_run_dir(tmp_path):
    target = tmp_path / "4p_fresh"

    path = build_run_dir(target, "4p")

    assert path == target
    assert (target / "checkpoints").is_dir()


def test_trajectory_recorder_stores_real_decision_fields():
    rec = EpisodeTrajectoryRecorder()
    obs = {"player": 0, "planets": [[0, 0, 50.0, 50.0, 2.0, 20, 4]]}
    feats = np.array([[0.2, 1.0, 1.0], [0.5, 1.5, 1.0]], dtype=np.float64)
    rec.record(
        player_id=0,
        observation=obs,
        candidate_features=feats,
        chosen_idx=1,
        old_logp=-0.9,
        value=0.2,
        mistake_penalty=-0.4,
        mistake_metrics={"sun_loss_ships": 3, "wasted_ships": 3, "invalid_actions": 0},
    )

    assert len(rec.decisions) == 1
    d = rec.decisions[0]
    assert d["player_id"] == 0
    assert d["chosen_idx"] == 1
    assert d["old_logp"] == -0.9
    assert d["value"] == 0.2
    assert d["candidate_features"].shape == (2, 3)
    assert d["mistake_penalty"] == -0.4
    assert d["mistake_metrics"]["sun_loss_ships"] == 3
    assert rec.latest_observation is obs


def test_finalize_episode_records_assigns_player_relative_returns_and_opposite_advantages():
    rec = EpisodeTrajectoryRecorder()
    f0 = np.array([[0.1, 1.0, 1.0], [0.2, 1.0, 1.0]], dtype=np.float64)
    f1 = np.array([[0.9, 1.0, 1.0], [0.4, 1.0, 1.0]], dtype=np.float64)
    rec.record(0, {"player": 0}, f0, 0, -0.7, 0.1, mistake_penalty=-0.25)
    rec.record(1, {"player": 1}, f1, 1, -0.6, -0.2, mistake_penalty=-0.5)

    batch = finalize_episode_records(rec.decisions, r0=3.0, r1=1.0)
    assert np.allclose(batch["returns"], np.array([1.75, -2.5]))
    assert np.allclose(batch["advantages"], np.array([1.65, -2.3]))


def test_step_efficiency_penalty_is_low_and_linear():
    assert step_efficiency_penalty(episode_steps=0, weight=0.001) == 0.0
    assert step_efficiency_penalty(episode_steps=120, weight=0.001) == -0.12
    assert step_efficiency_penalty(episode_steps=120, weight=0.0) == 0.0


def test_finalize_episode_records_applies_step_penalty_and_reward_scale():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    rec.record(0, {"player": 0}, feats, 0, -0.7, 0.1, mistake_penalty=-0.25)
    rec.record(1, {"player": 1}, feats, 0, -0.6, -0.2, mistake_penalty=-0.5)

    batch = finalize_episode_records(
        rec.decisions,
        r0=3.0,
        r1=1.0,
        episode_steps=50,
        step_penalty_weight=0.001,
        terminal_reward_scale=1.5,
    )
    assert np.allclose(batch["returns"], np.array([2.70, -3.55]))
    assert batch["step_penalty"] == -0.05


def test_finalize_episode_records_can_cap_total_mistake_penalty_per_decision():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    rec.record(0, {"player": 0}, feats, 0, -0.7, 0.1, mistake_penalty=-99.0)

    batch = finalize_episode_records(rec.decisions, r0=1.0, r1=0.0, max_mistake_penalty=0.5)

    assert np.allclose(batch["returns"], np.array([0.5]))


def test_finalize_episode_records_penalizes_2p_ties_as_losses_for_both_players():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    rec.record(0, {"player": 0}, feats, 0, -0.7, 0.0)
    rec.record(1, {"player": 1}, feats, 0, -0.6, 0.0)

    batch = finalize_episode_records(rec.decisions, r0=1.0, r1=1.0, tie_loss_penalty=1.0)

    assert np.allclose(batch["returns"], np.array([-1.0, -1.0]))


def test_finalize_episode_records_adds_asset_margin_shaping_without_overriding_tie_loss():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    rec.record(0, {"player": 0}, feats, 0, -0.7, 0.0)
    rec.record(1, {"player": 1}, feats, 0, -0.6, 0.0)

    batch = finalize_episode_records(
        rec.decisions,
        r0=1.0,
        r1=1.0,
        tie_loss_penalty=1.0,
        asset_margin_rewards=[0.2, -0.2],
    )

    assert np.allclose(batch["returns"], np.array([-0.8, -1.2]))
    assert np.all(batch["returns"] < 0.0)


def test_finalize_episode_records_dense_reward_gives_credit_for_neutral_capture_even_when_lost():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    before = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 40, 4],
            [1, -1, 20.0, 10.0, 2.0, 10, 3],
            [2, 1, 90.0, 90.0, 2.0, 70, 6],
        ],
        "fleets": [],
    }
    after_capture = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 35, 4],
            [1, 0, 20.0, 10.0, 2.0, 8, 3],
            [2, 1, 90.0, 90.0, 2.0, 75, 6],
        ],
        "fleets": [],
    }

    rec.record(0, before, feats, 0, -0.7, 0.0)
    rec.record(0, after_capture, feats, 0, -0.7, 0.0)

    batch = finalize_episode_records(
        rec.decisions,
        r0=0.0,
        r1=1.0,
        terminal_reward_scale=1.0,
        dense_reward_scale=2.0,
        reward_gamma=1.0,
    )

    assert batch["returns"][0] > 0.0
    assert batch["returns"][0] > batch["returns"][1]


def test_finalize_episode_records_dense_reward_penalizes_asset_loss_before_terminal():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    before = {
        "player": 0,
        "planets": [[0, 0, 10.0, 10.0, 2.0, 70, 6], [1, 1, 90.0, 90.0, 2.0, 40, 4]],
        "fleets": [],
    }
    after_loss = {
        "player": 0,
        "planets": [[0, 1, 10.0, 10.0, 2.0, 30, 6], [1, 1, 90.0, 90.0, 2.0, 50, 4]],
        "fleets": [],
    }

    rec.record(0, before, feats, 0, -0.7, 0.0)
    rec.record(0, after_loss, feats, 0, -0.7, 0.0)

    batch = finalize_episode_records(
        rec.decisions,
        r0=1.0,
        r1=0.0,
        terminal_reward_scale=0.0,
        dense_reward_scale=1.0,
        reward_gamma=1.0,
    )

    assert batch["returns"][0] < 0.0


def test_player_asset_margin_rewards_scores_final_asset_advantage_symmetrically():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 60, 5],
        [1, 1, 30.0, 10.0, 2.0, 20, 2],
    ]
    state = [_state(0, 0.0, planets), _state(1, 0.0, planets)]

    rewards = player_asset_margin_rewards(state, n_players=2, scale=0.25)

    assert rewards[0] > 0.0
    assert rewards[1] < 0.0
    assert np.isclose(rewards[0], -rewards[1])


def test_finalize_episode_records_sums_mistake_metrics():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    rec.record(
        0,
        {"player": 0},
        feats,
        0,
        -0.1,
        0.0,
        mistake_penalty=-0.2,
        mistake_metrics={"sun_loss_ships": 4, "wasted_ships": 5, "invalid_actions": 1},
    )
    rec.record(
        0,
        {"player": 0},
        feats,
        0,
        -0.1,
        0.0,
        mistake_penalty=-0.3,
        mistake_metrics={"sun_loss_ships": 2, "wasted_ships": 7, "invalid_actions": 0},
    )
    batch = finalize_episode_records(rec.decisions, r0=0.0, r1=1.0)
    assert batch["mistake_metrics"]["sun_loss_ships"] == 6
    assert batch["mistake_metrics"]["wasted_ships"] == 12
    assert batch["mistake_metrics"]["invalid_actions"] == 1


def test_finalize_episode_records_logs_action_and_noop_counts():
    rec = EpisodeTrajectoryRecorder()
    feats = np.array([[0.1, 1.0, 1.0]], dtype=np.float64)
    rec.record(0, {"player": 0}, feats, 0, -0.1, 0.0, action_kind="noop")
    rec.record(0, {"player": 0}, feats, 0, -0.1, 0.0, action_kind="launch")

    batch = finalize_episode_records(rec.decisions, r0=1.0, r1=0.0)

    assert batch["action_metrics"] == {"decisions": 2, "noop_actions": 1, "launch_actions": 1}


def test_featurize_viewer_payload_can_use_real_captured_observation():
    obs = {
        "player": 0,
        "planets": [[0, 0, 50.0, 50.0, 2.0, 20, 4], [1, 1, 60.0, 55.0, 2.0, 20, 3]],
    }
    rec = EpisodeTrajectoryRecorder()
    cands = [
        {"angle": 0.0, "ships": 3},
        {"angle": 1.5, "ships": 5},
    ]
    feats = np.stack([featurize(c) for c in cands], axis=0)
    rec.record(0, obs, feats, 0, -0.5, 0.1)
    assert rec.latest_observation == obs


def test_featurize_uses_richer_target_quality_features_for_rl_learning():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 50, 4],
            [1, -1, 40.0, 10.0, 2.0, 12, 6],
        ],
    }

    launch = next(c for c in build_action_candidates(obs) if c["kind"] == "launch" and c["ships"] == 27)
    features = featurize(launch)

    assert features.shape == (RL_FEATURE_DIM,)
    assert RL_FEATURE_DIM > 3
    assert features[0] == 1.0  # launch flag
    assert features[3] == 27.0 / 100.0  # sent ships normalized
    assert features[4] == 30.0 / 100.0  # distance normalized
    assert features[5] == 12.0 / 100.0  # target ships normalized
    assert features[6] == 6.0 / 10.0  # production normalized
    assert features[7] == 1.0  # neutral target flag
    assert features[8] == 0.0  # own target flag
    assert features[9] == 0.0  # enemy target flag


def test_build_action_candidates_exposes_geometry_for_feature_expansion():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 20.0, 3.0, 50, 4],
            [1, 1, 40.0, 60.0, 2.0, 12, 0],
        ],
    }

    launch = next(c for c in build_action_candidates(obs, avoid_waste=False) if c["kind"] == "launch")

    assert launch["source_x"] == 10.0
    assert launch["source_y"] == 20.0
    assert launch["target_x"] == 40.0
    assert launch["target_y"] == 60.0
    assert launch["target_radius"] == 2.0


def test_featurize_adds_geometry_capture_and_value_features_for_4p_capacity():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 20.0, 3.0, 160, 5],
            [1, 1, 40.0, 60.0, 2.0, 20, 4],
            [2, -1, 90.0, 10.0, 4.0, 10, 2],
        ],
    }

    launch = next(c for c in build_action_candidates(obs, avoid_waste=False) if c["target_id"] == 1 and c["ships"] == 88)
    features = featurize(launch)

    assert RL_FEATURE_DIM == 32
    assert features.shape == (RL_FEATURE_DIM,)
    assert features[19] == 10.0 / 100.0  # source x
    assert features[20] == 20.0 / 100.0  # source y
    assert features[21] == 40.0 / 100.0  # target x
    assert features[22] == 60.0 / 100.0  # target y
    assert features[23] == 2.0 / 10.0  # target radius
    assert features[24] == 88.0 / 160.0  # sent fraction of source ships
    assert features[25] == (88.0 - 20.0) / 100.0  # capture surplus proxy
    assert features[26] == 160.0 / 160.0  # source share of own ships
    expected_value_per_distance = (20.0 + 8.0 * 4.0 + 25.0) / (50.0 + 1.0) / 10.0
    assert np.isclose(features[27], expected_value_per_distance)
    assert np.isclose(features[28], 4.0 / (50.0 + 1.0))
    assert features[29] == (160.0 - 20.0) / 500.0  # ship advantage over enemies
    assert features[30] == (5.0 - 4.0) / 50.0  # production advantage over enemies
    assert features[31] == 1.0 / (1.0 + 50.0 / 100.0)  # inverse normalized distance


def test_featurize_distinguishes_reinforcement_enemy_and_global_context_for_rl_learning():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 3.0, 160, 5],
            [1, 0, 40.0, 10.0, 2.0, 20, 4],
            [2, 1, 70.0, 10.0, 2.0, 30, 3],
            [3, -1, 90.0, 10.0, 2.0, 10, 2],
        ],
    }

    launches = [c for c in build_action_candidates(obs, avoid_waste=False) if c["kind"] == "launch"]
    own = featurize(next(c for c in launches if c["target_id"] == 1 and c["ships"] == 88))
    enemy = featurize(next(c for c in launches if c["target_id"] == 2 and c["ships"] == 88))

    assert own.shape == (RL_FEATURE_DIM,)
    assert RL_FEATURE_DIM >= 19
    assert own[8] == 1.0
    assert own[9] == 0.0
    assert enemy[8] == 0.0
    assert enemy[9] == 1.0
    assert own[10] == 2.0 / 10.0  # my planet count
    assert own[11] == 1.0 / 10.0  # enemy planet count
    assert own[12] == 1.0 / 10.0  # neutral planet count
    assert own[13] == 180.0 / 500.0  # my total ships
    assert own[14] == 30.0 / 500.0  # enemy total ships
    assert own[15] == 10.0 / 500.0  # neutral total ships
    assert own[16] == 9.0 / 50.0  # my production
    assert own[17] == 3.0 / 50.0  # enemy production
    assert own[18] == 160.0 / 100.0  # source ships


def test_win_stay_seed_scheduler_changes_seed_after_win_and_repeats_after_loss():
    scheduler = WinStaySeedScheduler(seed=7)
    first = scheduler.current_seed

    scheduler.after_episode(won=False)
    assert scheduler.current_seed == first

    scheduler.after_episode(won=True)
    second = scheduler.current_seed
    assert second != first

    scheduler.after_episode(won=False)
    assert scheduler.current_seed == second


def test_win_stay_seed_scheduler_requires_configured_clean_gate_streak_before_advancing():
    scheduler = WinStaySeedScheduler(seed=7, advance_win_streak=3)
    first = scheduler.current_seed

    scheduler.after_episode(won=True)
    assert scheduler.current_seed == first
    assert scheduler.clean_win_streak == 1
    assert scheduler.last_seed_advanced is False

    scheduler.after_episode(won=True)
    assert scheduler.current_seed == first
    assert scheduler.clean_win_streak == 2

    scheduler.after_episode(won=True)
    assert scheduler.current_seed != first
    assert scheduler.clean_win_streak == 0
    assert scheduler.last_seed_advanced is True


def test_win_stay_seed_scheduler_resets_clean_gate_streak_on_loss():
    scheduler = WinStaySeedScheduler(seed=7, advance_win_streak=3)
    first = scheduler.current_seed

    scheduler.after_episode(won=True)
    scheduler.after_episode(won=False)

    assert scheduler.current_seed == first
    assert scheduler.clean_win_streak == 0
    assert scheduler.last_seed_advanced is False


def test_win_stay_seed_scheduler_is_reproducible_from_base_seed():
    a = WinStaySeedScheduler(seed=11)
    b = WinStaySeedScheduler(seed=11)

    for won in [True, True, False, True]:
        a.after_episode(won=won)
        b.after_episode(won=won)
        assert a.current_seed == b.current_seed


def test_player_outcome_rewards_only_unique_rank1_is_positive_for_4p_equal_env_rewards():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 40, 5],
        [1, 1, 30.0, 10.0, 2.0, 30, 4],
        [2, 2, 50.0, 10.0, 2.0, 20, 3],
        [3, 3, 70.0, 10.0, 2.0, 10, 2],
    ]
    state = [_state(pid, 1.0, planets) for pid in range(4)]
    rewards, ranks = player_outcome_rewards(state, n_players=4)
    assert rewards == [1.0, -1.0, -1.0, -1.0]
    assert ranks == [1, 2, 3, 4]


def test_player_outcome_rewards_can_strengthen_4p_strict_win_rewards():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 40, 5],
        [1, 1, 30.0, 10.0, 2.0, 30, 4],
        [2, 2, 50.0, 10.0, 2.0, 20, 3],
        [3, 3, 70.0, 10.0, 2.0, 10, 2],
    ]
    state = [_state(pid, 1.0, planets) for pid in range(4)]
    rewards, ranks = player_outcome_rewards(state, n_players=4, fourp_rank_reward_scale=2.0)
    assert rewards == [2.0, -2.0, -2.0, -2.0]
    assert ranks == [1, 2, 3, 4]


def test_player_outcome_rewards_does_not_break_equal_asset_ties_by_player_index():
    planets = [[pid, pid, 10.0 * pid, 10.0, 2.0, 20, 3] for pid in range(4)]
    state = [_state(pid, 1.0, planets) for pid in range(4)]
    rewards, ranks = player_outcome_rewards(state, n_players=4)
    assert rewards == [-1.0, -1.0, -1.0, -1.0]
    assert ranks == [2.5, 2.5, 2.5, 2.5]


def test_player_outcome_rewards_averages_partial_asset_ties():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 40, 5],
        [1, 1, 30.0, 10.0, 2.0, 40, 5],
        [2, 2, 50.0, 10.0, 2.0, 10, 2],
        [3, 3, 70.0, 10.0, 2.0, 10, 2],
    ]
    state = [_state(pid, 1.0, planets) for pid in range(4)]
    rewards, ranks = player_outcome_rewards(state, n_players=4)
    assert rewards == [-1.0, -1.0, -1.0, -1.0]
    assert ranks == [1.5, 1.5, 3.5, 3.5]


def test_player_outcome_rewards_env_reward_rank2_is_negative_for_4p():
    state = [
        _state(0, 10.0, []),
        _state(1, 7.0, []),
        _state(2, 3.0, []),
        _state(3, 1.0, []),
    ]

    rewards, ranks = player_outcome_rewards(state, n_players=4)

    assert ranks == [1, 2, 3, 4]
    assert rewards == [1.0, -1.0, -1.0, -1.0]


def test_player_outcome_rewards_preserves_distinct_env_rewards_for_2p():
    state = [_state(0, 3.0, []), _state(1, 1.0, [])]
    rewards, ranks = player_outcome_rewards(state, n_players=2)
    assert rewards == [3.0, 1.0]
    assert ranks == [1, 2]


def test_player0_won_requires_strict_first_place_not_tie():
    assert player0_won([1.0, 2.0]) is True
    assert player0_won([1.5, 1.5]) is False
    assert player0_won([1.0, 2.0, 3.0, 4.0]) is True
    assert player0_won([1.5, 1.5, 3.5, 3.5]) is False


def test_opponent_pool_keeps_snapshot_clones_and_caps_size():
    policy = LinearActorCritic(feature_dim=3, seed=1)
    pool = OpponentPool(capacity=2, seed=3)

    policy.actor_w[:] = 1.0
    pool.add_snapshot(policy)
    policy.actor_w[:] = 2.0
    pool.add_snapshot(policy)
    policy.actor_w[:] = 3.0
    pool.add_snapshot(policy)

    assert len(pool) == 2
    sampled = [pool.sample() for _ in range(8)]
    assert all(s is not None for s in sampled)
    assert {float(s.actor_w[0]) for s in sampled}.issubset({2.0, 3.0})
    policy.actor_w[:] = 99.0
    assert {float(s.actor_w[0]) for s in sampled}.issubset({2.0, 3.0})
