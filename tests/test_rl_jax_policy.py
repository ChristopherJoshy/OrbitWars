import numpy as np
import pytest

jax = pytest.importorskip("jax")


def _numpy_probs(features, actor_w):
    logits = np.asarray(features, dtype=np.float64) @ np.asarray(actor_w, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def test_jax_policy_probs_match_numpy_linear_actor_critic_softmax():
    from src.ow.rl import jax_policy

    features = np.array([[0.2, 10.0, 1.0], [1.1, 4.0, 1.0], [-0.4, 7.0, 1.0]], dtype=np.float64)
    actor_w = np.array([0.3, -0.02, 0.1], dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(jax_policy.policy_probs(features, actor_w)),
        _numpy_probs(features, actor_w),
        rtol=1e-9,
        atol=1e-9,
    )


def test_jax_logprob_and_value_match_numpy_policy_path():
    from src.ow.rl import jax_policy

    features = np.array([[0.2, 10.0, 1.0], [1.1, 4.0, 1.0], [-0.4, 7.0, 1.0]], dtype=np.float64)
    actor_w = np.array([0.3, -0.02, 0.1], dtype=np.float64)
    value_w = np.array([-0.1, 0.05, 0.2], dtype=np.float64)
    chosen_idx = 1

    probs = _numpy_probs(features, actor_w)
    expected_logp = np.log(np.clip(probs[chosen_idx], 1e-12, 1.0))
    expected_value = np.mean(features @ value_w)

    logp, value = jax_policy.logprob_and_value(features, actor_w, value_w, chosen_idx)

    assert float(logp) == pytest.approx(float(expected_logp), rel=1e-9, abs=1e-9)
    assert float(value) == pytest.approx(float(expected_value), rel=1e-9, abs=1e-9)


def test_jax_select_argmax_action_is_deterministic_for_kaggle_eval():
    from src.ow.rl import jax_policy

    features = np.array([[0.0, 0.0, 1.0], [1.0, 5.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float64)
    actor_w = np.array([1.0, -0.1, 0.0], dtype=np.float64)

    assert int(jax_policy.select_argmax_action(features, actor_w)) == 2


def test_jax_masked_argmax_candidate_ignores_inactive_padding_and_matches_numpy_path():
    from src.ow.rl import jax_policy

    features = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [100.0, 0.0, 1.0],  # inactive padding would win if not masked
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    active = np.array([True, True, False, True])
    actor_w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    value_w = np.array([0.1, 0.0, 0.2], dtype=np.float64)

    active_features = features[active]
    active_logits = active_features @ actor_w
    active_probs = np.exp(active_logits - active_logits.max())
    active_probs /= active_probs.sum()
    expected_active_offset = int(np.argmax(active_probs))
    expected_idx = int(np.flatnonzero(active)[expected_active_offset])
    expected_logp = float(np.log(np.clip(active_probs[expected_active_offset], 1e-12, 1.0)))
    expected_value = float(np.mean(active_features @ value_w))
    expected_entropy = float(-np.sum(active_probs * np.log(np.clip(active_probs, 1e-12, 1.0))))

    idx, logp, value, entropy = jax_policy.select_masked_argmax_candidate(features, active, actor_w, value_w)

    assert int(idx) == expected_idx
    assert float(logp) == pytest.approx(expected_logp, rel=1e-9, abs=1e-9)
    assert float(value) == pytest.approx(expected_value, rel=1e-9, abs=1e-9)
    assert float(entropy) == pytest.approx(expected_entropy, rel=1e-9, abs=1e-9)


def test_jax_sample_masked_candidate_threads_key_and_never_selects_inactive_padding():
    from src.ow.rl import jax_policy

    features = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [100.0, 0.0, 1.0],  # inactive padding would dominate if sampled unmasked
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    active = np.array([True, True, False, True])
    actor_w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    value_w = np.array([0.1, 0.0, 0.2], dtype=np.float64)
    key = jax.random.PRNGKey(7)

    idx, logp, value, entropy, next_key = jax_policy.sample_masked_candidate(key, features, active, actor_w, value_w)
    idx2, logp2, value2, entropy2, next_key2 = jax_policy.sample_masked_candidate(key, features, active, actor_w, value_w)
    probs, expected_value, expected_entropy = jax_policy.masked_policy_outputs(features, active, actor_w, value_w)

    assert bool(active[int(idx)]) is True
    assert int(idx) != 2
    assert int(idx2) == int(idx)
    np.testing.assert_array_equal(np.asarray(next_key2), np.asarray(next_key))
    assert float(logp) == pytest.approx(float(np.log(np.clip(np.asarray(probs)[int(idx)], 1e-12, 1.0))), rel=1e-9, abs=1e-9)
    assert float(logp2) == pytest.approx(float(logp), rel=1e-12, abs=1e-12)
    assert float(value) == pytest.approx(float(expected_value), rel=1e-9, abs=1e-9)
    assert float(value2) == pytest.approx(float(expected_value), rel=1e-9, abs=1e-9)
    assert float(entropy) == pytest.approx(float(expected_entropy), rel=1e-9, abs=1e-9)
    assert float(entropy2) == pytest.approx(float(expected_entropy), rel=1e-9, abs=1e-9)
    assert not np.array_equal(np.asarray(next_key), np.asarray(key))


def test_jax_update_batch_matches_numpy_linear_actor_critic_for_equal_action_counts():
    from src.ow.rl import jax_policy
    from src.ow.rl.policy import LinearActorCritic

    candidate_features = np.array(
        [
            [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [1.0, -0.1, 0.3]],
            [[0.2, 1.0, 0.4], [1.0, 0.1, -0.2], [0.0, 0.3, 0.7]],
        ],
        dtype=np.float64,
    )
    chosen_indices = np.array([2, 0], dtype=np.int64)
    old_logps = np.array([-1.2, -0.9], dtype=np.float64)
    returns = np.array([1.5, -0.25], dtype=np.float64)
    advantages = np.array([0.75, -0.5], dtype=np.float64)
    actor_w = np.array([0.05, -0.02, 0.03], dtype=np.float64)
    value_w = np.array([-0.01, 0.04, 0.02], dtype=np.float64)

    np_policy = LinearActorCritic(feature_dim=3, seed=0)
    np_policy.actor_w = actor_w.copy()
    np_policy.value_w = value_w.copy()
    expected_metrics = np_policy.update_batch(
        {
            "candidate_features": [candidate_features[0], candidate_features[1]],
            "chosen_indices": chosen_indices,
            "old_logps": old_logps,
            "returns": returns,
            "advantages": advantages,
        },
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    next_actor_w, next_value_w, metrics = jax_policy.update_batch(
        candidate_features,
        chosen_indices,
        old_logps,
        returns,
        advantages,
        actor_w,
        value_w,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    np.testing.assert_allclose(np.asarray(next_actor_w), np_policy.actor_w, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_value_w), np_policy.value_w, rtol=1e-9, atol=1e-9)
    assert float(metrics["loss_policy"]) == pytest.approx(expected_metrics["loss_policy"], rel=1e-9, abs=1e-9)
    assert float(metrics["loss_value"]) == pytest.approx(expected_metrics["loss_value"], rel=1e-9, abs=1e-9)
    assert float(metrics["ratio_mean"]) == pytest.approx(expected_metrics["ratio_mean"], rel=1e-9, abs=1e-9)


def test_jax_masked_update_batch_matches_numpy_for_padded_candidate_tensors():
    from src.ow.rl import jax_policy
    from src.ow.rl.policy import LinearActorCritic

    candidate_features = np.array(
        [
            [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [99.0, 99.0, 99.0], [1.0, -0.1, 0.3]],
            [[0.2, 1.0, 0.4], [99.0, 99.0, 99.0], [1.0, 0.1, -0.2], [0.0, 0.3, 0.7]],
        ],
        dtype=np.float64,
    )
    active = np.array([[True, True, False, True], [True, False, True, True]])
    chosen_fixed_indices = np.array([3, 2], dtype=np.int64)
    chosen_filtered_indices = np.array([2, 1], dtype=np.int64)
    old_logps = np.array([-1.2, -0.9], dtype=np.float64)
    returns = np.array([1.5, -0.25], dtype=np.float64)
    advantages = np.array([0.75, -0.5], dtype=np.float64)
    actor_w = np.array([0.05, -0.02, 0.03], dtype=np.float64)
    value_w = np.array([-0.01, 0.04, 0.02], dtype=np.float64)

    np_policy = LinearActorCritic(feature_dim=3, seed=0)
    np_policy.actor_w = actor_w.copy()
    np_policy.value_w = value_w.copy()
    expected_metrics = np_policy.update_batch(
        {
            "candidate_features": [candidate_features[0, active[0]], candidate_features[1, active[1]]],
            "chosen_indices": chosen_filtered_indices,
            "old_logps": old_logps,
            "returns": returns,
            "advantages": advantages,
        },
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    next_actor_w, next_value_w, metrics = jax_policy.masked_update_batch(
        candidate_features,
        active,
        chosen_fixed_indices,
        old_logps,
        returns,
        advantages,
        actor_w,
        value_w,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    np.testing.assert_allclose(np.asarray(next_actor_w), np_policy.actor_w, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_value_w), np_policy.value_w, rtol=1e-9, atol=1e-9)
    assert float(metrics["loss_policy"]) == pytest.approx(expected_metrics["loss_policy"], rel=1e-9, abs=1e-9)
    assert float(metrics["loss_value"]) == pytest.approx(expected_metrics["loss_value"], rel=1e-9, abs=1e-9)
    assert float(metrics["ratio_mean"]) == pytest.approx(expected_metrics["ratio_mean"], rel=1e-9, abs=1e-9)


def test_jax_discounted_returns_and_advantages_reset_on_done_and_bootstrap_tail():
    from src.ow.rl import jax_policy

    rewards = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [0.5, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    dones = np.array([False, True, False, False])
    values = np.array(
        [
            [0.1, -0.2],
            [0.4, -0.3],
            [0.25, 0.75],
            [0.0, 1.25],
        ],
        dtype=np.float64,
    )
    bootstrap_values = np.array([0.25, -0.5], dtype=np.float64)

    expected_returns = np.zeros_like(rewards)
    running = bootstrap_values.copy()
    for t in range(len(rewards) - 1, -1, -1):
        if dones[t]:
            running = rewards[t]
        else:
            running = rewards[t] + 0.9 * running
        expected_returns[t] = running
    expected_advantages = expected_returns - values

    returns, advantages = jax_policy.discounted_returns_and_advantages(
        rewards,
        dones,
        values,
        bootstrap_values,
        gamma=0.9,
    )

    np.testing.assert_allclose(np.asarray(returns), expected_returns, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(advantages), expected_advantages, rtol=1e-9, atol=1e-9)


def test_jax_multiagent_masked_update_batch_matches_per_player_updates():
    from src.ow.rl import jax_policy

    candidate_features = np.array(
        [
            [
                [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [99.0, 99.0, 99.0], [1.0, -0.1, 0.3]],
                [[0.2, 1.0, 0.4], [1.0, 0.1, -0.2], [0.0, 0.3, 0.7], [99.0, 99.0, 99.0]],
            ],
            [
                [[0.4, -0.2, 1.0], [99.0, 99.0, 99.0], [0.5, 0.0, 0.1], [1.2, 0.3, -0.4]],
                [[99.0, 99.0, 99.0], [0.7, -0.1, 0.2], [0.6, 0.4, 0.0], [0.1, 0.2, 0.8]],
            ],
        ],
        dtype=np.float64,
    )
    active = np.array(
        [
            [[True, True, False, True], [True, True, True, False]],
            [[True, False, True, True], [False, True, True, True]],
        ]
    )
    chosen_indices = np.array([[3, 1], [2, 3]], dtype=np.int64)
    old_logps = np.array([[-1.2, -0.9], [-0.7, -1.1]], dtype=np.float64)
    returns = np.array([[1.5, -0.25], [0.75, 1.0]], dtype=np.float64)
    advantages = np.array([[0.75, -0.5], [0.2, 0.4]], dtype=np.float64)
    actor_w = np.array([[0.05, -0.02, 0.03], [-0.03, 0.04, 0.01]], dtype=np.float64)
    value_w = np.array([[-0.01, 0.04, 0.02], [0.02, -0.03, 0.05]], dtype=np.float64)

    expected_actor = []
    expected_value = []
    expected_metrics = {"loss_policy": [], "loss_value": [], "ratio_mean": []}
    for player in range(2):
        actor_next, value_next, metrics = jax_policy.masked_update_batch(
            candidate_features[:, player],
            active[:, player],
            chosen_indices[:, player],
            old_logps[:, player],
            returns[:, player],
            advantages[:, player],
            actor_w[player],
            value_w[player],
            lr=0.01,
            clip_epsilon=0.2,
            entropy_coef=0.003,
            value_coef=0.5,
        )
        expected_actor.append(np.asarray(actor_next))
        expected_value.append(np.asarray(value_next))
        for key in expected_metrics:
            expected_metrics[key].append(float(metrics[key]))

    next_actor_w, next_value_w, metrics = jax_policy.multiagent_masked_update_batch(
        candidate_features,
        active,
        chosen_indices,
        old_logps,
        returns,
        advantages,
        actor_w,
        value_w,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    np.testing.assert_allclose(np.asarray(next_actor_w), np.stack(expected_actor), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_value_w), np.stack(expected_value), rtol=1e-9, atol=1e-9)
    for key, expected in expected_metrics.items():
        np.testing.assert_allclose(np.asarray(metrics[key]), np.asarray(expected), rtol=1e-9, atol=1e-9)


def test_jax_multiagent_ppo_update_from_rollout_matches_explicit_returns_then_update():
    from src.ow.rl import jax_policy

    candidate_features = np.array(
        [
            [
                [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [99.0, 99.0, 99.0], [1.0, -0.1, 0.3]],
                [[0.2, 1.0, 0.4], [1.0, 0.1, -0.2], [0.0, 0.3, 0.7], [99.0, 99.0, 99.0]],
            ],
            [
                [[0.4, -0.2, 1.0], [99.0, 99.0, 99.0], [0.5, 0.0, 0.1], [1.2, 0.3, -0.4]],
                [[99.0, 99.0, 99.0], [0.7, -0.1, 0.2], [0.6, 0.4, 0.0], [0.1, 0.2, 0.8]],
            ],
        ],
        dtype=np.float64,
    )
    active = np.array(
        [
            [[True, True, False, True], [True, True, True, False]],
            [[True, False, True, True], [False, True, True, True]],
        ]
    )
    chosen_indices = np.array([[3, 1], [2, 3]], dtype=np.int64)
    old_logps = np.array([[-1.2, -0.9], [-0.7, -1.1]], dtype=np.float64)
    rewards = np.array([[0.5, -1.0], [1.25, 0.75]], dtype=np.float64)
    dones = np.array([False, True])
    values = np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float64)
    bootstrap_values = np.array([0.1, -0.2], dtype=np.float64)
    actor_w = np.array([[0.05, -0.02, 0.03], [-0.03, 0.04, 0.01]], dtype=np.float64)
    value_w = np.array([[-0.01, 0.04, 0.02], [0.02, -0.03, 0.05]], dtype=np.float64)

    expected_returns, expected_advantages = jax_policy.discounted_returns_and_advantages(
        rewards,
        dones,
        values,
        bootstrap_values,
        gamma=0.9,
    )
    expected_actor, expected_value, expected_metrics = jax_policy.multiagent_masked_update_batch(
        candidate_features,
        active,
        chosen_indices,
        old_logps,
        expected_returns,
        expected_advantages,
        actor_w,
        value_w,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    next_actor_w, next_value_w, returns, advantages, metrics = jax_policy.multiagent_ppo_update_from_rollout(
        candidate_features,
        active,
        chosen_indices,
        old_logps,
        rewards,
        dones,
        values,
        bootstrap_values,
        actor_w,
        value_w,
        gamma=0.9,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    np.testing.assert_allclose(np.asarray(returns), np.asarray(expected_returns), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(advantages), np.asarray(expected_advantages), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_actor_w), np.asarray(expected_actor), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_value_w), np.asarray(expected_value), rtol=1e-9, atol=1e-9)
    for key in expected_metrics:
        np.testing.assert_allclose(np.asarray(metrics[key]), np.asarray(expected_metrics[key]), rtol=1e-9, atol=1e-9)


def test_jax_batched_multiagent_ppo_update_from_rollout_flattens_env_and_time_axes():
    from src.ow.rl import jax_policy

    candidate_features = np.array(
        [
            [
                [
                    [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [99.0, 99.0, 99.0], [1.0, -0.1, 0.3]],
                    [[0.2, 1.0, 0.4], [1.0, 0.1, -0.2], [0.0, 0.3, 0.7], [99.0, 99.0, 99.0]],
                ],
                [
                    [[0.4, -0.2, 1.0], [99.0, 99.0, 99.0], [0.5, 0.0, 0.1], [1.2, 0.3, -0.4]],
                    [[99.0, 99.0, 99.0], [0.7, -0.1, 0.2], [0.6, 0.4, 0.0], [0.1, 0.2, 0.8]],
                ],
            ],
            [
                [
                    [[0.9, 0.1, -0.2], [0.4, -0.3, 0.5], [0.2, 0.2, 0.2], [99.0, 99.0, 99.0]],
                    [[0.5, 0.6, -0.1], [99.0, 99.0, 99.0], [0.1, -0.4, 0.9], [0.3, 0.8, 0.0]],
                ],
                [
                    [[99.0, 99.0, 99.0], [0.6, 0.2, -0.3], [0.2, 0.9, 0.1], [0.1, 0.1, 0.7]],
                    [[0.8, -0.2, 0.2], [0.0, 0.5, 0.5], [99.0, 99.0, 99.0], [0.4, 0.3, 0.2]],
                ],
            ],
        ],
        dtype=np.float64,
    )
    active = np.array(
        [
            [
                [[True, True, False, True], [True, True, True, False]],
                [[True, False, True, True], [False, True, True, True]],
            ],
            [
                [[True, True, True, False], [True, False, True, True]],
                [[False, True, True, True], [True, True, False, True]],
            ],
        ]
    )
    chosen_indices = np.array([[[3, 1], [2, 3]], [[1, 2], [3, 0]]], dtype=np.int64)
    old_logps = np.array([[[-1.2, -0.9], [-0.7, -1.1]], [[-0.4, -1.3], [-0.8, -0.6]]], dtype=np.float64)
    rewards = np.array([[[0.5, -1.0], [1.25, 0.75]], [[-0.25, 0.4], [1.0, -0.5]]], dtype=np.float64)
    dones = np.array([[False, True], [False, False]])
    values = np.array([[[0.2, -0.1], [0.4, 0.3]], [[-0.2, 0.5], [0.1, -0.4]]], dtype=np.float64)
    bootstrap_values = np.array([[0.1, -0.2], [0.3, 0.0]], dtype=np.float64)
    actor_w = np.array([[0.05, -0.02, 0.03], [-0.03, 0.04, 0.01]], dtype=np.float64)
    value_w = np.array([[-0.01, 0.04, 0.02], [0.02, -0.03, 0.05]], dtype=np.float64)

    expected_returns = []
    expected_advantages = []
    for env_index in range(2):
        returns_i, advantages_i = jax_policy.discounted_returns_and_advantages(
            rewards[env_index],
            dones[env_index],
            values[env_index],
            bootstrap_values[env_index],
            gamma=0.9,
        )
        expected_returns.append(np.asarray(returns_i))
        expected_advantages.append(np.asarray(advantages_i))
    expected_returns = np.stack(expected_returns)
    expected_advantages = np.stack(expected_advantages)
    expected_actor, expected_value, expected_metrics = jax_policy.multiagent_masked_update_batch(
        candidate_features.reshape(4, 2, 4, 3),
        active.reshape(4, 2, 4),
        chosen_indices.reshape(4, 2),
        old_logps.reshape(4, 2),
        expected_returns.reshape(4, 2),
        expected_advantages.reshape(4, 2),
        actor_w,
        value_w,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    next_actor_w, next_value_w, returns, advantages, metrics = jax_policy.batched_multiagent_ppo_update_from_rollout(
        candidate_features,
        active,
        chosen_indices,
        old_logps,
        rewards,
        dones,
        values,
        bootstrap_values,
        actor_w,
        value_w,
        gamma=0.9,
        lr=0.01,
        clip_epsilon=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
    )

    np.testing.assert_allclose(np.asarray(returns), expected_returns, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(advantages), expected_advantages, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_actor_w), np.asarray(expected_actor), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(next_value_w), np.asarray(expected_value), rtol=1e-9, atol=1e-9)
    for key in expected_metrics:
        np.testing.assert_allclose(np.asarray(metrics[key]), np.asarray(expected_metrics[key]), rtol=1e-9, atol=1e-9)
