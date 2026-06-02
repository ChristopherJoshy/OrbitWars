import numpy as np

from src.ow.rl.policy import LinearActorCritic, normalize_advantages


def test_policy_sample_returns_valid_index_and_logprob():
    policy = LinearActorCritic(feature_dim=3, seed=7)
    features = np.array([[0.0, 0.0, 1.0], [1.0, 0.5, 1.0]], dtype=np.float64)
    idx, logp, value, entropy = policy.sample(features)
    assert idx in (0, 1)
    assert np.isfinite(logp)
    assert np.isfinite(value)
    assert entropy > 0.0


def test_policy_argmax_returns_deterministic_highest_score_action():
    policy = LinearActorCritic(feature_dim=2, seed=7)
    policy.actor_w[:] = np.array([0.25, -1.0], dtype=np.float64)
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -2.0]], dtype=np.float64)

    idx, logp, value, entropy = policy.argmax(features)

    assert idx == 2
    assert np.isfinite(logp)
    assert np.isfinite(value)
    assert entropy > 0.0


def test_policy_checkpoint_roundtrip(tmp_path):
    policy = LinearActorCritic(feature_dim=2, seed=1)
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = LinearActorCritic.load(path)
    assert np.allclose(policy.actor_w, loaded.actor_w)
    assert np.allclose(policy.value_w, loaded.value_w)


def test_policy_load_can_zero_pad_legacy_checkpoint_to_larger_feature_dim(tmp_path):
    policy = LinearActorCritic(feature_dim=2, seed=1)
    policy.actor_w[:] = np.array([1.0, 2.0], dtype=np.float64)
    policy.value_w[:] = np.array([3.0, 4.0], dtype=np.float64)
    path = tmp_path / "legacy_policy.npz"
    policy.save(path)

    loaded = LinearActorCritic.load(path, feature_dim=5)

    assert loaded.feature_dim == 5
    assert np.allclose(loaded.actor_w, [1.0, 2.0, 0.0, 0.0, 0.0])
    assert np.allclose(loaded.value_w, [3.0, 4.0, 0.0, 0.0, 0.0])


def test_policy_clone_can_seed_rng_for_reproducible_sample_streams():
    policy = LinearActorCritic(feature_dim=2, seed=1)

    a = policy.clone(seed=123)
    b = policy.clone(seed=123)
    c = policy.clone(seed=124)

    assert a.rng.random() == b.rng.random()
    assert a.rng.random() != c.rng.random()


def test_update_batch_keeps_ratio_near_one_with_matching_old_logps():
    policy = LinearActorCritic(feature_dim=3, seed=11)
    candidate_features = [
        np.array([[0.1, 1.0, 1.0], [0.2, 0.5, 1.0], [0.3, 0.2, 1.0]], dtype=np.float64),
        np.array([[0.8, 1.2, 1.0], [0.7, 0.4, 1.0]], dtype=np.float64),
    ]
    chosen_indices = np.array([1, 0], dtype=np.int64)
    old_logps = np.array(
        [policy.logprob_and_value(candidate_features[i], int(chosen_indices[i]))[0] for i in range(2)],
        dtype=np.float64,
    )
    returns = np.array([0.1, -0.2], dtype=np.float64)
    advantages = np.array([0.3, -0.1], dtype=np.float64)

    before = policy.actor_w.copy()
    metrics = policy.update_batch(
        {
            "candidate_features": candidate_features,
            "chosen_indices": chosen_indices,
            "old_logps": old_logps,
            "returns": returns,
            "advantages": advantages,
        },
        lr=5e-3,
        entropy_coef=0.0,
    )
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-6
    assert np.linalg.norm(policy.actor_w - before) < 1e-2


def test_advantage_normalization_preserves_positive_winning_batch_signs():
    normalized = normalize_advantages(np.array([0.25, 1.0, 2.0], dtype=np.float64))

    assert np.all(normalized > 0.0)
    assert np.isclose(np.mean(np.abs(normalized)), 1.0)


def test_update_batch_zeroes_actor_gradient_when_ppo_clip_selected():
    policy = LinearActorCritic(feature_dim=2, seed=13)
    policy.actor_w[:] = 0.0
    features = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
    ]
    before = policy.actor_w.copy()

    policy.update_batch(
        {
            "candidate_features": features,
            "chosen_indices": np.array([0, 0], dtype=np.int64),
            "old_logps": np.array([-10.0, 10.0], dtype=np.float64),
            "returns": np.array([1.0, -1.0], dtype=np.float64),
            "advantages": np.array([1.0, -1.0], dtype=np.float64),
        },
        lr=1.0,
        clip_epsilon=0.2,
        entropy_coef=0.0,
        value_coef=0.0,
    )

    assert np.allclose(policy.actor_w, before)
