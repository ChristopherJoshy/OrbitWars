from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def policy_logits(features, actor_w):
    return jnp.asarray(features, dtype=jnp.float64) @ jnp.asarray(actor_w, dtype=jnp.float64)


@jax.jit
def policy_probs(features, actor_w):
    logits = policy_logits(features, actor_w)
    shifted = logits - jnp.max(logits)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp)


@jax.jit
def state_value(features, value_w):
    feats = jnp.asarray(features, dtype=jnp.float64)
    weights = jnp.asarray(value_w, dtype=jnp.float64)
    return jnp.mean(feats @ weights)


def logprob_and_value(features, actor_w, value_w, chosen_idx: int):
    probs = policy_probs(features, actor_w)
    logp = jnp.log(jnp.clip(probs[int(chosen_idx)], 1e-12, 1.0))
    return logp, state_value(features, value_w)


def select_argmax_action(features, actor_w) -> int:
    return int(jnp.argmax(policy_logits(features, actor_w)))


@jax.jit
def masked_policy_outputs(features, active, actor_w, value_w):
    feats = jnp.asarray(features, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=jnp.bool_)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)

    logits = feats @ actor_w
    masked_logits = jnp.where(active, logits, -1.0e30)
    shifted = masked_logits - jnp.max(masked_logits)
    exp = jnp.where(active, jnp.exp(shifted), 0.0)
    probs = exp / jnp.maximum(jnp.sum(exp), 1e-12)

    active_count = jnp.maximum(jnp.sum(active.astype(jnp.float64)), 1.0)
    values = feats @ value_w
    value = jnp.sum(jnp.where(active, values, 0.0)) / active_count
    entropy = -jnp.sum(jnp.where(active, probs * jnp.log(jnp.clip(probs, 1e-12, 1.0)), 0.0))
    return probs, value, entropy


@jax.jit
def select_masked_argmax_candidate(features, active, actor_w, value_w):
    probs, value, entropy = masked_policy_outputs(features, active, actor_w, value_w)
    idx = jnp.argmax(probs)
    logp = jnp.log(jnp.clip(probs[idx], 1e-12, 1.0))
    return idx, logp, value, entropy


@jax.jit
def sample_masked_candidate(key, features, active, actor_w, value_w):
    feats = jnp.asarray(features, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=jnp.bool_)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    key, sample_key = jax.random.split(key)

    logits = feats @ actor_w
    masked_logits = jnp.where(active, logits, -1.0e30)
    idx = jax.random.categorical(sample_key, masked_logits).astype(jnp.int64)
    probs, value, entropy = masked_policy_outputs(feats, active, actor_w, value_w)
    logp = jnp.log(jnp.clip(probs[idx], 1e-12, 1.0))
    return idx, logp, value, entropy, key


@jax.jit
def update_batch(
    candidate_features,
    chosen_indices,
    old_logps,
    returns,
    advantages,
    actor_w,
    value_w,
    lr=3e-3,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    feats = jnp.asarray(candidate_features, dtype=jnp.float64)
    chosen = jnp.asarray(chosen_indices, dtype=jnp.int64)
    old_logps = jnp.asarray(old_logps, dtype=jnp.float64)
    returns = jnp.asarray(returns, dtype=jnp.float64)
    advantages = jnp.asarray(advantages, dtype=jnp.float64)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)

    advantages = advantages / (jnp.mean(jnp.abs(advantages)) + 1e-8)
    pooled_features = jnp.mean(feats, axis=1)
    logits = jnp.einsum("baf,f->ba", feats, actor_w)
    shifted = logits - jnp.max(logits, axis=1, keepdims=True)
    exp = jnp.exp(shifted)
    probs = exp / jnp.sum(exp, axis=1, keepdims=True)

    batch_idx = jnp.arange(feats.shape[0])
    new_logps = jnp.log(jnp.clip(probs[batch_idx, chosen], 1e-12, 1.0))
    ratios = jnp.exp(jnp.clip(new_logps - old_logps, -20.0, 20.0))
    clipped = jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    use_unclipped = (ratios * advantages) <= (clipped * advantages)

    weights = jnp.where(use_unclipped, ratios * advantages, 0.0)
    one_hot = jax.nn.one_hot(chosen, feats.shape[1], dtype=jnp.float64)
    grad_logits = (one_hot - probs) * weights[:, None]
    grad_actor = jnp.einsum("baf,ba->f", feats, grad_logits) / jnp.maximum(feats.shape[0], 1)
    grad_actor = grad_actor + entropy_coef * (-actor_w)

    values = pooled_features @ value_w
    value_err = returns - values
    grad_value = jnp.mean(value_err[:, None] * pooled_features, axis=0) * value_coef

    gnorm = jnp.maximum(jnp.maximum(jnp.linalg.norm(grad_actor), jnp.linalg.norm(grad_value)), 1e-12)
    scale = jnp.minimum(1.0, 1.0 / gnorm)
    next_actor_w = actor_w + lr * grad_actor * scale
    next_value_w = value_w + lr * grad_value * scale
    metrics = {
        "loss_policy": -jnp.mean(jnp.minimum(ratios * advantages, clipped * advantages)),
        "loss_value": jnp.mean(value_err**2),
        "ratio_mean": jnp.mean(ratios),
    }
    return next_actor_w, next_value_w, metrics


@jax.jit
def masked_update_batch(
    candidate_features,
    active,
    chosen_indices,
    old_logps,
    returns,
    advantages,
    actor_w,
    value_w,
    lr=3e-3,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    feats = jnp.asarray(candidate_features, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=jnp.bool_)
    chosen = jnp.asarray(chosen_indices, dtype=jnp.int64)
    old_logps = jnp.asarray(old_logps, dtype=jnp.float64)
    returns = jnp.asarray(returns, dtype=jnp.float64)
    advantages = jnp.asarray(advantages, dtype=jnp.float64)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)

    advantages = advantages / (jnp.mean(jnp.abs(advantages)) + 1e-8)
    active_f = active.astype(jnp.float64)
    active_count = jnp.maximum(jnp.sum(active_f, axis=1), 1.0)
    pooled_features = jnp.sum(feats * active_f[:, :, None], axis=1) / active_count[:, None]

    logits = jnp.einsum("baf,f->ba", feats, actor_w)
    masked_logits = jnp.where(active, logits, -1.0e30)
    shifted = masked_logits - jnp.max(masked_logits, axis=1, keepdims=True)
    exp = jnp.where(active, jnp.exp(shifted), 0.0)
    probs = exp / jnp.maximum(jnp.sum(exp, axis=1, keepdims=True), 1e-12)

    batch_idx = jnp.arange(feats.shape[0])
    new_logps = jnp.log(jnp.clip(probs[batch_idx, chosen], 1e-12, 1.0))
    ratios = jnp.exp(jnp.clip(new_logps - old_logps, -20.0, 20.0))
    clipped = jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    use_unclipped = (ratios * advantages) <= (clipped * advantages)

    weights = jnp.where(use_unclipped, ratios * advantages, 0.0)
    one_hot = jax.nn.one_hot(chosen, feats.shape[1], dtype=jnp.float64)
    grad_logits = (one_hot - probs) * weights[:, None]
    grad_logits = jnp.where(active, grad_logits, 0.0)
    grad_actor = jnp.einsum("baf,ba->f", feats, grad_logits) / jnp.maximum(feats.shape[0], 1)
    grad_actor = grad_actor + entropy_coef * (-actor_w)

    values = pooled_features @ value_w
    value_err = returns - values
    grad_value = jnp.mean(value_err[:, None] * pooled_features, axis=0) * value_coef

    gnorm = jnp.maximum(jnp.maximum(jnp.linalg.norm(grad_actor), jnp.linalg.norm(grad_value)), 1e-12)
    scale = jnp.minimum(1.0, 1.0 / gnorm)
    next_actor_w = actor_w + lr * grad_actor * scale
    next_value_w = value_w + lr * grad_value * scale
    metrics = {
        "loss_policy": -jnp.mean(jnp.minimum(ratios * advantages, clipped * advantages)),
        "loss_value": jnp.mean(value_err**2),
        "ratio_mean": jnp.mean(ratios),
    }
    return next_actor_w, next_value_w, metrics


@jax.jit
def discounted_returns_and_advantages(
    rewards,
    dones,
    values,
    bootstrap_values,
    gamma=0.99,
):
    rewards = jnp.asarray(rewards, dtype=jnp.float64)
    dones = jnp.asarray(dones, dtype=jnp.bool_)
    values = jnp.asarray(values, dtype=jnp.float64)
    bootstrap_values = jnp.asarray(bootstrap_values, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)

    def body(next_return, step_inputs):
        reward_t, done_t = step_inputs
        continuation = jnp.where(done_t, 0.0, gamma)
        return_t = reward_t + continuation * next_return
        return return_t, return_t

    _last, reversed_returns = jax.lax.scan(body, bootstrap_values, (rewards[::-1], dones[::-1]))
    returns = reversed_returns[::-1]
    advantages = returns - values
    return returns, advantages


@jax.jit
def multiagent_masked_update_batch(
    candidate_features,
    active,
    chosen_indices,
    old_logps,
    returns,
    advantages,
    actor_w,
    value_w,
    lr=3e-3,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    feats = jnp.asarray(candidate_features, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=jnp.bool_)
    chosen = jnp.asarray(chosen_indices, dtype=jnp.int64)
    old_logps = jnp.asarray(old_logps, dtype=jnp.float64)
    returns = jnp.asarray(returns, dtype=jnp.float64)
    advantages = jnp.asarray(advantages, dtype=jnp.float64)
    actor_w = jnp.asarray(actor_w, dtype=jnp.float64)
    value_w = jnp.asarray(value_w, dtype=jnp.float64)

    def update_one(feats_p, active_p, chosen_p, old_logps_p, returns_p, advantages_p, actor_p, value_p):
        return masked_update_batch(
            feats_p,
            active_p,
            chosen_p,
            old_logps_p,
            returns_p,
            advantages_p,
            actor_p,
            value_p,
            lr=lr,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )

    feats_by_player = jnp.swapaxes(feats, 0, 1)
    active_by_player = jnp.swapaxes(active, 0, 1)
    chosen_by_player = jnp.swapaxes(chosen, 0, 1)
    old_logps_by_player = jnp.swapaxes(old_logps, 0, 1)
    returns_by_player = jnp.swapaxes(returns, 0, 1)
    advantages_by_player = jnp.swapaxes(advantages, 0, 1)
    return jax.vmap(update_one)(
        feats_by_player,
        active_by_player,
        chosen_by_player,
        old_logps_by_player,
        returns_by_player,
        advantages_by_player,
        actor_w,
        value_w,
    )


@jax.jit
def multiagent_ppo_update_from_rollout(
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
    gamma=0.99,
    lr=3e-3,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    returns, advantages = discounted_returns_and_advantages(
        rewards,
        dones,
        values,
        bootstrap_values,
        gamma=gamma,
    )
    next_actor_w, next_value_w, metrics = multiagent_masked_update_batch(
        candidate_features,
        active,
        chosen_indices,
        old_logps,
        returns,
        advantages,
        actor_w,
        value_w,
        lr=lr,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
    )
    return next_actor_w, next_value_w, returns, advantages, metrics


@jax.jit
def batched_multiagent_ppo_update_from_rollout(
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
    gamma=0.99,
    lr=3e-3,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    feats = jnp.asarray(candidate_features, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=jnp.bool_)
    chosen = jnp.asarray(chosen_indices, dtype=jnp.int64)
    old_logps = jnp.asarray(old_logps, dtype=jnp.float64)
    rewards = jnp.asarray(rewards, dtype=jnp.float64)
    dones = jnp.asarray(dones, dtype=jnp.bool_)
    values = jnp.asarray(values, dtype=jnp.float64)
    bootstrap_values = jnp.asarray(bootstrap_values, dtype=jnp.float64)

    returns, advantages = jax.vmap(
        lambda r, d, v, b: discounted_returns_and_advantages(r, d, v, b, gamma=gamma)
    )(rewards, dones, values, bootstrap_values)
    batch_size, num_steps = feats.shape[:2]
    flat_samples = int(batch_size * num_steps)
    next_actor_w, next_value_w, metrics = multiagent_masked_update_batch(
        feats.reshape((flat_samples,) + feats.shape[2:]),
        active.reshape((flat_samples,) + active.shape[2:]),
        chosen.reshape((flat_samples,) + chosen.shape[2:]),
        old_logps.reshape((flat_samples,) + old_logps.shape[2:]),
        returns.reshape((flat_samples,) + returns.shape[2:]),
        advantages.reshape((flat_samples,) + advantages.shape[2:]),
        actor_w,
        value_w,
        lr=lr,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
    )
    return next_actor_w, next_value_w, returns, advantages, metrics
