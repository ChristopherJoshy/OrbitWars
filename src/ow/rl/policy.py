from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    advantages = np.asarray(advantages, dtype=np.float64)
    return advantages / (np.mean(np.abs(advantages)) + 1e-8)


class LinearActorCritic:
    def __init__(self, feature_dim: int, seed: int = 0) -> None:
        self.feature_dim = int(feature_dim)
        self.rng = np.random.default_rng(seed)
        self.actor_w = self.rng.normal(0.0, 0.01, size=(self.feature_dim,))
        self.value_w = self.rng.normal(0.0, 0.01, size=(self.feature_dim,))

    def score(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.actor_w

    def probs(self, features: np.ndarray) -> np.ndarray:
        logits = self.score(features)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def sample(self, features: np.ndarray) -> tuple[int, float, float, float]:
        p = self.probs(features)
        idx = int(self.rng.choice(len(p), p=p))
        logp = float(np.log(max(1e-12, p[idx])))
        feats = np.asarray(features, dtype=np.float64)
        value = float(np.mean(feats @ self.value_w))
        entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
        return idx, logp, value, entropy

    def argmax(self, features: np.ndarray) -> tuple[int, float, float, float]:
        p = self.probs(features)
        idx = int(np.argmax(p))
        logp = float(np.log(max(1e-12, p[idx])))
        feats = np.asarray(features, dtype=np.float64)
        value = float(np.mean(feats @ self.value_w))
        entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
        return idx, logp, value, entropy

    def logprob_and_value(self, candidate_features: np.ndarray, chosen_idx: int) -> tuple[float, float]:
        feats = np.asarray(candidate_features, dtype=np.float64)
        probs = self.probs(feats)
        logp = float(np.log(np.clip(probs[int(chosen_idx)], 1e-12, 1.0)))
        value = float(np.mean(feats @ self.value_w))
        return logp, value

    def update_batch(
        self,
        batch: dict,
        lr: float = 3e-3,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> dict:
        candidate_features = [np.asarray(f, dtype=np.float64) for f in batch["candidate_features"]]
        chosen_indices = np.asarray(batch["chosen_indices"], dtype=np.int64)
        old_logps = np.asarray(batch["old_logps"], dtype=np.float64)
        returns = np.asarray(batch["returns"], dtype=np.float64)
        advantages = np.asarray(batch["advantages"], dtype=np.float64)
        advantages = normalize_advantages(advantages)

        pooled_features = np.asarray([np.mean(f, axis=0) for f in candidate_features], dtype=np.float64)
        logits_list = [f @ self.actor_w for f in candidate_features]
        probs_list = []
        new_logps = np.zeros_like(old_logps)
        for i, logits in enumerate(logits_list):
            shifted = logits - np.max(logits)
            exp = np.exp(shifted)
            probs = exp / np.sum(exp)
            probs_list.append(probs)
            new_logps[i] = np.log(np.clip(probs[int(chosen_indices[i])], 1e-12, 1.0))

        ratios = np.exp(np.clip(new_logps - old_logps, -20.0, 20.0))
        clipped = np.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        use_unclipped = (ratios * advantages) <= (clipped * advantages)

        weights = np.where(use_unclipped, ratios * advantages, 0.0)
        grad_actor = np.zeros_like(self.actor_w)
        for i, feats in enumerate(candidate_features):
            probs = probs_list[i]
            one_hot = np.zeros_like(probs)
            one_hot[int(chosen_indices[i])] = 1.0
            grad_logits = (one_hot - probs) * weights[i]
            grad_actor += feats.T @ grad_logits
        grad_actor /= max(len(candidate_features), 1)
        grad_actor += entropy_coef * (-self.actor_w)

        values = pooled_features @ self.value_w
        value_err = returns - values
        grad_value = (value_err[:, None] * pooled_features).mean(axis=0) * value_coef

        gnorm = max(np.linalg.norm(grad_actor), np.linalg.norm(grad_value), 1e-12)
        scale = min(1.0, 1.0 / gnorm)
        self.actor_w += lr * grad_actor * scale
        self.value_w += lr * grad_value * scale
        return {
            "loss_policy": float(-np.mean(np.minimum(ratios * advantages, clipped * advantages))),
            "loss_value": float(np.mean(value_err**2)),
            "ratio_mean": float(np.mean(ratios)),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, actor_w=self.actor_w, value_w=self.value_w, feature_dim=self.feature_dim)

    def clone(self, seed: int | None = None) -> "LinearActorCritic":
        out = LinearActorCritic(feature_dim=self.feature_dim, seed=0 if seed is None else int(seed))
        out.actor_w = np.array(self.actor_w, copy=True)
        out.value_w = np.array(self.value_w, copy=True)
        return out

    @staticmethod
    def _resize_weights(weights: np.ndarray, feature_dim: int) -> np.ndarray:
        out = np.zeros((int(feature_dim),), dtype=np.float64)
        source = np.asarray(weights, dtype=np.float64)
        n = min(len(source), int(feature_dim))
        out[:n] = source[:n]
        return out

    @classmethod
    def load(cls, path: Path, feature_dim: int | None = None) -> "LinearActorCritic":
        data = np.load(path)
        target_dim = int(data["feature_dim"]) if feature_dim is None else int(feature_dim)
        out = cls(feature_dim=target_dim)
        out.actor_w = cls._resize_weights(data["actor_w"], target_dim)
        out.value_w = cls._resize_weights(data["value_w"], target_dim)
        return out
