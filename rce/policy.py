"""RL policy utilities for agent decision making."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import RCEConfig


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def init_policies(cfg: RCEConfig) -> tuple[np.ndarray, int]:
    """Initialize policy parameters for each agent."""
    n = cfg.n_agents
    state_dim = 3 + n  # [c_i, u_i, e_i, u_neighbors...]
    n_actions = cfg.n_actions

    if cfg.policy_init_path:
        init_path = Path(cfg.policy_init_path)
        if init_path.exists():
            try:
                loaded = np.load(init_path, allow_pickle=True)
                loaded = np.asarray(loaded, dtype=float)
                if loaded.shape == (n, n_actions, state_dim):
                    print(f"[info] Loaded policy parameters from {init_path}")
                    return loaded.copy(), state_dim
                else:
                    print(
                        f"[warn] Policy checkpoint shape {loaded.shape} does not match "
                        f"({n}, {n_actions}, {state_dim}); reinitializing."
                    )
            except OSError as exc:
                print(f"[warn] Failed to load policy checkpoint {init_path}: {exc}")
        else:
            print(f"[warn] Policy init path {init_path} not found; sampling new parameters.")

    rng = np.random.default_rng(cfg.seed + 1)
    policy_params = rng.normal(loc=0.0, scale=0.01, size=(n, n_actions, state_dim))
    return policy_params, state_dim


def build_states(c: np.ndarray, u: np.ndarray, e: np.ndarray, neighbors: dict[int, list[int]]) -> np.ndarray:
    """Build state vectors per agent."""
    n = len(c)
    states = np.zeros((n, 3 + n), dtype=float)
    for i in range(n):
        s = np.zeros(3 + n, dtype=float)
        s[0] = c[i]
        s[1] = u[i]
        s[2] = e[i]
        for j in neighbors[i]:
            s[3 + j] = u[j]
        states[i] = s
    return states


def select_actions(policy_params: np.ndarray, states: np.ndarray, cfg: RCEConfig, rng) -> tuple[np.ndarray, np.ndarray]:
    """Select actions for each agent based on current policies and states."""
    n = cfg.n_agents
    n_actions = cfg.n_actions

    actions = np.zeros(n, dtype=int)
    action_probs = np.zeros((n, n_actions), dtype=float)

    for i in range(n):
        logits = policy_params[i] @ states[i]
        probs = softmax(logits)
        action = rng.choice(n_actions, p=probs)
        actions[i] = action
        action_probs[i] = probs

    return actions, action_probs


def update_policies(
    policy_params: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    cfg: RCEConfig,
    update_mask: np.ndarray | None = None,
) -> np.ndarray:
    """REINFORCE update for each agent policy."""
    n = cfg.n_agents
    n_actions = cfg.n_actions
    lr = cfg.lr_policy

    for i in range(n):
        if update_mask is not None and not update_mask[i]:
            continue
        s = states[i]
        logits = policy_params[i] @ s
        probs = softmax(logits)
        a = actions[i]
        r = rewards[i]

        one_hot = np.zeros(n_actions, dtype=float)
        one_hot[a] = 1.0
        grad = (one_hot - probs)[:, None] * s[None, :]
        policy_params[i] += lr * r * grad

    return policy_params