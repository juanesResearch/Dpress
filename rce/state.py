"""Agent state initialization helpers."""

from __future__ import annotations

import numpy as np

from .config import RCEConfig


def initialize_states(cfg: RCEConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_agents

    c0 = rng.uniform(low=0.5, high=1.5, size=n)
    u0 = rng.uniform(low=0.0, high=1.0, size=n)
    e0 = rng.uniform(low=0.0, high=1.0, size=n)

    return c0, u0, e0
