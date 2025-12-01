"""Simulation configuration objects."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RCEConfig:
    """Container for simulation hyper-parameters."""

    n_agents: int = 10
    graph_type: str = "ring"
    T: int = 500
    gamma: float = 0.01
    lambda_reg: float = 0.0
    alpha_decay: float = 0.0
    eps: float = 1e-8
    seed: int = 42

    # RL params
    n_actions: int = 3
    lr_policy: float = 0.01

    # Extinction flag kept for parity with original design
    extinction_enabled: bool = False

    # Force full communication every N steps to improve synchronization
    force_comm_every: Optional[int] = None

    # Optional per-agent energy multipliers
    compute_energy_costs: Optional[list[float]] = None
    comm_energy_costs: Optional[list[float]] = None
    rest_energy_costs: Optional[list[float]] = None
