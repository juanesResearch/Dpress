"""Main experiment loop orchestrating the RCE simulation."""

from __future__ import annotations

import numpy as np

from .config import RCEConfig
from .data import build_agent_datasets, compute_accuracy, load_mnist_numpy
from .dynamics import step_env
from .graph import build_graph, build_trust_matrix
from .policy import build_states, init_policies, select_actions, update_policies
from .state import initialize_states


def _clone_models(agent_models):
    cloned = []
    for model in agent_models:
        cloned.append({k: v.copy() for k, v in model.items()})
    return cloned


def run_experiment(cfg: RCEConfig):
    G = build_graph(cfg.n_agents, cfg.graph_type)
    W = build_trust_matrix(G)
    neighbors = {i: list(G.neighbors(i)) for i in range(cfg.n_agents)}

    rng = np.random.default_rng(cfg.seed)
    c, u, e = initialize_states(cfg)
    cfg.initial_total_credit = float(c.sum())

    n = cfg.n_agents
    T = cfg.T

    X_mnist, y_mnist = load_mnist_numpy()
    agent_datasets = build_agent_datasets(X_mnist, y_mnist, n, rng)

    input_dim = 784
    hidden_dim = 8
    output_dim = 10

    agent_models = []
    for _ in range(n):
        W1 = rng.normal(0, 0.2, size=(hidden_dim, input_dim))
        b1 = rng.normal(0, 0.2, size=(hidden_dim,))
        W2 = rng.normal(0, 0.2, size=(output_dim, hidden_dim))
        b2 = rng.normal(0, 0.2, size=(output_dim,))
        agent_models.append({"W1": W1, "b1": b1, "W2": W2, "b2": b2})

    agent_models = np.array(agent_models, dtype=object)

    policy_params, _ = init_policies(cfg)

    credits_history = np.zeros((T + 1, n))
    total_credit = np.zeros(T + 1)
    mean_credit = np.zeros(T + 1)
    var_credit = np.zeros(T + 1)
    rewards_history = np.zeros((T, n))

    agent_accuracy = np.zeros((T, n))
    mean_accuracy = np.zeros(T)

    compute_actions = np.zeros(n)
    comm_actions = np.zeros(n)
    rest_actions = np.zeros(n)

    compute_used = np.zeros(T)
    comm_used = np.zeros(T)
    energy_used = np.zeros(T)
    energy_per_agent = np.zeros(n)

    W_history = np.zeros((T + 1, n, n))
    W_history[0] = W.copy()

    credits_history[0] = c

    utility_history = np.zeros((T + 1, n))
    utility_history[0] = u

    total_credit[0] = c.sum()
    mean_credit[0] = c.mean()
    var_credit[0] = c.var()

    for t in range(T):
        states = build_states(c, u, e, neighbors)
        actions, action_probs = select_actions(policy_params, states, cfg, rng)

        forced_comm = False
        if cfg.force_comm_every:
            interval = int(cfg.force_comm_every)
            if interval > 0 and (t + 1) % interval == 0:
                forced_comm = True
                actions[:] = 1

        for i in range(n):
            if actions[i] == 0:
                compute_actions[i] += 1
            elif actions[i] == 1:
                comm_actions[i] += 1
            else:
                rest_actions[i] += 1

        c, u, e, inflow, outflow, delta, rewards, W, agent_models = step_env(
            c,
            u,
            e,
            W,
            neighbors,
            actions,
            agent_models,
            agent_datasets,
            t,
            cfg,
            rng,
        )

        compute_used[t] = np.sum(actions == 0)
        comm_used[t] = np.sum(actions == 1)

        # Per-action energy cost model
        compute_energy_cost = 0.1
        comm_energy_cost = 0.01
        rest_energy_cost = 0.02

        step_energy = (
            compute_energy_cost * np.sum(actions == 0)
            + comm_energy_cost * np.sum(actions == 1)
            + rest_energy_cost * np.sum(actions == 2)
        )
        energy_used[t] = step_energy

        energy_per_agent += (
            (actions == 0) * compute_energy_cost
            + (actions == 1) * comm_energy_cost
            + (actions == 2) * rest_energy_cost
        )

        if not forced_comm:
            policy_params = update_policies(policy_params, states, actions, rewards, cfg)

        W_history[t + 1] = W.copy()

        eval_idx = rng.choice(len(X_mnist), size=200, replace=False)
        X_eval = X_mnist[eval_idx]
        y_eval = y_mnist[eval_idx]

        for i in range(n):
            agent_accuracy[t, i] = compute_accuracy(agent_models[i], X_eval, y_eval)

        mean_accuracy[t] = agent_accuracy[t].mean()

        credits_history[t + 1] = c
        total_credit[t + 1] = c.sum()
        mean_credit[t + 1] = c.mean()
        var_credit[t + 1] = c.var()
        utility_history[t + 1] = u
        rewards_history[t] = rewards

    return {
        "cfg": cfg,
        "G": G,
        "W": W,
        "neighbors": neighbors,
        "credits_history": credits_history,
        "total_credit": total_credit,
        "mean_credit": mean_credit,
        "var_credit": var_credit,
        "rewards_history": rewards_history,
        "agent_accuracy": agent_accuracy,
        "mean_accuracy": mean_accuracy,
        "trust_history": W_history,
        "utility_history": utility_history,
        "policy_params": policy_params,
        "compute_actions": compute_actions,
        "comm_actions": comm_actions,
        "rest_actions": rest_actions,
        "compute_used": compute_used,
        "comm_used": comm_used,
        "energy_used": energy_used,
        "energy_per_agent": energy_per_agent,
        "final_accuracy": mean_accuracy[-1],
        "total_compute": compute_actions.sum(),
        "total_comm": comm_actions.sum(),
        "total_energy": energy_used.sum(),
        "acc_per_energy": mean_accuracy[-1] / (energy_used.sum() + 1e-9),
        "acc_per_comm": mean_accuracy[-1] / (comm_actions.sum() + 1e-9),
        "agent_models": _clone_models(agent_models),
    }
