"""Main experiment loop orchestrating the RCE simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import RCEConfig
from .data import build_agent_datasets, compute_accuracy, load_dataset
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

    X_data, y_data = load_dataset(getattr(cfg, "dataset_name", "mnist"))
    agent_datasets = build_agent_datasets(X_data, y_data, n, rng)

    eval_rng = np.random.default_rng(cfg.seed + 2024)
    eval_size = min(200, len(X_data))
    eval_idx = eval_rng.choice(len(X_data), size=eval_size, replace=False)
    X_eval_fixed = X_data[eval_idx]
    y_eval_fixed = y_data[eval_idx]

    input_dim = X_data.shape[1]
    output_dim = int(np.unique(y_data).size)
    hidden_dim = 8 if input_dim <= 1024 else 32

    agent_models = []
    for _ in range(n):
        W1 = rng.normal(0, 0.2, size=(hidden_dim, input_dim))
        b1 = rng.normal(0, 0.2, size=(hidden_dim,))
        W2 = rng.normal(0, 0.2, size=(output_dim, hidden_dim))
        b2 = rng.normal(0, 0.2, size=(output_dim,))
        agent_models.append({"W1": W1, "b1": b1, "W2": W2, "b2": b2})

    agent_models = np.array(agent_models, dtype=object)

    baseline_acc = [compute_accuracy(model, X_eval_fixed, y_eval_fixed) for model in agent_models]
    prev_mean_acc = float(np.mean(baseline_acc))
    acc_trend = 0.0
    comm_trend = 0.0
    comm_trace = np.zeros(n, dtype=float)

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
    rest_used = np.zeros(T)
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

    if cfg.compute_energy_costs is not None:
        compute_costs = np.asarray(cfg.compute_energy_costs, dtype=float)
    else:
        compute_costs = np.full(n, 0.1)

    if cfg.comm_energy_costs is not None:
        comm_costs = np.asarray(cfg.comm_energy_costs, dtype=float)
    else:
        comm_costs = np.full(n, 0.01)

    if cfg.rest_energy_costs is not None:
        rest_costs = np.asarray(cfg.rest_energy_costs, dtype=float)
    else:
        rest_costs = np.full(n, 0.02)

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
        rest_used[t] = np.sum(actions == 2)

        # Per-action energy cost model
        per_agent_energy = (
            (actions == 0) * compute_costs
            + (actions == 1) * comm_costs
            + (actions == 2) * rest_costs
        )
        energy_per_agent += per_agent_energy
        energy_used[t] = per_agent_energy.sum()

        for i in range(n):
            agent_accuracy[t, i] = compute_accuracy(agent_models[i], X_eval_fixed, y_eval_fixed)

        mean_accuracy[t] = agent_accuracy[t].mean()

        acc_delta = max(0.0, mean_accuracy[t] - prev_mean_acc)
        prev_mean_acc = mean_accuracy[t]
        acc_trend = 0.4 * acc_trend + 0.6 * acc_delta

        horizon = max(1, T)
        phase = max(0.0, 1.0 - (t / horizon))
        fast_decay = np.exp(-t / max(1.0, 0.2 * horizon))
        bonus_scale = 0.5 * (phase + fast_decay)
        bonus = bonus_scale * acc_trend

        comm_rate = comm_used[t] / max(1.0, float(n))
        comm_trend = 0.9 * comm_trend + 0.1 * comm_rate
        late_start = 0.75 * horizon
        late_frac = max(0.0, (t - late_start) / max(1.0, horizon - late_start))
        late_penalty = 0.02 * comm_trend * late_frac

        trace_decay = 0.6
        comm_trace = trace_decay * comm_trace + (actions == 1).astype(float)

        if bonus > 0:
            trace = comm_trace.copy()
            total_trace = trace.sum()
            if total_trace > 0:
                rewards += bonus * (trace / total_trace)

            if late_penalty > 0:
                comm_mask = actions == 1
                rewards[comm_mask] -= late_penalty

            compute_mask = actions == 0
            rewards[compute_mask] -= 0.1 * bonus

        policy_params = update_policies(
            policy_params,
            states,
            actions,
            rewards,
            cfg,
        )

        W_history[t + 1] = W.copy()

        credits_history[t + 1] = c
        total_credit[t + 1] = c.sum()
        mean_credit[t + 1] = c.mean()
        var_credit[t + 1] = c.var()
        utility_history[t + 1] = u
        rewards_history[t] = rewards

    if cfg.policy_save_path:
        save_path = Path(cfg.policy_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, policy_params)
        print(f"[info] Saved policy parameters to {save_path}")

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
        "rest_used": rest_used,
        "energy_used": energy_used,
        "energy_per_agent": energy_per_agent,
        "final_accuracy": mean_accuracy[-1],
        "total_compute": compute_actions.sum(),
        "total_comm": comm_actions.sum(),
        "total_rest": rest_actions.sum(),
        "total_energy": energy_used.sum(),
        "acc_per_energy": mean_accuracy[-1] / (energy_used.sum() + 1e-9),
        "acc_per_comm": mean_accuracy[-1] / (comm_actions.sum() + 1e-9),
        "acc_per_rest": mean_accuracy[-1] / (rest_actions.sum() + 1e-9),
        "agent_models": _clone_models(agent_models),
    }
