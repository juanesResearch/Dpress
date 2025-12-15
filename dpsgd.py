"""Self-contained D-PSGD style baseline to avoid cogdl dependency."""

from __future__ import annotations

import numpy as np
from rce.graph import build_graph

DEFAULT_INPUT_DIM = 784
DEFAULT_HIDDEN_DIM = 8
DEFAULT_OUTPUT_DIM = 10
BATCH_SIZE = 32
LR = 0.05
GRAPH_SYNC_INTERVAL = {
    "ring": 5,
    "smallworld": 3,
    "complete": 1,
}
GRAPH_COMM_FACTOR = {
    "ring": 1.0,
    "smallworld": 1.0,
    "complete": 1.0,
}


def _init_model(rng, input_dim, hidden_dim, output_dim):
    return {
        "W1": rng.normal(0, 0.2, size=(hidden_dim, input_dim)),
        "b1": rng.normal(0, 0.2, size=(hidden_dim,)),
        "W2": rng.normal(0, 0.2, size=(output_dim, hidden_dim)),
        "b2": rng.normal(0, 0.2, size=(output_dim,)),
    }


def _forward(model, X):
    H = X @ model["W1"].T + model["b1"]
    H_relu = np.maximum(H, 0)
    logits = H_relu @ model["W2"].T + model["b2"]
    return logits, H_relu


def _loss_and_grads(model, X, y_onehot):
    logits, H_relu = _forward(model, X)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

    dlogits = (probs - y_onehot) / X.shape[0]
    dW2 = dlogits.T @ H_relu
    db2 = dlogits.sum(axis=0)
    dH = dlogits @ model["W2"]
    dH[H_relu <= 0] = 0
    dW1 = dH.T @ X
    db1 = dH.sum(axis=0)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return loss, grads


def _apply_grads(model, grads, lr):
    model["W1"] -= lr * grads["dW1"]
    model["b1"] -= lr * grads["db1"]
    model["W2"] -= lr * grads["dW2"]
    model["b2"] -= lr * grads["db2"]


def _sample_batch(X, y, rng):
    idx = rng.choice(len(X), size=BATCH_SIZE, replace=False)
    return X[idx], y[idx]


def _average_models(models):
    avg = {}
    for key in models[0].keys():
        stacked = np.stack([m[key] for m in models], axis=0)
        avg[key] = stacked.mean(axis=0)
    return avg


def _clone_model(model):
    return {k: v.copy() for k, v in model.items()}


def _compute_accuracy(model, X, y):
    logits, _ = _forward(model, X)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def run_dpsgd_baseline(
    X,
    y,
    n_agents=10,
    T=10000,
    graph_type="ring",
    seed=0,
    compute_costs=None,
    comm_costs=None,
    agent_datasets=None,
    eval_stride=200,
    exclusive_comm_rounds=False,
):
    """Lightweight NumPy implementation of synchronous D-PSGD."""

    rng = np.random.default_rng(seed)

    if agent_datasets is not None:
        clients = agent_datasets
    else:
        data_splits = np.array_split(np.arange(len(X)), n_agents)
        clients = [(X[idx], y[idx]) for idx in data_splits]

    input_dim = X.shape[1] if X.ndim > 1 else DEFAULT_INPUT_DIM
    unique_labels = np.unique(y)
    output_dim = int(unique_labels.size) if unique_labels.size else DEFAULT_OUTPUT_DIM
    hidden_dim = DEFAULT_HIDDEN_DIM if input_dim <= 1024 else 32

    models = [_init_model(rng, input_dim, hidden_dim, output_dim) for _ in range(n_agents)]

    G = build_graph(n_agents, graph_type)
    if G.number_of_edges() == 0:
        for i in range(n_agents):
            G.add_edge(i, (i + 1) % n_agents)

    neighbors = []
    for node in range(n_agents):
        neigh = list(G.neighbors(node))
        if not neigh:
            neigh = [node]
        neighbors.append(neigh)

    compute_actions = 0
    communication_events = 0
    compute_energy = 0.0
    comm_energy = 0.0
    accuracy_history = []
    comm_counts_history = []

    eval_idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]
    sync_interval = GRAPH_SYNC_INTERVAL.get(graph_type, GRAPH_SYNC_INTERVAL["ring"])
    if exclusive_comm_rounds and sync_interval < 2:
        sync_interval = 2  # keep at least one pure compute step between comm rounds
    comm_factor = GRAPH_COMM_FACTOR.get(graph_type, 1.0)

    if compute_costs is None:
        compute_costs = np.full(n_agents, 0.1)
    else:
        compute_costs = np.asarray(compute_costs, dtype=float)

    if comm_costs is None:
        comm_costs = np.full(n_agents, 0.01)
    else:
        comm_costs = np.asarray(comm_costs, dtype=float)

    for step in range(T):
        step_comm_actions = 0
        do_sync = (step + 1) % sync_interval == 0
        skip_compute = exclusive_comm_rounds and do_sync

        if not skip_compute:
            for i, (Xi, yi) in enumerate(clients):
                Xb, yb = _sample_batch(Xi, yi, rng)
                y_onehot = np.eye(output_dim)[yb.astype(int)]
                _, grads = _loss_and_grads(models[i], Xb, y_onehot)
                _apply_grads(models[i], grads, LR)
                compute_actions += 1
            compute_energy += compute_costs.sum()

        if do_sync:
            mixed_models = []
            event_comm_energy = 0.0
            event_comm_actions = 0
            for agent_idx in range(n_agents):
                partner_idxs = [agent_idx] + neighbors[agent_idx]
                partner_models = [models[idx] for idx in partner_idxs]
                avg_model = _average_models(partner_models)
                mixed_models.append(avg_model)

                deg = len(neighbors[agent_idx])
                agent_comm_cost = comm_factor * comm_costs[agent_idx] * deg
                event_comm_energy += agent_comm_cost
                event_comm_actions += deg

            models = [_clone_model(m) for m in mixed_models]
            comm_energy += event_comm_energy
            communication_events += event_comm_actions
            step_comm_actions += event_comm_actions

        comm_counts_history.append(step_comm_actions)

        if (step + 1) % eval_stride == 0 or step == T - 1:
            avg_model = _average_models(models)
            acc = _compute_accuracy(avg_model, X_eval, y_eval)
            accuracy_history.append((step + 1, acc))

    global_model = _average_models(models)

    eval_idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    final_accuracy = _compute_accuracy(global_model, X[eval_idx], y[eval_idx])

    total_energy = compute_energy + comm_energy

    return {
        "final_accuracy": final_accuracy,
        "total_compute": float(compute_actions),
        "total_comm": float(communication_events),
        "total_energy": float(total_energy),
        "acc_per_energy": final_accuracy / (total_energy + 1e-9),
        "acc_per_comm": final_accuracy / (communication_events + 1e-9),
        "graph_type": graph_type,
        "accuracy_history": accuracy_history,
        "comm_counts_history": comm_counts_history,
    }
