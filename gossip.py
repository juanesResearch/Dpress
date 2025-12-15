"""Gossip-style decentralized SGD baseline."""

from __future__ import annotations

import numpy as np

from rce.graph import build_graph

DEFAULT_INPUT_DIM = 784
DEFAULT_HIDDEN_DIM = 8
DEFAULT_OUTPUT_DIM = 10
BATCH_SIZE = 32
LR = 0.05

GRAPH_COMM_FACTOR = {
    "ring": 1.0,
    "smallworld": 1.2,
    "complete": 1.4,
}

GOSSIP_COMM_INTERVAL = {
    "ring": 5,
    "smallworld": 3,
    "complete": 1,
}


def _init_model(rng, input_dim, hidden_dim, output_dim):
    return {
        "W1": rng.normal(0, 0.2, size=(hidden_dim, input_dim)),
        "b1": rng.normal(0, 0.2, size=(hidden_dim,)),
        "W2": rng.normal(0, 0.2, size=(output_dim, hidden_dim)),
        "b2": rng.normal(0, 0.2, size=(output_dim,)),
    }


def _clone_model(model):
    return {k: v.copy() for k, v in model.items()}


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

    return loss, {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def _apply_grads(model, grads, lr):
    model["W1"] -= lr * grads["dW1"]
    model["b1"] -= lr * grads["db1"]
    model["W2"] -= lr * grads["dW2"]
    model["b2"] -= lr * grads["db2"]


def _sample_batch(X, y, rng):
    idx = rng.choice(len(X), size=BATCH_SIZE, replace=False)
    return X[idx], y[idx]


def _average_pair(model_i, model_j, mix=0.5):
    new_i = {}
    new_j = {}
    for key in model_i.keys():
        blended = mix * model_i[key] + (1 - mix) * model_j[key]
        new_i[key] = blended.copy()
        new_j[key] = blended.copy()
    return new_i, new_j


def _average_models(models):
    avg = {k: np.zeros_like(v) for k, v in models[0].items()}
    for model in models:
        for key in avg.keys():
            avg[key] += model[key]
    for key in avg.keys():
        avg[key] /= len(models)
    return avg


def _compute_accuracy(model, X, y):
    logits, _ = _forward(model, X)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def run_gossip_sgd_baseline(
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
    pairs_per_round=1,
    exclusive_comm_rounds=False,
):
    """Run a lightweight gossip SGD baseline with configurable pair exchanges."""

    rng = np.random.default_rng(seed)
    G = build_graph(n_agents, graph_type)
    edges = list(G.edges())
    if not edges:
        edges = [(i, (i + 1) % n_agents) for i in range(n_agents)]

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

    if compute_costs is None:
        compute_costs = np.full(n_agents, 0.1)
    else:
        compute_costs = np.asarray(compute_costs, dtype=float)

    if comm_costs is None:
        comm_costs = np.full(n_agents, 0.01)
    else:
        comm_costs = np.asarray(comm_costs, dtype=float)

    compute_actions = 0
    communication_events = 0.0
    compute_energy = 0.0
    comm_energy = 0.0
    comm_factor = GRAPH_COMM_FACTOR.get(graph_type, 1.0)
    accuracy_history = []
    comm_counts_history = []

    eval_idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]

    pairs_per_round = max(1, int(pairs_per_round))
    comm_interval = GOSSIP_COMM_INTERVAL.get(graph_type, 1)
    if exclusive_comm_rounds and comm_interval < 2:
        comm_interval = 2  # ensure compute and comm alternate instead of being skipped entirely

    for step in range(T):
        step_comm_actions = 0
        do_comm = (step + 1) % comm_interval == 0
        skip_compute = exclusive_comm_rounds and do_comm

        if not skip_compute:
            for i, (Xi, yi) in enumerate(clients):
                Xb, yb = _sample_batch(Xi, yi, rng)
                y_onehot = np.eye(output_dim)[yb.astype(int)]
                _, grads = _loss_and_grads(models[i], Xb, y_onehot)
                _apply_grads(models[i], grads, LR)
                compute_actions += 1
            compute_energy += compute_costs.sum()

        if do_comm:
            n_pairs = min(pairs_per_round, len(edges))
            selected = rng.choice(len(edges), size=n_pairs, replace=False)
            for idx in selected:
                a, b = edges[idx]
                new_a, new_b = _average_pair(models[a], models[b], mix=0.5)
                models[a] = new_a
                models[b] = new_b
                comm_energy += comm_factor * (comm_costs[a] + comm_costs[b])
                communication_events += 2
                step_comm_actions += 2

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
