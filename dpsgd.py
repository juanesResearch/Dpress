"""Self-contained D-PSGD style baseline to avoid cogdl dependency."""

from __future__ import annotations

import numpy as np

INPUT_DIM = 784
HIDDEN_DIM = 8
OUTPUT_DIM = 10
BATCH_SIZE = 32
LR = 0.05
GRAPH_SYNC_INTERVAL = {
    "ring": 5,
    "smallworld": 3,
    "complete": 1,
}
GRAPH_COMM_FACTOR = {
    "ring": 1.0,
    "smallworld": 1.5,
    "complete": 2.5,
}


def _init_model(rng):
    return {
        "W1": rng.normal(0, 0.2, size=(HIDDEN_DIM, INPUT_DIM)),
        "b1": rng.normal(0, 0.2, size=(HIDDEN_DIM,)),
        "W2": rng.normal(0, 0.2, size=(OUTPUT_DIM, HIDDEN_DIM)),
        "b2": rng.normal(0, 0.2, size=(OUTPUT_DIM,)),
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


def run_dpsgd_baseline(X, y, n_agents=10, T=10000, graph_type="ring", seed=0):
    """Lightweight NumPy implementation of synchronous D-PSGD."""

    rng = np.random.default_rng(seed)

    data_splits = np.array_split(np.arange(len(X)), n_agents)
    clients = [(X[idx], y[idx]) for idx in data_splits]

    models = [_init_model(rng) for _ in range(n_agents)]

    compute_cost = 0
    communication_cost = 0
    sync_interval = GRAPH_SYNC_INTERVAL.get(graph_type, GRAPH_SYNC_INTERVAL["ring"])
    comm_factor = GRAPH_COMM_FACTOR.get(graph_type, 1.0)

    for step in range(T):
        for i, (Xi, yi) in enumerate(clients):
            Xb, yb = _sample_batch(Xi, yi, rng)
            y_onehot = np.eye(OUTPUT_DIM)[yb]
            _, grads = _loss_and_grads(models[i], Xb, y_onehot)
            _apply_grads(models[i], grads, LR)
            compute_cost += 1

        if (step + 1) % sync_interval == 0:
            avg_model = _average_models(models)
            models = [_clone_model(avg_model) for _ in range(n_agents)]
            communication_cost += n_agents * comm_factor

    global_model = _average_models(models)

    eval_idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    final_accuracy = _compute_accuracy(global_model, X[eval_idx], y[eval_idx])

    total_energy = compute_cost * 0.1 + communication_cost * 0.01

    return {
        "final_accuracy": final_accuracy,
        "total_compute": float(compute_cost),
        "total_comm": float(communication_cost),
        "total_energy": float(total_energy),
        "acc_per_energy": final_accuracy / (total_energy + 1e-9),
        "acc_per_comm": final_accuracy / (communication_cost + 1e-9),
        "graph_type": graph_type,
    }
