"""Dataset utilities for the RCE simulation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_numpy() -> tuple[np.ndarray, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    return X, y


def load_cifar10_numpy() -> tuple[np.ndarray, np.ndarray]:
    try:
        from tensorflow.keras.datasets import cifar10  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TensorFlow required'."
        ) from exc

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32) / 255.0
    y = np.concatenate([y_train, y_test], axis=0).astype(int).ravel()
    X = X.reshape(X.shape[0], -1)
    return X, y


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    name = name.lower()
    if name in {"mnist", "mnist_784"}:
        return load_mnist_numpy()
    if name in {"cifar", "cifar10", "cifar_10"}:
        return load_cifar10_numpy()
    raise ValueError(f"Unsupported dataset: {name}")


def build_agent_datasets(
    X: np.ndarray,
    y: np.ndarray,
    n_agents: int,
    rng,
    size_profile: Sequence[int] | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if size_profile is None:
        size_profile = np.array([8000] * max(n_agents, 10), dtype=int)
    else:
        size_profile = np.asarray(size_profile, dtype=int)

    if n_agents > len(size_profile):
        size_profile = np.concatenate(
            [size_profile, np.full(n_agents - len(size_profile), size_profile[-1], dtype=int)]
        )
    elif n_agents < len(size_profile):
        size_profile = size_profile[:n_agents]

    digit_indices = {d: np.where(y == d)[0] for d in range(10)}
    for d in range(10):
        rng.shuffle(digit_indices[d])

    all_other_indices = {d: np.where(y != d)[0] for d in range(10)}

    multi_digit_agents = min(3, n_agents)
    multi_digit_values = np.arange(1, 8)  # digits 1-7 inclusive
    multi_digit_pool = np.concatenate([digit_indices[d] for d in multi_digit_values])
    multi_digit_other_pool = np.concatenate([digit_indices[d] for d in range(10) if d not in multi_digit_values])

    agent_datasets = []
    for i in range(n_agents):
        home_digit = i % 10
        total_size = int(size_profile[i])

        main_size = int(0.7 * total_size)
        other_size = total_size - main_size

        if i < multi_digit_agents:
            main_idx = rng.choice(multi_digit_pool, size=main_size, replace=True)
            other_idx = rng.choice(multi_digit_other_pool, size=other_size, replace=True)
        else:
            main_idx = rng.choice(digit_indices[home_digit], size=main_size, replace=True)
            other_idx = rng.choice(all_other_indices[home_digit], size=other_size, replace=True)

        idx_i = np.concatenate([main_idx, other_idx])
        rng.shuffle(idx_i)

        Xi = X[idx_i]
        yi = y[idx_i]

        agent_datasets.append((Xi, yi))

    return agent_datasets


def compute_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    H = X @ model["W1"].T + model["b1"]
    H_relu = np.maximum(H, 0)
    logits = H_relu @ model["W2"].T + model["b2"]
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))
