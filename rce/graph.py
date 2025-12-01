"""Graph generation and trust matrix utilities."""

from __future__ import annotations

import numpy as np
import networkx as nx


def build_graph(n: int, graph_type: str = "ring") -> nx.Graph:
    """Construct supported graph topologies for the agent network."""
    if graph_type == "ring":
        return nx.cycle_graph(n)
    if graph_type == "complete":
        return nx.complete_graph(n)
    if graph_type == "smallworld":
        return nx.watts_strogatz_graph(n, k=4, p=0.2)
    if graph_type == "scale-free":
        return nx.barabasi_albert_graph(n, m=2)
    raise ValueError(f"Unknown graph_type: {graph_type}")


def sinkhorn_knopp_doubly_stochastic(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Approximate a doubly stochastic matrix using Sinkhorn-Knopp."""
    W = A.astype(float)
    for _ in range(max_iter):
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        W /= row_sums

        col_sums = W.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        W /= col_sums

        if np.allclose(W.sum(axis=1), 1.0, atol=tol) and np.allclose(
            W.sum(axis=0), 1.0, atol=tol
        ):
            break
    return W


def build_trust_matrix(G: nx.Graph) -> np.ndarray:
    """Create an initial row-stochastic trust matrix over the graph."""
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=float)

    for i in range(n):
        A[i, i] = 1.0
        for j in G.neighbors(i):
            A[i, j] = 1.0

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return A / row_sums
