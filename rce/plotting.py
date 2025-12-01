"""Plotting and reporting helpers for RCE experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def moving_average(x, w: int = 50):
    """Moving average that preserves array length."""
    x = np.asarray(x)
    w = int(w)

    if w < 1:
        return x
    if w > len(x):
        w = len(x)

    pad = w // 2
    x_padded = np.pad(x, (pad, pad), mode="edge")
    smooth_full = np.convolve(x_padded, np.ones(w) / w, mode="same")
    return smooth_full[pad:-pad]


def plot_results(logs):
    from scipy.signal import savgol_filter

    cfg = logs["cfg"]
    credits_history = logs["credits_history"]
    rewards_history = logs["rewards_history"]
    utility_history = logs["utility_history"]

    T = credits_history.shape[0] - 1
    n = cfg.n_agents
    t_axis = np.arange(T + 1)

    plt.figure(figsize=(10, 6))
    window = 50

    for i in range(n):
        credit_curve = credits_history[:, i]
        credit_smooth = moving_average(credit_curve, w=window)
        plt.plot(t_axis, credit_smooth, label=f"Agent {i}")

    plt.xlabel("Time step")
    plt.ylabel("Credit c_i(t)")
    plt.title(f"Smoothed Credit Trajectories (N={n}, graph={cfg.graph_type})")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    plt.figure(figsize=(10, 6))

    for i in range(n):
        util = utility_history[:, i]
        util_ma = moving_average(util, w=200)
        util_smooth = savgol_filter(util_ma, window_length=201, polyorder=3)
        t = np.arange(len(util_smooth))
        plt.plot(t, util_smooth, label=f"Agent {i}")

    plt.xlabel("Time step")
    plt.ylabel("Smoothed utility u_i(t)")
    plt.title("Smoothed Utility Trajectories (Curved)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    avg_reward = rewards_history.mean(axis=1)
    plt.plot(np.arange(T), avg_reward)
    plt.xlabel("Time step")
    plt.ylabel("Average reward")
    plt.title("Average reward over time")
    plt.tight_layout()

    plt.show()


def credit_conservation_report(logs):
    cfg = logs["cfg"]
    total_credit = logs["total_credit"]

    initial = total_credit[0]
    final = total_credit[-1]
    change = final - initial
    percent_change = (change / initial) * 100.0

    print("========== CREDIT DYNAMICS SUMMARY ==========")
    print(f"Graph type:           {cfg.graph_type}")
    print(f"N agents:             {cfg.n_agents}")
    print(f"T steps:              {cfg.T}")
    print(f"gamma (exchange):     {cfg.gamma}")
    print(f"lambda_reg (regen):   {cfg.lambda_reg}")
    print(f"alpha_decay:          {cfg.alpha_decay}")
    print("---------------------------------------------")
    print(f"Initial total credit: {initial:.4f}")
    print(f"Final total credit:   {final:.4f}")
    print(f"Absolute change:      {change:.4f}")
    print(f"Percent change:       {percent_change:.2f}%")
    print("=============================================")


def plot_rce_graph(
    G: nx.Graph,
    final_credit,
    final_utility,
    final_W,
    graph_type: str = "unknown",
    seed: int | None = None,
):
    import matplotlib.cm as cm

    n = len(final_credit)

    layout_seed = seed if seed is not None else 123

    if graph_type == "ring":
        pos = nx.circular_layout(G)
    elif graph_type == "complete":
        pos = nx.spring_layout(G, seed=layout_seed, k=2.0)
    elif graph_type == "scale-free":
        pos = nx.spring_layout(G, seed=layout_seed, k=1.2)
    elif graph_type == "smallworld":
        pos = nx.spring_layout(G, seed=layout_seed, k=0.9)
    else:
        pos = nx.spring_layout(G, seed=layout_seed)

    credit = np.array(final_credit)
    credit_norm = (credit - credit.min()) / (credit.max() - credit.min() + 1e-8)
    node_sizes = 800 * (0.3 + 2.0 * credit_norm)

    utility = np.array(final_utility)
    util_norm = (utility - utility.min()) / (utility.max() - utility.min() + 1e-8)
    cmap = cm.viridis
    node_colors = cmap(util_norm)

    edges = []
    widths = []
    for i, j in G.edges():
        edges.append((i, j))
        widths.append(5 * final_W[i, j])

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.2,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.8,
    )

    for i in range(n):
        x, y = pos[i]
        label_text = f"{i}\nC={credit[i]:.2f}\nU={utility[i]:.2f}"
        plt.text(
            x,
            y - 0.07,
            label_text,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.25",
                alpha=0.85,
            ),
        )

    plt.title("RCE Graph — Node Size = Credit,  Node Color = Utility")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_trust_heatmap(W_hist):
    T, n, _ = W_hist.shape

    trust_into = np.zeros((T, n))
    for t in range(T):
        trust_into[t] = W_hist[t].sum(axis=0)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        trust_into,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )

    plt.colorbar(label="Incoming Trust")
    plt.xlabel("Agent")
    plt.ylabel("Time Step")
    plt.title("Trust Evolution Heatmap (0 → T)")
    plt.xticks(range(n))
    plt.tight_layout()
    plt.show()
