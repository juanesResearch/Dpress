import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from pathlib import Path

from dpsgd import run_dpsgd_baseline
from gossip import run_gossip_sgd_baseline
from rce.config import RCEConfig
from rce.data import build_agent_datasets, compute_accuracy, load_dataset
from rce.experiment import run_experiment
from rce.plotting import (
    credit_conservation_report,
    plot_rce_graph,
    plot_results,
    plot_trust_heatmap,
)


# Lower temperature so ensemble prioritizes high-utility agents.
ENSEMBLE_TEMPERATURE = 0.05

# Periodically force a full communication round to keep models synchronized.
FORCE_COMM_INTERVAL = 5000
ACCURACY_PLOT_STRIDE = 50
POLICY_CKPT_PATH = "checkpoints/rce_policy.npy"
DATASET_NAME = "mnist"


def _sample_history(values, stride):
    if stride <= 1:
        stride = 1
    if values is None:
        return []

    arr = np.asarray(values)
    if arr.size == 0:
        return []

    if arr.ndim == 2 and arr.shape[1] == 2:
        steps = arr[:, 0].astype(float)
        series = arr[:, 1].astype(float)
    else:
        series = arr.astype(float).ravel()
        steps = np.arange(1, series.size + 1, dtype=float)

    order = np.argsort(steps)
    steps = steps[order]
    series = series[order]

    while series.size > 1 and series[-1] <= 0.0 and series[-2] > 0.0:
        steps = steps[:-1]
        series = series[:-1]

    if steps.size == 1:
        return [(steps[0], series[0])]

    max_step = steps[-1]
    sample_steps = np.arange(stride, max_step + 1, stride, dtype=float)
    if sample_steps.size == 0 or sample_steps[0] > steps[0]:
        sample_steps = np.insert(sample_steps, 0, steps[0])
    if sample_steps[-1] != max_step:
        sample_steps = np.append(sample_steps, max_step)

    sampled_values = np.interp(sample_steps, steps, series)
    return list(zip(sample_steps, sampled_values))


def _smooth_series(y, window=5):
    arr = np.asarray(y, dtype=float)
    if arr.size < 2 or window <= 1:
        return arr
    window = min(window, arr.size)
    kernel = np.ones(window, dtype=float) / window
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = arr[0]
    smoothed[-1] = arr[-1]
    return smoothed


def utility_weighted_average(agent_models, utilities, temperature: float = 1.0):
    util = np.asarray(utilities, dtype=float)
    if util.ndim != 1:
        util = util.ravel()
    temp = max(temperature, 1e-6)
    scaled = util / temp
    scaled -= scaled.max()
    weights = np.exp(scaled)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()

    averaged = {}
    for key in agent_models[0].keys():
        stacked = np.stack([m[key] for m in agent_models], axis=0)
        averaged[key] = np.tensordot(weights, stacked, axes=1)

    return averaged, weights


def evaluate_rce_ensemble(logs, X, y, temperature: float = 1.0, eval_size: int = 5000):
    agent_models = logs["agent_models"]
    utilities = logs["utility_history"][-1]
    ensemble_model, weights = utility_weighted_average(agent_models, utilities, temperature=temperature)

    rng = np.random.default_rng(logs["cfg"].seed + 999)
    sample = min(eval_size, len(X))
    eval_idx = rng.choice(len(X), size=sample, replace=False)
    accuracy = compute_accuracy(ensemble_model, X[eval_idx], y[eval_idx])

    return {
        "model": ensemble_model,
        "weights": weights,
        "accuracy": accuracy,
    }


def summarize_methods(rce_logs, dpsgd_logs):
    metrics = [
        ("Final Accuracy", "final_accuracy"),
        ("Total Energy", "total_energy"),
        ("ACC/E", "acc_per_energy"),
        ("ACC/COMM", "acc_per_comm"),
    ]

    print("\n===== RCE vs D-PSGD =====")
    header = f"{'Metric':<18}{'RCE':>14}{'D-PSGD':>14}{'Delta':>14}{'Delta%':>10}"
    print(header)
    print("-" * len(header))

    for label, key in metrics:
        r_val = rce_logs.get(key, float("nan"))
        d_val = dpsgd_logs.get(key, float("nan"))
        delta = r_val - d_val
        rel = np.nan
        if not np.isnan(d_val) and d_val != 0:
            rel = delta / d_val * 100.0
        print(f"{label:<18}{r_val:>14.4f}{d_val:>14.4f}{delta:>14.4f}{rel:>9.2f}%")


SCENARIOS = [
    {
        "name": "decay_ext",
        "lambda_reg": 0.0,
        "alpha_decay": 0.003,
        "extinction_enabled": True,
    },
    {
        "name": "pure_diffusion",
        "lambda_reg": 0.0,
        "alpha_decay": 0.0,
        "extinction_enabled": False,
    },
]

GRAPH_TYPES = ["ring", "complete", "smallworld"]


def run_multi_graph_experiments(
    base_cfg,
    scenarios,
    graph_types,
    X,
    y,
    ensemble_temp=ENSEMBLE_TEMPERATURE,
):
    results = []
    reference = None

    for s_idx, scenario in enumerate(scenarios):
        for g_idx, graph_type in enumerate(graph_types):
            cfg_variant = replace(
                base_cfg,
                graph_type=graph_type,
                lambda_reg=scenario["lambda_reg"],
                alpha_decay=scenario["alpha_decay"],
                extinction_enabled=scenario["extinction_enabled"],
                seed=base_cfg.seed + 100 * s_idx + g_idx,
            )

            print(
                f"\n=== RCE RUN scenario={scenario['name']} graph={graph_type} seed={cfg_variant.seed} ==="
            )
            logs = run_experiment(cfg_variant)
            ensemble_info = evaluate_rce_ensemble(logs, X, y, temperature=ensemble_temp)

            baseline_rng = np.random.default_rng(cfg_variant.seed + 4242)
            baseline_datasets = build_agent_datasets(
                X,
                y,
                cfg_variant.n_agents,
                baseline_rng,
            )
            dpsgd_metrics = run_dpsgd_baseline(
                X,
                y,
                cfg_variant.n_agents,
                cfg_variant.T,
                graph_type=graph_type,
                seed=cfg_variant.seed,
                compute_costs=cfg_variant.compute_energy_costs,
                comm_costs=cfg_variant.comm_energy_costs,
                agent_datasets=baseline_datasets,
                eval_stride=ACCURACY_PLOT_STRIDE,
                exclusive_comm_rounds=True,
            )
            gossip_metrics = run_gossip_sgd_baseline(
                X,
                y,
                cfg_variant.n_agents,
                cfg_variant.T,
                graph_type=graph_type,
                seed=cfg_variant.seed,
                compute_costs=cfg_variant.compute_energy_costs,
                comm_costs=cfg_variant.comm_energy_costs,
                agent_datasets=baseline_datasets,
                eval_stride=ACCURACY_PLOT_STRIDE,
                pairs_per_round=1,
                exclusive_comm_rounds=True,
            )

            metrics = {
                "final_accuracy": logs["final_accuracy"],
                "total_energy": logs["total_energy"],
                "total_compute": logs["total_compute"],
                "total_comm": logs["total_comm"],
                "total_rest": logs.get("total_rest", float("nan")),
                "acc_per_energy": logs["acc_per_energy"],
                "acc_per_comm": logs["acc_per_comm"],
                "acc_per_rest": logs.get("acc_per_rest", float("nan")),
                "ensemble_accuracy": ensemble_info["accuracy"],
            }

            rce_comm_history = list(logs.get("comm_used", []))
            dpsgd_comm_history = dpsgd_metrics.get("comm_counts_history", [])
            gossip_comm_history = gossip_metrics.get("comm_counts_history", [])

            results.append(
                {
                    "scenario": scenario["name"],
                    "graph": graph_type,
                    "cfg": cfg_variant,
                    "metrics": metrics,
                    "dpsgd": dpsgd_metrics,
                    "seed": cfg_variant.seed,
                    "gossip": gossip_metrics,
                    "rce_history": list(logs.get("mean_accuracy", [])),
                    "dpsgd_history": dpsgd_metrics.get("accuracy_history", []),
                    "gossip_history": gossip_metrics.get("accuracy_history", []),
                    "rce_comm_history": rce_comm_history,
                    "dpsgd_comm_history": dpsgd_comm_history,
                    "gossip_comm_history": gossip_comm_history,
                }
            )

            if reference is None:
                reference = {
                    "logs": logs,
                    "cfg": cfg_variant,
                    "ensemble": ensemble_info,
                    "dpsgd": dpsgd_metrics,
                    "gossip": gossip_metrics,
                    "scenario": scenario["name"],
                    "graph": graph_type,
                }

    expected_pairs = {
        (scenario["name"], graph_type)
        for scenario in scenarios
        for graph_type in graph_types
    }
    completed_pairs = {(row["scenario"], row["graph"]) for row in results}
    missing_pairs = sorted(expected_pairs - completed_pairs)
    if missing_pairs:
        print("\n[warn] Missing scenario/graph runs:")
        for scenario_name, graph_type in missing_pairs:
            print(f"  - scenario={scenario_name}, graph={graph_type}")

    return results, reference


def summarize_multi_graph_results(results):
    if not results:
        return

    print("\n===== MULTI-GRAPH RCE vs D-PSGD =====")
    header = (
        f"{'Scenario':<16}{'Graph':<12}{'RCE Acc':>10}{'D Acc':>10}{'G Acc':>10}"
        f"{'ΔR-D':>10}{'ΔR-G':>10}"
        f"{'RCE Energy':>12}{'D Energy':>12}{'G Energy':>12}"
        f"{'RCE ACC/E':>12}{'D ACC/E':>12}{'G ACC/E':>12}{'Ens Acc':>10}"
        f"{'R Compute':>12}{'R Comm':>12}{'R Rest':>12}{'R ACC/R':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        r = row["metrics"]
        d = row["dpsgd"]
        g = row.get("gossip", {})
        acc_delta_d = r["final_accuracy"] - d["final_accuracy"]
        acc_delta_g = r["final_accuracy"] - g.get("final_accuracy", float("nan"))
        print(
            f"{row['scenario']:<16}{row['graph']:<12}"
            f"{r['final_accuracy']:>10.4f}{d['final_accuracy']:>10.4f}{g.get('final_accuracy', float('nan')):>10.4f}"
            f"{acc_delta_d:>10.4f}{acc_delta_g:>10.4f}"
            f"{r['total_energy']:>12.1f}{d['total_energy']:>12.1f}{g.get('total_energy', float('nan')):>12.1f}"
            f"{r['acc_per_energy']:>12.6f}{d['acc_per_energy']:>12.6f}{g.get('acc_per_energy', float('nan')):>12.6f}"
            f"{r['ensemble_accuracy']:>10.4f}"
            f"{r.get('total_compute', float('nan')):>12.1f}"
            f"{r.get('total_comm', float('nan')):>12.1f}"
            f"{r.get('total_rest', float('nan')):>12.1f}"
            f"{r.get('acc_per_rest', float('nan')):>12.6f}"
        )


def plot_accuracy_comparison(results, graph_order=None):
    if not results:
        return

    if graph_order is None:
        graph_order = GRAPH_TYPES

    graph_order = list(graph_order)
    graph_positions = np.arange(len(graph_order))

    scenario_data = {}
    for row in results:
        scenario = row["scenario"]
        idx = graph_order.index(row["graph"])
        data = scenario_data.setdefault(
            scenario,
            {
                "rce": np.zeros(len(graph_order)),
                "dpsgd": np.zeros(len(graph_order)),
                "gossip": np.zeros(len(graph_order)),
            },
        )
        data["rce"][idx] = row["metrics"]["final_accuracy"]
        data["dpsgd"][idx] = row["dpsgd"]["final_accuracy"]
        data["gossip"][idx] = row["gossip"]["final_accuracy"]

    fine_x = np.linspace(graph_positions.min(), graph_positions.max(), 200)

    for scenario, values in scenario_data.items():
        plt.figure(figsize=(8, 4))
        for label, arr, color in [
            ("RCE", values["rce"], "#1f77b4"),
            ("D-PSGD", values["dpsgd"], "#ff7f0e"),
            ("Gossip", values["gossip"], "#2ca02c"),
        ]:
            smoothed = np.interp(fine_x, graph_positions, arr)
            plt.plot(fine_x, smoothed, label=label, linewidth=2, color=color)
            plt.scatter(graph_positions, arr, color=color, s=50)

        plt.xticks(graph_positions, graph_order)
        plt.ylim(0.7, 1.0)
        plt.xlabel("Graph Topology")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Comparison per Graph — {scenario}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_graph_accuracy_trajectories(
    results,
    graph_types=None,
    stride=ACCURACY_PLOT_STRIDE,
    smooth_window=None,
):
    if not results:
        return

    if graph_types is None:
        graph_types = GRAPH_TYPES

    scenario_names = sorted({row["scenario"] for row in results})
    linestyles = ["-", "--", ":", "-."]
    scenario_style = {
        name: linestyles[i % len(linestyles)] for i, name in enumerate(scenario_names)
    }
    method_colors = {
        "RCE": "#1f77b4",
        "D-PSGD": "#ff7f0e",
        "Gossip": "#2ca02c",
    }

    for graph in graph_types:
        plt.figure(figsize=(9, 4))
        for scenario in scenario_names:
            row = next(
                (r for r in results if r["graph"] == graph and r["scenario"] == scenario),
                None,
            )
            if not row:
                continue

            histories = {
                "RCE": row.get("rce_history"),
                "D-PSGD": row.get("dpsgd_history"),
                "Gossip": row.get("gossip_history"),
            }

            for method, history in histories.items():
                sampled = _sample_history(history, stride)
                if not sampled:
                    continue
                steps, values = zip(*sampled)
                steps = np.asarray(steps, dtype=float)
                if smooth_window and smooth_window > 1:
                    series = _smooth_series(values, window=smooth_window)
                else:
                    series = np.asarray(values, dtype=float)
                label = f"{scenario} {method}"
                plt.plot(
                    steps,
                    series,
                    label=label,
                    color=method_colors[method],
                    linestyle=scenario_style.get(scenario, "-"),
                    linewidth=2,
                )

        plt.xlabel("Training Steps")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Over Time — {graph}")
        plt.ylim(0.6, 1.0)
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_comm_activity(
    results,
    graph_types=None,
    stride=ACCURACY_PLOT_STRIDE,
):
    if not results:
        return

    if graph_types is None:
        graph_types = GRAPH_TYPES

    scenario_names = sorted({row["scenario"] for row in results})
    linestyles = ["-", "--", ":", "-."]
    scenario_style = {
        name: linestyles[i % len(linestyles)] for i, name in enumerate(scenario_names)
    }
    method_colors = {
        "RCE": "#1f77b4",
        "D-PSGD": "#ff7f0e",
        "Gossip": "#2ca02c",
    }

    for graph in graph_types:
        plt.figure(figsize=(9, 4))
        for scenario in scenario_names:
            row = next(
                (r for r in results if r["graph"] == graph and r["scenario"] == scenario),
                None,
            )
            if not row:
                continue

            comm_histories = {
                "RCE": row.get("rce_comm_history"),
                "D-PSGD": row.get("dpsgd_comm_history"),
                "Gossip": row.get("gossip_comm_history"),
            }

            for method, history in comm_histories.items():
                if not history:
                    continue
                sampled = _sample_history(history, stride)
                if not sampled:
                    continue
                steps, values = zip(*sampled)
                plt.plot(
                    steps,
                    values,
                    label=f"{scenario} {method}",
                    color=method_colors[method],
                    linestyle=scenario_style.get(scenario, "-"),
                    linewidth=2,
                )

        plt.xlabel("Training Steps")
        plt.ylabel("Comm actions per step")
        plt.title(f"Communication Activity — {graph}")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_comm_accuracy_diagnostics(logs):
    mean_accuracy = np.asarray(logs.get("mean_accuracy", []), dtype=float)
    comm_used = np.asarray(logs.get("comm_used", []), dtype=float)

    if mean_accuracy.size == 0 or comm_used.size == 0:
        print("[warn] Missing accuracy or communication history; skipping diagnostic plot")
        return

    steps = np.arange(1, mean_accuracy.size + 1, dtype=float)
    window_long = min(201, mean_accuracy.size)
    window_short = min(101, comm_used.size)

    acc_smoothed = _smooth_series(mean_accuracy, window=window_long)
    comm_smoothed = _smooth_series(comm_used, window=window_short)

    acc_delta = np.diff(np.insert(mean_accuracy, 0, mean_accuracy[0]))
    acc_delta = np.maximum(0.0, acc_delta)
    delta_smoothed = _smooth_series(acc_delta, window=min(51, acc_delta.size))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(steps, mean_accuracy, color="#9ecae1", linewidth=1, label="Raw accuracy")
    axes[0].plot(steps, acc_smoothed, color="#1f77b4", linewidth=2, label="Smoothed accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(steps, comm_used, color="#fdd0a2", linewidth=1, label="Comm actions/step")
    axes[1].plot(steps, comm_smoothed, color="#ff7f0e", linewidth=2, label="Smoothed comm")
    axes[1].set_ylabel("Comm count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].bar(steps, delta_smoothed, color="#31a354", width=1.0, alpha=0.7)
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Positive Δ accuracy")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Accuracy vs Communication Activity (single RCE run)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def main():
    base_compute_costs = [0.01, 0.01, 0.012, 0.012, 0.014, 0.014, 0.016, 0.016, 0.018, 0.018]
    base_comm_costs = [0.12, 0.12, 0.125, 0.125, 0.13, 0.13, 0.135, 0.135, 0.14, 0.14]
    base_rest_costs = [0.02, 0.02, 0.022, 0.022, 0.024, 0.024, 0.026, 0.026, 0.028, 0.028]

    policy_ckpt = Path(POLICY_CKPT_PATH)
    policy_init = str(policy_ckpt) if policy_ckpt.exists() else None

    base_cfg = RCEConfig(
        n_agents=10,
        graph_type="ring",
        T=4000,
        gamma=0.035,
        lambda_reg=0.0,
        alpha_decay=0.0,
        eps=1e-8,
        seed=131,
        n_actions=2,
        lr_policy=0.025,
        extinction_enabled=False,
        force_comm_every=FORCE_COMM_INTERVAL,
        compute_energy_costs=base_compute_costs,
        comm_energy_costs=base_comm_costs,
        rest_energy_costs=base_rest_costs,
        policy_init_path=policy_init,
        policy_save_path=str(policy_ckpt),
        dataset_name=DATASET_NAME,
    )

    X_data, y_data = load_dataset(DATASET_NAME)

    batch_results, reference = run_multi_graph_experiments(
        base_cfg,
        SCENARIOS,
        GRAPH_TYPES,
        X_data,
        y_data,
        ensemble_temp=ENSEMBLE_TEMPERATURE,
    )

    summarize_multi_graph_results(batch_results)
    plot_accuracy_comparison(batch_results)
    plot_graph_accuracy_trajectories(batch_results)
    plot_comm_activity(batch_results)

    if not reference:
        return

    ref_logs = reference["logs"]
    ref_cfg = reference["cfg"]
    ref_ensemble = reference["ensemble"]
    ref_dpsgd = reference["dpsgd"]

    print(
        f"\n===== RCE FINAL METRICS (scenario={reference['scenario']}, graph={reference['graph']}) ====="
    )
    print("Final Accuracy:", ref_logs["final_accuracy"])
    print("Total Compute Actions:", ref_logs["total_compute"])
    print("Total Communication Actions:", ref_logs["total_comm"])
    print("Total Rest Actions:", ref_logs.get("total_rest"))
    print("Total Energy:", ref_logs["total_energy"])
    print("Accuracy per Energy:", ref_logs["acc_per_energy"])
    print("Accuracy per Comm:", ref_logs["acc_per_comm"])
    print("Accuracy per Rest:", ref_logs.get("acc_per_rest"))

    print("\nUtility-weighted ensemble accuracy (RCE):", round(ref_ensemble["accuracy"], 4))
    top_agent = int(np.argmax(ref_ensemble["weights"]))
    print(
        "Top contributing agent:",
        top_agent,
        "with weight",
        round(float(ref_ensemble["weights"][top_agent]), 3),
    )

    print("\n===== D-PSGD RESULTS (reference graph) =====")
    print("Final Accuracy:", ref_dpsgd["final_accuracy"])
    print("Total Compute:", ref_dpsgd["total_compute"])
    print("Total Comm:", ref_dpsgd["total_comm"])
    print("Total Energy:", ref_dpsgd["total_energy"])
    print("ACC/E:", ref_dpsgd["acc_per_energy"])
    print("ACC/COMM:", ref_dpsgd["acc_per_comm"])

    summarize_methods(ref_logs, ref_dpsgd)

    final_credit = ref_logs["credits_history"][-1]
    final_utility = ref_logs["utility_history"][-1]
    final_W = ref_logs["trust_history"][-1]

    plot_rce_graph(
        ref_logs["G"],
        final_credit,
        final_utility,
        final_W,
        graph_type=ref_cfg.graph_type,
        seed=ref_cfg.seed,
    )

    plot_comm_accuracy_diagnostics(ref_logs)

    credit_conservation_report(ref_logs)
    plot_results(ref_logs)
    plot_trust_heatmap(ref_logs["trust_history"])


if __name__ == "__main__":
    main()


