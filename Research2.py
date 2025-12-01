import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

from dpsgd import run_dpsgd_baseline
from rce.config import RCEConfig
from rce.data import compute_accuracy, load_mnist_numpy
from rce.experiment import run_experiment
from rce.plotting import (
    credit_conservation_report,
    plot_rce_graph,
    plot_results,
    plot_trust_heatmap,
)


# Lower temperature so ensemble prioritizes high-utility agents.
ENSEMBLE_TEMPERATURE = 0.2

# Periodically force a full communication round to keep models synchronized.
FORCE_COMM_INTERVAL = 400


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
            dpsgd_metrics = run_dpsgd_baseline(
                X,
                y,
                cfg_variant.n_agents,
                cfg_variant.T,
                graph_type=graph_type,
                seed=cfg_variant.seed,
            )

            metrics = {
                "final_accuracy": logs["final_accuracy"],
                "total_energy": logs["total_energy"],
                "total_compute": logs["total_compute"],
                "total_comm": logs["total_comm"],
                "acc_per_energy": logs["acc_per_energy"],
                "acc_per_comm": logs["acc_per_comm"],
                "ensemble_accuracy": ensemble_info["accuracy"],
            }

            results.append(
                {
                    "scenario": scenario["name"],
                    "graph": graph_type,
                    "cfg": cfg_variant,
                    "metrics": metrics,
                    "dpsgd": dpsgd_metrics,
                    "seed": cfg_variant.seed,
                }
            )

            if reference is None:
                reference = {
                    "logs": logs,
                    "cfg": cfg_variant,
                    "ensemble": ensemble_info,
                    "dpsgd": dpsgd_metrics,
                    "scenario": scenario["name"],
                    "graph": graph_type,
                }

    return results, reference


def summarize_multi_graph_results(results):
    if not results:
        return

    print("\n===== MULTI-GRAPH RCE vs D-PSGD =====")
    header = (
        f"{'Scenario':<16}{'Graph':<12}{'RCE Acc':>10}{'D Acc':>10}{'ΔAcc':>10}"
        f"{'RCE Energy':>12}{'D Energy':>12}{'ΔEnergy':>12}{'RCE ACC/E':>12}{'D ACC/E':>12}{'Ens Acc':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        r = row["metrics"]
        d = row["dpsgd"]
        acc_delta = r["final_accuracy"] - d["final_accuracy"]
        energy_delta = r["total_energy"] - d["total_energy"]
        print(
            f"{row['scenario']:<16}{row['graph']:<12}"
            f"{r['final_accuracy']:>10.4f}{d['final_accuracy']:>10.4f}{acc_delta:>10.4f}"
            f"{r['total_energy']:>12.1f}{d['total_energy']:>12.1f}{energy_delta:>12.1f}"
            f"{r['acc_per_energy']:>12.6f}{d['acc_per_energy']:>12.6f}"
            f"{r['ensemble_accuracy']:>10.4f}"
        )




def main():
    base_cfg = RCEConfig(
        n_agents=10,
        graph_type="ring",
        T=10000,
        gamma=0.035,
        lambda_reg=0.0,
        alpha_decay=0.0,
        eps=1e-8,
        seed=131,
        n_actions=3,
        lr_policy=0.025,
        extinction_enabled=False,
        force_comm_every=FORCE_COMM_INTERVAL,
    )

    X_mnist, y_mnist = load_mnist_numpy()

    batch_results, reference = run_multi_graph_experiments(
        base_cfg,
        SCENARIOS,
        GRAPH_TYPES,
        X_mnist,
        y_mnist,
        ensemble_temp=ENSEMBLE_TEMPERATURE,
    )

    summarize_multi_graph_results(batch_results)

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
    print("Total Energy:", ref_logs["total_energy"])
    print("Accuracy per Energy:", ref_logs["acc_per_energy"])
    print("Accuracy per Comm:", ref_logs["acc_per_comm"])

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

    plt.figure(figsize=(6, 4))
    plt.scatter(ref_logs["total_energy"], ref_logs["final_accuracy"], s=160)
    plt.xlabel("Total Energy Used")
    plt.ylabel("Final Accuracy")
    plt.title("Accuracy vs Energy (RCE)")
    plt.grid(True)
    plt.show()

    plt.plot(ref_logs["mean_accuracy"])
    plt.title("Mean Accuracy Over Time")
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.show()

    credit_conservation_report(ref_logs)
    plot_results(ref_logs)
    plot_trust_heatmap(ref_logs["trust_history"])


if __name__ == "__main__":
    main()


