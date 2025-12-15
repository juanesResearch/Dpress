"""Environment dynamics, rewards, and credit flow logic."""

from __future__ import annotations

import numpy as np

from .config import RCEConfig


def update_utilities(u: np.ndarray, t: int, cfg: RCEConfig, rng) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=0.02, size=u.shape)
    u_new = u + noise
    return np.clip(u_new, 0.0, 1.0)


def update_energy_costs(e: np.ndarray, t: int, cfg: RCEConfig, rng) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=0.02, size=e.shape)
    e_new = e + noise
    return np.clip(e_new, 0.0, 1.0)


def apply_action_effects(u: np.ndarray, e: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_new = u.copy()
    e_new = e.copy()

    for i, a in enumerate(actions):
        if a == 0:
            u_new[i] += 0.10
            e_new[i] += 0.10
        elif a == 1:
            u_new[i] += 0.05
            e_new[i] += 0.05
        elif a == 2:
            u_new[i] -= 0.02
            e_new[i] -= 0.10

    u_new = np.clip(u_new, 0.0, 1.0)
    e_new = np.clip(e_new, 0.0, 1.0)
    return u_new, e_new


def compute_flows(
    c: np.ndarray,
    u: np.ndarray,
    W: np.ndarray,
    neighbors: dict[int, list[int]],
    cfg: RCEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(c)
    gamma = cfg.gamma
    eps = cfg.eps

    denom = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for k in neighbors[i]:
            if u[k] > 0:
                acc += u[k] * W[i, k]
        denom[i] = acc + eps

    delta = np.zeros((n, n))

    for i in range(n):
        if c[i] <= 0:
            continue
        Zi = denom[i]
        for j in neighbors[i]:
            if u[j] <= 0:
                continue
            delta[i, j] = gamma * c[i] * (u[j] * W[i, j] / Zi)

    outflow = delta.sum(axis=1)
    inflow = delta.sum(axis=0)

    net_flow = inflow.sum() - outflow.sum()

    if abs(net_flow) > 1e-12:
        correction = net_flow / n
        inflow -= correction / 2
        outflow += correction / 2

    return inflow, outflow, delta


def compute_reward(u: np.ndarray, e: np.ndarray) -> np.ndarray:
    beta = 1.0
    gamma_e = 0.8
    return beta * u - gamma_e * e


def compute_relational_utility(r: np.ndarray, W: np.ndarray, neighbors: dict[int, list[int]]) -> np.ndarray:
    n = len(r)
    u_new = np.zeros(n, dtype=float)
    for i in range(n):
        total = r[i]
        for j in neighbors[i]:
            total += W[i, j] * r[j]
        u_new[i] = total
    return u_new


def update_trust(
    W: np.ndarray,
    u: np.ndarray,
    neighbors: dict[int, list[int]],
    eta: float = 0.05,
) -> np.ndarray:
    n = len(u)
    W_new = np.zeros_like(W)

    for i in range(n):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            W_new[i, i] = 1.0
            continue

        u_vals = np.array([u[j] for j in nbrs])
        total = u_vals.sum()

        if total > 0:
            target = u_vals / total
        else:
            target = np.ones(len(nbrs)) / len(nbrs)

        for idx, j in enumerate(nbrs):
            W_new[i, j] = (1 - eta) * W[i, j] + eta * target[idx]

        W_new[i, i] = 0.0

    row_sums = W_new.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W_new / row_sums


def step_env(
    c: np.ndarray,
    u: np.ndarray,
    e: np.ndarray,
    W: np.ndarray,
    neighbors: dict[int, list[int]],
    actions: np.ndarray,
    agent_models: np.ndarray,
    agent_datasets: list[tuple[np.ndarray, np.ndarray]],
    t: int,
    cfg: RCEConfig,
    rng,
):
    n = len(c)
    batch_size = 32
    lr_compute = 0.05
    lr_comm = 0.05

    e_rw = update_energy_costs(e, t, cfg, rng)
    e_new = e_rw.copy()

    for i, a in enumerate(actions):
        if a == 0:
            e_new[i] += 0.10
        elif a == 1:
            e_new[i] += 0.05
        elif a == 2:
            e_new[i] -= 0.05

    e_new = np.clip(e_new, 0, 1)

    batches_X = []
    batches_yoh = []

    for i in range(n):
        Xi, yi = agent_datasets[i]
        idx_i = rng.choice(len(Xi), size=batch_size, replace=False)
        Xb = Xi[idx_i]
        yb = yi[idx_i]
        batches_X.append(Xb)
        batches_yoh.append(np.eye(10)[yb])

    def batch_accuracy(model, Xb, yb):
        H = Xb @ model["W1"].T + model["b1"]
        H_relu = np.maximum(H, 0)
        logits = H_relu @ model["W2"].T + model["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return np.mean(np.argmax(probs, axis=1) == yb)

    losses_before = np.zeros(n)
    grads_list = []

    for i in range(n):
        model = agent_models[i]
        W1, b1 = model["W1"], model["b1"]
        W2, b2 = model["W2"], model["b2"]

        Xb = batches_X[i]
        y_onehot = batches_yoh[i]

        H = Xb @ W1.T + b1
        H_relu = np.maximum(H, 0)

        logits = H_relu @ W2.T + b2
        logits -= logits.max(axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
        losses_before[i] = loss

        dlogits = (probs - y_onehot) / batch_size

        dW2 = dlogits.T @ H_relu
        db2 = dlogits.sum(axis=0)

        dH = dlogits @ W2
        dH[H <= 0] = 0

        dW1 = dH.T @ Xb
        db1 = dH.sum(axis=0)

        grads_list.append({
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        })

    rewards = np.zeros(n)

    for i in range(n):
        a = actions[i]
        g = grads_list[i]
        model_i = agent_models[i]

        if a == 0:
            Xb = batches_X[i]
            y_onehot = batches_yoh[i]

            H_before = Xb @ model_i["W1"].T + model_i["b1"]
            H_relu_before = np.maximum(H_before, 0)
            logits_before = H_relu_before @ model_i["W2"].T + model_i["b2"]

            expL = np.exp(logits_before - logits_before.max(axis=1, keepdims=True))
            probs = expL / expL.sum(axis=1, keepdims=True)

            loss_before = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

            flat_grad_before = np.concatenate([
                g["dW1"].ravel(),
                g["db1"].ravel(),
                g["dW2"].ravel(),
                g["db2"].ravel(),
            ])

            model_i["W1"] -= lr_compute * g["dW1"]
            model_i["b1"] -= lr_compute * g["db1"]
            model_i["W2"] -= lr_compute * g["dW2"]
            model_i["b2"] -= lr_compute * g["db2"]

            H_after = Xb @ model_i["W1"].T + model_i["b1"]
            H_relu_after = np.maximum(H_after, 0)
            logits_after = H_relu_after @ model_i["W2"].T + model_i["b2"]

            expL2 = np.exp(logits_after - logits_after.max(axis=1, keepdims=True))
            probs2 = expL2 / expL2.sum(axis=1, keepdims=True)

            loss_after = -np.mean(np.sum(y_onehot * np.log(probs2 + 1e-9), axis=1))
            loss_delta = loss_before - loss_after

            dlogits_after = (probs2 - y_onehot) / batch_size
            dW2_after = dlogits_after.T @ H_relu_after
            db2_after = dlogits_after.sum(axis=0)

            dH_after = dlogits_after @ model_i["W2"]
            dH_after[H_after <= 0] = 0

            dW1_after = dH_after.T @ Xb
            db1_after = dH_after.sum(axis=0)

            flat_grad_after = np.concatenate([
                dW1_after.ravel(),
                db1_after.ravel(),
                dW2_after.ravel(),
                db2_after.ravel(),
            ])
            cos_align = np.dot(flat_grad_before, flat_grad_after)
            norm_before = np.linalg.norm(flat_grad_before)
            norm_after = np.linalg.norm(flat_grad_after)
            cos_align /= (norm_before * norm_after + 1e-9)
            rewards[i] = cos_align * loss_delta

        elif a == 1:
            Xb = batches_X[i]
            y_onehot = batches_yoh[i]
            model_i = agent_models[i]

            # Run a local SGD step (same as compute) so communication never skips learning.
            H_before = Xb @ model_i["W1"].T + model_i["b1"]
            H_relu_before = np.maximum(H_before, 0)
            logits_before = H_relu_before @ model_i["W2"].T + model_i["b2"]

            expL = np.exp(logits_before - logits_before.max(axis=1, keepdims=True))
            probs = expL / expL.sum(axis=1, keepdims=True)

            loss_before = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

            g = grads_list[i]
            flat_grad_before = np.concatenate([
                g["dW1"].ravel(),
                g["db1"].ravel(),
                g["dW2"].ravel(),
                g["db2"].ravel(),
            ])

            model_i["W1"] -= lr_comm * g["dW1"]
            model_i["b1"] -= lr_comm * g["db1"]
            model_i["W2"] -= lr_comm * g["dW2"]
            model_i["b2"] -= lr_comm * g["db2"]

            H_after = Xb @ model_i["W1"].T + model_i["b1"]
            H_relu_after = np.maximum(H_after, 0)
            logits_after = H_relu_after @ model_i["W2"].T + model_i["b2"]

            expL2 = np.exp(logits_after - logits_after.max(axis=1, keepdims=True))
            probs2 = expL2 / expL2.sum(axis=1, keepdims=True)

            loss_after = -np.mean(np.sum(y_onehot * np.log(probs2 + 1e-9), axis=1))
            loss_delta = loss_before - loss_after

            dlogits_after = (probs2 - y_onehot) / batch_size
            dW2_after = dlogits_after.T @ H_relu_after
            db2_after = dlogits_after.sum(axis=0)

            dH_after = dlogits_after @ model_i["W2"]
            dH_after[H_after <= 0] = 0

            dW1_after = dH_after.T @ Xb
            db1_after = dH_after.sum(axis=0)

            flat_grad_after = np.concatenate([
                dW1_after.ravel(),
                db1_after.ravel(),
                dW2_after.ravel(),
                db2_after.ravel(),
            ])
            cos_align = np.dot(flat_grad_before, flat_grad_after)
            norm_before = np.linalg.norm(flat_grad_before)
            norm_after = np.linalg.norm(flat_grad_after)
            cos_align /= (norm_before * norm_after + 1e-9)
            grad_reward = cos_align * loss_delta

            nbrs = neighbors[i]
            partners = [i] + nbrs

            avg_params = {}
            for key in model_i.keys():
                stacked = np.stack([agent_models[idx][key] for idx in partners], axis=0)
                avg_params[key] = stacked.mean(axis=0)

            for idx_partner in partners:
                for key, value in avg_params.items():
                    agent_models[idx_partner][key] = value.copy()

            rewards[i] = grad_reward

        else:
            # Temporarily disable rest rewards for testing; previously 0.1 * (1 - e_new[i]).
            rewards[i] = 0.0

    # Preserve relative reward magnitudes while keeping utilities in a healthy band.
    r = np.tanh(rewards) + 0.2
    r = np.clip(r, -1.0, 1.0)

    u_rel_raw = compute_relational_utility(r, W, neighbors)
    u_rel = 0.5 * u + 0.5 * u_rel_raw
    u_rel = np.clip(u_rel, 0.0, 2.0)
    W = update_trust(W, u_rel, neighbors, eta=0.05)

    inflow, outflow, delta = compute_flows(c, u_rel, W, neighbors, cfg)

    if cfg.alpha_decay > 0:
        r_regen = cfg.lambda_reg * (1 - e_new)
        c_next = (1 - cfg.alpha_decay) * c + inflow - outflow + r_regen
    else:
        c_next = c + inflow - outflow

    total_before = c.sum()
    total_after = c_next.sum()
    drift = total_after - total_before

    if abs(drift) > 1e-12:
        c_next -= drift / len(c_next)

    if t % 200 == 0:
        print(f"\n========== DEBUG STEP {t} ==========")
        print("u_rel per agent:", np.round(u_rel, 3))
        print("u_rel variance:", round(np.var(u_rel), 6))
        print("u_rel (min,max):", round(u_rel.min(), 3), round(u_rel.max(), 3))
        print("c:", np.round(c, 3))
        print("inflow:", np.round(inflow, 4))
        print("outflow:", np.round(outflow, 4))
        print("Δc:", np.round(inflow - outflow, 4))
        print("Total Δc:", round((inflow - outflow).sum(), 6))
        print("Trust row-mean:", round(W.sum(axis=1).mean(), 5))
        print("Trust col-mean:", round(W.sum(axis=0).mean(), 5))
        print("delta sample:")
        print(np.round(delta[:3, :3], 4))

    c_next = np.maximum(c_next, 0)

    return c_next, u_rel, e_new, inflow, outflow, delta, r, W, agent_models