from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import run_feasibility as base
import run_f1_f2 as f2


OUT = base.ROOT / "results" / "optimizer_stability"
N_GRID = (100, 250, 500)
R_GRID = (2, 4, 8)
K_FRACTIONS = (0.02, 0.05)
DELTAS = (0.03, 0.05, 0.10)
SEEDS = 20
MULTISTARTS = 10
MILP_LIMIT = 60.0
MASTER_SEED = 20261022


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_instance(n: int, r: int, seed: int):
    rng_graph = np.random.default_rng(MASTER_SEED + 1_000_003 * seed + 101 * n + r)
    start = time.perf_counter()
    A, planted = base.generate_directed_sbm(n, max(8, r), 6, rng_graph)
    labels = base.coarsen_labels(planted, r)
    degree = np.asarray(A.sum(axis=1)).ravel()
    gamma = base.gamma_vector(
        "lognormal_moderate", degree,
        np.random.default_rng(MASTER_SEED + 2_000_003 * seed + n + 17 * r),
    )
    W = base.make_W(labels)
    H, iterations, residual = base.solve_h(A, degree, gamma, W, tol=1e-11)
    innate = np.random.default_rng(
        MASTER_SEED + 3_000_017 * seed + 3 * n + 31 * r
    ).uniform(0.0, 0.8, n)
    raw = (1.0 - innate)[:, None] * H
    proportions = np.bincount(labels, minlength=r).astype(float) / n
    efficiency = raw @ proportions
    gain_seconds = time.perf_counter() - start
    return raw, efficiency, gain_seconds, iterations, residual


def normalized_gain(raw: np.ndarray, k: int):
    tau = np.array([
        np.partition(raw[:, j], -k)[-k:].sum() for j in range(raw.shape[1])
    ])
    return raw / np.maximum(tau, 1e-15)[None, :]


def repair_efficiency(selected, efficiency, threshold, k):
    selected = np.asarray(selected, dtype=np.int32).copy()
    inside = np.zeros(len(efficiency), dtype=bool)
    inside[selected] = True
    total = float(efficiency[selected].sum())
    while total < threshold - 1e-12:
        outside = np.flatnonzero(~inside)
        old_pos = int(np.argmin(efficiency[selected]))
        old = int(selected[old_pos])
        new = int(outside[np.argmax(efficiency[outside])])
        improvement = float(efficiency[new] - efficiency[old])
        if improvement <= 1e-15:
            return base.topk_indices(efficiency, k).astype(np.int32)
        inside[old] = False
        inside[new] = True
        selected[old_pos] = new
        total += improvement
    return selected


def local_search_from(G, efficiency, threshold, selected, max_passes=12):
    selected = repair_efficiency(selected, efficiency, threshold, len(selected))
    n = G.shape[0]
    inside = np.zeros(n, dtype=bool)
    inside[selected] = True
    totals = G[selected].sum(axis=0)
    eff_total = float(efficiency[selected].sum())
    current = float(totals.min())
    for _ in range(max_passes):
        best = None
        outside = np.flatnonzero(~inside)
        # Preserve the current implementation's path rule: the last selected
        # position with an improving best swap becomes the next move.
        for pos, old in enumerate(selected):
            candidate_eff = eff_total - efficiency[old] + efficiency[outside]
            feasible = candidate_eff >= threshold - 1e-12
            if not np.any(feasible):
                continue
            candidates = outside[feasible]
            candidate_totals = totals[None, :] - G[old] + G[candidates]
            scores = candidate_totals.min(axis=1)
            q = int(np.argmax(scores))
            if float(scores[q]) > current + 1e-12:
                best = (float(scores[q]), pos, int(old), int(candidates[q]),
                        candidate_totals[q], float(candidate_eff[feasible][q]))
        if best is None:
            break
        current, pos, old, new, totals, eff_total = best
        inside[old] = False
        inside[new] = True
        selected[pos] = new
    return selected


def initial_sets(G, efficiency, threshold, k, count, rng):
    starts = [base.topk_indices(efficiency, k).astype(np.int32)]
    # Deterministic max-min greedy is included as one alternative start.
    starts.append(repair_efficiency(base.fair_greedy(G, k), efficiency, threshold, k))
    eff_scaled = efficiency / max(float(efficiency.max()), 1e-15)
    for q in range(max(0, count - len(starts))):
        group_weights = rng.dirichlet(np.ones(G.shape[1]))
        fairness_score = G @ group_weights
        fairness_score /= max(float(fairness_score.max()), 1e-15)
        lam = q / max(count - 3, 1)
        score = (1.0 - lam) * fairness_score + lam * eff_scaled
        selected = base.topk_indices(score, k).astype(np.int32)
        starts.append(repair_efficiency(selected, efficiency, threshold, k))
    return starts[:count]


def optimize_single(G, efficiency, threshold, k):
    start = time.perf_counter()
    selected = f2.constrained_local_search(G, efficiency, threshold, k)[0]
    return selected, time.perf_counter() - start


def optimize_multistart(G, efficiency, threshold, k, rng):
    start = time.perf_counter()
    best_selected, best_value = None, -np.inf
    for initial in initial_sets(G, efficiency, threshold, k, MULTISTARTS, rng):
        selected = local_search_from(G, efficiency, threshold, initial)
        value = float(G[selected].sum(axis=0).min())
        if value > best_value:
            best_value = value
            best_selected = selected.copy()
    return best_selected, time.perf_counter() - start


def evaluate(selected, G, efficiency, threshold):
    value = float(G[selected].sum(axis=0).min())
    eff = float(efficiency[selected].sum())
    return value, eff, bool(eff >= threshold - 1e-10)


def summarize(rows):
    summary = []
    methods = sorted(set(row["method"] for row in rows))
    for method in methods:
        subset = [row for row in rows if row["method"] == method and row["oracle_exact"]]
        regrets = np.asarray([row["normalized_regret"] for row in subset], float)
        ratios = np.asarray([row["optimization_fraction"] for row in subset], float)
        feasible = np.asarray([row["feasible"] for row in subset], bool)
        summary.append({
            "method": method, "exact_oracle_trials": len(subset),
            "median_regret": float(np.median(regrets)),
            "p90_regret": float(np.quantile(regrets, 0.90)),
            "p95_regret": float(np.quantile(regrets, 0.95)),
            "max_regret": float(regrets.max()),
            "rate_regret_gt_1pct": float(np.mean(regrets > 0.01)),
            "rate_regret_gt_2pct": float(np.mean(regrets > 0.02)),
            "rate_regret_gt_5pct": float(np.mean(regrets > 0.05)),
            "feasibility_rate": float(feasible.mean()),
            "median_runtime_seconds": float(np.median([row["runtime_seconds"] for row in subset])),
            "median_optimization_fraction": float(np.median(ratios)),
            "p95_optimization_fraction": float(np.quantile(ratios, 0.95)),
            "passes_regret_gate": bool(np.median(regrets) <= 0.01 and np.quantile(regrets, 0.95) <= 0.02),
            "passes_feasibility_gate": bool(feasible.mean() >= 0.99),
            "passes_time_gate_median": bool(np.median(ratios) <= 0.50),
        })
    return summary


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    n_grid = (100,) if smoke else N_GRID
    r_grid = (2, 4) if smoke else R_GRID
    seeds = 2 if smoke else SEEDS
    rows = []
    start_all = time.perf_counter()
    for n in n_grid:
        for r in r_grid:
            for seed in range(seeds):
                raw, efficiency, gain_seconds, iterations, residual = make_instance(n, r, seed)
                for k_fraction in K_FRACTIONS:
                    k = max(2, int(round(n * k_fraction)))
                    G = normalized_gain(raw, k)
                    efficiency_optimum = float(efficiency[base.topk_indices(efficiency, k)].sum())
                    for delta in DELTAS:
                        threshold = (1.0 - delta) * efficiency_optimum
                        oracle_sel, oracle_result, oracle_time = f2.constrained_milp(
                            G, efficiency, threshold, k, time_limit=MILP_LIMIT
                        )
                        gap = f2.result_gap(oracle_result)
                        oracle_exact = bool(oracle_result.x is not None and gap is not None and gap <= 1e-8)
                        if oracle_result.x is None:
                            continue
                        oracle_value, oracle_eff, oracle_feasible = evaluate(
                            oracle_sel, G, efficiency, threshold
                        )
                        methods = [("MILP", oracle_sel, oracle_time)]
                        sel, elapsed = optimize_single(G, efficiency, threshold, k)
                        methods.append(("SingleStart-LS", sel, elapsed))
                        sel, elapsed = optimize_multistart(
                            G, efficiency, threshold, k,
                            np.random.default_rng(MASTER_SEED + seed + 13*n + 97*r + k),
                        )
                        methods.append((f"MultiStart{MULTISTARTS}-LS", sel, elapsed))
                        for method, selected, elapsed in methods:
                            value, eff_value, feasible = evaluate(selected, G, efficiency, threshold)
                            regret = max(0.0, (oracle_value - value) / max(oracle_value, 1e-15))
                            rows.append({
                                "n": n, "r": r, "seed": seed, "k": k,
                                "k_fraction": k_fraction, "delta": delta,
                                "method": method, "fair_value": value,
                                "oracle_value": oracle_value, "normalized_regret": regret,
                                "efficiency_value": eff_value, "efficiency_threshold": threshold,
                                "feasible": feasible, "runtime_seconds": elapsed,
                                "gain_computation_seconds": gain_seconds,
                                "optimization_fraction": elapsed / (gain_seconds + elapsed),
                                "oracle_exact": oracle_exact, "oracle_mip_gap": gap,
                                "oracle_status": int(oracle_result.status),
                                "oracle_runtime_seconds": oracle_time,
                                "reference_iterations": iterations,
                                "reference_residual": residual,
                            })
                print(f"completed n={n} r={r} seed={seed}", flush=True)
    summary = summarize(rows)
    suffix = "_smoke" if smoke else ""
    write_csv(OUT / f"optimizer_trials{suffix}.csv", rows)
    write_csv(OUT / f"optimizer_summary{suffix}.csv", summary)
    metadata = {
        "smoke": smoke, "wall_seconds": time.perf_counter() - start_all,
        "trial_rows": len(rows), "instance_scenarios": len(rows)//3,
        "n_grid": n_grid, "r_grid": r_grid, "k_fractions": K_FRACTIONS,
        "deltas": DELTAS, "seeds": seeds, "multistarts": MULTISTARTS,
        "milp_limit_seconds": MILP_LIMIT,
    }
    (OUT / f"metadata{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
