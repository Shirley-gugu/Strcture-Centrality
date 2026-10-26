from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import run_feasibility as base


OUT = base.ROOT / "results" / "f1_f2"
DELTAS = (0.0, 0.01, 0.03, 0.05, 0.10)


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_n250():
    n = 250
    A, planted = base.generate_directed_sbm(
        n, 8, 6, np.random.default_rng(base.SEED + n)
    )
    degree = np.asarray(A.sum(axis=1)).ravel()
    gamma = base.gamma_vector(
        "lognormal_moderate", degree, np.random.default_rng(base.SEED + 2 * n)
    )
    labels = base.coarsen_labels(planted, 4)
    W = base.make_W(labels)
    H, iterations, residual = base.solve_h(A, degree, gamma, W, tol=1e-11)
    innate = np.random.default_rng(base.SEED + 3 * n).uniform(0.0, 0.8, n)
    raw = (1.0 - innate)[:, None] * H
    k = max(5, min(20, n // 20))
    tau = np.array([np.sort(raw[:, j])[-k:].sum() for j in range(raw.shape[1])])
    G = raw / np.maximum(tau, 1e-15)[None, :]

    Fest = base.forest_estimate_shared(
        base.adjacency_lists(A), gamma, labels, 80, base.SEED + 4 * n
    )
    Hest = Fest
    for _ in range(8):
        Hest = base.h_step_block(A, degree, gamma, Hest, W)
    raw_est = (1.0 - innate)[:, None] * Hest
    Gest = raw_est / np.maximum(tau, 1e-15)[None, :]

    proportions = np.bincount(labels, minlength=G.shape[1]).astype(float) / n
    efficiency = raw @ proportions
    efficiency_est = raw_est @ proportions
    return {
        "A": A, "degree": degree, "gamma": gamma, "labels": labels,
        "W": W, "H": H, "Hest": Hest, "raw": raw, "raw_est": raw_est,
        "G": G, "Gest": Gest, "efficiency": efficiency,
        "efficiency_est": efficiency_est, "k": k, "tau": tau,
        "iterations": iterations, "residual": residual,
    }


def constrained_milp(G_opt, efficiency_constraint, threshold, k, time_limit=120.0):
    n, r = G_opt.shape
    c = np.r_[np.zeros(n), -1.0]
    rows, lbs, ubs = [], [], []
    for j in range(r):
        rows.append(np.r_[-G_opt[:, j], 1.0])
        lbs.append(-np.inf)
        ubs.append(0.0)
    rows.append(np.r_[np.ones(n), 0.0])
    lbs.append(-np.inf)
    ubs.append(float(k))
    rows.append(np.r_[-efficiency_constraint, 0.0])
    lbs.append(-np.inf)
    ubs.append(-float(threshold))
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lbs), np.asarray(ubs))
    bounds = Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf])
    integrality = np.r_[np.ones(n, dtype=int), 0]
    start = time.perf_counter()
    result = milp(
        c, integrality=integrality, bounds=bounds, constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 1e-9},
    )
    elapsed = time.perf_counter() - start
    if result.x is None:
        return np.array([], dtype=np.int32), result, elapsed
    return np.flatnonzero(result.x[:n] > 0.5), result, elapsed


def constrained_local_search(G_opt, efficiency_constraint, threshold, k, max_passes=12):
    selected = base.topk_indices(efficiency_constraint, k).astype(np.int32)
    n = G_opt.shape[0]
    inside = np.zeros(n, dtype=bool)
    inside[selected] = True
    group_totals = G_opt[selected].sum(axis=0)
    eff_total = float(efficiency_constraint[selected].sum())
    current = float(group_totals.min())
    start = time.perf_counter()
    for _ in range(max_passes):
        best = None
        outside = np.flatnonzero(~inside)
        for pos, old in enumerate(selected):
            candidate_eff = eff_total - efficiency_constraint[old] + efficiency_constraint[outside]
            feasible = candidate_eff >= threshold - 1e-12
            if not np.any(feasible):
                continue
            candidates = outside[feasible]
            candidate_totals = group_totals[None, :] - G_opt[old] + G_opt[candidates]
            scores = candidate_totals.min(axis=1)
            q = int(np.argmax(scores))
            score = float(scores[q])
            if score > current + 1e-12:
                best = (score, pos, int(old), int(candidates[q]),
                        candidate_totals[q], float(candidate_eff[feasible][q]))
        if best is None:
            break
        current, pos, old, new, group_totals, eff_total = best
        inside[old] = False
        inside[new] = True
        selected[pos] = new
    return selected, time.perf_counter() - start


def evaluate(selected, G_true, efficiency_true, efficiency_optimum):
    gains = G_true[selected].sum(axis=0)
    retention = float(efficiency_true[selected].sum() / efficiency_optimum)
    bottleneck = int(np.argmin(gains))
    return {
        "retention": retention,
        "fair_value": float(gains[bottleneck]),
        "bottleneck": bottleneck,
        "gains": gains,
    }


def result_gap(result):
    value = getattr(result, "mip_gap", None)
    return None if value is None else float(value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = reconstruct_n250()
    G, Gest = d["G"], d["Gest"]
    eff, eff_est, k = d["efficiency"], d["efficiency_est"], d["k"]
    total_topk = base.topk_indices(eff, k)
    total_optimum = float(eff[total_topk].sum())

    f1_rows = []
    f1_solutions = {}
    for delta in DELTAS:
        threshold = (1.0 - delta) * total_optimum
        selected, result, elapsed = constrained_milp(G, eff, threshold, k)
        ev = evaluate(selected, G, eff, total_optimum)
        f1_solutions[delta] = selected
        f1_rows.append({
            "delta": delta,
            "efficiency_threshold": threshold,
            "efficiency_retention": ev["retention"],
            "min_normalized_group_gain": ev["fair_value"],
            "bottleneck_group": ev["bottleneck"],
            "group_0": ev["gains"][0], "group_1": ev["gains"][1],
            "group_2": ev["gains"][2], "group_3": ev["gains"][3],
            "runtime_seconds": elapsed, "solver_success": bool(result.success),
            "solver_status": int(result.status), "mip_gap": result_gap(result),
            "message": str(result.message),
        })
    write_csv(OUT / "f1_pareto.csv", f1_rows)

    delta = 0.05
    threshold = (1.0 - delta) * total_optimum
    # A: true coefficients + exact optimization.
    sel_A, res_A, time_A = constrained_milp(G, eff, threshold, k)
    # B: estimated group gains + exact optimization; true efficiency domain.
    sel_B, res_B, time_B = constrained_milp(Gest, eff, threshold, k)
    # C: true gains + current constrained local-search optimizer.
    sel_C, time_C = constrained_local_search(G, eff, threshold, k)
    # D: estimated gains + current optimizer; true efficiency domain.
    sel_D, time_D = constrained_local_search(Gest, eff, threshold, k)
    configurations = [
        ("A_true_exact", sel_A, time_A, res_A),
        ("B_estimated_exact", sel_B, time_B, res_B),
        ("C_true_approx", sel_C, time_C, None),
        ("D_estimated_approx", sel_D, time_D, None),
    ]
    A_eval = evaluate(sel_A, G, eff, total_optimum)
    f2_rows = []
    for name, selected, elapsed, solver in configurations:
        ev = evaluate(selected, G, eff, total_optimum)
        est_gains = Gest[selected].sum(axis=0)
        f2_rows.append({
            "configuration": name,
            "true_fair_value": ev["fair_value"],
            "true_fair_regret": A_eval["fair_value"] - ev["fair_value"],
            "normalized_fair_regret": (A_eval["fair_value"] - ev["fair_value"]) / A_eval["fair_value"],
            "true_efficiency_retention": ev["retention"],
            "true_efficiency_feasible": ev["retention"] >= 0.95 - 1e-10,
            "estimated_efficiency_retention": float(eff_est[selected].sum() / np.sort(eff_est)[-k:].sum()),
            "true_bottleneck": ev["bottleneck"],
            "estimated_bottleneck": int(np.argmin(est_gains)),
            "group_0": ev["gains"][0], "group_1": ev["gains"][1],
            "group_2": ev["gains"][2], "group_3": ev["gains"][3],
            "selection_overlap_with_A": len(set(selected) & set(sel_A)) / k,
            "runtime_seconds": elapsed,
            "mip_gap": "" if solver is None else result_gap(solver),
            "solver_success": "" if solver is None else bool(solver.success),
        })
    write_csv(OUT / "f2_decomposition.csv", f2_rows)

    # Counterfactual: estimated efficiency coefficients are used operationally.
    est_total_optimum = float(np.sort(eff_est)[-k:].sum())
    est_threshold = 0.95 * est_total_optimum
    sel_BE, res_BE, time_BE = constrained_milp(Gest, eff_est, est_threshold, k)
    sel_DE, time_DE = constrained_local_search(Gest, eff_est, est_threshold, k)
    counter_rows = []
    for name, selected, elapsed, solver in (
        ("B_estimated_exact_estimated_constraint", sel_BE, time_BE, res_BE),
        ("D_estimated_approx_estimated_constraint", sel_DE, time_DE, None),
    ):
        ev = evaluate(selected, G, eff, total_optimum)
        counter_rows.append({
            "configuration": name,
            "true_efficiency_retention": ev["retention"],
            "true_efficiency_feasible": ev["retention"] >= 0.95 - 1e-10,
            "true_fair_value": ev["fair_value"],
            "true_fair_regret_vs_A": A_eval["fair_value"] - ev["fair_value"],
            "true_bottleneck": ev["bottleneck"],
            "runtime_seconds": elapsed,
            "mip_gap": "" if solver is None else result_gap(solver),
        })
    write_csv(OUT / "f2_constraint_misjudgment.csv", counter_rows)

    abs_H = np.abs(d["Hest"] - d["H"])
    abs_G = np.abs(Gest - G)
    flat = np.sort(abs_G.ravel())[::-1]
    top_count = max(1, int(np.ceil(0.01 * len(flat))))
    union_nodes = np.array(sorted(set(sel_A) | set(sel_B) | set(sel_C) | set(sel_D)), dtype=int)
    diagnostics = {
        "n": 250, "r": 4, "k": k, "delta": delta,
        "reference_iterations": d["iterations"],
        "reference_relative_step": d["residual"],
        "max_raw_H_entry_error": float(abs_H.max()),
        "max_normalized_G_entry_error": float(abs_G.max()),
        "mean_normalized_G_entry_error": float(abs_G.mean()),
        "top_1pct_error_mass_fraction": float(flat[:top_count].sum() / max(flat.sum(), 1e-15)),
        "max_G_error_on_any_selected_node": float(abs_G[union_nodes].max()),
        "selected_union_node_count": int(len(union_nodes)),
        "true_efficiency_optimum": total_optimum,
        "estimated_efficiency_optimum": est_total_optimum,
        "efficiency_optimum_relative_error": (est_total_optimum - total_optimum) / total_optimum,
    }
    with (OUT / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps({"f1": f1_rows, "f2": f2_rows,
                      "constraint_counterfactual": counter_rows,
                      "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
