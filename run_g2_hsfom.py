from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import bicgstab, gmres

import run_f1_f2 as f2
import run_g1_pokec as g1
import run_optimizer_stability_audit as audit


OUT = g1.ROOT / "results" / "g2_hsfom"
R_GRID = (4, 8)
DELTAS = (0.05, 0.10)
GAMMA_NAMES = ("identity", "random_lognormal")
FOREST_SEED = 20261021
SHARED_S = 20
SHARED_K = 5
STATIONARY_K = 36
PER_LIST = 250
MILP_LIMIT = 300.0


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    q = min(count, len(values))
    if q == len(values):
        return np.arange(len(values), dtype=np.int64)
    return np.argpartition(values, -q)[-q:].astype(np.int64)


def candidate_pool(G: np.ndarray, efficiency: np.ndarray,
                   per_list: int = PER_LIST) -> np.ndarray:
    parts = [top_indices(G[:, j], per_list) for j in range(G.shape[1])]
    parts.append(top_indices(G.mean(axis=1), per_list))
    parts.append(top_indices(efficiency, per_list))
    return np.unique(np.concatenate(parts))


def efficiency_optimum(efficiency: np.ndarray, k: int) -> float:
    return float(efficiency[top_indices(efficiency, k)].sum())


def constrained_lp_upper_bound(G: np.ndarray, efficiency: np.ndarray,
                               threshold: float, k: int):
    """Full-network LP relaxation of the efficiency-constrained max-min problem."""
    n, r = G.shape
    c = np.r_[np.zeros(n), -1.0]
    rows = [
        sparse.hstack([
            -sparse.csr_matrix(G[:, j][None, :]),
            sparse.csr_matrix([[1.0]]),
        ])
        for j in range(r)
    ]
    rows.append(sparse.hstack([
        sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix([[0.0]])
    ]))
    rows.append(sparse.hstack([
        -sparse.csr_matrix(efficiency[None, :]), sparse.csr_matrix([[0.0]])
    ]))
    A_ub = sparse.vstack(rows, format="csr")
    b_ub = np.r_[np.zeros(r), float(k), -float(threshold)]
    start = time.perf_counter()
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        bounds=[(0.0, 1.0)] * n + [(None, None)], method="highs",
    )
    elapsed = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"full-network LP failed: {result.message}")
    return float(-result.fun), elapsed


def exact_krylov_augmented(A, degree, gamma, W, rtol=1e-10):
    """High-accuracy block reference with a checked GMRES breakdown fallback."""
    M = sparse.diags(gamma + degree, format="csc") - A.T.tocsc()
    H = np.empty_like(W)
    iterations, solvers = [], []
    start = time.perf_counter()
    for j in range(W.shape[1]):
        count = [0]

        def callback(_):
            count[0] += 1

        u, info = bicgstab(
            M, W[:, j], rtol=rtol, atol=0.0, maxiter=5000,
            callback=callback,
        )
        solver = "bicgstab"
        if info != 0:
            count[0] = 0
            u, info = gmres(
                M, W[:, j], rtol=rtol, atol=0.0, restart=100,
                maxiter=5000, callback=callback, callback_type="legacy",
            )
            solver = "gmres"
        if info != 0:
            raise RuntimeError(f"reference solve failed for column {j}: info={info}")
        H[:, j] = gamma * u
        iterations.append(count[0])
        solvers.append(solver)
    elapsed = time.perf_counter() - start
    residual = np.linalg.norm((M @ (H / gamma[:, None])) - W) / np.linalg.norm(W)
    if not np.isfinite(residual) or residual > 5e-9:
        raise RuntimeError(f"reference residual too large: {residual}")
    return H, elapsed, iterations, float(residual), solvers


def gap(upper: float, value: float) -> float:
    if upper <= 1e-15:
        return 0.0
    return max(0.0, float((upper - value) / upper))


def evaluate(selected: np.ndarray, G: np.ndarray, efficiency: np.ndarray,
             efficiency_star: float, delta: float) -> dict:
    fair_value = float(G[selected].sum(axis=0).min())
    efficiency_value = float(efficiency[selected].sum())
    retention = efficiency_value / max(efficiency_star, 1e-15)
    return {
        "fair_value": fair_value,
        "efficiency_value": efficiency_value,
        "efficiency_retention": retention,
        "true_feasible": bool(retention >= 1.0 - delta - 1e-10),
        "true_violation": max(0.0, (1.0 - delta) - retention),
    }


def optimize_pool(G_opt: np.ndarray, efficiency_opt: np.ndarray,
                  threshold: float, k: int, pool: np.ndarray, seed: int):
    start = time.perf_counter()
    selected_local, search_seconds = audit.optimize_multistart(
        G_opt[pool], efficiency_opt[pool], threshold, k,
        np.random.default_rng(seed),
    )
    return pool[selected_local], time.perf_counter() - start, search_seconds


def candidate_milp(G: np.ndarray, efficiency: np.ndarray, threshold: float,
                   k: int, pool: np.ndarray):
    selected_local, result, elapsed = f2.constrained_milp(
        G[pool], efficiency[pool], threshold, k, time_limit=MILP_LIMIT
    )
    if result.x is None:
        raise RuntimeError(f"candidate MILP has no incumbent: {result.message}")
    selected = pool[selected_local]
    mip_gap = getattr(result, "mip_gap", None)
    exact = mip_gap is not None and float(mip_gap) <= 1e-8
    return selected, float(mip_gap) if mip_gap is not None else None, exact, elapsed


def forest_initializations(A, gamma, group_data, n: int, gamma_index: int):
    roots_list = []
    forest_seconds = 0.0
    for sample_index in range(1, SHARED_S + 1):
        start = time.perf_counter()
        roots = g1.sample_forest(
            A.indptr, A.indices, gamma,
            FOREST_SEED + sample_index + 100000 * gamma_index,
        )
        forest_seconds += time.perf_counter() - start
        roots_list.append(roots)

    outputs = {}
    for r, (labels, W, sizes, _) in group_data.items():
        cumulative = np.zeros((n, r + 1), np.float64)
        stat_start = time.perf_counter()
        for roots in roots_list:
            g1.add_group_stat(cumulative[:, :r], roots, labels, sizes)
            cumulative[:, r] += np.bincount(roots, minlength=n) / float(n)
        stat_seconds = time.perf_counter() - stat_start
        W_aug = np.column_stack([W, np.full(n, 1.0 / n)])
        refine_start = time.perf_counter()
        H = g1.refine(A, np.asarray(A.sum(axis=1)).ravel(), gamma,
                      cumulative / SHARED_S, W_aug, SHARED_K)
        refine_seconds = time.perf_counter() - refine_start
        outputs[r] = (
            H, forest_seconds + stat_seconds + refine_seconds,
            forest_seconds, stat_seconds, refine_seconds,
        )
    return outputs


def make_estimates(A, degree, gamma, group_data, n: int, gamma_index: int):
    shared = forest_initializations(A, gamma, group_data, n, gamma_index)
    estimates = {}
    for r, (_, W, _, _) in group_data.items():
        W_aug = np.column_stack([W, np.full(n, 1.0 / n)])
        start = time.perf_counter()
        H_stationary = g1.refine(
            A, degree, gamma, np.zeros_like(W_aug), W_aug, STATIONARY_K
        )
        stationary_seconds = time.perf_counter() - start
        estimates[r] = {
            "Shared-IFR": {
                "H": shared[r][0], "estimation_seconds": shared[r][1],
                "forest_seconds": shared[r][2], "statistic_seconds": shared[r][3],
                "refinement_seconds": shared[r][4],
                "S": SHARED_S, "K": SHARED_K,
            },
            "Batched-Stationary": {
                "H": H_stationary, "estimation_seconds": stationary_seconds,
                "forest_seconds": 0.0, "statistic_seconds": 0.0,
                "refinement_seconds": stationary_seconds,
                "S": 0, "K": STATIONARY_K,
            },
        }
    return estimates


def make_report(rows: list[dict], references: list[dict], metadata: dict):
    lines = [
        f"# G2: Full HS-FOM on soc-Pokec-{metadata['nodes']}",
        "",
        "## Material Passport",
        "",
        "- Verification status: COMPLETED; internal consistency checks passed",
        f"- Node budget: k={metadata['k']}",
        f"- Candidate rule: Top-{PER_LIST} per group, average group gain, and efficiency",
        f"- Shared estimator: S={SHARED_S}, K={SHARED_K}, forest seed={FOREST_SEED}",
        f"- Batched stationary baseline: K={STATIONARY_K}",
        "- Optimizer: Multi-start-10 one-swap local search for every evaluated method",
        "- Full-network certificate: constrained LP over all nodes with true coefficients",
        "",
        "## Candidate/reference audit",
        "",
        "| Gamma | r | delta | |C_true| | candidate MILP exact | search regret | screening certificate | reference certified gap |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for q in references:
        lines.append(
            f"| {q['gamma']} | {q['r']} | {100*q['delta']:.0f}% | "
            f"{q['candidate_pool']} | {q['candidate_milp_exact']} | "
            f"{100*q['reference_search_regret']:.3f}% | "
            f"{100*q['candidate_screening_certificate']:.3f}% | "
            f"{100*q['reference_certified_lp_gap']:.3f}% |"
        )
    lines += [
        "",
        "The screening certificate is conservative because its numerator uses the full-network LP upper bound. It isolates screening from local-search error by using the exact candidate-pool MILP value as the feasible lower bound.",
        "",
        "## End-to-end results",
        "",
        "| Gamma | r | delta | Method | |C_hat| | true efficiency retained | violation | true fair value | reference gap | certified LP gap | total time (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in rows:
        cert = "--" if q["certified_lp_gap"] == "" else f"{100*float(q['certified_lp_gap']):.3f}%"
        lines.append(
            f"| {q['gamma']} | {q['r']} | {100*q['delta']:.0f}% | {q['method']} | "
            f"{q['candidate_pool']} | {100*q['true_efficiency_retention']:.3f}% | "
            f"{100*q['true_efficiency_violation']:.3f}% | {q['true_fair_value']:.6g} | "
            f"{100*q['reference_relative_gap']:.3f}% | {cert} | {q['end_to_end_seconds']:.3f} |"
        )
    lines += [
        "",
        "A certified LP gap is reported only for sets that satisfy the true full-network efficiency constraint. Reference-relative gap compares with the common exact-coefficient candidate procedure and is not a global optimality claim.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(n: int, k: int):
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    start_all = time.perf_counter()
    A, gender, age = g1.load_pokec(n)
    degree = np.asarray(A.sum(axis=1)).ravel()
    group_data = {r: g1.make_groups(gender, age, r) for r in R_GRID}
    rng = np.random.default_rng(g1.SEED)
    innate = rng.uniform(0.0, 1.0, n)
    gammas = g1.gamma_vectors(degree, rng)

    # Compile Numba kernels before measured regions.
    g1.sample_forest(np.array([0, 0], np.int64), np.empty(0, np.int32), np.ones(1), 1)
    dummy = np.zeros((1, 1))
    g1.add_group_stat(dummy, np.zeros(1, np.int64), np.zeros(1, np.int16), np.ones(1))

    rows: list[dict] = []
    reference_rows: list[dict] = []
    for gamma_index, gamma_name in enumerate(GAMMA_NAMES):
        gamma = gammas[gamma_name]
        estimates = make_estimates(A, degree, gamma, group_data, n, gamma_index)
        for r in R_GRID:
            _, W, sizes, cuts = group_data[r]
            W_aug = np.column_stack([W, np.full(n, 1.0 / n)])
            H_true, exact_seconds, iterations, residual, solvers = exact_krylov_augmented(
                A, degree, gamma, W_aug
            )
            raw_true = (1.0 - innate)[:, None] * H_true
            # G2 uses raw group-average gains (tau_j=1), so candidate screening
            # and optimization do not access oracle-derived normalization scales.
            G_true = raw_true[:, :r]
            e_true = raw_true[:, r]
            E_true_star = efficiency_optimum(e_true, k)

            for delta in DELTAS:
                true_threshold = (1.0 - delta) * E_true_star
                screen_start = time.perf_counter()
                true_pool = candidate_pool(G_true, e_true)
                true_screen_seconds = time.perf_counter() - screen_start
                ref_selected, ref_opt_seconds, _ = optimize_pool(
                    G_true, e_true, true_threshold, k, true_pool,
                    g1.SEED + 1009 * r + int(100 * delta) + 100000 * gamma_index,
                )
                ref_eval = evaluate(ref_selected, G_true, e_true, E_true_star, delta)
                milp_selected, mip_gap, mip_exact, milp_seconds = candidate_milp(
                    G_true, e_true, true_threshold, k, true_pool
                )
                milp_eval = evaluate(milp_selected, G_true, e_true, E_true_star, delta)
                lp_upper, lp_seconds = constrained_lp_upper_bound(
                    G_true, e_true, true_threshold, k
                )
                search_regret = gap(milp_eval["fair_value"], ref_eval["fair_value"])
                screening_certificate = gap(lp_upper, milp_eval["fair_value"])
                reference_certificate = gap(lp_upper, ref_eval["fair_value"])
                reference_rows.append({
                    "dataset": f"soc-Pokec-{n}", "gamma": gamma_name, "r": r,
                    "delta": delta, "k": k, "candidate_pool": len(true_pool),
                    "candidate_pool_upper_bound": PER_LIST * (r + 2),
                    "candidate_milp_exact": mip_exact,
                    "candidate_milp_gap": mip_gap,
                    "candidate_milp_seconds": milp_seconds,
                    "candidate_optimum": milp_eval["fair_value"],
                    "reference_fair_value": ref_eval["fair_value"],
                    "reference_efficiency_retention": ref_eval["efficiency_retention"],
                    "reference_search_regret": search_regret,
                    "full_lp_upper_bound": lp_upper,
                    "full_lp_seconds": lp_seconds,
                    "candidate_screening_certificate": screening_certificate,
                    "reference_certified_lp_gap": reference_certificate,
                    "reference_screen_seconds": true_screen_seconds,
                    "reference_optimization_seconds": ref_opt_seconds,
                    "exact_coefficient_seconds": exact_seconds,
                    "exact_iterations": ";".join(map(str, iterations)),
                    "exact_solvers": ";".join(solvers),
                    "exact_residual": residual,
                    "group_sizes": ";".join(map(str, sizes)),
                    "age_cuts": ";".join(map(str, cuts)),
                })

                for method, info in estimates[r].items():
                    H_est = info["H"]
                    raw_est = (1.0 - innate)[:, None] * H_est
                    G_est = raw_est[:, :r]
                    e_est = raw_est[:, r]
                    E_est_star = efficiency_optimum(e_est, k)
                    estimated_threshold = (1.0 - delta) * E_est_star
                    screen_start = time.perf_counter()
                    pool = candidate_pool(G_est, e_est)
                    screening_seconds = time.perf_counter() - screen_start
                    selected, optimization_seconds, _ = optimize_pool(
                        G_est, e_est, estimated_threshold, k, pool,
                        g1.SEED + 1009 * r + int(100 * delta)
                        + 100000 * gamma_index + (0 if method == "Shared-IFR" else 50000),
                    )
                    true_eval = evaluate(selected, G_true, e_true, E_true_star, delta)
                    estimated_retention = float(e_est[selected].sum() / max(E_est_star, 1e-15))
                    ref_gap = gap(ref_eval["fair_value"], true_eval["fair_value"])
                    certified = (
                        gap(lp_upper, true_eval["fair_value"])
                        if true_eval["true_feasible"] else ""
                    )
                    rows.append({
                        "dataset": f"soc-Pokec-{n}", "gamma": gamma_name, "r": r,
                        "delta": delta, "budget": k, "method": method,
                        "forest_samples": info["S"],
                        "refinement_rounds": info["K"],
                        "candidate_pool": len(pool),
                        "candidate_pool_upper_bound": PER_LIST * (r + 2),
                        "candidate_uses_reference_coefficients": False,
                        "true_fair_value": true_eval["fair_value"],
                        "reference_fair_value": ref_eval["fair_value"],
                        "reference_relative_gap": ref_gap,
                        "full_lp_upper_bound": lp_upper,
                        "certified_lp_gap": certified,
                        "estimated_efficiency_retention": estimated_retention,
                        "true_efficiency_retention": true_eval["efficiency_retention"],
                        "true_efficiency_violation": true_eval["true_violation"],
                        "true_feasible": true_eval["true_feasible"],
                        "gain_matrix_max_error": float(np.max(np.abs(G_est - G_true))),
                        "efficiency_max_error": float(np.max(np.abs(e_est - e_true))),
                        "candidate_optimum_node_recall": float(np.mean(np.isin(milp_selected, pool))),
                        "reference_set_overlap": float(len(np.intersect1d(selected, ref_selected)) / k),
                        "estimation_seconds": info["estimation_seconds"],
                        "screening_seconds": screening_seconds,
                        "optimization_seconds": optimization_seconds,
                        "end_to_end_seconds": info["estimation_seconds"] + screening_seconds + optimization_seconds,
                    })
                print(f"completed gamma={gamma_name} r={r} delta={delta}", flush=True)

    write_csv(OUT / "g2_methods.csv", rows)
    write_csv(OUT / "g2_references.csv", reference_rows)
    metadata = {
        "date": "2026-10-25", "nodes": n, "edges": int(A.nnz), "k": k,
        "r_grid": R_GRID, "deltas": DELTAS, "gamma_names": GAMMA_NAMES,
        "per_list": PER_LIST, "candidate_pool_bound": "250(r+2)",
        "forest_seed": FOREST_SEED, "shared_S": SHARED_S,
        "shared_K": SHARED_K, "stationary_K": STATIONARY_K,
        "multistarts": audit.MULTISTARTS, "milp_limit_seconds": MILP_LIMIT,
        "normalization": "tau_j=1 (raw group-average gains; no oracle normalization in G2)",
        "python": platform.python_version(), "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - start_all,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_report(rows, reference_rows, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()
    run(args.nodes, args.budget)
