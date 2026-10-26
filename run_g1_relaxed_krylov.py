from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import bicgstab

import run_g1_pokec as g1


RTOLS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ERROR_TARGETS = (1e-2, 5e-3, 2e-3)
REPEATS = 5


def relaxed_krylov(A, degree, gamma, W, rtol):
    M = sparse.diags(gamma + degree, format="csc") - A.T.tocsc()
    H = np.empty_like(W)
    iterations = []
    start = time.perf_counter()
    for j in range(W.shape[1]):
        count = [0]

        def callback(_):
            count[0] += 1

        u, info = bicgstab(
            M, W[:, j], rtol=rtol, atol=0.0, maxiter=5000, callback=callback
        )
        if info != 0:
            raise RuntimeError(
                f"BiCGSTAB failed: rtol={rtol}, group={j}, info={info}"
            )
        H[:, j] = gamma * u
        iterations.append(count[0])
    elapsed = time.perf_counter() - start
    residual = np.linalg.norm((M @ (H / gamma[:, None])) - W) / np.linalg.norm(W)
    return H, elapsed, iterations, float(residual)


def run(n=100000, k_budget=50):
    g1.OUT.mkdir(parents=True, exist_ok=True)
    A, gender, age = g1.load_pokec(n)
    degree = np.asarray(A.sum(axis=1)).ravel()
    groups = {r: g1.make_groups(gender, age, r) for r in g1.R_GRID}
    rng = np.random.default_rng(g1.SEED)
    innate = rng.uniform(0.0, 1.0, n)
    gammas = g1.gamma_vectors(degree, rng)
    rows = []
    matched = []
    started = time.perf_counter()

    for gamma_name, gamma in gammas.items():
        for r in g1.R_GRID:
            _, W, _, _ = groups[r]
            Htrue, reference_seconds, reference_iterations, reference_residual = (
                g1.exact_krylov(A, degree, gamma, W)
            )
            raw_true = (1.0 - innate)[:, None] * Htrue
            tau = np.array(
                [
                    np.partition(raw_true[:, j], -k_budget)[-k_budget:].sum()
                    for j in range(r)
                ]
            )
            Gtrue = raw_true / np.maximum(tau, 1e-15)[None, :]
            reference_set, _, _ = g1.candidate_fair_optimizer(Gtrue, k_budget)
            reference_value = float(Gtrue[reference_set].sum(axis=0).min())

            config_rows = []
            for rtol in RTOLS:
                times = []
                outputs = []
                for repeat in range(REPEATS):
                    H, elapsed, iterations, residual = relaxed_krylov(
                        A, degree, gamma, W, rtol
                    )
                    times.append(elapsed)
                    outputs.append((H, iterations, residual))
                median_index = int(np.argsort(times)[REPEATS // 2])
                H, iterations, residual = outputs[median_index]
                estimation_seconds = float(times[median_index])
                Gest = ((1.0 - innate)[:, None] * H) / np.maximum(
                    tau, 1e-15
                )[None, :]
                gain_error = float(np.max(np.abs(Gest - Gtrue)))
                selected, optimization_seconds, pool = g1.candidate_fair_optimizer(
                    Gest, k_budget
                )
                true_fair_value = float(Gtrue[selected].sum(axis=0).min())
                row = {
                    "dataset": f"soc-Pokec-{n}",
                    "gamma": gamma_name,
                    "r": r,
                    "method": "Relaxed-Batched-Krylov",
                    "rtol": rtol,
                    "repeats": REPEATS,
                    "time_min": float(np.min(times)),
                    "estimation_seconds": estimation_seconds,
                    "time_max": float(np.max(times)),
                    "iterations_mean": float(np.mean(iterations)),
                    "iterations_max": int(np.max(iterations)),
                    "linear_residual": residual,
                    "gain_matrix_max_error": gain_error,
                    "optimization_seconds": optimization_seconds,
                    "end_to_end_seconds": estimation_seconds + optimization_seconds,
                    "true_fair_value": true_fair_value,
                    "reference_regret_ratio": g1.safe_regret(
                        reference_value, true_fair_value
                    ),
                    "candidate_pool": pool,
                    "reference_seconds": reference_seconds,
                    "reference_residual": reference_residual,
                    "reference_iterations_mean": float(
                        np.mean(reference_iterations)
                    ),
                    "peak_rss_mib": psutil.Process().memory_info().rss / 2**20,
                }
                rows.append(row)
                config_rows.append(row)
                print(
                    gamma_name,
                    r,
                    f"rtol={rtol:g}",
                    f"gain_error={gain_error:.6g}",
                    f"median_seconds={estimation_seconds:.3f}",
                    flush=True,
                )

            for target in ERROR_TARGETS:
                eligible = [
                    row
                    for row in config_rows
                    if row["gain_matrix_max_error"] <= target
                ]
                if eligible:
                    best = min(eligible, key=lambda row: row["estimation_seconds"])
                    matched.append(
                        {
                            "gamma": gamma_name,
                            "r": r,
                            "target_gain_error": target,
                            **{
                                key: best[key]
                                for key in (
                                    "rtol",
                                    "gain_matrix_max_error",
                                    "linear_residual",
                                    "iterations_mean",
                                    "estimation_seconds",
                                    "optimization_seconds",
                                    "end_to_end_seconds",
                                    "reference_regret_ratio",
                                )
                            },
                        }
                    )
                else:
                    matched.append(
                        {
                            "gamma": gamma_name,
                            "r": r,
                            "target_gain_error": target,
                            "rtol": "NA",
                        }
                    )

    g1.write_csv(g1.OUT / "g1_relaxed_krylov.csv", rows)
    g1.write_csv(g1.OUT / "g1_relaxed_krylov_matched.csv", matched)
    metadata = {
        "date": "2026-10-22",
        "nodes": n,
        "edges": int(A.nnz),
        "budget": k_budget,
        "r_grid": list(g1.R_GRID),
        "rtols": list(RTOLS),
        "error_targets": list(ERROR_TARGETS),
        "timing_repeats": REPEATS,
        "threads_visible": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": psutil.Process().memory_info().rss / 2**20,
        "interpretation_note": (
            "This is the directed transferable group-adjoint Krylov baseline, "
            "not a verbatim reproduction of Lu et al.'s undirected signed-"
            "Laplacian implementation. Linear residual and gain-matrix error "
            "are reported separately."
        ),
    }
    (g1.OUT / "g1_relaxed_krylov_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    run()
