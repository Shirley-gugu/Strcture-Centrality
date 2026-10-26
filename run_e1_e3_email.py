from __future__ import annotations

import csv
import gzip
import json
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.linalg import spsolve

import run_feasibility as base


DATA = base.ROOT / "data"
OUT = base.ROOT / "results" / "e1_e3_email"
S_GRID = (1, 5, 10, 20, 50)
K_GRID = (0, 1, 2, 5, 10)
REPETITIONS = 5
SEED = 20261020
K_BUDGET = 20
MILP_TIMEOUT = 60.0


class PeakRSS:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = 0
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = None

    def _watch(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop_event.wait(0.002)

    def __enter__(self):
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def peak_mib(self):
        return self.peak / 2**20

    @property
    def incremental_mib(self):
        return max(0, self.peak - self.baseline) / 2**20


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_email():
    edge_path = DATA / "email-Eu-core.txt.gz"
    label_path = DATA / "email-Eu-core-department-labels.txt.gz"
    labels_by_id = {}
    with gzip.open(label_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            node, department = map(int, line.split())
            labels_by_id[node] = department
    n = max(labels_by_id) + 1
    rows, cols = [], []
    with gzip.open(edge_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0] in "#%":
                continue
            u, v = map(int, line.split()[:2])
            rows.append(u)
            cols.append(v)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    counts = Counter(labels_by_id.values())
    departments = [department for department, _ in counts.most_common(5)]
    W = np.zeros((n, len(departments)), dtype=np.float64)
    for j, department in enumerate(departments):
        members = [node for node, dep in labels_by_id.items() if dep == department]
        W[members, j] = 1.0 / len(members)
    return A, W, departments, [counts[d] for d in departments]


def rel_errors(H, reference):
    denominators = np.linalg.norm(reference, axis=0)
    values = np.linalg.norm(H - reference, axis=0) / denominators
    return values


def refine(A, degree, H, W, rounds):
    out = H
    scale = 1.0 / (1.0 + degree)
    for _ in range(rounds):
        out = scale[:, None] * (W + A.T @ out)
    return out


def solve_fair_milp(G, k, time_limit):
    n, r = G.shape
    c = np.r_[np.zeros(n), -1.0]
    rows, lower, upper = [], [], []
    for j in range(r):
        rows.append(np.r_[-G[:, j], 1.0])
        lower.append(-np.inf)
        upper.append(0.0)
    rows.append(np.r_[np.ones(n), 0.0])
    lower.append(-np.inf)
    upper.append(float(k))
    constraint = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
    bounds = Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf])
    integrality = np.r_[np.ones(n, dtype=int), 0]
    start = time.perf_counter()
    result = milp(
        c, integrality=integrality, bounds=bounds, constraints=constraint,
        options={"time_limit": time_limit, "mip_rel_gap": 1e-8},
    )
    elapsed = time.perf_counter() - start
    selected = np.array([], dtype=np.int32) if result.x is None else np.flatnonzero(result.x[:n] > 0.5)
    return selected, result, elapsed


def mip_gap(result):
    value = getattr(result, "mip_gap", None)
    return None if value is None else float(value)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    A, W, departments, sizes = load_email()
    n, r = W.shape
    degree = np.asarray(A.sum(axis=1)).ravel()
    gamma = np.ones(n)
    adj = base.adjacency_lists(A)

    # Offline high-precision sparse direct reference.
    M = sparse.eye(n, format="csc") + sparse.diags(degree, format="csc") - A.T.tocsc()
    with PeakRSS() as reference_memory:
        start = time.perf_counter()
        Hstar = np.asarray(spsolve(M, W))
        reference_time = time.perf_counter() - start
    reference_residual = float(np.linalg.norm(M @ Hstar - W) / np.linalg.norm(W))

    # Fixed opinions are used only in E3; E1 estimates H independently of s.
    innate = np.random.default_rng(SEED).uniform(0.0, 1.0, n)
    raw_true = (1.0 - innate)[:, None] * Hstar
    tau = np.array([np.sort(raw_true[:, j])[-K_BUDGET:].sum() for j in range(r)])
    Gtrue = raw_true / np.maximum(tau, 1e-15)[None, :]
    exact_selected, exact_result, exact_time = solve_fair_milp(Gtrue, K_BUDGET, MILP_TIMEOUT)
    if exact_result.x is None:
        raise RuntimeError(f"reference MILP has no incumbent: {exact_result.message}")
    exact_group_gains = Gtrue[exact_selected].sum(axis=0)
    exact_fair = float(exact_group_gains.min())

    e1_rows, e3_rows = [], []
    for repetition in range(1, REPETITIONS + 1):
        rng = np.random.default_rng(SEED + repetition * 1_000_003)
        cumulative = np.zeros_like(W)
        sampling_elapsed = 0.0
        next_checkpoint = 0
        with PeakRSS() as run_memory:
            for sample_index in range(1, max(S_GRID) + 1):
                start = time.perf_counter()
                roots = base.sample_weighted_forest(adj, gamma, rng)
                sample_stat = np.zeros_like(W)
                for j in range(r):
                    members = np.flatnonzero(W[:, j] > 0)
                    np.add.at(sample_stat[:, j], roots[members], W[members, j])
                cumulative += sample_stat
                sampling_elapsed += time.perf_counter() - start
                if sample_index != S_GRID[next_checkpoint]:
                    continue
                S = sample_index
                forest = cumulative / S
                for K in K_GRID:
                    start = time.perf_counter()
                    estimate = refine(A, degree, forest, W, K)
                    refine_elapsed = time.perf_counter() - start
                    errors = rel_errors(estimate, Hstar)
                    method = "Shared-Forest" if K == 0 else ("Shared-RPF" if K == 1 else "Shared-IFR")
                    e1_rows.append({
                        "dataset": "email-Eu-core", "repetition": repetition,
                        "S": S, "K": K, "method": method,
                        "department_0": departments[0], "department_1": departments[1],
                        "department_2": departments[2], "department_3": departments[3],
                        "department_4": departments[4],
                        "relerr_group_0": errors[0], "relerr_group_1": errors[1],
                        "relerr_group_2": errors[2], "relerr_group_3": errors[3],
                        "relerr_group_4": errors[4],
                        "mean_relerr": float(errors.mean()), "max_relerr": float(errors.max()),
                        "sampling_seconds": sampling_elapsed,
                        "refinement_seconds": refine_elapsed,
                        "total_estimation_seconds": sampling_elapsed + refine_elapsed,
                        "process_peak_rss_mib": run_memory.peak_mib,
                        "incremental_peak_rss_mib": run_memory.incremental_mib,
                    })

                    raw_est = (1.0 - innate)[:, None] * estimate
                    Gest = raw_est / np.maximum(tau, 1e-15)[None, :]
                    selected, result, optimization_time = solve_fair_milp(Gest, K_BUDGET, MILP_TIMEOUT)
                    if len(selected):
                        true_gains = Gtrue[selected].sum(axis=0)
                        true_fair = float(true_gains.min())
                        bottleneck = int(np.argmin(true_gains))
                        overlap = len(set(selected) & set(exact_selected)) / K_BUDGET
                    else:
                        true_gains = np.full(r, np.nan)
                        true_fair = np.nan
                        bottleneck = -1
                        overlap = np.nan
                    e3_rows.append({
                        "dataset": "email-Eu-core", "repetition": repetition,
                        "S": S, "K": K, "method": method,
                        "true_fair_value": true_fair,
                        "fair_regret": exact_fair - true_fair,
                        "normalized_fair_regret": (exact_fair - true_fair) / exact_fair,
                        "true_bottleneck_group": bottleneck,
                        "true_group_0": true_gains[0], "true_group_1": true_gains[1],
                        "true_group_2": true_gains[2], "true_group_3": true_gains[3],
                        "true_group_4": true_gains[4],
                        "selection_overlap": overlap,
                        "coefficient_mean_relerr": float(errors.mean()),
                        "coefficient_max_relerr": float(errors.max()),
                        "estimation_seconds": sampling_elapsed + refine_elapsed,
                        "optimization_seconds": optimization_time,
                        "end_to_end_seconds": sampling_elapsed + refine_elapsed + optimization_time,
                        "solver_success": bool(result.success), "solver_status": int(result.status),
                        "mip_gap": mip_gap(result), "solver_message": str(result.message),
                    })
                next_checkpoint += 1
                if next_checkpoint == len(S_GRID):
                    break

    # Batched deterministic iteration baseline, independent of repetitions.
    batched_rows = []
    H = np.zeros_like(W)
    elapsed = 0.0
    checkpoints = set(K_GRID[1:])
    for iteration in range(1, 10001):
        start = time.perf_counter()
        nxt = refine(A, degree, H, W, 1)
        elapsed += time.perf_counter() - start
        relstep = float(np.linalg.norm(nxt - H) / max(np.linalg.norm(nxt), 1e-15))
        H = nxt
        if iteration in checkpoints or relstep <= 1e-12:
            errors = rel_errors(H, Hstar)
            batched_rows.append({
                "iteration": iteration, "mean_relerr": float(errors.mean()),
                "max_relerr": float(errors.max()), "elapsed_seconds": elapsed,
                "relative_step": relstep,
            })
        if relstep <= 1e-12:
            break

    write_csv(OUT / "e1_trials.csv", e1_rows)
    write_csv(OUT / "e1_batched_iteration.csv", batched_rows)
    write_csv(OUT / "e3_trials.csv", e3_rows)
    metadata = {
        "dataset": "email-Eu-core", "n": n, "m": int(A.nnz),
        "departments": departments, "department_sizes": sizes,
        "S_grid": S_GRID, "K_grid": K_GRID, "repetitions": REPETITIONS,
        "gamma": "identity", "opinion_seed": SEED,
        "reference_solve_seconds": reference_time,
        "reference_residual": reference_residual,
        "reference_peak_rss_mib": reference_memory.peak_mib,
        "reference_incremental_peak_rss_mib": reference_memory.incremental_mib,
        "k_budget": K_BUDGET, "tau_rule": "single-group exact top-k",
        "exact_fair_value": exact_fair, "exact_group_gains": exact_group_gains.tolist(),
        "exact_milp_seconds": exact_time, "exact_mip_gap": mip_gap(exact_result),
        "exact_solver_success": bool(exact_result.success),
    }
    with (OUT / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
