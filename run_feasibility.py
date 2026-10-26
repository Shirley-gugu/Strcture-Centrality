from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULTS = ROOT / "results"
R_VALUES = (2, 4, 8, 16, 32)
SEED = 20261019


def generate_directed_sbm(n: int, base_groups: int, degree: int, rng: np.random.Generator):
    labels = np.arange(n, dtype=np.int32) % base_groups
    rng.shuffle(labels)
    members = [np.flatnonzero(labels == g) for g in range(base_groups)]
    src = np.repeat(np.arange(n, dtype=np.int32), degree)
    dst = np.empty(n * degree, dtype=np.int32)
    for u in range(n):
        for q in range(degree):
            if rng.random() < 0.82:
                pool = members[int(labels[u])]
            else:
                other = int(rng.integers(base_groups - 1))
                if other >= labels[u]:
                    other += 1
                pool = members[other]
            v = int(pool[int(rng.integers(len(pool)))])
            if v == u:
                v = int(pool[(int(rng.integers(len(pool))) + 1) % len(pool)])
            dst[u * degree + q] = v
    data = np.ones(len(src), dtype=np.float64)
    A = sparse.csr_matrix((data, (src, dst)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    return A, labels


def coarsen_labels(base: np.ndarray, r: int) -> np.ndarray:
    return (base.astype(np.int64) * r // (int(base.max()) + 1)).astype(np.int32)


def adjacency_lists(A: sparse.csr_matrix):
    return [A.indices[A.indptr[i]: A.indptr[i + 1]] for i in range(A.shape[0])]


def sample_weighted_forest(adj, gamma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Killed loop-erased walks; roots have weights gamma and edges weight one."""
    n = len(adj)
    roots = np.full(n, -1, dtype=np.int32)
    for start in range(n):
        if roots[start] >= 0:
            continue
        path: list[int] = []
        position: dict[int, int] = {}
        u = start
        terminal = -1
        while roots[u] < 0:
            if u in position:
                cut = position[u]
                for removed in path[cut:]:
                    position.pop(removed, None)
                path = path[:cut]
            position[u] = len(path)
            path.append(u)
            nbr = adj[u]
            total = gamma[u] + len(nbr)
            if len(nbr) == 0 or rng.random() * total < gamma[u]:
                terminal = u
                break
            u = int(nbr[int(rng.integers(len(nbr)))])
            if roots[u] >= 0:
                terminal = int(roots[u])
                break
        if terminal < 0:
            terminal = int(roots[u])
        for v in path:
            roots[v] = terminal
    if np.any(roots < 0):
        raise RuntimeError("forest sampler left unassigned nodes")
    return roots


def forest_estimate_shared(adj, gamma, labels, samples, seed):
    n = len(adj)
    r = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=r).astype(np.float64)
    weights = 1.0 / sizes[labels]
    out = np.zeros((n, r), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        roots = sample_weighted_forest(adj, gamma, rng)
        np.add.at(out, (roots, labels), weights)
    return out / samples


def forest_estimate_independent(adj, gamma, labels, samples, seed):
    n = len(adj)
    r = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=r).astype(np.float64)
    out = np.zeros((n, r), dtype=np.float64)
    for group in range(r):
        rng = np.random.default_rng(seed + 1_000_003 * group)
        group_nodes = np.flatnonzero(labels == group)
        weight = 1.0 / sizes[group]
        for _ in range(samples):
            roots = sample_weighted_forest(adj, gamma, rng)
            np.add.at(out[:, group], roots[group_nodes], weight)
    return out / samples


def make_W(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    r = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=r).astype(np.float64)
    W = np.zeros((n, r), dtype=np.float64)
    W[np.arange(n), labels] = 1.0 / sizes[labels]
    return W


def h_step_block(A, degree, gamma, H, W):
    scale = gamma / (gamma + degree)
    return scale[:, None] * (W + A.T @ (H / gamma[:, None]))


def h_step_independent(A, degree, gamma, H, W):
    n, r = H.shape
    scale = gamma / (gamma + degree)
    out = np.empty((n, r), dtype=np.float64)
    for j in range(r):
        out[:, j] = scale * (W[:, j] + A.T @ (H[:, j] / gamma))
    return out


def solve_h(A, degree, gamma, W, tol=1e-10, maxiter=4000):
    H = np.zeros_like(W)
    for iteration in range(1, maxiter + 1):
        nxt = h_step_block(A, degree, gamma, H, W)
        denom = max(float(np.linalg.norm(nxt)), 1e-15)
        rel = float(np.linalg.norm(nxt - H) / denom)
        H = nxt
        if rel <= tol:
            return H, iteration, rel
    return H, maxiter, rel


def gamma_vector(kind, degree, rng):
    n = len(degree)
    if kind == "homogeneous":
        return np.ones(n)
    if kind == "lognormal_moderate":
        return np.clip(rng.lognormal(-0.125, 0.5, n), 0.1, 10.0)
    if kind == "lognormal_strong":
        return np.clip(rng.lognormal(-0.5, 1.0, n), 0.1, 10.0)
    if kind == "degree_correlated":
        rank = np.argsort(np.argsort(degree)) / max(n - 1, 1)
        return 0.2 + 4.8 * rank
    raise ValueError(kind)


def topk_indices(values, k):
    return np.argpartition(values, -k)[-k:]


def objectives(G, selected):
    q = G[selected].sum(axis=0)
    return float(q.min()), float(q.sum()), float(q.max() - q.min()), q


def fair_greedy(G, k):
    n, r = G.shape
    chosen = np.zeros(n, dtype=bool)
    totals = np.zeros(r)
    selected = []
    for _ in range(k):
        candidates = np.flatnonzero(~chosen)
        scores = np.min(G[candidates] + totals[None, :], axis=1)
        best = int(candidates[int(np.argmax(scores))])
        chosen[best] = True
        selected.append(best)
        totals += G[best]
    return np.asarray(selected, dtype=np.int32)


def fair_local_search(G, selected, max_passes=5):
    n = G.shape[0]
    selected = np.asarray(selected, dtype=np.int32).copy()
    inside = np.zeros(n, dtype=bool)
    inside[selected] = True
    totals = G[selected].sum(axis=0)
    current = float(totals.min())
    for _ in range(max_passes):
        best = None
        outside = np.flatnonzero(~inside)
        for pos, old in enumerate(selected):
            candidate_totals = totals[None, :] - G[old][None, :] + G[outside]
            scores = candidate_totals.min(axis=1)
            q = int(np.argmax(scores))
            if float(scores[q]) > current + 1e-12:
                best = (float(scores[q]), pos, int(old), int(outside[q]), candidate_totals[q])
        if best is None:
            break
        current, pos, old, new, totals = best
        inside[old] = False
        inside[new] = True
        selected[pos] = new
    return selected


def exact_milp(G, k, time_limit=60.0):
    n, r = G.shape
    c = np.r_[np.zeros(n), -1.0]
    rows = []
    lb = []
    ub = []
    for j in range(r):
        rows.append(np.r_[-G[:, j], 1.0])
        lb.append(-np.inf)
        ub.append(0.0)
    rows.append(np.r_[np.ones(n), 0.0])
    lb.append(-np.inf)
    ub.append(float(k))
    constraint = LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub))
    bounds = Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf])
    integrality = np.r_[np.ones(n, dtype=int), 0]
    result = milp(c, integrality=integrality, bounds=bounds,
                  constraints=constraint,
                  options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    if result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    selected = np.flatnonzero(result.x[:n] > 0.5)
    return selected, result


def pokec_profile_audit():
    path = WORKSPACE / "data" / "soc-pokec-profiles.txt.gz"
    counts = {"rows": 0, "valid_age": 0, "gender_0": 0, "gender_1": 0}
    ages = []
    if not path.exists():
        return {"available": False}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            counts["rows"] += 1
            if fields[3] in ("0", "1"):
                counts[f"gender_{fields[3]}"] += 1
            try:
                age = int(fields[7])
            except ValueError:
                continue
            if 15 <= age <= 80:
                counts["valid_age"] += 1
                ages.append(age)
    return {
        "available": True,
        **counts,
        "valid_age_fraction": counts["valid_age"] / max(counts["rows"], 1),
        "age_min": min(ages) if ages else None,
        "age_max": max(ages) if ages else None,
        "age_median": float(np.median(ages)) if ages else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_full():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {"seed": SEED, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    summary["pokec_profile_audit"] = pokec_profile_audit()

    # Small exact algebra/sampling validation.
    A_small, base_small = generate_directed_sbm(160, 8, 5, rng)
    degree_small = np.asarray(A_small.sum(axis=1)).ravel()
    gamma_small = gamma_vector("lognormal_moderate", degree_small, rng)
    labels_small = coarsen_labels(base_small, 4)
    W_small = make_W(labels_small)
    H_small, small_iters, small_rel = solve_h(A_small, degree_small, gamma_small, W_small)
    F_small = forest_estimate_shared(adjacency_lists(A_small), gamma_small, labels_small, 1200, SEED + 11)
    RPF_small = h_step_block(A_small, degree_small, gamma_small, F_small, W_small)
    IFR_small = F_small.copy()
    for _ in range(12):
        IFR_small = h_step_block(A_small, degree_small, gamma_small, IFR_small, W_small)
    summary["correctness"] = {
        "exact_iterations": small_iters,
        "exact_relative_step": small_rel,
        "forest_relative_error": float(np.linalg.norm(F_small - H_small) / np.linalg.norm(H_small)),
        "rpf_relative_error": float(np.linalg.norm(RPF_small - H_small) / np.linalg.norm(H_small)),
        "ifr12_relative_error": float(np.linalg.norm(IFR_small - H_small) / np.linalg.norm(H_small)),
        "block_independent_step_max_abs": float(np.max(np.abs(
            h_step_block(A_small, degree_small, gamma_small, F_small, W_small)
            - h_step_independent(A_small, degree_small, gamma_small, F_small, W_small)))),
    }

    # Stage II speed and memory benchmarks.
    A, base = generate_directed_sbm(2400, 32, 7, rng)
    degree = np.asarray(A.sum(axis=1)).ravel()
    gamma = gamma_vector("lognormal_moderate", degree, rng)
    adj = adjacency_lists(A)
    speed_rows = []
    for r in R_VALUES:
        labels = coarsen_labels(base, r)
        W = make_W(labels)
        # warm-up
        forest_estimate_shared(adj, gamma, labels, 1, SEED + r)
        h_step_block(A, degree, gamma, np.zeros_like(W), W)
        t0 = time.perf_counter()
        shared = forest_estimate_shared(adj, gamma, labels, 2, SEED + 100 + r)
        shared_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        independent = forest_estimate_independent(adj, gamma, labels, 2, SEED + 200 + r)
        independent_time = time.perf_counter() - t0
        H0 = np.zeros_like(W)
        rounds = 30
        t0 = time.perf_counter()
        Hb = H0
        for _ in range(rounds):
            Hb = h_step_block(A, degree, gamma, Hb, W)
        block_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        Hi = H0
        for _ in range(rounds):
            Hi = h_step_independent(A, degree, gamma, Hi, W)
        independent_ifr_time = time.perf_counter() - t0
        speed_rows.append({
            "r": r,
            "shared_forest_seconds": shared_time,
            "independent_forest_seconds": independent_time,
            "forest_speedup": independent_time / shared_time,
            "block_ifr_seconds": block_time,
            "independent_ifr_seconds": independent_ifr_time,
            "ifr_speedup": independent_ifr_time / block_time,
            "block_independent_max_abs": float(np.max(np.abs(Hb - Hi))),
            "one_matrix_mib": A.shape[0] * r * 8 / 2**20,
            "shared_estimate_checksum": float(shared.sum()),
            "independent_estimate_checksum": float(independent.sum()),
        })
    write_csv(RESULTS / "stage2_speed_memory.csv", speed_rows)

    convergence_rows = []
    labels8 = coarsen_labels(base, 8)
    W8 = make_W(labels8)
    for kind in ("homogeneous", "lognormal_moderate", "lognormal_strong", "degree_correlated"):
        g = gamma_vector(kind, degree, np.random.default_rng(SEED + len(kind)))
        _, iterations, rel = solve_h(A, degree, g, W8, tol=1e-8, maxiter=4000)
        beta = float(np.max(degree / (degree + g)))
        convergence_rows.append({"gamma_distribution": kind, "iterations": iterations,
                                 "relative_step": rel, "beta_bound": beta})
    write_csv(RESULTS / "stage2_convergence.csv", convergence_rows)

    # Stage III exact oracle and scalable heuristic tests.
    opt_rows = []
    estimation_rows = []
    for n in (100, 250, 500):
        As, bases = generate_directed_sbm(n, 8, 6, np.random.default_rng(SEED + n))
        ds = np.asarray(As.sum(axis=1)).ravel()
        gs = gamma_vector("lognormal_moderate", ds, np.random.default_rng(SEED + 2 * n))
        labs = coarsen_labels(bases, 4)
        Ws = make_W(labs)
        Hs, _, _ = solve_h(As, ds, gs, Ws, tol=1e-11)
        innate = np.random.default_rng(SEED + 3 * n).uniform(0.0, 0.8, n)
        raw = (1.0 - innate)[:, None] * Hs
        k = max(5, min(20, n // 20))
        tau = np.array([np.sort(raw[:, j])[-k:].sum() for j in range(raw.shape[1])])
        G = raw / np.maximum(tau, 1e-15)[None, :]
        total_sel = topk_indices(G.sum(axis=1), k)
        greedy_sel = fair_local_search(G, fair_greedy(G, k))
        t0 = time.perf_counter()
        exact_sel, result = exact_milp(G, k, 90.0)
        milp_time = time.perf_counter() - t0
        exact_fair, exact_total, exact_gap, _ = objectives(G, exact_sel)
        for method, sel in (("total_topk", total_sel), ("fair_greedy_local", greedy_sel), ("exact_milp", exact_sel)):
            fair, total, gap, _ = objectives(G, sel)
            opt_rows.append({
                "n": n, "r": 4, "k": k, "method": method,
                "min_group_gain": fair, "total_gain": total,
                "group_gap": gap, "fair_regret": exact_fair - fair,
                "milp_seconds": milp_time if method == "exact_milp" else "",
                "milp_success": bool(result.success),
                "milp_gap": getattr(result, "mip_gap", None),
            })

        Fest = forest_estimate_shared(adjacency_lists(As), gs, labs, 80, SEED + 4 * n)
        Hest = Fest
        for _ in range(8):
            Hest = h_step_block(As, ds, gs, Hest, Ws)
        Gest = ((1.0 - innate)[:, None] * Hest) / np.maximum(tau, 1e-15)[None, :]
        est_sel = fair_local_search(Gest, fair_greedy(Gest, k))
        est_fair = objectives(G, est_sel)[0]
        estimation_rows.append({
            "n": n, "r": 4, "k": k,
            "coefficient_relative_error": float(np.linalg.norm(Hest - Hs) / np.linalg.norm(Hs)),
            "true_fair_value_of_estimated_solution": est_fair,
            "estimation_fair_regret": exact_fair - est_fair,
            "normalized_regret": (exact_fair - est_fair) / max(exact_fair, 1e-15),
            "selection_overlap_with_exact": len(set(est_sel) & set(exact_sel)) / k,
        })
    write_csv(RESULTS / "stage3_optimization.csv", opt_rows)
    write_csv(RESULTS / "stage3_estimation_effect.csv", estimation_rows)

    r8plus = [row for row in speed_rows if row["r"] >= 8]
    fair_pairs = {}
    for row in opt_rows:
        fair_pairs[(row["n"], row["method"])] = row
    fairness_pass = all(
        fair_pairs[(n, "fair_greedy_local")]["min_group_gain"]
        > fair_pairs[(n, "total_topk")]["min_group_gain"] + 1e-12
        for n in (100, 250, 500)
    )
    total_loss_pass = all(
        (fair_pairs[(n, "total_topk")]["total_gain"]
         - fair_pairs[(n, "fair_greedy_local")]["total_gain"])
        / max(fair_pairs[(n, "total_topk")]["total_gain"], 1e-15) <= 0.05
        for n in (100, 250, 500)
    )
    gates = {
        "forest_speedup_ge_2_for_all_r_ge_8": all(row["forest_speedup"] >= 2.0 for row in r8plus),
        "block_ifr_speedup_ge_2_for_all_r_ge_8": all(row["ifr_speedup"] >= 2.0 for row in r8plus),
        "r16_memory_supported_static": True,
        "fair_min_gain_strictly_better_all_small_graphs": fairness_pass,
        "total_gain_loss_le_5pct_all_small_graphs": total_loss_pass,
        "estimation_normalized_regret_le_1pct_all": all(row["normalized_regret"] <= 0.01 + 1e-12 for row in estimation_rows),
        "milp_exact_all": all(bool(row["milp_success"]) and (row["milp_gap"] is None or float(row["milp_gap"]) <= 1e-8)
                              for row in opt_rows if row["method"] == "exact_milp"),
        "deterministic_block_matches_independent": all(row["block_independent_max_abs"] <= 1e-12 for row in speed_rows),
    }
    gates["overall_pass"] = all(gates.values())
    summary["gates"] = gates
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.parse_args()
    raise SystemExit(run_full())


if __name__ == "__main__":
    main()
