from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import psutil
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import bicgstab


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "results" / "g1_pokec"
S_GRID = (5, 10, 20, 50)
K_GRID = (0, 1, 2, 5, 10)
R_GRID = (4, 8, 16)
SEED = 20261021
FOREST_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pokec(n: int):
    edge_path = WORKSPACE / "data" / "soc-pokec-relationships.txt"
    profile_path = WORKSPACE / "data" / "soc-pokec-profiles.txt.gz"
    gender = np.full(n, -1, np.int8)
    age = np.full(n, -1, np.int16)
    with gzip.open(profile_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            node = int(fields[0])
            if node > n:
                continue
            if fields[3] in ("0", "1"):
                gender[node - 1] = int(fields[3])
            try:
                value = int(fields[7])
                if 14 <= value <= 80:
                    age[node - 1] = value
            except ValueError:
                pass

    src, dst = [], []
    with edge_path.open("rt", encoding="utf-8") as f:
        for line in f:
            u, v = map(int, line.split()[:2])
            if u <= n and v <= n and u != v:
                src.append(u - 1)
                dst.append(v - 1)
    A = sparse.csr_matrix(
        (np.ones(len(src), np.float64), (np.asarray(src), np.asarray(dst))),
        shape=(n, n),
    )
    A.sum_duplicates()
    A.data[:] = 1.0
    A.sort_indices()
    return A, gender, age


def make_groups(gender: np.ndarray, age: np.ndarray, r: int):
    bins = r // 2
    valid = (gender >= 0) & (age >= 14)
    # Fixed, reproducible quantile cuts from the observed Pokec-100k population.
    cuts = np.quantile(age[valid], np.linspace(0, 1, bins + 1)[1:-1])
    age_bin = np.searchsorted(cuts, age, side="right")
    labels = np.full(len(age), -1, np.int16)
    labels[valid] = gender[valid] * bins + age_bin[valid]
    sizes = np.bincount(labels[valid], minlength=r).astype(np.float64)
    W = np.zeros((len(age), r), np.float64)
    rows = np.flatnonzero(valid)
    W[rows, labels[rows]] = 1.0 / sizes[labels[rows]]
    return labels, W, sizes.astype(int), cuts.tolist()


@nb.njit(cache=True)
def sample_forest(indptr, indices, gamma, seed):
    np.random.seed(seed)
    n = len(gamma)
    inside = np.zeros(n, np.uint8)
    nxt = np.empty(n, np.int64)
    root = np.empty(n, np.int64)
    for start in range(n):
        u = start
        while inside[u] == 0:
            degree = indptr[u + 1] - indptr[u]
            if degree == 0 or np.random.random() * (gamma[u] + degree) < gamma[u]:
                inside[u] = 1
                root[u] = u
                break
            nxt[u] = indices[indptr[u] + np.random.randint(degree)]
            u = nxt[u]
        terminal = root[u]
        u = start
        while inside[u] == 0:
            inside[u] = 1
            root[u] = terminal
            u = nxt[u]
    return root


@nb.njit(cache=True)
def add_group_stat(cumulative, roots, labels, sizes):
    for v in range(len(roots)):
        group = labels[v]
        if group >= 0:
            cumulative[roots[v], group] += 1.0 / sizes[group]


def refine(A, degree, gamma, H, W, rounds):
    out = H
    scale = gamma / (gamma + degree)
    for _ in range(rounds):
        out = scale[:, None] * (W + A.T @ (out / gamma[:, None]))
    return out


def exact_krylov(A, degree, gamma, W, rtol=1e-10):
    M = sparse.diags(gamma + degree, format="csc") - A.T.tocsc()
    H = np.empty_like(W)
    iterations = []
    start = time.perf_counter()
    for j in range(W.shape[1]):
        count = [0]
        def callback(_):
            count[0] += 1
        u, info = bicgstab(M, W[:, j], rtol=rtol, atol=0.0,
                           maxiter=5000, callback=callback)
        if info != 0:
            raise RuntimeError(f"BiCGSTAB failed for group {j}: info={info}")
        H[:, j] = gamma * u
        iterations.append(count[0])
    elapsed = time.perf_counter() - start
    residual = np.linalg.norm((M @ (H / gamma[:, None])) - W) / np.linalg.norm(W)
    return H, elapsed, iterations, float(residual)


def errors(estimate, truth):
    rel = np.linalg.norm(estimate - truth, axis=0) / np.maximum(
        np.linalg.norm(truth, axis=0), 1e-15
    )
    gain_max = float(np.max(np.abs(estimate - truth)))
    return float(rel.mean()), float(rel.max()), gain_max


def safe_gap(upper, value):
    return 0.0 if abs(upper) <= 1e-15 else float((upper - value) / upper)


def safe_regret(reference, value):
    return 0.0 if abs(reference) <= 1e-15 else max(0.0, float((reference - value) / reference))


def candidate_fair_optimizer(G, k, per_group=250):
    start = time.perf_counter()
    n, r = G.shape
    pool_parts = [np.argpartition(G[:, j], -min(per_group, n))[-min(per_group, n):]
                  for j in range(r)]
    aggregate = G.mean(axis=1)
    pool_parts.append(np.argpartition(aggregate, -min(per_group, n))[-min(per_group, n):])
    pool = np.unique(np.concatenate(pool_parts))
    selected = []
    chosen = np.zeros(len(pool), bool)
    totals = np.zeros(r)
    for _ in range(k):
        candidates = np.flatnonzero(~chosen)
        scores = np.min(totals[None, :] + G[pool[candidates]], axis=1)
        q = int(candidates[np.argmax(scores)])
        chosen[q] = True
        selected.append(int(pool[q]))
        totals += G[pool[q]]
    return np.asarray(selected, np.int64), time.perf_counter() - start, len(pool)


def lp_upper_bound(G, k):
    # Continuous relaxation: max t, t <= G[:,j]^T x, sum x <= k.
    n, r = G.shape
    c = np.r_[np.zeros(n), -1.0]
    A_ub = sparse.vstack([
        sparse.hstack([-sparse.csr_matrix(G[:, j][None, :]), sparse.csr_matrix([[1.0]])])
        for j in range(r)
    ] + [sparse.hstack([sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix([[0.0]])])],
        format="csr")
    b_ub = np.r_[np.zeros(r), float(k)]
    start = time.perf_counter()
    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(0, 1)] * n + [(None, None)], method="highs")
    elapsed = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"LP relaxation failed: {result.message}")
    return float(-result.fun), elapsed


def gamma_vectors(degree, rng):
    n = len(degree)
    alpha = rng.uniform(0.2, 0.8, n)
    return {
        "identity": np.ones(n),
        "random_lognormal": np.clip(rng.lognormal(-0.125, 0.5, n), 0.1, 10.0),
        "alpha_converted": np.maximum(degree, 1.0) * alpha / (1.0 - alpha),
    }


def plot_results(rows):
    for metric_x, metric_y, filename, xlabel, ylabel in [
        ("estimation_seconds", "gain_matrix_max_error", "runtime_vs_gain_error.pdf",
         "Estimation runtime (s)", "Gain-matrix max error"),
        ("end_to_end_seconds", "reference_regret_ratio", "runtime_vs_fair_regret.pdf",
         "End-to-end runtime (s)", "Fair regret relative to common reference"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
        for ax, r in zip(axes, R_GRID):
            subset = [x for x in rows if int(x["r"]) == r and x["gamma"] == "identity"]
            for method in sorted(set(x["method"] for x in subset)):
                points = [x for x in subset if x["method"] == method]
                if not points:
                    continue
                points.sort(key=lambda x: float(x[metric_x]))
                ax.plot([float(x[metric_x]) for x in points],
                        [float(x[metric_y]) for x in points], marker="o", ms=3,
                        label=method)
            ax.set_title(f"r={r}")
            ax.set_xlabel(xlabel)
            ax.set_yscale("log")
            ax.grid(alpha=.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / filename)
        plt.close(fig)


def run(n: int, k_budget: int, forest_seeds=FOREST_SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    start_all = time.perf_counter()
    A, gender, age = load_pokec(n)
    degree = np.asarray(A.sum(axis=1)).ravel()
    groups = {r: make_groups(gender, age, r) for r in R_GRID}
    rng = np.random.default_rng(SEED)
    innate = rng.uniform(0.0, 1.0, n)
    gammas = gamma_vectors(degree, rng)
    # Compile Numba kernels outside measured regions.
    sample_forest(np.array([0, 0], np.int64), np.empty(0, np.int32), np.ones(1), 1)
    dummy = np.zeros((1, 1)); add_group_stat(dummy, np.zeros(1, np.int64), np.zeros(1, np.int16), np.ones(1))

    rows, reference_rows = [], []
    for gamma_name, gamma in gammas.items():
        truths = {}
        references = {}
        for r in R_GRID:
            labels, W, sizes, cuts = groups[r]
            Htrue, krylov_time, krylov_iters, residual = exact_krylov(A, degree, gamma, W)
            raw = (1.0 - innate)[:, None] * Htrue
            tau = np.array([np.partition(raw[:, j], -k_budget)[-k_budget:].sum() for j in range(r)])
            Gtrue = raw / np.maximum(tau, 1e-15)[None, :]
            reference_set, opt_time, pool_size = candidate_fair_optimizer(Gtrue, k_budget)
            reference_value = float(Gtrue[reference_set].sum(axis=0).min())
            lp_ub, lp_time = lp_upper_bound(Gtrue, k_budget)
            truths[r] = (Htrue, Gtrue, tau)
            references[r] = (reference_value, lp_ub)
            reference_rows.append({
                "dataset": f"soc-Pokec-{n}", "gamma": gamma_name, "r": r,
                "nodes": n, "edges": A.nnz, "labeled_nodes": int((labels >= 0).sum()),
                "group_sizes": ";".join(map(str, sizes)), "age_cuts": ";".join(map(str, cuts)),
                "k": k_budget, "krylov_seconds": krylov_time,
                "krylov_iterations": ";".join(map(str, krylov_iters)),
                "krylov_residual": residual, "reference_optimizer_seconds": opt_time,
                "reference_candidate_pool": pool_size, "reference_fair_value": reference_value,
                "lp_upper_bound": lp_ub, "reference_verified_gap": safe_gap(lp_ub, reference_value),
                "lp_seconds": lp_time,
            })
            rows.append({
                "dataset": f"soc-Pokec-{n}", "gamma": gamma_name, "r": r,
                "forest_seed": 0,
                "method": "Batched-Krylov", "S": 0, "K": 0,
                "mean_group_rel_error": 0.0, "max_group_rel_error": 0.0,
                "gain_matrix_max_error": 0.0, "estimation_seconds": krylov_time,
                "optimization_seconds": opt_time, "end_to_end_seconds": krylov_time + opt_time,
                "true_fair_value": reference_value, "reference_regret": 0.0,
                "reference_regret_ratio": 0.0, "verified_gap_to_lp": safe_gap(lp_ub, reference_value),
                "candidate_pool": pool_size, "peak_rss_mib": psutil.Process().memory_info().rss/2**20,
            })

            # Deterministic fixed-point checkpoints and converged endpoint.
            H = np.zeros_like(W)
            fp_elapsed = 0.0
            checkpoints = set(K_GRID[1:])
            for iteration in range(1, 5001):
                t0 = time.perf_counter(); H = refine(A, degree, gamma, H, W, 1); fp_elapsed += time.perf_counter()-t0
                if iteration in checkpoints:
                    raw_est = (1.0-innate)[:, None] * H
                    Gest = raw_est / np.maximum(tau, 1e-15)[None, :]
                    selected, opt_t, pool = candidate_fair_optimizer(Gest, k_budget)
                    fair = float(Gtrue[selected].sum(axis=0).min())
                    mean_e, max_e, _ = errors(H, Htrue)
                    gain_e = float(np.max(np.abs(Gest-Gtrue)))
                    rows.append({
                        "dataset": f"soc-Pokec-{n}", "gamma": gamma_name, "r": r,
                        "forest_seed": 0,
                        "method": "Batched-Fixed-Point", "S": 0, "K": iteration,
                        "mean_group_rel_error": mean_e, "max_group_rel_error": max_e,
                        "gain_matrix_max_error": gain_e, "estimation_seconds": fp_elapsed,
                        "optimization_seconds": opt_t, "end_to_end_seconds": fp_elapsed+opt_t,
                        "true_fair_value": fair, "reference_regret": reference_value-fair,
                        "reference_regret_ratio": safe_regret(reference_value, fair),
                        "verified_gap_to_lp": safe_gap(lp_ub, fair), "candidate_pool": pool,
                        "peak_rss_mib": psutil.Process().memory_info().rss/2**20,
                    })
                if iteration >= 10 and np.linalg.norm(H-Htrue)/max(np.linalg.norm(Htrue),1e-15) < 1e-8:
                    break

        for forest_seed in forest_seeds:
            cumulatives = {r: np.zeros((n, r), np.float64) for r in R_GRID}
            stat_times = {r: 0.0 for r in R_GRID}
            forest_time = 0.0
            checkpoint = 0
            for sample_index in range(1, max(S_GRID)+1):
                t0 = time.perf_counter()
                roots = sample_forest(
                    A.indptr, A.indices, gamma,
                    forest_seed + sample_index + 100000*list(gammas).index(gamma_name),
                )
                forest_time += time.perf_counter()-t0
                for r in R_GRID:
                    labels, _, sizes, _ = groups[r]
                    t0 = time.perf_counter()
                    add_group_stat(cumulatives[r], roots, labels, sizes)
                    stat_times[r] += time.perf_counter()-t0
                if sample_index != S_GRID[checkpoint]:
                    continue
                S = sample_index
                for r in R_GRID:
                    _, W, _, _ = groups[r]
                    H0 = cumulatives[r] / S
                    Htrue, Gtrue, tau = truths[r]
                    reference_value, lp_ub = references[r]
                    for K in K_GRID:
                        t0 = time.perf_counter()
                        H = refine(A, degree, gamma, H0, W, K)
                        refinement_time = time.perf_counter()-t0
                        raw_est = (1.0-innate)[:, None]*H
                        Gest = raw_est/np.maximum(tau,1e-15)[None,:]
                        selected,opt_t,pool = candidate_fair_optimizer(Gest,k_budget)
                        fair=float(Gtrue[selected].sum(axis=0).min())
                        mean_e,max_e,_=errors(H,Htrue)
                        method="Shared-Forest" if K==0 else ("Shared-RPF" if K==1 else "Shared-IFR")
                        est_t=forest_time+stat_times[r]+refinement_time
                        rows.append({
                            "dataset":f"soc-Pokec-{n}","gamma":gamma_name,"r":r,
                            "forest_seed":forest_seed,"method":method,"S":S,"K":K,
                            "mean_group_rel_error":mean_e,"max_group_rel_error":max_e,
                            "gain_matrix_max_error":float(np.max(np.abs(Gest-Gtrue))),"estimation_seconds":est_t,
                            "optimization_seconds":opt_t,"end_to_end_seconds":est_t+opt_t,"true_fair_value":fair,
                            "reference_regret":reference_value-fair,
                            "reference_regret_ratio":safe_regret(reference_value, fair),
                            "verified_gap_to_lp":safe_gap(lp_ub, fair),"candidate_pool":pool,
                            "peak_rss_mib":psutil.Process().memory_info().rss/2**20,
                        })
                checkpoint += 1

    write_csv(OUT/"g1_methods.csv",rows)
    write_csv(OUT/"g1_references.csv",reference_rows)
    plot_results(rows)
    metadata={
        "date":"2026-10-22","seed":SEED,"forest_seeds":list(forest_seeds),
        "nodes":n,"edges":int(A.nnz),"k":k_budget,
        "S_grid":S_GRID,"K_grid":K_GRID,"r_grid":R_GRID,"threads":os.cpu_count(),
        "python":platform.python_version(),"platform":platform.platform(),
        "wall_seconds":time.perf_counter()-start_all,"peak_rss_mib":psutil.Process().memory_info().rss/2**20,
        "interpretation_note":"FairRegret is relative to a common scalable reference optimizer; verified_gap_to_lp uses a global LP upper bound and is not integer regret."
    }
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--nodes",type=int,default=100000)
    parser.add_argument("--budget",type=int,default=50)
    parser.add_argument("--forest-seeds",type=int,nargs="+",default=list(FOREST_SEEDS))
    args=parser.parse_args()
    run(args.nodes,args.budget,args.forest_seeds)
