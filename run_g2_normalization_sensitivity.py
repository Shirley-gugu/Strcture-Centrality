from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

import run_g1_pokec as g1
import run_g2_hsfom as g2


OUT = g1.ROOT / "results" / "g2_normalization_sensitivity"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_set(selected: np.ndarray, raw_gain: np.ndarray,
                 tau_oracle: np.ndarray, efficiency: np.ndarray,
                 efficiency_star: float) -> dict:
    group_gains = raw_gain[selected].sum(axis=0)
    potential_profile = group_gains / tau_oracle
    return {
        "raw_value": float(group_gains.min()),
        "potential_value": float(potential_profile.min()),
        "efficiency_retention": float(
            efficiency[selected].sum() / max(efficiency_star, 1e-15)
        ),
        "raw_bottleneck": int(np.argmin(group_gains)) + 1,
        "potential_bottleneck": int(np.argmin(potential_profile)) + 1,
        "group_gains": group_gains,
        "potential_profile": potential_profile,
    }


def relative_loss(best: float, alternative: float) -> float:
    return max(0.0, float((best - alternative) / max(best, 1e-15)))


def format_vector(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.12g}" for value in values)


def make_report(rows: list[dict], metadata: dict) -> None:
    lines = [
        "# G2 normalization sensitivity on Pokec-100k",
        "",
        "## Material Passport",
        "",
        "- Verification status: COMPLETED; all candidate MILPs returned zero reported gap",
        f"- Configurations: {len(rows)} (two stubbornness settings, two group counts, two efficiency budgets)",
        f"- Intervention budget: k={metadata['k']}",
        "- Coefficients: high-accuracy augmented Krylov solves used only for this offline diagnostic",
        "- Compared objectives: raw group-average gain and oracle potential-normalized gain",
        "- Controlled factors: one common union candidate pool, the same true efficiency constraint, and exact candidate-pool MILP optimization",
        "- Runtime status: excluded from deployable pipeline timing because oracle normalization is unavailable online",
        "",
        "## Cross-objective sensitivity",
        "",
        "Raw loss is the relative loss in raw fairness when the potential-normalized solution is used. Potential loss is the relative loss in potential-normalized fairness when the raw solution is used. Profile change is the maximum absolute change in group achieved-potential ratios.",
        "",
        "| Gamma | r | delta | |C_union| | set overlap | raw loss | potential loss | profile change | E_raw/E* | E_pot/E* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gamma']} | {row['r']} | {100*row['delta']:.0f}% | "
            f"{row['union_candidate_pool']} | {100*row['set_overlap']:.1f}% | "
            f"{100*row['raw_cross_loss']:.3f}% | "
            f"{100*row['potential_cross_loss']:.3f}% | "
            f"{100*row['potential_profile_max_change']:.3f} pp | "
            f"{100*row['raw_solution_efficiency_retention']:.3f}% | "
            f"{100*row['potential_solution_efficiency_retention']:.3f}% |"
        )
    overlap = np.array([row["set_overlap"] for row in rows])
    raw_loss = np.array([row["raw_cross_loss"] for row in rows])
    potential_loss = np.array([row["potential_cross_loss"] for row in rows])
    profile_change = np.array([row["potential_profile_max_change"] for row in rows])
    lines += [
        "",
        "## Summary",
        "",
        f"- Set overlap range: {100*overlap.min():.1f}%--{100*overlap.max():.1f}% (median {100*np.median(overlap):.1f}%).",
        f"- Raw-fairness cross-loss range: {100*raw_loss.min():.3f}%--{100*raw_loss.max():.3f}%.",
        f"- Potential-normalized cross-loss range: {100*potential_loss.min():.3f}%--{100*potential_loss.max():.3f}%.",
        f"- Maximum achieved-potential profile change: {100*profile_change.max():.3f} percentage points.",
        "",
        "The diagnostic isolates the normative choice of fairness scale. It does not compare estimators and is not included in online runtime claims.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(n: int, k: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    A, gender, age = g1.load_pokec(n)
    degree = np.asarray(A.sum(axis=1)).ravel()
    group_data = {r: g1.make_groups(gender, age, r) for r in g2.R_GRID}
    rng = np.random.default_rng(g1.SEED)
    innate = rng.uniform(0.0, 1.0, n)
    gammas = g1.gamma_vectors(degree, rng)

    rows: list[dict] = []
    solve_audit: list[dict] = []
    for gamma_name in g2.GAMMA_NAMES:
        gamma = gammas[gamma_name]
        for r in g2.R_GRID:
            _, W, sizes, cuts = group_data[r]
            W_aug = np.column_stack([W, np.full(n, 1.0 / n)])
            H_true, exact_seconds, iterations, residual, solvers = (
                g2.exact_krylov_augmented(A, degree, gamma, W_aug)
            )
            augmented_gain = (1.0 - innate)[:, None] * H_true
            raw_gain = augmented_gain[:, :r]
            efficiency = augmented_gain[:, r]
            tau_oracle = np.array([
                g2.efficiency_optimum(raw_gain[:, j], k) for j in range(r)
            ])
            potential_gain = raw_gain / tau_oracle
            efficiency_star = g2.efficiency_optimum(efficiency, k)

            raw_pool = g2.candidate_pool(raw_gain, efficiency)
            potential_pool = g2.candidate_pool(potential_gain, efficiency)
            union_pool = np.union1d(raw_pool, potential_pool)

            for delta in g2.DELTAS:
                threshold = (1.0 - delta) * efficiency_star
                raw_selected, raw_gap, raw_exact, raw_seconds = g2.candidate_milp(
                    raw_gain, efficiency, threshold, k, union_pool
                )
                potential_selected, potential_gap, potential_exact, potential_seconds = (
                    g2.candidate_milp(
                        potential_gain, efficiency, threshold, k, union_pool
                    )
                )
                if not raw_exact or not potential_exact:
                    raise RuntimeError(
                        "normalization sensitivity requires exact candidate MILPs"
                    )

                raw_eval = evaluate_set(
                    raw_selected, raw_gain, tau_oracle,
                    efficiency, efficiency_star,
                )
                potential_eval = evaluate_set(
                    potential_selected, raw_gain, tau_oracle,
                    efficiency, efficiency_star,
                )
                overlap = len(np.intersect1d(raw_selected, potential_selected)) / k
                profile_change = float(np.max(np.abs(
                    raw_eval["potential_profile"]
                    - potential_eval["potential_profile"]
                )))
                row = {
                    "dataset": f"soc-Pokec-{n}",
                    "gamma": gamma_name,
                    "r": r,
                    "delta": delta,
                    "k": k,
                    "raw_candidate_pool": len(raw_pool),
                    "potential_candidate_pool": len(potential_pool),
                    "union_candidate_pool": len(union_pool),
                    "tau_min": float(tau_oracle.min()),
                    "tau_max": float(tau_oracle.max()),
                    "tau_max_min_ratio": float(tau_oracle.max() / tau_oracle.min()),
                    "set_overlap": overlap,
                    "raw_solution_raw_value": raw_eval["raw_value"],
                    "potential_solution_raw_value": potential_eval["raw_value"],
                    "raw_cross_loss": relative_loss(
                        raw_eval["raw_value"], potential_eval["raw_value"]
                    ),
                    "raw_solution_potential_value": raw_eval["potential_value"],
                    "potential_solution_potential_value": potential_eval["potential_value"],
                    "potential_cross_loss": relative_loss(
                        potential_eval["potential_value"], raw_eval["potential_value"]
                    ),
                    "potential_profile_max_change": profile_change,
                    "raw_solution_efficiency_retention": raw_eval["efficiency_retention"],
                    "potential_solution_efficiency_retention": potential_eval["efficiency_retention"],
                    "raw_solution_raw_bottleneck": raw_eval["raw_bottleneck"],
                    "potential_solution_raw_bottleneck": potential_eval["raw_bottleneck"],
                    "raw_solution_potential_bottleneck": raw_eval["potential_bottleneck"],
                    "potential_solution_potential_bottleneck": potential_eval["potential_bottleneck"],
                    "raw_solution_group_gains": format_vector(raw_eval["group_gains"]),
                    "potential_solution_group_gains": format_vector(potential_eval["group_gains"]),
                    "raw_solution_potential_profile": format_vector(raw_eval["potential_profile"]),
                    "potential_solution_potential_profile": format_vector(potential_eval["potential_profile"]),
                    "raw_milp_seconds": raw_seconds,
                    "potential_milp_seconds": potential_seconds,
                    "raw_milp_gap": raw_gap,
                    "potential_milp_gap": potential_gap,
                }
                rows.append(row)
                print(
                    f"completed gamma={gamma_name} r={r} delta={delta}: "
                    f"overlap={overlap:.3f}, raw_loss={row['raw_cross_loss']:.6f}, "
                    f"potential_loss={row['potential_cross_loss']:.6f}",
                    flush=True,
                )

            solve_audit.append({
                "gamma": gamma_name,
                "r": r,
                "exact_seconds": exact_seconds,
                "exact_iterations": ";".join(map(str, iterations)),
                "exact_solvers": ";".join(solvers),
                "exact_residual": residual,
                "group_sizes": ";".join(map(str, sizes)),
                "age_cuts": ";".join(map(str, cuts)),
                "tau_oracle": format_vector(tau_oracle),
            })

    write_csv(OUT / "normalization_sensitivity.csv", rows)
    write_csv(OUT / "reference_solve_audit.csv", solve_audit)
    metadata = {
        "date": "2026-10-25",
        "nodes": n,
        "edges": int(A.nnz),
        "k": k,
        "r_grid": g2.R_GRID,
        "deltas": g2.DELTAS,
        "gamma_names": g2.GAMMA_NAMES,
        "raw_definition": "min_j Q_j(S)",
        "potential_definition": "min_j Q_j(S)/tau_j^oracle",
        "tau_oracle_definition": "sum of the k largest exact raw gains for each group",
        "candidate_control": "union of exact raw and potential-normalized G2 candidate pools",
        "efficiency_control": "same exact efficiency threshold for both objectives",
        "optimizer": "candidate-pool MILP; zero reported gap required",
        "runtime_scope": "offline diagnostic; excluded from deployment timing",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - start_all,
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_report(rows, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    run(args.n, args.k)
