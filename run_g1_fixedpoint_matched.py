from __future__ import annotations

import time

import numpy as np
import psutil

import run_g1_pokec as g1


THRESHOLDS = (0.05, 0.02, 0.01, 0.005, 0.002)


def run(n=100000, k_budget=50):
    g1.OUT.mkdir(parents=True, exist_ok=True)
    A, gender, age = g1.load_pokec(n)
    degree = np.asarray(A.sum(axis=1)).ravel()
    groups = {r: g1.make_groups(gender, age, r) for r in g1.R_GRID}
    rng = np.random.default_rng(g1.SEED)
    innate = rng.uniform(0.0, 1.0, n)
    gammas = g1.gamma_vectors(degree, rng)
    rows = []
    for gamma_name, gamma in gammas.items():
        for r in g1.R_GRID:
            _, W, _, _ = groups[r]
            Htrue, krylov_time, _, residual = g1.exact_krylov(A, degree, gamma, W)
            raw = (1.0-innate)[:, None]*Htrue
            tau = np.array([np.partition(raw[:,j],-k_budget)[-k_budget:].sum() for j in range(r)])
            Gtrue = raw/np.maximum(tau,1e-15)[None,:]
            reference_set, _, _ = g1.candidate_fair_optimizer(Gtrue,k_budget)
            reference_value=float(Gtrue[reference_set].sum(axis=0).min())
            H=np.zeros_like(W); elapsed=0.0; remaining=list(THRESHOLDS)
            for iteration in range(1,1001):
                t0=time.perf_counter(); H=g1.refine(A,degree,gamma,H,W,1); elapsed+=time.perf_counter()-t0
                Gest=((1.0-innate)[:,None]*H)/np.maximum(tau,1e-15)[None,:]
                error=float(np.max(np.abs(Gest-Gtrue)))
                crossed=[x for x in remaining if error<=x]
                for threshold in crossed:
                    selected,opt_t,pool=g1.candidate_fair_optimizer(Gest,k_budget)
                    fair=float(Gtrue[selected].sum(axis=0).min())
                    rows.append({
                        "dataset":f"soc-Pokec-{n}","gamma":gamma_name,"r":r,
                        "method":"Batched-Fixed-Point","target_gain_error":threshold,
                        "achieved_gain_error":error,"iterations":iteration,
                        "estimation_seconds":elapsed,"optimization_seconds":opt_t,
                        "end_to_end_seconds":elapsed+opt_t,"true_fair_value":fair,
                        "reference_regret_ratio":g1.safe_regret(reference_value,fair),
                        "candidate_pool":pool,"krylov_reference_seconds":krylov_time,
                        "krylov_residual":residual,"peak_rss_mib":psutil.Process().memory_info().rss/2**20,
                    })
                    remaining.remove(threshold)
                if not remaining:
                    break
            if remaining:
                raise RuntimeError(f"thresholds not reached for {gamma_name}, r={r}: {remaining}")
            print(gamma_name,r,"iterations",iteration,"seconds",elapsed,flush=True)
    g1.write_csv(g1.OUT/"g1_fixedpoint_matched.csv",rows)


if __name__ == "__main__":
    run()
