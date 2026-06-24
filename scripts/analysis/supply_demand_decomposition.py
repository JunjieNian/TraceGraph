#!/usr/bin/env python3
"""RQ3 — Capability decomposition: model×benchmark bilinear factorization.

Decomposes the 5×5 resolve_rate matrix into model capability supply × benchmark demand.

Step 1: Supply vectors a_m ∈ R^4 (task-centered residual of 4-axis profile)
Step 2: Demand vectors d_b ∈ R^4
  - Primary: weighted label-contrast (success vs failure weighted means)
  - Secondary: OLS regression (saved separately as demand_ols.json)
Step 3: Bilinear decomposition: ResolveRate_{m,b} ≈ μ + a_m^T d_b + η_m + γ_b
Step 4: Transfer prediction (leave-one-benchmark-out with bootstrap CI + permutation test)
Step 5: Mismatch analysis (additive residual: actual - (μ + η_m + γ_b))

Depends on: script 91 (capability_axes.json must exist)

Output: results/cxcmu/capability_decomp/
  - supply_demand.json    (primary: weighted label-contrast demand)
  - demand_ols.json       (secondary: OLS demand)
  - bilinear_fit.json
  - transfer_prediction.json
  - mismatch_analysis.json

Usage:
    python scripts/92_capability_decomposition.py [--n-bootstrap 1000] [--n-perm 500]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path("results/cxcmu/capability_decomp")
DYNAMICS_PATH = Path("results/cxcmu/typed_state_dynamics.jsonl")
AXES_PATH = Path("results/cxcmu/enhanced_analysis/capability_axes.json")

AXES = ["productive_reachability", "gate_commitment", "basin_avoidance", "recovery"]

METRICS = [
    "core_visit_rate", "committor_dp", "mfpt_to_core",
    "basin_entry_rate", "basin_escape_rate", "escape_hazard",
    "return_prob", "res_vs_fail_tv",
]


def load_dynamics_data() -> list[dict]:
    data = []
    with open(DYNAMICS_PATH) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _compute_z_params(data: list[dict]) -> dict[str, tuple[float, float]]:
    """Compute mean/std for z-scoring each metric."""
    raw = defaultdict(list)
    for row in data:
        for model_id, metrics in row.get("per_model", {}).items():
            for m in METRICS:
                val = metrics.get(m)
                if val is not None:
                    raw[m].append(val)
    params = {}
    for m, vals in raw.items():
        arr = np.array(vals)
        params[m] = (float(np.mean(arr)), float(np.std(arr)))
    return params


def _z(val, metric, z_params):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    mean, std = z_params.get(metric, (0.0, 1.0))
    if std < 1e-8:
        return 0.0
    return (val - mean) / std


def _axes_for_metrics(metrics: dict, z_params: dict) -> dict[str, float]:
    cv = _z(metrics.get("core_visit_rate"), "core_visit_rate", z_params)
    mfpt = _z(metrics.get("mfpt_to_core"), "mfpt_to_core", z_params)
    cdp = _z(metrics.get("committor_dp"), "committor_dp", z_params)
    be = _z(metrics.get("basin_entry_rate"), "basin_entry_rate", z_params)
    eh = _z(metrics.get("escape_hazard"), "escape_hazard", z_params)
    besc = _z(metrics.get("basin_escape_rate"), "basin_escape_rate", z_params)
    rp = _z(metrics.get("return_prob"), "return_prob", z_params)
    return {
        "productive_reachability": cv - mfpt,
        "gate_commitment": cdp,
        "basin_avoidance": -(be + eh),
        "recovery": besc + rp,
    }


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Supply Vectors
# ═══════════════════════════════════════════════════════════════════════

def _collect_raw_axes(data: list[dict], z_params: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Collect raw global-z-scored axes for every (model, task) pair.

    Returns: {task_id: {model_id: {axis: value}}}
    """
    task_model_axes: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in data:
        task_id = row.get("task_id", row.get("benchmark", "unknown"))
        bench = row.get("benchmark", "")
        # Use (benchmark, task_id) as unique task key
        task_key = f"{bench}::{task_id}"
        for model_id, metrics in row.get("per_model", {}).items():
            axes = _axes_for_metrics(metrics, z_params)
            task_model_axes[task_key][model_id] = axes
    return task_model_axes


def _task_center_axes(
    task_model_axes: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Center axes within each task: z̃_{m,t,a} = z_{m,t,a} - mean_{m'∈M_t}(z_{m',t,a}).

    Returns: {task_key: {model_id: {axis: centered_value}}}
    """
    centered: dict[str, dict[str, dict[str, float]]] = {}
    for task_key, model_dict in task_model_axes.items():
        models_in_task = list(model_dict.keys())
        if not models_in_task:
            continue
        # Compute task mean per axis
        task_mean = {}
        for a in AXES:
            vals = [model_dict[m].get(a, 0.0) for m in models_in_task]
            task_mean[a] = np.mean(vals)
        # Center
        centered[task_key] = {}
        for m in models_in_task:
            centered[task_key][m] = {
                a: model_dict[m].get(a, 0.0) - task_mean[a] for a in AXES
            }
    return centered


def compute_supply_vectors(data: list[dict], z_params: dict) -> dict[str, np.ndarray]:
    """Per model: average of task-centered 4-axis residuals across all tasks.

    1. Compute raw axes z_{m,t,a} for all (model, task) pairs using global z-scores.
    2. Center within each task: z̃_{m,t,a} = z_{m,t,a} - (1/|M_t|) Σ_{m'} z_{m',t,a}
    3. Supply vector = average of task-centered residuals per model.
    """
    task_model_axes = _collect_raw_axes(data, z_params)
    centered = _task_center_axes(task_model_axes)

    model_residuals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for task_key, model_dict in centered.items():
        for model_id, axes_dict in model_dict.items():
            for a in AXES:
                model_residuals[model_id][a].append(axes_dict[a])

    supply = {}
    for model_id, axis_data in model_residuals.items():
        vec = np.array([np.mean(axis_data[a]) for a in AXES])
        supply[model_id] = vec

    return supply


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Demand Vectors
# ═══════════════════════════════════════════════════════════════════════

def compute_demand_vectors(data: list[dict], z_params: dict) -> dict[str, np.ndarray]:
    """Per benchmark: weighted label-contrast demand.

    D_{b,a} = weighted_mean(z̃ | rr) - weighted_mean(z̃ | 1-rr)

    where rr_{m,t} is the resolve_rate for model m on task t, and z̃ is the
    task-centered axis value.
    """
    # Collect task-centered axes
    task_model_axes = _collect_raw_axes(data, z_params)
    centered = _task_center_axes(task_model_axes)

    # Group by benchmark with resolve_rate
    bench_records: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    for row in data:
        bench = row["benchmark"]
        task_id = row.get("task_id", row.get("benchmark", "unknown"))
        task_key = f"{bench}::{task_id}"
        if task_key not in centered:
            continue
        for model_id, metrics in row.get("per_model", {}).items():
            rr = metrics.get("resolve_rate")
            if rr is None:
                continue
            if model_id not in centered[task_key]:
                continue
            bench_records[bench].append((rr, centered[task_key][model_id]))

    demand = {}
    for bench, records in bench_records.items():
        if len(records) < 5:
            continue
        d_vec = []
        for a in AXES:
            # Weighted positive mean (weighted by rr)
            sum_pos_w = sum(rr for rr, _ in records)
            sum_neg_w = sum(1.0 - rr for rr, _ in records)
            if sum_pos_w < 1e-12 or sum_neg_w < 1e-12:
                d_vec.append(0.0)
                continue
            pos_mean = sum(rr * axes[a] for rr, axes in records) / sum_pos_w
            neg_mean = sum((1.0 - rr) * axes[a] for rr, axes in records) / sum_neg_w
            d_vec.append(pos_mean - neg_mean)
        demand[bench] = np.array(d_vec)

    return demand


def compute_demand_vectors_ols(data: list[dict], z_params: dict) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Per benchmark: OLS regression of resolve_rate ~ 4 axes → coefficients = demand.

    Legacy OLS-based demand computation, kept as secondary method.
    """
    bench_data = defaultdict(lambda: {"X": [], "y": []})

    for row in data:
        bench = row["benchmark"]
        for model_id, metrics in row.get("per_model", {}).items():
            rr = metrics.get("resolve_rate")
            if rr is None:
                continue
            axes = _axes_for_metrics(metrics, z_params)
            x = [axes[a] for a in AXES]
            bench_data[bench]["X"].append(x)
            bench_data[bench]["y"].append(rr)

    demand = {}
    demand_r2 = {}
    for bench, bd in bench_data.items():
        X = np.array(bd["X"])
        y = np.array(bd["y"])
        if len(y) < 10 or X.shape[1] == 0:
            continue

        # OLS with intercept
        X_aug = np.column_stack([X, np.ones(len(X))])
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
            y_pred = X_aug @ beta
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            demand[bench] = beta[:4]  # coefficients for 4 axes
            demand_r2[bench] = r2
        except np.linalg.LinAlgError:
            continue

    return demand, demand_r2


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Bilinear Decomposition
# ═══════════════════════════════════════════════════════════════════════

def bilinear_decomposition(
    data: list[dict],
    supply: dict[str, np.ndarray],
    demand: dict[str, np.ndarray],
) -> dict:
    """Fit: ResolveRate_{m,b} ≈ μ + a_m^T d_b + η_m + γ_b.

    Compare R² of:
    - additive: μ + η_m + γ_b
    - bilinear: μ + a_m^T d_b
    - full: μ + a_m^T d_b + η_m + γ_b
    """
    # Build resolve rate matrix
    models = sorted(supply.keys())
    benchmarks = sorted(demand.keys())

    # Average resolve rate per (model, benchmark)
    rr_matrix = defaultdict(lambda: defaultdict(list))
    for row in data:
        bench = row["benchmark"]
        if bench not in benchmarks:
            continue
        for model_id, metrics in row.get("per_model", {}).items():
            if model_id not in models:
                continue
            rr = metrics.get("resolve_rate")
            if rr is not None:
                rr_matrix[model_id][bench].append(rr)

    # Compute averages
    R = np.full((len(models), len(benchmarks)), np.nan)
    for i, m in enumerate(models):
        for j, b in enumerate(benchmarks):
            vals = rr_matrix[m].get(b, [])
            if vals:
                R[i, j] = np.mean(vals)

    # Fill NaN with row/col mean
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isnan(R[i, j]):
                row_mean = np.nanmean(R[i, :])
                col_mean = np.nanmean(R[:, j])
                R[i, j] = (row_mean + col_mean) / 2 if not np.isnan(row_mean + col_mean) else 0.5

    mu = float(np.mean(R))

    # Additive model: R ≈ μ + η_m + γ_b
    eta = R.mean(axis=1) - mu  # model effects
    gamma = R.mean(axis=0) - mu  # benchmark effects
    R_additive = mu + eta[:, None] + gamma[None, :]
    ss_res_add = float(np.sum((R - R_additive) ** 2))
    ss_tot = float(np.sum((R - mu) ** 2))
    r2_additive = 1 - ss_res_add / max(ss_tot, 1e-12)

    # Bilinear model: R ≈ μ + a_m^T d_b
    R_bilinear = np.full_like(R, mu)
    for i, m in enumerate(models):
        for j, b in enumerate(benchmarks):
            if m in supply and b in demand:
                R_bilinear[i, j] = mu + float(supply[m] @ demand[b])
    ss_res_bil = float(np.sum((R - R_bilinear) ** 2))
    r2_bilinear = 1 - ss_res_bil / max(ss_tot, 1e-12)

    # Full model: R ≈ μ + a_m^T d_b + η_m + γ_b
    R_full = np.full_like(R, mu)
    for i, m in enumerate(models):
        for j, b in enumerate(benchmarks):
            bilinear_term = float(supply[m] @ demand[b]) if (m in supply and b in demand) else 0.0
            R_full[i, j] = mu + bilinear_term + eta[i] + gamma[j]
    ss_res_full = float(np.sum((R - R_full) ** 2))
    r2_full = 1 - ss_res_full / max(ss_tot, 1e-12)

    return {
        "models": models,
        "benchmarks": benchmarks,
        "resolve_rate_matrix": {m: {b: round(float(R[i, j]), 4)
                                    for j, b in enumerate(benchmarks)}
                                for i, m in enumerate(models)},
        "mu": round(mu, 4),
        "model_effects_eta": {m: round(float(eta[i]), 4) for i, m in enumerate(models)},
        "benchmark_effects_gamma": {b: round(float(gamma[j]), 4) for j, b in enumerate(benchmarks)},
        "r2_additive": round(r2_additive, 4),
        "r2_bilinear": round(r2_bilinear, 4),
        "r2_full": round(r2_full, 4),
        "interaction_gain": round(r2_full - r2_additive, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Transfer Prediction (LOO)
# ═══════════════════════════════════════════════════════════════════════

def transfer_prediction(
    data: list[dict],
    z_params: dict,
    n_bootstrap: int = 1000,
    n_perm: int = 500,
) -> dict:
    """Leave-one-benchmark-out: predict held-out ranking from supply×demand on rest."""
    rng = np.random.default_rng(42)

    benchmarks = sorted({row["benchmark"] for row in data})
    models = sorted({m for row in data for m in row.get("per_model", {}).keys()})

    if len(benchmarks) < 3 or len(models) < 3:
        return {"error": "Not enough benchmarks or models"}

    loo_results = []
    for held_out in benchmarks:
        train_benchs = [b for b in benchmarks if b != held_out]

        # Compute supply from training benchmarks only
        train_data = [row for row in data if row["benchmark"] in train_benchs]
        supply_train = compute_supply_vectors(train_data, z_params)

        # Compute demand from training benchmarks
        demand_train = compute_demand_vectors(train_data, z_params)

        # Average demand vector from training benchmarks → predict held-out demand
        if not demand_train:
            continue
        avg_demand = np.mean([d for d in demand_train.values()], axis=0)

        # Predict model ranking on held-out = supply × avg_demand
        pred_scores = []
        for m in models:
            if m in supply_train:
                pred_scores.append(float(supply_train[m] @ avg_demand))
            else:
                pred_scores.append(0.0)

        # True resolve rates on held-out
        true_rr = defaultdict(list)
        for row in data:
            if row["benchmark"] != held_out:
                continue
            for m, metrics in row.get("per_model", {}).items():
                rr = metrics.get("resolve_rate")
                if rr is not None:
                    true_rr[m].append(rr)

        true_scores = [np.mean(true_rr[m]) if true_rr[m] else 0.5 for m in models]

        # Kendall τ
        tau, p_tau = scipy_stats.kendalltau(true_scores, pred_scores)
        if math.isnan(tau):
            continue

        # Bootstrap CI for τ
        boot_taus = []
        n = len(models)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            t_boot = [true_scores[i] for i in idx]
            p_boot = [pred_scores[i] for i in idx]
            bt, _ = scipy_stats.kendalltau(t_boot, p_boot)
            if not math.isnan(bt):
                boot_taus.append(bt)

        ci_lo = float(np.percentile(boot_taus, 2.5)) if boot_taus else None
        ci_hi = float(np.percentile(boot_taus, 97.5)) if boot_taus else None

        # Permutation test
        perm_taus = []
        for _ in range(n_perm):
            perm_pred = rng.permutation(pred_scores).tolist()
            pt, _ = scipy_stats.kendalltau(true_scores, perm_pred)
            if not math.isnan(pt):
                perm_taus.append(pt)

        perm_p = (1 + sum(1 for pt in perm_taus if pt >= tau)) / (1 + len(perm_taus)) if perm_taus else 1.0

        loo_results.append({
            "held_out": held_out,
            "tau": round(float(tau), 4),
            "p_tau": round(float(p_tau), 4),
            "bootstrap_ci_95": [round(ci_lo, 4) if ci_lo is not None else None,
                                round(ci_hi, 4) if ci_hi is not None else None],
            "permutation_p": round(perm_p, 4),
        })

    mean_tau = float(np.mean([r["tau"] for r in loo_results])) if loo_results else None
    return {
        "n_folds": len(loo_results),
        "mean_tau": round(mean_tau, 4) if mean_tau is not None else None,
        "folds": loo_results,
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_perm,
    }


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Mismatch Analysis
# ═══════════════════════════════════════════════════════════════════════

def mismatch_analysis(
    data: list[dict],
    supply: dict[str, np.ndarray],
    demand: dict[str, np.ndarray],
    mu: float,
    model_effects_eta: dict[str, float],
    benchmark_effects_gamma: dict[str, float],
) -> list[dict]:
    """Find cases where models over/underperform relative to the additive model.

    Expected rate uses the additive model from bilinear decomposition:
        expected_{m,b} = mu + eta_m + gamma_b
    """
    models = sorted(supply.keys())
    benchmarks = sorted(demand.keys())

    # Actual resolve rate per (model, benchmark)
    model_bench_rr = defaultdict(lambda: defaultdict(list))
    for row in data:
        bench = row["benchmark"]
        for m, metrics in row.get("per_model", {}).items():
            rr = metrics.get("resolve_rate")
            if rr is not None:
                model_bench_rr[m][bench].append(rr)

    mismatches = []
    for m in models:
        if m not in supply:
            continue
        eta_m = model_effects_eta.get(m, 0.0)
        for b in benchmarks:
            if b not in demand:
                continue
            gamma_b = benchmark_effects_gamma.get(b, 0.0)
            actual_rr = np.mean(model_bench_rr[m].get(b, [0.5]))
            expected_rr = mu + eta_m + gamma_b
            residual = actual_rr - expected_rr

            # Report all model-benchmark cells; flag larger deviations
            if True:  # report all cells for analysis
                # Explain via capability alignment
                alignment = float(supply[m] @ demand[b])
                # Which axis contributes most to misalignment?
                axis_contributions = {}
                for k, axis in enumerate(AXES):
                    axis_contributions[axis] = round(float(supply[m][k] * demand[b][k]), 4)

                mismatches.append({
                    "model": m,
                    "benchmark": b,
                    "actual_rr": round(float(actual_rr), 4),
                    "expected_rr": round(float(expected_rr), 4),
                    "residual": round(float(residual), 4),
                    "capability_alignment": round(alignment, 4),
                    "axis_contributions": axis_contributions,
                    "direction": "underperform" if residual < 0 else "overperform",
                })

    mismatches.sort(key=lambda x: abs(x["residual"]), reverse=True)
    return mismatches


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(n_bootstrap: int = 1000, n_perm: int = 500):
    if not DYNAMICS_PATH.exists():
        print(f"Error: {DYNAMICS_PATH} not found.")
        return

    print("Loading dynamics data...")
    data = load_dynamics_data()
    print(f"  {len(data)} task records")

    z_params = _compute_z_params(data)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Supply vectors
    print("\n── Step 1: Supply Vectors ──")
    supply = compute_supply_vectors(data, z_params)
    for m, vec in sorted(supply.items()):
        print(f"  {m}: [{', '.join(f'{v:.3f}' for v in vec)}]")

    # Step 2a: Demand vectors (primary — weighted label-contrast)
    print("\n── Step 2a: Demand Vectors (weighted label-contrast) ──")
    demand = compute_demand_vectors(data, z_params)
    for b, vec in sorted(demand.items()):
        print(f"  {b}: [{', '.join(f'{v:.3f}' for v in vec)}]")

    # Step 2b: Demand vectors (secondary — OLS, saved separately)
    print("\n── Step 2b: Demand Vectors (OLS, secondary) ──")
    demand_ols, demand_r2 = compute_demand_vectors_ols(data, z_params)
    for b, vec in sorted(demand_ols.items()):
        r2 = demand_r2.get(b, 0)
        print(f"  {b}: [{', '.join(f'{v:.3f}' for v in vec)}] (R²={r2:.3f})")

    # Save primary supply/demand
    sd_output = {
        "axes": AXES,
        "supply_vectors": {m: [round(float(v), 4) for v in vec] for m, vec in supply.items()},
        "demand_vectors": {b: [round(float(v), 4) for v in vec] for b, vec in demand.items()},
        "demand_method": "weighted_label_contrast",
    }
    with open(RESULTS_DIR / "supply_demand.json", "w") as f:
        json.dump(sd_output, f, indent=2)

    # Save secondary OLS demand
    ols_output = {
        "axes": AXES,
        "demand_vectors_ols": {b: [round(float(v), 4) for v in vec] for b, vec in demand_ols.items()},
        "demand_r2": {b: round(r2, 4) for b, r2 in demand_r2.items()},
        "demand_method": "ols",
    }
    with open(RESULTS_DIR / "demand_ols.json", "w") as f:
        json.dump(ols_output, f, indent=2)

    # Step 3: Bilinear decomposition
    print("\n── Step 3: Bilinear Decomposition ──")
    bilinear = bilinear_decomposition(data, supply, demand)
    print(f"  R² additive:  {bilinear['r2_additive']}")
    print(f"  R² bilinear:  {bilinear['r2_bilinear']}")
    print(f"  R² full:      {bilinear['r2_full']}")
    print(f"  Interaction gain: {bilinear['interaction_gain']}")

    with open(RESULTS_DIR / "bilinear_fit.json", "w") as f:
        json.dump(bilinear, f, indent=2)

    # Step 4: Transfer prediction
    print(f"\n── Step 4: Transfer Prediction (LOO, boot={n_bootstrap}, perm={n_perm}) ──")
    transfer = transfer_prediction(data, z_params, n_bootstrap, n_perm)
    if "error" not in transfer:
        print(f"  Mean LOO τ: {transfer['mean_tau']}")
        for fold in transfer.get("folds", []):
            print(f"    {fold['held_out']}: τ={fold['tau']}, "
                  f"CI={fold['bootstrap_ci_95']}, perm_p={fold['permutation_p']}")
    else:
        print(f"  {transfer['error']}")

    with open(RESULTS_DIR / "transfer_prediction.json", "w") as f:
        json.dump(transfer, f, indent=2)

    # Step 5: Mismatch analysis (using additive model from bilinear decomposition)
    print("\n── Step 5: Mismatch Analysis ──")
    mismatches = mismatch_analysis(
        data, supply, demand,
        mu=bilinear["mu"],
        model_effects_eta=bilinear["model_effects_eta"],
        benchmark_effects_gamma=bilinear["benchmark_effects_gamma"],
    )
    print(f"  {len(mismatches)} mismatches (|residual| > 0.15)")
    for mm in mismatches[:5]:
        print(f"    {mm['model']} on {mm['benchmark']}: "
              f"actual={mm['actual_rr']}, expected={mm['expected_rr']}, "
              f"Δ={mm['residual']} ({mm['direction']})")

    with open(RESULTS_DIR / "mismatch_analysis.json", "w") as f:
        json.dump(mismatches, f, indent=2)

    print(f"\n── Output ──")
    print(f"  {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=500)
    args = parser.parse_args()
    main(n_bootstrap=args.n_bootstrap, n_perm=args.n_perm)
