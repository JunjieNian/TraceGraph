#!/usr/bin/env python3
"""RQ2 — Enhanced separability: graph metrics surpass baselines in predicting success.

Part A: Task fixed-effect logistic regression
  M0: model-only (no graph metrics)
  M1: graph-only (no model dummies)
  M2: full (model + graph)
  Compare: McFadden R², LR test

Part B: Baselines
  - outcome_only: raw resolve_rate ranking
  - trajectory_length: average step count
  - tool_count: unique tools used
  - action_ngram: action-type bigram transition entropy
  Compare: Spearman ρ and Kendall τ with resolve_rate ranking

Part C: 4 Capability Axes (task-centered residuals of z-scored composites)
  - Productive Reachability = z(core_visit) - z(mfpt_to_core)
  - Gate Commitment         = z(committor_dp)
  - Basin Avoidance         = -(z(basin_entry) + z(escape_hazard))
  - Recovery                = z(basin_escape) + z(return_prob)
  After computing raw z-scored axes, task-center by subtracting the per-task
  mean across all models: z̃_{m,t,a} = z_{m,t,a} - mean_task(z_{*,t,a})

Part D: Composite metrics
  - productive_navigation: P(visit core before entering any basin)
  - safe_exploration: core_visit_rate - λ·basin_entry_rate

Part E: Bootstrap CI (1000 resamples per model per metric)

Output: results/cxcmu/enhanced_analysis/
  - logistic_regression.json
  - baselines.json
  - capability_axes.json
  - bootstrap_ci.json

Usage:
    python scripts/91_enhanced_separability.py [--n-bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

RESULTS_DIR = Path("results/cxcmu/enhanced_analysis")
DYNAMICS_PATH = Path("results/cxcmu/typed_state_dynamics.jsonl")
PARSED_DIR = Path("data/cxcmu/parsed")

METRICS = [
    "core_visit_rate",
    "committor_dp",
    "mfpt_to_core",
    "basin_entry_rate",
    "basin_escape_rate",
    "escape_hazard",
    "return_prob",
    "res_vs_fail_tv",
]

METRIC_SIGN = {
    "core_visit_rate": +1,
    "committor_dp": +1,
    "mfpt_to_core": -1,
    "basin_entry_rate": -1,
    "basin_escape_rate": +1,
    "escape_hazard": -1,
    "return_prob": +1,
    "res_vs_fail_tv": +1,
}


def load_dynamics_data() -> list[dict]:
    data = []
    with open(DYNAMICS_PATH) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _build_observation_table(data: list[dict]) -> list[dict]:
    """Build flat observation table: one row per (task, model) with metrics + outcome."""
    rows = []
    for task_row in data:
        bench = task_row["benchmark"]
        task_id = task_row["task_id"]
        for model_id, metrics in task_row.get("per_model", {}).items():
            row = {
                "benchmark": bench,
                "task_id": task_id,
                "model_id": model_id,
                "resolve_rate": metrics.get("resolve_rate", 0.0),
                "n_runs": metrics.get("n_runs", 0),
                "n_resolved": metrics.get("n_resolved", 0),
            }
            for m in METRICS:
                row[m] = metrics.get(m)
            rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Part A: Logistic Regression
# ═══════════════════════════════════════════════════════════════════════

def _logistic_regression(obs_table: list[dict]) -> dict:
    """Fixed-effect logistic regression with model dummies and graph metrics."""
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit
    except ImportError:
        return {"error": "statsmodels not installed; skipping logistic regression"}

    # Prepare data
    models = sorted({r["model_id"] for r in obs_table})
    benchmarks = sorted({r["benchmark"] for r in obs_table})
    tasks = sorted({r["task_id"] for r in obs_table})

    # Encode
    y = []
    X_model = []  # model dummies
    X_graph = []  # graph metrics
    X_task = []   # task fixed effects (benchmark dummies as proxy)

    for row in obs_table:
        # Binary outcome: any success in this model's runs on this task
        outcome = 1.0 if row["resolve_rate"] > 0 else 0.0
        y.append(outcome)

        # Model dummies (reference = first model)
        m_dummy = [1.0 if row["model_id"] == m else 0.0 for m in models[1:]]
        X_model.append(m_dummy)

        # Graph metrics (impute NaN with 0)
        g_feats = []
        for m in METRICS:
            val = row.get(m)
            g_feats.append(float(val) if val is not None else 0.0)
        X_graph.append(g_feats)

        # Benchmark dummies (fixed effects proxy)
        b_dummy = [1.0 if row["benchmark"] == b else 0.0 for b in benchmarks[1:]]
        X_task.append(b_dummy)

    y = np.array(y)
    X_model = np.array(X_model)
    X_graph = np.array(X_graph)
    X_task = np.array(X_task)

    # Standardize graph metrics
    for col in range(X_graph.shape[1]):
        col_std = X_graph[:, col].std()
        if col_std > 1e-8:
            X_graph[:, col] = (X_graph[:, col] - X_graph[:, col].mean()) / col_std

    # Check if there's variance in y
    if y.sum() == 0 or y.sum() == len(y):
        return {"error": "No variance in outcome variable"}

    results = {}

    # M0: model-only (+ task fixed effects)
    X0 = np.column_stack([X_task, X_model, np.ones((len(y), 1))])
    try:
        m0 = Logit(y, X0).fit(disp=0, maxiter=100)
        results["M0_model_only"] = {
            "llf": float(m0.llf),
            "llnull": float(m0.llnull),
            "mcfadden_r2": float(1 - m0.llf / m0.llnull),
            "aic": float(m0.aic),
            "n_params": int(m0.df_model),
        }
    except Exception as e:
        results["M0_model_only"] = {"error": str(e)}

    # M1: graph-only (+ task fixed effects)
    X1 = np.column_stack([X_task, X_graph, np.ones((len(y), 1))])
    try:
        m1 = Logit(y, X1).fit(disp=0, maxiter=100)
        results["M1_graph_only"] = {
            "llf": float(m1.llf),
            "llnull": float(m1.llnull),
            "mcfadden_r2": float(1 - m1.llf / m1.llnull),
            "aic": float(m1.aic),
            "n_params": int(m1.df_model),
        }
    except Exception as e:
        results["M1_graph_only"] = {"error": str(e)}

    # M2: full (model + graph + task fixed effects)
    X2 = np.column_stack([X_task, X_model, X_graph, np.ones((len(y), 1))])
    try:
        m2 = Logit(y, X2).fit(disp=0, maxiter=100)
        results["M2_full"] = {
            "llf": float(m2.llf),
            "llnull": float(m2.llnull),
            "mcfadden_r2": float(1 - m2.llf / m2.llnull),
            "aic": float(m2.aic),
            "n_params": int(m2.df_model),
            "graph_coefs": {METRICS[i]: round(float(m2.params[X_task.shape[1] + X_model.shape[1] + i]), 4)
                           for i in range(len(METRICS))},
        }
    except Exception as e:
        results["M2_full"] = {"error": str(e)}

    # LR test: M2 vs M0 (does adding graph metrics help beyond model identity?)
    if "llf" in results.get("M0_model_only", {}) and "llf" in results.get("M2_full", {}):
        lr_stat = 2 * (results["M2_full"]["llf"] - results["M0_model_only"]["llf"])
        df_diff = results["M2_full"]["n_params"] - results["M0_model_only"]["n_params"]
        if df_diff > 0 and lr_stat > 0:
            lr_p = float(scipy_stats.chi2.sf(lr_stat, df_diff))
            results["LR_test_M2_vs_M0"] = {
                "chi2": round(lr_stat, 4),
                "df": df_diff,
                "p_value": round(lr_p, 6),
                "significant": lr_p < 0.05,
            }

    # LR test: M2 vs M1 (does model identity add beyond graph metrics?)
    if "llf" in results.get("M1_graph_only", {}) and "llf" in results.get("M2_full", {}):
        lr_stat = 2 * (results["M2_full"]["llf"] - results["M1_graph_only"]["llf"])
        df_diff = results["M2_full"]["n_params"] - results["M1_graph_only"]["n_params"]
        if df_diff > 0 and lr_stat > 0:
            lr_p = float(scipy_stats.chi2.sf(lr_stat, df_diff))
            results["LR_test_M2_vs_M1"] = {
                "chi2": round(lr_stat, 4),
                "df": df_diff,
                "p_value": round(lr_p, 6),
                "significant": lr_p < 0.05,
            }

    results["n_observations"] = len(y)
    results["n_success"] = int(y.sum())
    return results


# ═══════════════════════════════════════════════════════════════════════
# Part B: Baselines
# ═══════════════════════════════════════════════════════════════════════

def _compute_baselines(data: list[dict]) -> dict:
    """Compute baseline metrics and compare with graph-metric ranking."""
    # Load parsed data for trajectory-level baselines
    model_baselines = defaultdict(lambda: defaultdict(list))  # model → metric → values
    model_resolve = defaultdict(list)

    for task_row in data:
        bench = task_row["benchmark"]
        task_id = task_row["task_id"]

        # Load parsed trajectories for length/tool baselines
        parsed_path = PARSED_DIR / bench / f"{task_id}.jsonl"
        task_parsed = {}
        if parsed_path.exists():
            with open(parsed_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        task_parsed[r.get("model_id", "unknown")] = task_parsed.get(
                            r.get("model_id", "unknown"), []
                        )
                        task_parsed.setdefault(r["model_id"], []).append(r)

        for model_id, metrics in task_row.get("per_model", {}).items():
            rr = metrics.get("resolve_rate", 0.0)
            model_resolve[model_id].append(rr)

            # Trajectory length
            if model_id in task_parsed:
                lengths = [r.get("n_steps", 0) for r in task_parsed[model_id]]
                model_baselines[model_id]["trajectory_length"].append(
                    np.mean(lengths) if lengths else 0
                )

                # Tool count
                tools_per_run = []
                for r in task_parsed[model_id]:
                    tools = {s.get("tool_name", "") for s in r.get("steps", []) if s.get("tool_name")}
                    tools_per_run.append(len(tools))
                model_baselines[model_id]["tool_count"].append(
                    np.mean(tools_per_run) if tools_per_run else 0
                )

                # Action n-gram entropy (bigram transition entropy)
                for r in task_parsed[model_id]:
                    actions = [s.get("action_type", "other") for s in r.get("steps", [])]
                    if len(actions) >= 2:
                        bigrams = [(actions[i], actions[i+1]) for i in range(len(actions)-1)]
                        counts = defaultdict(int)
                        for bg in bigrams:
                            counts[bg] += 1
                        total = sum(counts.values())
                        entropy = -sum(
                            (c / total) * math.log2(c / total)
                            for c in counts.values() if c > 0
                        )
                        model_baselines[model_id]["action_ngram"].append(entropy)

    # Compute model-level averages
    models = sorted(model_resolve.keys())
    model_avg_resolve = {m: np.mean(model_resolve[m]) for m in models}

    # True ranking by resolve rate
    true_ranking = sorted(models, key=lambda m: model_avg_resolve[m], reverse=True)
    true_scores = [model_avg_resolve[m] for m in models]

    results = {"models": models, "true_resolve_rates": {m: round(model_avg_resolve[m], 4) for m in models}}
    baseline_metrics = ["trajectory_length", "tool_count", "action_ngram"]

    for bl_name in baseline_metrics:
        bl_scores = []
        for m in models:
            vals = model_baselines[m].get(bl_name, [])
            bl_scores.append(np.mean(vals) if vals else 0.0)

        if len(bl_scores) >= 3:
            rho, p_rho = scipy_stats.spearmanr(true_scores, bl_scores)
            tau, p_tau = scipy_stats.kendalltau(true_scores, bl_scores)
            results[bl_name] = {
                "model_values": {m: round(float(bl_scores[i]), 4) for i, m in enumerate(models)},
                "spearman_rho": round(float(rho), 4) if not math.isnan(rho) else None,
                "spearman_p": round(float(p_rho), 6) if not math.isnan(p_rho) else None,
                "kendall_tau": round(float(tau), 4) if not math.isnan(tau) else None,
                "kendall_p": round(float(p_tau), 6) if not math.isnan(p_tau) else None,
            }

    # Graph metric rankings (from dynamics data)
    for metric in METRICS:
        sign = METRIC_SIGN[metric]
        metric_scores = []
        for m in models:
            vals = []
            for task_row in data:
                pm = task_row.get("per_model", {}).get(m, {})
                v = pm.get(metric)
                if v is not None:
                    vals.append(v)
            metric_scores.append(np.mean(vals) * sign if vals else 0.0)

        if len(metric_scores) >= 3:
            rho, p_rho = scipy_stats.spearmanr(true_scores, metric_scores)
            tau, p_tau = scipy_stats.kendalltau(true_scores, metric_scores)
            results[f"graph_{metric}"] = {
                "spearman_rho": round(float(rho), 4) if not math.isnan(rho) else None,
                "spearman_p": round(float(p_rho), 6) if not math.isnan(p_rho) else None,
                "kendall_tau": round(float(tau), 4) if not math.isnan(tau) else None,
                "kendall_p": round(float(p_tau), 6) if not math.isnan(p_tau) else None,
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Part C: 4 Capability Axes
# ═══════════════════════════════════════════════════════════════════════

def _compute_capability_axes(obs_table: list[dict]) -> dict:
    """Compute 4 task-centered capability axis composites per model.

    Steps:
      1. Global z-score each raw metric (for scale comparability).
      2. Combine z-scored metrics into 4 axis composites per observation.
      3. Task-center: for each task, subtract the task mean across all models
         that attempted that task.  This yields residuals that isolate each
         model's navigation *style* from task difficulty:
         z̃_{m,t,a} = z_{m,t,a} - mean_task(z_{*,t,a})
      4. Aggregate task-centered residuals into model / benchmark profiles
         and axis–reward correlations.
    """
    # Collect raw values per observation
    raw = defaultdict(list)
    for row in obs_table:
        for m in METRICS:
            val = row.get(m)
            if val is not None:
                raw[m].append(val)
            else:
                raw[m].append(np.nan)

    # Step 1: Global z-score each metric (for scale comparability)
    z_scores = {}
    for m in METRICS:
        arr = np.array(raw[m], dtype=float)
        valid = ~np.isnan(arr)
        if valid.sum() > 1:
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            if std > 1e-8:
                z = (arr - mean) / std
            else:
                z = np.zeros_like(arr)
        else:
            z = np.zeros(len(arr))
        z_scores[m] = z

    # Step 2: Compute 4 raw axes per observation
    n = len(obs_table)
    axes = {
        "productive_reachability": np.zeros(n),
        "gate_commitment": np.zeros(n),
        "basin_avoidance": np.zeros(n),
        "recovery": np.zeros(n),
    }

    for i in range(n):
        # Core Reachability = z(core_visit) - z(mfpt_to_core)
        cv = z_scores["core_visit_rate"][i] if not np.isnan(z_scores["core_visit_rate"][i]) else 0.0
        mfpt = z_scores["mfpt_to_core"][i] if not np.isnan(z_scores["mfpt_to_core"][i]) else 0.0
        axes["productive_reachability"][i] = cv - mfpt

        # Decision Commitment = z(committor_dp)
        cdp = z_scores["committor_dp"][i] if not np.isnan(z_scores["committor_dp"][i]) else 0.0
        axes["gate_commitment"][i] = cdp

        # Basin Vulnerability = -(z(basin_entry) + z(escape_hazard))
        be = z_scores["basin_entry_rate"][i] if not np.isnan(z_scores["basin_entry_rate"][i]) else 0.0
        eh = z_scores["escape_hazard"][i] if not np.isnan(z_scores["escape_hazard"][i]) else 0.0
        axes["basin_avoidance"][i] = -(be + eh)

        # Recovery Ability = z(basin_escape) + z(return_prob)
        besc = z_scores["basin_escape_rate"][i] if not np.isnan(z_scores["basin_escape_rate"][i]) else 0.0
        rp = z_scores["return_prob"][i] if not np.isnan(z_scores["return_prob"][i]) else 0.0
        axes["recovery"][i] = besc + rp

    # Step 3: Task-center — subtract per-task mean to isolate model style
    # Group observation indices by task_id
    task_indices = defaultdict(list)
    for i, row in enumerate(obs_table):
        task_indices[row["task_id"]].append(i)

    for axis_name in axes:
        for task_id, idxs in task_indices.items():
            task_vals = axes[axis_name][idxs]
            task_mean = float(np.mean(task_vals))
            axes[axis_name][idxs] -= task_mean

    # Step 4: Aggregate task-centered residuals
    # Average per model
    model_axes = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(obs_table):
        model_id = row["model_id"]
        for axis_name, arr in axes.items():
            model_axes[model_id][axis_name].append(arr[i])

    model_profiles = {}
    for model_id, axis_vals in model_axes.items():
        model_profiles[model_id] = {
            axis: round(float(np.mean(vals)), 4)
            for axis, vals in axis_vals.items()
        }

    # Per-benchmark model profiles
    bench_model_axes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, row in enumerate(obs_table):
        for axis_name, arr in axes.items():
            bench_model_axes[row["benchmark"]][row["model_id"]][axis_name].append(arr[i])

    bench_profiles = {}
    for bench, models_data in bench_model_axes.items():
        bench_profiles[bench] = {}
        for model_id, axis_vals in models_data.items():
            bench_profiles[bench][model_id] = {
                axis: round(float(np.mean(vals)), 4)
                for axis, vals in axis_vals.items()
            }

    # Correlation of task-centered axes with resolve rate
    axis_reward_corr = {}
    resolve_arr = np.array([row["resolve_rate"] for row in obs_table])
    for axis_name, arr in axes.items():
        valid = ~np.isnan(arr) & ~np.isnan(resolve_arr)
        if valid.sum() >= 10:
            rho, p = scipy_stats.spearmanr(arr[valid], resolve_arr[valid])
            axis_reward_corr[axis_name] = {
                "spearman_rho": round(float(rho), 4) if not math.isnan(rho) else None,
                "p_value": round(float(p), 6) if not math.isnan(p) else None,
            }

    return {
        "axes": ["productive_reachability", "gate_commitment", "basin_avoidance", "recovery"],
        "model_profiles": model_profiles,
        "benchmark_profiles": bench_profiles,
        "axis_reward_correlation": axis_reward_corr,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part D: Composite Metrics
# ═══════════════════════════════════════════════════════════════════════

def _compute_composite_metrics(obs_table: list[dict]) -> dict:
    """Compute productive_navigation and safe_exploration composites."""
    LAMBDA = 1.5  # penalty weight for basin entry

    model_composites = defaultdict(lambda: defaultdict(list))

    for row in obs_table:
        model_id = row["model_id"]
        cvr = row.get("core_visit_rate")
        ber = row.get("basin_entry_rate")

        if cvr is not None and ber is not None:
            # safe_exploration: core_visit - λ·basin_entry
            safe = cvr - LAMBDA * ber
            model_composites[model_id]["safe_exploration"].append(safe)

            # productive_navigation: binary — visited core AND didn't enter basin
            productive = 1.0 if (cvr > 0 and ber == 0) else 0.0
            model_composites[model_id]["productive_navigation"].append(productive)

    results = {}
    for model_id, composites in sorted(model_composites.items()):
        results[model_id] = {}
        for comp_name, vals in composites.items():
            results[model_id][comp_name] = round(float(np.mean(vals)), 4)

    return {"lambda": LAMBDA, "model_composites": results}


# ═══════════════════════════════════════════════════════════════════════
# Part E: Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════

def _bootstrap_ci(obs_table: list[dict], n_bootstrap: int = 1000) -> dict:
    """Bootstrap 95% CI for each model's mean on each metric + capability axis.

    Capability axes are task-centered: after computing raw global-z axes for
    every observation, we subtract the per-task mean so that bootstrap
    resamples reflect model navigation style, not task difficulty.
    """
    rng = np.random.default_rng(42)

    # Group by model
    model_data = defaultdict(list)
    for row in obs_table:
        model_data[row["model_id"]].append(row)

    all_metrics = METRICS + ["productive_reachability", "gate_commitment",
                             "basin_avoidance", "recovery"]
    axis_names = ["productive_reachability", "gate_commitment",
                  "basin_avoidance", "recovery"]

    # Precompute capability axes for each observation
    # (reuse z-score logic)
    raw_arrs = {m: [] for m in METRICS}
    for row in obs_table:
        for m in METRICS:
            val = row.get(m)
            raw_arrs[m].append(float(val) if val is not None else np.nan)

    z_means = {}
    z_stds = {}
    for m in METRICS:
        arr = np.array(raw_arrs[m])
        z_means[m] = float(np.nanmean(arr))
        z_stds[m] = float(np.nanstd(arr))
        if z_stds[m] < 1e-8:
            z_stds[m] = 1.0

    def _z(val, metric):
        if val is None or np.isnan(val):
            return 0.0
        return (val - z_means[metric]) / z_stds[metric]

    def _axes_for_row(row):
        cv = _z(row.get("core_visit_rate"), "core_visit_rate")
        mfpt = _z(row.get("mfpt_to_core"), "mfpt_to_core")
        cdp = _z(row.get("committor_dp"), "committor_dp")
        be = _z(row.get("basin_entry_rate"), "basin_entry_rate")
        eh = _z(row.get("escape_hazard"), "escape_hazard")
        besc = _z(row.get("basin_escape_rate"), "basin_escape_rate")
        rp = _z(row.get("return_prob"), "return_prob")
        return {
            "productive_reachability": cv - mfpt,
            "gate_commitment": cdp,
            "basin_avoidance": -(be + eh),
            "recovery": besc + rp,
        }

    # Compute raw axes for every observation, then task-center
    obs_axes = [_axes_for_row(row) for row in obs_table]

    # Group observation indices by task_id
    task_indices = defaultdict(list)
    for i, row in enumerate(obs_table):
        task_indices[row["task_id"]].append(i)

    # Task-center: subtract per-task mean for each axis
    for axis in axis_names:
        for task_id, idxs in task_indices.items():
            task_mean = np.mean([obs_axes[i][axis] for i in idxs])
            for i in idxs:
                obs_axes[i][axis] -= task_mean

    # Build index mapping: for each model, which obs_table indices belong to it
    model_indices = defaultdict(list)
    for i, row in enumerate(obs_table):
        model_indices[row["model_id"]].append(i)

    results = {}
    for model_id, m_idxs in sorted(model_indices.items()):
        n = len(m_idxs)
        if n < 5:
            continue

        model_rows = [obs_table[i] for i in m_idxs]
        model_obs_axes = [obs_axes[i] for i in m_idxs]

        model_results = {}
        for metric in all_metrics:
            # Get values
            if metric in METRICS:
                vals = [row.get(metric) for row in model_rows]
                vals = [v for v in vals if v is not None]
            else:
                # Use task-centered axis values
                vals = [oa[metric] for oa in model_obs_axes]

            if len(vals) < 5:
                continue

            vals_arr = np.array(vals)
            observed_mean = float(np.mean(vals_arr))

            # Bootstrap
            boot_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(vals_arr, size=len(vals_arr), replace=True)
                boot_means.append(float(np.mean(sample)))

            boot_means = np.array(boot_means)
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            model_results[metric] = {
                "mean": round(observed_mean, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
                "se": round(float(np.std(boot_means)), 4),
            }

        results[model_id] = model_results

    return {"n_bootstrap": n_bootstrap, "model_ci": results}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(n_bootstrap: int = 1000):
    if not DYNAMICS_PATH.exists():
        print(f"Error: {DYNAMICS_PATH} not found. Run 86_typed_state_cxcmu.py first.")
        return

    print("Loading dynamics data...")
    data = load_dynamics_data()
    print(f"  {len(data)} task records")

    obs_table = _build_observation_table(data)
    print(f"  {len(obs_table)} observations (task × model)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Part A: Logistic regression
    print("\n── Part A: Logistic Regression ──")
    lr_results = _logistic_regression(obs_table)
    if "error" not in lr_results:
        for model_name in ["M0_model_only", "M1_graph_only", "M2_full"]:
            r = lr_results.get(model_name, {})
            r2 = r.get("mcfadden_r2")
            if r2 is not None:
                print(f"  {model_name}: McFadden R² = {r2:.4f}")
        lr_test = lr_results.get("LR_test_M2_vs_M0", {})
        if lr_test:
            print(f"  LR test (M2 vs M0): χ²={lr_test.get('chi2')}, p={lr_test.get('p_value')}")
    else:
        print(f"  {lr_results['error']}")

    with open(RESULTS_DIR / "logistic_regression.json", "w") as f:
        json.dump(lr_results, f, indent=2, default=str)

    # Part B: Baselines
    print("\n── Part B: Baselines ──")
    baseline_results = _compute_baselines(data)
    for key in ["trajectory_length", "tool_count", "action_ngram"]:
        br = baseline_results.get(key, {})
        rho = br.get("spearman_rho")
        if rho is not None:
            print(f"  {key}: ρ={rho}")
    # Show graph metrics
    graph_rhos = []
    for metric in METRICS:
        gr = baseline_results.get(f"graph_{metric}", {})
        rho = gr.get("spearman_rho")
        if rho is not None:
            graph_rhos.append((metric, rho))
    graph_rhos.sort(key=lambda x: abs(x[1]), reverse=True)
    for m, r in graph_rhos[:3]:
        print(f"  graph_{m}: ρ={r}")

    with open(RESULTS_DIR / "baselines.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)

    # Part C: Capability Axes
    print("\n── Part C: Capability Axes ──")
    axes_results = _compute_capability_axes(obs_table)
    for axis, corr in axes_results.get("axis_reward_correlation", {}).items():
        print(f"  {axis}: ρ={corr.get('spearman_rho')}")
    print("  Model profiles:")
    for model_id, profile in sorted(axes_results.get("model_profiles", {}).items()):
        print(f"    {model_id}: {profile}")

    with open(RESULTS_DIR / "capability_axes.json", "w") as f:
        json.dump(axes_results, f, indent=2, default=str)

    # Part D: Composite metrics
    print("\n── Part D: Composite Metrics ──")
    composite_results = _compute_composite_metrics(obs_table)
    for model_id, comps in sorted(composite_results.get("model_composites", {}).items()):
        print(f"  {model_id}: {comps}")

    with open(RESULTS_DIR / "composite_metrics.json", "w") as f:
        json.dump(composite_results, f, indent=2, default=str)

    # Part E: Bootstrap CI
    print(f"\n── Part E: Bootstrap CI (n={n_bootstrap}) ──")
    ci_results = _bootstrap_ci(obs_table, n_bootstrap)
    for model_id, metrics_ci in sorted(ci_results.get("model_ci", {}).items()):
        cdp = metrics_ci.get("committor_dp", {})
        if cdp:
            print(f"  {model_id} committor_dp: "
                  f"{cdp['mean']:.3f} [{cdp['ci_95_lo']:.3f}, {cdp['ci_95_hi']:.3f}]")

    with open(RESULTS_DIR / "bootstrap_ci.json", "w") as f:
        json.dump(ci_results, f, indent=2, default=str)

    print(f"\n── Output ──")
    print(f"  {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    main(n_bootstrap=args.n_bootstrap)
