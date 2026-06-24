#!/usr/bin/env python3
"""Cross-benchmark analysis of cx-cmu per-model graph metrics.

5 Analyses:
  A1: Profile heatmap — 5 models × 5 benchmarks × 8 metrics
  A2: Model separability — per-benchmark ANOVA (F-test)
  A3: Cross-benchmark rank consistency — Kendall τ
  A4: Reward correlation — graph metrics vs task reward (Spearman ρ)
  A5: Leave-one-benchmark-out — transfer prediction

Output: results/cxcmu/analysis.json

Usage:
    python scripts/87_cxcmu_analysis.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path("results/cxcmu")
DYNAMICS_PATH = RESULTS_DIR / "typed_state_dynamics.jsonl"

# 8 graph-derived metrics
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

# Expected direction (higher is better = +1, lower is better = -1)
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
    """Load typed_state_dynamics.jsonl."""
    data = []
    with open(DYNAMICS_PATH) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_profile_matrix(data: list[dict]) -> dict:
    """Build (model × benchmark) profile matrix.

    Returns: {model: {benchmark: {metric: mean_value}}}
    """
    # Collect per-(model, benchmark, metric) values
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in data:
        bench = row["benchmark"]
        for model_id, metrics in row.get("per_model", {}).items():
            for metric in METRICS:
                val = metrics.get(metric)
                if val is not None:
                    raw[model_id][bench][metric].append(val)

    # Average
    profiles = {}
    for model_id, bench_data in raw.items():
        profiles[model_id] = {}
        for bench, metric_data in bench_data.items():
            profiles[model_id][bench] = {}
            for metric, vals in metric_data.items():
                profiles[model_id][bench][metric] = float(np.mean(vals))

    return profiles


def analysis_a1_profile_heatmap(profiles: dict) -> dict:
    """A1: Profile heatmap data — models × benchmarks × metrics."""
    models = sorted(profiles.keys())
    benchmarks = set()
    for m in profiles.values():
        benchmarks.update(m.keys())
    benchmarks = sorted(benchmarks)

    heatmap = {}
    for model in models:
        heatmap[model] = {}
        for bench in benchmarks:
            heatmap[model][bench] = profiles.get(model, {}).get(bench, {})

    return {
        "models": models,
        "benchmarks": benchmarks,
        "metrics": METRICS,
        "heatmap": heatmap,
    }


def analysis_a2_separability(data: list[dict]) -> dict:
    """A2: Per-benchmark ANOVA — test if models differ significantly per metric."""
    # Group by benchmark
    by_bench = defaultdict(list)
    for row in data:
        by_bench[row["benchmark"]].append(row)

    results = {}
    for bench, rows in sorted(by_bench.items()):
        bench_results = {}
        for metric in METRICS:
            # Group metric values by model
            model_groups = defaultdict(list)
            for row in rows:
                for model_id, metrics in row.get("per_model", {}).items():
                    val = metrics.get(metric)
                    if val is not None:
                        model_groups[model_id].append(val)

            # Need at least 2 groups with ≥2 values
            groups = [v for v in model_groups.values() if len(v) >= 2]
            if len(groups) >= 2:
                try:
                    f_stat, p_val = scipy_stats.f_oneway(*groups)
                    bench_results[metric] = {
                        "F": round(float(f_stat), 4) if not math.isnan(f_stat) else None,
                        "p": round(float(p_val), 6) if not math.isnan(p_val) else None,
                        "significant": bool(p_val < 0.05) if not math.isnan(p_val) else False,
                        "n_groups": len(groups),
                    }
                except Exception:
                    bench_results[metric] = {"F": None, "p": None, "significant": False}
            else:
                bench_results[metric] = {"F": None, "p": None, "significant": False}

        results[bench] = bench_results

    # Summary: how many metrics are significant per benchmark
    summary = {}
    for bench, br in results.items():
        n_sig = sum(1 for m in br.values() if m.get("significant", False))
        summary[bench] = f"{n_sig}/{len(METRICS)} significant"

    return {"per_benchmark": results, "summary": summary}


def analysis_a3_rank_consistency(profiles: dict) -> dict:
    """A3: Cross-benchmark rank consistency via Kendall τ.

    For each metric, rank models within each benchmark,
    then compute pairwise Kendall τ between benchmarks.
    """
    models = sorted(profiles.keys())
    benchmarks = set()
    for m in profiles.values():
        benchmarks.update(m.keys())
    benchmarks = sorted(benchmarks)

    if len(models) < 3 or len(benchmarks) < 2:
        return {"error": "Not enough models or benchmarks for rank analysis"}

    results = {}
    for metric in METRICS:
        sign = METRIC_SIGN[metric]

        # Rank models per benchmark
        bench_ranks = {}
        for bench in benchmarks:
            vals = []
            for model in models:
                v = profiles.get(model, {}).get(bench, {}).get(metric)
                vals.append(v if v is not None else float('nan'))

            # Rank (accounting for direction)
            arr = np.array(vals) * sign  # higher = better after sign
            # Handle NaN by giving worst rank
            valid_mask = ~np.isnan(arr)
            if valid_mask.sum() < 2:
                continue
            ranks = np.full(len(arr), float('nan'))
            ranks[valid_mask] = scipy_stats.rankdata(arr[valid_mask])
            bench_ranks[bench] = ranks

        # Pairwise Kendall τ between benchmarks
        taus = []
        for b1, b2 in combinations(bench_ranks.keys(), 2):
            r1, r2 = bench_ranks[b1], bench_ranks[b2]
            valid = ~(np.isnan(r1) | np.isnan(r2))
            if valid.sum() >= 3:
                tau, p = scipy_stats.kendalltau(r1[valid], r2[valid])
                if not math.isnan(tau):
                    taus.append({"pair": f"{b1}_vs_{b2}", "tau": round(tau, 4), "p": round(p, 4)})

        mean_tau = float(np.mean([t["tau"] for t in taus])) if taus else math.nan
        results[metric] = {
            "mean_kendall_tau": round(mean_tau, 4) if not math.isnan(mean_tau) else None,
            "n_pairs": len(taus),
            "pairs": taus,
        }

    # Overall summary
    all_taus = []
    for metric_res in results.values():
        mt = metric_res.get("mean_kendall_tau")
        if mt is not None:
            all_taus.append(mt)

    return {
        "per_metric": results,
        "overall_mean_tau": round(float(np.mean(all_taus)), 4) if all_taus else None,
        "n_metrics_positive_tau": sum(1 for t in all_taus if t > 0),
    }


def analysis_a4_reward_correlation(data: list[dict]) -> dict:
    """A4: Spearman ρ between graph metrics and task reward (resolve_rate)."""
    results = {}

    for metric in METRICS:
        metric_vals = []
        reward_vals = []

        for row in data:
            for model_id, metrics in row.get("per_model", {}).items():
                m_val = metrics.get(metric)
                r_val = metrics.get("resolve_rate")
                if m_val is not None and r_val is not None:
                    metric_vals.append(m_val)
                    reward_vals.append(r_val)

        if len(metric_vals) >= 10:
            rho, p = scipy_stats.spearmanr(metric_vals, reward_vals)
            results[metric] = {
                "spearman_rho": round(float(rho), 4) if not math.isnan(rho) else None,
                "p_value": round(float(p), 6) if not math.isnan(p) else None,
                "n_observations": len(metric_vals),
                "significant": bool(p < 0.05) if not math.isnan(p) else False,
            }
        else:
            results[metric] = {"spearman_rho": None, "p_value": None, "n_observations": len(metric_vals)}

    # Summary
    sig_metrics = [m for m, r in results.items() if r.get("significant", False)]
    return {
        "per_metric": results,
        "n_significant": len(sig_metrics),
        "significant_metrics": sig_metrics,
    }


def analysis_a5_leave_one_out(profiles: dict) -> dict:
    """A5: Leave-one-benchmark-out transfer prediction.

    For each held-out benchmark, predict model ranking using
    average ranking from remaining benchmarks. Evaluate with Kendall τ.
    """
    models = sorted(profiles.keys())
    benchmarks = set()
    for m in profiles.values():
        benchmarks.update(m.keys())
    benchmarks = sorted(benchmarks)

    if len(benchmarks) < 3 or len(models) < 3:
        return {"error": "Not enough benchmarks or models for leave-one-out"}

    results = {}
    for metric in METRICS:
        sign = METRIC_SIGN[metric]
        loo_taus = []

        for held_out in benchmarks:
            train_benchs = [b for b in benchmarks if b != held_out]

            # Get model scores on held-out benchmark
            true_scores = []
            for model in models:
                v = profiles.get(model, {}).get(held_out, {}).get(metric)
                true_scores.append(v * sign if v is not None else float('nan'))

            # Predict from training benchmarks (average score)
            pred_scores = []
            for model in models:
                train_vals = []
                for b in train_benchs:
                    v = profiles.get(model, {}).get(b, {}).get(metric)
                    if v is not None:
                        train_vals.append(v * sign)
                pred_scores.append(float(np.mean(train_vals)) if train_vals else float('nan'))

            # Compute Kendall τ
            true_arr = np.array(true_scores)
            pred_arr = np.array(pred_scores)
            valid = ~(np.isnan(true_arr) | np.isnan(pred_arr))
            if valid.sum() >= 3:
                tau, p = scipy_stats.kendalltau(true_arr[valid], pred_arr[valid])
                if not math.isnan(tau):
                    loo_taus.append({
                        "held_out": held_out,
                        "tau": round(tau, 4),
                        "p": round(p, 4),
                    })

        mean_tau = float(np.mean([t["tau"] for t in loo_taus])) if loo_taus else math.nan
        results[metric] = {
            "mean_loo_tau": round(mean_tau, 4) if not math.isnan(mean_tau) else None,
            "n_folds": len(loo_taus),
            "folds": loo_taus,
        }

    # Overall
    all_taus = [r["mean_loo_tau"] for r in results.values() if r.get("mean_loo_tau") is not None]
    return {
        "per_metric": results,
        "overall_mean_loo_tau": round(float(np.mean(all_taus)), 4) if all_taus else None,
        "n_metrics_positive": sum(1 for t in all_taus if t > 0),
    }


def main():
    if not DYNAMICS_PATH.exists():
        print(f"Error: {DYNAMICS_PATH} not found. Run 86_typed_state_cxcmu.py first.")
        return

    print("Loading dynamics data...")
    data = load_dynamics_data()
    print(f"  {len(data)} task records loaded")

    # Build profile matrix
    profiles = build_profile_matrix(data)
    models = sorted(profiles.keys())
    benchmarks = set()
    for m in profiles.values():
        benchmarks.update(m.keys())
    benchmarks = sorted(benchmarks)
    print(f"  {len(models)} models × {len(benchmarks)} benchmarks")

    # Run 5 analyses
    print("\nRunning A1: Profile heatmap...")
    a1 = analysis_a1_profile_heatmap(profiles)

    print("Running A2: Model separability (ANOVA)...")
    a2 = analysis_a2_separability(data)
    for bench, summary in a2.get("summary", {}).items():
        print(f"  {bench}: {summary}")

    print("Running A3: Cross-benchmark rank consistency (Kendall τ)...")
    a3 = analysis_a3_rank_consistency(profiles)
    print(f"  Overall mean τ: {a3.get('overall_mean_tau')}")
    print(f"  Metrics with positive τ: {a3.get('n_metrics_positive_tau')}/{len(METRICS)}")

    print("Running A4: Reward correlation (Spearman ρ)...")
    a4 = analysis_a4_reward_correlation(data)
    print(f"  Significant metrics: {a4.get('n_significant')}/{len(METRICS)}")
    for m in a4.get("significant_metrics", []):
        rho = a4["per_metric"][m]["spearman_rho"]
        print(f"    {m}: ρ={rho}")

    print("Running A5: Leave-one-benchmark-out prediction...")
    a5 = analysis_a5_leave_one_out(profiles)
    print(f"  Overall mean LOO τ: {a5.get('overall_mean_loo_tau')}")
    print(f"  Metrics with positive LOO τ: {a5.get('n_metrics_positive')}/{len(METRICS)}")

    # Save all results
    output = {
        "n_tasks": len(data),
        "n_models": len(models),
        "n_benchmarks": len(benchmarks),
        "models": models,
        "benchmarks": benchmarks,
        "metrics": METRICS,
        "A1_profile_heatmap": a1,
        "A2_separability": a2,
        "A3_rank_consistency": a3,
        "A4_reward_correlation": a4,
        "A5_leave_one_out": a5,
    }

    out_path = RESULTS_DIR / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
