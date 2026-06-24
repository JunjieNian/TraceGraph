#!/usr/bin/env python3
"""RQ1 — Signature ablation: which key types drive model separability?

7 ablation conditions:
  - full:        all keys (baseline)
  - tool_only:   only TOOL: keys
  - action_only: only ACTION: keys
  - obs_only:    only OBS: keys
  - file_only:   only FILE_PATH: + FILE_EXT: keys
  - no_phase:    remove PHASE: keys
  - random:      random key assignment (destroys semantics, preserves cardinality)

For each condition, re-run: IDF → distances → kNN → graph → reward field →
typed-state → committor. Compare ANOVA F-stat and silhouette score for
model separability.

Output: results/cxcmu/signature_ablation/ablation_results.json

Usage:
    python scripts/90_signature_ablation.py [--max-tasks 30] [--benchmark BENCH] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Set

import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

from tracegraph.constants import (
    CORE_POS_Q,
    DIST_SCALE,
    EOS_FAILED,
    EOS_RESOLVED,
    LAPLACE_ALPHA,
    MIN_RUN_SUPPORT,
    NEIGHBOR_K,
    PROPAGATION_ALPHA,
    PROPAGATION_STEPS,
    SUPPORT_SHRINK_EXP,
)
from tracegraph.graph_construction import (
    build_mutual_knn_edges,
    compute_bcc_analysis,
)
from tracegraph.reward_field import (
    build_block_graph,
    build_run_resolved,
    build_transition_matrix,
    build_visit_sets,
    compute_seed_vector,
    diffuse_reward_field,
    make_core_mask,
    nontrivial_block_info,
    reconstruct_run_sequences,
)
from tracegraph.signature import (
    build_idf_weights,
    compute_knn,
    compute_pairwise_distances,
)
from tracegraph.typed_state_mdp import (
    build_kernel,
    build_typed_sequences,
    compute_committor,
)

GRAPH_DIR = Path("data/cxcmu/graphs")
SIGNATURES_DIR = Path("data/cxcmu/signatures")
RESULTS_DIR = Path("results/cxcmu/signature_ablation")

# Key-type prefix filters
KEY_FILTERS = {
    "full": None,  # no filter
    "tool_only": lambda k: k.startswith("TOOL:"),
    "action_only": lambda k: k.startswith("ACTION:"),
    "obs_only": lambda k: k.startswith("OBS:"),
    "file_only": lambda k: k.startswith("FILE_PATH:") or k.startswith("FILE_EXT:"),
    "no_phase": lambda k: not k.startswith("PHASE:"),
    "random": "random",  # special handling
}


def _load_key_sets(bench: str, task_id: str) -> list[Set[str]] | None:
    """Load pre-computed key sets from signatures directory."""
    sig_dir = SIGNATURES_DIR / bench / task_id
    ks_path = sig_dir / "key_sets.pkl"
    if not ks_path.exists():
        return None
    with open(ks_path, "rb") as f:
        return pickle.load(f)


def _filter_key_sets(
    key_sets: list[Set[str]],
    condition: str,
    rng: random.Random,
) -> list[Set[str]]:
    """Apply ablation filter to key sets."""
    if condition == "full":
        return key_sets

    if condition == "random":
        # Collect all unique keys and randomly reassign
        all_keys = sorted({k for ks in key_sets for k in ks})
        if not all_keys:
            return key_sets
        result = []
        for ks in key_sets:
            n = len(ks)
            new_ks = set(rng.sample(all_keys, min(n, len(all_keys))))
            result.append(new_ks)
        return result

    filt = KEY_FILTERS[condition]
    return [
        {k for k in ks if filt(k)} for ks in key_sets
    ]


def _run_pipeline_on_keys(
    key_sets: list[Set[str]],
    payload: dict,
) -> dict | None:
    """Run graph → reward field → typed-state → committor on filtered keys.

    Returns per-model committor_dp values or None if pipeline fails.
    """
    # Filter out empty key sets (but keep indices consistent)
    n_slices = len(key_sets)
    if n_slices < 6:
        return None

    # IDF + distances
    non_empty = [ks for ks in key_sets if ks]
    if len(non_empty) < 6:
        return None

    idf = build_idf_weights(key_sets)
    if not idf:
        return None

    distances = compute_pairwise_distances(key_sets, idf)
    knn_indices, knn_dists = compute_knn(distances, NEIGHBOR_K)

    # Build mutual-kNN edges
    edges = build_mutual_knn_edges(
        knn_indices, knn_dists,
        neighbor_k=NEIGHBOR_K,
        dist_scale=DIST_SCALE,
    )
    if len(edges) < 3:
        return None

    # Build adjacency list for BCC (List[List[int]] format)
    adj_list = [[] for _ in range(n_slices)]
    for e in edges:
        s, t = e["source"], e["target"]
        if s < n_slices and t < n_slices:
            adj_list[s].append(t)
            adj_list[t].append(s)

    # BCC analysis
    cache = payload.get("role_threshold_cache", {})
    slice_runs = list(cache.get("slice_runs", []))
    slice_positions = list(cache.get("slice_positions", []))
    trajectories = payload.get("trajectories", [])
    run_resolved_map = {int(t["run_id"]): t.get("is_resolved", False) for t in trajectories}

    # Ensure arrays match n_slices
    slice_run_arr = np.array(slice_runs[:n_slices], dtype=int) if slice_runs else np.zeros(n_slices, dtype=int)
    slice_pos_arr = np.array(slice_positions[:n_slices], dtype=int) if slice_positions else np.arange(n_slices, dtype=int)
    is_resolved_list = [
        run_resolved_map.get(int(slice_run_arr[i]), False) if i < len(slice_run_arr) else False
        for i in range(n_slices)
    ]

    bcc_result = compute_bcc_analysis(
        adj_list, n_slices, slice_run_arr, slice_pos_arr, is_resolved_list,
    )
    if bcc_result is None:
        return None

    # Build a synthetic payload with new BCC for typed-state extraction
    new_payload = dict(payload)
    new_payload["bcc_analysis"] = bcc_result

    block_meta = nontrivial_block_info(new_payload)
    if len(block_meta) < 3:
        return None

    run_sequences = reconstruct_run_sequences(new_payload)
    if not run_sequences:
        return None

    run_block_sets, block_run_sets = build_visit_sets(run_sequences)
    run_resolved = build_run_resolved(new_payload)
    if len(run_resolved) < 3:
        return None

    # Block graph + transition matrix
    block_adj = build_block_graph(new_payload, run_sequences)
    nodes = sorted(block_meta.keys())
    P_block, node_to_idx = build_transition_matrix(nodes, block_adj)

    # Reward field + core mask
    seed = compute_seed_vector(
        run_id=None, nodes=nodes, block_run_sets=block_run_sets,
        run_resolved=run_resolved, min_run_support=max(1, MIN_RUN_SUPPORT),
        support_shrink_exp=SUPPORT_SHRINK_EXP,
    )
    field = diffuse_reward_field(seed, P_block, alpha=PROPAGATION_ALPHA, n_steps=PROPAGATION_STEPS)
    core_mask = make_core_mask(field, CORE_POS_Q)

    # Build typed sequences
    run_typed = build_typed_sequences(run_sequences, block_meta, core_mask, node_to_idx)
    if not run_typed:
        return None

    all_states_set = set()
    for seq in run_typed.values():
        all_states_set.update(seq)
    all_states_set.add(EOS_RESOLVED)
    all_states_set.add(EOS_FAILED)
    all_states = sorted(all_states_set)

    all_rids = sorted(run_resolved.keys())

    # Build kernel + committor
    P_all, visits_all, state_idx = build_kernel(
        run_typed, all_rids, run_resolved, all_states, LAPLACE_ALPHA,
    )
    committor = compute_committor(P_all, state_idx, all_states)

    # Extract per-model committor_dp
    model_runs = defaultdict(set)
    slice_model_ids = payload.get("slice_model_ids", [])
    for i, mid in enumerate(slice_model_ids):
        if i < len(slice_runs):
            model_runs[mid].add(int(slice_runs[i]))

    per_model = {}
    for model_id, rids in model_runs.items():
        model_typed_rids = [r for r in sorted(rids) if r in run_typed]
        if len(model_typed_rids) < 2:
            continue

        P_m, _, sidx_m = build_kernel(
            run_typed, model_typed_rids, run_resolved, all_states, LAPLACE_ALPHA,
        )
        c_m = compute_committor(P_m, sidx_m, all_states)

        m_committor_dp = math.nan
        if c_m is not None:
            dp_idx = [sidx_m[s] for s in all_states
                      if "decision_point" in s and s in sidx_m]
            if dp_idx:
                dp_vals = c_m[dp_idx]
                dp_vals = dp_vals[~np.isnan(dp_vals)]
                if len(dp_vals) > 0:
                    m_committor_dp = float(np.mean(dp_vals))

        # Also get resolve rate
        n_resolved = sum(1 for r in rids if run_resolved.get(r, False))
        per_model[model_id] = {
            "committor_dp": m_committor_dp,
            "resolve_rate": n_resolved / len(rids) if rids else 0.0,
            "n_runs": len(rids),
        }

    if len(per_model) < 2:
        return None

    return per_model


def _compute_separability_metrics(per_model_results: list[dict]) -> dict:
    """Compute ANOVA F-stat and silhouette-like metric from per-model committors."""
    # Collect committor values grouped by model
    model_groups = defaultdict(list)
    for task_result in per_model_results:
        for model_id, metrics in task_result.items():
            cdp = metrics.get("committor_dp")
            if cdp is not None and not math.isnan(cdp):
                model_groups[model_id].append(cdp)

    # ANOVA F-stat
    groups = [v for v in model_groups.values() if len(v) >= 2]
    f_stat = None
    p_val = None
    if len(groups) >= 2:
        try:
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            if math.isnan(f_stat):
                f_stat = None
                p_val = None
        except Exception:
            pass

    # Silhouette-like: ratio of between-model variance to total variance
    all_vals = [v for g in groups for v in g]
    if len(all_vals) >= 4 and len(groups) >= 2:
        total_var = float(np.var(all_vals))
        group_means = [np.mean(g) for g in groups]
        between_var = float(np.var(group_means))
        silhouette_ratio = between_var / max(total_var, 1e-12)
    else:
        silhouette_ratio = None

    # Spearman correlation between committor and resolve rate
    committor_vals = []
    resolve_vals = []
    for task_result in per_model_results:
        for model_id, metrics in task_result.items():
            cdp = metrics.get("committor_dp")
            rr = metrics.get("resolve_rate")
            if cdp is not None and not math.isnan(cdp) and rr is not None:
                committor_vals.append(cdp)
                resolve_vals.append(rr)

    rho = None
    if len(committor_vals) >= 5:
        try:
            rho_val, _ = scipy_stats.spearmanr(committor_vals, resolve_vals)
            if not math.isnan(rho_val):
                rho = float(rho_val)
        except Exception:
            pass

    return {
        "f_stat": round(float(f_stat), 4) if f_stat is not None else None,
        "p_value": round(float(p_val), 6) if p_val is not None else None,
        "silhouette_ratio": round(silhouette_ratio, 6) if silhouette_ratio is not None else None,
        "reward_correlation_rho": round(rho, 4) if rho is not None else None,
        "n_tasks_valid": len(per_model_results),
        "n_models": len(groups),
    }


def main(max_tasks: int = 30, benchmark: str | None = None, seed: int = 42):
    rng = random.Random(seed)
    np.random.seed(seed)

    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    print(f"Signature ablation: {len(benchmarks)} benchmarks, max {max_tasks} tasks each")
    print(f"Conditions: {list(KEY_FILTERS.keys())}")

    # Collect tasks
    task_list: list[tuple[str, str, Path]] = []
    for bench in benchmarks:
        bench_dir = GRAPH_DIR / bench
        graph_files = sorted(bench_dir.glob("*.pkl"))[:max_tasks]
        for gpath in graph_files:
            task_list.append((bench, gpath.stem, gpath))

    print(f"Total tasks: {len(task_list)}")

    # Run ablation for each condition
    results = {}
    for condition in KEY_FILTERS:
        print(f"\n── Condition: {condition} ──")
        per_model_results = []
        n_skipped = 0

        for bench, task_id, gpath in tqdm(task_list, desc=f"  {condition}"):
            # Load key sets
            key_sets = _load_key_sets(bench, task_id)
            if key_sets is None:
                n_skipped += 1
                continue

            # Apply filter
            filtered = _filter_key_sets(key_sets, condition, rng)

            # Check if enough non-empty key sets remain
            non_empty = [ks for ks in filtered if ks]
            if len(non_empty) < 6:
                n_skipped += 1
                continue

            # Load payload
            with open(gpath, "rb") as f:
                payload = pickle.load(f)

            # Run pipeline
            task_result = _run_pipeline_on_keys(filtered, payload)
            if task_result is not None:
                per_model_results.append(task_result)
            else:
                n_skipped += 1

        # Compute separability metrics
        metrics = _compute_separability_metrics(per_model_results)
        metrics["n_skipped"] = n_skipped
        results[condition] = metrics

        print(f"  Valid tasks: {metrics['n_tasks_valid']}, "
              f"F={metrics['f_stat']}, "
              f"silhouette={metrics['silhouette_ratio']}, "
              f"ρ={metrics['reward_correlation_rho']}")

    # Summary comparison
    print("\n── Summary ──")
    print(f"{'Condition':<15} {'F-stat':<10} {'Silhouette':<12} {'ρ(reward)':<10} {'N_valid':<8}")
    print("-" * 55)
    for cond, m in results.items():
        f_str = f"{m['f_stat']:.2f}" if m['f_stat'] is not None else "N/A"
        s_str = f"{m['silhouette_ratio']:.4f}" if m['silhouette_ratio'] is not None else "N/A"
        r_str = f"{m['reward_correlation_rho']:.3f}" if m['reward_correlation_rho'] is not None else "N/A"
        print(f"{cond:<15} {f_str:<10} {s_str:<12} {r_str:<10} {m['n_tasks_valid']:<8}")

    # Ranking
    ranked = sorted(
        [(c, m) for c, m in results.items() if m.get("f_stat") is not None],
        key=lambda x: x[1]["f_stat"],
        reverse=True,
    )
    if ranked:
        print(f"\nRanking by F-stat: {' > '.join(c for c, _ in ranked)}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "conditions": list(KEY_FILTERS.keys()),
            "n_benchmarks": len(benchmarks),
            "max_tasks_per_bench": max_tasks,
            "total_tasks": len(task_list),
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=30,
                        help="Max tasks per benchmark (default 30 for speed)")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(max_tasks=args.max_tasks, benchmark=args.benchmark, seed=args.seed)
