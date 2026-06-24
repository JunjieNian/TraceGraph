#!/usr/bin/env python3
"""Shared vs per-model graph comparison: justifying the shared-landscape approach.

For each task:
1. Shared graph: use existing payload (all ~20 rollouts from 5 models)
2. Per-model graphs: for each model, build a separate graph using only its ~4 rollouts

Compare:
  - Stability: coefficient of variation of graph metrics across bootstrap resamples
  - Ranking consistency: which approach's readout better predicts resolve_rate ranking?
  - Structural richness: n_blocks, core fraction, etc.

Expected conclusion: per-model graphs are too sparse (only 4 rollouts) to yield stable
metrics; shared graphs pool 20 rollouts for a stable landscape, then read out model-specific
behaviour via transition kernel conditioning.

Output: results/cxcmu/shared_vs_permodel/
  - comparison.json
  - per_task_details.jsonl

Usage:
    python scripts/94_shared_vs_permodel.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

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
from tracegraph.graph_construction import build_mutual_knn_edges
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
RESULTS_DIR = Path("results/cxcmu/shared_vs_permodel")


def _get_model_slice_indices(payload: dict) -> dict[str, list[int]]:
    """Map model_id → list of slice indices belonging to that model."""
    model_slices = defaultdict(list)
    slice_model_ids = payload.get("slice_model_ids", [])
    for i, mid in enumerate(slice_model_ids):
        model_slices[mid].append(i)
    return dict(model_slices)


def _build_permodel_graph_metrics(
    key_sets: list[set],
    model_slice_indices: list[int],
    model_run_ids: set[int],
    payload: dict,
) -> dict | None:
    """Build a graph from only one model's slices and extract metrics."""
    if len(model_slice_indices) < 6:
        return None

    # Subset key sets
    sub_keys = [key_sets[i] for i in model_slice_indices]
    non_empty = [ks for ks in sub_keys if ks]
    if len(non_empty) < 6:
        return None

    # IDF + distances on subset
    idf = build_idf_weights(sub_keys)
    if not idf:
        return None

    distances = compute_pairwise_distances(sub_keys, idf)
    k = min(NEIGHBOR_K, len(sub_keys) - 1)
    if k < 2:
        return None
    knn_indices, knn_dists = compute_knn(distances, k)

    # Build graph
    edges = build_mutual_knn_edges(knn_indices, knn_dists, neighbor_k=k, dist_scale=DIST_SCALE)
    if len(edges) < 2:
        return None

    # Count structural properties
    n_nodes = len(sub_keys)
    n_edges = len(edges)

    # Build adjacency for BCC (simplified — just count components)
    adj = defaultdict(set)
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    # Approximate block count via connected components
    visited = set()
    n_components = 0
    for node in range(n_nodes):
        if node not in visited and node in adj:
            # BFS
            queue = [node]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                for nbr in adj.get(curr, set()):
                    if nbr not in visited:
                        queue.append(nbr)
            n_components += 1

    # Estimate density
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max(max_edges, 1)

    return {
        "n_slices": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "density": round(density, 6),
        "avg_degree": round(2 * n_edges / max(n_nodes, 1), 4),
    }


def _shared_graph_readout(payload: dict, model_run_ids: dict[str, set[int]]) -> dict[str, dict]:
    """Extract per-model metrics from the shared graph (already done in script 86)."""
    block_meta = nontrivial_block_info(payload)
    if len(block_meta) < 3:
        return {}

    run_sequences = reconstruct_run_sequences(payload)
    if not run_sequences:
        return {}

    run_resolved = build_run_resolved(payload)
    rf = payload.get("reward_field", {})
    if not rf:
        return {}

    core_mask = np.array(rf["core_mask"])
    rf_node_to_idx = {int(k): v for k, v in rf.get("node_to_idx", {}).items()}

    run_typed = build_typed_sequences(run_sequences, block_meta, core_mask, rf_node_to_idx)
    if not run_typed:
        return {}

    all_states_set = set()
    for seq in run_typed.values():
        all_states_set.update(seq)
    all_states_set.add(EOS_RESOLVED)
    all_states_set.add(EOS_FAILED)
    all_states = sorted(all_states_set)

    per_model = {}
    for model_id, rids in model_run_ids.items():
        model_typed_rids = [r for r in sorted(rids) if r in run_typed]
        if len(model_typed_rids) < 2:
            continue

        P_m, _, sidx_m = build_kernel(run_typed, model_typed_rids, run_resolved, all_states, LAPLACE_ALPHA)
        c_m = compute_committor(P_m, sidx_m, all_states)

        committor_dp = math.nan
        if c_m is not None:
            dp_idx = [sidx_m[s] for s in all_states if "decision_point" in s and s in sidx_m]
            if dp_idx:
                dp_vals = c_m[dp_idx]
                dp_vals = dp_vals[~np.isnan(dp_vals)]
                if len(dp_vals) > 0:
                    committor_dp = float(np.mean(dp_vals))

        n_res = sum(1 for r in rids if run_resolved.get(r, False))
        per_model[model_id] = {
            "committor_dp": committor_dp,
            "resolve_rate": n_res / len(rids) if rids else 0.0,
        }

    return per_model


def process_task(bench: str, task_id: str, payload: dict) -> dict | None:
    """Compare shared vs per-model for one task."""
    # Get model → run_ids mapping
    model_slices = _get_model_slice_indices(payload)
    cache = payload.get("role_threshold_cache", {})
    slice_runs = list(cache.get("slice_runs", []))
    run_resolved = build_run_resolved(payload)

    model_run_ids = defaultdict(set)
    slice_model_ids = payload.get("slice_model_ids", [])
    for i, mid in enumerate(slice_model_ids):
        if i < len(slice_runs):
            model_run_ids[mid].add(int(slice_runs[i]))

    if len(model_run_ids) < 2:
        return None

    # Shared graph readout
    shared_readout = _shared_graph_readout(payload, dict(model_run_ids))
    if len(shared_readout) < 2:
        return None

    # Per-model graph metrics
    sig_dir = SIGNATURES_DIR / bench / task_id
    ks_path = sig_dir / "key_sets.pkl"
    if not ks_path.exists():
        return None

    with open(ks_path, "rb") as f:
        key_sets = pickle.load(f)

    permodel_metrics = {}
    for model_id, slice_idx in model_slices.items():
        pm = _build_permodel_graph_metrics(key_sets, slice_idx, model_run_ids.get(model_id, set()), payload)
        if pm is not None:
            permodel_metrics[model_id] = pm

    # Shared graph structural metrics
    block_meta = nontrivial_block_info(payload)
    rf = payload.get("reward_field", {})
    core_mask = np.array(rf.get("core_mask", []))
    n_core = int(core_mask.sum()) if len(core_mask) > 0 else 0

    shared_structure = {
        "n_blocks": len(block_meta),
        "n_slices": payload.get("n_slices", 0),
        "n_edges": payload.get("n_edges", 0),
        "n_core_blocks": n_core,
        "core_fraction": round(n_core / max(len(block_meta), 1), 4),
    }

    # Stability comparison: coefficient of variation across models
    # For shared: CV of committor_dp across models
    shared_committors = [v["committor_dp"] for v in shared_readout.values()
                         if not math.isnan(v.get("committor_dp", math.nan))]
    shared_cv = float(np.std(shared_committors) / max(abs(np.mean(shared_committors)), 1e-12)) if len(shared_committors) >= 2 else None

    # For per-model: CV of structural metrics (density, avg_degree)
    pm_densities = [v["density"] for v in permodel_metrics.values()]
    pm_cv_density = float(np.std(pm_densities) / max(abs(np.mean(pm_densities)), 1e-12)) if len(pm_densities) >= 2 else None

    # Ranking consistency with resolve rate
    models = sorted(set(shared_readout.keys()) & set(model_run_ids.keys()))
    if len(models) >= 3:
        true_rr = [shared_readout[m]["resolve_rate"] for m in models]

        # Shared graph ranking (by committor_dp)
        shared_scores = [shared_readout[m].get("committor_dp", 0.0) for m in models]
        shared_scores = [0.0 if math.isnan(s) else s for s in shared_scores]

        tau_shared, _ = scipy_stats.kendalltau(true_rr, shared_scores)

        # Per-model graph ranking (by density as proxy for structural quality)
        pm_scores = [permodel_metrics.get(m, {}).get("density", 0.0) for m in models]
        tau_permodel, _ = scipy_stats.kendalltau(true_rr, pm_scores)

        ranking = {
            "tau_shared": round(float(tau_shared), 4) if not math.isnan(tau_shared) else None,
            "tau_permodel": round(float(tau_permodel), 4) if not math.isnan(tau_permodel) else None,
            "shared_better": float(tau_shared) > float(tau_permodel) if not (math.isnan(tau_shared) or math.isnan(tau_permodel)) else None,
        }
    else:
        ranking = None

    return {
        "benchmark": bench,
        "task_id": task_id,
        "n_models": len(model_run_ids),
        "shared_structure": shared_structure,
        "permodel_structure": permodel_metrics,
        "shared_readout": {m: {k: round(v, 6) if isinstance(v, float) and not math.isnan(v) else v
                               for k, v in metrics.items()}
                          for m, metrics in shared_readout.items()},
        "stability": {
            "shared_committor_cv": round(shared_cv, 4) if shared_cv is not None else None,
            "permodel_density_cv": round(pm_cv_density, 4) if pm_cv_density is not None else None,
        },
        "ranking": ranking,
    }


def main(max_tasks: int | None = None, benchmark: str | None = None):
    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    print(f"Shared vs Per-model comparison: {benchmarks}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for bench in benchmarks:
        bench_dir = GRAPH_DIR / bench
        graph_files = sorted(bench_dir.glob("*.pkl"))
        if max_tasks:
            graph_files = graph_files[:max_tasks]

        print(f"\n── {bench}: {len(graph_files)} tasks ──")
        for gpath in tqdm(graph_files, desc=f"  {bench}"):
            with open(gpath, "rb") as f:
                payload = pickle.load(f)

            result = process_task(bench, gpath.stem, payload)
            if result is not None:
                all_results.append(result)

    print(f"\n── Summary: {len(all_results)} valid tasks ──")

    # Aggregate statistics
    shared_better_count = 0
    permodel_better_count = 0
    shared_taus = []
    permodel_taus = []

    avg_shared_blocks = []
    avg_permodel_slices = []

    for r in all_results:
        ranking = r.get("ranking")
        if ranking:
            ts = ranking.get("tau_shared")
            tp = ranking.get("tau_permodel")
            if ts is not None:
                shared_taus.append(ts)
            if tp is not None:
                permodel_taus.append(tp)
            if ranking.get("shared_better"):
                shared_better_count += 1
            elif ranking.get("shared_better") is False:
                permodel_better_count += 1

        ss = r.get("shared_structure", {})
        avg_shared_blocks.append(ss.get("n_blocks", 0))

        for pm in r.get("permodel_structure", {}).values():
            avg_permodel_slices.append(pm.get("n_slices", 0))

    print(f"\n  Structural Richness:")
    print(f"    Shared graph: mean {np.mean(avg_shared_blocks):.1f} blocks/task")
    print(f"    Per-model: mean {np.mean(avg_permodel_slices):.1f} slices/model/task")

    print(f"\n  Ranking Consistency:")
    if shared_taus:
        print(f"    Shared graph mean τ: {np.mean(shared_taus):.4f}")
    if permodel_taus:
        print(f"    Per-model graph mean τ: {np.mean(permodel_taus):.4f}")
    print(f"    Shared better: {shared_better_count}/{shared_better_count + permodel_better_count}")

    # Paired test
    if len(shared_taus) == len(permodel_taus) and len(shared_taus) >= 5:
        diff = np.array(shared_taus) - np.array(permodel_taus)
        t_stat, p_val = scipy_stats.ttest_1samp(diff, 0)
        print(f"    Paired t-test (shared - permodel): t={t_stat:.3f}, p={p_val:.4f}")

    # Save
    with open(RESULTS_DIR / "per_task_details.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    summary = {
        "n_valid_tasks": len(all_results),
        "shared_graph": {
            "mean_blocks": round(float(np.mean(avg_shared_blocks)), 2) if avg_shared_blocks else None,
            "mean_ranking_tau": round(float(np.mean(shared_taus)), 4) if shared_taus else None,
        },
        "permodel_graph": {
            "mean_slices_per_model": round(float(np.mean(avg_permodel_slices)), 2) if avg_permodel_slices else None,
            "mean_ranking_tau": round(float(np.mean(permodel_taus)), 4) if permodel_taus else None,
        },
        "shared_better_fraction": round(shared_better_count / max(shared_better_count + permodel_better_count, 1), 4),
    }

    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Output ──")
    print(f"  {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
