#!/usr/bin/env python3
"""Shared vs per-model graph size: why TraceGraph pools rollouts.

For each task:
1. Shared graph: use the existing pooled payload (all ~20 rollouts from 5 models)
2. Per-model graphs: for each model, build a separate mutual-kNN graph using
   only its ~4 rollouts

A task is counted when its shared graph has at least three retained blocks,
a reward field, and at least two models with two or more rollouts on the
landscape, i.e. when it supports the multi-model profiles.

Output: results/cxcmu/shared_vs_permodel/
  - comparison.json
  - per_task_details.jsonl

Usage:
    python scripts/analysis/shared_vs_permodel.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tracegraph.constants import NEIGHBOR_K
from tracegraph.graph_construction import build_mutual_knn_edges
from tracegraph.reward_field import (
    nontrivial_block_info,
    reconstruct_run_sequences,
)
from tracegraph.signature import (
    build_idf_weights,
    compute_knn,
    compute_pairwise_distances,
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
) -> dict | None:
    """Build a graph from only one model's slices and report its size."""
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
    edges = build_mutual_knn_edges(knn_indices, knn_dists, neighbor_k=k)
    if len(edges) < 2:
        return None

    n_nodes = len(sub_keys)
    n_edges = len(edges)

    adj = defaultdict(set)
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    # Connected components of the per-model graph
    visited = set()
    n_components = 0
    for node in range(n_nodes):
        if node not in visited and node in adj:
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

    return {
        "n_slices": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
    }


def _supports_multimodel_profiles(payload: dict, model_run_ids: dict[str, set[int]]) -> bool:
    """True if the shared graph supports per-model readout on the landscape."""
    block_meta = nontrivial_block_info(payload)
    if len(block_meta) < 3:
        return False
    run_sequences = reconstruct_run_sequences(payload)
    if not run_sequences:
        return False
    if not payload.get("reward_field", {}):
        return False
    n_models = sum(
        1 for rids in model_run_ids.values()
        if sum(1 for r in rids if r in run_sequences) >= 2
    )
    return n_models >= 2


def process_task(bench: str, task_id: str, payload: dict) -> dict | None:
    """Compare shared vs per-model graph size for one task."""
    model_slices = _get_model_slice_indices(payload)
    cache = payload.get("role_threshold_cache", {})
    slice_runs = list(cache.get("slice_runs", []))

    model_run_ids = defaultdict(set)
    slice_model_ids = payload.get("slice_model_ids", [])
    for i, mid in enumerate(slice_model_ids):
        if i < len(slice_runs):
            model_run_ids[mid].add(int(slice_runs[i]))

    if len(model_run_ids) < 2:
        return None
    if not _supports_multimodel_profiles(payload, dict(model_run_ids)):
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
        pm = _build_permodel_graph_metrics(key_sets, slice_idx)
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

    return {
        "benchmark": bench,
        "task_id": task_id,
        "n_models": len(model_run_ids),
        "shared_structure": shared_structure,
        "permodel_structure": permodel_metrics,
    }


def main(max_tasks: int | None = None, benchmark: str | None = None):
    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    print(f"Shared vs per-model comparison: {benchmarks}")
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

    avg_shared_blocks = [r["shared_structure"]["n_blocks"] for r in all_results]
    avg_permodel_slices = [
        pm["n_slices"]
        for r in all_results
        for pm in r["permodel_structure"].values()
    ]

    print("\n  Structural richness:")
    if avg_shared_blocks:
        print(f"    Shared graph: mean {np.mean(avg_shared_blocks):.1f} blocks/task")
    if avg_permodel_slices:
        print(f"    Per-model: mean {np.mean(avg_permodel_slices):.1f} slices/model/task")

    # Save
    with open(RESULTS_DIR / "per_task_details.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    summary = {
        "n_valid_tasks": len(all_results),
        "shared_graph": {
            "mean_blocks": round(float(np.mean(avg_shared_blocks)), 2) if avg_shared_blocks else None,
        },
        "permodel_graph": {
            "mean_slices_per_model": round(float(np.mean(avg_permodel_slices)), 2) if avg_permodel_slices else None,
        },
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
