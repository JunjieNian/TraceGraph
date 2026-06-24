#!/usr/bin/env python3
"""Build per-task shared-state graphs for cx-cmu data.

Per task (all models pooled):
1. Load kNN arrays and metadata
2. build_mutual_knn_edges()
3. build_adjacency_list()
4. compute_bcc_analysis() with resolved labels
5. Save graph payload with slice_model_ids annotation

Key addition: payload["slice_model_ids"] maps each slice to its model,
enabling per-model analysis in downstream scripts.

Usage:
    python scripts/83_build_cxcmu_graphs.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tracegraph.constants import DIST_SCALE, NEIGHBOR_K
from tracegraph.graph_construction import (
    build_adjacency_list,
    build_mutual_knn_edges,
    compute_bcc_analysis,
)

SIG_DIR = Path("data/cxcmu/signatures")
PARSED_DIR = Path("data/cxcmu/parsed")
GRAPH_DIR = Path("data/cxcmu/graphs")


def _set_data_root(data_root: Path) -> None:
    """Re-point SIG_DIR / PARSED_DIR / GRAPH_DIR under ``data_root``."""
    global SIG_DIR, PARSED_DIR, GRAPH_DIR
    data_root = Path(data_root)
    SIG_DIR = data_root / "signatures"
    PARSED_DIR = data_root / "parsed"
    GRAPH_DIR = data_root / "graphs"


def build_task_graph(task_id: str, sig_dir: Path, parsed_path: Path) -> dict | None:
    """Build graph for a single task (all models pooled)."""
    task_sig = sig_dir / task_id
    if not task_sig.exists():
        return None

    # Load kNN arrays
    knn_path = task_sig / "knn_indices.npy"
    if not knn_path.exists():
        return None

    knn_indices = np.load(task_sig / "knn_indices.npy")
    knn_dists = np.load(task_sig / "knn_dists.npy")

    with open(task_sig / "slice_metadata.json") as f:
        slice_metadata = json.load(f)

    n_slices = knn_indices.shape[0]
    if n_slices < 3:
        return None

    # Build run_id mapping: assign numeric run_ids from run_id strings
    unique_runs = []
    run_str_to_int = {}
    for m in slice_metadata:
        rid_str = m["run_id"]
        if rid_str not in run_str_to_int:
            run_str_to_int[rid_str] = len(unique_runs)
            unique_runs.append(rid_str)

    slice_run = np.array([run_str_to_int[m["run_id"]] for m in slice_metadata], dtype=np.int32)
    slice_pos = np.array([m["step_idx"] for m in slice_metadata], dtype=np.int32)
    slice_model_ids = [m.get("model_id", "unknown") for m in slice_metadata]

    # Load parsed data for resolved labels
    is_resolved_map = {}  # run_str → bool
    if parsed_path.exists():
        with open(parsed_path) as f:
            for line in f:
                traj = json.loads(line.strip())
                is_resolved_map[traj["run_id"]] = traj.get("resolved", False)

    # Build is_resolved array indexed by numeric run_id
    is_resolved = [is_resolved_map.get(r, False) for r in unique_runs]

    # Build graph
    k = min(NEIGHBOR_K, knn_indices.shape[1])
    edges = build_mutual_knn_edges(knn_indices, knn_dists, k, DIST_SCALE)
    adj = build_adjacency_list(edges, n_slices)

    # BCC analysis
    bcc_result = compute_bcc_analysis(
        adj=adj,
        n_nodes=n_slices,
        slice_run=slice_run,
        slice_pos=slice_pos,
        is_resolved=is_resolved,
    )

    # Build node_bcc_map
    node_bcc_map = defaultdict(list)
    for block in bcc_result.get("blocks", []):
        for node_idx in block["node_indices"]:
            node_bcc_map[node_idx].append(block["block_id"])

    # Assemble payload
    payload = {
        "task_id": task_id,
        "n_slices": n_slices,
        "n_edges": len(edges),
        "bcc_analysis": bcc_result,
        "slice_model_ids": slice_model_ids,
        "run_id_map": run_str_to_int,  # str → int mapping
        "unique_runs": unique_runs,    # int → str mapping
        "role_threshold_cache": {
            "selected_slices": list(range(n_slices)),
            "slice_runs": slice_run.tolist(),
            "slice_positions": slice_pos.tolist(),
            "node_bcc_map": {str(k): v for k, v in node_bcc_map.items()},
        },
        "trajectories": [
            {"run_id": i, "run_str": unique_runs[i], "is_resolved": is_resolved[i]}
            for i in range(len(unique_runs))
        ],
    }

    return payload


def main(max_tasks: int | None = None, benchmark: str | None = None):
    benchmarks = sorted(d.name for d in SIG_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {SIG_DIR}")
        return

    print(f"Processing benchmarks: {benchmarks}")

    stats = {"total": 0, "n_blocks": [], "n_nt_blocks": [], "n_edges": []}

    for bench in benchmarks:
        bench_sig = SIG_DIR / bench
        bench_parsed = PARSED_DIR / bench
        bench_graph = GRAPH_DIR / bench
        bench_graph.mkdir(parents=True, exist_ok=True)

        task_dirs = sorted(d.name for d in bench_sig.iterdir() if d.is_dir())
        if max_tasks is not None:
            task_dirs = task_dirs[:max_tasks]

        print(f"\n── {bench}: {len(task_dirs)} tasks ──")

        for task_id in tqdm(task_dirs, desc=f"  Building {bench}"):
            parsed_path = bench_parsed / f"{task_id}.jsonl"
            payload = build_task_graph(task_id, bench_sig, parsed_path)
            if payload is None:
                continue

            out_path = bench_graph / f"{task_id}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(payload, f)

            bcc = payload["bcc_analysis"]
            stats["total"] += 1
            stats["n_blocks"].append(bcc.get("n_blocks", 0))
            stats["n_nt_blocks"].append(bcc.get("n_nontrivial", 0))
            stats["n_edges"].append(payload["n_edges"])

    print(f"\n── Summary ──")
    print(f"Built {stats['total']} graphs")
    if stats["total"] > 0:
        for key in ("n_blocks", "n_nt_blocks", "n_edges"):
            vals = stats[key]
            print(f"  {key}: mean={np.mean(vals):.1f}, "
                  f"median={np.median(vals):.0f}, "
                  f"min={min(vals)}, max={max(vals)}")
    print(f"Output directory: {GRAPH_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the data root (default: data/cxcmu). "
             "Expects <data_root>/signatures and <data_root>/parsed, "
             "writes to <data_root>/graphs.",
    )
    args = parser.parse_args()
    if args.data_root is not None:
        _set_data_root(Path(args.data_root))
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
