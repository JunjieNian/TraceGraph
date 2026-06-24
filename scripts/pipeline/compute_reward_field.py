#!/usr/bin/env python3
"""Compute reward field propagation and core mask for cx-cmu task graphs.

Per task:
1. Load graph payload
2. Reconstruct run sequences
3. Build block graph + transition matrix
4. Compute reward seed → diffuse → core mask
5. Save enriched payload back to pickle

Usage:
    python scripts/84_reward_field_cxcmu.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from tracegraph.constants import (
    CORE_POS_Q,
    MIN_RUN_SUPPORT,
    PROPAGATION_ALPHA,
    PROPAGATION_STEPS,
    SUPPORT_SHRINK_EXP,
)
from tracegraph.graph_construction import nontrivial_blocks
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

GRAPH_DIR = Path("data/cxcmu/graphs")
RESULTS_DIR = Path("results/cxcmu/reward_field")


def _set_data_root(data_root: Path, results_root: Optional[Path] = None) -> None:
    """Re-point GRAPH_DIR and (optionally) RESULTS_DIR."""
    global GRAPH_DIR, RESULTS_DIR
    data_root = Path(data_root)
    GRAPH_DIR = data_root / "graphs"
    if results_root is not None:
        RESULTS_DIR = Path(results_root) / "reward_field"


def main(max_tasks: int | None = None, benchmark: str | None = None):
    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing benchmarks: {benchmarks}")
    all_results = []

    for bench in benchmarks:
        bench_graph = GRAPH_DIR / bench
        graph_files = sorted(bench_graph.glob("*.pkl"))
        if max_tasks is not None:
            graph_files = graph_files[:max_tasks]

        print(f"\n── {bench}: {len(graph_files)} graphs ──")

        for gpath in tqdm(graph_files, desc=f"  Reward field {bench}"):
            task_id = gpath.stem
            with open(gpath, "rb") as f:
                payload = pickle.load(f)

            # Block info
            block_meta = nontrivial_block_info(payload)
            if not block_meta:
                continue

            # Run sequences
            run_sequences = reconstruct_run_sequences(payload)
            if not run_sequences:
                continue

            # Visit sets
            run_block_sets, block_run_sets = build_visit_sets(run_sequences)
            run_resolved = build_run_resolved(payload)

            # Block graph
            block_adj = build_block_graph(payload, run_sequences)
            nodes = sorted(block_meta.keys())

            if len(nodes) < 2:
                continue

            # Transition matrix
            P, node_to_idx = build_transition_matrix(nodes, block_adj)

            # Seed vector (global)
            seed = compute_seed_vector(
                run_id=None,
                nodes=nodes,
                block_run_sets=block_run_sets,
                run_resolved=run_resolved,
                min_run_support=max(1, MIN_RUN_SUPPORT),
                support_shrink_exp=SUPPORT_SHRINK_EXP,
            )

            # Diffuse
            field = diffuse_reward_field(
                seed, P, alpha=PROPAGATION_ALPHA, n_steps=PROPAGATION_STEPS,
            )

            # Core mask
            core_mask = make_core_mask(field, CORE_POS_Q)

            # Summary
            n_core = int(core_mask.sum())
            row = {
                "benchmark": bench,
                "task_id": task_id,
                "n_blocks": len(nodes),
                "n_core_blocks": n_core,
                "core_fraction": round(n_core / max(1, len(nodes)), 4),
                "field_mean": round(float(field.mean()), 6),
                "field_std": round(float(field.std()), 6),
            }
            all_results.append(row)

            # Save enriched payload
            payload["reward_field"] = {
                "field": field.tolist(),
                "core_mask": core_mask.tolist(),
                "nodes": nodes,
                "node_to_idx": {str(k): v for k, v in node_to_idx.items()},
            }
            with open(gpath, "wb") as f:
                pickle.dump(payload, f)

    # Save summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n── Summary ──")
    print(f"Processed {len(all_results)} tasks")
    if all_results:
        core_fracs = [r["core_fraction"] for r in all_results]
        print(f"  Mean core fraction: {np.mean(core_fracs):.4f}")
        print(f"  Tasks with core > 0: {sum(1 for cf in core_fracs if cf > 0)}/{len(core_fracs)}")
    print(f"Output: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the data root (default: data/cxcmu). "
             "Expects <data_root>/graphs.",
    )
    parser.add_argument(
        "--results-root", type=str, default=None,
        help="Override the results root (default: results/cxcmu). "
             "Reward-field summary is written to <results_root>/reward_field/.",
    )
    args = parser.parse_args()
    if args.data_root is not None or args.results_root is not None:
        _set_data_root(
            Path(args.data_root) if args.data_root else GRAPH_DIR.parent,
            Path(args.results_root) if args.results_root else None,
        )
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
