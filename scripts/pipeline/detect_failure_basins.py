#!/usr/bin/env python3
"""Detect failure basins and recovery gates for cx-cmu task graphs.

Per task:
1. Load graph payload (with reward field)
2. Detect failure basins
3. Detect recovery gates
4. Detect loop motifs
5. Save basin/gate info to graph payload

Usage:
    python scripts/85_failure_basins_cxcmu.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tracegraph.constants import (
    FAILURE_BASIN_MAX_ESCAPE_3STEP,
    FAILURE_BASIN_MIN_FAIL_RATE,
    RECOVERY_GATE_MIN_UPLIFT,
)
from tracegraph.failure_basins import (
    basin_summary_statistics,
    detect_failure_basins,
    detect_loop_motifs,
    detect_recovery_gates,
)
from tracegraph.graph_construction import nontrivial_blocks
from tracegraph.reward_field import (
    build_block_graph,
    build_run_resolved,
    build_visit_sets,
    nontrivial_block_info,
    reconstruct_run_sequences,
)

GRAPH_DIR = Path("data/cxcmu/graphs")
RESULTS_DIR = Path("results/cxcmu/failure_basins")


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

        for gpath in tqdm(graph_files, desc=f"  Basins {bench}"):
            task_id = gpath.stem
            with open(gpath, "rb") as f:
                payload = pickle.load(f)

            # Block info
            blocks = payload.get("bcc_analysis", {}).get("blocks", [])
            nt_blocks = nontrivial_blocks(blocks)
            if not nt_blocks:
                continue

            # Run sequences
            run_sequences = reconstruct_run_sequences(payload)
            if not run_sequences:
                continue

            # Visit sets
            run_block_sets, block_run_sets = build_visit_sets(run_sequences)
            run_resolved = build_run_resolved(payload)

            if len(run_resolved) < 2:
                continue

            # Block adjacency
            block_adj = build_block_graph(payload, run_sequences)

            # Detect failure basins
            basins = detect_failure_basins(
                blocks=nt_blocks,
                block_adjacency=block_adj,
                run_block_sets=run_block_sets,
                run_resolved=run_resolved,
                run_sequences=run_sequences,
                min_fail_rate=FAILURE_BASIN_MIN_FAIL_RATE,
                max_escape_3step=FAILURE_BASIN_MAX_ESCAPE_3STEP,
            )

            # Get success cores from reward field
            rf = payload.get("reward_field", {})
            success_cores = rf.get("components", [])
            if not success_cores:
                # Use core_mask to identify core blocks
                core_mask = np.array(rf.get("core_mask", []))
                nodes = rf.get("nodes", [])
                if len(core_mask) > 0 and len(nodes) > 0:
                    core_blocks = [nodes[i] for i in range(len(core_mask)) if core_mask[i]]
                    if core_blocks:
                        success_cores = [core_blocks]

            # Detect recovery gates
            gates = detect_recovery_gates(
                basins=basins,
                success_cores=success_cores,
                blocks=nt_blocks,
                block_adjacency=block_adj,
                run_block_sets=run_block_sets,
                run_resolved=run_resolved,
                run_sequences=run_sequences,
            )

            # Detect loop motifs
            motifs = detect_loop_motifs(run_sequences, run_resolved)

            # Summary
            summary = basin_summary_statistics(basins, gates, run_resolved)

            row = {
                "benchmark": bench,
                "task_id": task_id,
                **summary,
                "n_significant_gates": sum(
                    1 for g in gates if g.get("success_uplift", 0) >= RECOVERY_GATE_MIN_UPLIFT
                ),
                "n_loop_motifs": len(motifs),
            }
            all_results.append(row)

            # Save basin info to payload
            payload["failure_basins"] = {
                "basins": basins,
                "gates": gates,
                "loop_motifs": motifs[:20],
                "summary": summary,
            }
            with open(gpath, "wb") as f:
                pickle.dump(payload, f)

    # Save summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n── Summary ──")
    print(f"Processed {len(all_results)} tasks")
    if all_results:
        n_basins = [r.get("n_basins", 0) for r in all_results]
        n_gates = [r.get("n_gates", 0) for r in all_results]
        print(f"  Tasks with ≥1 basin: {sum(1 for n in n_basins if n >= 1)}/{len(all_results)}")
        print(f"  Mean basins/task: {np.mean(n_basins):.2f}")
        print(f"  Mean gates/task: {np.mean(n_gates):.2f}")
    print(f"Output: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
