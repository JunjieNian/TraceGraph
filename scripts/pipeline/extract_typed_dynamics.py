#!/usr/bin/env python3
"""Typed-state dynamics + per-model graph metric extraction for cx-cmu data.

CORE SCRIPT: This is where per-model graph metrics are extracted.

Per task:
1. Load graph payload (with reward field + basins)
2. Build typed-state sequences
3. Compute task-level metrics (committor, MFPT, escape hazard, return prob)
4. Split rollouts by model using slice_model_ids
5. Compute per-model metrics:
   - core_visit_rate: fraction of core blocks visited by this model
   - committor_dp: committor at decision points for this model's rollouts
   - mfpt_to_core: mean steps to reach core for this model
   - basin_entry_rate: fraction of this model's rollouts entering basins
   - basin_escape_rate: fraction that escape after entering
   - escape_hazard: core escape risk for this model
   - return_prob: return-to-core probability after escape
   - res_vs_fail_tv: TV distance between resolved/failed kernels

Output: results/cxcmu/typed_state_dynamics.jsonl

Usage:
    python scripts/86_typed_state_cxcmu.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tracegraph.constants import (
    CORE_POS_Q,
    EOS_FAILED,
    EOS_RESOLVED,
    LAPLACE_ALPHA,
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
from tracegraph.typed_state_mdp import (
    build_kernel,
    build_typed_sequences,
    compute_committor,
    compute_escape_hazard_3step,
    compute_mfpt_to_core,
    compute_return_prob_3step,
)

GRAPH_DIR = Path("data/cxcmu/graphs")
RESULTS_DIR = Path("results/cxcmu")


def _set_data_root(data_root: "Path | None", results_root: "Path | None") -> None:
    """Re-point GRAPH_DIR and RESULTS_DIR.

    Important: typed_state_dynamics.jsonl is a GLOBAL output (no per-bench
    subdir), so external runs MUST pass ``--results-root`` to avoid
    clobbering ``results/cxcmu/typed_state_dynamics.jsonl``.
    """
    global GRAPH_DIR, RESULTS_DIR
    if data_root is not None:
        GRAPH_DIR = Path(data_root) / "graphs"
    if results_root is not None:
        RESULTS_DIR = Path(results_root)


def _compute_tv(P_res: np.ndarray, P_fail: np.ndarray, n_states: int) -> float:
    """Standard resolved vs failed TV over all rows."""
    row_tvs = np.zeros(n_states)
    for i in range(n_states):
        row_tvs[i] = 0.5 * float(np.abs(P_res[i] - P_fail[i]).sum())
    return float(np.mean(row_tvs))


def _get_model_run_ids(payload: dict) -> dict[str, list[int]]:
    """Get mapping: model_id → list of numeric run_ids."""
    model_runs = defaultdict(set)

    # Use slice_model_ids to identify which runs belong to which model
    slice_model_ids = payload.get("slice_model_ids", [])
    cache = payload.get("role_threshold_cache", {})
    slice_runs = cache.get("slice_runs", [])

    for i, model_id in enumerate(slice_model_ids):
        if i < len(slice_runs):
            run_id = slice_runs[i]
            model_runs[model_id].add(run_id)

    return {m: sorted(rids) for m, rids in model_runs.items()}


def _compute_core_visit_rate(
    run_ids: list[int], run_sequences: dict, block_meta: dict,
    core_mask: np.ndarray, node_to_idx: dict,
) -> float:
    """Fraction of core blocks visited by these runs."""
    nodes = sorted(block_meta.keys())
    core_blocks = set()
    for n in nodes:
        idx = node_to_idx.get(n)
        if idx is not None and 0 <= idx < len(core_mask) and core_mask[idx]:
            core_blocks.add(n)

    if not core_blocks:
        return 0.0

    visited_core = set()
    for rid in run_ids:
        seq = run_sequences.get(rid, [])
        for step in seq:
            bid = step.get("primary_block")
            if bid is not None and int(bid) in core_blocks:
                visited_core.add(int(bid))

    return len(visited_core) / len(core_blocks)


def _compute_basin_rates(
    run_ids: list[int], run_block_sets: dict, basins: list[dict],
) -> tuple[float, float]:
    """Compute basin_entry_rate and basin_escape_rate for given runs."""
    if not basins or not run_ids:
        return 0.0, 0.0

    all_basin_blocks = set()
    for b in basins:
        all_basin_blocks.update(b.get("block_ids", []))

    if not all_basin_blocks:
        return 0.0, 0.0

    n_entered = 0
    n_escaped = 0  # entered but also visited non-basin blocks after

    for rid in run_ids:
        blocks_visited = run_block_sets.get(rid, set())
        basin_visited = blocks_visited & all_basin_blocks
        if basin_visited:
            n_entered += 1
            # "Escape" = visited blocks outside basin
            non_basin_visited = blocks_visited - all_basin_blocks
            if non_basin_visited:
                n_escaped += 1

    entry_rate = n_entered / len(run_ids) if run_ids else 0.0
    escape_rate = n_escaped / n_entered if n_entered > 0 else 0.0
    return entry_rate, escape_rate


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

        for gpath in tqdm(graph_files, desc=f"  Dynamics {bench}"):
            task_id = gpath.stem
            with open(gpath, "rb") as f:
                payload = pickle.load(f)

            block_meta = nontrivial_block_info(payload)
            if len(block_meta) < 3:
                continue

            run_sequences = reconstruct_run_sequences(payload)
            if not run_sequences:
                continue

            run_block_sets, block_run_sets = build_visit_sets(run_sequences)
            run_resolved = build_run_resolved(payload)
            if len(run_resolved) < 3:
                continue

            # Block graph + transition matrix
            block_adj = build_block_graph(payload, run_sequences)
            nodes = sorted(block_meta.keys())
            P_block, node_to_idx = build_transition_matrix(nodes, block_adj)

            # Reward field + core mask
            rf = payload.get("reward_field", {})
            if rf:
                core_mask = np.array(rf["core_mask"])
                field = np.array(rf["field"])
                rf_node_to_idx = {int(k): v for k, v in rf.get("node_to_idx", {}).items()}
            else:
                seed = compute_seed_vector(
                    run_id=None, nodes=nodes, block_run_sets=block_run_sets,
                    run_resolved=run_resolved, min_run_support=max(1, MIN_RUN_SUPPORT),
                    support_shrink_exp=SUPPORT_SHRINK_EXP,
                )
                field = diffuse_reward_field(seed, P_block, alpha=PROPAGATION_ALPHA, n_steps=PROPAGATION_STEPS)
                core_mask = make_core_mask(field, CORE_POS_Q)
                rf_node_to_idx = node_to_idx

            # Build typed sequences
            run_typed = build_typed_sequences(run_sequences, block_meta, core_mask, rf_node_to_idx)
            if not run_typed:
                continue

            all_states_set = set()
            for seq in run_typed.values():
                all_states_set.update(seq)
            all_states_set.add(EOS_RESOLVED)
            all_states_set.add(EOS_FAILED)
            all_states = sorted(all_states_set)

            core_states = [s for s in all_states if "|core" in s and s not in (EOS_RESOLVED, EOS_FAILED)]
            non_core_states = [s for s in all_states if "|outer" in s and s not in (EOS_RESOLVED, EOS_FAILED)]

            all_rids = sorted(run_resolved.keys())
            resolved_rids = [r for r in all_rids if run_resolved.get(r, False)]
            failed_rids = [r for r in all_rids if not run_resolved.get(r, False)]

            # ── Task-level metrics ──
            P_all, visits_all, state_idx = build_kernel(
                run_typed, all_rids, run_resolved, all_states, LAPLACE_ALPHA,
            )

            # Committor
            committor = compute_committor(P_all, state_idx, all_states)
            committor_dp = math.nan
            if committor is not None:
                dp_idx = [state_idx[s] for s in all_states
                          if "decision_point" in s and s in state_idx]
                if dp_idx:
                    dp_vals = committor[dp_idx]
                    dp_vals = dp_vals[~np.isnan(dp_vals)]
                    if len(dp_vals) > 0:
                        committor_dp = float(np.mean(dp_vals))

            # MFPT to core
            mfpt = compute_mfpt_to_core(P_all, state_idx, all_states, core_states)
            mfpt_dp = math.nan
            if mfpt is not None:
                dp_idx = [state_idx[s] for s in all_states
                          if "decision_point" in s and s in state_idx]
                if dp_idx:
                    dp_vals = mfpt[dp_idx]
                    dp_vals = dp_vals[~np.isnan(dp_vals)]
                    if len(dp_vals) > 0:
                        mfpt_dp = float(np.mean(dp_vals))

            # Escape hazard
            escape_hazard = compute_escape_hazard_3step(
                P_all, visits_all, state_idx, core_states, non_core_states,
            )

            # Return prob
            return_prob = compute_return_prob_3step(
                run_typed, all_rids, run_resolved, P_all, state_idx, core_states,
            )

            # TV: resolved vs failed
            task_tv = math.nan
            if resolved_rids and failed_rids:
                P_res, _, _ = build_kernel(
                    run_typed, resolved_rids, run_resolved, all_states, LAPLACE_ALPHA,
                )
                P_fail, _, _ = build_kernel(
                    run_typed, failed_rids, run_resolved, all_states, LAPLACE_ALPHA,
                )
                task_tv = _compute_tv(P_res, P_fail, len(all_states))

            # ── Per-model metrics ──
            model_run_ids = _get_model_run_ids(payload)
            basins = payload.get("failure_basins", {}).get("basins", [])

            per_model = {}
            for model_id, model_rids in model_run_ids.items():
                if len(model_rids) < 2:
                    continue

                # Core visit rate
                cvr = _compute_core_visit_rate(
                    model_rids, run_sequences, block_meta, core_mask, rf_node_to_idx,
                )

                # Basin entry/escape rates
                basin_entry, basin_escape = _compute_basin_rates(
                    model_rids, run_block_sets, basins,
                )

                # Per-model committor (from model-specific rollouts)
                model_typed_rids = [r for r in model_rids if r in run_typed]
                model_resolved = [r for r in model_typed_rids if run_resolved.get(r, False)]
                model_failed = [r for r in model_typed_rids if not run_resolved.get(r, False)]

                m_committor_dp = math.nan
                m_mfpt_dp = math.nan
                m_escape = math.nan
                m_return = math.nan
                m_tv = math.nan

                if len(model_typed_rids) >= 2:
                    # Build model-specific kernel
                    P_m, visits_m, sidx_m = build_kernel(
                        run_typed, model_typed_rids, run_resolved, all_states, LAPLACE_ALPHA,
                    )

                    # Model committor
                    c_m = compute_committor(P_m, sidx_m, all_states)
                    if c_m is not None:
                        dp_idx = [sidx_m[s] for s in all_states
                                  if "decision_point" in s and s in sidx_m]
                        if dp_idx:
                            dp_vals = c_m[dp_idx]
                            dp_vals = dp_vals[~np.isnan(dp_vals)]
                            if len(dp_vals) > 0:
                                m_committor_dp = float(np.mean(dp_vals))

                    # Model MFPT
                    mfpt_m = compute_mfpt_to_core(P_m, sidx_m, all_states, core_states)
                    if mfpt_m is not None:
                        dp_idx = [sidx_m[s] for s in all_states
                                  if "decision_point" in s and s in sidx_m]
                        if dp_idx:
                            dp_vals = mfpt_m[dp_idx]
                            dp_vals = dp_vals[~np.isnan(dp_vals)]
                            if len(dp_vals) > 0:
                                m_mfpt_dp = float(np.mean(dp_vals))

                    # Model escape hazard
                    m_escape = compute_escape_hazard_3step(
                        P_m, visits_m, sidx_m, core_states, non_core_states,
                    )

                    # Model return prob
                    m_return = compute_return_prob_3step(
                        run_typed, model_typed_rids, run_resolved,
                        P_m, sidx_m, core_states,
                    )

                    # Model TV (resolved vs failed)
                    if model_resolved and model_failed:
                        P_mr, _, _ = build_kernel(
                            run_typed, model_resolved, run_resolved, all_states, LAPLACE_ALPHA,
                        )
                        P_mf, _, _ = build_kernel(
                            run_typed, model_failed, run_resolved, all_states, LAPLACE_ALPHA,
                        )
                        m_tv = _compute_tv(P_mr, P_mf, len(all_states))

                per_model[model_id] = {
                    "n_runs": len(model_rids),
                    "n_resolved": sum(1 for r in model_rids if run_resolved.get(r, False)),
                    "resolve_rate": sum(1 for r in model_rids if run_resolved.get(r, False)) / len(model_rids),
                    "core_visit_rate": round(cvr, 6),
                    "committor_dp": round(m_committor_dp, 6) if not math.isnan(m_committor_dp) else None,
                    "mfpt_to_core": round(m_mfpt_dp, 4) if not math.isnan(m_mfpt_dp) else None,
                    "basin_entry_rate": round(basin_entry, 6),
                    "basin_escape_rate": round(basin_escape, 6),
                    "escape_hazard": round(m_escape, 6) if not math.isnan(m_escape) else None,
                    "return_prob": round(m_return, 6) if not math.isnan(m_return) else None,
                    "res_vs_fail_tv": round(m_tv, 6) if not math.isnan(m_tv) else None,
                }

            row = {
                "benchmark": bench,
                "task_id": task_id,
                "n_blocks": len(block_meta),
                "n_typed_states": len(all_states),
                "n_runs": len(all_rids),
                "n_resolved": len(resolved_rids),
                # Task-level metrics
                "committor_dp": round(committor_dp, 6) if not math.isnan(committor_dp) else None,
                "mfpt_dp": round(mfpt_dp, 4) if not math.isnan(mfpt_dp) else None,
                "escape_hazard_3step": round(escape_hazard, 6) if not math.isnan(escape_hazard) else None,
                "return_prob_3step": round(return_prob, 6) if not math.isnan(return_prob) else None,
                "res_vs_fail_tv": round(task_tv, 6) if not math.isnan(task_tv) else None,
                # Per-model metrics
                "per_model": per_model,
            }
            all_results.append(row)

    # Save results as JSONL
    out_path = RESULTS_DIR / "typed_state_dynamics.jsonl"
    with open(out_path, "w") as f:
        for row in all_results:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"\n── Summary ──")
    print(f"Processed {len(all_results)} tasks")
    if all_results:
        def _m(key):
            vals = [r[key] for r in all_results if r.get(key) is not None]
            return f"{np.mean(vals):.4f}" if vals else "N/A"

        print(f"  committor_dp:        {_m('committor_dp')}")
        print(f"  mfpt_dp:             {_m('mfpt_dp')}")
        print(f"  escape_hazard_3step: {_m('escape_hazard_3step')}")
        print(f"  return_prob_3step:   {_m('return_prob_3step')}")
        print(f"  res_vs_fail_tv:      {_m('res_vs_fail_tv')}")

        # Per-model summary
        model_metrics = defaultdict(lambda: defaultdict(list))
        for row in all_results:
            for model_id, metrics in row.get("per_model", {}).items():
                for k, v in metrics.items():
                    if v is not None and k not in ("n_runs", "n_resolved"):
                        model_metrics[model_id][k].append(v)

        if model_metrics:
            print(f"\n  Per-model means (across tasks):")
            for model_id in sorted(model_metrics.keys()):
                mm = model_metrics[model_id]
                cvr = np.mean(mm.get("core_visit_rate", [0]))
                c_dp = np.mean(mm.get("committor_dp", [])) if mm.get("committor_dp") else float('nan')
                print(f"    {model_id}: core_visit={cvr:.3f} committor={c_dp:.3f}")

    print(f"Output: {out_path}")


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
             "External runs MUST set this to avoid clobbering the cxcmu "
             "typed_state_dynamics.jsonl global output.",
    )
    args = parser.parse_args()
    if args.data_root is not None or args.results_root is not None:
        _set_data_root(
            Path(args.data_root) if args.data_root else None,
            Path(args.results_root) if args.results_root else None,
        )
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
