#!/usr/bin/env python3
"""RQ4 — Counterfactual MDP interventions on typed-state kernels.

4 intervention types:
  A. Basin-edge deletion: remove gate→basin transitions, renormalize, recompute committor
  B. Transition patching: replace weak model's gate outgoing with strong model's
  C. Recovery-edge boosting: amplify basin→core transitions ×2
  D. Matched random controls: same perturbation on non-gate blocks of equal degree/phase

For each intervention, report Δ_committor at decision-point states and
statistical significance vs matched random controls.

Output: results/cxcmu/counterfactual/
  - basin_deletion.json
  - transition_patching.json
  - recovery_boosting.json
  - matched_controls.json
  - summary.json

Usage:
    python scripts/93_counterfactual.py [--max-tasks N] [--benchmark BENCH]
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
    EOS_FAILED,
    EOS_RESOLVED,
    LAPLACE_ALPHA,
)
from tracegraph.reward_field import (
    build_block_graph,
    build_run_resolved,
    build_transition_matrix,
    build_visit_sets,
    nontrivial_block_info,
    reconstruct_run_sequences,
)
from tracegraph.typed_state_mdp import (
    build_kernel,
    build_typed_sequences,
    compute_committor,
)

GRAPH_DIR = Path("data/cxcmu/graphs")
RESULTS_DIR = Path("results/cxcmu/counterfactual")


def _get_dp_committor(committor: np.ndarray | None, state_idx: dict, all_states: list) -> float:
    """Mean committor at decision_point states."""
    if committor is None:
        return math.nan
    dp_idx = [state_idx[s] for s in all_states if "decision_point" in s and s in state_idx]
    if not dp_idx:
        return math.nan
    dp_vals = committor[dp_idx]
    dp_vals = dp_vals[~np.isnan(dp_vals)]
    return float(np.mean(dp_vals)) if len(dp_vals) > 0 else math.nan


def _identify_gate_states(all_states: list, basins_data: list, block_meta: dict,
                          core_mask: np.ndarray, rf_node_to_idx: dict) -> tuple[list[str], list[str]]:
    """Identify gate states (decision_point adjacent to basins) and basin states."""
    # Basin blocks
    all_basin_blocks = set()
    for basin in basins_data:
        all_basin_blocks.update(basin.get("block_ids", []))

    # Core blocks
    core_blocks = set()
    for bid in block_meta:
        idx = rf_node_to_idx.get(bid)
        if idx is not None and idx < len(core_mask) and core_mask[idx]:
            core_blocks.add(bid)

    # Gate states: decision_point states that are "outer" (near basins)
    gate_states = [s for s in all_states
                   if "decision_point" in s and s not in (EOS_RESOLVED, EOS_FAILED)]
    basin_states = [s for s in all_states
                    if "weak_basin" in s and s not in (EOS_RESOLVED, EOS_FAILED)]

    return gate_states, basin_states


def _modify_kernel_delete_basin_edges(
    P: np.ndarray,
    state_idx: dict,
    gate_states: list[str],
    basin_states: list[str],
) -> np.ndarray:
    """Delete gate→basin transitions and renormalize."""
    P_new = P.copy()
    basin_idx = {state_idx[s] for s in basin_states if s in state_idx}

    for gs in gate_states:
        if gs not in state_idx:
            continue
        gi = state_idx[gs]
        # Zero out transitions to basin states
        for bi in basin_idx:
            P_new[gi, bi] = 0.0
        # Renormalize row
        row_sum = P_new[gi].sum()
        if row_sum > 1e-12:
            P_new[gi] /= row_sum
        else:
            # If all mass was removed, set self-loop
            P_new[gi, gi] = 1.0

    return P_new


def _modify_kernel_patch_transitions(
    P_weak: np.ndarray,
    P_strong: np.ndarray,
    state_idx: dict,
    gate_states: list[str],
) -> np.ndarray:
    """Replace weak model's gate outgoing distribution with strong model's."""
    P_patched = P_weak.copy()
    for gs in gate_states:
        if gs not in state_idx:
            continue
        gi = state_idx[gs]
        P_patched[gi] = P_strong[gi].copy()
    return P_patched


def _modify_kernel_boost_recovery(
    P: np.ndarray,
    state_idx: dict,
    basin_states: list[str],
    core_states: list[str],
    boost_factor: float = 2.0,
) -> np.ndarray:
    """Boost basin→core transition probabilities by factor, then renormalize."""
    P_new = P.copy()
    core_idx = {state_idx[s] for s in core_states if s in state_idx}

    for bs in basin_states:
        if bs not in state_idx:
            continue
        bi = state_idx[bs]
        # Boost transitions to core states
        for ci in core_idx:
            P_new[bi, ci] *= boost_factor
        # Renormalize
        row_sum = P_new[bi].sum()
        if row_sum > 1e-12:
            P_new[bi] /= row_sum

    return P_new


def _matched_random_control(
    P: np.ndarray,
    state_idx: dict,
    all_states: list[str],
    target_states: list[str],
    intervention_fn,
    n_controls: int = 100,
    rng: np.random.Generator = None,
) -> list[float]:
    """Apply the same intervention to random non-target states of similar phase."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Find non-target transient states (not EOS, not in target set)
    target_set = set(target_states)
    eligible = [s for s in all_states
                if s not in target_set
                and s not in (EOS_RESOLVED, EOS_FAILED)
                and s in state_idx]

    if not eligible or not target_states:
        return []

    n_target = len(target_states)
    control_deltas = []

    for _ in range(n_controls):
        # Sample random states of same count
        if len(eligible) >= n_target:
            random_states = rng.choice(eligible, size=n_target, replace=False).tolist()
        else:
            random_states = list(eligible)

        # Apply intervention
        P_ctrl = intervention_fn(P, state_idx, random_states)

        # Compute committor on modified kernel
        committor_ctrl = compute_committor(P_ctrl, state_idx, all_states)
        dp_ctrl = _get_dp_committor(committor_ctrl, state_idx, all_states)
        if not math.isnan(dp_ctrl):
            control_deltas.append(dp_ctrl)

    return control_deltas


def process_task(payload: dict) -> dict | None:
    """Run all 4 counterfactual interventions on one task."""
    block_meta = nontrivial_block_info(payload)
    if len(block_meta) < 3:
        return None

    run_sequences = reconstruct_run_sequences(payload)
    if not run_sequences:
        return None

    run_resolved = build_run_resolved(payload)
    if len(run_resolved) < 3:
        return None

    # Reward field
    rf = payload.get("reward_field", {})
    if not rf:
        return None
    core_mask = np.array(rf["core_mask"])
    rf_node_to_idx = {int(k): v for k, v in rf.get("node_to_idx", {}).items()}

    # Build typed sequences
    run_typed = build_typed_sequences(run_sequences, block_meta, core_mask, rf_node_to_idx)
    if not run_typed:
        return None

    all_states_set = set()
    for seq in run_typed.values():
        all_states_set.update(seq)
    all_states_set.add(EOS_RESOLVED)
    all_states_set.add(EOS_FAILED)
    all_states = sorted(all_states_set)

    core_states = [s for s in all_states if "|core" in s and s not in (EOS_RESOLVED, EOS_FAILED)]
    basins_data = payload.get("failure_basins", {}).get("basins", [])

    # Identify gate and basin states
    gate_states, basin_states = _identify_gate_states(
        all_states, basins_data, block_meta, core_mask, rf_node_to_idx,
    )

    if not gate_states and not basin_states:
        return None

    all_rids = sorted(run_resolved.keys())

    # Build baseline kernel
    P_base, visits_base, state_idx = build_kernel(
        run_typed, all_rids, run_resolved, all_states, LAPLACE_ALPHA,
    )
    committor_base = compute_committor(P_base, state_idx, all_states)
    base_dp = _get_dp_committor(committor_base, state_idx, all_states)

    if math.isnan(base_dp):
        return None

    rng = np.random.default_rng(42)
    result = {"base_committor_dp": round(base_dp, 6)}

    # ── A. Basin-edge deletion ──
    if gate_states and basin_states:
        P_del = _modify_kernel_delete_basin_edges(P_base, state_idx, gate_states, basin_states)
        committor_del = compute_committor(P_del, state_idx, all_states)
        del_dp = _get_dp_committor(committor_del, state_idx, all_states)
        delta_del = del_dp - base_dp if not math.isnan(del_dp) else math.nan

        result["basin_deletion"] = {
            "new_committor_dp": round(del_dp, 6) if not math.isnan(del_dp) else None,
            "delta": round(delta_del, 6) if not math.isnan(delta_del) else None,
            "n_gate_states": len(gate_states),
            "n_basin_states": len(basin_states),
        }

    # ── B. Transition patching (strongest vs weakest model) ──
    model_runs = defaultdict(set)
    slice_model_ids = payload.get("slice_model_ids", [])
    cache = payload.get("role_threshold_cache", {})
    slice_runs = cache.get("slice_runs", [])
    for i, mid in enumerate(slice_model_ids):
        if i < len(slice_runs):
            model_runs[mid].add(int(slice_runs[i]))

    # Find strongest/weakest model by resolve rate
    model_rr = {}
    for mid, rids in model_runs.items():
        n_res = sum(1 for r in rids if run_resolved.get(r, False))
        model_rr[mid] = n_res / len(rids) if rids else 0.0

    if len(model_rr) >= 2:
        strongest = max(model_rr, key=model_rr.get)
        weakest = min(model_rr, key=model_rr.get)

        if model_rr[strongest] > model_rr[weakest]:
            # Build per-model kernels
            strong_rids = sorted(r for r in model_runs[strongest] if r in run_typed)
            weak_rids = sorted(r for r in model_runs[weakest] if r in run_typed)

            if len(strong_rids) >= 2 and len(weak_rids) >= 2:
                P_strong, _, _ = build_kernel(run_typed, strong_rids, run_resolved, all_states, LAPLACE_ALPHA)
                P_weak, _, _ = build_kernel(run_typed, weak_rids, run_resolved, all_states, LAPLACE_ALPHA)

                # Patch weak with strong at gates
                if gate_states:
                    P_patched = _modify_kernel_patch_transitions(P_weak, P_strong, state_idx, gate_states)
                    committor_patched = compute_committor(P_patched, state_idx, all_states)
                    patched_dp = _get_dp_committor(committor_patched, state_idx, all_states)

                    committor_weak = compute_committor(P_weak, state_idx, all_states)
                    weak_dp = _get_dp_committor(committor_weak, state_idx, all_states)

                    delta_patch = patched_dp - weak_dp if not (math.isnan(patched_dp) or math.isnan(weak_dp)) else math.nan

                    result["transition_patching"] = {
                        "strong_model": strongest,
                        "weak_model": weakest,
                        "strong_rr": round(model_rr[strongest], 4),
                        "weak_rr": round(model_rr[weakest], 4),
                        "weak_committor_dp": round(weak_dp, 6) if not math.isnan(weak_dp) else None,
                        "patched_committor_dp": round(patched_dp, 6) if not math.isnan(patched_dp) else None,
                        "delta": round(delta_patch, 6) if not math.isnan(delta_patch) else None,
                    }

    # ── C. Recovery-edge boosting ──
    if basin_states and core_states:
        P_boost = _modify_kernel_boost_recovery(P_base, state_idx, basin_states, core_states)
        committor_boost = compute_committor(P_boost, state_idx, all_states)
        boost_dp = _get_dp_committor(committor_boost, state_idx, all_states)
        delta_boost = boost_dp - base_dp if not math.isnan(boost_dp) else math.nan

        result["recovery_boosting"] = {
            "new_committor_dp": round(boost_dp, 6) if not math.isnan(boost_dp) else None,
            "delta": round(delta_boost, 6) if not math.isnan(delta_boost) else None,
            "boost_factor": 2.0,
        }

    # ── D. Matched random controls (for basin deletion) ──
    if gate_states and basin_states:
        def _deletion_intervention(P, sidx, states):
            P_new = P.copy()
            basin_idx_set = {sidx[s] for s in basin_states if s in sidx}
            for gs in states:
                if gs not in sidx:
                    continue
                gi = sidx[gs]
                for bi in basin_idx_set:
                    P_new[gi, bi] = 0.0
                row_sum = P_new[gi].sum()
                if row_sum > 1e-12:
                    P_new[gi] /= row_sum
                else:
                    P_new[gi, gi] = 1.0
            return P_new

        control_dps = _matched_random_control(
            P_base, state_idx, all_states, gate_states,
            _deletion_intervention, n_controls=100, rng=rng,
        )

        if control_dps and "basin_deletion" in result:
            del_dp_val = result["basin_deletion"].get("new_committor_dp")
            if del_dp_val is not None:
                # Compare deltas from baseline, not absolute values
                obs_delta = del_dp_val - base_dp
                ctrl_deltas = [cd - base_dp for cd in control_dps]
                # p-value: fraction of controls with delta >= observed delta
                # Using Phipson & Smyth (2010) correction: (1 + count) / (1 + total)
                p_val = (1 + sum(1 for cd in ctrl_deltas if cd >= obs_delta)) / (1 + len(ctrl_deltas))
                result["matched_controls"] = {
                    "n_controls": len(control_dps),
                    "control_mean_delta": round(float(np.mean(ctrl_deltas)), 6),
                    "control_std_delta": round(float(np.std(ctrl_deltas)), 6),
                    "intervention_delta": round(obs_delta, 6),
                    "p_value": round(p_val, 4),
                    "significant": p_val < 0.05,
                }

    return result


def main(max_tasks: int | None = None, benchmark: str | None = None):
    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    print(f"Counterfactual interventions: {benchmarks}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for bench in benchmarks:
        bench_dir = GRAPH_DIR / bench
        graph_files = sorted(bench_dir.glob("*.pkl"))
        if max_tasks:
            graph_files = graph_files[:max_tasks]

        print(f"\n── {bench}: {len(graph_files)} tasks ──")
        for gpath in tqdm(graph_files, desc=f"  CF {bench}"):
            with open(gpath, "rb") as f:
                payload = pickle.load(f)

            result = process_task(payload)
            if result is not None:
                result["benchmark"] = bench
                result["task_id"] = gpath.stem
                all_results.append(result)

    # Aggregate results
    print(f"\n── Summary: {len(all_results)} valid tasks ──")

    # Basin deletion statistics
    del_deltas = [r["basin_deletion"]["delta"] for r in all_results
                  if "basin_deletion" in r and r["basin_deletion"].get("delta") is not None]
    if del_deltas:
        print(f"\n  A. Basin Deletion:")
        print(f"     Mean Δ_committor: {np.mean(del_deltas):.4f} (±{np.std(del_deltas):.4f})")
        print(f"     Δ > 0: {sum(1 for d in del_deltas if d > 0)}/{len(del_deltas)}")
        # One-sample t-test
        t_stat, p_val = scipy_stats.ttest_1samp(del_deltas, 0)
        print(f"     t-test H0: Δ=0: t={t_stat:.3f}, p={p_val:.4f}")

    # Transition patching
    patch_deltas = [r["transition_patching"]["delta"] for r in all_results
                    if "transition_patching" in r and r["transition_patching"].get("delta") is not None]
    if patch_deltas:
        print(f"\n  B. Transition Patching (weak→strong at gates):")
        print(f"     Mean Δ_committor: {np.mean(patch_deltas):.4f} (±{np.std(patch_deltas):.4f})")
        print(f"     Δ > 0: {sum(1 for d in patch_deltas if d > 0)}/{len(patch_deltas)}")

    # Recovery boosting
    boost_deltas = [r["recovery_boosting"]["delta"] for r in all_results
                    if "recovery_boosting" in r and r["recovery_boosting"].get("delta") is not None]
    if boost_deltas:
        print(f"\n  C. Recovery Boosting (×2):")
        print(f"     Mean Δ_committor: {np.mean(boost_deltas):.4f} (±{np.std(boost_deltas):.4f})")
        print(f"     Δ > 0: {sum(1 for d in boost_deltas if d > 0)}/{len(boost_deltas)}")

    # Matched controls
    ctrl_results = [r for r in all_results if "matched_controls" in r]
    if ctrl_results:
        n_sig = sum(1 for r in ctrl_results if r["matched_controls"].get("significant", False))
        print(f"\n  D. Matched Controls:")
        print(f"     {n_sig}/{len(ctrl_results)} tasks: gate intervention > random (p < 0.05)")
        p_vals = [r["matched_controls"]["p_value"] for r in ctrl_results
                  if not math.isnan(r["matched_controls"]["p_value"])]
        avg_p = np.mean(p_vals) if p_vals else float("nan")
        print(f"     Mean p-value: {avg_p:.4f}")

    # Save detailed results
    with open(RESULTS_DIR / "all_tasks.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save summary
    summary = {
        "n_valid_tasks": len(all_results),
        "basin_deletion": {
            "n_tasks": len(del_deltas),
            "mean_delta": round(float(np.mean(del_deltas)), 6) if del_deltas else None,
            "std_delta": round(float(np.std(del_deltas)), 6) if del_deltas else None,
            "frac_positive": round(sum(1 for d in del_deltas if d > 0) / max(len(del_deltas), 1), 4) if del_deltas else None,
        },
        "transition_patching": {
            "n_tasks": len(patch_deltas),
            "mean_delta": round(float(np.mean(patch_deltas)), 6) if patch_deltas else None,
            "std_delta": round(float(np.std(patch_deltas)), 6) if patch_deltas else None,
            "frac_positive": round(sum(1 for d in patch_deltas if d > 0) / max(len(patch_deltas), 1), 4) if patch_deltas else None,
        },
        "recovery_boosting": {
            "n_tasks": len(boost_deltas),
            "mean_delta": round(float(np.mean(boost_deltas)), 6) if boost_deltas else None,
            "std_delta": round(float(np.std(boost_deltas)), 6) if boost_deltas else None,
            "frac_positive": round(sum(1 for d in boost_deltas if d > 0) / max(len(boost_deltas), 1), 4) if boost_deltas else None,
        },
        "matched_controls": {
            "n_tasks": len(ctrl_results),
            "n_significant": n_sig if ctrl_results else 0,
            "frac_significant": round(n_sig / max(len(ctrl_results), 1), 4) if ctrl_results else None,
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Output ──")
    print(f"  {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
