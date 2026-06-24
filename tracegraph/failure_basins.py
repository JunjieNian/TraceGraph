"""Failure basin detection and recovery gate identification.

NEW module for TraceGraph — no SliceGraph counterpart.

A failure basin is a connected region of BCC blocks with high fail rate
and low escape probability.  Recovery gates are blocks (often at
articulation points) that connect failure basins to success cores and
exhibit measurable success uplift.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Failure basin detection
# ═══════════════════════════════════════════════════════════════════════

def _block_fail_rate(
    block_id: int,
    run_block_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
) -> Tuple[float, int, int]:
    """Compute fail rate for a single block.

    Returns (fail_rate, n_visitors, n_failed).
    """
    visitors = [
        rid for rid, bset in run_block_sets.items()
        if block_id in bset
    ]
    if not visitors:
        return 0.0, 0, 0
    n_failed = sum(1 for rid in visitors if not run_resolved.get(rid, False))
    return n_failed / len(visitors), len(visitors), n_failed


def detect_failure_basins(
    blocks: List[dict],
    block_adjacency: Dict[int, Set[int]],
    run_block_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
    run_sequences: Dict[int, List[dict]],
    min_fail_rate: float = 0.7,
    max_escape_3step: float = 0.3,
) -> List[dict]:
    """Find connected block regions with high fail rate and low escape probability.

    Algorithm:
    1. Identify seed blocks with fail_rate >= min_fail_rate
    2. Grow connected components from seed blocks (only including blocks
       with fail_rate >= min_fail_rate * 0.8)
    3. Filter by 3-step escape probability
    4. Return basin dicts with metadata

    Uses ``run_sequences`` to compute transition counts from compact block
    sequences (temporal ordering) rather than approximating from sorted
    block set pairs.

    Returns list of basin dicts: {basin_id, block_ids, fail_rate, escape_prob,
    n_trajectories, n_failed, is_sticky}
    """
    # Step 1: compute per-block fail rates
    block_fail: Dict[int, float] = {}
    block_visitors: Dict[int, int] = {}
    nontrivial_ids = set()
    for b in blocks:
        if b.get("is_trivial", False):
            continue
        bid = b["block_id"]
        nontrivial_ids.add(bid)
        fr, nv, nf = _block_fail_rate(bid, run_block_sets, run_resolved)
        block_fail[bid] = fr
        block_visitors[bid] = nv

    # Step 2: find connected components of high-fail-rate blocks
    seed_blocks = {bid for bid, fr in block_fail.items() if fr >= min_fail_rate}
    relaxed_threshold = min_fail_rate * 0.8
    expandable = {bid for bid, fr in block_fail.items() if fr >= relaxed_threshold}

    seen: Set[int] = set()
    basin_components: List[Set[int]] = []
    for start in sorted(seed_blocks):
        if start in seen:
            continue
        comp: Set[int] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            for v in block_adjacency.get(u, set()):
                if v in expandable and v not in seen:
                    stack.append(v)
        if comp:
            basin_components.append(comp)

    # Step 3: build transition counts from compact block sequences
    block_transition_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for rid, seq in run_sequences.items():
        # Build compact block sequence: consecutive duplicates removed
        compact_path: List[int] = []
        for step in seq:
            primary = step.get("primary_block")
            if primary is None:
                continue
            bid = int(primary)
            if not compact_path or compact_path[-1] != bid:
                compact_path.append(bid)
        # Count directed transitions between consecutive blocks
        for a, b in zip(compact_path[:-1], compact_path[1:]):
            if a != b:
                block_transition_counts[(a, b)] += 1

    # Step 4: filter and build basin dicts
    basins: List[dict] = []
    for basin_id, comp in enumerate(basin_components):
        # Aggregate fail rate
        total_visits = 0
        total_failed = 0
        for bid in comp:
            visitors = [
                rid for rid, bset in run_block_sets.items()
                if bid in bset
            ]
            for rid in visitors:
                total_visits += 1
                if not run_resolved.get(rid, False):
                    total_failed += 1
        fail_rate = total_failed / max(1, total_visits)

        # Compute escape probability
        escape_prob = compute_escape_probability(
            comp, block_transition_counts, horizon=3,
        )

        # Filter by escape probability
        if escape_prob > max_escape_3step and len(comp) <= 2:
            continue

        # Determine stickiness (high self-transition ratio)
        is_sticky = escape_prob < 0.15

        # Unique trajectories through basin
        traj_in_basin = {
            rid for rid, bset in run_block_sets.items()
            if bset & comp
        }
        n_failed_trajs = sum(
            1 for rid in traj_in_basin
            if not run_resolved.get(rid, False)
        )

        basins.append({
            "basin_id": basin_id,
            "block_ids": sorted(comp),
            "n_blocks": len(comp),
            "fail_rate": round(fail_rate, 4),
            "escape_prob": round(escape_prob, 4),
            "n_trajectories": len(traj_in_basin),
            "n_failed": n_failed_trajs,
            "is_sticky": is_sticky,
        })

    return basins


def compute_escape_probability(
    basin_blocks: Set[int],
    block_transition_counts: Dict[Tuple[int, int], int],
    horizon: int = 3,
) -> float:
    """P(B_{t+h} not in basin | B_t in basin) over h steps.

    Uses a simple Markov approximation: from observed transition counts,
    estimate transition probabilities out of the basin, then compute
    h-step escape probability.
    """
    if not basin_blocks:
        return 1.0

    # Count transitions from basin blocks
    total_out = 0
    total_transitions = 0
    for (src, tgt), count in block_transition_counts.items():
        if src in basin_blocks:
            total_transitions += count
            if tgt not in basin_blocks:
                total_out += count

    if total_transitions == 0:
        return 0.5  # uninformative prior

    # Single-step escape probability
    p_escape_1 = total_out / total_transitions

    # h-step: P(escape in h steps) = 1 - (1 - p_escape_1)^h
    p_stay_h = (1.0 - p_escape_1) ** horizon
    return 1.0 - p_stay_h


# ═══════════════════════════════════════════════════════════════════════
# Block-level articulation point detection
# ═══════════════════════════════════════════════════════════════════════

def find_block_articulation_points(
    block_adjacency: Dict[int, Set[int]],
) -> Set[int]:
    """Find articulation points in the block adjacency graph.

    These are block_ids whose removal would disconnect the block adjacency
    graph (a *block-level* articulation, as opposed to the slice-node-level
    articulation points returned by ``graph_construction.find_articulation_points``).

    Uses iterative Tarjan's algorithm on the block adjacency graph.
    """
    # Build a compact index mapping for blocks present in adjacency
    all_blocks = set(block_adjacency.keys())
    for nbrs in block_adjacency.values():
        all_blocks |= nbrs
    block_list = sorted(all_blocks)
    if len(block_list) <= 2:
        return set()

    bid_to_idx: Dict[int, int] = {bid: i for i, bid in enumerate(block_list)}
    n = len(block_list)

    # Build adjacency list in compact index space
    adj: List[List[int]] = [[] for _ in range(n)]
    for bid, nbrs in block_adjacency.items():
        u = bid_to_idx[bid]
        for nb in nbrs:
            v = bid_to_idx.get(nb)
            if v is not None and v != u:
                adj[u].append(v)

    # Iterative Tarjan's
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_ap = [False] * n
    timer = 0

    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        child_count = 0
        stack = [(root, 0)]

        while stack:
            u, idx = stack[-1]
            if idx < len(adj[u]):
                stack[-1] = (u, idx + 1)
                v = adj[u][idx]
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    if u == root:
                        child_count += 1
                    stack.append((v, 0))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                if stack:
                    p = parent[u]
                    low[p] = min(low[p], low[u])
                    if p != root and low[u] >= disc[p]:
                        is_ap[p] = True
        if child_count > 1:
            is_ap[root] = True

    return {block_list[i] for i in range(n) if is_ap[i]}


# ═══════════════════════════════════════════════════════════════════════
# Recovery gate detection
# ═══════════════════════════════════════════════════════════════════════

def detect_recovery_gates(
    basins: List[dict],
    success_cores: List[List[int]],
    blocks: List[dict],
    block_adjacency: Dict[int, Set[int]],
    run_block_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
    run_sequences: Dict[int, List[dict]],
) -> List[dict]:
    """Find blocks connecting failure basins to success cores.

    A recovery gate is a block (often an articulation point) that:
    1. Is adjacent to at least one failure basin block
    2. Is adjacent to or part of a success core
    3. Has positive success uplift Δ(g) >= threshold

    Block-level articulation is determined by ``find_block_articulation_points``
    (removal of the block disconnects the block adjacency graph), NOT by
    slice-node articulation points from the BCC decomposition.

    Success uplift is computed temporally via ``compute_success_uplift``
    using ``run_sequences``.

    Returns list of gate dicts: {gate_id, block_id, is_articulation,
    connected_basins, connected_cores, success_uplift, n_visits_after_basin,
    n_resolved_after_basin}
    """
    if not basins or not success_cores:
        return []

    # Block-level articulation points (not slice-node articulation points)
    block_aps = find_block_articulation_points(block_adjacency)

    basin_block_to_basin: Dict[int, int] = {}
    all_basin_blocks: Set[int] = set()
    for basin in basins:
        for bid in basin["block_ids"]:
            basin_block_to_basin[bid] = basin["basin_id"]
            all_basin_blocks.add(bid)

    core_block_to_core: Dict[int, int] = {}
    all_core_blocks: Set[int] = set()
    for ci, core in enumerate(success_cores):
        for bid in core:
            core_block_to_core[bid] = ci
            all_core_blocks.add(bid)

    # Candidate gates: blocks adjacent to both basin and core regions
    nontrivial_ids = {b["block_id"] for b in blocks if not b.get("is_trivial", False)}
    candidates: Set[int] = set()
    for bid in nontrivial_ids:
        if bid in all_basin_blocks:
            continue
        nbrs = block_adjacency.get(bid, set())
        touches_basin = bool(nbrs & all_basin_blocks)
        touches_core = bool((nbrs & all_core_blocks) or (bid in all_core_blocks))
        if touches_basin and touches_core:
            candidates.add(bid)

    # Also consider block-level articulation points adjacent to basins
    for ap_bid in block_aps:
        if ap_bid in all_basin_blocks:
            continue
        nbrs = block_adjacency.get(ap_bid, set())
        if nbrs & all_basin_blocks:
            candidates.add(ap_bid)

    gates: List[dict] = []
    for gate_id, bid in enumerate(sorted(candidates)):
        connected_basins = sorted({
            basin_block_to_basin[nb]
            for nb in block_adjacency.get(bid, set())
            if nb in basin_block_to_basin
        })
        connected_cores = sorted({
            core_block_to_core.get(nb, core_block_to_core.get(bid, -1))
            for nb in (block_adjacency.get(bid, set()) | {bid})
            if nb in core_block_to_core
        })
        connected_cores = [c for c in connected_cores if c >= 0]

        # Temporal success uplift via run_sequences
        uplift = compute_success_uplift(
            bid, all_basin_blocks, run_block_sets, run_resolved, run_sequences,
        )

        # Count visits and resolutions for runs that visit gate after basin
        n_visits, n_resolved = _count_gate_visits(
            bid, all_basin_blocks, run_block_sets, run_resolved, run_sequences,
        )

        gates.append({
            "gate_id": gate_id,
            "block_id": bid,
            "is_articulation": bid in block_aps,
            "connected_basins": connected_basins,
            "connected_cores": connected_cores,
            "success_uplift": round(uplift, 4),
            "n_visits_after_basin": n_visits,
            "n_resolved_after_basin": n_resolved,
        })

    return gates


def _compute_gate_uplift(
    gate_block: int,
    basin_blocks: Set[int],
    run_block_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
) -> Tuple[float, int, int]:
    """Compute success uplift for a gate block.

    Δ(g) = P(resolved | visit g AND visit basin) - P(resolved | visit basin, no g)
    """
    # Runs that visit the basin
    basin_runs = {
        rid for rid, bset in run_block_sets.items()
        if bset & basin_blocks
    }
    if not basin_runs:
        return 0.0, 0, 0

    # Among basin runs, split by whether they visit the gate
    with_gate = {
        rid for rid in basin_runs
        if gate_block in run_block_sets.get(rid, set())
    }
    without_gate = basin_runs - with_gate

    n_with = len(with_gate)
    n_without = len(without_gate)

    if n_with == 0:
        return 0.0, 0, 0

    p_resolved_with = sum(
        1 for rid in with_gate if run_resolved.get(rid, False)
    ) / n_with
    p_resolved_without = (
        sum(1 for rid in without_gate if run_resolved.get(rid, False)) / n_without
        if n_without > 0 else 0.0
    )

    uplift = p_resolved_with - p_resolved_without
    n_resolved = sum(1 for rid in with_gate if run_resolved.get(rid, False))
    return uplift, n_with, n_resolved


def _count_gate_visits(
    gate_block: int,
    basin_blocks: Set[int],
    run_block_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
    run_sequences: Dict[int, List[dict]],
) -> Tuple[int, int]:
    """Count runs that visit *gate_block* after entering *basin_blocks*.

    Returns (n_visits_after_basin, n_resolved_after_basin).
    Uses temporal ordering from run_sequences.
    """
    basin_runs = {
        rid for rid, bset in run_block_sets.items()
        if bset & basin_blocks
    }
    n_visits = 0
    n_resolved = 0
    for rid in basin_runs:
        seq = run_sequences.get(rid, [])
        entered_basin = False
        visited_gate_after = False
        for step in seq:
            primary = step.get("primary_block")
            if primary is None:
                continue
            bids = step.get("block_ids", (primary,))
            if any(int(b) in basin_blocks for b in bids):
                entered_basin = True
            if entered_basin and int(primary) == gate_block:
                visited_gate_after = True
                break
        if visited_gate_after:
            n_visits += 1
            if run_resolved.get(rid, False):
                n_resolved += 1
    return n_visits, n_resolved


def compute_success_uplift(
    gate_block: int,
    basin_blocks: Set[int],
    run_block_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
    run_sequences: Dict[int, List[dict]],
) -> float:
    """Δ(g) = P(resolved | visit g after basin) - P(resolved | enter basin, no g).

    Uses temporal ordering from run_sequences to determine "after basin".
    """
    basin_runs = {
        rid for rid, bset in run_block_sets.items()
        if bset & basin_blocks
    }
    if not basin_runs:
        return 0.0

    visits_gate_after_basin = set()
    visits_basin_no_gate = set()

    for rid in basin_runs:
        seq = run_sequences.get(rid, [])
        if not seq:
            if gate_block in run_block_sets.get(rid, set()):
                visits_gate_after_basin.add(rid)
            else:
                visits_basin_no_gate.add(rid)
            continue

        entered_basin = False
        visited_gate_after = False
        for step in seq:
            primary = step.get("primary_block")
            if primary is None:
                continue
            bids = step.get("block_ids", (primary,))
            if any(int(b) in basin_blocks for b in bids):
                entered_basin = True
            if entered_basin and int(primary) == gate_block:
                visited_gate_after = True
                break

        if visited_gate_after:
            visits_gate_after_basin.add(rid)
        else:
            visits_basin_no_gate.add(rid)

    n_with = len(visits_gate_after_basin)
    n_without = len(visits_basin_no_gate)
    if n_with == 0:
        return 0.0

    p_with = sum(
        1 for r in visits_gate_after_basin if run_resolved.get(r, False)
    ) / n_with
    p_without = (
        sum(1 for r in visits_basin_no_gate if run_resolved.get(r, False)) / n_without
        if n_without > 0 else 0.0
    )
    return p_with - p_without


# ═══════════════════════════════════════════════════════════════════════
# Loop motif detection
# ═══════════════════════════════════════════════════════════════════════

def detect_loop_motifs(
    run_sequences: Dict[int, List[dict]],
    run_resolved: Dict[int, bool],
    min_repeat: int = 3,
) -> List[dict]:
    """Detect repeated block visit patterns (retry loops, dead-end cycles).

    Common agent motifs:
    - pytest fail → edit → pytest fail (test-retry loop)
    - grep/search → grep/search (search loop)
    - edit → error → edit → error (fix-retry loop)

    Returns list of loop dicts with pattern, frequency, and resolution info.
    """
    loop_counts: Dict[Tuple[int, ...], Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "resolved": 0, "failed": 0}
    )

    for rid, seq in run_sequences.items():
        # Extract primary block sequence
        block_seq = []
        for step in seq:
            primary = step.get("primary_block")
            if primary is not None:
                block_seq.append(int(primary))

        if len(block_seq) < min_repeat:
            continue

        is_resolved = run_resolved.get(rid, False)

        # Detect repeating subsequences of length 1-3
        for pattern_len in range(1, 4):
            for start in range(len(block_seq) - pattern_len * min_repeat + 1):
                pattern = tuple(block_seq[start:start + pattern_len])
                # Count consecutive repetitions
                repeats = 1
                pos = start + pattern_len
                while pos + pattern_len <= len(block_seq):
                    if tuple(block_seq[pos:pos + pattern_len]) == pattern:
                        repeats += 1
                        pos += pattern_len
                    else:
                        break
                if repeats >= min_repeat:
                    loop_counts[pattern]["total"] += 1
                    if is_resolved:
                        loop_counts[pattern]["resolved"] += 1
                    else:
                        loop_counts[pattern]["failed"] += 1

    # Build results
    motifs: List[dict] = []
    for pattern, counts in sorted(loop_counts.items(), key=lambda x: -x[1]["total"]):
        total = counts["total"]
        motifs.append({
            "pattern": list(pattern),
            "pattern_length": len(pattern),
            "occurrences": total,
            "n_resolved": counts["resolved"],
            "n_failed": counts["failed"],
            "fail_rate": counts["failed"] / max(1, total),
        })

    return motifs


# ═══════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════

def basin_summary_statistics(
    basins: List[dict],
    gates: List[dict],
    run_resolved: Dict[int, bool],
) -> dict:
    """Aggregate statistics for failure basins and recovery gates.

    Reports tiered basin classification:
    - failure_dense: all detected basins (high fail rate connected regions)
    - sticky (escape <= 0.3): subset with low temporal escape probability
    - very_sticky (escape <= 0.15): strictest subset

    Distinguishes:
    - visit_weighted_fail_rate: block-level fail purity (how many run-block
      visits are failed)
    - trajectory_level_fail_rate: P(failed | trajectory enters basin)
    """
    total_runs = len(run_resolved)
    total_failed = sum(1 for r in run_resolved.values() if not r)

    if not basins:
        return {
            "n_basins": 0,
            "n_failure_dense": 0,
            "n_sticky_basins": 0,
            "n_very_sticky_basins": 0,
            "visit_weighted_fail_rate": math.nan,
            "trajectory_level_fail_rate": math.nan,
            "mean_escape_prob": math.nan,
            "pct_failed_in_basins": math.nan,
            "n_gates": len(gates),
            "mean_uplift": math.nan,
            "n_gates_with_positive_uplift": 0,
            "n_significant_gates_with_support": 0,
        }

    fail_rates = [b["fail_rate"] for b in basins]
    escape_probs = [b["escape_prob"] for b in basins]
    basin_failed = sum(b["n_failed"] for b in basins)

    # Tiered classification
    n_sticky = sum(1 for b in basins if b["escape_prob"] <= 0.3)
    n_very_sticky = sum(1 for b in basins if b["escape_prob"] <= 0.15)

    # Trajectory-level fail rate: P(failed | enters basin)
    traj_fail_rates = []
    for b in basins:
        n_traj = b.get("n_trajectories", 0)
        n_fail = b.get("n_failed", 0)
        if n_traj > 0:
            traj_fail_rates.append(n_fail / n_traj)

    uplifts = [g["success_uplift"] for g in gates] if gates else []
    n_positive_uplift = sum(1 for u in uplifts if u > 0)

    # Significant gates with sufficient support (n_visits_after_basin >= 3)
    n_sig_with_support = sum(
        1 for g in gates
        if g.get("success_uplift", 0) >= 0.1
        and g.get("n_visits_after_basin", 0) >= 3
    )

    return {
        "n_basins": len(basins),
        "n_failure_dense": len(basins),
        "n_sticky_basins": n_sticky,
        "n_very_sticky_basins": n_very_sticky,
        "visit_weighted_fail_rate": round(float(np.mean(fail_rates)), 4),
        "trajectory_level_fail_rate": round(
            float(np.mean(traj_fail_rates)), 4
        ) if traj_fail_rates else math.nan,
        "mean_escape_prob": round(float(np.mean(escape_probs)), 4),
        "pct_failed_in_basins": round(
            basin_failed / max(1, total_failed), 4
        ),
        "n_gates": len(gates),
        "mean_uplift": (
            round(float(np.mean(uplifts)), 4) if uplifts else math.nan
        ),
        "n_gates_with_positive_uplift": n_positive_uplift,
        "n_significant_gates_with_support": n_sig_with_support,
    }
