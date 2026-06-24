"""Reward-field propagation and core-mask extraction.

Adapted from SliceGraph §3.4 for SWE agent trajectory analysis.
Correctness labels (resolved/failed) are mapped to block-level seeds,
then diffused over the block graph via a personalised PageRank-style
iteration.  The high-value core is the set of positive-field blocks
above a quantile threshold.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────

def nontrivial_block_info(payload: dict) -> Dict[int, dict]:
    """Extract metadata for non-trivial BCC blocks from a TraceGraph payload."""
    blocks = payload.get("bcc_analysis", {}).get("blocks", [])
    return {
        int(block["block_id"]): block
        for block in blocks
        if not block.get("is_trivial", False)
    }


def choose_primary_block(
    block_ids: Sequence[int],
    block_meta: Dict[int, dict],
) -> Optional[int]:
    """Among candidate block ids, choose the primary (largest, most-visited)."""
    bids = [int(bid) for bid in block_ids if int(bid) in block_meta]
    if not bids:
        return None
    return max(
        bids,
        key=lambda bid: (
            int(block_meta[bid].get("n_nodes", 0)),
            int(block_meta[bid].get("n_unique_runs", 0)),
            -int(block_meta[bid].get("min_pos", 0)),
            -int(bid),
        ),
    )


# ── Run-level sequence reconstruction ────────────────────────────────

def reconstruct_run_sequences(
    payload: dict,
) -> Dict[int, List[dict]]:
    """Return per-run ordered slice states with attached block ids.

    Each step is a dict with keys: pos, local_idx, progress, block_ids,
    primary_block.

    Adapted for TraceGraph payload format: uses 'trace_info' instead of
    SliceGraph's 'slice_info' from the problem dict.
    """
    rtc = payload.get("role_threshold_cache", {})
    selected = list(rtc.get("selected_slices", []))
    slice_runs = list(rtc.get("slice_runs", []))
    slice_positions = list(rtc.get("slice_positions", []))
    node_bcc_map = rtc.get("node_bcc_map", {})

    block_meta = nontrivial_block_info(payload)
    if not selected or not block_meta:
        return {}

    per_run: Dict[int, List[Tuple[int, int, Tuple[int, ...]]]] = defaultdict(list)
    for local_idx in range(len(selected)):
        if local_idx >= len(slice_runs):
            continue
        rid = int(slice_runs[local_idx])
        pos = int(slice_positions[local_idx]) if local_idx < len(slice_positions) else local_idx
        raw_bids = node_bcc_map.get(local_idx, node_bcc_map.get(str(local_idx), []))
        bids = sorted({int(bid) for bid in raw_bids if int(bid) in block_meta})
        if not bids:
            continue
        per_run[rid].append((pos, local_idx, tuple(bids)))

    sequences: Dict[int, List[dict]] = {}
    for rid, rows in per_run.items():
        rows.sort(key=lambda item: (item[0], item[1]))
        denom = max(1, len(rows) - 1)
        seq: List[dict] = []
        for idx, (pos, local_idx, bids) in enumerate(rows):
            seq.append({
                "pos": int(pos),
                "local_idx": int(local_idx),
                "progress": float(idx / denom),
                "block_ids": bids,
                "primary_block": choose_primary_block(bids, block_meta),
            })
        sequences[rid] = seq
    return sequences


# ── Resolved status and visit sets ──────────────────────────────────

def build_run_resolved(payload: dict) -> Dict[int, bool]:
    """Extract {run_id: is_resolved} from a TraceGraph payload."""
    trajectories = payload.get("trajectories", [])
    result: Dict[int, bool] = {}
    for trajectory in trajectories:
        rid = trajectory.get("run_id")
        if rid is None:
            continue
        result[int(rid)] = bool(trajectory.get("is_resolved", False))
    return result


def build_visit_sets(
    run_sequences: Dict[int, List[dict]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Return (run_block_sets, block_run_sets) from run sequences."""
    run_block_sets: Dict[int, Set[int]] = {}
    block_run_sets: Dict[int, Set[int]] = defaultdict(set)
    for rid, seq in run_sequences.items():
        bids = {
            int(bid)
            for step in seq
            for bid in step.get("block_ids", ())
        }
        run_block_sets[rid] = bids
        for bid in bids:
            block_run_sets[bid].add(int(rid))
    return run_block_sets, dict(block_run_sets)


# ── Block graph ──────────────────────────────────────────────────────

def build_block_graph(
    payload: dict,
    run_sequences: Dict[int, List[dict]],
) -> Dict[int, Set[int]]:
    """Build an undirected adjacency dict over non-trivial blocks.

    Edges come from two sources:
    1. The block-cut tree (blocks sharing a cut vertex).
    2. Temporal transitions observed in run sequences (compact path:
       consecutive duplicate blocks removed).
    """
    block_meta = nontrivial_block_info(payload)
    adj: Dict[int, Set[int]] = {bid: set() for bid in block_meta}

    # Edges from block-cut tree
    tree = payload.get("bcc_analysis", {}).get("block_cut_tree", {})
    cut_to_blocks: Dict[str, Set[int]] = defaultdict(set)
    for edge in tree.get("edges", []):
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if src.startswith("cut-") and tgt.startswith("block-"):
            bid = int(tgt.split("-", 1)[1])
            if bid in adj:
                cut_to_blocks[src].add(bid)
        elif tgt.startswith("cut-") and src.startswith("block-"):
            bid = int(src.split("-", 1)[1])
            if bid in adj:
                cut_to_blocks[tgt].add(bid)
    for bids in cut_to_blocks.values():
        for a, b in combinations(sorted(bids), 2):
            adj[a].add(b)
            adj[b].add(a)

    # Edges from temporal transitions
    for seq in run_sequences.values():
        compact_path: List[int] = []
        for step in seq:
            primary = step.get("primary_block")
            if primary is None:
                continue
            if not compact_path or compact_path[-1] != int(primary):
                compact_path.append(int(primary))
        for a, b in zip(compact_path[:-1], compact_path[1:]):
            if a == b:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

    return adj


# ── Transition matrix ────────────────────────────────────────────────

def build_transition_matrix(
    nodes: Sequence[int],
    adjacency: Dict[int, Set[int]],
    *,
    self_loop_weight: float = 1.0,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Row-normalised transition matrix over block nodes.

    Returns (P, node_to_idx).
    """
    node_to_idx = {int(bid): idx for idx, bid in enumerate(nodes)}
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=float)
    for bid in nodes:
        i = node_to_idx[int(bid)]
        matrix[i, i] += self_loop_weight
        for nbr in adjacency.get(int(bid), set()):
            j = node_to_idx.get(int(nbr))
            if j is None:
                continue
            matrix[i, j] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    matrix = matrix / row_sums
    return matrix, node_to_idx


# ── Seed vector ──────────────────────────────────────────────────────

def support_shrinkage(n_visit: int, n_total_runs: int, exponent: float) -> float:
    """Support-shrinkage factor  (n_visit / n_total)^β."""
    if n_total_runs <= 0:
        return 0.0
    frac = max(0.0, min(1.0, n_visit / n_total_runs))
    return float(frac ** exponent)


def compute_seed_vector(
    *,
    run_id: Optional[int],
    nodes: Sequence[int],
    block_run_sets: Dict[int, Set[int]],
    run_resolved: Dict[int, bool],
    min_run_support: int,
    support_shrink_exp: float,
) -> np.ndarray:
    """Compute global or leave-one-out (LOO) block reward seeds.

    When ``run_id`` is None, the global seed is computed.  Otherwise the
    label of ``run_id`` is excluded (LOO), so the reward field used to
    score a run never includes that run's own label.
    """
    all_runs = sorted(run_resolved)
    n_runs = len(all_runs)
    if n_runs == 0:
        return np.zeros(len(nodes), dtype=float)

    y_excluded = 0
    denom_runs = n_runs
    if run_id is not None and run_id in run_resolved:
        y_excluded = 1 if run_resolved[run_id] else 0
        denom_runs = max(1, n_runs - 1)

    n_resolved_total = sum(1 for rid in all_runs if run_resolved[rid])
    p_base = (n_resolved_total - y_excluded) / denom_runs

    seed = np.zeros(len(nodes), dtype=float)
    for idx, bid in enumerate(nodes):
        visitors = set(block_run_sets.get(int(bid), set()))
        n_visit = len(visitors)
        if n_visit < min_run_support or n_visit > n_runs - min_run_support:
            seed[idx] = 0.0
            continue
        resolved_visitors = sum(
            1 for rid in visitors if run_resolved.get(int(rid), False)
        )
        shrink = support_shrinkage(n_visit, n_runs, support_shrink_exp)
        if run_id is not None and run_id in visitors:
            denom = n_visit - 1
            if denom <= 0:
                seed[idx] = 0.0
                continue
            ratio = (resolved_visitors - y_excluded) / denom
        else:
            ratio = resolved_visitors / n_visit
        seed[idx] = (ratio - p_base) * shrink
    return seed


# ── Field diffusion ──────────────────────────────────────────────────

def diffuse_reward_field(
    seed: np.ndarray,
    transition_matrix: np.ndarray,
    *,
    alpha: float,
    n_steps: int,
) -> np.ndarray:
    """Personalised PageRank-style diffusion of the seed vector.

        f^{(t+1)} = α · seed  +  (1 − α) · P · f^{(t)}

    The result is L∞-normalised to [−1, 1].
    """
    if seed.size == 0:
        return seed.copy()
    field = seed.astype(float, copy=True)
    for _ in range(max(1, int(n_steps))):
        field = alpha * seed + (1.0 - alpha) * (transition_matrix @ field)
    scale = float(np.max(np.abs(field)))
    if scale > 1e-12:
        field = field / scale
    return field


# ── Core mask ────────────────────────────────────────────────────────

def make_core_mask(field: np.ndarray, positive_quantile: float) -> np.ndarray:
    """Boolean mask selecting high-value blocks above the quantile threshold.

    Only positive-field blocks are candidates; the threshold is the
    ``positive_quantile``-th quantile of positive entries.
    """
    if field.size == 0:
        return np.zeros(0, dtype=bool)
    positive = field[field > 0]
    if positive.size == 0:
        return np.zeros_like(field, dtype=bool)
    if positive.size <= 3:
        return field > 0
    threshold = float(np.quantile(positive, positive_quantile))
    return np.logical_and(field > 0, field >= threshold)
