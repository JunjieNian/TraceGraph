"""TraceGraph construction: Mutual-kNN graph + BCC decomposition & role assignment.

Adapted from SliceGraph §3.1–3.2 for SWE agent trajectory analysis.
Transforms pre-computed kNN arrays (from IDF-weighted Jaccard distances on
symbolic action-observation key sets) into block-level analysis objects.

Pipeline:
    kNN arrays → mutual-kNN edges → undirected graph → BCC decomposition
    → block statistics → role assignment (5 roles) → block-cut tree
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Mutual-kNN graph construction
# ═══════════════════════════════════════════════════════════════════════

def build_mutual_knn_edges(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    neighbor_k: int = 6,
    dist_scale: float = 0.35,
    slice_run: Optional[np.ndarray] = None,
    exclude_same_run: bool = False,
) -> List[Dict[str, Any]]:
    """Build mutual-kNN edge list from pre-computed kNN arrays.

    An edge (i, j) exists iff i is in j's top-k AND j is in i's top-k.
    Edge weight = exp(-dist / dist_scale).

    Args:
        knn_indices: (n_slices, K) array of neighbor indices.
        knn_dists:   (n_slices, K) array of Jaccard distances.
        neighbor_k:  number of neighbors to consider.
        dist_scale:  RBF scale for edge weight.
        slice_run:   optional array mapping node index → run_id.
        exclude_same_run: if True and slice_run is given, drop edges
                          where both endpoints belong to the same run.

    Returns:
        List of edge dicts with {source, target, weight, distance}.
    """
    n_slices = knn_indices.shape[0]
    k = min(neighbor_k, knn_indices.shape[1])

    # Build forward neighbor sets
    forward: List[set] = [set() for _ in range(n_slices)]
    dist_map: Dict[Tuple[int, int], float] = {}
    for i in range(n_slices):
        for col in range(k):
            j = int(knn_indices[i, col])
            if 0 <= j < n_slices and j != i:
                forward[i].add(j)
                dist_map[(i, j)] = float(knn_dists[i, col])

    # Mutual filtering: keep edge only if both directions present
    edges = []
    seen = set()
    for i in range(n_slices):
        for j in forward[i]:
            if j in forward[i] and i in forward[j]:
                edge_key = (min(i, j), max(i, j))
                if edge_key not in seen:
                    seen.add(edge_key)
                    # Optionally exclude same-run edges
                    if exclude_same_run and slice_run is not None:
                        if slice_run[edge_key[0]] == slice_run[edge_key[1]]:
                            continue
                    d = min(dist_map.get((i, j), 1.0), dist_map.get((j, i), 1.0))
                    w = np.exp(-d / max(dist_scale, 1e-6))
                    edges.append({
                        "source": edge_key[0],
                        "target": edge_key[1],
                        "weight": round(float(w), 6),
                        "distance": round(d, 6),
                    })
    return edges


def build_adjacency_list(edges: List[Dict[str, Any]], n_nodes: int) -> List[List[int]]:
    """Build undirected adjacency list from edge list."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for e in edges:
        u, v = e["source"], e["target"]
        adj[u].append(v)
        adj[v].append(u)
    return adj


# ═══════════════════════════════════════════════════════════════════════
# Biconnected component (BCC) decomposition
# ═══════════════════════════════════════════════════════════════════════

def find_articulation_points(adj: Sequence[Sequence[int]]) -> List[int]:
    """Find articulation points (cut vertices) via iterative Tarjan's algorithm."""
    n = len(adj)
    disc = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    is_ap = np.zeros(n, dtype=bool)
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
                v = int(adj[u][idx])
                if v == u or v < 0 or v >= n:
                    continue
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
                    p = int(parent[u])
                    low[p] = min(low[p], low[u])
                    if p != root and low[u] >= disc[p]:
                        is_ap[p] = True
        if child_count > 1:
            is_ap[root] = True

    return sorted(np.nonzero(is_ap)[0].tolist())


def find_biconnected_components(adj: Sequence[Sequence[int]]) -> List[frozenset]:
    """Find BCCs via iterative Tarjan with edge stack.

    Returns list of frozensets of node indices. Articulation points may appear
    in multiple BCCs. Isolated nodes appear in none. K2 bridges (|BCC|=2) are
    marked as trivial by downstream code.
    """
    n = len(adj)
    if n == 0:
        return []

    disc = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    timer = 0
    edge_stack: List[Tuple[int, int]] = []
    bccs: List[frozenset] = []

    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        dfs_stack = [(root, 0)]

        while dfs_stack:
            u, it = dfs_stack[-1]
            advanced = False
            while it < len(adj[u]):
                v = int(adj[u][it])
                it += 1
                if v == u or v < 0 or v >= n:
                    continue
                if disc[v] == -1:
                    parent[v] = u
                    edge_stack.append((u, v))
                    disc[v] = low[v] = timer
                    timer += 1
                    dfs_stack[-1] = (u, it)
                    dfs_stack.append((v, 0))
                    advanced = True
                    break
                elif v != parent[u]:
                    if disc[v] < disc[u]:
                        edge_stack.append((u, v))
                    low[u] = min(low[u], disc[v])

            if not advanced:
                dfs_stack.pop()
                if dfs_stack:
                    p = int(parent[u])
                    low[p] = min(low[p], low[u])
                    if low[u] >= disc[p]:
                        bcc_nodes: set = set()
                        while edge_stack:
                            e = edge_stack.pop()
                            bcc_nodes.add(e[0])
                            bcc_nodes.add(e[1])
                            if e == (p, u):
                                break
                        if bcc_nodes:
                            bccs.append(frozenset(bcc_nodes))
    return bccs


# ═══════════════════════════════════════════════════════════════════════
# Block-cut tree construction
# ═══════════════════════════════════════════════════════════════════════

def build_block_cut_tree(
    bccs: List[frozenset],
    articulation_points: List[int],
) -> Dict[str, Any]:
    """Build the block-cut tree from BCC blocks and articulation points.

    The block-cut tree is a bipartite tree whose nodes are either
    *block nodes* (one per BCC) or *cut nodes* (one per articulation point).
    An edge connects a cut node to a block node iff the articulation point
    belongs to that BCC.

    Args:
        bccs: list of frozensets of node indices (each frozenset is a BCC).
        articulation_points: list of articulation-point node indices.

    Returns:
        Dict with:
            nodes  — list of {id, type, block_id | node_index, size}
            edges  — list of {source, target}
    """
    ap_set = set(articulation_points)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []

    # Block nodes
    for bid, bcc_nodes in enumerate(bccs):
        nodes.append({
            "id": f"block-{bid}",
            "type": "block",
            "block_id": bid,
            "size": len(bcc_nodes),
        })

    # Cut nodes and edges
    for ap in sorted(ap_set):
        cut_id = f"cut-{ap}"
        nodes.append({
            "id": cut_id,
            "type": "cut",
            "node_index": ap,
        })
        # Connect to every BCC that contains this articulation point
        for bid, bcc_nodes in enumerate(bccs):
            if ap in bcc_nodes:
                edges.append({
                    "source": cut_id,
                    "target": f"block-{bid}",
                })

    return {"nodes": nodes, "edges": edges}


# ═══════════════════════════════════════════════════════════════════════
# Block role assignment (5 roles, agent-adapted)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RoleThresholdConfig:
    """Thresholds for the 5-role classification (agent-adapted).

    Roles:
        common_setup    — high coverage, early position, large (≈ shared_trunk)
        success_outcome — late, high resolved purity, many runs (≈ answer_basin)
        weak_basin      — late, large, but low purity
        decision_point  — small, at articulation point
        intermediate    — everything else
    """
    setup_min_coverage: float = 0.4       # min run coverage for common_setup
    setup_min_size: int = 6               # min nodes
    setup_pos_quantile: float = 0.5       # must be in early half
    outcome_min_size: int = 6             # min nodes for success_outcome
    outcome_min_purity: float = 0.6       # min resolved purity
    outcome_min_unique_runs: int = 3      # min distinct runs
    outcome_pos_quantile: float = 0.5     # must be in late half
    decision_max_size: int = 5            # max nodes for decision_point
    decision_require_articulation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_block_role(
    block: Dict[str, Any],
    role_config: RoleThresholdConfig,
    setup_pos_cutoff: float,
    outcome_pos_cutoff: float,
) -> str:
    """Classify a single BCC block into one of 5 roles.

    Roles:
        common_setup     — high coverage, early position, large
        success_outcome  — late, high resolved purity, many runs
        weak_basin       — late, large, but low purity
        decision_point   — small, at articulation point
        intermediate     — everything else
    """
    coverage = float(block.get("response_coverage", 0))
    n_nodes = int(block.get("n_nodes", 0))
    median_pos = float(block.get("temporal_span", 0))
    purity = float(block.get("resolved_purity", 0))
    n_runs = int(block.get("n_unique_runs", 0))
    has_ap = bool(block.get("has_articulation", False))
    cfg = role_config

    if (coverage > cfg.setup_min_coverage
            and n_nodes >= cfg.setup_min_size
            and median_pos <= setup_pos_cutoff):
        return "common_setup"

    if n_nodes >= cfg.outcome_min_size and median_pos > outcome_pos_cutoff:
        if purity >= cfg.outcome_min_purity and n_runs >= cfg.outcome_min_unique_runs:
            return "success_outcome"
        return "weak_basin"

    if n_nodes <= cfg.decision_max_size and (
            not cfg.decision_require_articulation or has_ap):
        return "decision_point"

    return "intermediate"


def assign_block_roles(
    blocks: List[Dict[str, Any]],
    all_positions: Sequence[int],
    role_config: Optional[RoleThresholdConfig] = None,
) -> List[Dict[str, Any]]:
    """Assign roles to all blocks and deduplicate success_outcome blocks.

    Returns blocks with added 'block_type' field.
    """
    cfg = role_config or RoleThresholdConfig()
    positions = np.asarray(all_positions, dtype=float)
    setup_cutoff = float(np.quantile(positions, cfg.setup_pos_quantile)) if len(positions) else 0
    outcome_cutoff = float(np.quantile(positions, cfg.outcome_pos_quantile)) if len(positions) else 0

    for b in blocks:
        b["block_type"] = classify_block_role(b, cfg, setup_cutoff, outcome_cutoff)

    # Deduplicate: only the largest block keeps success_outcome per dominant outcome
    outcome_groups: Dict[str, List[Dict]] = defaultdict(list)
    for b in blocks:
        if b["block_type"] == "success_outcome":
            outcome_groups[str(b.get("dominant_outcome", ""))].append(b)
    for group in outcome_groups.values():
        if len(group) <= 1:
            continue
        largest = max(group, key=lambda x: x["n_nodes"])
        for b in group:
            if b is not largest:
                b["block_type"] = "weak_basin"

    return blocks


# ═══════════════════════════════════════════════════════════════════════
# Full BCC analysis (combines all steps above)
# ═══════════════════════════════════════════════════════════════════════

def compute_bcc_analysis(
    adj: List[List[int]],
    n_nodes: int,
    slice_run: np.ndarray,
    slice_pos: np.ndarray,
    is_resolved: List[bool],
) -> Dict[str, Any]:
    """Full BCC analysis: decompose → compute block stats → assign roles.

    Args:
        adj:          undirected adjacency list (n_nodes).
        n_nodes:      number of nodes in the graph.
        slice_run:    array mapping node index → run_id.
        slice_pos:    array mapping node index → position.
        is_resolved:  per-run resolved status flag.

    Returns:
        Dict with: blocks (list), n_blocks, n_trivial,
                   n_nontrivial, articulation_points, block_cut_tree.
    """
    aps = set(find_articulation_points(adj))
    bccs = find_biconnected_components(adj)
    if not bccs:
        return {
            "blocks": [], "n_blocks": 0, "n_trivial": 0,
            "n_nontrivial": 0, "block_cut_tree": {"nodes": [], "edges": []},
        }

    bccs = sorted(bccs, key=lambda s: -len(s))
    total_runs = len(set(int(slice_run[i]) for i in range(n_nodes)))
    all_positions = [int(slice_pos[i]) for i in range(n_nodes)]

    blocks = []
    for bid, bcc_nodes in enumerate(bccs):
        runs_in = [int(slice_run[i]) for i in bcc_nodes]
        pos_in = np.array([int(slice_pos[i]) for i in bcc_nodes])

        unique_runs = len(set(runs_in))
        coverage = unique_runs / max(1, total_runs)

        # Resolved purity (fraction of run visits that are resolved)
        resolved_count = sum(
            1 for r in runs_in
            if r < len(is_resolved) and is_resolved[r]
        )
        resolved_purity = resolved_count / max(1, len(runs_in))

        # Dominant outcome
        resolved_unique = sum(
            1 for r in set(runs_in)
            if r < len(is_resolved) and is_resolved[r]
        )
        dominant_outcome = (
            "resolved" if resolved_unique > unique_runs / 2 else "failed"
        )

        blocks.append({
            "block_id": bid,
            "n_nodes": len(bcc_nodes),
            "node_indices": sorted(bcc_nodes),
            "is_trivial": len(bcc_nodes) == 2,
            "response_coverage": round(coverage, 4),
            "n_unique_runs": unique_runs,
            "resolved_purity": round(resolved_purity, 4),
            "dominant_outcome": dominant_outcome,
            "correctness_ratio": round(resolved_purity, 4),
            "temporal_span": round(float(np.median(pos_in)), 2),
            "min_pos": int(pos_in.min()),
            "max_pos": int(pos_in.max()),
            "has_articulation": any(i in aps for i in bcc_nodes),
        })

    blocks = assign_block_roles(blocks, all_positions)

    # Build block-cut tree
    block_cut_tree = build_block_cut_tree(bccs, sorted(aps))

    n_trivial = sum(1 for b in blocks if b["is_trivial"])
    return {
        "blocks": blocks,
        "n_blocks": len(blocks),
        "n_trivial": n_trivial,
        "n_nontrivial": len(blocks) - n_trivial,
        "articulation_points": sorted(aps),
        "block_cut_tree": block_cut_tree,
    }


def nontrivial_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to non-trivial blocks (|BCC| >= 3)."""
    return [b for b in blocks if not b.get("is_trivial", False)]
