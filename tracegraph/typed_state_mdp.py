"""Typed-state transition kernel behind the signature-ablation statistic.

Each retained BCC block visited by a run is mapped to a typed state
``role|phase|core_tag`` where:
  - role ∈ {common_setup, decision_point, intermediate, weak_basin, success_outcome}
  - phase ∈ {early, mid, late}  (based on normalised temporal position)
  - core_tag ∈ {core, outer}     (based on the reward-field core mask)

A Laplace-smoothed transition kernel with absorbing resolved/failed end
states is estimated per model, and the committor (probability of reaching
the resolved end state first) averaged over decision-point states gives the
per-model value compared across signature conditions in
``scripts/analysis/signature_ablation.py``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import EOS_RESOLVED, EOS_FAILED


# ── Typed-state construction ─────────────────────────────────────────

def phase_of(progress: float) -> str:
    """Map normalised progress ∈ [0, 1] to a phase label."""
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "mid"
    return "late"


def typed_code(role: str, phase: str, core_tag: str) -> str:
    """Encode a typed state as ``role|phase|core_tag``."""
    return f"{role}|{phase}|{core_tag}"


def build_typed_sequences(
    run_sequences: Dict[int, List[dict]],
    block_meta: Dict[int, dict],
    core_mask: np.ndarray,
    node_to_idx: Dict[int, int],
) -> Dict[int, List[str]]:
    """Per-run compact typed-code sequence (consecutive duplicates removed)."""
    per_run: Dict[int, List[str]] = {}
    for rid, seq in run_sequences.items():
        typed: List[str] = []
        for step in seq:
            bid = step.get("primary_block")
            if bid is None or int(bid) not in block_meta:
                continue
            role = str(block_meta[int(bid)].get("block_type", "intermediate"))
            phase = phase_of(float(step.get("progress", 0.0)))
            idx = node_to_idx.get(int(bid))
            core_tag = (
                "core"
                if (idx is not None and 0 <= idx < len(core_mask)
                    and bool(core_mask[idx]))
                else "outer"
            )
            code = typed_code(role, phase, core_tag)
            if not typed or typed[-1] != code:
                typed.append(code)
        if typed:
            per_run[int(rid)] = typed
    return per_run


# ── Kernel estimation (Laplace smoothed) ─────────────────────────────

def build_kernel(
    run_typed: Dict[int, List[str]],
    run_ids: Sequence[int],
    run_resolved: Dict[int, bool],
    all_states: Sequence[str],
    alpha_smoothing: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Build Laplace-smoothed transition kernel P over ``all_states``.

    Trajectories are ``typed_seq + [EOS_*]`` (EOS picked from
    ``run_resolved``).  EOS states are forced absorbing after smoothing:
    ``P[EOS, :] = 0; P[EOS, EOS] = 1``.

    Returns (P, visit_counts, state_idx).
    """
    n = len(all_states)
    state_idx = {s: i for i, s in enumerate(all_states)}
    counts = np.zeros((n, n), dtype=float)
    visits = np.zeros(n, dtype=float)
    for rid in run_ids:
        typed_seq = run_typed.get(int(rid), [])
        if not typed_seq:
            continue
        eos = EOS_RESOLVED if run_resolved.get(int(rid), False) else EOS_FAILED
        full = list(typed_seq) + [eos]
        for t in range(len(full) - 1):
            si, sj = full[t], full[t + 1]
            if si in state_idx and sj in state_idx:
                counts[state_idx[si], state_idx[sj]] += 1.0
        for s in full:
            if s in state_idx:
                visits[state_idx[s]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    P = (counts + alpha_smoothing) / (row_sums + alpha_smoothing * n)
    # Force absorbing sinks AFTER smoothing
    for eos in (EOS_RESOLVED, EOS_FAILED):
        if eos in state_idx:
            ei = state_idx[eos]
            P[ei, :] = 0.0
            P[ei, ei] = 1.0
    return P, visits, state_idx


# ── Committor  q(i) = Pr(hit EOS_resolved before EOS_failed | start i) ──

def compute_committor(
    P: np.ndarray,
    state_idx: Dict[str, int],
    all_states: Sequence[str],
) -> Optional[np.ndarray]:
    """Solve  (I − Q) q = r_A  on transients T (everything except EOS_*)."""
    if EOS_RESOLVED not in state_idx or EOS_FAILED not in state_idx:
        return None
    absorbing = {state_idx[EOS_RESOLVED], state_idx[EOS_FAILED]}
    T_idx = np.array(
        [state_idx[s] for s in all_states if state_idx[s] not in absorbing],
        dtype=int,
    )
    if T_idx.size == 0:
        return None
    Q = P[np.ix_(T_idx, T_idx)]
    r_A = P[T_idx, state_idx[EOS_RESOLVED]]
    try:
        q_T = np.linalg.solve(np.eye(len(T_idx)) - Q, r_A)
    except np.linalg.LinAlgError:
        return None
    q = np.full(P.shape[0], np.nan, dtype=float)
    q[T_idx] = q_T
    q[state_idx[EOS_RESOLVED]] = 1.0
    q[state_idx[EOS_FAILED]] = 0.0
    return q
