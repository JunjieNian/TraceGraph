"""Typed-state transition encoding and mechanism metrics.

Adapted from SliceGraph §3.5 for SWE agent trajectory analysis.
Each non-trivial BCC block is mapped to a typed state
``(role, phase, core_tag)`` where:
  - role ∈ {common_setup, decision_point, intermediate, weak_basin, success_outcome}
  - phase ∈ {early, mid, late}  (based on normalised temporal position)
  - core_tag ∈ {core, outer}     (based on reward-field core mask)

Transition kernels are estimated under Laplace smoothing, and four
mechanism metrics are computed for resolved-only vs failed-only kernels:
  1. Committor  q(i)  at decision-point states
  2. MFPT  (decision → core)
  3. 3-step core escape hazard
  4. 3-step post-escape return probability

Family-level TV distance measures how distinct family transition policies are.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


# ── MFPT to core ────────────────────────────────────────────────────

def compute_mfpt_to_core(
    P: np.ndarray,
    state_idx: Dict[str, int],
    all_states: Sequence[str],
    core_states: Sequence[str],
) -> Optional[np.ndarray]:
    """Expected first-passage time to core-tagged absorbing set.

    Absorbing = core_states + EOS_resolved + EOS_failed.
    Solve  (I − Q') μ = 1  on transients.
    """
    if not core_states:
        return None
    absorbing = {state_idx[s] for s in core_states if s in state_idx}
    absorbing |= {state_idx[EOS_RESOLVED], state_idx[EOS_FAILED]}
    T_idx = np.array(
        [state_idx[s] for s in all_states if state_idx[s] not in absorbing],
        dtype=int,
    )
    if T_idx.size == 0:
        return None
    Q = P[np.ix_(T_idx, T_idx)]
    try:
        mu_T = np.linalg.solve(np.eye(len(T_idx)) - Q, np.ones(len(T_idx)))
    except np.linalg.LinAlgError:
        return None
    mu = np.full(P.shape[0], np.nan, dtype=float)
    mu[T_idx] = mu_T
    for s in core_states:
        if s in state_idx:
            mu[state_idx[s]] = 0.0
    return mu


# ── 3-step core escape hazard ───────────────────────────────────────

def compute_escape_hazard_3step(
    P: np.ndarray,
    visits: np.ndarray,
    state_idx: Dict[str, int],
    core_states: Sequence[str],
    non_core_states: Sequence[str],
) -> float:
    """π̂-weighted probability of transitioning from core to outer in 3 steps."""
    if not core_states or not non_core_states:
        return math.nan
    core_idx = np.array(
        [state_idx[s] for s in core_states if s in state_idx], dtype=int,
    )
    outer_idx = np.array(
        [state_idx[s] for s in non_core_states if s in state_idx], dtype=int,
    )
    if core_idx.size == 0 or outer_idx.size == 0:
        return math.nan
    total_core_visits = float(visits[core_idx].sum())
    if total_core_visits <= 1e-12:
        return math.nan
    pi_hat = visits[core_idx] / total_core_visits
    P3 = P @ P @ P
    escape_rows = P3[np.ix_(core_idx, outer_idx)].sum(axis=1)
    return float(np.sum(pi_hat * escape_rows))


# ── 3-step post-escape return probability ────────────────────────────

def compute_return_prob_3step(
    run_typed: Dict[int, List[str]],
    run_ids: Sequence[int],
    run_resolved: Dict[int, bool],
    P: np.ndarray,
    state_idx: Dict[str, int],
    core_states: Sequence[str],
) -> float:
    """At each observed escape (core → outer), average P^3[j, core_set]."""
    if not core_states:
        return math.nan
    core_set = set(core_states)
    core_idx = np.array(
        [state_idx[s] for s in core_states if s in state_idx], dtype=int,
    )
    if core_idx.size == 0:
        return math.nan
    P3 = P @ P @ P
    returns: List[float] = []
    for rid in run_ids:
        seq = run_typed.get(int(rid), [])
        if not seq:
            continue
        eos = EOS_RESOLVED if run_resolved.get(int(rid), False) else EOS_FAILED
        full = list(seq) + [eos]
        for t in range(len(full) - 1):
            si, sj = full[t], full[t + 1]
            if si in core_set and sj not in core_set and sj in state_idx:
                returns.append(float(P3[state_idx[sj], core_idx].sum()))
    if not returns:
        return math.nan
    return float(np.mean(returns))


# ── Successor representation ─────────────────────────────────────────

def compute_successor_repr(
    P: np.ndarray,
    gamma: float,
) -> Optional[np.ndarray]:
    """M = (I − γ P)⁻¹.  Returns None if singular."""
    n = P.shape[0]
    try:
        M = np.linalg.solve(np.eye(n) - gamma * P, np.eye(n))
    except np.linalg.LinAlgError:
        return None
    return M


# ── Family-TV distance ──────────────────────────────────────────────

def compute_family_tv(
    family_kernels: Dict[int, Tuple[np.ndarray, Dict[str, int]]],
) -> Dict[int, float]:
    """Mean pairwise total-variation between family transition kernels.

    TV(P_i, P_j) = 0.5 · mean_s Σ_s' |P_i(s,s') − P_j(s,s')|.
    Returns {family_id: mean TV to other families}.
    """
    fam_ids = sorted(family_kernels.keys())
    tv_by_fam: Dict[int, List[float]] = defaultdict(list)
    for i_idx, fi in enumerate(fam_ids):
        for j_idx, fj in enumerate(fam_ids):
            if j_idx <= i_idx:
                continue
            P_i, _ = family_kernels[fi]
            P_j, _ = family_kernels[fj]
            tv = 0.5 * float(np.mean(np.abs(P_i - P_j).sum(axis=1)))
            tv_by_fam[fi].append(tv)
            tv_by_fam[fj].append(tv)
    result: Dict[int, float] = {}
    for fid in fam_ids:
        tvs = tv_by_fam.get(fid, [])
        result[fid] = float(np.mean(tvs)) if tvs else math.nan
    return result


# ── Cohen's d (for effect-size aggregation) ──────────────────────────

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled Cohen's d  = (mean(x) − mean(y)) / s_pooled."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    return float((np.mean(x) - np.mean(y)) / max(sp, 1e-12))
