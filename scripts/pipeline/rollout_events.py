#!/usr/bin/env python3
"""Per-rollout four-event process profiles for cx-cmu agent trajectories.

Reads enriched graph payloads (data/cxcmu/graphs/<bench>/<task>.pkl) and the
parsed per-rollout rows (data/cxcmu/parsed/<bench>/<task>.jsonl), and emits
per-rollout records with the four process events used in the analysis-paper
main text:

    A_r  = 1[exists step with primary_block in C_t]           (Access)
    E_r  = 1[exists step with primary_block in B_t]           (Trap exposure)
    R_r  = 1[exists s<u with seq[s] in B_t and seq[u] in C_t] (Repair after trap)
    G_r  = mean_{g in V_r ∩ G_t} q_t(g)                       (Gate resolution)

with B_t the bottom-quartile mask of the diffused outcome-normalized reward
field (note: NOT the dual-threshold failure_basins; trap and repair are thus
not circular), C_t the existing top-quartile core mask, G_t the set of blocks
flagged has_articulation=True, and the empirical gate-outcome map

    q_t(g) = (sum_{r: g in V_r} y_r + 0.5) / (|{r: g in V_r}| + 1.0)

with Laplace smoothing. Rollouts without any visited gate are excluded from
the G_r denominator (G_r = null in the JSONL row).

Outcomes y_r are read from parsed metadata:
    metadata.resolved_score in {0, 1}        for binary benchmarks
    metadata.resolved_score in [0, 1]        for mcpbench (raw_reward / task_max)

Outputs:
    results/cxcmu/rollout_events.jsonl
        one row per rollout: {benchmark, task_id, model_id, run_id, y_r,
                              A_r, E_r, R_r, G_r, visited_gate}
    results/cxcmu/rollout_events_supply_demand.json
        per-model supply S_{m,a} and per-benchmark demand D_{b,a},
        plus the model and benchmark lists used.

The script does not modify any payloads on disk.

Usage:
    python scripts/100_rollout_events.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from tracegraph.reward_field import (
    build_run_resolved,
    nontrivial_block_info,
    reconstruct_run_sequences,
)

GRAPH_DIR = Path("data/cxcmu/graphs")
PARSED_DIR = Path("data/cxcmu/parsed")
RESULTS_DIR = Path("results/cxcmu")
EVENTS_PATH = RESULTS_DIR / "rollout_events.jsonl"
SD_PATH = RESULTS_DIR / "rollout_events_supply_demand.json"


def _set_data_root(data_root: Optional[Path], results_root: Optional[Path]) -> None:
    """Re-point GRAPH_DIR, PARSED_DIR, RESULTS_DIR and the two output paths."""
    global GRAPH_DIR, PARSED_DIR, RESULTS_DIR, EVENTS_PATH, SD_PATH
    if data_root is not None:
        data_root = Path(data_root)
        GRAPH_DIR = data_root / "graphs"
        PARSED_DIR = data_root / "parsed"
    if results_root is not None:
        RESULTS_DIR = Path(results_root)
    EVENTS_PATH = RESULTS_DIR / "rollout_events.jsonl"
    SD_PATH = RESULTS_DIR / "rollout_events_supply_demand.json"

TRAP_QUANTILE = 0.25  # bottom quartile of the negative field


# ── Helpers ─────────────────────────────────────────────────────────

def load_parsed_outcomes(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {run_str: {model_id, resolved_score, raw_reward}} per row.

    run_str follows the {model_id}__pass{pass_tag} convention used in
    unique_runs/run_id_map of the graph payloads.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            model_id = row.get("model_id")
            scaffold = row.get("scaffold_id")
            run_id_str = row.get("run_id")  # parsed row's own run_id (string)
            md = row.get("metadata", {}) or {}
            pass_tag = md.get("pass")
            if pass_tag is None and run_id_str is not None:
                # Fallback: try to derive from run_id_str if needed.
                pass_tag = run_id_str
            if run_id_str is not None:
                key = run_id_str
            elif pass_tag is not None:
                pass_tag_str = str(pass_tag)
                if pass_tag_str.startswith("pass"):
                    key = f"{model_id}__{pass_tag_str}"
                else:
                    key = f"{model_id}__pass{pass_tag_str}"
            else:
                key = None
            score = md.get("resolved_score")
            if score is None:
                # Last-resort fallback to binary resolved flag.
                score = 1.0 if row.get("resolved") else 0.0
            entry = {
                "model_id": model_id,
                "scaffold_id": scaffold,
                "resolved_score": float(score),
                "raw_reward": md.get("raw_reward"),
                "task_max_reward": md.get("task_max_reward"),
                "run_id_str": run_id_str,
            }
            if key is not None:
                out[key] = entry
    return out


def compute_trap_mask(field: np.ndarray, quantile: float = TRAP_QUANTILE) -> np.ndarray:
    """Return boolean mask of blocks in the bottom quantile of the negative field.

    Only strictly-negative entries are eligible. The threshold is the
    ``quantile``-th quantile of negative entries (i.e. the most negative
    fraction). For very small populations (≤3 negatives), every negative
    block is flagged as trap.
    """
    if field.size == 0:
        return np.zeros(0, dtype=bool)
    negative = field[field < 0]
    if negative.size == 0:
        return np.zeros_like(field, dtype=bool)
    if negative.size <= 3:
        return field < 0
    threshold = float(np.quantile(negative, quantile))  # most negative side
    return np.logical_and(field < 0, field <= threshold)


def gate_block_set(block_meta: Dict[int, dict]) -> set:
    """Set of block ids with at least one articulation point."""
    return {
        int(bid)
        for bid, meta in block_meta.items()
        if bool(meta.get("has_articulation", False))
    }


def compact_block_sequence(steps: List[dict]) -> List[int]:
    """Collapse a per-step sequence to consecutive-unique primary blocks."""
    out: List[int] = []
    for step in steps:
        prim = step.get("primary_block")
        if prim is None:
            continue
        prim = int(prim)
        if not out or out[-1] != prim:
            out.append(prim)
    return out


# ── Per-task event computation ─────────────────────────────────────

def process_task(
    payload: dict,
    parsed_lookup: Dict[str, Dict[str, Any]],
    benchmark: str,
    task_id: str,
) -> List[Dict[str, Any]]:
    """Compute per-rollout events for a single (benchmark, task) pair."""
    rf = payload.get("reward_field") or {}
    nodes = rf.get("nodes")
    field = rf.get("field")
    core_mask = rf.get("core_mask")
    if not nodes or field is None or core_mask is None:
        return []

    field_arr = np.asarray(field, dtype=float)
    core_arr = np.asarray(core_mask, dtype=bool)
    trap_arr = compute_trap_mask(field_arr, TRAP_QUANTILE)

    node_to_idx = {int(bid): i for i, bid in enumerate(nodes)}
    core_blocks = {int(nodes[i]) for i, c in enumerate(core_arr) if c}
    trap_blocks = {int(nodes[i]) for i, c in enumerate(trap_arr) if c}

    block_meta = nontrivial_block_info(payload)
    gate_blocks = gate_block_set(block_meta)

    run_sequences = reconstruct_run_sequences(payload)
    if not run_sequences:
        return []

    # rid -> {model_id, y_r, run_str, ...}
    unique_runs = payload.get("unique_runs", []) or []
    run_id_map = payload.get("run_id_map", {}) or {}
    rid_to_runstr: Dict[int, str] = {}
    for run_str, rid in run_id_map.items():
        rid_to_runstr[int(rid)] = str(run_str)
    if not rid_to_runstr and unique_runs:
        rid_to_runstr = {i: str(s) for i, s in enumerate(unique_runs)}

    # First pass: per-rollout visit/sequence summary, and y_r.
    rollouts: List[Dict[str, Any]] = []
    run_resolved_fallback = build_run_resolved(payload)
    for rid, steps in run_sequences.items():
        run_str = rid_to_runstr.get(int(rid))
        if run_str is None:
            continue
        meta = parsed_lookup.get(run_str)
        if meta is not None:
            model_id = meta["model_id"]
            y_r = float(meta["resolved_score"])
        else:
            # Fallback: split "{model}__{pass}" and use binary resolved.
            model_id = run_str.split("__", 1)[0]
            y_r = 1.0 if run_resolved_fallback.get(int(rid), False) else 0.0

        compact = compact_block_sequence(steps)
        visit_set = set(compact)

        A_r = 1 if (visit_set & core_blocks) else 0
        E_r = 1 if (visit_set & trap_blocks) else 0

        R_r = 0
        if E_r and (visit_set & core_blocks):
            saw_trap = False
            for b in compact:
                if b in trap_blocks:
                    saw_trap = True
                elif saw_trap and b in core_blocks:
                    R_r = 1
                    break

        visited_gates = sorted(visit_set & gate_blocks)
        rollouts.append({
            "rid": int(rid),
            "run_str": run_str,
            "model_id": model_id,
            "y_r": float(y_r),
            "A_r": int(A_r),
            "E_r": int(E_r),
            "R_r": int(R_r),
            "visited_gates": visited_gates,
        })

    # Second pass: empirical q_t(g) per visited gate, then G_r per rollout.
    gate_num: Dict[int, float] = defaultdict(float)
    gate_den: Dict[int, int] = defaultdict(int)
    for r in rollouts:
        for g in r["visited_gates"]:
            gate_num[int(g)] += float(r["y_r"])
            gate_den[int(g)] += 1

    q_g: Dict[int, float] = {}
    for g, n in gate_den.items():
        q_g[int(g)] = (gate_num[int(g)] + 0.5) / (n + 1.0)

    out_rows: List[Dict[str, Any]] = []
    for r in rollouts:
        visited = r["visited_gates"]
        if visited:
            vals = [q_g[int(g)] for g in visited if int(g) in q_g]
            G_r: Optional[float] = float(np.mean(vals)) if vals else None
            visited_gate = True
        else:
            G_r = None
            visited_gate = False
        out_rows.append({
            "benchmark": benchmark,
            "task_id": task_id,
            "model_id": r["model_id"],
            "run_id": r["rid"],
            "run_str": r["run_str"],
            "y_r": r["y_r"],
            "A_r": r["A_r"],
            "E_r": r["E_r"],
            "R_r": r["R_r"],
            "G_r": G_r,
            "visited_gate": bool(visited_gate),
        })
    return out_rows


# ── Aggregation: supply (model) and demand (benchmark) ─────────────

EVENTS = ["A_r", "E_r", "R_r", "G_r"]


def compute_supply_demand(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute task-centered supply and reward-weighted demand."""
    # Group: (benchmark, task_id, model_id) -> per-event lists.
    cells: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {e: [] for e in EVENTS}
    )
    for r in rows:
        key = (r["benchmark"], r["task_id"], r["model_id"])
        for e in EVENTS:
            v = r.get(e)
            if v is None:
                continue
            cells[key][e].append(float(v))

    # Cell means a_{m,t} per event.
    cell_means: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for key, ev_lists in cells.items():
        cell_means[key] = {
            e: (float(np.mean(vs)) if vs else float("nan")) for e, vs in ev_lists.items()
        }

    # Task means E_{m'}[a_{m',t}] per event (over the models present in that task).
    task_event_sums: Dict[Tuple[str, str, str], Dict[str, Tuple[float, int]]] = {}
    by_task: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    for key in cell_means:
        b, t, m = key
        by_task[(b, t)].append(key)
    task_means: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (b, t), keys in by_task.items():
        task_means[(b, t)] = {}
        for e in EVENTS:
            vals = [
                cell_means[k][e]
                for k in keys
                if not (cell_means[k][e] != cell_means[k][e])  # not NaN
            ]
            task_means[(b, t)][e] = float(np.mean(vals)) if vals else float("nan")

    # Supply: per model, mean over tasks of (a_{m,t} - mean_{m'} a_{m',t}).
    by_model: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for (b, t, m) in cell_means:
        by_model[m].append((b, t))

    supply: Dict[str, Dict[str, float]] = {}
    for m, tlist in by_model.items():
        supply[m] = {}
        for e in EVENTS:
            diffs = []
            for (b, t) in tlist:
                a_mt = cell_means[(b, t, m)][e]
                a_bar = task_means[(b, t)][e]
                if (a_mt != a_mt) or (a_bar != a_bar):
                    continue
                diffs.append(a_mt - a_bar)
            supply[m][e] = float(np.mean(diffs)) if diffs else float("nan")

    # Demand: per benchmark, reward-weighted contrast over rollouts.
    #   D_{b,a} = sum_r w_r a_r / sum_r w_r  -  sum_r (1-w_r) a_r / sum_r (1-w_r)
    # with w_r = y_r in [0,1]. For binary y this reduces to mean(resolved) - mean(failed).
    by_bench_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bench_rows[r["benchmark"]].append(r)

    demand: Dict[str, Dict[str, float]] = {}
    for b, rs in by_bench_rows.items():
        demand[b] = {}
        for e in EVENTS:
            num_w = 0.0
            den_w = 0.0
            num_uw = 0.0
            den_uw = 0.0
            for r in rs:
                v = r.get(e)
                if v is None:
                    continue
                v = float(v)
                w = float(r["y_r"])
                num_w += w * v
                den_w += w
                num_uw += (1.0 - w) * v
                den_uw += (1.0 - w)
            hi = (num_w / den_w) if den_w > 0 else float("nan")
            lo = (num_uw / den_uw) if den_uw > 0 else float("nan")
            demand[b][e] = float(hi - lo)

    return {
        "events": EVENTS,
        "supply": supply,
        "demand": demand,
        "task_means": {
            f"{b}::{t}": vs for (b, t), vs in task_means.items()
        },
    }


# ── Main ───────────────────────────────────────────────────────────

def main(max_tasks: Optional[int] = None, benchmark: Optional[str] = None) -> None:
    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]
    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for bench in benchmarks:
        gdir = GRAPH_DIR / bench
        pdir = PARSED_DIR / bench
        graph_files = sorted(gdir.glob("*.pkl"))
        if max_tasks is not None:
            graph_files = graph_files[:max_tasks]
        print(f"\n── {bench}: {len(graph_files)} tasks ──")

        for gpath in tqdm(graph_files, desc=f"  Rollout events {bench}"):
            task_id = gpath.stem
            try:
                with open(gpath, "rb") as fh:
                    payload = pickle.load(fh)
            except Exception as exc:
                print(f"    skip {bench}/{task_id}: load failed ({exc})")
                continue

            parsed_path = pdir / f"{task_id}.jsonl"
            parsed_lookup = load_parsed_outcomes(parsed_path)

            rows = process_task(payload, parsed_lookup, bench, task_id)
            all_rows.extend(rows)

    # Write per-rollout JSONL.
    with open(EVENTS_PATH, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")

    # Aggregate and write supply/demand.
    sd = compute_supply_demand(all_rows)
    sd["n_rollouts"] = len(all_rows)
    sd["n_with_gate"] = sum(1 for r in all_rows if r.get("visited_gate"))
    sd["benchmarks"] = sorted({r["benchmark"] for r in all_rows})
    sd["models"] = sorted({r["model_id"] for r in all_rows})
    with open(SD_PATH, "w") as fh:
        json.dump(sd, fh, indent=2)

    print(f"\n── Summary ──")
    print(f"Wrote {len(all_rows)} rollout rows → {EVENTS_PATH}")
    print(f"Wrote supply/demand for "
          f"{len(sd['models'])} models × {len(sd['benchmarks'])} benchmarks → {SD_PATH}")
    print(f"Rollouts with a visited gate: {sd['n_with_gate']} / {sd['n_rollouts']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the data root (default: data/cxcmu). "
             "Expects <data_root>/graphs and <data_root>/parsed.",
    )
    parser.add_argument(
        "--results-root", type=str, default=None,
        help="Override the results root (default: results/cxcmu). "
             "External runs MUST set this to avoid overwriting the cxcmu "
             "rollout_events.jsonl global output.",
    )
    args = parser.parse_args()
    if args.data_root is not None or args.results_root is not None:
        _set_data_root(
            Path(args.data_root) if args.data_root else None,
            Path(args.results_root) if args.results_root else None,
        )
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
