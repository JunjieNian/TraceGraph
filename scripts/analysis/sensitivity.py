#!/usr/bin/env python3
"""S1-S5 sensitivity + bootstrap CI for paper Tables 2 and 3.

Reads existing graph payloads in data/cxcmu/graphs/<bench>/<task>.pkl and
parsed rollouts in data/cxcmu/parsed/<bench>/<task>.jsonl. NO graph rebuild.

Outputs all into results/cxcmu/sensitivity/:
  supply_ci.json            S1 — bootstrap CI on each S_{m,a}
  demand_ci.json            S2 — bootstrap CI on each D_{b,a}
  trap_quantile_sweep.json  S3 — sweep over trap quantile
  core_quantile_sweep.json  S4 — sweep over core quantile (re-derived from field)
  laplace_sweep.json        S5 — sweep over Laplace constant in q_t(g)

CI / bootstrap design:
  Supply (S_{m,a}): bootstrap unit = (benchmark, task). On each resample, recompute
    a_{m,t} and the task-centered residual, then the model average.
  Demand (D_{b,a}): bootstrap unit = rollout within benchmark; the reward-weighted
    contrast is recomputed on each resample.
  B = 2000 resamples (default).

Sweeps share the same pipeline as script 100 but vary the corresponding parameter;
each setting produces one supply table and one demand table.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from tracegraph.reward_field import (
    build_run_resolved,
    nontrivial_block_info,
    reconstruct_run_sequences,
)

# Paths
GRAPH_DIR = Path("data/cxcmu/graphs")
PARSED_DIR = Path("data/cxcmu/parsed")
OUT_DIR = Path("results/cxcmu/sensitivity")

# Defaults match the paper main text
TRAP_Q_DEFAULT = 0.25
CORE_Q_DEFAULT = 0.75
LAPLACE_DEFAULT = 0.5

ARMS = ("A_r", "E_r", "R_r", "G_r")


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_parsed(bench: str, task_id: str) -> Dict[str, Dict[str, Any]]:
    fp = PARSED_DIR / bench / f"{task_id}.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    if not fp.exists():
        return out
    for line in open(fp):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        md = row.get("metadata") or {}
        run_id_str = row.get("run_id")
        pass_tag = md.get("pass")
        if run_id_str is not None:
            key = run_id_str
        elif pass_tag is not None:
            pass_tag_str = str(pass_tag)
            if pass_tag_str.startswith("pass"):
                key = f"{row.get('model_id')}__{pass_tag_str}"
            else:
                key = f"{row.get('model_id')}__pass{pass_tag_str}"
        else:
            continue
        score = md.get("resolved_score")
        if score is None:
            score = 1.0 if row.get("resolved") else 0.0
        out[key] = {
            "model_id": row.get("model_id"),
            "resolved_score": float(score),
        }
    return out


def compute_trap_mask(field: np.ndarray, quantile: float) -> np.ndarray:
    if field.size == 0:
        return np.zeros(0, dtype=bool)
    negative = field[field < 0]
    if negative.size == 0:
        return np.zeros_like(field, dtype=bool)
    if negative.size <= 3:
        return field < 0
    thr = float(np.quantile(negative, quantile))
    return np.logical_and(field < 0, field <= thr)


def compute_core_mask(field: np.ndarray, quantile: float) -> np.ndarray:
    if field.size == 0:
        return np.zeros(0, dtype=bool)
    positive = field[field > 0]
    if positive.size == 0:
        return np.zeros_like(field, dtype=bool)
    if positive.size <= 3:
        return field > 0
    thr = float(np.quantile(positive, quantile))
    return np.logical_and(field > 0, field >= thr)


def gate_block_set(block_meta: Dict[int, dict]) -> set:
    return {int(bid) for bid, m in block_meta.items() if m.get("has_articulation")}


def compact_block_sequence(steps: List[dict]) -> List[int]:
    out: List[int] = []
    for step in steps:
        prim = step.get("primary_block")
        if prim is None:
            continue
        prim = int(prim)
        if not out or out[-1] != prim:
            out.append(prim)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-task event computation, with knobs for sweeps
# ──────────────────────────────────────────────────────────────────────────────
def process_task(payload: dict, parsed_lookup: Dict[str, Dict[str, Any]],
                 *,
                 trap_quantile: float = TRAP_Q_DEFAULT,
                 core_quantile: Optional[float] = None,  # None = use baked core_mask
                 laplace: float = LAPLACE_DEFAULT,
                 benchmark: str = "",
                 task_id: str = "") -> List[Dict[str, Any]]:
    rf = payload.get("reward_field") or {}
    nodes = rf.get("nodes")
    field = rf.get("field")
    baked_core = rf.get("core_mask")
    if not nodes or field is None:
        return []
    field_arr = np.asarray(field, dtype=float)
    if core_quantile is None and baked_core is not None:
        core_arr = np.asarray(baked_core, dtype=bool)
    else:
        core_arr = compute_core_mask(field_arr, core_quantile if core_quantile is not None else CORE_Q_DEFAULT)
    trap_arr = compute_trap_mask(field_arr, trap_quantile)

    core_blocks = {int(nodes[i]) for i, c in enumerate(core_arr) if c}
    trap_blocks = {int(nodes[i]) for i, c in enumerate(trap_arr) if c}
    block_meta = nontrivial_block_info(payload)
    gate_blocks = gate_block_set(block_meta)

    run_sequences = reconstruct_run_sequences(payload)
    if not run_sequences:
        return []

    unique_runs = payload.get("unique_runs", []) or []
    run_id_map = payload.get("run_id_map", {}) or {}
    rid_to_runstr: Dict[int, str] = {int(rid): str(rs) for rs, rid in run_id_map.items()}
    if not rid_to_runstr and unique_runs:
        rid_to_runstr = {i: str(s) for i, s in enumerate(unique_runs)}

    run_resolved = build_run_resolved(payload)
    rollouts: List[Dict[str, Any]] = []
    for rid, steps in run_sequences.items():
        rs = rid_to_runstr.get(int(rid))
        if rs is None:
            continue
        meta = parsed_lookup.get(rs)
        if meta is not None:
            model_id = meta["model_id"]
            y_r = float(meta["resolved_score"])
        else:
            model_id = rs.split("__", 1)[0]
            y_r = 1.0 if run_resolved.get(int(rid), False) else 0.0

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
            "rid": int(rid), "run_str": rs, "model_id": model_id, "y_r": y_r,
            "A_r": A_r, "E_r": E_r, "R_r": R_r,
            "visited_gates": visited_gates,
        })

    # gate map
    gate_num: Dict[int, float] = defaultdict(float)
    gate_den: Dict[int, int] = defaultdict(int)
    for r in rollouts:
        for g in r["visited_gates"]:
            gate_num[int(g)] += float(r["y_r"])
            gate_den[int(g)] += 1
    q_g: Dict[int, float] = {}
    for g, n in gate_den.items():
        q_g[int(g)] = (gate_num[g] + laplace) / (n + 2.0 * laplace)

    out: List[Dict[str, Any]] = []
    for r in rollouts:
        if r["visited_gates"]:
            vals = [q_g[int(g)] for g in r["visited_gates"] if int(g) in q_g]
            G_r = float(np.mean(vals)) if vals else None
        else:
            G_r = None
        out.append({
            "benchmark": benchmark, "task_id": task_id,
            "model_id": r["model_id"], "run_str": r["run_str"],
            "y_r": r["y_r"], "A_r": r["A_r"], "E_r": r["E_r"], "R_r": r["R_r"],
            "G_r": G_r, "visited_gate": G_r is not None,
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────
def supply_demand_from_rollouts(rows: List[Dict[str, Any]]) -> Tuple[Dict, Dict]:
    """Return (supply, demand) dicts.
    supply[model][event] = S_{m,a}; demand[benchmark][event] = D_{b,a}.
    """
    # per-(bench,task,model) cell averages
    cells: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["benchmark"], r["task_id"], r["model_id"])
        for ev in ARMS:
            v = r.get(ev)
            if v is None:
                continue
            cells[key][ev].append(float(v))

    # a_{m,t}
    cell_means: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for key, ev_lists in cells.items():
        cell_means[key] = {ev: (sum(vs) / len(vs)) for ev, vs in ev_lists.items() if vs}

    # task-centered residual then model average
    by_task_event: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (bench, tid, m), cm in cell_means.items():
        for ev, v in cm.items():
            by_task_event[(bench, tid)][ev].append((m, v))
    supply_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (bench, tid), ev_lists in by_task_event.items():
        for ev, mlist in ev_lists.items():
            mean_v = sum(v for _, v in mlist) / len(mlist)
            for m, v in mlist:
                supply_acc[m][ev].append(v - mean_v)
    supply = {m: {ev: (sum(xs) / len(xs)) if xs else 0.0 for ev, xs in d.items()} for m, d in supply_acc.items()}

    # demand
    demand: Dict[str, Dict[str, float]] = {}
    by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)
    for bench, rs in by_bench.items():
        ydem = {}
        for ev in ARMS:
            num1 = den1 = num0 = den0 = 0.0
            for r in rs:
                v = r.get(ev)
                if v is None:
                    continue
                w = float(r.get("y_r", 0))
                num1 += w * float(v); den1 += w
                num0 += (1.0 - w) * float(v); den0 += (1.0 - w)
            a = num1 / den1 if den1 > 0 else float("nan")
            b = num0 / den0 if den0 > 0 else float("nan")
            ydem[ev] = a - b
        demand[bench] = ydem
    return supply, demand


def bootstrap_supply(rows: List[Dict[str, Any]], B: int, seed: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Bootstrap CI on supply S_{m,a}.
    Resampling unit: (benchmark, task). On each resample we recompute the cell
    averages, task-centered residual, and per-model supply average.
    """
    # group rows by (bench, task)
    by_bt: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bt[(r["benchmark"], r["task_id"])].append(r)
    keys = list(by_bt.keys())
    n_keys = len(keys)
    rng = random.Random(seed)
    samples: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for b in range(B):
        idx = [rng.randrange(n_keys) for _ in range(n_keys)]
        rep_rows: List[Dict[str, Any]] = []
        # task id may repeat; we treat each sampled task as a distinct copy
        # by appending an index suffix to its task_id so it doesn't collide.
        for j, i in enumerate(idx):
            kk = keys[i]
            tag = f"{kk[1]}__b{j}"
            for r in by_bt[kk]:
                rep_rows.append({**r, "task_id": tag})
        sup, _ = supply_demand_from_rollouts(rep_rows)
        for m, d in sup.items():
            for ev, v in d.items():
                samples[m][ev].append(v)
    cis: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m, d in samples.items():
        cis[m] = {}
        for ev, xs in d.items():
            xs.sort()
            cis[m][ev] = {
                "median": xs[len(xs) // 2] if xs else float("nan"),
                "lo95": xs[int(0.025 * len(xs))] if xs else float("nan"),
                "hi95": xs[int(0.975 * len(xs))] if xs else float("nan"),
            }
    return cis


def bootstrap_demand(rows: List[Dict[str, Any]], B: int, seed: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Bootstrap CI on demand D_{b,a}.
    Resampling unit: rollouts within each benchmark, with replacement.
    """
    by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)
    rng = random.Random(seed)
    samples: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for bench, rs in by_bench.items():
        n = len(rs)
        for b in range(B):
            idx = [rng.randrange(n) for _ in range(n)]
            rep_rows = [rs[i] for i in idx]
            for ev in ARMS:
                num1 = den1 = num0 = den0 = 0.0
                for r in rep_rows:
                    v = r.get(ev)
                    if v is None:
                        continue
                    w = float(r.get("y_r", 0))
                    num1 += w * float(v); den1 += w
                    num0 += (1.0 - w) * float(v); den0 += (1.0 - w)
                if den1 > 0 and den0 > 0:
                    samples[bench][ev].append(num1 / den1 - num0 / den0)
    cis: Dict[str, Dict[str, Dict[str, float]]] = {}
    for bench, d in samples.items():
        cis[bench] = {}
        for ev, xs in d.items():
            xs.sort()
            cis[bench][ev] = {
                "median": xs[len(xs) // 2] if xs else float("nan"),
                "lo95": xs[int(0.025 * len(xs))] if xs else float("nan"),
                "hi95": xs[int(0.975 * len(xs))] if xs else float("nan"),
            }
    return cis


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ──────────────────────────────────────────────────────────────────────────────
def collect_rows(*,
                 trap_quantile: float = TRAP_Q_DEFAULT,
                 core_quantile: Optional[float] = None,
                 laplace: float = LAPLACE_DEFAULT,
                 max_tasks_per_bench: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bench_dir in sorted(GRAPH_DIR.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        graph_files = sorted(bench_dir.glob("*.pkl"))
        if max_tasks_per_bench:
            graph_files = graph_files[:max_tasks_per_bench]
        for gp in graph_files:
            task_id = gp.stem
            try:
                payload = pickle.load(open(gp, "rb"))
            except Exception:
                continue
            parsed_lookup = load_parsed(bench, task_id)
            rows.extend(process_task(payload, parsed_lookup,
                                      trap_quantile=trap_quantile,
                                      core_quantile=core_quantile,
                                      laplace=laplace,
                                      benchmark=bench,
                                      task_id=task_id))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="skip S1/S2 bootstrap CIs (sweeps only)")
    ap.add_argument("--skip-sweeps", action="store_true",
                    help="skip S3/S4/S5 sweeps (bootstrap only)")
    ap.add_argument("--max-tasks-per-bench", type=int, default=0)
    args = ap.parse_args()
    max_tasks = args.max_tasks_per_bench if args.max_tasks_per_bench > 0 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline pass (default parameters) ────────────────────────────────────
    print("loading rows at default thresholds", flush=True)
    rows = collect_rows(max_tasks_per_bench=max_tasks)
    print(f"  collected {len(rows)} rollouts", flush=True)
    sup0, dem0 = supply_demand_from_rollouts(rows)
    json.dump({"supply": sup0, "demand": dem0},
              open(OUT_DIR / "baseline_point.json", "w"), indent=2)

    if not args.skip_bootstrap:
        # ── S1: supply bootstrap ──────────────────────────────────────────────
        print(f"S1: supply bootstrap (B={args.B})", flush=True)
        sup_ci = bootstrap_supply(rows, args.B, args.seed)
        json.dump({"point": sup0, "ci": sup_ci},
                  open(OUT_DIR / "supply_ci.json", "w"), indent=2)
        for m in sorted(sup_ci):
            for ev in ARMS:
                p = sup0.get(m, {}).get(ev, float("nan"))
                lo = sup_ci.get(m, {}).get(ev, {}).get("lo95", float("nan"))
                hi = sup_ci.get(m, {}).get(ev, {}).get("hi95", float("nan"))
                print(f"  {m:20s} {ev}: {p:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]")

        # ── S2: demand bootstrap ──────────────────────────────────────────────
        print(f"S2: demand bootstrap (B={args.B})", flush=True)
        dem_ci = bootstrap_demand(rows, args.B, args.seed)
        json.dump({"point": dem0, "ci": dem_ci},
                  open(OUT_DIR / "demand_ci.json", "w"), indent=2)
        for b in sorted(dem_ci):
            for ev in ARMS:
                p = dem0.get(b, {}).get(ev, float("nan"))
                lo = dem_ci.get(b, {}).get(ev, {}).get("lo95", float("nan"))
                hi = dem_ci.get(b, {}).get(ev, {}).get("hi95", float("nan"))
                print(f"  {b:14s} {ev}: {p:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]")

    if not args.skip_sweeps:
        # ── S3: trap quantile sweep ───────────────────────────────────────────
        print("S3: trap quantile sweep", flush=True)
        s3: Dict[str, Any] = {}
        for q in (0.10, 0.20, 0.25, 0.30, 0.50):
            rows_q = collect_rows(trap_quantile=q, max_tasks_per_bench=max_tasks)
            sup, dem = supply_demand_from_rollouts(rows_q)
            s3[f"q={q}"] = {"supply": sup, "demand": dem}
            print(f"  q={q}: collected {len(rows_q)} rollouts", flush=True)
        json.dump(s3, open(OUT_DIR / "trap_quantile_sweep.json", "w"), indent=2)

        # ── S4: core quantile sweep ───────────────────────────────────────────
        print("S4: core quantile sweep", flush=True)
        s4: Dict[str, Any] = {}
        for q in (0.50, 0.65, 0.75, 0.85, 0.90):
            rows_q = collect_rows(core_quantile=q, max_tasks_per_bench=max_tasks)
            sup, dem = supply_demand_from_rollouts(rows_q)
            s4[f"q={q}"] = {"supply": sup, "demand": dem}
            print(f"  q={q}: collected {len(rows_q)} rollouts", flush=True)
        json.dump(s4, open(OUT_DIR / "core_quantile_sweep.json", "w"), indent=2)

        # ── S5: Laplace sweep ────────────────────────────────────────────────
        print("S5: Laplace constant sweep", flush=True)
        s5: Dict[str, Any] = {}
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
            rows_l = collect_rows(laplace=lam, max_tasks_per_bench=max_tasks)
            sup, dem = supply_demand_from_rollouts(rows_l)
            s5[f"lambda={lam}"] = {"supply": sup, "demand": dem}
            print(f"  λ={lam}: collected {len(rows_l)} rollouts", flush=True)
        json.dump(s5, open(OUT_DIR / "laplace_sweep.json", "w"), indent=2)

    print(f"\nwrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
