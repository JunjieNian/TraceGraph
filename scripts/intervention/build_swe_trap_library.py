#!/usr/bin/env python3
"""Build a cleaned SWE-bench intervention library (trap / core / nontrap + IDF).

The script scans cx-cmu SWE-bench graph artifacts and emits the detector
libraries consumed by `scripts/intervention/swe_runner.py`
to detect trap-like (low-outcome) steps and core-like (productive) steps
during a live multi-step SWE agent run on the swebench docker images.

Per task graph:
  1. Trap blocks = bottom-quartile of the diffused negative reward field
     (same rule as scripts/100_rollout_events.py).
  2. Core blocks = existing top-quartile core mask in the graph payload.
  3. Walk node_bcc_map to harvest step-level key sets.
  4. Strip generic detection keys (PHASE:*, ACTION:other).
  5. Aggregate by sorted detection-key tuple; deduplicate ambiguous trap/core.

For SWE the relevant signal-bearing keys include:
  TOOL:execute_bash, ACTION:edit / view / create, CMD:pytest / grep / cat / sed / python / git
  OBS:error, OBS:traceback, OBS:test_failed, OBS:test_passed, etc.

Outputs:
  data/cxcmu/intervention/swebench_trap_library.json
  data/cxcmu/intervention/swebench_core_library.json
  data/cxcmu/intervention/swebench_nontrap_library.json
  data/cxcmu/intervention/swebench_idf_corpus.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from tracegraph.signature import build_idf_weights

GRAPH_DIR = Path("data/cxcmu/graphs/swebench")
SIG_DIR = Path("data/cxcmu/signatures/swebench")
OUT_DIR = Path("data/cxcmu/intervention")

TRAP_QUANTILE = 0.25

GENERIC_KEY_PREFIXES = ("PHASE:",)
GENERIC_KEYS = {"ACTION:other"}


def trap_mask_from_field(field: np.ndarray, q: float = TRAP_QUANTILE) -> np.ndarray:
    if field.size == 0:
        return np.zeros(0, dtype=bool)
    neg = field[field < 0]
    if neg.size == 0:
        return np.zeros_like(field, dtype=bool)
    if neg.size <= 3:
        return field < 0
    thr = float(np.quantile(neg, q))
    return np.logical_and(field < 0, field <= thr)


def strip_generic_keys(keys: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for k in keys:
        if any(k.startswith(p) for p in GENERIC_KEY_PREFIXES):
            continue
        if k in GENERIC_KEYS:
            continue
        out.add(k)
    return out


def tool_names_from_keys(keys: Iterable[str]) -> List[str]:
    return sorted(k.split("TOOL:", 1)[1] for k in keys if k.startswith("TOOL:"))


def make_example(task_id: str, raw_keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    key_set = set(raw_keys)
    detection = strip_generic_keys(key_set)
    if not detection:
        return None
    return {
        "task_id": task_id,
        "keys": sorted(key_set),
        "detection_keys": sorted(detection),
        "tool_names": tool_names_from_keys(key_set),
        "task_ids": [task_id],
        "n_raw": 1,
    }


def aggregate_examples(examples: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, ...]]]:
    by_sig: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for ex in examples:
        sig = tuple(ex["detection_keys"])
        cur = by_sig.get(sig)
        if cur is None:
            by_sig[sig] = {
                "keys": list(ex["keys"]),
                "detection_keys": list(ex["detection_keys"]),
                "task_ids": set(ex["task_ids"]),
                "tool_names": set(ex["tool_names"]),
                "n_raw": int(ex.get("n_raw", 1)),
            }
            continue
        cur["task_ids"].update(ex["task_ids"])
        cur["tool_names"].update(ex["tool_names"])
        cur["n_raw"] += int(ex.get("n_raw", 1))
    out: List[Dict[str, Any]] = []
    for sig, ex in sorted(by_sig.items()):
        out.append({
            "keys": sorted(ex["keys"]),
            "detection_keys": sorted(ex["detection_keys"]),
            "task_ids": sorted(ex["task_ids"]),
            "tool_names": sorted(ex["tool_names"]),
            "n_raw": int(ex["n_raw"]),
        })
    return out, set(by_sig.keys())


def write_library(path: Path, *, kind: str, examples: Sequence[Dict[str, Any]], n_tasks: int, n_steps: int) -> None:
    with open(path, "w") as fh:
        json.dump({
            "domain": "swebench",
            "kind": kind,
            "n_tasks": int(n_tasks),
            "n_steps": int(n_steps),
            "examples": list(examples),
        }, fh, indent=2)


def main(graph_dir: Path = GRAPH_DIR, sig_dir: Path = SIG_DIR, out_dir: Path = OUT_DIR) -> None:
    graph_dir = Path(graph_dir)
    sig_dir = Path(sig_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trap_raw: List[Dict[str, Any]] = []
    core_raw: List[Dict[str, Any]] = []
    nontrap_raw: List[Dict[str, Any]] = []
    step_sets: List[Set[str]] = []
    task_ids_seen: Set[str] = set()
    n_steps_total = 0

    for gp in sorted(graph_dir.glob("*.pkl")):
        tid = gp.stem
        sig_root = sig_dir / tid
        sig_path = sig_root / "key_sets.pkl"
        if not sig_path.exists():
            continue
        try:
            payload = pickle.load(open(gp, "rb"))
            key_sets = pickle.load(open(sig_path, "rb"))
        except Exception:
            continue
        rf = payload.get("reward_field") or {}
        nodes = rf.get("nodes")
        field = rf.get("field")
        core_mask = rf.get("core_mask")
        if not nodes or field is None:
            continue
        field_arr = np.asarray(field, dtype=float)
        trap_arr = trap_mask_from_field(field_arr, TRAP_QUANTILE)
        core_arr = (np.asarray(core_mask, dtype=bool) if core_mask is not None
                    else np.zeros_like(field_arr, dtype=bool))
        trap_blocks = {int(nodes[i]) for i, on in enumerate(trap_arr) if on}
        core_blocks = {int(nodes[i]) for i, on in enumerate(core_arr) if on}

        rtc = payload.get("role_threshold_cache") or {}
        node_bcc_map = rtc.get("node_bcc_map", {})
        task_ids_seen.add(tid)
        n_steps_total += len(key_sets)

        for li, bids in (node_bcc_map.items() if isinstance(node_bcc_map, dict) else []):
            li_int = int(li)
            if li_int >= len(key_sets):
                continue
            example = make_example(tid, key_sets[li_int])
            if example is None:
                continue
            step_sets.append(set(example["detection_keys"]))
            bids_int = {int(b) for b in bids}
            if bids_int & trap_blocks:
                trap_raw.append(example)
            elif bids_int & core_blocks:
                core_raw.append(example)
                nontrap_raw.append(example)
            else:
                nontrap_raw.append(example)

    trap_examples, trap_sigs = aggregate_examples(trap_raw)
    core_examples, core_sigs = aggregate_examples(core_raw)
    nontrap_examples, nontrap_sigs = aggregate_examples(nontrap_raw)

    ambiguous = trap_sigs & core_sigs
    trap_clean = [ex for ex in trap_examples if tuple(ex["detection_keys"]) not in ambiguous]
    core_clean = [ex for ex in core_examples if tuple(ex["detection_keys"]) not in ambiguous]
    forbidden = trap_sigs | core_sigs
    nontrap_clean = [ex for ex in nontrap_examples if tuple(ex["detection_keys"]) not in forbidden]

    idf = build_idf_weights(step_sets)

    write_library(out_dir / "swebench_trap_library.json",
                  kind="trap", examples=trap_clean,
                  n_tasks=len(task_ids_seen), n_steps=n_steps_total)
    write_library(out_dir / "swebench_core_library.json",
                  kind="core", examples=core_clean,
                  n_tasks=len(task_ids_seen), n_steps=n_steps_total)
    write_library(out_dir / "swebench_nontrap_library.json",
                  kind="nontrap", examples=nontrap_clean,
                  n_tasks=len(task_ids_seen), n_steps=n_steps_total)
    with open(out_dir / "swebench_idf_corpus.json", "w") as fh:
        json.dump(idf, fh, indent=2)

    print(f"── swebench library summary ──")
    print(f"Tasks scanned:        {len(task_ids_seen)}")
    print(f"Steps total:          {n_steps_total}")
    print(f"Trap signatures:      {len(trap_clean)}")
    print(f"Core signatures:      {len(core_clean)}")
    print(f"Non-trap signatures:  {len(nontrap_clean)}")
    print(f"Ambiguous removed:    {len(ambiguous)}")
    print(f"IDF keys:             {len(idf)}")
    cnt = Counter()
    for ex in trap_clean:
        for k in ex["detection_keys"]:
            cnt[k] += 1
    if cnt:
        print("Top-10 trap detection keys:")
        for k, n in cnt.most_common(10):
            print(f"  {n:4d}  {k}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--sig-dir", type=Path, default=SIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(graph_dir=args.graph_dir, sig_dir=args.sig_dir, out_dir=args.out_dir)
