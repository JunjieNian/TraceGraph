#!/usr/bin/env python3
"""RQ1 — Generate annotation samples for semantic validation of graph structure.

Sampling targets:
  1. Within vs Across BCC pairs (200 pairs): same BCC vs different BCC step pairs
     with raw action/observation text for human annotation
  2. Articulation transitions (100): consecutive steps crossing articulation points,
     to judge whether phase transitions exist
  3. Core/Basin block samples (50+50): blocks labelled as core or basin,
     with representative steps for productive/looping/dead-end classification

Output: results/cxcmu/annotation_samples/
  - bcc_pairs.json          (200 within + 200 across pairs)
  - articulation_transitions.json  (100 transitions)
  - block_samples.json      (100 core + basin samples)

Usage:
    python scripts/89_annotation_sampling.py [--max-tasks N] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from tracegraph.graph_construction import nontrivial_blocks
from tracegraph.reward_field import (
    build_run_resolved,
    build_visit_sets,
    nontrivial_block_info,
    reconstruct_run_sequences,
)

GRAPH_DIR = Path("data/cxcmu/graphs")
PARSED_DIR = Path("data/cxcmu/parsed")
RESULTS_DIR = Path("results/cxcmu/annotation_samples")

# Sampling targets
N_BCC_PAIRS = 200       # 200 within + 200 across
N_ARTICULATION = 100
N_CORE_SAMPLES = 50
N_BASIN_SAMPLES = 50


def _load_parsed_steps(bench: str, task_id: str) -> dict[str, list[dict]]:
    """Load parsed steps indexed by (model_id, run_id) → steps list."""
    parsed_path = PARSED_DIR / bench / f"{task_id}.jsonl"
    if not parsed_path.exists():
        return {}
    runs = {}
    with open(parsed_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row.get('model_id', 'unknown')}_{row.get('run_id', '')}"
            runs[key] = row.get("steps", [])
    return runs


def _get_step_text(steps: list[dict], step_idx: int) -> dict:
    """Extract raw action/observation text for a step."""
    if step_idx < 0 or step_idx >= len(steps):
        return {"action": "", "observation": "", "tool_name": "", "action_type": ""}
    s = steps[step_idx]
    return {
        "action": s.get("raw_action", "")[:500],
        "observation": s.get("raw_observation", "")[:500],
        "tool_name": s.get("tool_name", ""),
        "action_type": s.get("action_type", ""),
        "command_class": s.get("command_class", ""),
    }


def _find_run_key_for_slice(payload: dict, slice_idx: int, parsed_runs: dict) -> tuple[str | None, int]:
    """Map a graph slice index back to a parsed run + step index."""
    cache = payload.get("role_threshold_cache", {})
    slice_runs = cache.get("slice_runs", [])
    slice_positions = cache.get("slice_positions", [])

    if slice_idx >= len(slice_runs):
        return None, -1

    run_id = int(slice_runs[slice_idx])
    step_pos = int(slice_positions[slice_idx]) if slice_idx < len(slice_positions) else 0

    # Map numeric run_id to parsed run key
    model_ids = payload.get("slice_model_ids", [])
    model_id = model_ids[slice_idx] if slice_idx < len(model_ids) else "unknown"

    # Try to find matching run
    run_id_map = payload.get("run_id_map", {})
    run_str = run_id_map.get(run_id, run_id_map.get(str(run_id), str(run_id)))

    # Try various key formats
    for key in [f"{model_id}_{run_str}", f"{model_id}_{run_id}"]:
        if key in parsed_runs:
            return key, step_pos

    # Fallback: try by run_id order
    keys = sorted(parsed_runs.keys())
    if run_id < len(keys):
        return keys[run_id], step_pos

    return None, step_pos


def sample_bcc_pairs(
    all_payloads: list[tuple[str, str, dict]],
    rng: random.Random,
    n_pairs: int = N_BCC_PAIRS,
) -> list[dict]:
    """Sample within-BCC and across-BCC pairs with raw text."""
    within_candidates = []
    across_candidates = []

    for bench, task_id, payload in all_payloads:
        blocks = payload.get("bcc_analysis", {}).get("blocks", [])
        nontrivial = [b for b in blocks if not b.get("is_trivial", False) and b.get("n_nodes", 0) >= 3]
        if len(nontrivial) < 2:
            continue

        cache = payload.get("role_threshold_cache", {})
        node_bcc_map = cache.get("node_bcc_map", {})

        # Build block → node set
        block_nodes = defaultdict(set)
        for node_idx_str, bids in node_bcc_map.items():
            node_idx = int(node_idx_str) if isinstance(node_idx_str, str) else node_idx_str
            for bid in bids:
                bid = int(bid)
                if any(b["block_id"] == bid and not b.get("is_trivial", False) for b in blocks):
                    block_nodes[bid].add(node_idx)

        # Within-BCC pairs: two slices in the same block
        for bid, nodes in block_nodes.items():
            nodes_list = sorted(nodes)
            if len(nodes_list) >= 2:
                for _ in range(min(3, len(nodes_list) // 2)):
                    pair = rng.sample(nodes_list, 2)
                    within_candidates.append({
                        "bench": bench, "task_id": task_id,
                        "block_id": bid, "type": "within_bcc",
                        "slice_a": pair[0], "slice_b": pair[1],
                    })

        # Across-BCC pairs: two slices in different blocks
        block_ids = sorted(block_nodes.keys())
        if len(block_ids) >= 2:
            for _ in range(min(5, len(block_ids))):
                b1, b2 = rng.sample(block_ids, 2)
                if block_nodes[b1] and block_nodes[b2]:
                    s1 = rng.choice(sorted(block_nodes[b1]))
                    s2 = rng.choice(sorted(block_nodes[b2]))
                    across_candidates.append({
                        "bench": bench, "task_id": task_id,
                        "block_a": b1, "block_b": b2, "type": "across_bcc",
                        "slice_a": s1, "slice_b": s2,
                    })

    # Sample
    within_sampled = rng.sample(within_candidates, min(n_pairs, len(within_candidates)))
    across_sampled = rng.sample(across_candidates, min(n_pairs, len(across_candidates)))

    # Enrich with text
    results = []
    for item in within_sampled + across_sampled:
        bench, task_id = item["bench"], item["task_id"]
        parsed_runs = _load_parsed_steps(bench, task_id)
        # Reload payload for slice lookup
        gpath = GRAPH_DIR / bench / f"{task_id}.pkl"
        with open(gpath, "rb") as f:
            payload = pickle.load(f)

        run_key_a, pos_a = _find_run_key_for_slice(payload, item["slice_a"], parsed_runs)
        run_key_b, pos_b = _find_run_key_for_slice(payload, item["slice_b"], parsed_runs)

        text_a = _get_step_text(parsed_runs.get(run_key_a, []), pos_a) if run_key_a else {}
        text_b = _get_step_text(parsed_runs.get(run_key_b, []), pos_b) if run_key_b else {}

        results.append({
            **item,
            "text_a": text_a,
            "text_b": text_b,
            "annotation_question": "Are these two steps semantically similar (same phase/strategy)?",
            "expected_label": "yes" if item["type"] == "within_bcc" else "unknown",
        })

    return results


def sample_articulation_transitions(
    all_payloads: list[tuple[str, str, dict]],
    rng: random.Random,
    n_samples: int = N_ARTICULATION,
) -> list[dict]:
    """Sample transitions across articulation points."""
    candidates = []

    for bench, task_id, payload in all_payloads:
        art_points = set(payload.get("bcc_analysis", {}).get("articulation_points", []))
        if not art_points:
            continue

        run_sequences = reconstruct_run_sequences(payload)
        cache = payload.get("role_threshold_cache", {})
        node_bcc_map = cache.get("node_bcc_map", {})
        block_meta = nontrivial_block_info(payload)

        for rid, seq in run_sequences.items():
            for i in range(len(seq) - 1):
                step_a = seq[i]
                step_b = seq[i + 1]
                local_a = step_a.get("local_idx")
                local_b = step_b.get("local_idx")

                # Check if the node at step boundary is an articulation point
                if local_a is not None and int(local_a) in art_points:
                    block_a = step_a.get("primary_block")
                    block_b = step_b.get("primary_block")
                    if block_a is not None and block_b is not None and block_a != block_b:
                        role_a = block_meta.get(int(block_a), {}).get("block_type", "unknown")
                        role_b = block_meta.get(int(block_b), {}).get("block_type", "unknown")
                        candidates.append({
                            "bench": bench, "task_id": task_id,
                            "run_id": rid,
                            "slice_a": local_a, "slice_b": local_b,
                            "block_a": block_a, "block_b": block_b,
                            "role_a": role_a, "role_b": role_b,
                            "articulation_node": int(local_a),
                        })

    sampled = rng.sample(candidates, min(n_samples, len(candidates)))

    # Enrich with text
    results = []
    for item in sampled:
        bench, task_id = item["bench"], item["task_id"]
        parsed_runs = _load_parsed_steps(bench, task_id)
        gpath = GRAPH_DIR / bench / f"{task_id}.pkl"
        with open(gpath, "rb") as f:
            payload = pickle.load(f)

        run_key_a, pos_a = _find_run_key_for_slice(payload, item["slice_a"], parsed_runs)
        run_key_b, pos_b = _find_run_key_for_slice(payload, item["slice_b"], parsed_runs)

        text_a = _get_step_text(parsed_runs.get(run_key_a, []), pos_a) if run_key_a else {}
        text_b = _get_step_text(parsed_runs.get(run_key_b, []), pos_b) if run_key_b else {}

        results.append({
            **item,
            "text_before": text_a,
            "text_after": text_b,
            "annotation_question": "Does this transition represent a phase change in strategy?",
            "annotation_options": ["clear_phase_change", "minor_shift", "no_change", "unclear"],
        })

    return results


def sample_core_basin_blocks(
    all_payloads: list[tuple[str, str, dict]],
    rng: random.Random,
    n_core: int = N_CORE_SAMPLES,
    n_basin: int = N_BASIN_SAMPLES,
) -> list[dict]:
    """Sample blocks from core and basin regions with representative steps."""
    core_candidates = []
    basin_candidates = []

    for bench, task_id, payload in all_payloads:
        block_meta = nontrivial_block_info(payload)
        rf = payload.get("reward_field", {})
        basins_data = payload.get("failure_basins", {}).get("basins", [])

        if not rf:
            continue

        core_mask = rf.get("core_mask", [])
        field = rf.get("field", [])
        rf_nodes = rf.get("nodes", [])
        rf_node_to_idx = {int(k): v for k, v in rf.get("node_to_idx", {}).items()}

        # Core blocks
        for bid, meta in block_meta.items():
            idx = rf_node_to_idx.get(bid)
            if idx is not None and idx < len(core_mask) and core_mask[idx]:
                core_candidates.append({
                    "bench": bench, "task_id": task_id,
                    "block_id": bid,
                    "block_type": meta.get("block_type", "unknown"),
                    "n_nodes": meta.get("n_nodes", 0),
                    "resolved_purity": meta.get("resolved_purity", 0.0),
                    "field_value": float(field[idx]) if idx < len(field) else None,
                    "region": "core",
                })

        # Basin blocks
        all_basin_blocks = set()
        for basin in basins_data:
            for bid in basin.get("block_ids", []):
                all_basin_blocks.add(int(bid))

        for bid in all_basin_blocks:
            if bid in block_meta:
                meta = block_meta[bid]
                idx = rf_node_to_idx.get(bid)
                basin_candidates.append({
                    "bench": bench, "task_id": task_id,
                    "block_id": bid,
                    "block_type": meta.get("block_type", "unknown"),
                    "n_nodes": meta.get("n_nodes", 0),
                    "resolved_purity": meta.get("resolved_purity", 0.0),
                    "field_value": float(field[idx]) if idx is not None and idx < len(field) else None,
                    "region": "basin",
                })

    core_sampled = rng.sample(core_candidates, min(n_core, len(core_candidates)))
    basin_sampled = rng.sample(basin_candidates, min(n_basin, len(basin_candidates)))

    # Enrich with representative step text
    results = []
    for item in core_sampled + basin_sampled:
        bench, task_id, bid = item["bench"], item["task_id"], item["block_id"]
        parsed_runs = _load_parsed_steps(bench, task_id)
        gpath = GRAPH_DIR / bench / f"{task_id}.pkl"
        with open(gpath, "rb") as f:
            payload = pickle.load(f)

        # Get slices in this block
        cache = payload.get("role_threshold_cache", {})
        node_bcc_map = cache.get("node_bcc_map", {})
        block_slices = []
        for node_idx_str, bids in node_bcc_map.items():
            node_idx = int(node_idx_str) if isinstance(node_idx_str, str) else node_idx_str
            if bid in [int(b) for b in bids]:
                block_slices.append(node_idx)

        # Sample up to 3 representative steps
        sample_slices = rng.sample(block_slices, min(3, len(block_slices))) if block_slices else []
        step_texts = []
        for s_idx in sample_slices:
            run_key, pos = _find_run_key_for_slice(payload, s_idx, parsed_runs)
            if run_key:
                step_texts.append(_get_step_text(parsed_runs.get(run_key, []), pos))

        results.append({
            **item,
            "representative_steps": step_texts,
            "annotation_question": "What is the nature of activity in this block?",
            "annotation_options": ["productive_progress", "exploration", "looping_retry", "dead_end", "recovery", "unclear"],
        })

    return results


def main(max_tasks: int | None = None, seed: int = 42):
    rng = random.Random(seed)
    np.random.seed(seed)

    benchmarks = sorted(d.name for d in GRAPH_DIR.iterdir() if d.is_dir())
    if not benchmarks:
        print(f"No benchmarks found in {GRAPH_DIR}")
        return

    # Load all payloads
    print("Loading graph payloads...")
    all_payloads: list[tuple[str, str, dict]] = []
    for bench in benchmarks:
        bench_dir = GRAPH_DIR / bench
        graph_files = sorted(bench_dir.glob("*.pkl"))
        if max_tasks:
            graph_files = graph_files[:max_tasks]
        for gpath in tqdm(graph_files, desc=f"  Loading {bench}"):
            with open(gpath, "rb") as f:
                payload = pickle.load(f)
            all_payloads.append((bench, gpath.stem, payload))

    print(f"Loaded {len(all_payloads)} task payloads")

    # Sample
    print("\n── Sampling BCC pairs ──")
    bcc_pairs = sample_bcc_pairs(all_payloads, rng)
    n_within = sum(1 for p in bcc_pairs if p["type"] == "within_bcc")
    n_across = sum(1 for p in bcc_pairs if p["type"] == "across_bcc")
    print(f"  {n_within} within-BCC + {n_across} across-BCC = {len(bcc_pairs)} pairs")

    print("\n── Sampling articulation transitions ──")
    art_transitions = sample_articulation_transitions(all_payloads, rng)
    print(f"  {len(art_transitions)} transitions sampled")

    print("\n── Sampling core/basin blocks ──")
    block_samples = sample_core_basin_blocks(all_payloads, rng)
    n_core_s = sum(1 for s in block_samples if s["region"] == "core")
    n_basin_s = sum(1 for s in block_samples if s["region"] == "basin")
    print(f"  {n_core_s} core + {n_basin_s} basin = {len(block_samples)} blocks")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "bcc_pairs.json", "w") as f:
        json.dump(bcc_pairs, f, indent=2, default=str)

    with open(RESULTS_DIR / "articulation_transitions.json", "w") as f:
        json.dump(art_transitions, f, indent=2, default=str)

    with open(RESULTS_DIR / "block_samples.json", "w") as f:
        json.dump(block_samples, f, indent=2, default=str)

    print(f"\n── Output ──")
    print(f"  {RESULTS_DIR / 'bcc_pairs.json'}")
    print(f"  {RESULTS_DIR / 'articulation_transitions.json'}")
    print(f"  {RESULTS_DIR / 'block_samples.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(max_tasks=args.max_tasks, seed=args.seed)
