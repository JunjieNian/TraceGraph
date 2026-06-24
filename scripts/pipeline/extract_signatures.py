#!/usr/bin/env python3
"""Extract key sets from parsed cx-cmu trajectories, compute IDF and pairwise distances.

Per task (all rollouts pooled):
1. Load all parsed trajectories for this task
2. For each step: extract key set using extract_slice_keys()
3. Compute IDF weights across all slices in this task
4. Compute pairwise weighted Jaccard distances
5. Compute kNN arrays (indices + distances)
6. Save: key_sets.pkl, idf_weights.json, slice_metadata.json, knn_*.npy

Usage:
    python scripts/82_extract_cxcmu_signatures.py [--max-tasks N] [--benchmark BENCH]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tracegraph.constants import MAX_NODES, NEIGHBOR_K
from tracegraph.signature import (
    build_idf_weights,
    compute_knn,
    compute_pairwise_distances,
)

PARSED_DIR = Path("data/cxcmu/parsed")
SIG_DIR = Path("data/cxcmu/signatures")


def _set_data_root(data_root: Path) -> None:
    """Re-point PARSED_DIR / SIG_DIR under ``data_root``.

    Layout convention: ``<data_root>/{parsed,signatures}/...``.
    Module-level globals are rebound so existing references continue to work.
    """
    global PARSED_DIR, SIG_DIR
    data_root = Path(data_root)
    PARSED_DIR = data_root / "parsed"
    SIG_DIR = data_root / "signatures"

# ── Search-benchmark evidence-quality enrichment ────────────────────
# The base signature (TOOL/CMD/OBS/FILE/PHASE) does not capture evidence
# quality, which is what the search split is mostly about.  When the
# benchmark is `search`, we add three families of keys that describe
# what each slice contributes to the run's evidence state:
#   URL_DOMAIN:{etld+1}       - which sources are touched
#   QUERY_NOVELTY:{new|refine|repeat}  - whether this query reuses prior tokens
#   EVIDENCE_COUNT:{low|mid|high}      - cumulative distinct domains seen so far
# These are computed at slice time but require run-level state, so they
# are injected in `process_task` rather than in `extract_cxcmu_slice_keys`.

_URL_RE = re.compile(r"https?://([^/\s\")\]\}>]+)", re.IGNORECASE)


def _extract_domains(text: str) -> set[str]:
    if not text:
        return set()
    out = set()
    for m in _URL_RE.finditer(text[:5000]):
        host = m.group(1).lower().rstrip(".,;:")
        # Drop port
        host = host.split(":", 1)[0]
        parts = [p for p in host.split(".") if p]
        if len(parts) < 2:
            continue
        # eTLD+1 approximation: last 2 dotted parts (good enough for search results)
        etld1 = ".".join(parts[-2:])
        out.add(etld1)
    return out


def _extract_query_tokens(action_text: str) -> set[str]:
    """Pull the `query`/`q` argument out of a JSON-ish action string and tokenize."""
    if not action_text:
        return set()
    q = ""
    try:
        obj = json.loads(action_text)
        if isinstance(obj, dict):
            for key in ("query", "q", "search_query", "input", "text"):
                v = obj.get(key)
                if isinstance(v, str) and v:
                    q = v
                    break
    except Exception:
        # Fall back: regex for "query": "..."
        m = re.search(r'"(?:query|q|search_query|input|text)"\s*:\s*"([^"]+)"', action_text)
        if m:
            q = m.group(1)
    if not q:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2}


def _query_novelty_label(cur: set[str], prior: list[set[str]]) -> str:
    if not cur:
        return "new"
    if not prior:
        return "new"
    max_j = 0.0
    for p in prior:
        union = cur | p
        if not union:
            continue
        j = len(cur & p) / len(union)
        if j > max_j:
            max_j = j
    if max_j >= 0.6:
        return "repeat"
    if max_j >= 0.2:
        return "refine"
    return "new"


def _evidence_bucket(n_domains: int) -> str:
    if n_domains < 3:
        return "low"
    if n_domains < 8:
        return "mid"
    return "high"


def extract_cxcmu_slice_keys(step: dict, progress: float) -> set[str]:
    """Extract key set from a cx-cmu parsed step.

    Produces keys compatible with tracegraph/signature.py format:
    TOOL:{name}, CMD:{class}, OBS:{pattern}, FILE_PATH:{path}, FILE_EXT:{ext}, PHASE:{label}
    """
    keys = set()

    # TOOL key
    tool_name = step.get("tool_name", "")
    if tool_name:
        keys.add(f"TOOL:{tool_name}")

    # CMD key (command class)
    cmd_class = step.get("command_class", "other")
    if cmd_class and cmd_class != "other":
        keys.add(f"CMD:{cmd_class}")

    # Action type as a key
    action_type = step.get("action_type", "other")
    if action_type:
        keys.add(f"ACTION:{action_type}")

    # OBS keys (observation signatures)
    for sig in step.get("observation_signature", []):
        keys.add(f"OBS:{sig}")

    # FILE_PATH and FILE_EXT keys
    for fp in step.get("files_touched", []) + step.get("files_read", []):
        if "/" in fp:
            # Last 2 path components
            parts = fp.rsplit("/", 2)
            short = "/".join(parts[-2:]) if len(parts) >= 2 else fp
            keys.add(f"FILE_PATH:{short}")
        # Extension
        if "." in fp:
            ext = fp.rsplit(".", 1)[-1]
            if ext and len(ext) <= 6:
                keys.add(f"FILE_EXT:{ext}")

    # PHASE key
    if progress <= 0.33:
        phase = "early"
    elif progress <= 0.67:
        phase = "mid"
    else:
        phase = "late"
    keys.add(f"PHASE:{phase}")

    return keys


def process_task(parsed_path: Path, benchmark: str = "") -> dict | None:
    """Process a single task: extract key sets, compute distances.

    For benchmark == 'search', also enrich each slice with evidence-quality
    keys (URL_DOMAIN, QUERY_NOVELTY, EVIDENCE_COUNT) computed using run-level
    state so the shared landscape can separate evidence-rich productive
    traffic from evidence-poor looping.
    """
    trajectories = []
    with open(parsed_path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    if not trajectories:
        return None

    # Extract key sets from all steps across all rollouts
    all_key_sets = []
    slice_metadata = []

    for traj in trajectories:
        run_id = traj["run_id"]
        model_id = traj["model_id"]
        steps = traj.get("steps", [])
        n_steps = len(steps)

        # Run-level state for the search enrichment
        run_domains: set[str] = set()
        prior_queries: list[set[str]] = []

        for step in steps:
            step_idx = step["step_idx"]
            progress = step_idx / max(1, n_steps - 1) if n_steps > 1 else 0.0

            keys = extract_cxcmu_slice_keys(step, progress)
            if not keys:
                continue

            if benchmark == "search":
                # Domains harvested from the observation text
                obs_domains = _extract_domains(step.get("raw_observation", ""))
                run_domains.update(obs_domains)
                for d in obs_domains:
                    keys.add(f"URL_DOMAIN:{d}")

                # Query novelty based on prior queries in this run
                cur_q_tokens = _extract_query_tokens(step.get("raw_action", ""))
                if cur_q_tokens or step.get("action_type") == "search":
                    novelty = _query_novelty_label(cur_q_tokens, prior_queries)
                    keys.add(f"QUERY_NOVELTY:{novelty}")
                    if cur_q_tokens:
                        prior_queries.append(cur_q_tokens)

                # Cumulative evidence breadth bucket (updated each step)
                keys.add(f"EVIDENCE_COUNT:{_evidence_bucket(len(run_domains))}")

            all_key_sets.append(keys)
            slice_metadata.append({
                "run_id": run_id,
                "model_id": model_id,
                "slice_idx": len(all_key_sets) - 1,
                "step_idx": step_idx,
                "progress": round(progress, 4),
            })

    if len(all_key_sets) < 3:
        return None

    # Cap nodes to avoid memory issues
    if len(all_key_sets) > MAX_NODES:
        indices = np.random.RandomState(42).choice(
            len(all_key_sets), MAX_NODES, replace=False,
        )
        indices = sorted(indices)
        all_key_sets = [all_key_sets[i] for i in indices]
        slice_metadata = [slice_metadata[i] for i in indices]

    # Compute IDF weights
    idf = build_idf_weights(all_key_sets)

    # Compute pairwise distances
    distances = compute_pairwise_distances(all_key_sets, idf)

    # Compute kNN
    k = min(NEIGHBOR_K, len(all_key_sets) - 1)
    if k < 1:
        return None
    knn_indices, knn_dists = compute_knn(distances, k)

    return {
        "key_sets": all_key_sets,
        "idf": idf,
        "slice_metadata": slice_metadata,
        "knn_indices": knn_indices,
        "knn_dists": knn_dists,
        "n_slices": len(all_key_sets),
    }


def main(max_tasks: int | None = None, benchmark: str | None = None):
    benchmarks = sorted(d.name for d in PARSED_DIR.iterdir() if d.is_dir())
    if benchmark:
        benchmarks = [b for b in benchmarks if b == benchmark]

    if not benchmarks:
        print(f"No benchmarks found in {PARSED_DIR}")
        return

    print(f"Processing benchmarks: {benchmarks}")
    total_processed = 0

    for bench in benchmarks:
        bench_parsed = PARSED_DIR / bench
        bench_sig = SIG_DIR / bench
        bench_sig.mkdir(parents=True, exist_ok=True)

        parsed_files = sorted(bench_parsed.glob("*.jsonl"))
        if max_tasks is not None:
            parsed_files = parsed_files[:max_tasks]

        print(f"\n── {bench}: {len(parsed_files)} tasks ──")

        for parsed_path in tqdm(parsed_files, desc=f"  Extracting {bench}"):
            task_id = parsed_path.stem
            task_dir = bench_sig / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            result = process_task(parsed_path, benchmark=bench)
            if result is None:
                continue

            # Save artifacts
            with open(task_dir / "key_sets.pkl", "wb") as f:
                pickle.dump(result["key_sets"], f)

            with open(task_dir / "idf_weights.json", "w") as f:
                json.dump(result["idf"], f)

            with open(task_dir / "slice_metadata.json", "w") as f:
                json.dump(result["slice_metadata"], f)

            np.save(task_dir / "knn_indices.npy", result["knn_indices"])
            np.save(task_dir / "knn_dists.npy", result["knn_dists"])
            total_processed += 1

        print(f"  Processed: {total_processed} tasks with signatures")

    print(f"\nTotal tasks with signatures: {total_processed}")
    print(f"Output directory: {SIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the data root (default: data/cxcmu). "
             "Expects <data_root>/parsed and writes to <data_root>/signatures.",
    )
    args = parser.parse_args()
    if args.data_root is not None:
        _set_data_root(Path(args.data_root))
    main(max_tasks=args.max_tasks, benchmark=args.benchmark)
