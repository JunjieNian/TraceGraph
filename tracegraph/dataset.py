"""Dataset path helpers and registry for multi-dataset support.

Provides standardized path resolution and a registry of known datasets.
All scripts use these helpers to locate data and results directories.

HuggingFace mirror: set env var HF_ENDPOINT=https://hf-mirror.com (or any
mirror URL) before running download scripts.  The ``load_from_hub`` helper
in ``00_download_data.py`` honours this variable automatically via the
``datasets`` library.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ── HuggingFace mirror (optional) ────────────────────────────────────
# If you need a HuggingFace mirror (e.g. in China), set env var HF_ENDPOINT
# before running download scripts:
#   export HF_ENDPOINT=https://hf-mirror.com


# ── Dataset registry ─────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "openhands": {
        "hub_id": "nebius/SWE-rebench-openhands-trajectories",
        "id_field": "instance_id",
        "resolved_field": "resolved",
        "trajectory_field": "trajectory",
    },
    "swe_smith": {
        "hub_id": "SWE-bench/SWE-smith-trajectories",
        "id_field": "instance_id",
        "resolved_field": "resolved",
        "trajectory_field": "trajectory",
    },
    "swe_agent": {
        "hub_id": "nebius/SWE-agent-trajectories",
        "id_field": "instance_id",
        "resolved_field": "resolved",
        "trajectory_field": "trajectory",
    },
}


# ── Path helpers ─────────────────────────────────────────────────────

def get_data_dirs(dataset: str) -> Dict[str, Path]:
    """Return standardized data directory paths for a dataset."""
    base = Path(f"data/{dataset}")
    return {
        "hf_raw": base / "hf_raw",
        "raw": base / "raw",
        "parsed": base / "parsed",
        "signatures": base / "signatures",
        "graphs": base / "graphs",
    }


def get_results_dir(dataset: str, experiment: str) -> Path:
    """Return results directory for a dataset + experiment."""
    return Path(f"results/{dataset}/{experiment}")


def add_dataset_arg(parser) -> None:
    """Add the standard --dataset CLI argument to an argparse parser."""
    parser.add_argument(
        "--dataset", type=str, default="openhands",
        help="Dataset name (default: openhands)",
    )


# ── Parsed per-rollout outcomes ──────────────────────────────────────

def load_parsed_outcomes(jsonl_path) -> Dict[str, Dict[str, object]]:
    """Return {run_str: {model_id, resolved_score, raw_reward, ...}} per row.

    ``run_str`` follows the ``{model_id}__pass{pass_tag}`` convention used in
    ``unique_runs`` / ``run_id_map`` of the graph payloads.  ``resolved_score``
    is the binary flag on the four binary splits and the per-task
    max-normalised reward on continuous-reward splits such as MCPBench.
    """
    import json
    from pathlib import Path as _Path

    jsonl_path = _Path(jsonl_path)
    out: Dict[str, Dict[str, object]] = {}
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
            run_id_str = row.get("run_id")
            md = row.get("metadata", {}) or {}
            pass_tag = md.get("pass")
            if pass_tag is None and run_id_str is not None:
                pass_tag = run_id_str
            if run_id_str is not None:
                key = run_id_str
            elif pass_tag is not None:
                pass_tag_str = str(pass_tag)
                key = (
                    f"{model_id}__{pass_tag_str}"
                    if pass_tag_str.startswith("pass")
                    else f"{model_id}__pass{pass_tag_str}"
                )
            else:
                key = None
            score = md.get("resolved_score")
            if score is None:
                score = 1.0 if row.get("resolved") else 0.0
            if key is not None:
                out[key] = {
                    "model_id": model_id,
                    "scaffold_id": scaffold,
                    "resolved_score": float(score),
                    "raw_reward": md.get("raw_reward"),
                    "task_max_reward": md.get("task_max_reward"),
                    "run_id_str": run_id_str,
                }
    return out
