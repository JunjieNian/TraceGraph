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
