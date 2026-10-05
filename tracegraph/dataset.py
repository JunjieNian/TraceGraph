"""Loader for the parsed per-rollout outcomes written by
``scripts/data/parse_cxcmu.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_parsed_outcomes(jsonl_path) -> Dict[str, Dict[str, object]]:
    """Return {run_str: {model_id, resolved_score, raw_reward, ...}} per row.

    ``run_str`` follows the ``{model_id}__pass{pass_tag}`` convention used in
    ``unique_runs`` / ``run_id_map`` of the graph payloads.  ``resolved_score``
    is the binary flag on the four binary splits and the per-task
    max-normalised reward on continuous-reward splits such as MCPBench.
    """
    jsonl_path = Path(jsonl_path)
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
