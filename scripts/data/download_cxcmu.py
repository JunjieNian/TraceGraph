#!/usr/bin/env python3
"""Download cx-cmu/agent_trajectories from HuggingFace.

Downloads 5 benchmark JSONL files (excluding mathhay which has no tool_calls)
and saves them as data/cxcmu/raw/{benchmark}.jsonl.

The dataset provides per-benchmark JSONL files directly, so we download them
individually using hf_hub_download.

Models: DeepSeek-R1, DeepSeek-V3.2, Gemini-2.5-Flash, Qwen3-235B, Qwen3-Next
Benchmarks: tau2bench, swebench, terminalbench, search, mcpbench

Usage:
    python scripts/data/download_cxcmu.py

Set HF_TOKEN if the dataset requires gated access. Set HF_ENDPOINT only if you need a mirror.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Optional: set HF_ENDPOINT before running if you need a HuggingFace mirror.

REPO_ID = "cx-cmu/agent_trajectories"
BENCHMARKS = ["tau2bench", "swebench", "terminalbench", "search", "mcpbench"]
OUTPUT_DIR = Path("data/cxcmu/raw")


def main(repo_id: str = REPO_ID, benchmarks: list[str] | None = None, output_dir: Path = OUTPUT_DIR):
    from huggingface_hub import hf_hub_download

    benchmarks = list(benchmarks or BENCHMARKS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    print(f"Downloading from: {repo_id}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"Benchmarks: {benchmarks}")
    print()

    total_records = 0
    for bench in benchmarks:
        filename = f"{bench}.jsonl"
        print(f"Downloading {filename}...")

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Copy to output directory
        out_path = output_dir / filename
        shutil.copy2(local_path, out_path)

        # Count records
        n_records = 0
        models = set()
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    n_records += 1
                    try:
                        rec = json.loads(line)
                        models.add(rec.get("source_model", rec.get("model", "unknown")))
                    except json.JSONDecodeError:
                        pass

        total_records += n_records
        print(f"  {bench}: {n_records} records, {len(models)} models")
        for m in sorted(models):
            print(f"    - {m}")

    print(f"\nTotal: {total_records} records across {len(benchmarks)} benchmarks")
    print(f"Output directory: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(repo_id=args.repo_id, benchmarks=args.benchmarks, output_dir=args.output_dir)
