#!/usr/bin/env python3
"""Evaluate SWE-bench Verified patches from a pilot JSONL via the official
swebench harness.

Pipeline:
  1. Read pilot_swe_*.jsonl rollouts (each row has instance_id, arm, patch).
  2. For each arm, write a predictions.jsonl in the standard SWE-bench format:
       {instance_id, model_name_or_path, model_patch}
  3. Run `python -m swebench.harness.run_evaluation` per arm via docker.
  4. Parse the resulting `*.json` report; merge `resolved` and per-instance
     reward info back into a results file per arm.

Usage:
  python scripts/109_eval_swe_patches.py \\
      --pilot-glob 'results/cxcmu/intervention/pilot_swe_postfix_s*.jsonl' \\
      --out-dir results/cxcmu/intervention/swe_eval \\
      --max-workers 4 \\
      --arms placebo tracegraph shuffled
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_pilot(globs: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in globs:
        for p in sorted(glob.glob(g)):
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def write_predictions(rows: List[Dict[str, Any]], arm: str, model_tag: str,
                      out_path: Path) -> int:
    """Write a SWE-bench-format predictions file for one arm. Returns the
    number of predictions written (skips rows that errored or have no
    instance_id; empty patches ARE included as that's the harness convention)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        for r in rows:
            if r.get("arm") != arm:
                continue
            iid = r.get("instance_id")
            if not iid:
                continue
            patch = r.get("patch") or ""
            rec = {
                "instance_id": iid,
                "model_name_or_path": model_tag,
                "model_patch": patch,
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def run_harness(predictions_path: Path, run_id: str,
                dataset: str = "princeton-nlp/SWE-bench_Verified",
                max_workers: int = 4,
                namespace: str = "none",
                cache_level: str = "instance",
                log_path: Optional[Path] = None) -> int:
    """Invoke swebench.harness.run_evaluation; return exit code."""
    env = os.environ.copy()
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", str(predictions_path),
        "--dataset_name", dataset,
        "--run_id", run_id,
        "--max_workers", str(max_workers),
        "--namespace", namespace,
        "--cache_level", cache_level,
    ]
    if log_path is None:
        proc = subprocess.run(cmd, env=env)
        return proc.returncode
    with open(log_path, "ab") as logf:
        logf.write(f"\n== {' '.join(shlex.quote(c) for c in cmd)} ==\n".encode())
        logf.flush()
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        return proc.returncode


def find_report(model_tag: str, run_id: str) -> Optional[Path]:
    """Locate the harness output report file.

    swebench.harness.run_evaluation writes <model>.<run_id>.json at the CWD
    of the python process (or sometimes inside logs/run_evaluation/). We
    search the obvious candidates.
    """
    candidates = [
        Path(f"{model_tag}.{run_id}.json"),
        Path("logs") / "run_evaluation" / run_id / model_tag / "report.json",
        Path("logs") / "run_evaluation" / run_id / f"{model_tag}.{run_id}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last-resort: scan for .json files that mention this model+run_id.
    for p in Path(".").rglob(f"{model_tag}.{run_id}.json"):
        return p
    return None


def find_per_instance_reports(run_id: str) -> List[Path]:
    base = Path("logs") / "run_evaluation" / run_id / run_id
    if not base.exists():
        return []
    return sorted(base.glob("*/report.json"))


def parse_report(report_path: Path) -> Dict[str, Any]:
    """Parse SWE-bench report. The standard format is one of:
       - {model_tag: {instance_id: {resolved: bool, ...}}}
       - {instance_id: {resolved: bool, ...}}  (older)
       - {"resolved_instances": [...], "unresolved_instances": [...], ...}
    Returns a dict instance_id → {resolved: bool, ...}.
    """
    with open(report_path) as fh:
        data = json.load(fh)
    out: Dict[str, Any] = {}

    # Standard summary format (preferred)
    if isinstance(data, dict) and "resolved_ids" in data:
        for iid in data.get("resolved_ids", []) or []:
            out[iid] = {"resolved": True}
        for iid in data.get("unresolved_ids", []) or []:
            out.setdefault(iid, {})["resolved"] = False
        for iid in data.get("empty_patch_ids", []) or []:
            out.setdefault(iid, {})["resolved"] = False
            out[iid]["empty_patch"] = True
        for iid in data.get("error_ids", []) or []:
            out.setdefault(iid, {})["resolved"] = False
            out[iid]["error"] = True
        return out

    # Per-instance dict format
    if isinstance(data, dict):
        # one-layer: {iid: {resolved: ...}}
        for iid, info in data.items():
            if isinstance(info, dict) and "resolved" in info:
                out[iid] = {"resolved": bool(info["resolved"]), **{k: v for k, v in info.items() if k != "resolved"}}
        if out:
            return out
        # nested: {model: {iid: {resolved: ...}}}
        for model, per_iid in data.items():
            if isinstance(per_iid, dict):
                for iid, info in per_iid.items():
                    if isinstance(info, dict) and "resolved" in info:
                        out[iid] = {"resolved": bool(info["resolved"]), **{k: v for k, v in info.items() if k != "resolved"}}
                if out:
                    return out

    return out


def parse_per_instance_reports(report_paths: Sequence[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for report_path in report_paths:
        try:
            data = json.loads(report_path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for iid, info in data.items():
            if isinstance(info, dict) and "resolved" in info:
                out[iid] = {
                    "resolved": bool(info.get("resolved", False)),
                    **{k: v for k, v in info.items() if k != "resolved"},
                    "report_path": str(report_path),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-glob", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["placebo", "tracegraph", "shuffled"])
    ap.add_argument("--out-dir", default="results/cxcmu/intervention/swe_eval")
    ap.add_argument("--run-id-prefix", default="tracegraph_pilot")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Parallel swebench docker workers per arm.")
    ap.add_argument("--harness-timeout-sec", type=int, default=180,
                    help="Kill a harness subprocess if it still hasn't exited after this many "
                         "seconds. Per-instance `report.json` files are still harvested.")
    ap.add_argument("--parallel-arms", action="store_true",
                    help="Run arms in parallel (3× docker pressure).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If the arm's resolved-info file already exists, skip the harness call.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pilot(args.pilot_glob)
    print(f"Loaded {len(rows)} pilot rows", flush=True)
    if not rows:
        sys.exit("No rollouts to evaluate.")

    # Per-arm prediction files
    arm_pred_paths: Dict[str, Path] = {}
    arm_tags: Dict[str, str] = {}
    for arm in args.arms:
        tag = f"{args.run_id_prefix}_{arm}"
        pred_path = out_dir / f"predictions_{arm}.jsonl"
        n = write_predictions(rows, arm, tag, pred_path)
        arm_pred_paths[arm] = pred_path
        arm_tags[arm] = tag
        print(f"  {arm}: wrote {n} predictions → {pred_path}", flush=True)

    # Run harness per arm
    arm_results: Dict[str, Dict[str, Any]] = {}
    procs: Dict[str, subprocess.Popen] = {}

    def kick_arm(arm: str) -> Optional[subprocess.Popen]:
        tag = arm_tags[arm]
        pred = arm_pred_paths[arm]
        run_id = tag
        out_info = out_dir / f"resolved_{arm}.json"
        if args.skip_existing and out_info.exists():
            print(f"  [{arm}] skip (existing {out_info})", flush=True)
            return None
        log_path = out_dir / f"harness_{arm}.log"
        log_path.write_bytes(b"")
        env = os.environ.copy()
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--predictions_path", str(pred),
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--run_id", run_id,
            "--max_workers", str(args.max_workers),
            "--namespace", "none",
            "--cache_level", "instance",
        ]
        print(f"  [{arm}] launching harness, log={log_path}", flush=True)
        logf = open(log_path, "ab")
        return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)

    if args.parallel_arms:
        for arm in args.arms:
            procs[arm] = kick_arm(arm)
        for arm, p in procs.items():
            if p is None:
                continue
            try:
                rc = p.wait(timeout=args.harness_timeout_sec if args.harness_timeout_sec > 0 else None)
            except subprocess.TimeoutExpired:
                print(f"  [{arm}] harness timeout after {args.harness_timeout_sec}s; terminating", flush=True)
                p.terminate()
                try:
                    rc = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = p.wait(timeout=10)
            print(f"  [{arm}] harness exit={rc}", flush=True)
    else:
        for arm in args.arms:
            p = kick_arm(arm)
            if p is None:
                continue
            try:
                rc = p.wait(timeout=args.harness_timeout_sec if args.harness_timeout_sec > 0 else None)
            except subprocess.TimeoutExpired:
                print(f"  [{arm}] harness timeout after {args.harness_timeout_sec}s; terminating", flush=True)
                p.terminate()
                try:
                    rc = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = p.wait(timeout=10)
            print(f"  [{arm}] harness exit={rc}", flush=True)

    # Locate reports + parse
    for arm in args.arms:
        tag = arm_tags[arm]
        report = find_report(tag, tag)
        per_iid: Dict[str, Any] = {}
        report_paths: List[str] = []
        if report is not None:
            per_iid = parse_report(report)
            report_paths.append(str(report))
        if not per_iid:
            inst_reports = find_per_instance_reports(tag)
            if inst_reports:
                per_iid = parse_per_instance_reports(inst_reports)
                report_paths.extend(str(p) for p in inst_reports)
        if not per_iid:
            print(f"  [{arm}] WARNING: no report found", flush=True)
            arm_results[arm] = {}
            continue
        arm_results[arm] = per_iid
        out_info = out_dir / f"resolved_{arm}.json"
        with open(out_info, "w") as fh:
            json.dump({"report_path": str(report) if report is not None else None,
                       "report_paths": report_paths,
                       "instances": per_iid}, fh, indent=2)
        n_resolved = sum(1 for v in per_iid.values() if v.get("resolved"))
        n_total = len(per_iid)
        print(f"  [{arm}] resolved {n_resolved}/{n_total} → {out_info}", flush=True)

    # Merge resolved status back into a final per-rollout JSONL
    merged_path = out_dir / "pilot_swe_with_resolved.jsonl"
    with open(merged_path, "w") as fh:
        for r in rows:
            arm = r.get("arm")
            iid = r.get("instance_id")
            info = arm_results.get(arm, {}).get(iid, {})
            r2 = dict(r)
            r2["resolved"] = bool(info.get("resolved", False))
            r2["resolved_info"] = info
            # outcome (reward = 1 if resolved else 0) for downstream paired Δ
            r2["reward"] = 1.0 if r2["resolved"] else 0.0
            r2["success"] = int(r2["resolved"])
            fh.write(json.dumps(r2) + "\n")
    print(f"\nMerged {len(rows)} rows with resolved status → {merged_path}", flush=True)


if __name__ == "__main__":
    main()
