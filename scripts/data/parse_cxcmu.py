#!/usr/bin/env python3
"""Parse cx-cmu/agent_trajectories messages into UnifiedTrajectory format.

For each record, extracts tool_calls from assistant messages and pairs them
with tool responses to build UnifiedStep sequences.

Per-benchmark classification rules:
- swebench: str_replace_editor(view) → read, execute_bash(pytest) → bash/pytest
- tau2bench: airline_get_* → search/read, airline_book_* → other(mutate)
- terminalbench: execute_bash(...) → bash/command_class
- search: tool_name direct, query in arguments
- mcpbench: tool_name direct

Output: data/cxcmu/parsed/{benchmark}/{task_id}.jsonl
  Each task has all rollouts (across models, passes) in one file.

Usage:
    python scripts/data/parse_cxcmu.py [--max-tasks N]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

RAW_DIR = Path("data/cxcmu/raw")
PARSED_DIR = Path("data/cxcmu/parsed")

# ── Command classification ──────────────────────────────────────────

BASH_COMMANDS = {
    "pytest": "pytest", "python": "python", "grep": "grep", "find": "find",
    "sed": "sed", "cat": "cat", "pip": "pip", "git": "git", "cd": "cd",
    "ls": "ls", "echo": "echo", "mkdir": "mkdir", "rm": "rm", "curl": "curl",
    "awk": "awk", "sort": "sort", "head": "head", "tail": "tail",
    "wc": "wc", "diff": "diff", "chmod": "chmod", "mv": "mv", "cp": "cp",
    "tar": "tar", "unzip": "unzip", "wget": "wget", "npm": "npm",
    "node": "node", "java": "java", "javac": "javac", "gcc": "gcc",
    "make": "make", "cargo": "cargo", "go": "go", "ruby": "ruby",
}

# ── Observation signature patterns ──────────────────────────────────

OBS_PATTERNS = [
    (re.compile(r"error|Error|ERROR|exception|Exception|EXCEPTION", re.IGNORECASE), "error"),
    (re.compile(r"Traceback \(most recent call last\)"), "traceback"),
    (re.compile(r"PASSED|passed|PASS|pass", re.IGNORECASE), "test_passed"),
    (re.compile(r"FAILED|failed|FAIL|fail", re.IGNORECASE), "test_failed"),
    (re.compile(r"SyntaxError"), "syntax_error"),
    (re.compile(r"ImportError|ModuleNotFoundError"), "import_error"),
    (re.compile(r"FileNotFoundError|No such file"), "file_not_found"),
    (re.compile(r"permission denied|PermissionError", re.IGNORECASE), "permission_error"),
    (re.compile(r"timeout|TimeoutError|Timeout", re.IGNORECASE), "timeout"),
    (re.compile(r"success|Success|SUCCESS|completed", re.IGNORECASE), "success"),
]

# ── Per-benchmark tool classification ───────────────────────────────

# tau2bench
TAU_EVIDENCE_PREFIXES = ("get_", "search_", "list_", "look", "check", "find", "view")
TAU_MUTATE_PREFIXES = ("book_", "cancel_", "update_", "create_", "delete_", "set_",
                       "modify_", "change_", "add_", "remove_")
TAU_FINALIZE = {"transfer_to_human_agents", "submit", "finish", "done"}

# swebench tools
SWE_READ_TOOLS = {"str_replace_editor__view", "view_file", "read_file"}
SWE_EDIT_TOOLS = {"str_replace_editor__create", "str_replace_editor__str_replace",
                  "str_replace_editor__insert", "edit_file", "write_file",
                  "str_replace_editor"}
SWE_BASH_TOOL = {"execute_bash", "bash", "terminal"}
SWE_SUBMIT = {"submit", "finish"}


@dataclass
class UnifiedStep:
    step_idx: int
    action_type: str  # bash|edit|read|search|submit|other
    command_class: str  # pytest|grep|cat|sed|python|git|other
    tool_name: str
    files_touched: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    observation_signature: List[str] = field(default_factory=list)
    raw_action: str = ""
    raw_observation: str = ""


@dataclass
class UnifiedTrajectory:
    task_id: str
    model_id: str
    scaffold_id: str
    run_id: str
    resolved: bool
    n_steps: int
    steps: List[UnifiedStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def classify_bash_command(cmd: str) -> str:
    """Classify a bash command string into a command class."""
    cmd_stripped = cmd.strip().lstrip("!")
    # Get first token
    first = cmd_stripped.split()[0] if cmd_stripped.split() else ""
    # Strip path
    first = first.rsplit("/", 1)[-1]
    return BASH_COMMANDS.get(first, "other")


def extract_obs_signatures(text: str) -> List[str]:
    """Extract observation signature patterns from response text."""
    if not text:
        return []
    sigs = []
    for pattern, label in OBS_PATTERNS:
        if pattern.search(text[:2000]):
            sigs.append(label)
    return sigs


def extract_file_paths(text: str) -> List[str]:
    """Extract file paths from text."""
    if not text:
        return []
    # Match common file path patterns
    paths = re.findall(r'(?:^|[\s"\'(,])(/[a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)', text[:3000])
    paths += re.findall(r'(?:^|[\s"\'(,])([a-zA-Z0-9_./]+\.[a-zA-Z]{1,6})', text[:3000])
    # Deduplicate and filter
    seen = set()
    result = []
    for p in paths:
        p = p.strip()
        if p and p not in seen and len(p) > 2 and "/" in p:
            seen.add(p)
            result.append(p)
    return result[:10]  # cap


def _strip_bench_prefix(name: str, benchmark: str) -> str:
    """Strip benchmark prefix from tool names.

    e.g. 'swebench__swebench_str_replace_editor' → 'str_replace_editor'
         'terminalbench__terminalbench_execute_bash' → 'execute_bash'
         'search__web_search' → 'web_search'
    """
    # Strip '{bench}__' prefix
    prefixes = [
        f"{benchmark}__{benchmark}_",
        f"{benchmark}__",
        f"{benchmark}_",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _get_tool_call_info(tool_call: dict, benchmark: str = "") -> tuple[str, str, str]:
    """Extract (tool_name, arguments_str, full_name) from a tool_call."""
    func = tool_call.get("function", tool_call)
    name = func.get("name", "")
    full_name = name
    args = func.get("arguments", "")
    if isinstance(args, dict):
        args = json.dumps(args)
    # Strip benchmark prefix for classification
    name = _strip_bench_prefix(name, benchmark)
    return name, args, full_name


def classify_swebench(tool_name: str, args_str: str) -> tuple[str, str]:
    """Classify a swebench tool call → (action_type, command_class)."""
    tn_lower = tool_name.lower()

    # Check for view command in str_replace_editor
    if "str_replace_editor" in tn_lower:
        if "view" in args_str.lower():
            return "read", "cat"
        return "edit", "sed"

    if any(t in tn_lower for t in ("execute_bash", "bash", "terminal")):
        cmd_class = classify_bash_command(args_str)
        return "bash", cmd_class

    if any(t in tn_lower for t in ("read", "view", "cat")):
        return "read", "cat"

    if any(t in tn_lower for t in ("submit", "finish")):
        return "submit", "other"

    if any(t in tn_lower for t in ("write", "edit", "create", "insert", "replace")):
        return "edit", "sed"

    return "other", "other"


def classify_tau2bench(tool_name: str, args_str: str) -> tuple[str, str]:
    """Classify a tau2bench tool call → (action_type, command_class)."""
    tn_lower = tool_name.lower()

    if tn_lower in TAU_FINALIZE or any(f in tn_lower for f in ("transfer", "submit", "finish")):
        return "submit", "other"

    if any(tn_lower.startswith(p) for p in TAU_EVIDENCE_PREFIXES):
        return "search", "other"

    if any(tn_lower.startswith(p) for p in TAU_MUTATE_PREFIXES):
        return "other", "other"  # mutate

    if "think" in tn_lower or "calculate" in tn_lower:
        return "search", "other"

    return "other", "other"


def classify_terminalbench(tool_name: str, args_str: str) -> tuple[str, str]:
    """Classify a terminalbench tool call → (action_type, command_class)."""
    tn_lower = tool_name.lower()

    if any(t in tn_lower for t in ("bash", "terminal", "execute", "shell")):
        cmd_class = classify_bash_command(args_str)
        if cmd_class in ("cat", "head", "tail", "less"):
            return "read", cmd_class
        return "bash", cmd_class

    if any(t in tn_lower for t in ("submit", "finish")):
        return "submit", "other"

    return "bash", classify_bash_command(args_str)


def classify_search_bench(tool_name: str, args_str: str) -> tuple[str, str]:
    """Classify a search benchmark tool call → (action_type, command_class)."""
    tn_lower = tool_name.lower()

    if any(t in tn_lower for t in ("search", "query", "google", "bing", "browse")):
        return "search", "other"

    if any(t in tn_lower for t in ("click", "navigate", "open", "visit")):
        return "read", "other"

    if any(t in tn_lower for t in ("submit", "finish", "answer")):
        return "submit", "other"

    return "search", "other"


def classify_mcpbench(tool_name: str, args_str: str) -> tuple[str, str]:
    """Classify an MCP benchmark tool call → (action_type, command_class)."""
    tn_lower = tool_name.lower()

    if any(t in tn_lower for t in ("read", "get", "list", "search", "find", "view")):
        return "read", "other"

    if any(t in tn_lower for t in ("write", "create", "update", "delete", "set", "edit")):
        return "edit", "other"

    if any(t in tn_lower for t in ("execute", "run", "bash", "shell")):
        cmd_class = classify_bash_command(args_str)
        return "bash", cmd_class

    if any(t in tn_lower for t in ("submit", "finish")):
        return "submit", "other"

    return "other", "other"


BENCHMARK_CLASSIFIERS = {
    "swebench": classify_swebench,
    "tau2bench": classify_tau2bench,
    "terminalbench": classify_terminalbench,
    "search": classify_search_bench,
    "mcpbench": classify_mcpbench,
}


def parse_record(
    record: dict,
    benchmark: str,
    task_max_reward: Optional[float] = None,
) -> Optional[UnifiedTrajectory]:
    """Parse a single cx-cmu record into a UnifiedTrajectory.

    For mcpbench, reward is a continuous multi-criteria sum (range 0..~8) and
    97% of records have reward > 0, so a > 0 threshold makes the split
    near-ceiling.  The analysis therefore uses the per-task max-normalised
    reward stored in ``metadata.resolved_score``; the binary ``resolved`` flag
    (reward >= 0.5 * task_max_reward) is kept only as block-level bookkeeping.
    Other benchmarks use their binary 0/1 reward with the > 0 rule.
    """
    messages = record.get("messages", [])
    if not messages:
        return None

    task_id = str(record.get("task_id", record.get("instance_id", "unknown")))
    model_id = record.get("source_model", record.get("model", "unknown"))
    pass_num = record.get("pass", record.get("pass_id", 0))
    run_id = f"{model_id}__pass{pass_num}"

    reward = record.get("reward", 0) or 0.0
    if benchmark == "mcpbench":
        tmax = task_max_reward if task_max_reward and task_max_reward > 0 else None
        if tmax is None:
            # Degenerate task (all zeros): fall back to > 0 rule
            resolved = bool(reward > 0)
            resolved_score = 0.0
        else:
            resolved_score = float(reward) / tmax
            resolved = bool(reward >= 0.5 * tmax)
    else:
        # Binary-reward benchmarks: keep original rule
        resolved = bool(reward and reward > 0)
        resolved_score = 1.0 if resolved else 0.0

    classifier = BENCHMARK_CLASSIFIERS.get(benchmark, classify_mcpbench)

    steps: List[UnifiedStep] = []
    step_idx = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                # Check for function_call in content
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    i += 1
                    continue
                i += 1
                continue

            for tc in tool_calls:
                tool_name, args_str, full_name = _get_tool_call_info(tc, benchmark)
                if not tool_name:
                    continue

                # Get corresponding tool response
                obs_text = ""
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") in ("tool", "function"):
                        obs_text = next_msg.get("content", "")
                        if isinstance(obs_text, list):
                            obs_text = " ".join(
                                item.get("text", "") if isinstance(item, dict) else str(item)
                                for item in obs_text
                            )

                action_type, command_class = classifier(tool_name, args_str)
                obs_sigs = extract_obs_signatures(obs_text)
                files_touched = extract_file_paths(args_str) if action_type == "edit" else []
                files_read = extract_file_paths(args_str) if action_type == "read" else []

                steps.append(UnifiedStep(
                    step_idx=step_idx,
                    action_type=action_type,
                    command_class=command_class,
                    tool_name=tool_name,
                    files_touched=files_touched,
                    files_read=files_read,
                    observation_signature=obs_sigs,
                    raw_action=args_str[:500],
                    raw_observation=obs_text[:500],
                ))
                step_idx += 1

            # Skip the tool response messages
            while i + 1 < len(messages) and messages[i + 1].get("role") in ("tool", "function"):
                i += 1

        i += 1

    if not steps:
        return None

    return UnifiedTrajectory(
        task_id=task_id,
        model_id=model_id,
        scaffold_id=benchmark,
        run_id=run_id,
        resolved=resolved,
        n_steps=len(steps),
        steps=steps,
        metadata={
            "benchmark": benchmark,
            "pass": pass_num,
            "n_messages": len(messages),
            "raw_reward": float(reward),
            "resolved_score": float(resolved_score),
            "task_max_reward": float(task_max_reward) if task_max_reward is not None else None,
        },
    )


def main(max_tasks: Optional[int] = None):
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print(f"No raw files found in {RAW_DIR}")
        return

    print(f"Found {len(raw_files)} benchmark files")

    total_parsed = 0
    total_skipped = 0

    for raw_path in raw_files:
        benchmark = raw_path.stem
        print(f"\n── Processing {benchmark} ──")

        out_dir = PARSED_DIR / benchmark
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load all records
        records = []
        with open(raw_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        print(f"  Total records: {len(records)}")

        # Group by task_id
        by_task: dict[str, list] = defaultdict(list)
        for rec in records:
            tid = str(rec.get("task_id", rec.get("instance_id", "unknown")))
            by_task[tid].append(rec)

        task_ids = sorted(by_task.keys())
        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]

        bench_parsed = 0
        bench_skipped = 0

        # Pre-pass: for mcpbench, compute per-task max reward across all rollouts
        # for the max-normalised resolved_score.
        task_max_reward: dict[str, float] = {}
        if benchmark == "mcpbench":
            for tid, recs in by_task.items():
                rewards = [float(r.get("reward", 0) or 0.0) for r in recs]
                if rewards:
                    task_max_reward[tid] = max(rewards)

        for task_id in tqdm(task_ids, desc=f"  Parsing {benchmark}"):
            task_records = by_task[task_id]
            trajectories = []

            tmax = task_max_reward.get(task_id) if benchmark == "mcpbench" else None

            for rec in task_records:
                traj = parse_record(rec, benchmark, task_max_reward=tmax)
                if traj is not None:
                    trajectories.append(traj)
                else:
                    bench_skipped += 1

            if not trajectories:
                continue

            # Save all rollouts for this task
            safe_id = task_id.replace("/", "__").replace(" ", "_")
            out_path = out_dir / f"{safe_id}.jsonl"
            with open(out_path, "w") as f:
                for traj in trajectories:
                    row = {
                        "task_id": traj.task_id,
                        "model_id": traj.model_id,
                        "scaffold_id": traj.scaffold_id,
                        "run_id": traj.run_id,
                        "resolved": traj.resolved,
                        "n_steps": traj.n_steps,
                        "steps": [asdict(s) for s in traj.steps],
                        "metadata": traj.metadata,
                    }
                    f.write(json.dumps(row, default=str) + "\n")
            bench_parsed += len(trajectories)

        total_parsed += bench_parsed
        total_skipped += bench_skipped
        print(f"  Parsed: {bench_parsed} trajectories, Skipped: {bench_skipped}")
        print(f"  Tasks with data: {len(list(out_dir.glob('*.jsonl')))}")

    print(f"\n── Summary ──")
    print(f"Total parsed: {total_parsed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output: {PARSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    main(max_tasks=args.max_tasks)
