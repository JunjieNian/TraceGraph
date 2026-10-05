#!/usr/bin/env python3
"""SWE-bench Verified prefix-fork recovery runner.

Builds on the bundled TraceGraph SWE agent runtime, a MiniSWEAgent-style
multi-step bash agent.

For each instance a probe rollout runs at the base temperature with the
trap detector active.  At the first step where the detector fires, the
Docker workspace and the message prefix are snapshotted, and every arm
continues from that identical state:

  tg_baseline     (Baseline)  T=0.6, no note
  tg_hot          (Hot)       only the temperature changes, to T=0.9
  tg_repair_cool  (Note)      T=0.6, diagnosis note appended at the fork

All arms share top-p=0.95 and the same total step budget.  A trigger fires
when the IDF-weighted Jaccard similarity to the historical trap library
reaches --sim-threshold and the warmup, cooldown, intent, and file-cue gates
pass; the trap-versus-reference margin is logged per step and never gates
firing.  Defaults reproduce the settings of the paper.

The trap / reference libraries, IDF corpus, and diagnosis sidecar come from
`data/cxcmu/intervention/` when rebuilt locally, or otherwise from the
bundled `resources/swebench_detector/`.

Outputs one JSONL row per (instance_id, arm, seed) plus a sidecar JSONL with
the saved trigger prefixes.  Each row carries the final patch; score the
patches with `scripts/intervention/eval_patches.py`.

Usage:
  python scripts/intervention/swe_runner.py \\
      --instances django__django-11066 \\
      --model-base-url http://localhost:8000/v1 --model-name <served-model> \\
      --output results/cxcmu/intervention/example.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- repo-local imports ---
from tracegraph.signature import extract_observation_keys, weighted_jaccard

# --- bundled MiniSWEAgent-style runtime ---
from tracegraph.sweagent import (
    Agent,
    AgentResult,
    DockerEnvironment,
    ModelResponse,
    StepRecord,
    VLLMModel,
    make_observation_message,
    make_system_message,
    make_user_message,
    parse_response,
)


import openai


class DeepSeekModel:
    """Lightweight OpenAI-compatible model shim for DeepSeek (or any
    OpenAI-compatible endpoint that requires a real API key and prefers
    NOT to return token-level logprobs).

    Exposes a `query(messages)` method returning `ModelResponse`, matching the
    interface expected by the bundled TraceGraph SWE agent.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-v4-pro",
        api_key: str,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        request_timeout: int = 180,
    ):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self.model_name = model_name
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = max_tokens
        self.thinking_enabled = bool(thinking_enabled)

    def query(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,  # ignored; DeepSeek charges for logprobs
    ) -> ModelResponse:
        t0 = time.time()
        kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature if temperature is None else float(temperature),
            top_p=self.top_p if top_p is None else float(top_p),
        )
        eff_max = self.max_tokens if max_tokens is None else max_tokens
        if eff_max is not None:
            kwargs["max_tokens"] = int(eff_max)
        extra_body: Dict[str, Any] = {}
        if not self.thinking_enabled:
            extra_body["thinking"] = {"type": "disabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**kwargs)
        inference_time = time.time() - t0

        choice = response.choices[0]
        content = choice.message.content or ""

        usage: Dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }
        return ModelResponse(
            content=content,
            logprobs=None,
            request_id=getattr(response, "id", None),
            usage=usage,
            inference_time=inference_time,
            finish_reason=getattr(choice, "finish_reason", None),
            raw_logprobs=None,
        )

INT_DIR = Path("data/cxcmu/intervention")
TRAP_PATH = INT_DIR / "swebench_trap_library.json"
CORE_PATH = INT_DIR / "swebench_core_library.json"
IDF_PATH = INT_DIR / "swebench_idf_corpus.json"
# Per-family pattern diagnosis that fills the fifth slot of the Note arm.
# Built by scripts/intervention/build_swe_trap_diagnosis.py.
TRAP_DIAGNOSIS_PATH = INT_DIR / "swebench_trap_diagnosis.json"
RESOURCE_INT_DIR = Path(__file__).resolve().parents[2] / "resources" / "swebench_detector"

DEFAULT_OUT = Path("results/cxcmu/intervention/swe_prefix_fork.jsonl")

GENERIC_KEY_PREFIXES = ("PHASE:",)
GENERIC_KEYS = {"ACTION:other"}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    note_type: str          # "repair" | "none"
    bump_temp: bool         # True: continue at --hot-temperature instead of --temperature


ARM_SPECS: Dict[str, ArmSpec] = {
    "tg_baseline":    ArmSpec("tg_baseline",    "none",   bump_temp=False),  # Baseline
    "tg_hot":         ArmSpec("tg_hot",         "none",   bump_temp=True),   # Hot
    "tg_repair_cool": ArmSpec("tg_repair_cool", "repair", bump_temp=False),  # Note
}
ARM_CHOICES = tuple(ARM_SPECS.keys())
ARM_DISPLAY_ORDER = ["tg_baseline", "tg_hot", "tg_repair_cool"]
MAX_TRIGGERS = 1   # at most one trigger per rollout; the fork happens there


def parse_arm(arm: str) -> ArmSpec:
    try:
        return ARM_SPECS[arm]
    except KeyError as exc:
        raise ValueError(f"Unsupported arm: {arm}") from exc


# ── diagnosis note ─────────────────────────────────────────────────
# Four slots ({ctx_command}, {ctx_signal}, {ctx_files}, {ctx_confidence})
# come from the agent's own per-step log; the fifth ({ctx_trap_pattern}) is
# the family diagnosis stored with the best-matching historical trap.

NOTE_HEADER = "[INTERNAL DIAGNOSTIC — not visible to graders]"

_NOTE_TEMPLATE = (
    "{header}\n"
    "A trajectory pattern previously associated with low task-resolution rates\n"
    "fired at this step. Concretely:\n"
    "\n"
    "  - Last command: {ctx_command}\n"
    "  - Last error/test signal: {ctx_signal}\n"
    "  - File(s) touched in this region: {ctx_files}\n"
    "  - Detector confidence (trap similarity): {ctx_confidence}\n"
    "\n"
    "  Trap pattern (from prior failed trajectories with this signature):\n"
    "  {ctx_trap_pattern}\n"
    "\n"
    "Before continuing:\n"
    "  1. Re-read the failing test/traceback for the specific assertion or\n"
    "     unexpected value (do not rely on memory of earlier steps).\n"
    "  2. Localize to the smallest function or class implicated by that\n"
    "     evidence, in the file(s) above.\n"
    "  3. Propose ONE minimal change consistent with the evidence; do not\n"
    "     rewrite unrelated code.\n"
    "  4. Run the narrowest relevant test or check before submitting.\n"
    "  5. If your current patch is not supported by the error/test evidence,\n"
    "     revise or discard it.\n"
    "\n"
    "Respond in the normal THOUGHT / ACTION format with exactly one bash\n"
    "command."
)


def render_note(
    *,
    trigger_record: Optional[Dict[str, Any]] = None,
    recent_action: Optional[str] = None,
    recent_obs: Optional[str] = None,
    trap_diagnosis_text: Optional[str] = None,
) -> str:
    """Populate the diagnosis note from the trigger step.

    Only text the agent itself produced or observed fills the slots, plus
    the pre-specified family diagnosis; no gold patch, verified-test list,
    or grader signal is used.
    """
    slots = _repair_slots(trigger_record, recent_action, recent_obs,
                          trap_diagnosis_text)
    return _NOTE_TEMPLATE.format(header=NOTE_HEADER, **slots)


# ── repair-note slot rendering ────────────────────────────────────


_REPAIR_TRUNC_CMD = 200
_REPAIR_TRUNC_SIGNAL = 300
_FILE_PATH_KEY_PREFIX = "FILE_PATH:"


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text or "(empty)"
    return text[: max(0, limit - 1)].rstrip() + "…"


def _extract_error_summary(observation: str, limit: int = _REPAIR_TRUNC_SIGNAL) -> str:
    """Extract a short error/test summary from the agent's last observation.

    Operates only on text the agent itself just observed; never reads gold
    patches or the verified-test list.
    """
    obs = (observation or "").strip()
    if not obs:
        return "(no observation captured)"
    # Prefer a line containing an obvious failure cue; fall back to the
    # tail of the observation (which is where pytest puts its summary).
    cue_pat = re.compile(
        r"^(.*?(Traceback|FAILED|ERROR|AssertionError|Error|Exception|"
        r"not found|denied|invalid).*)$",
        re.I | re.M,
    )
    m = cue_pat.search(obs[:6000])
    if m:
        return _truncate(m.group(1), limit)
    # Fallback: last non-empty line of the observation window.
    tail = obs[-1500:]
    last_lines = [ln for ln in tail.splitlines() if ln.strip()]
    if last_lines:
        return _truncate(last_lines[-1], limit)
    return _truncate(obs, limit)


def _top_file_paths_from_trigger(
    trigger_record: Optional[Dict[str, Any]],
    fallback_action: Optional[str],
    fallback_obs: Optional[str],
    *,
    k: int = 3,
) -> List[str]:
    """Pull up to k FILE_PATH:* values from the trigger record's keys.

    These keys were generated by `encode_step` from the agent's own command
    + observation, so they reveal nothing the agent has not already seen.
    Falls back to re-extracting from the action / observation text if the
    trigger record is missing (e.g. tests).
    """
    seen: List[str] = []
    if trigger_record:
        for k_ in (trigger_record.get("trigger_keys") or []):
            if isinstance(k_, str) and k_.startswith(_FILE_PATH_KEY_PREFIX):
                path = k_[len(_FILE_PATH_KEY_PREFIX):]
                if path and path not in seen:
                    seen.append(path)
                    if len(seen) >= k:
                        return seen
    for blob in (fallback_action or "", (fallback_obs or "")[:2000]):
        for m in _FILE_PATH_RE.findall(blob):
            if m and m not in seen:
                seen.append(m)
                if len(seen) >= k:
                    return seen
    return seen


_FALLBACK_TRAP_PATTERN = (
    "(no pattern diagnosis available for this trap signature) Treat "
    "this as a generic low-yield action signal: re-check whether the "
    "next planned action is the smallest change actually supported by "
    "current evidence; if not, narrow the plan or revert speculative "
    "edits before continuing."
)


def _repair_slots(
    trigger_record: Optional[Dict[str, Any]],
    recent_action: Optional[str],
    recent_obs: Optional[str],
    trap_diagnosis_text: Optional[str] = None,
) -> Dict[str, str]:
    files = _top_file_paths_from_trigger(trigger_record, recent_action, recent_obs)
    files_str = ", ".join(files) if files else "(none localized yet)"
    sim_trap = None
    if trigger_record:
        try:
            sim_trap = float(trigger_record.get("sim_trap"))
        except (TypeError, ValueError):
            sim_trap = None
    conf_str = f"{sim_trap:.2f}" if sim_trap is not None else "n/a"
    diag = (trap_diagnosis_text or "").strip() or _FALLBACK_TRAP_PATTERN
    return {
        "ctx_command": _truncate(recent_action or "(no command recorded)", _REPAIR_TRUNC_CMD),
        "ctx_signal": _extract_error_summary(recent_obs or ""),
        "ctx_files": files_str,
        "ctx_confidence": conf_str,
        "ctx_trap_pattern": diag,
    }


# ── trap-library loading ──────────────────────────────────────────


def strip_generic_keys(keys: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for k in keys:
        if any(k.startswith(p) for p in GENERIC_KEY_PREFIXES):
            continue
        if k in GENERIC_KEYS:
            continue
        out.add(k)
    return out


def _resource_or_data_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = RESOURCE_INT_DIR / path.name
    if fallback.exists():
        return fallback
    return path


def load_libraries() -> Tuple[List[Set[str]], List[Set[str]], Dict[str, float]]:
    """Load the trap library, the core-side reference library, and the IDF corpus."""
    trap_path = _resource_or_data_path(TRAP_PATH)
    core_path = _resource_or_data_path(CORE_PATH)
    idf_path = _resource_or_data_path(IDF_PATH)
    if not trap_path.exists():
        raise SystemExit(
            f"Missing {TRAP_PATH} and bundled {RESOURCE_INT_DIR / TRAP_PATH.name}; "
            "run scripts/intervention/build_swe_trap_library.py first"
        )
    trap_raw = json.load(open(trap_path))["examples"]
    trap = [
        set(ex.get("detection_keys") or strip_generic_keys(ex.get("keys", [])))
        for ex in trap_raw
    ]
    core_raw = json.load(open(core_path))["examples"] if core_path.exists() else []
    core = [
        set(ex.get("detection_keys") or strip_generic_keys(ex.get("keys", [])))
        for ex in core_raw
    ]
    if not idf_path.exists():
        raise SystemExit(
            f"Missing {IDF_PATH} and bundled {RESOURCE_INT_DIR / IDF_PATH.name}; "
            "run scripts/intervention/build_swe_trap_library.py first"
        )
    idf = json.load(open(idf_path))
    return trap, core, idf


def max_similarity(query: Set[str], library: Sequence[Set[str]], idf: Dict[str, float]) -> float:
    if not query or not library:
        return 0.0
    best = 0.0
    for ref in library:
        s = weighted_jaccard(query, ref, idf)
        if s > best:
            best = s
    return best


def max_similarity_with_index(
    query: Set[str], library: Sequence[Set[str]], idf: Dict[str, float]
) -> Tuple[float, int]:
    """Same as max_similarity but also returns the best-matching library
    index (or -1 if no library / no overlap). Used to look up the
    per-trap family diagnosis for the Note arm.
    """
    if not query or not library:
        return (0.0, -1)
    best = 0.0
    best_idx = -1
    for i, ref in enumerate(library):
        s = weighted_jaccard(query, ref, idf)
        if s > best:
            best = s
            best_idx = i
    return (best, best_idx)


def load_trap_diagnosis() -> Dict[int, str]:
    """Load the `swebench_trap_diagnosis.json` sidecar and return a dict
    mapping trap example index → pattern-family diagnosis text.
    Returns {} with a warning if the sidecar is missing; the Note arm then
    falls back to a generic low-yield-action pattern text."""
    diag_path = _resource_or_data_path(TRAP_DIAGNOSIS_PATH)
    if not diag_path.exists():
        print(f"WARNING: no trap diagnosis sidecar at {TRAP_DIAGNOSIS_PATH} "
              f"or {RESOURCE_INT_DIR / TRAP_DIAGNOSIS_PATH.name}", file=sys.stderr)
        return {}
    try:
        data = json.loads(diag_path.read_text())
    except Exception as exc:
        print(f"WARNING: failed to load {diag_path}: {exc}", file=sys.stderr)
        return {}
    families = data.get("families") or {}
    by_idx = data.get("by_trap_index") or {}
    out: Dict[int, str] = {}
    for k, fk in by_idx.items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        fam = families.get(fk) or {}
        diag = fam.get("diagnosis")
        if isinstance(diag, str) and diag.strip():
            out[i] = diag.strip()
    return out


# ── step encoding ─────────────────────────────────────────────────

_FILE_PATH_RE = re.compile(r"[\w./-]+\.(?:py|rst|md|c|h|cpp|txt|cfg|toml|yaml|yml|json)")
_FILE_EXT_RE = re.compile(r"\.(\w{1,5})$")
_CMD_TOKEN_RE = re.compile(r"^\s*([a-zA-Z_][\w-]*)")


def _classify_first_cmd(bash_cmd: str) -> str:
    m = _CMD_TOKEN_RE.match(bash_cmd or "")
    if not m:
        return "other"
    token = m.group(1).lower()
    if token in {"pytest", "py.test"}:
        return "pytest"
    if token == "grep" or token == "rg":
        return "grep"
    if token in {"cat", "head", "tail", "less"}:
        return "cat"
    if token == "sed":
        return "sed"
    if token in {"python", "python3", "py"}:
        return "python"
    if token == "git":
        return "git"
    if token in {"ls", "find", "tree"}:
        return "find"
    if token in {"awk"}:
        return "awk"
    if token in {"echo"}:
        return "echo"
    return "other"


def _intent_from_cmd(bash_cmd: str) -> str:
    s = (bash_cmd or "").lower()
    if any(kw in s for kw in (" >>", " > ", "<<eof", "heredoc", "tee ")):
        return "edit"
    if s.startswith(("sed ", "awk ")):
        return "edit"
    if any(kw in s for kw in ("cat ", "less ", "head ", "tail ", "view ")):
        return "read"
    if any(kw in s for kw in ("ls ", "find ", "tree ")):
        return "read"
    if any(kw in s for kw in ("grep ", "rg ")):
        return "search"
    if any(kw in s for kw in ("pytest", "unittest", "py.test", "python -m ", "python3 -m ")):
        return "test"
    return "other"


_HARD_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|FAILED|FAILED tests|ERROR collecting|"
    r"AssertionError|command not found|No such file or directory|Permission denied|"
    r"pytest.+failed|=+ FAILURES =+",
    re.I,
)
_PYTHONISH_ERROR_RE = re.compile(r"\b(?:Exception|Error|Failure|failed)\b", re.I)


def hard_error_tag(bash_cmd: str, observation: str) -> bool:
    """Conservative post-error detector for real execution failures.

    Avoids firing on source-code reads that merely contain words like
    `ValueError` or `error`.
    """
    cmd_class = _classify_first_cmd(bash_cmd)
    text = observation if isinstance(observation, str) else str(observation or "")
    snippet = text[:4000]

    if _HARD_ERROR_RE.search(snippet):
        return True
    if cmd_class in {"python", "pytest"} and _PYTHONISH_ERROR_RE.search(snippet):
        return True
    return False


def _is_check_action(action: str, intent: str, cmd_class: str) -> bool:
    s = (action or "").lower()
    return (
        intent == "test"
        or cmd_class in {"pytest", "python"}
        or "python -c" in s
        or "python3 -c" in s
    )


def encode_step(bash_cmd: str, observation: str) -> Set[str]:
    """Encode a live SWE agent step into the detector key alphabet.

    The detector libraries use TOOL: / ACTION: / CMD: / FILE_PATH: /
    FILE_EXT: / OBS:* keys. We synthesize TOOL:execute_bash + ACTION:<intent>
    + CMD:<classified> and file cues, then add OBS keys from the
    observation text.
    """
    bash_cmd = bash_cmd or ""
    obs_text = observation if isinstance(observation, str) else str(observation or "")
    keys: Set[str] = set()

    keys.add("TOOL:execute_bash")
    intent = _intent_from_cmd(bash_cmd)
    if intent != "other":
        keys.add(f"ACTION:{intent}")
    cmd_class = _classify_first_cmd(bash_cmd)
    if cmd_class != "other":
        keys.add(f"CMD:{cmd_class}")

    # File path / extension cues from the bash command (and observation).
    for blob in (bash_cmd, obs_text[:1000]):
        for m in _FILE_PATH_RE.findall(blob):
            keys.add(f"FILE_PATH:{m}")
        for m in _FILE_PATH_RE.finditer(blob):
            ext = _FILE_EXT_RE.search(m.group(0))
            if ext:
                keys.add(f"FILE_EXT:{ext.group(1)}")

    # Regex-based OBS keys (exception names, test/traceback/success cues).
    # These match on any observation text, including source code the agent
    # has just read, not only on runtime tracebacks.
    try:
        keys |= extract_observation_keys({"content": obs_text[:4000]})
    except Exception:
        pass

    # Generic English error detector for common SWE failure phrases that the
    # regex families above do not catch.
    if hard_error_tag(bash_cmd, obs_text):
        keys.add("OBS:hard_error")
    if _english_error_tag(obs_text):
        keys.add("OBS:error")
    return keys


_GENERIC_OBS_ERROR_PATS = [
    re.compile(r"\berror\b", re.I),
    re.compile(r"\btraceback\b", re.I),
    re.compile(r"FAILED", re.I),
    re.compile(r"AssertionError|ImportError|TypeError|ValueError|AttributeError|KeyError|IndexError|FileNotFoundError|RuntimeError|SyntaxError|NameError", re.I),
    re.compile(r"\bnot\s+found\b", re.I),
    re.compile(r"command not found", re.I),
    re.compile(r"\bdenied\b", re.I),
    re.compile(r"\binvalid\b", re.I),
]


def _english_error_tag(text: str) -> bool:
    if not isinstance(text, str):
        return False
    snippet = text[:4000]
    return any(p.search(snippet) for p in _GENERIC_OBS_ERROR_PATS)


# ── intervention agent ────────────────────────────────────────────


def _safe_frac(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return round(float(num) / float(den), 4)


def _step_intent(step: Dict[str, Any]) -> str:
    intent = step.get("intent")
    if intent:
        return str(intent)
    if step.get("action") == "submit":
        return "submit"
    return "other"


def _step_cmd_class(step: Dict[str, Any]) -> str:
    cmd_class = step.get("cmd_class")
    if cmd_class:
        return str(cmd_class)
    if step.get("action") == "submit":
        return "submit"
    return "other"


def _step_is_check(step: Dict[str, Any]) -> bool:
    return _is_check_action(
        str(step.get("action") or ""),
        _step_intent(step),
        _step_cmd_class(step),
    )


def summarize_trigger_behavior(per_step_log: Sequence[Dict[str, Any]], *, window: int = 2) -> Dict[str, Any]:
    trigger_positions = [idx for idx, st in enumerate(per_step_log) if st.get("triggered")]
    n_trigger_steps = len(trigger_positions)
    trigger_intent_counts: Dict[str, int] = {}
    trigger_cmd_class_counts: Dict[str, int] = {}
    n_trigger_steps_obs_error = 0
    n_trigger_steps_obs_hard_error = 0
    n_trigger_steps_is_trap = 0
    n_trigger_steps_obs_error_and_is_trap = 0
    n_with_check = 0
    n_with_test = 0
    n_with_read = 0
    n_with_edit = 0
    n_with_submit = 0
    trigger_step_values: List[float] = []

    for pos in trigger_positions:
        step = per_step_log[pos]
        intent = _step_intent(step)
        cmd_class = _step_cmd_class(step)
        trigger_intent_counts[intent] = trigger_intent_counts.get(intent, 0) + 1
        trigger_cmd_class_counts[cmd_class] = trigger_cmd_class_counts.get(cmd_class, 0) + 1

        obs_has_error = bool(step.get("obs_has_error"))
        obs_has_hard_error = bool(step.get("obs_has_hard_error"))
        is_trap = bool(step.get("is_trap"))
        n_trigger_steps_obs_error += int(obs_has_error)
        n_trigger_steps_obs_hard_error += int(obs_has_hard_error)
        n_trigger_steps_is_trap += int(is_trap)
        n_trigger_steps_obs_error_and_is_trap += int(obs_has_error and is_trap)
        if step.get("step") is not None:
            try:
                trigger_step_values.append(float(step["step"]))
            except Exception:
                pass

        future_steps = per_step_log[pos + 1: pos + 1 + max(1, int(window))]
        future_intents = {_step_intent(st) for st in future_steps}
        n_with_check += int(any(_step_is_check(st) for st in future_steps))
        n_with_test += int("test" in future_intents)
        n_with_read += int("read" in future_intents)
        n_with_edit += int("edit" in future_intents)
        n_with_submit += int("submit" in future_intents)

    mean_trigger_step = None
    if trigger_step_values:
        mean_trigger_step = round(sum(trigger_step_values) / len(trigger_step_values), 2)

    return {
        "n_trigger_steps": int(n_trigger_steps),
        "n_trigger_steps_obs_error": int(n_trigger_steps_obs_error),
        "n_trigger_steps_obs_hard_error": int(n_trigger_steps_obs_hard_error),
        "n_trigger_steps_is_trap": int(n_trigger_steps_is_trap),
        "n_trigger_steps_obs_error_and_is_trap": int(n_trigger_steps_obs_error_and_is_trap),
        "trigger_obs_error_frac": _safe_frac(n_trigger_steps_obs_error, n_trigger_steps),
        "trigger_obs_hard_error_frac": _safe_frac(n_trigger_steps_obs_hard_error, n_trigger_steps),
        "trigger_is_trap_frac": _safe_frac(n_trigger_steps_is_trap, n_trigger_steps),
        "trigger_obs_error_and_is_trap_frac": _safe_frac(
            n_trigger_steps_obs_error_and_is_trap, n_trigger_steps
        ),
        "trigger_intent_counts": trigger_intent_counts,
        "trigger_cmd_class_counts": trigger_cmd_class_counts,
        "mean_trigger_step": mean_trigger_step,
        "n_triggers_with_check_within_2_steps": int(n_with_check),
        "n_triggers_with_test_within_2_steps": int(n_with_test),
        "n_triggers_with_read_within_2_steps": int(n_with_read),
        "n_triggers_with_edit_within_2_steps": int(n_with_edit),
        "n_triggers_with_submit_within_2_steps": int(n_with_submit),
        "check_within_2_steps_after_trigger_frac": _safe_frac(n_with_check, n_trigger_steps),
        "test_within_2_steps_after_trigger_frac": _safe_frac(n_with_test, n_trigger_steps),
        "read_within_2_steps_after_trigger_frac": _safe_frac(n_with_read, n_trigger_steps),
        "edit_within_2_steps_after_trigger_frac": _safe_frac(n_with_edit, n_trigger_steps),
        "submit_within_2_steps_after_trigger_frac": _safe_frac(n_with_submit, n_trigger_steps),
    }


@dataclass
class InterventionRunState:
    messages: List[Dict[str, str]]
    steps: List[StepRecord] = field(default_factory=list)
    next_step_index: int = 0
    n_triggers: int = 0
    n_trigger_candidates: int = 0
    cd_left: int = 0
    visited_trap: int = 0
    visited_core: int = 0
    first_trap_step: Optional[int] = None
    per_step_log: List[Dict[str, Any]] = field(default_factory=list)
    tool_error_count: int = 0
    trigger_records: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "InterventionRunState":
        return InterventionRunState(
            messages=copy.deepcopy(self.messages),
            steps=list(self.steps),
            next_step_index=self.next_step_index,
            n_triggers=self.n_triggers,
            n_trigger_candidates=self.n_trigger_candidates,
            cd_left=self.cd_left,
            visited_trap=self.visited_trap,
            visited_core=self.visited_core,
            first_trap_step=self.first_trap_step,
            per_step_log=copy.deepcopy(self.per_step_log),
            tool_error_count=self.tool_error_count,
            trigger_records=copy.deepcopy(self.trigger_records),
        )


class SnapshotDockerEnvironment(DockerEnvironment):
    """DockerEnvironment with optional reset skipping for fork continuations."""

    def __init__(
        self,
        *,
        image: str,
        container_name: str,
        timeout: int = 30,
        max_output_chars: int = 10000,
        reset_to_base: bool = True,
        base_commit_override: Optional[str] = None,
    ):
        super().__init__(
            image=image,
            container_name=container_name,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        self.reset_to_base = bool(reset_to_base)
        self.base_commit_override = base_commit_override

    def start(self) -> None:
        if self._started:
            return

        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.container_name,
                self.image,
                "tail", "-f", "/dev/null",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self._started = True
        if self.base_commit_override is not None:
            self._base_commit = self.base_commit_override
        else:
            out = self.execute("cd /testbed && git rev-parse HEAD~1", timeout=10)
            self._base_commit = out.strip().split("\n")[0] if out.strip() else None

        if self.reset_to_base and self._base_commit:
            self.execute(f"cd /testbed && git checkout {self._base_commit}", timeout=10)


def _docker_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", value.lower())
    return slug.strip("-") or "snapshot"


def snapshot_env_state(env: DockerEnvironment, snapshot_name: str) -> Dict[str, Any]:
    snapshot_image = f"tracegraph-prefixfork:{_docker_slug(snapshot_name)}"
    result = subprocess.run(
        ["docker", "commit", env.container_name, snapshot_image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker commit failed: {result.stderr}")
    return {
        "image": snapshot_image,
        "base_commit": getattr(env, "_base_commit", None),
        "git_status": env.execute("cd /testbed && git status --short", timeout=10),
        "prefix_patch": env.get_patch(),
    }


def cleanup_snapshot_image(image: Optional[str]) -> None:
    if not image:
        return
    subprocess.run(["docker", "rmi", "-f", image], capture_output=True)


class InterventionAgent(Agent):
    """TraceGraph SWE agent with the step-level trap detector.

    `run()` executes the agent loop, scores every step against the trap and
    reference libraries, and records triggers subject to the warmup,
    cooldown, intent, file-cue, and optional observation-key gates.  The
    probe rollout stops at the first trigger; forked continuations resume
    from the saved state.
    """

    def __init__(
        self,
        model,
        env,
        arm: str,
        trap_lib: List[Set[str]],
        core_lib: List[Set[str]],
        idf: Dict[str, float],
        sim_threshold: float = 0.35,
        margin: float = 0.03,
        cooldown: int = 4,
        warmup_steps: int = 2,
        require_obs_keys: Optional[Sequence[str]] = None,
        max_steps: int = 30,
        allowed_intents: Optional[Sequence[str]] = ("edit", "submit"),
    ):
        super().__init__(model=model, env=env, hooks=[], max_steps=max_steps)
        self.arm = arm
        self.arm_spec = parse_arm(arm)
        self.note_type = self.arm_spec.note_type
        self.trap_lib = trap_lib
        self.core_lib = core_lib
        self.idf = idf
        self.sim_threshold = float(sim_threshold)
        self.margin = float(margin)
        self.cooldown = int(cooldown)
        self.max_triggers = MAX_TRIGGERS
        self.warmup_steps = int(warmup_steps)
        self.require_obs_keys = set(require_obs_keys or [])
        self.allowed_intents = set(allowed_intents) if allowed_intents else None

    def run(
        self,
        problem_statement: Optional[str] = None,
        *,
        initial_messages: Optional[List[Dict[str, str]]] = None,
        initial_state: Optional[InterventionRunState] = None,
        stop_on_first_trigger: bool = False,
        return_state: bool = False,
    ):
        if initial_state is None:
            if initial_messages is not None:
                messages = copy.deepcopy(initial_messages)
            else:
                if problem_statement is None:
                    raise ValueError("problem_statement required when no initial messages/state are provided")
                messages = [make_system_message(), make_user_message(problem_statement)]
            state = InterventionRunState(messages=messages)
        else:
            state = initial_state.clone()
            if initial_messages is not None:
                state.messages = copy.deepcopy(initial_messages)

        messages = state.messages
        exit_reason = "max_steps"

        s = state.next_step_index
        while s < self.max_steps:
            try:
                response = self.model.query(messages)
            except Exception as e:
                exit_reason = f"model_error: {e}"
                break

            thinking, thought, action_cmd = parse_response(response.content)
            if not action_cmd:
                observation = (
                    "[ERROR] Could not parse an ACTION from your response. "
                    "Please respond with THOUGHT: and ACTION: sections, "
                    "with the action in a ```bash code block."
                )
                messages.append({"role": "assistant", "content": response.content})
                messages.append(make_observation_message(observation))
                state.per_step_log.append({
                    "step": s,
                    "action": "",
                    "no_action": True,
                    "intent": "parse_error",
                    "cmd_class": "parse_error",
                    "triggered": False,
                    "observation_preview": observation[:200],
                })
                state.next_step_index = s + 1
                s += 1
                continue

            if action_cmd.strip().lower() == "submit":
                exit_reason = "submit"
                state.steps.append(StepRecord(
                    step_index=s,
                    thinking=thinking,
                    thought=thought,
                    action="submit",
                    observation="",
                    response=response,
                ))
                state.per_step_log.append({
                    "step": s,
                    "action": "submit",
                    "intent": "submit",
                    "cmd_class": "submit",
                    "triggered": False,
                    "obs_has_error": False,
                    "obs_has_hard_error": False,
                    "is_trap": False,
                    "is_core": False,
                })
                state.next_step_index = s + 1
                break

            observation = self.env.execute(action_cmd)

            keys = encode_step(action_cmd, observation)
            det_keys = strip_generic_keys(keys)
            sim_trap, sim_trap_idx = max_similarity_with_index(det_keys, self.trap_lib, self.idf)
            sim_core = max_similarity(det_keys, self.core_lib, self.idf) if self.core_lib else 0.0
            obs_has_error = "OBS:error" in det_keys
            obs_has_hard_error = "OBS:hard_error" in det_keys
            if obs_has_error:
                state.tool_error_count += 1
            # Bookkeeping only: the trap-versus-reference margin never gates firing.
            is_trap = sim_trap >= self.sim_threshold and (sim_trap - sim_core) >= self.margin
            is_core = sim_core >= self.sim_threshold and (sim_core - sim_trap) >= self.margin
            if is_trap:
                state.visited_trap += 1
                if state.first_trap_step is None:
                    state.first_trap_step = s
            if is_core:
                state.visited_core += 1

            warmup_ok = s >= self.warmup_steps
            obs_gate_ok = (not self.require_obs_keys) or bool(self.require_obs_keys & det_keys)
            step_intent = _intent_from_cmd(action_cmd)
            intent_ok = (self.allowed_intents is None) or (step_intent in self.allowed_intents)
            file_cue_ok = any(k.startswith("FILE_PATH:") for k in det_keys)
            gates_ok = warmup_ok and obs_gate_ok and intent_ok and file_cue_ok
            triggered = False
            if (
                gates_ok
                and state.cd_left == 0
                and state.n_triggers < self.max_triggers
                and sim_trap >= self.sim_threshold
            ):
                triggered = True
                state.n_triggers += 1
                state.n_trigger_candidates += 1
                state.cd_left = self.cooldown
            elif gates_ok and sim_trap >= self.sim_threshold:
                state.n_trigger_candidates += 1
                state.cd_left = max(0, state.cd_left - 1)
            else:
                state.cd_left = max(0, state.cd_left - 1)

            messages.append({"role": "assistant", "content": response.content})
            messages.append(make_observation_message(observation))

            state.steps.append(StepRecord(
                step_index=s,
                thinking=thinking,
                thought=thought,
                action=action_cmd,
                observation=observation,
                response=response,
            ))
            step_log = {
                "step": s,
                "action": action_cmd,
                "cmd_class": _classify_first_cmd(action_cmd),
                "intent": step_intent,
                "sim_trap": round(float(sim_trap), 4),
                "sim_core": round(float(sim_core), 4),
                "obs_has_error": bool(obs_has_error),
                "obs_has_hard_error": bool(obs_has_hard_error),
                "is_trap": bool(is_trap),
                "is_core": bool(is_core),
                "triggered": bool(triggered),
                "n_obs_chars": len(observation or ""),
                "raw_assistant": (response.content if response is not None else ""),
                "observation": observation or "",
            }
            state.per_step_log.append(step_log)
            if triggered:
                state.trigger_records.append({
                    "trigger_step": s,
                    "trigger_keys": sorted(det_keys),
                    "sim_trap": round(float(sim_trap), 4),
                    "sim_trap_idx": int(sim_trap_idx),
                    "sim_core": round(float(sim_core), 4),
                    "obs_has_error": bool(obs_has_error),
                    "obs_has_hard_error": bool(obs_has_hard_error),
                    "is_trap": bool(is_trap),
                    "is_core": bool(is_core),
                })

            state.next_step_index = s + 1

            if triggered and stop_on_first_trigger:
                exit_reason = "prefix_fork_trigger"
                break
            s += 1

        patch = ""
        try:
            patch = self.env.get_patch()
        except Exception:
            pass

        result = AgentResult(steps=state.steps, patch=patch, exit_reason=exit_reason)
        trigger_meta = summarize_trigger_behavior(state.per_step_log, window=2)
        meta = {
            "arm": self.arm,
            "note_type": self.note_type,
            "exit_reason": exit_reason,
            "n_steps": len(state.steps),
            "n_triggers": int(state.n_triggers),
            "n_trigger_candidates": int(state.n_trigger_candidates),
            "visited_trap": int(state.visited_trap),
            "visited_core": int(state.visited_core),
            "first_trap_step": state.first_trap_step,
            "tool_error_count": int(state.tool_error_count),
            "patch_chars": len(patch),
            "per_step": state.per_step_log,
            "trigger_records": state.trigger_records,
            **trigger_meta,
        }
        if return_state:
            return result, meta, state
        return result, meta


# ── dataset + driver ─────────────────────────────────────────────


def load_swebench_verified(max_instances: Optional[int], instance_ids: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """Load SWE-bench Verified test split.

    Prefers the normal Hugging Face loader, but falls back to the locally
    cached Arrow shard when the shared home cache is readable but not writable
    under sandboxing.
    """
    from datasets import Dataset, load_dataset
    ds = None
    last_exc: Optional[Exception] = None
    for attempt, env_overrides in enumerate([
        {},
        {"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
    ]):
        old_env = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)
        try:
            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            break
        except Exception as exc:
            last_exc = exc
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    if ds is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "princeton-nlp___swe-bench_verified"
        arrow_candidates = sorted(cache_root.glob("default/*/*/swe-bench_verified-test.arrow"))
        for arrow_path in reversed(arrow_candidates):
            try:
                ds = Dataset.from_file(str(arrow_path))
                print(f"Loaded SWE-bench_Verified from cached Arrow: {arrow_path}", flush=True)
                break
            except Exception as exc:
                last_exc = exc
    if ds is None:
        raise SystemExit(
            "Could not load SWE-bench_Verified. "
            f"Last error: {last_exc}"
        )
    instances = list(ds)
    if instance_ids:
        s = set(instance_ids)
        instances = [x for x in instances if x["instance_id"] in s]
    if max_instances:
        instances = instances[:max_instances]
    return instances


def load_completed(out_path: Path) -> Set[Tuple[str, str, int]]:
    done: Set[Tuple[str, str, int]] = set()
    if not out_path.exists():
        return done
    with open(out_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("instance_id"), r.get("arm"), int(r.get("seed", 0))))
    return done


def _ordered_arms(arms: Sequence[str]) -> List[str]:
    rank = {arm: idx for idx, arm in enumerate(ARM_DISPLAY_ORDER)}
    return sorted(arms, key=lambda arm: (rank.get(arm, 999), arm))


def _container_slug(instance_id: str) -> str:
    return instance_id.replace("/", "_").replace("__", "_")


def _instance_image(instance_id: str) -> str:
    return f"sweb.eval.x86_64.{instance_id.lower()}:latest"


def _row_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "sim_threshold": args.sim_threshold,
        "margin": args.margin,
        "cooldown": args.cooldown,
        "max_triggers": MAX_TRIGGERS,
        "warmup_steps": args.warmup_steps,
        "require_obs_keys": list(args.require_obs_keys or []),
        "allowed_intents": list(args.allowed_intents or []),
        "require_file_cue": True,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "hot_temperature": args.hot_temperature,
        "top_p": args.top_p,
    }


def _write_jsonl_row(fh, row: Dict[str, Any]) -> None:
    fh.write(json.dumps(row) + "\n")
    fh.flush()


def build_agent(
    *,
    args: argparse.Namespace,
    model: Any,
    env: Any,
    arm: str,
    trap_lib: List[Set[str]],
    core_lib: List[Set[str]],
    idf: Dict[str, float],
) -> InterventionAgent:
    return InterventionAgent(
        model=model,
        env=env,
        arm=arm,
        trap_lib=trap_lib,
        core_lib=core_lib,
        idf=idf,
        sim_threshold=args.sim_threshold,
        margin=args.margin,
        cooldown=args.cooldown,
        warmup_steps=args.warmup_steps,
        require_obs_keys=args.require_obs_keys,
        max_steps=args.max_steps,
        allowed_intents=args.allowed_intents,
    )


def build_rollout_row(
    *,
    instance_id: str,
    arm: str,
    seed: int,
    duration_sec: float,
    result: Any,
    err: Optional[str],
    args: argparse.Namespace,
    meta: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "instance_id": instance_id,
        "arm": arm,
        "seed": seed,
        "duration_sec": round(duration_sec, 1),
        "exit_reason": getattr(result, "exit_reason", "?"),
        "patch": getattr(result, "patch", "") or "",
        "error": err,
        "config": _row_config(args),
        **{k: v for k, v in meta.items() if k != "per_step"},
    }
    if "per_step" in meta:
        row["per_step"] = meta["per_step"]
    if extra:
        row.update(extra)
    return row


def _prefix_probe_row(
    *,
    instance_id: str,
    seed: int,
    args: argparse.Namespace,
    base_result: Any,
    base_meta: Dict[str, Any],
    prefix_id: Optional[str],
    prefix_state: Optional[InterventionRunState],
    prefix_snapshot: Optional[Dict[str, Any]],
    probe_error: Optional[str] = None,
) -> Dict[str, Any]:
    first_trigger = ((base_meta.get("trigger_records") or [None])[0] if base_meta else None)
    row: Dict[str, Any] = {
        "record_type": "prefix_probe",
        "instance_id": instance_id,
        "seed": seed,
        "found_prefix": bool(first_trigger),
        "prefix_id": prefix_id,
        "base_exit_reason": getattr(base_result, "exit_reason", base_meta.get("exit_reason") if base_meta else None),
        "error": probe_error,
        "config": _row_config(args),
    }
    if base_meta:
        row.update({
            "n_steps": base_meta.get("n_steps"),
            "n_triggers": base_meta.get("n_triggers"),
            "n_trigger_candidates": base_meta.get("n_trigger_candidates"),
            "tool_error_count": base_meta.get("tool_error_count"),
        })
    if first_trigger:
        row.update({
            "trigger_step": first_trigger.get("trigger_step"),
            "trigger_keys": first_trigger.get("trigger_keys") or [],
            "sim_trap": first_trigger.get("sim_trap"),
            "sim_trap_idx": first_trigger.get("sim_trap_idx"),
            "sim_core": first_trigger.get("sim_core"),
            "obs_has_error": first_trigger.get("obs_has_error"),
            "obs_has_hard_error": first_trigger.get("obs_has_hard_error"),
            "is_trap": first_trigger.get("is_trap"),
            "is_core": first_trigger.get("is_core"),
        })
    if prefix_state is not None and first_trigger:
        row["messages_prefix"] = prefix_state.messages
        row["per_step_prefix"] = prefix_state.per_step_log
    if prefix_snapshot is not None:
        row["env_snapshot"] = {
            "kind": "docker_commit",
            "base_commit": prefix_snapshot.get("base_commit"),
            "git_status": prefix_snapshot.get("git_status"),
        }
        row["prefix_patch"] = prefix_snapshot.get("prefix_patch", "")
    return row


def run_prefix_fork_rollouts(
    *,
    args: argparse.Namespace,
    model: Any,
    instances: Sequence[Dict[str, Any]],
    trap_lib: List[Set[str]],
    core_lib: List[Set[str]],
    idf: Dict[str, float],
    out_path: Path,
    prefix_log_path: Path,
    done: Set[Tuple[str, str, int]],
    trap_diagnosis_map: Optional[Dict[int, str]] = None,
) -> int:
    written = 0
    t0 = time.time()
    # The probe never injects anything; its arm label only names the agent.
    probe_arm = "tg_baseline"

    with open(out_path, "a") as rollout_fh, open(prefix_log_path, "a") as prefix_fh:
        for inst in instances:
            instance_id = inst["instance_id"]
            missing_arms = [arm for arm in args.arms if (instance_id, arm, args.seed) not in done]
            if not missing_arms:
                continue

            image_name = _instance_image(instance_id)
            base_env = SnapshotDockerEnvironment(
                image=image_name,
                container_name=f"{args.container_name_prefix}_prefixprobe_{_container_slug(instance_id)}",
                timeout=args.timeout,
                max_output_chars=args.max_output_chars,
                reset_to_base=True,
            )
            base_result = SimpleNamespace(patch="", steps=[], exit_reason="not_started")
            base_meta: Dict[str, Any] = {}
            base_state: Optional[InterventionRunState] = None
            probe_error: Optional[str] = None
            prefix_snapshot: Optional[Dict[str, Any]] = None
            prefix_id: Optional[str] = None
            first_trigger: Optional[Dict[str, Any]] = None

            try:
                base_env.start()
                probe_agent = build_agent(
                    args=args,
                    model=model,
                    env=base_env,
                    arm=probe_arm,
                    trap_lib=trap_lib,
                    core_lib=core_lib,
                    idf=idf,
                )
                base_result, base_meta, base_state = probe_agent.run(
                    inst["problem_statement"],
                    stop_on_first_trigger=True,
                    return_state=True,
                )
                first_trigger = ((base_meta.get("trigger_records") or [None])[0] if base_meta else None)
                if first_trigger:
                    prefix_id = (
                        f"{_docker_slug(instance_id)}-seed{args.seed}-"
                        f"step{first_trigger.get('trigger_step')}"
                    )
                    prefix_snapshot = snapshot_env_state(base_env, prefix_id)
            except Exception as exc:
                probe_error = str(exc)[:300]
            finally:
                try:
                    base_env.stop()
                except Exception:
                    pass

            prefix_row = _prefix_probe_row(
                instance_id=instance_id,
                seed=args.seed,
                args=args,
                base_result=base_result,
                base_meta=base_meta,
                prefix_id=prefix_id,
                prefix_state=base_state,
                prefix_snapshot=prefix_snapshot,
                probe_error=probe_error,
            )
            _write_jsonl_row(prefix_fh, prefix_row)

            if probe_error:
                print(f"[probe] {instance_id} error={probe_error}", flush=True)
                cleanup_snapshot_image(prefix_snapshot.get("image") if prefix_snapshot else None)
                continue
            if not first_trigger or base_state is None or prefix_snapshot is None:
                print(f"[probe] {instance_id} no prefix trigger found", flush=True)
                cleanup_snapshot_image(prefix_snapshot.get("image") if prefix_snapshot else None)
                continue

            try:
                for arm in missing_arms:
                    arm_spec = parse_arm(arm)
                    fork_env = SnapshotDockerEnvironment(
                        image=prefix_snapshot["image"],
                        container_name=f"{args.container_name_prefix}_prefixfork_{_container_slug(instance_id)}_{arm}",
                        timeout=args.timeout,
                        max_output_chars=args.max_output_chars,
                        reset_to_base=False,
                        base_commit_override=prefix_snapshot.get("base_commit"),
                    )
                    t_start = time.time()
                    # Save base temp/top_p so we can restore between forks. The
                    # same `model` instance is shared across probe + all forks.
                    base_temp = getattr(model, "temperature", None)
                    base_top_p = getattr(model, "top_p", None)
                    try:
                        fork_env.start()
                        fork_agent = build_agent(
                            args=args,
                            model=model,
                            env=fork_env,
                            arm=arm,
                            trap_lib=trap_lib,
                            core_lib=core_lib,
                            idf=idf,
                        )
                        # The Note arm renders the diagnosis from the trigger
                        # step's own command, observation, and keys plus the
                        # family diagnosis of the best-matching trap.
                        sim_idx = first_trigger.get("sim_trap_idx", -1)
                        try:
                            sim_idx_int = int(sim_idx) if sim_idx is not None else -1
                        except (TypeError, ValueError):
                            sim_idx_int = -1
                        diag_text = None
                        if trap_diagnosis_map and sim_idx_int >= 0:
                            diag_text = trap_diagnosis_map.get(sim_idx_int)
                        trigger_action = ""
                        trigger_obs = ""
                        try:
                            if base_state is not None and len(base_state.steps) > 0:
                                trigger_action = getattr(base_state.steps[-1], "action", "") or ""
                                trigger_obs = getattr(base_state.steps[-1], "observation", "") or ""
                        except Exception:
                            pass
                        fork_messages = copy.deepcopy(base_state.messages)
                        rendered_note = ""
                        if arm_spec.note_type == "repair":
                            rendered_note = render_note(
                                trigger_record={
                                    "trigger_step": first_trigger.get("trigger_step"),
                                    "trigger_keys": first_trigger.get("trigger_keys"),
                                    "sim_trap": first_trigger.get("sim_trap"),
                                    "sim_trap_idx": sim_idx_int,
                                    "obs_has_hard_error": first_trigger.get("obs_has_hard_error", False),
                                },
                                recent_action=trigger_action,
                                recent_obs=trigger_obs,
                                trap_diagnosis_text=diag_text,
                            )
                            fork_messages.append({
                                "role": "user",
                                "content": "[NOTE FROM SUPERVISOR] " + rendered_note,
                            })
                        # Only the Hot arm changes the temperature; top-p stays
                        # at its shared value in every arm.
                        if arm_spec.bump_temp:
                            model.temperature = float(args.hot_temperature)
                        else:
                            if base_temp is not None:
                                model.temperature = base_temp
                        if base_top_p is not None:
                            model.top_p = base_top_p
                        result, meta = fork_agent.run(
                            initial_messages=fork_messages,
                            initial_state=base_state,
                        )
                        err = None
                    except Exception as exc:
                        result = SimpleNamespace(patch="", steps=[], exit_reason=f"error: {exc}")
                        meta = {
                            "arm": arm,
                            "note_type": arm_spec.note_type,
                            "error": str(exc)[:300],
                        }
                        err = str(exc)[:300]
                    finally:
                        try:
                            fork_env.stop()
                        except Exception:
                            pass
                        # Restore base temperature/top_p so the next fork
                        # (or next instance's probe) starts at the base.
                        if base_temp is not None:
                            try: model.temperature = base_temp
                            except Exception: pass
                        if base_top_p is not None:
                            try: model.top_p = base_top_p
                            except Exception: pass

                    dur = time.time() - t_start
                    row = build_rollout_row(
                        instance_id=instance_id,
                        arm=arm,
                        seed=args.seed,
                        duration_sec=dur,
                        result=result,
                        err=err,
                        args=args,
                        meta=meta,
                        extra={
                            "prefix_id": prefix_id,
                            "prefix_trigger_step": first_trigger.get("trigger_step"),
                            "prefix_trigger_keys": first_trigger.get("trigger_keys") or [],
                            "prefix_obs_has_error": first_trigger.get("obs_has_error"),
                            "prefix_obs_has_hard_error": first_trigger.get("obs_has_hard_error"),
                            "prefix_sim_trap": first_trigger.get("sim_trap"),
                            "prefix_sim_core": first_trigger.get("sim_core"),
                            "note_has_family_diagnosis": bool(diag_text) if arm_spec.note_type == "repair" else None,
                        },
                    )
                    _write_jsonl_row(rollout_fh, row)
                    written += 1
                    done.add((instance_id, arm, args.seed))
                    elapsed = time.time() - t0
                    patch_chars = len(row.get("patch") or "")
                    print(
                        f"[{written}] {instance_id} arm={arm} prefix={prefix_id} "
                        f"exit={row['exit_reason']} steps={meta.get('n_steps', '?')} "
                        f"trig={meta.get('n_triggers', '?')}/cand={meta.get('n_trigger_candidates', '?')} "
                        f"patch={patch_chars}ch dur={dur:.0f}s elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )
            finally:
                cleanup_snapshot_image(prefix_snapshot.get("image") if prefix_snapshot else None)

    return written


def main():
    ap = argparse.ArgumentParser(
        description="Prefix-fork recovery runner (defaults match the paper settings).")
    ap.add_argument("--arms", nargs="+", default=list(ARM_DISPLAY_ORDER), choices=ARM_CHOICES)
    ap.add_argument("--instances", nargs="*", default=None,
                    help="Specific instance_ids; otherwise --start-index/--max-instances "
                         "select a slice of SWE-bench Verified.")
    ap.add_argument("--max-instances", type=int, default=500)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11,
                    help="Recorded with every row and used to name the saved prefixes.")
    # Decoding
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Base temperature of the probe and of the Baseline and Note arms.")
    ap.add_argument("--hot-temperature", type=float, default=0.9,
                    help="Temperature of the Hot arm after the fork.")
    ap.add_argument("--top-p", type=float, default=0.95, help="Shared top-p of every arm.")
    ap.add_argument("--max-steps", type=int, default=30,
                    help="Total step budget per rollout, prefix included.")
    # Detector
    ap.add_argument("--sim-threshold", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="Trap-versus-reference margin, logged per step for bookkeeping only.")
    ap.add_argument("--cooldown", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--allowed-intents", nargs="+", default=["edit", "submit"],
                    choices=["edit", "submit", "test", "read", "search", "other"],
                    help="Only fire when the current step's intent is in this set.")
    ap.add_argument("--require-obs-keys", nargs="+", default=[],
                    help="Optional observation-key gate (off by default).")
    # Model endpoint
    ap.add_argument("--api-provider", choices=["vllm", "deepseek", "glm"], default="vllm",
                    help="vllm = OpenAI-compatible local server (tracegraph.sweagent.VLLMModel); "
                         "deepseek / glm = provider APIs through the OpenAI-compatible shim.")
    ap.add_argument("--model-base-url", default=None,
                    help="Endpoint URL. Defaults: http://localhost:8000/v1 (vllm), "
                         "https://api.deepseek.com (deepseek), "
                         "https://open.bigmodel.cn/api/paas/v4 (glm).")
    ap.add_argument("--model-name", default=None,
                    help="Served model name. Defaults: deepseek-v4-pro (deepseek), glm-5.1 (glm); "
                         "required for vllm.")
    ap.add_argument("--api-key", default=None,
                    help="Provider API key. Falls back to DEEPSEEK_API_KEY / GLM_API_KEY.")
    ap.add_argument("--thinking-enabled", action="store_true",
                    help="For the deepseek / glm providers: enable thinking mode (off by default).")
    ap.add_argument("--max-model-tokens", type=int, default=4096,
                    help="Per-call max_tokens for the model.")
    # Environment
    ap.add_argument("--timeout", type=int, default=90,
                    help="Per-command timeout inside the container, in seconds.")
    ap.add_argument("--max-output-chars", type=int, default=12000)
    ap.add_argument("--container-name-prefix", default="tracegraph",
                    help="Prefix for docker container names; use one per provider to run "
                         "several providers on the same instance in parallel.")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--prefix-log-output", default=None,
                    help="Sidecar JSONL for saved trigger prefixes. "
                         "Default: <output stem>_prefixes.jsonl")
    args = ap.parse_args()
    args.arms = _ordered_arms(args.arms)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed(out_path)
    print(f"Found {len(done)} completed rollouts in {out_path}", flush=True)
    prefix_log_path = Path(args.prefix_log_output) if args.prefix_log_output else (
        out_path.with_name(f"{out_path.stem}_prefixes.jsonl")
    )
    prefix_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Prefix logs → {prefix_log_path}", flush=True)

    trap_lib, core_lib, idf = load_libraries()
    trap_diagnosis_map = load_trap_diagnosis()
    print(
        f"SWE libs | trap={len(trap_lib)} reference={len(core_lib)} "
        f"idf={len(idf)} trap_diagnosis={len(trap_diagnosis_map)}",
        flush=True,
    )

    # Build the model interface.
    if args.api_provider in ("deepseek", "glm"):
        env_var = "DEEPSEEK_API_KEY" if args.api_provider == "deepseek" else "GLM_API_KEY"
        api_key = args.api_key or os.environ.get(env_var)
        if not api_key:
            raise SystemExit(f"--api-provider {args.api_provider} requires --api-key or {env_var}")
        default_url, default_name = {
            "deepseek": ("https://api.deepseek.com", "deepseek-v4-pro"),
            "glm": ("https://open.bigmodel.cn/api/paas/v4", "glm-5.1"),
        }[args.api_provider]
        model = DeepSeekModel(
            base_url=args.model_base_url or default_url,
            model_name=args.model_name or default_name,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_model_tokens,
            thinking_enabled=args.thinking_enabled,
            request_timeout=max(60, args.timeout * 2),
        )
        print(f"Using {args.api_provider} API: {model.model_name} @ {model.client.base_url} "
              f"thinking={model.thinking_enabled}", flush=True)
    else:
        if not args.model_name:
            raise SystemExit("--api-provider vllm requires --model-name")
        base_url = args.model_base_url or "http://localhost:8000/v1"
        model = VLLMModel(
            base_url=base_url,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_logprobs=0,
            max_tokens=args.max_model_tokens,
        )
        print(f"Using VLLMModel: {model.model_name} @ {base_url}", flush=True)

    instances = load_swebench_verified(
        max_instances=None if args.instances else args.max_instances + args.start_index,
        instance_ids=args.instances,
    )
    if args.instances is None:
        instances = instances[args.start_index:args.start_index + args.max_instances]
    print(
        f"Running {len(instances)} instances in prefix-fork mode "
        f"with arms={args.arms}",
        flush=True,
    )
    written = run_prefix_fork_rollouts(
        args=args,
        model=model,
        instances=instances,
        trap_lib=trap_lib,
        core_lib=core_lib,
        idf=idf,
        out_path=out_path,
        prefix_log_path=prefix_log_path,
        done=done,
        trap_diagnosis_map=trap_diagnosis_map,
    )

    print(f"\nDone. Wrote {written} new rollouts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
