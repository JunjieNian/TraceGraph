#!/usr/bin/env python3
"""SWE-bench Verified causal intervention runner.

Builds on the bundled TraceGraph SWE agent runtime, a MiniSWEAgent-style
multi-step bash agent.
Supports two experiment designs:

  independent rollout:
    placebo:    same trigger schedule as tracegraph, generic "review carefully" note
    tracegraph: real trap-library trigger, repair-coaching note
    shuffled:   matched-size random non-trap-library trigger, repair-coaching note

  prefix-fork:
    tg_generic: stop at the first TraceGraph trigger prefix, then continue from
                the exact same prefix with a generic note
    tg_repair:  same prefix, but continue with the TraceGraph repair note

Prefix-fork removes no-trigger drift by comparing two continuations from the
same post-error state. It also supports `shuf_generic` / `shuf_repair` for a
future 2×2 trigger-source × note-type design.

The trap/core/nontrap libraries come from
`data/cxcmu/intervention/swebench_*_library.json` or the bundled fallback
resources in `resources/swebench_detector/`.

Outputs one JSONL row per (instance_id, arm, seed). Patch from the agent is
saved per-row; the SWE-bench `run_evaluation.py` harness can be invoked later
to score the patches in a batch.

Usage:
  python scripts/intervention/swe_runner.py \\
      --instances django__django-12419 astropy__astropy-8707 \\
      --arms placebo tracegraph \\
      --output results/cxcmu/intervention/pilot_swe_smoke.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
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
from tracegraph.signature import extract_slice_keys, weighted_jaccard

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
    strip_thinking,
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
            routed_experts=None,
        )

INT_DIR = Path("data/cxcmu/intervention")
TRAP_PATH = INT_DIR / "swebench_trap_library.json"
CORE_PATH = INT_DIR / "swebench_core_library.json"
NONTRAP_PATH = INT_DIR / "swebench_nontrap_library.json"
IDF_PATH = INT_DIR / "swebench_idf_corpus.json"
# Optional sidecar (RQ4 v2): per-family pattern diagnosis text to enrich
# the `repair` note when a trap fires. Built by
# scripts/_build_trap_diagnosis.py. If absent the repair note falls back
# to the local-evidence-only template.
TRAP_DIAGNOSIS_PATH = INT_DIR / "swebench_trap_diagnosis.json"
RESOURCE_INT_DIR = Path(__file__).resolve().parents[2] / "resources" / "swebench_detector"

DEFAULT_OUT = Path("results/cxcmu/intervention/pilot_swe.jsonl")

GENERIC_KEY_PREFIXES = ("PHASE:",)
GENERIC_KEYS = {"ACTION:other"}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    trigger_source: str
    note_type: str         # "generic" | "repair" | "none"
    bump_temp: bool = True  # when False, fork continues at base temperature (no post-trigger T=0.9 bump)


ARM_SPECS: Dict[str, ArmSpec] = {
    "placebo": ArmSpec("placebo", "tracegraph", "generic"),
    "tracegraph": ArmSpec("tracegraph", "tracegraph", "repair"),
    "shuffled": ArmSpec("shuffled", "shuffled", "repair"),
    "tg_generic": ArmSpec("tg_generic", "tracegraph", "generic"),
    "tg_repair": ArmSpec("tg_repair", "tracegraph", "repair"),
    "shuf_generic": ArmSpec("shuf_generic", "shuffled", "generic"),
    "shuf_repair": ArmSpec("shuf_repair", "shuffled", "repair"),
    # ---- 4-arm factorial decomposition of the intervention ----
    # tg_baseline: T stays at base (0.6), no note    → pure no-intervention continuation
    # tg_hot:      T bumps to fork (0.9), no note    → isolates temperature-bump effect alone
    # tg_generic:  T bumps to fork (0.9), generic note → +temp bump +neutral note
    # tg_repair:   T bumps to fork (0.9), repair note  → full intervention
    # Useful contrasts:
    #   tg_hot - tg_baseline     = temperature-bump effect
    #   tg_generic - tg_hot      = generic-note effect (controlled for temp bump)
    #   tg_repair - tg_generic   = repair-note content effect (the current v4 measurement)
    #   tg_repair - tg_baseline  = full intervention effect (vs no-intervention)
    "tg_baseline": ArmSpec("tg_baseline", "tracegraph", "none",   bump_temp=False),
    "tg_hot":      ArmSpec("tg_hot",      "tracegraph", "none",   bump_temp=True),
    "tg_repair_cool": ArmSpec("tg_repair_cool", "tracegraph", "repair", bump_temp=False),
    "shuf_baseline": ArmSpec("shuf_baseline", "shuffled", "none", bump_temp=False),
}
ARM_CHOICES = tuple(ARM_SPECS.keys())
LEGACY_ARM_DEFAULT = ["placebo", "tracegraph", "shuffled"]
PREFIX_FORK_ARM_DEFAULT = ["tg_baseline", "tg_hot", "tg_generic", "tg_repair"]
ARM_DISPLAY_ORDER = [
    "placebo", "tracegraph", "shuffled",
    "tg_baseline", "tg_hot", "tg_generic", "tg_repair",
    "shuf_baseline", "shuf_generic", "shuf_repair",
]


def parse_arm(arm: str) -> ArmSpec:
    try:
        return ARM_SPECS[arm]
    except KeyError as exc:
        raise ValueError(f"Unsupported arm: {arm}") from exc


def prompt_for_note_type(
    note_type: str,
    *,
    trigger_record: Optional[Dict[str, Any]] = None,
    recent_action: Optional[str] = None,
    recent_obs: Optional[str] = None,
    trap_diagnosis_text: Optional[str] = None,
) -> str:
    """Return a length-matched, structurally identical note.

    Both `repair` and `generic` notes share the same 5-block template
    (header / local-evidence context / trap-pattern diagnosis / numbered
    procedure / format reminder) so that the arm contrast isolates the
    *content* of the populated slots, not the length or layout.

    The repair note populates slots from the actual trigger record +
    agent observation AND from the per-trap pattern diagnosis sidecar.
    The generic note populates them with content-free placeholders
    (length-matched).
    """
    if note_type not in {"generic", "repair"}:
        raise ValueError(f"Unsupported note type: {note_type}")
    is_repair = (note_type == "repair")
    if is_repair:
        slots = _repair_slots(trigger_record, recent_action, recent_obs,
                              trap_diagnosis_text)
        header = TG_REPAIR_HEADER
        return _NOTE_TEMPLATE.format(header=header, **slots)
    # Generic: honor scripts/174 A/A override if set; otherwise build the
    # length-matched template from slots.
    if _PLACEBO_OVERRIDE is not None:
        return _PLACEBO_OVERRIDE
    slots = _generic_slots()
    header = PLACEBO_HEADER
    return _NOTE_TEMPLATE.format(header=header, **slots)


# ── prompts ────────────────────────────────────────────────────────
# Both note types share `_NOTE_TEMPLATE` for length matching. Only the
# header and the four content slots ({ctx_command}, {ctx_signal},
# {ctx_files}, {ctx_confidence}) differ between repair and generic. The
# numbered procedure is identical so any DiD effect attributes to the
# trigger-grounded context, not to the procedural instructions or length.

TG_REPAIR_HEADER = "[INTERNAL DIAGNOSTIC — not visible to graders]"
PLACEBO_HEADER   = "[INTERNAL PROCESS NOTE — not visible to graders]"

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

# Kept as module-level constants so external callers / scripts/174 A/A
# paraphrase code can still reference the un-populated default versions.
TG_REPAIR_COACHING = _NOTE_TEMPLATE.format(
    header=TG_REPAIR_HEADER,
    ctx_command="<populated at fire time>",
    ctx_signal="<populated at fire time>",
    ctx_files="<populated at fire time>",
    ctx_confidence="<populated at fire time>",
    ctx_trap_pattern="<populated at fire time from trap diagnosis sidecar>",
)
PLACEBO_NOTE = _NOTE_TEMPLATE.format(
    header=PLACEBO_HEADER,
    ctx_command="your recent commands",
    ctx_signal="the most recent test or error output you produced",
    ctx_files="the file(s) you have been working with",
    ctx_confidence="n/a (process check)",
    ctx_trap_pattern=(
        "(generic process reminder, not specific to this step) Before "
        "extending the current chain of actions, briefly check that the "
        "next step is the smallest test or edit that could move the task "
        "forward. If the next action would simply repeat a prior read or "
        "stack another speculative edit on top of a previously failing "
        "one, consider instead: narrowing the hypothesis to a single "
        "failing assertion, running the smallest test that exercises "
        "only that assertion, and committing to a minimal edit only when "
        "that test fails for the expected reason. No pattern-specific "
        "information is provided here."
    ),
)


_PLACEBO_OVERRIDE: Optional[str] = None


def _maybe_override_placebo_from_env() -> None:
    """Allow scripts/174 A/A to swap the generic note with a paraphrase
    by setting RQ4V2_PLACEBO_NOTE_PATH=/path/to/note.txt before launching
    108. The override applies only to the `generic` note type and bypasses
    the length-matched template entirely (the paraphrase file is used
    verbatim)."""
    global _PLACEBO_OVERRIDE
    p = os.environ.get("RQ4V2_PLACEBO_NOTE_PATH")
    if not p:
        return
    try:
        text = Path(p).read_text()
        if text.strip():
            _PLACEBO_OVERRIDE = text
            print(f"[108] PLACEBO override loaded from {p} ({len(text)} chars)",
                  flush=True)
    except Exception as exc:
        print(f"[108] WARNING: failed to load RQ4V2_PLACEBO_NOTE_PATH={p}: {exc}",
              file=sys.stderr)


_maybe_override_placebo_from_env()


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


def _generic_slots() -> Dict[str, str]:
    """Length-matched generic slot values. Same structural template, no
    trigger-specific content. Keeps the placebo arm note size on the same
    order of magnitude as a populated repair note so length is not a
    confound.
    """
    return {
        "ctx_command": "your recent commands",
        "ctx_signal": "the most recent test or error output you produced",
        "ctx_files": "the file(s) you have been working with",
        "ctx_confidence": "n/a (process check)",
        "ctx_trap_pattern": (
            "(generic process reminder, not specific to this step) Before "
            "extending the current chain of actions, briefly check that the "
            "next step is the smallest test or edit that could move the "
            "task forward. If the next action would simply repeat a prior "
            "read or stack another speculative edit on top of a previously "
            "failing one, consider instead: narrowing the hypothesis to a "
            "single failing assertion, running the smallest test that "
            "exercises only that assertion, and committing to a minimal "
            "edit only when that test fails for the expected reason. No "
            "pattern-specific information is provided here."
        ),
    }


def render_repair_note(
    trigger_record: Optional[Dict[str, Any]] = None,
    recent_action: Optional[str] = None,
    recent_obs: Optional[str] = None,
    trap_diagnosis_text: Optional[str] = None,
) -> str:
    """Public helper exported for scripts/174 A/A check + tests."""
    return prompt_for_note_type(
        "repair",
        trigger_record=trigger_record,
        recent_action=recent_action,
        recent_obs=recent_obs,
        trap_diagnosis_text=trap_diagnosis_text,
    )


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


def load_libraries() -> Tuple[List[Set[str]], List[Set[str]], List[Set[str]], Dict[str, float]]:
    trap_path = _resource_or_data_path(TRAP_PATH)
    core_path = _resource_or_data_path(CORE_PATH)
    nontrap_path = _resource_or_data_path(NONTRAP_PATH)
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
    nontrap_raw = json.load(open(nontrap_path))["examples"] if nontrap_path.exists() else []
    nontrap = [
        set(ex.get("detection_keys") or strip_generic_keys(ex.get("keys", [])))
        for ex in nontrap_raw
    ]
    if not idf_path.exists():
        raise SystemExit(
            f"Missing {IDF_PATH} and bundled {RESOURCE_INT_DIR / IDF_PATH.name}; "
            "run scripts/intervention/build_swe_trap_library.py first"
        )
    idf = json.load(open(idf_path))
    return trap, core, nontrap, idf


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
    per-trap diagnosis text in the RQ4 v2 stronger repair note.
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
    """Load the optional `swebench_trap_diagnosis.json` sidecar and return
    a dict mapping trap example index → pattern-family diagnosis text.
    Returns {} silently if the sidecar is missing, so old configurations
    still work."""
    if not TRAP_DIAGNOSIS_PATH.exists():
        return {}
    try:
        data = json.loads(TRAP_DIAGNOSIS_PATH.read_text())
    except Exception as exc:
        print(f"[108] WARNING: failed to load {TRAP_DIAGNOSIS_PATH}: {exc}",
              file=sys.stderr)
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


def encode_step(bash_cmd: str, observation: str, progress: float) -> Set[str]:
    """Encode a SWE agent step into the cxcmu key alphabet.

    The cxcmu library uses TOOL: / ACTION: / CMD: / FILE_PATH: / FILE_EXT: /
    OBS:* keys. We synthesize TOOL:execute_bash + ACTION:<intent> +
    CMD:<classified>, then run the cxcmu OBS regex extractor on the
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

    # Hand OBS keys to cxcmu's regex-based extractor via extract_slice_keys.
    action_msg = {"tool_calls": [
        {"function": {"name": "execute_bash", "arguments": json.dumps({"command": bash_cmd})}}
    ]}
    obs_msg = {"content": obs_text[:4000]}
    try:
        cxcmu_keys = extract_slice_keys(action_msg, obs_msg, progress)
    except Exception:
        cxcmu_keys = set()
    # We add OBS:* and any other useful keys produced by the library, but
    # ignore PHASE which we leave to the cxcmu library to set; we strip in
    # the detector so PHASE doesn't dominate similarity.
    for k in cxcmu_keys:
        if k.startswith("OBS:"):
            keys.add(k)

    # Generic English error detector for common SWE failure phrases that the
    # cxcmu regex does not catch.
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


def shuffled_library_match(nontrap: List[Set[str]], n_target: int, rng: random.Random) -> List[Set[str]]:
    if not nontrap:
        return []
    n = min(n_target, len(nontrap))
    return rng.sample(nontrap, n)


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
    any_trap_then_core: bool = False
    first_trap_step: Optional[int] = None
    per_step_log: List[Dict[str, Any]] = field(default_factory=list)
    tool_error_count: int = 0
    hard_error_context_left: int = 0
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
            any_trap_then_core=self.any_trap_then_core,
            first_trap_step=self.first_trap_step,
            per_step_log=copy.deepcopy(self.per_step_log),
            tool_error_count=self.tool_error_count,
            hard_error_context_left=self.hard_error_context_left,
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
    snapshot_image = f"agentgraph-prefixfork:{_docker_slug(snapshot_name)}"
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
    """TraceGraph SWE agent + step-level repair-coaching intervention.

    Overrides `run()` to inject an intervention prompt after a trap-like
    observation is detected (subject to warmup, cooldown, max_triggers, and
    OBS:error gating).
    """

    def __init__(
        self,
        model,
        env,
        arm: str,
        trap_lib: List[Set[str]],
        core_lib: List[Set[str]],
        shuffled_lib: List[Set[str]],
        idf: Dict[str, float],
        sim_threshold: float = 0.55,
        margin: float = 0.10,
        cooldown: int = 3,
        max_triggers: int = 3,
        warmup_steps: int = 2,
        require_obs_keys: Optional[Sequence[str]] = ("OBS:error",),
        max_steps: int = 40,
        no_intervention: bool = False,
        hard_error_context_window: int = 0,
        require_hard_error_at_trigger: bool = False,
        allowed_intents: Optional[Sequence[str]] = None,
        require_evidence_locality: bool = False,
        trap_diagnosis_map: Optional[Dict[int, str]] = None,
        temperature_after_trigger: Optional[float] = None,
        top_p_after_trigger: Optional[float] = None,
        enable_retrace: bool = False,
        retrace_budget: int = 1,
        temperature_on_retrace: Optional[float] = None,
        top_p_on_retrace: Optional[float] = None,
    ):
        super().__init__(model=model, env=env, hooks=[], max_steps=max_steps)
        self.arm = arm
        self.arm_spec = parse_arm(arm)
        self.trigger_source = self.arm_spec.trigger_source
        self.note_type = self.arm_spec.note_type
        self.trap_lib = trap_lib
        self.core_lib = core_lib
        self.shuffled_lib = shuffled_lib
        self.idf = idf
        self.sim_threshold = float(sim_threshold)
        self.margin = float(margin)
        self.cooldown = int(cooldown)
        self.max_triggers = int(max_triggers)
        self.warmup_steps = int(warmup_steps)
        self.require_obs_keys = set(require_obs_keys or [])
        self.no_intervention = bool(no_intervention)
        self.hard_error_context_window = max(0, int(hard_error_context_window))
        self.require_hard_error_at_trigger = bool(require_hard_error_at_trigger)
        self.allowed_intents = set(allowed_intents) if allowed_intents else None
        self.require_evidence_locality = bool(require_evidence_locality)
        self.trap_diagnosis_map = dict(trap_diagnosis_map or {})
        # RQ4 v2 enhancement: after a trigger fires, optionally bump the
        # model's sampling temperature (and top_p) so the supervisor note
        # actually has a chance to alter the agent's high-probability
        # decoding chain. None means "no bump, keep base temperature".
        # Applied symmetrically to all arms (no temp confound across arms).
        self.temperature_after_trigger = (
            float(temperature_after_trigger) if temperature_after_trigger is not None else None
        )
        self.top_p_after_trigger = (
            float(top_p_after_trigger) if top_p_after_trigger is not None else None
        )
        # RQ4 v2 Option C: retrace-on-trigger. When True, the FIRST trigger
        # within the rollout's retrace_budget triggers:
        #   (1) restore docker env state to the post-step-(s-1) snapshot,
        #   (2) truncate messages/steps back to end-of-step-(s-1),
        #   (3) inject a [GUIDANCE] user msg containing the trap-family
        #       diagnosis as a "reconsider before continuing" suffix,
        #   (4) bump model.temperature/top_p to temperature_on_retrace /
        #       top_p_on_retrace to force exploration of an alternate plan.
        # Snapshot strategy: commit env after every step but cleanup the
        # previous snapshot (keep only the most recent) to bound disk use.
        self.enable_retrace = bool(enable_retrace)
        self.retrace_budget = int(retrace_budget)
        self.temperature_on_retrace = (
            float(temperature_on_retrace) if temperature_on_retrace is not None else None
        )
        self.top_p_on_retrace = (
            float(top_p_on_retrace) if top_p_on_retrace is not None else None
        )

    def _trigger_lib(self) -> List[Set[str]]:
        if self.trigger_source == "tracegraph":
            return self.trap_lib
        if self.trigger_source == "shuffled":
            return self.shuffled_lib
        raise ValueError(f"Unsupported trigger source: {self.trigger_source}")

    def _intervention_prompt(
        self,
        *,
        trigger_record: Optional[Dict[str, Any]] = None,
        recent_action: Optional[str] = None,
        recent_obs: Optional[str] = None,
        trap_diagnosis_text: Optional[str] = None,
    ) -> str:
        return prompt_for_note_type(
            self.note_type,
            trigger_record=trigger_record,
            recent_action=recent_action,
            recent_obs=recent_obs,
            trap_diagnosis_text=trap_diagnosis_text,
        )

    def run(
        self,
        problem_statement: Optional[str] = None,
        *,
        initial_messages: Optional[List[Dict[str, str]]] = None,
        initial_state: Optional[InterventionRunState] = None,
        inject_on_trigger: bool = True,
        stop_on_first_trigger: bool = False,
        return_state: bool = False,
        retain_snapshots: bool = False,
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

        # Option C: per-step docker snapshots + per-step msg-end indices, to
        # support rolling back to before any past step. Only populated if
        # self.enable_retrace; otherwise both lists stay empty and the loop
        # is byte-for-byte the legacy behavior.
        snapshots: List[Optional[Dict[str, Any]]] = []
        msg_end_idx: List[int] = []
        used_retraces: int = 0

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
                    "recent_hard_error_context": bool(state.hard_error_context_left > 0),
                    "is_trap": False,
                    "is_core": False,
                })
                state.next_step_index = s + 1
                break

            observation = self.env.execute(action_cmd)

            progress = (s + 1) / self.max_steps
            keys = encode_step(action_cmd, observation, progress)
            det_keys = strip_generic_keys(keys)
            sim_trap, sim_trap_idx = max_similarity_with_index(det_keys, self.trap_lib, self.idf)
            sim_core = max_similarity(det_keys, self.core_lib, self.idf) if self.core_lib else 0.0
            sim_detect = max_similarity(det_keys, self._trigger_lib(), self.idf)
            obs_has_error = "OBS:error" in det_keys
            obs_has_hard_error = "OBS:hard_error" in det_keys
            if obs_has_error:
                state.tool_error_count += 1
            if obs_has_hard_error and self.hard_error_context_window > 0:
                state.hard_error_context_left = max(
                    state.hard_error_context_left,
                    self.hard_error_context_window,
                )
            recent_hard_error_context = (
                state.hard_error_context_left > 0 if self.hard_error_context_window > 0 else False
            )
            is_trap = sim_trap >= self.sim_threshold and (sim_trap - sim_core) >= self.margin
            is_core = sim_core >= self.sim_threshold and (sim_core - sim_trap) >= self.margin
            if is_trap:
                state.visited_trap += 1
                if state.first_trap_step is None:
                    state.first_trap_step = s
            if is_core:
                state.visited_core += 1
                if state.first_trap_step is not None and state.first_trap_step <= s:
                    state.any_trap_then_core = True

            warmup_ok = s >= self.warmup_steps
            obs_gate_ok = (not self.require_obs_keys) or bool(self.require_obs_keys & det_keys)
            trigger_gate_ok = recent_hard_error_context if self.hard_error_context_window > 0 else obs_gate_ok
            # New AND-composed gates (no-op when their flags are unset)
            step_intent = _intent_from_cmd(action_cmd)
            hard_at_step_ok = (not self.require_hard_error_at_trigger) or obs_has_hard_error
            intent_ok = (self.allowed_intents is None) or (step_intent in self.allowed_intents)
            evidence_ok = (not self.require_evidence_locality) or any(
                k.startswith("FILE_PATH:") for k in det_keys
            )
            extra_gates_ok = hard_at_step_ok and intent_ok and evidence_ok
            triggered = False
            if (
                not self.no_intervention
                and warmup_ok
                and trigger_gate_ok
                and extra_gates_ok
                and state.cd_left == 0
                and state.n_triggers < self.max_triggers
                and sim_detect >= self.sim_threshold
            ):
                triggered = True
                state.n_triggers += 1
                state.n_trigger_candidates += 1
                state.cd_left = self.cooldown
            elif warmup_ok and trigger_gate_ok and extra_gates_ok and sim_detect >= self.sim_threshold:
                state.n_trigger_candidates += 1
                state.cd_left = max(0, state.cd_left - 1)
            else:
                state.cd_left = max(0, state.cd_left - 1)

            messages.append({"role": "assistant", "content": response.content})
            messages.append(make_observation_message(observation))

            # ====================================================================
            # OPTION C — retrace-on-trigger branch.
            # When enabled and a trigger fires with retrace budget remaining,
            # rewind docker + messages to the post-step-(s-1) snapshot, append
            # a [GUIDANCE] suffix, and bump temperature for exploration. The
            # current step's trigger/step records are intentionally NOT
            # appended (the step is being undone). The agent then re-runs
            # step s from the rewound state.
            # ====================================================================
            do_retrace = (
                triggered
                and inject_on_trigger
                and self.enable_retrace
                and used_retraces < self.retrace_budget
                and s >= 1
                and len(snapshots) >= s
                and snapshots[s - 1] is not None
                and len(msg_end_idx) >= s
            )
            if do_retrace:
                prev_snap = snapshots[s - 1]
                prev_msg_end = msg_end_idx[s - 1]
                # Build the diagnosis suffix BEFORE we tear down env, so we still
                # have access to action_cmd / observation / det_keys.
                diag_text = self.trap_diagnosis_map.get(int(sim_trap_idx)) if sim_trap_idx >= 0 else None
                guidance_record = {
                    "trigger_step": s,
                    "trigger_keys": sorted(det_keys),
                    "sim_trap": round(float(sim_trap), 4),
                    "sim_trap_idx": int(sim_trap_idx),
                    "obs_has_hard_error": bool(obs_has_hard_error),
                }
                guidance_body = self._intervention_prompt(
                    trigger_record=guidance_record,
                    recent_action=action_cmd,
                    recent_obs=observation,
                    trap_diagnosis_text=diag_text,
                )
                # Restore docker env from snapshot.
                try:
                    self.env.stop()
                except Exception:
                    pass
                retrace_container = f"{self.env.container_name}_retr{used_retraces}"
                subprocess.run(["docker", "rm", "-f", retrace_container], capture_output=True)
                new_env = SnapshotDockerEnvironment(
                    image=prev_snap["image"],
                    container_name=retrace_container,
                    timeout=getattr(self.env, "timeout", 30),
                    max_output_chars=getattr(self.env, "max_output_chars", 10000),
                    reset_to_base=False,
                    base_commit_override=prev_snap.get("base_commit"),
                )
                try:
                    new_env.start()
                    self.env = new_env
                except Exception as exc:
                    print(f"[retrace] env restore failed at step {s}: {exc}", flush=True)
                    # Fall through to the legacy inject path below.
                    do_retrace = False

            if do_retrace:
                # Truncate messages back to end-of-step-(s-1).
                del messages[prev_msg_end:]
                # Truncate per-step / steps / snapshots / msg_end_idx to keep
                # records 0..s-1.
                state.steps = state.steps[:s]
                state.per_step_log = state.per_step_log[:s]
                # We have NOT yet appended the trigger for this step, so
                # state.trigger_records is already clean.
                # Discard the snapshot we'd otherwise keep for step s.
                snapshots = snapshots[:s]
                msg_end_idx = msg_end_idx[:s]
                # Inject [GUIDANCE] user msg.
                messages.append({
                    "role": "user",
                    "content": (
                        f"[GUIDANCE — pattern detector flagged the action you "
                        f"were about to take at step {s} as high-risk based on "
                        f"prior failed trajectories. The environment has been "
                        f"rolled back to before that action. Please reconsider "
                        f"the plan using the diagnostic below before issuing "
                        f"your next ACTION.]\n\n"
                        + guidance_body
                    ),
                })
                if self.temperature_on_retrace is not None:
                    try:
                        self.model.temperature = self.temperature_on_retrace
                    except Exception:
                        pass
                if self.top_p_on_retrace is not None:
                    try:
                        self.model.top_p = self.top_p_on_retrace
                    except Exception:
                        pass
                used_retraces += 1
                # Roll back trigger / candidate counters so the retraced
                # step is not double-counted if it re-fires.
                state.n_triggers = max(0, state.n_triggers - 1)
                state.n_trigger_candidates = max(0, state.n_trigger_candidates - 1)
                state.cd_left = 0
                state.next_step_index = s
                # Do NOT advance s; loop will re-run step s with rewound state.
                continue

            if triggered and inject_on_trigger:
                # Build the trigger record now so the note can render grounded
                # slots (no oracle leakage: only fields the agent itself
                # produced this step are passed in).
                # For the trap-pattern slot, look up the per-family diagnosis
                # text by the best-matching trap example's index. The trap
                # diagnosis is generic pattern advice keyed off abstract
                # action signatures, not task-specific knowledge.
                diag_text = self.trap_diagnosis_map.get(int(sim_trap_idx)) if sim_trap_idx >= 0 else None
                trigger_record_for_note = {
                    "trigger_step": s,
                    "trigger_keys": sorted(det_keys),
                    "sim_trap": round(float(sim_trap), 4),
                    "sim_trap_idx": int(sim_trap_idx),
                    "sim_core": round(float(sim_core), 4),
                    "sim_detect": round(float(sim_detect), 4),
                    "obs_has_error": bool(obs_has_error),
                    "obs_has_hard_error": bool(obs_has_hard_error),
                    "recent_hard_error_context": bool(recent_hard_error_context),
                    "is_trap": bool(is_trap),
                    "is_core": bool(is_core),
                    "trap_diagnosis_present": bool(diag_text),
                }
                rendered_note = self._intervention_prompt(
                    trigger_record=trigger_record_for_note,
                    recent_action=action_cmd,
                    recent_obs=observation,
                    trap_diagnosis_text=diag_text,
                )
                messages.append({
                    "role": "user",
                    "content": "[NOTE FROM SUPERVISOR] " + rendered_note,
                })
                # RQ4 v2: bump sampling temperature/top_p AFTER trigger so the
                # injected note can actually divert the agent's high-probability
                # decoding chain. Applied symmetrically across arms (placebo and
                # tracegraph both get the same bump) so the only differential
                # treatment between arms is the *content* of the note slot.
                if self.temperature_after_trigger is not None:
                    try:
                        self.model.temperature = self.temperature_after_trigger
                    except Exception:
                        pass
                if self.top_p_after_trigger is not None:
                    try:
                        self.model.top_p = self.top_p_after_trigger
                    except Exception:
                        pass

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
                "intent": _intent_from_cmd(action_cmd),
                "sim_trap": round(float(sim_trap), 4),
                "sim_core": round(float(sim_core), 4),
                "sim_detect": round(float(sim_detect), 4),
                "obs_has_error": bool(obs_has_error),
                "obs_has_hard_error": bool(obs_has_hard_error),
                "recent_hard_error_context": bool(recent_hard_error_context),
                "is_trap": bool(is_trap),
                "is_core": bool(is_core),
                "triggered": bool(triggered),
                "n_obs_chars": len(observation or ""),
                # Raw assistant content (thinking + thought + action block) so
                # downstream path-B env-replay scripts can reconstruct the
                # original conversation prefix without resampling the LLM.
                "raw_assistant": (response.content if response is not None else ""),
                "observation": observation or "",
            }
            state.per_step_log.append(step_log)
            if triggered:
                state.trigger_records.append({
                    "trigger_step": s,
                    "trigger_keys": sorted(det_keys),
                    "sim_trap": round(float(sim_trap), 4),
                    "sim_core": round(float(sim_core), 4),
                    "sim_detect": round(float(sim_detect), 4),
                    "obs_has_error": bool(obs_has_error),
                    "obs_has_hard_error": bool(obs_has_hard_error),
                    "recent_hard_error_context": bool(recent_hard_error_context),
                    "is_trap": bool(is_trap),
                    "is_core": bool(is_core),
                })

            state.next_step_index = s + 1
            if self.hard_error_context_window > 0:
                state.hard_error_context_left = max(0, state.hard_error_context_left - 1)

            # Option C: snapshot the env after this step so a future trigger
            # could roll back to here. Keep only the most recent snapshot per
            # rollout to bound disk usage.
            if self.enable_retrace:
                try:
                    # Cleanup previous snapshot.
                    if s >= 1 and len(snapshots) >= s and snapshots[s - 1] is not None:
                        cleanup_snapshot_image(snapshots[s - 1]["image"])
                        snapshots[s - 1] = None
                    snap_name = f"{self.env.container_name}-step{s}"
                    snap = snapshot_env_state(self.env, snap_name)
                    while len(snapshots) <= s:
                        snapshots.append(None)
                    snapshots[s] = snap
                except Exception as exc:
                    print(f"[retrace] snapshot failed at step {s}: {exc}", flush=True)
            msg_end_idx.append(len(messages))

            if triggered and stop_on_first_trigger:
                exit_reason = "prefix_fork_trigger"
                break
            s += 1

        # Attach snapshots + msg_end_idx to state so callers (e.g. the
        # probe-fork driver) can read them. Caller is responsible for
        # cleanup when retain_snapshots=True.
        state.snapshots = list(snapshots)  # type: ignore[attr-defined]
        state.msg_end_idx = list(msg_end_idx)  # type: ignore[attr-defined]

        # Cleanup remaining snapshot images at exit UNLESS caller asked to
        # keep them (e.g. probe-fork driver wants to fork from a snapshot).
        if self.enable_retrace and not retain_snapshots:
            for snap in snapshots:
                if snap is not None:
                    try:
                        cleanup_snapshot_image(snap["image"])
                    except Exception:
                        pass

        patch = ""
        try:
            patch = self.env.get_patch()
        except Exception:
            pass

        result = AgentResult(steps=state.steps, patch=patch, exit_reason=exit_reason)
        trigger_meta = summarize_trigger_behavior(state.per_step_log, window=2)
        meta = {
            "arm": self.arm,
            "trigger_source": self.trigger_source,
            "note_type": self.note_type,
            "exit_reason": exit_reason,
            "n_steps": len(state.steps),
            "n_triggers": int(state.n_triggers),
            "n_trigger_candidates": int(state.n_trigger_candidates),
            "visited_trap": int(state.visited_trap),
            "visited_core": int(state.visited_core),
            "first_trap_step": state.first_trap_step,
            "E_r": int(state.visited_trap > 0),
            "A_r": int(state.visited_core > 0),
            "R_r": int(state.any_trap_then_core),
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
        "design": args.design,
        "sim_threshold": args.sim_threshold,
        "margin": args.margin,
        "cooldown": args.cooldown,
        "max_triggers": args.max_triggers,
        "warmup_steps": args.warmup_steps,
        "require_obs_keys": list(args.require_obs_keys or []),
        "hard_error_context_window": int(args.hard_error_context_window),
        "require_hard_error_at_trigger": bool(getattr(args, "require_hard_error_at_trigger", False)),
        "allowed_intents": list(getattr(args, "allowed_intents", None) or []),
        "require_evidence_locality": bool(getattr(args, "require_evidence_locality", False)),
        "no_intervention": bool(args.no_intervention),
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
    shuffled_lib: List[Set[str]],
    idf: Dict[str, float],
    trap_diagnosis_map: Optional[Dict[int, str]] = None,
) -> InterventionAgent:
    return InterventionAgent(
        model=model,
        env=env,
        arm=arm,
        trap_lib=trap_lib,
        core_lib=core_lib,
        shuffled_lib=shuffled_lib,
        idf=idf,
        sim_threshold=args.sim_threshold,
        margin=args.margin,
        cooldown=args.cooldown,
        max_triggers=args.max_triggers,
        warmup_steps=args.warmup_steps,
        require_obs_keys=args.require_obs_keys,
        max_steps=args.max_steps,
        no_intervention=args.no_intervention,
        hard_error_context_window=args.hard_error_context_window,
        require_hard_error_at_trigger=getattr(args, "require_hard_error_at_trigger", False),
        allowed_intents=getattr(args, "allowed_intents", None),
        require_evidence_locality=getattr(args, "require_evidence_locality", False),
        trap_diagnosis_map=trap_diagnosis_map,
        temperature_after_trigger=getattr(args, "temperature_after_trigger", None),
        top_p_after_trigger=getattr(args, "top_p_after_trigger", None),
        enable_retrace=getattr(args, "enable_retrace", False),
        retrace_budget=getattr(args, "retrace_budget", 1),
        temperature_on_retrace=getattr(args, "temperature_on_retrace", None),
        top_p_on_retrace=getattr(args, "top_p_on_retrace", None),
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
        "design": args.design,
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
    trigger_source: str,
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
        "design": args.design,
        "trigger_source": trigger_source,
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
            "sim_core": first_trigger.get("sim_core"),
            "sim_detect": first_trigger.get("sim_detect"),
            "obs_has_error": first_trigger.get("obs_has_error"),
            "obs_has_hard_error": first_trigger.get("obs_has_hard_error"),
            "recent_hard_error_context": first_trigger.get("recent_hard_error_context"),
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


def run_independent_rollouts(
    *,
    args: argparse.Namespace,
    model: Any,
    instances: Sequence[Dict[str, Any]],
    trap_lib: List[Set[str]],
    core_lib: List[Set[str]],
    shuffled_lib: List[Set[str]],
    idf: Dict[str, float],
    out_path: Path,
    done: Set[Tuple[str, str, int]],
    trap_diagnosis_map: Optional[Dict[int, str]] = None,
) -> int:
    written = 0
    t0 = time.time()
    with open(out_path, "a") as fh:
        for inst in instances:
            instance_id = inst["instance_id"]
            image_name = _instance_image(instance_id)
            for arm in args.arms:
                key = (instance_id, arm, args.seed)
                if key in done:
                    continue
                container_name = f"{args.container_name_prefix}_ablate_{_container_slug(instance_id)}_{arm}"
                env = DockerEnvironment(
                    image=image_name,
                    container_name=container_name,
                    timeout=args.timeout,
                    max_output_chars=args.max_output_chars,
                )
                t_start = time.time()
                try:
                    env.start()
                except Exception as exc:
                    print(f"  docker start failed for {instance_id} arm={arm}: {exc}", flush=True)
                    _write_jsonl_row(fh, {
                        "instance_id": instance_id,
                        "arm": arm,
                        "seed": args.seed,
                        "design": args.design,
                        "error": f"docker_start: {str(exc)[:300]}",
                    })
                    continue

                agent = build_agent(
                    args=args,
                    model=model,
                    env=env,
                    arm=arm,
                    trap_lib=trap_lib,
                    core_lib=core_lib,
                    shuffled_lib=shuffled_lib,
                    idf=idf,
                    trap_diagnosis_map=trap_diagnosis_map,
                )
                try:
                    result, meta = agent.run(inst["problem_statement"])
                    err = None
                except Exception as exc:
                    result = SimpleNamespace(patch="", steps=[], exit_reason=f"error: {exc}")
                    meta = {"arm": arm, "error": str(exc)[:300]}
                    err = str(exc)[:300]
                finally:
                    try:
                        env.stop()
                    except Exception:
                        pass

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
                )
                _write_jsonl_row(fh, row)
                written += 1
                done.add(key)
                elapsed = time.time() - t0
                patch_chars = len(row.get("patch") or "")
                print(
                    f"[{written}] {instance_id} arm={arm} "
                    f"exit={row['exit_reason']} steps={meta.get('n_steps', '?')} "
                    f"trig={meta.get('n_triggers', '?')}/cand={meta.get('n_trigger_candidates', '?')} "
                    f"E={meta.get('E_r', '?')} R={meta.get('R_r', '?')} "
                    f"patch={patch_chars}ch dur={dur:.0f}s elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
    return written


def run_prefix_fork_rollouts(
    *,
    args: argparse.Namespace,
    model: Any,
    instances: Sequence[Dict[str, Any]],
    trap_lib: List[Set[str]],
    core_lib: List[Set[str]],
    shuffled_lib: List[Set[str]],
    idf: Dict[str, float],
    out_path: Path,
    prefix_log_path: Path,
    done: Set[Tuple[str, str, int]],
    trap_diagnosis_map: Optional[Dict[int, str]] = None,
) -> int:
    written = 0
    t0 = time.time()
    arm_specs = [parse_arm(arm) for arm in args.arms]
    trigger_sources = {spec.trigger_source for spec in arm_specs}
    if len(trigger_sources) != 1:
        raise SystemExit("prefix-fork mode currently requires a single trigger source per run.")
    trigger_source = next(iter(trigger_sources))
    probe_arm = "tg_repair" if trigger_source == "tracegraph" else "shuf_repair"

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
                    shuffled_lib=shuffled_lib,
                    idf=idf,
                    trap_diagnosis_map=trap_diagnosis_map,
                )
                base_result, base_meta, base_state = probe_agent.run(
                    inst["problem_statement"],
                    inject_on_trigger=False,
                    stop_on_first_trigger=True,
                    return_state=True,
                )
                first_trigger = ((base_meta.get("trigger_records") or [None])[0] if base_meta else None)
                if first_trigger:
                    prefix_id = (
                        f"{_docker_slug(instance_id)}-seed{args.seed}-"
                        f"{trigger_source}-step{first_trigger.get('trigger_step')}"
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
                trigger_source=trigger_source,
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
                            shuffled_lib=shuffled_lib,
                            idf=idf,
                            trap_diagnosis_map=trap_diagnosis_map,
                        )
                        # Render the repair / generic note WITH actual trigger
                        # context (G2 diagnosis populates the ctx_trap_pattern
                        # slot only when these fields are passed in). The
                        # legacy call passed only note_type, leaving slots as
                        # "<populated at fire time>" placeholders.
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
                        if arm_spec.note_type != "none":
                            rendered_note = prompt_for_note_type(
                                arm_spec.note_type,
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
                        # Temperature / top_p decisions are independent of note
                        # injection: tg_hot has note=none but bump_temp=True,
                        # tg_baseline has note=none but bump_temp=False.
                        if arm_spec.bump_temp:
                            if getattr(args, "temperature_after_trigger", None) is not None:
                                try: model.temperature = float(args.temperature_after_trigger)
                                except Exception: pass
                            if getattr(args, "top_p_after_trigger", None) is not None:
                                try: model.top_p = float(args.top_p_after_trigger)
                                except Exception: pass
                        else:
                            # Explicitly restore base temperature/top_p (in case
                            # a prior arm in the loop bumped them).
                            if base_temp is not None:
                                try: model.temperature = base_temp
                                except Exception: pass
                            if base_top_p is not None:
                                try: model.top_p = base_top_p
                                except Exception: pass
                        result, meta = fork_agent.run(
                            initial_messages=fork_messages,
                            initial_state=base_state,
                        )
                        err = None
                    except Exception as exc:
                        result = SimpleNamespace(patch="", steps=[], exit_reason=f"error: {exc}")
                        meta = {
                            "arm": arm,
                            "trigger_source": arm_spec.trigger_source,
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
                            "prefix_trigger_source": trigger_source,
                            "prefix_trigger_step": first_trigger.get("trigger_step"),
                            "prefix_trigger_keys": first_trigger.get("trigger_keys") or [],
                            "prefix_obs_has_error": first_trigger.get("obs_has_error"),
                            "prefix_obs_has_hard_error": first_trigger.get("obs_has_hard_error"),
                            "prefix_recent_hard_error_context": first_trigger.get("recent_hard_error_context"),
                            "prefix_sim_trap": first_trigger.get("sim_trap"),
                            "prefix_sim_core": first_trigger.get("sim_core"),
                            "prefix_sim_detect": first_trigger.get("sim_detect"),
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


def run_probe_then_retrace_fork(
    *,
    args: argparse.Namespace,
    model: Any,
    instances: Sequence[Dict[str, Any]],
    trap_lib: List[Set[str]],
    core_lib: List[Set[str]],
    shuffled_lib: List[Set[str]],
    idf: Dict[str, float],
    out_path: Path,
    done: Set[Tuple[str, str, int]],
    trap_diagnosis_map: Optional[Dict[int, str]] = None,
) -> int:
    """Option C: shared-prefix probe → fork into 2 branches.

    For each (instance, seed):
      1. PROBE: one rollout at args.temperature with NO injection and
         per-step env snapshots (--enable-retrace + retrace_budget=0).
         Runs all 30 steps. Written as arm = `baseline_optC`.
      2. If a trigger fired during probe at step k, RESTORE the docker env
         from the snapshot at end-of-step-(k-1) (pre-trap state), TRUNCATE
         messages to end-of-step-(k-1), INJECT a [GUIDANCE] user msg
         containing the trap-family diagnosis, BUMP model.temperature to
         args.temperature_on_retrace, then resume to max_steps. Written as
         arm = `retrace_optC`.
    Both arms share the prefix steps 0..k-1 (same docker history, same
    decoded tokens) so paired comparison is uniform.
    """
    written = 0
    base_temp = getattr(model, "temperature", None)
    base_top_p = getattr(model, "top_p", None)
    BASELINE_ARM = "baseline_optC"
    RETRACE_ARM = "retrace_optC"

    with open(out_path, "a") as rollout_fh:
        for inst in instances:
            instance_id = inst["instance_id"]
            seed = args.seed
            need_baseline = (instance_id, BASELINE_ARM, seed) not in done
            need_retrace = (instance_id, RETRACE_ARM, seed) not in done
            if not need_baseline and not need_retrace:
                continue

            image_name = _instance_image(instance_id)
            slug = _container_slug(instance_id)

            # -------- PROBE phase (baseline) --------
            probe_env = SnapshotDockerEnvironment(
                image=image_name,
                container_name=f"{args.container_name_prefix}_probe_{slug}",
                timeout=args.timeout,
                max_output_chars=args.max_output_chars,
                reset_to_base=True,
            )
            probe_result = SimpleNamespace(patch="", steps=[], exit_reason="probe_not_started")
            probe_meta: Dict[str, Any] = {}
            probe_state: Optional[InterventionRunState] = None
            probe_err: Optional[str] = None
            t_probe_start = time.time()
            try:
                probe_env.start()
                if base_temp is not None:
                    try: model.temperature = base_temp
                    except Exception: pass
                if base_top_p is not None:
                    try: model.top_p = base_top_p
                    except Exception: pass
                probe_agent = build_agent(
                    args=args, model=model, env=probe_env, arm="placebo",
                    trap_lib=trap_lib, core_lib=core_lib, shuffled_lib=shuffled_lib,
                    idf=idf, trap_diagnosis_map=trap_diagnosis_map,
                )
                # Probe: snapshot per step but never actually retrace, never inject.
                probe_agent.enable_retrace = True
                probe_agent.retrace_budget = 0
                probe_agent.no_intervention = True
                probe_result, probe_meta, probe_state = probe_agent.run(
                    inst["problem_statement"],
                    inject_on_trigger=False,
                    stop_on_first_trigger=False,
                    return_state=True,
                    retain_snapshots=True,
                )
            except Exception as exc:
                probe_err = str(exc)[:300]
                print(f"[probe-fork] {instance_id} probe failed: {exc}", flush=True)
            finally:
                try: probe_env.stop()
                except Exception: pass
            probe_dur = time.time() - t_probe_start

            if need_baseline:
                row = build_rollout_row(
                    instance_id=instance_id, arm=BASELINE_ARM, seed=seed,
                    duration_sec=probe_dur, result=probe_result, err=probe_err,
                    args=args, meta=probe_meta or {},
                    extra={"design": "probe-fork", "phase": "probe_baseline",
                           "arm": BASELINE_ARM, "arm_internal": (probe_meta or {}).get("arm")},
                )
                _write_jsonl_row(rollout_fh, row)
                written += 1
                done.add((instance_id, BASELINE_ARM, seed))
                print(f"[probe-fork] {instance_id} baseline: "
                      f"steps={(probe_meta or {}).get('n_steps')} "
                      f"trig={(probe_meta or {}).get('n_triggers')} "
                      f"patch={len(probe_result.patch)}ch dur={probe_dur:.0f}s",
                      flush=True)

            # -------- RETRACE FORK phase --------
            def _cleanup_probe_snapshots():
                if probe_state is None: return
                for snap in (getattr(probe_state, "snapshots", []) or []):
                    if snap is not None:
                        try: cleanup_snapshot_image(snap["image"])
                        except Exception: pass

            if not need_retrace:
                _cleanup_probe_snapshots()
                continue
            if probe_state is None or not probe_state.trigger_records:
                # No trigger fired → can't fork. Write empty retrace row.
                _write_jsonl_row(rollout_fh, build_rollout_row(
                    instance_id=instance_id, arm=RETRACE_ARM, seed=seed,
                    duration_sec=0.0,
                    result=SimpleNamespace(patch="", steps=[], exit_reason="no_trigger_during_probe"),
                    err="no trigger during probe; nothing to retrace",
                    args=args, meta={},
                    extra={"design": "probe-fork", "phase": "retrace_fork", "arm": RETRACE_ARM},
                ))
                written += 1
                done.add((instance_id, RETRACE_ARM, seed))
                _cleanup_probe_snapshots()
                continue

            first_trig = probe_state.trigger_records[0]
            trap_step = int(first_trig.get("trigger_step", 0))
            snapshots_list = list(getattr(probe_state, "snapshots", []) or [])
            msg_end_idx_list = list(getattr(probe_state, "msg_end_idx", []) or [])
            if (trap_step < 1
                or len(snapshots_list) <= trap_step - 1
                or snapshots_list[trap_step - 1] is None):
                _write_jsonl_row(rollout_fh, build_rollout_row(
                    instance_id=instance_id, arm=RETRACE_ARM, seed=seed,
                    duration_sec=0.0,
                    result=SimpleNamespace(patch="", steps=[], exit_reason="no_pre_trigger_snapshot"),
                    err=f"trap_step={trap_step}, no pre-trigger snapshot available",
                    args=args, meta={},
                    extra={"design": "probe-fork", "phase": "retrace_fork", "arm": RETRACE_ARM},
                ))
                written += 1
                done.add((instance_id, RETRACE_ARM, seed))
                _cleanup_probe_snapshots()
                continue

            pre_snap = snapshots_list[trap_step - 1]
            pre_msg_end = (msg_end_idx_list[trap_step - 1]
                           if len(msg_end_idx_list) > trap_step - 1
                           else len(probe_state.messages))

            # Build the [GUIDANCE] body from trap diagnosis + the trigger
            # step's actual action / observation (recovered from probe state).
            sim_trap_idx = first_trig.get("sim_trap_idx", -1)
            try:
                sim_trap_idx_int = int(sim_trap_idx) if sim_trap_idx is not None else -1
            except (TypeError, ValueError):
                sim_trap_idx_int = -1
            diag_text = None
            if trap_diagnosis_map and sim_trap_idx_int >= 0:
                diag_text = trap_diagnosis_map.get(sim_trap_idx_int)
            trigger_action = ""
            trigger_obs = ""
            if len(probe_state.steps) > trap_step:
                trigger_action = getattr(probe_state.steps[trap_step], "action", "") or ""
                trigger_obs = getattr(probe_state.steps[trap_step], "observation", "") or ""
            guidance_body = prompt_for_note_type(
                "repair",
                trigger_record={
                    "trigger_step": trap_step,
                    "trigger_keys": first_trig.get("trigger_keys"),
                    "sim_trap": first_trig.get("sim_trap"),
                    "sim_trap_idx": sim_trap_idx_int,
                    "obs_has_hard_error": first_trig.get("obs_has_hard_error", False),
                },
                recent_action=trigger_action,
                recent_obs=trigger_obs,
                trap_diagnosis_text=diag_text,
            )

            fork_container = f"{args.container_name_prefix}_retracefork_{slug}"
            subprocess.run(["docker", "rm", "-f", fork_container], capture_output=True)
            fork_env = SnapshotDockerEnvironment(
                image=pre_snap["image"],
                container_name=fork_container,
                timeout=args.timeout,
                max_output_chars=args.max_output_chars,
                reset_to_base=False,
                base_commit_override=pre_snap.get("base_commit"),
            )
            fork_result = SimpleNamespace(patch="", steps=[], exit_reason="fork_not_started")
            fork_meta: Dict[str, Any] = {}
            fork_err: Optional[str] = None
            t_fork_start = time.time()
            try:
                fork_env.start()
                # Bump model sampling for retrace.
                if args.temperature_on_retrace is not None:
                    try: model.temperature = float(args.temperature_on_retrace)
                    except Exception: pass
                if args.top_p_on_retrace is not None:
                    try: model.top_p = float(args.top_p_on_retrace)
                    except Exception: pass
                fork_agent = build_agent(
                    args=args, model=model, env=fork_env, arm="tracegraph",
                    trap_lib=trap_lib, core_lib=core_lib, shuffled_lib=shuffled_lib,
                    idf=idf, trap_diagnosis_map=trap_diagnosis_map,
                )
                fork_agent.enable_retrace = False
                fork_agent.no_intervention = True  # [GUIDANCE] is already in messages
                # Build fork messages: probe messages truncated to end-of-(k-1) + [GUIDANCE].
                fork_messages = copy.deepcopy(probe_state.messages[:pre_msg_end])
                fork_messages.append({
                    "role": "user",
                    "content": (
                        f"[GUIDANCE — pattern detector flagged the action you were about to "
                        f"take at step {trap_step} as high-risk based on prior failed "
                        f"trajectories. The environment has been rolled back to before that "
                        f"action. Please reconsider before issuing your next ACTION.]\n\n"
                        + guidance_body
                    ),
                })
                fork_state = InterventionRunState(messages=fork_messages)
                fork_state.next_step_index = trap_step
                fork_state.steps = list(probe_state.steps[:trap_step])
                fork_state.per_step_log = list(probe_state.per_step_log[:trap_step])
                fork_result, fork_meta = fork_agent.run(
                    initial_state=fork_state,
                    inject_on_trigger=False,
                    stop_on_first_trigger=False,
                    return_state=False,
                )
            except Exception as exc:
                fork_err = str(exc)[:300]
                print(f"[probe-fork] {instance_id} retrace fork failed: {exc}", flush=True)
            finally:
                try: fork_env.stop()
                except Exception: pass
                _cleanup_probe_snapshots()
                # Restore base temperature for next instance's probe.
                if base_temp is not None:
                    try: model.temperature = base_temp
                    except Exception: pass
                if base_top_p is not None:
                    try: model.top_p = base_top_p
                    except Exception: pass
            fork_dur = time.time() - t_fork_start
            _write_jsonl_row(rollout_fh, build_rollout_row(
                instance_id=instance_id, arm=RETRACE_ARM, seed=seed,
                duration_sec=fork_dur, result=fork_result, err=fork_err,
                args=args, meta=fork_meta,
                extra={
                    "design": "probe-fork", "phase": "retrace_fork",
                    "arm": RETRACE_ARM,
                    "arm_internal": fork_meta.get("arm"),
                    "fork_from_step": trap_step - 1,
                    "trap_step": trap_step,
                    "fork_temperature": args.temperature_on_retrace,
                    "fork_top_p": args.top_p_on_retrace,
                },
            ))
            written += 1
            done.add((instance_id, RETRACE_ARM, seed))
            print(f"[probe-fork] {instance_id} retrace: "
                  f"steps={fork_meta.get('n_steps')} "
                  f"patch={len(fork_result.patch)}ch dur={fork_dur:.0f}s "
                  f"fork_from=step{trap_step - 1}",
                  flush=True)

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["independent", "prefix-fork", "probe-fork"], default="independent",
                    help="probe-fork = Option C: 1 probe rollout at base temp with no inject "
                         "(written as baseline_optC arm), then if a trigger fired, fork from the "
                         "pre-trigger snapshot with [GUIDANCE] inject + bumped temperature "
                         "(written as retrace_optC arm).")
    ap.add_argument("--arms", nargs="+", default=None, choices=ARM_CHOICES)
    ap.add_argument("--instances", nargs="*", default=None,
                    help="Specific instance_ids; mutually exclusive with --max-instances")
    ap.add_argument("--max-instances", type=int, default=10)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--sim-threshold", type=float, default=0.55)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--cooldown", type=int, default=3)
    ap.add_argument("--max-triggers", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--require-obs-keys", nargs="+", default=["OBS:error"],
                    help="Trigger only if any of these detection keys are present. "
                         "Default: OBS:error")
    ap.add_argument("--no-obs-gate", action="store_true",
                    help="Disable observation-key gating for triggering.")
    ap.add_argument("--hard-error-context-window", type=int, default=0,
                    help="If > 0, gate triggers on recent hard-error context for this many steps "
                         "instead of current-step OBS:error matching.")
    # New gating flags (added 2026-05-21 after detector-only sweep; AND-composed
    # with the existing gates so legacy behavior is preserved when unset).
    ap.add_argument("--require-hard-error-at-trigger", action="store_true",
                    help="Additionally require obs_has_hard_error==True at the trigger step itself "
                         "(not just within the lookback window).")
    ap.add_argument("--allowed-intents", nargs="+", default=None,
                    choices=["edit", "submit", "test", "read", "search", "other"],
                    help="If set, only fire when the current step's intent is in this set "
                         "(e.g. 'edit submit' to skip read-after-error states).")
    ap.add_argument("--require-evidence-locality", action="store_true",
                    help="Additionally require at least one FILE_PATH:* key in the trigger step's "
                         "detection keys (proxy for evidence-localizable repair targets).")
    ap.add_argument("--no-intervention", action="store_true",
                    help="A/A mode: do not actually inject the intervention; detection metrics still logged.")
    # RQ4 v2: post-trigger sampling bump (apply symmetrically to all arms so
    # the only cross-arm difference is the *content* of the injected note).
    ap.add_argument("--temperature-after-trigger", type=float, default=None,
                    help="If set, after the first trigger fires, bump model.temperature to this "
                         "value for all subsequent steps. Helps escape deterministic decoding "
                         "(at temp=0 the injected note is often ignored). Qwen3 thinking "
                         "recommended = 0.6.")
    ap.add_argument("--top-p-after-trigger", type=float, default=None,
                    help="If set, similarly bump model.top_p after the first trigger. "
                         "Qwen3 thinking recommended = 0.95.")
    # Option C: retrace-on-trigger (rollback docker + truncate messages +
    # inject [GUIDANCE] suffix + bump temperature).
    ap.add_argument("--enable-retrace", action="store_true",
                    help="Option C: on trigger, restore the docker env to the post-step-(s-1) "
                         "snapshot, truncate messages, inject a [GUIDANCE] user msg with the "
                         "trap diagnosis, and bump sampling temperature for exploration. "
                         "Requires --temperature-on-retrace to actually bump.")
    ap.add_argument("--retrace-budget", type=int, default=1,
                    help="Max number of retraces per rollout (default 1).")
    ap.add_argument("--temperature-on-retrace", type=float, default=None,
                    help="model.temperature value after retrace (e.g. 0.9 for exploration).")
    ap.add_argument("--top-p-on-retrace", type=float, default=None,
                    help="model.top_p value after retrace.")
    ap.add_argument("--model-base-url", default="http://localhost:8192/v1")
    ap.add_argument("--model-name", default="YOUR_MODEL_NAME_OR_PATH")
    ap.add_argument("--api-provider", choices=["vllm", "deepseek", "glm"], default="vllm",
                    help="vllm = local model via vLLM (uses tracegraph.sweagent.VLLMModel); "
                         "deepseek = DeepSeek API (uses DeepSeekModel shim); "
                         "glm = Zhipu GLM API (also OpenAI-compatible).")
    ap.add_argument("--api-key", default=None,
                    help="API key for deepseek provider. Falls back to DEEPSEEK_API_KEY env var.")
    ap.add_argument("--thinking-enabled", action="store_true",
                    help="For deepseek provider: enable thinking mode (otherwise disabled).")
    ap.add_argument("--max-model-tokens", type=int, default=None,
                    help="Per-call max_tokens for the model (None = provider default).")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-output-chars", type=int, default=15000)
    ap.add_argument("--container-name-prefix", default="tracegraph",
                    help="Prefix for docker container names. Use a per-provider "
                         "prefix (e.g. tracegraph_qwen, tracegraph_deepseek) to allow "
                         "multiple providers to run on the same instance in parallel.")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--prefix-log-output", default=None,
                    help="Prefix-fork sidecar JSONL for saved trigger prefixes. "
                         "Default: <output stem>_prefixes.jsonl")
    args = ap.parse_args()
    if args.arms is None:
        args.arms = list(PREFIX_FORK_ARM_DEFAULT if args.design == "prefix-fork" else LEGACY_ARM_DEFAULT)
    args.arms = _ordered_arms(args.arms)
    if args.no_obs_gate:
        args.require_obs_keys = []
    if args.design == "prefix-fork" and args.no_intervention:
        raise SystemExit("prefix-fork mode requires actual note injection; do not use --no-intervention.")
    if args.design == "prefix-fork" and args.max_triggers != 1:
        raise SystemExit("prefix-fork mode currently requires --max-triggers 1.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed(out_path)
    print(f"Found {len(done)} completed rollouts in {out_path}", flush=True)
    prefix_log_path = None
    if args.design == "prefix-fork":
        prefix_log_path = Path(args.prefix_log_output) if args.prefix_log_output else (
            out_path.with_name(f"{out_path.stem}_prefixes.jsonl")
        )
        prefix_log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Prefix logs → {prefix_log_path}", flush=True)

    trap_lib, core_lib, nontrap_lib, idf = load_libraries()
    rng = random.Random(args.seed)
    shuffled_lib = shuffled_library_match(nontrap_lib, len(trap_lib), rng)
    trap_diagnosis_map = load_trap_diagnosis()
    print(
        f"SWE libs | trap={len(trap_lib)} core={len(core_lib)} "
        f"nontrap={len(nontrap_lib)} shuffled={len(shuffled_lib)} "
        f"idf={len(idf)} trap_diagnosis={len(trap_diagnosis_map)}",
        flush=True,
    )

    # Build the model interface.
    if args.api_provider == "deepseek":
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("--api-provider deepseek requires --api-key or DEEPSEEK_API_KEY env var")
        model = DeepSeekModel(
            base_url=args.model_base_url if args.model_base_url != "http://localhost:8192/v1"
                else "https://api.deepseek.com",
            model_name=args.model_name if args.model_name.startswith("deepseek")
                else "deepseek-v4-pro",
            api_key=api_key,
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_model_tokens,
            thinking_enabled=args.thinking_enabled,
            request_timeout=max(60, args.timeout * 2),
        )
        print(f"Using DeepSeekModel: {model.model_name} @ {model.client.base_url} "
              f"thinking={model.thinking_enabled}", flush=True)
    elif args.api_provider == "glm":
        api_key = args.api_key or os.environ.get("GLM_API_KEY")
        if not api_key:
            raise SystemExit("--api-provider glm requires --api-key or GLM_API_KEY env var")
        model = DeepSeekModel(
            base_url=args.model_base_url if args.model_base_url != "http://localhost:8192/v1"
                else "https://open.bigmodel.cn/api/paas/v4",
            model_name=args.model_name if (args.model_name.startswith("glm") or "glm" in args.model_name.lower())
                else "glm-5.1",
            api_key=api_key,
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_model_tokens,
            thinking_enabled=args.thinking_enabled,
            request_timeout=max(60, args.timeout * 2),
        )
        print(f"Using GLM (via DeepSeekModel shim): {model.model_name} @ {model.client.base_url} "
              f"thinking={model.thinking_enabled}", flush=True)
    else:
        model = VLLMModel(
            base_url=args.model_base_url,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=0.95,
            top_logprobs=0,
            max_tokens=args.max_model_tokens,
        )
        print(f"Using VLLMModel: {model.model_name} @ {args.model_base_url}", flush=True)

    instances = load_swebench_verified(
        max_instances=None if args.instances else args.max_instances + args.start_index,
        instance_ids=args.instances,
    )
    if args.instances is None:
        instances = instances[args.start_index:args.start_index + args.max_instances]
    if args.design == "prefix-fork":
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
            shuffled_lib=shuffled_lib,
            idf=idf,
            out_path=out_path,
            prefix_log_path=prefix_log_path,
            done=done,
            trap_diagnosis_map=trap_diagnosis_map,
        )
    elif args.design == "probe-fork":
        print(
            f"Running {len(instances)} instances in probe-fork (Option C) mode "
            f"→ 2 arms per (iid,seed): baseline_optC + retrace_optC",
            flush=True,
        )
        written = run_probe_then_retrace_fork(
            args=args,
            model=model,
            instances=instances,
            trap_lib=trap_lib,
            core_lib=core_lib,
            shuffled_lib=shuffled_lib,
            idf=idf,
            out_path=out_path,
            done=done,
            trap_diagnosis_map=trap_diagnosis_map,
        )
    else:
        print(f"Running {len(instances)} instances × {len(args.arms)} arms "
              f"= {len(instances) * len(args.arms)} rollouts", flush=True)
        written = run_independent_rollouts(
            args=args,
            model=model,
            instances=instances,
            trap_lib=trap_lib,
            core_lib=core_lib,
            shuffled_lib=shuffled_lib,
            idf=idf,
            out_path=out_path,
            done=done,
            trap_diagnosis_map=trap_diagnosis_map,
        )

    print(f"\nDone. Wrote {written} new rollouts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
