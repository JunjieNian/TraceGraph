#!/usr/bin/env python3
"""Build a sidecar trap-diagnosis JSON for the RQ4 v2 stronger repair note.

The base trap library (`data/cxcmu/intervention/swebench_trap_library.json`)
stores only abstract key-sets per trap example. This script classifies
each example into one of a fixed set of "failure families" based purely
on its keys, then attaches a ~120-word pattern diagnosis to each family.

Crucially, the diagnosis text is GENERAL software-engineering advice
keyed off the *abstract action pattern* (read-then-stuck, blind-edit-
then-error, etc.) — it never reads the source task's problem statement,
gold patch, or verified-test list, so the repair note remains free of
oracle leakage.

Output:
  data/cxcmu/intervention/swebench_trap_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
TRAP_PATH = ROOT / "data/cxcmu/intervention/swebench_trap_library.json"
OUT_PATH = ROOT / "data/cxcmu/intervention/swebench_trap_diagnosis.json"
RESOURCE_TRAP_PATH = ROOT / "resources/swebench_detector/swebench_trap_library.json"


FAMILIES: Dict[str, Dict[str, str]] = {
    # E2 (conservative): the v1/v2/v3 "explore-and-search-elsewhere" advice
    # caused systematic over-editing on Qwen (it acted on the suggestion to
    # grep / look at sibling files and then made *extra* destructive edits).
    # These reformulations instead push the model to VERIFY the current edit
    # is correct using the narrowest available test, REVERT (not extend) if
    # not, and AVOID adding new edits before confirmation.
    "blind_edit_error": {
        "label": "blind-edit-then-error",
        "diagnosis": (
            "Past trajectories with this signature applied a sed/replace edit "
            "and the next observation contained an error. Do not add another "
            "edit yet. Before issuing any further changes: (a) run only the "
            "single narrowest failing test that is most directly relevant to "
            "what this region of the code does, (b) if the same error "
            "persists or a new one appears, REVERT this edit instead of "
            "writing a different one, and (c) do not edit any sibling file "
            "until the current location is verified to be the right one. "
            "Doing less and verifying is safer than doing more."
        ),
    },
    "blind_edit_silent": {
        "label": "blind-edit-silent-success",
        "diagnosis": (
            "Past trajectories with this signature applied an edit that "
            "appeared to succeed but then failed the task. Do not stack new "
            "edits on top. Before any other change: (a) run only the "
            "specific failing test cited by the issue, (b) compare the new "
            "failure (if any) to the original failure cited in the issue, "
            "(c) if the original failure is identical or unchanged, REVERT "
            "this edit and reconsider — adding more edits will compound the "
            "drift. Verify before extending."
        ),
    },
    "stuck_reading_error": {
        "label": "stuck-reading-error",
        "diagnosis": (
            "Past trajectories with this signature kept reading the same "
            "region after an error appeared instead of acting on it. Stop "
            "reading more. Make exactly ONE minimal edit to the smallest "
            "scope implicated by the error you have already seen, then "
            "immediately run only the failing test. Do not edit any other "
            "file in the same step. If the test outcome is unchanged, "
            "REVERT this edit; do not add a second edit."
        ),
    },
    "misread_partial_success": {
        "label": "misread-partial-success",
        "diagnosis": (
            "Past trajectories with this signature treated a partial "
            "`test passed` line as global success and submitted. Do not "
            "submit yet. Before any submit: (a) identify the specific test "
            "(or specific assertion in the issue) that proves the requested "
            "behavior changed, (b) run that exact test alone, (c) only if "
            "that test passes is the change complete. Do not add or revise "
            "edits in the meantime — verify the current state first."
        ),
    },
    "aimless_reading": {
        "label": "aimless-reading",
        "diagnosis": (
            "Past trajectories with this signature kept exploring instead "
            "of acting on what they already knew. Stop exploring. State, "
            "in one sentence, the single most likely root cause given the "
            "evidence already in front of you, then make ONE minimal edit "
            "to the smallest scope implicated by that hypothesis. Run only "
            "the narrowest test. Do not open additional files before that "
            "test runs; verify before broadening."
        ),
    },
    "premature_submit": {
        "label": "premature-submit",
        "diagnosis": (
            "Past trajectories with this signature submitted on a partial "
            "success and then failed the harness. Do not submit yet. "
            "Re-read the issue statement and identify the single behavior "
            "that must change, then run the test that exercises that "
            "behavior specifically. Only submit when that test passes. "
            "Do not add edits in the interim — verify the current state "
            "before any further action."
        ),
    },
    "unknown": {
        "label": "unknown-low-yield-pattern",
        "diagnosis": (
            "This pattern fired the trap detector but does not match a "
            "known sub-family. Be conservative: do not introduce a new "
            "edit on a different file in your next action. Verify the "
            "effect of your most recent change first by running the "
            "single narrowest relevant test, and revert that change if "
            "the test outcome is unchanged."
        ),
    },
}


def classify_trap(keys: List[str]) -> str:
    """Map a trap example's abstract keys to one of the FAMILIES."""
    s: Set[str] = set(keys or [])
    has_edit = "ACTION:edit" in s
    has_read = "ACTION:read" in s
    has_submit = "ACTION:submit" in s
    has_sed = "CMD:sed" in s
    has_cat = "CMD:cat" in s
    has_err = "OBS:error" in s or "OBS:traceback" in s
    has_succ = "OBS:success" in s
    has_test_passed = "OBS:test_passed" in s
    if has_submit:
        return "premature_submit"
    if has_edit and has_sed:
        return "blind_edit_error" if has_err else "blind_edit_silent"
    if has_read and has_cat:
        if has_err:
            return "stuck_reading_error"
        if has_test_passed:
            return "misread_partial_success"
        return "aimless_reading"
    return "unknown"


def main(trap_path: Path = TRAP_PATH, out_path: Path = OUT_PATH) -> None:
    trap_path = Path(trap_path)
    out_path = Path(out_path)
    if not trap_path.exists() and RESOURCE_TRAP_PATH.exists():
        trap_path = RESOURCE_TRAP_PATH
    trap = json.loads(trap_path.read_text())
    examples = trap.get("examples") or []
    by_index: Dict[str, str] = {}
    family_counts: Dict[str, int] = {fk: 0 for fk in FAMILIES}
    for i, e in enumerate(examples):
        keys = e.get("detection_keys") or e.get("keys") or []
        family_key = classify_trap(keys)
        by_index[str(i)] = family_key
        family_counts[family_key] = family_counts.get(family_key, 0) + 1

    out: Dict[str, Any] = {
        "schema": "swebench_trap_diagnosis.v1",
        "source_trap_library": str(trap_path.relative_to(ROOT) if trap_path.is_relative_to(ROOT) else trap_path),
        "n_traps": len(examples),
        "families": FAMILIES,
        "by_trap_index": by_index,
        "family_counts": family_counts,
        "notes": (
            "All diagnosis text is generic software-engineering advice keyed "
            "off the abstract action pattern; it does not read any source "
            "task's problem statement, gold patch, or verified-test list."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"family distribution: {family_counts}")
    # Length sanity
    for fk, fv in FAMILIES.items():
        print(f"  {fk:<28s}  diagnosis_chars={len(fv['diagnosis']):>4d}  count={family_counts.get(fk, 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trap-path", type=Path, default=TRAP_PATH)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(trap_path=args.trap_path, out_path=args.out_path)
