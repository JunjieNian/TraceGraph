"""Key-set extraction and IDF-weighted Jaccard distance for agent traces.

Replaces activation-based key sets from SliceGraph with observable symbolic
action-observation key sets extracted from SWE agent trajectories.

Key types:
    TOOL:{name}      — tool/function name from assistant tool_calls
    CMD:{class}      — classified bash command (pytest, grep, sed, etc.)
    OBS:{pattern}    — error/outcome patterns from observations
    FILE_EXT:{ext}   — file extension of edited/viewed files
    FILE_PATH:{path} — normalised file path
    DIFF:{type}      — edit operation type (add_function, modify_function, etc.)
    PHASE:{label}    — temporal phase (early, mid, late)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Bash command classification ──────────────────────────────────────

_CMD_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("pytest",  re.compile(r"\b(pytest|py\.test)\b")),
    ("python",  re.compile(r"\b(python3?|ipython)\b")),
    ("grep",    re.compile(r"\b(grep|rg|ag|ack)\b")),
    ("find",    re.compile(r"\b(find|fd|locate)\b")),
    ("sed",     re.compile(r"\b(sed)\b")),
    ("cat",     re.compile(r"\b(cat|head|tail|less|more)\b")),
    ("pip",     re.compile(r"\b(pip3?|conda)\s+install\b")),
    ("git",     re.compile(r"\b(git)\b")),
    ("cd",      re.compile(r"^\s*cd\b")),
    ("ls",      re.compile(r"\b(ls|dir)\b")),
    ("echo",    re.compile(r"\b(echo|printf)\b")),
    ("mkdir",   re.compile(r"\b(mkdir)\b")),
    ("rm",      re.compile(r"\b(rm|rmdir)\b")),
    ("curl",    re.compile(r"\b(curl|wget)\b")),
]

# ── Observation error patterns ───────────────────────────────────────

_OBS_ERROR_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("OBS:AssertionError",    re.compile(r"AssertionError|assert\s+.*failed", re.I)),
    ("OBS:ImportError",       re.compile(r"ImportError|ModuleNotFoundError", re.I)),
    ("OBS:SyntaxError",       re.compile(r"SyntaxError", re.I)),
    ("OBS:NameError",         re.compile(r"NameError", re.I)),
    ("OBS:TypeError",         re.compile(r"TypeError", re.I)),
    ("OBS:ValueError",        re.compile(r"ValueError", re.I)),
    ("OBS:AttributeError",    re.compile(r"AttributeError", re.I)),
    ("OBS:KeyError",          re.compile(r"KeyError", re.I)),
    ("OBS:IndexError",        re.compile(r"IndexError", re.I)),
    ("OBS:FileNotFoundError", re.compile(r"FileNotFoundError|No such file", re.I)),
    ("OBS:PermissionError",   re.compile(r"PermissionError|Permission denied", re.I)),
    ("OBS:TimeoutError",      re.compile(r"TimeoutError|timed?\s*out", re.I)),
    ("OBS:RuntimeError",      re.compile(r"RuntimeError", re.I)),
    ("OBS:OSError",           re.compile(r"OSError|IOError", re.I)),
]

_OBS_OUTCOME_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("OBS:test_passed",  re.compile(r"\bpassed\b.*\btest", re.I)),
    ("OBS:test_failed",  re.compile(r"\bfailed\b.*\btest|\bFAILED\b", re.I)),
    ("OBS:test_error",   re.compile(r"\bERROR\b.*\btest|test.*\bERROR\b", re.I)),
    ("OBS:traceback",    re.compile(r"Traceback \(most recent call last\)")),
    ("OBS:success",      re.compile(r"\bsuccess(?:ful(?:ly)?)?\b", re.I)),
]

# ── Diff operation patterns ──────────────────────────────────────────

_DIFF_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("DIFF:add_import",     re.compile(r"^\+\s*(import |from .* import )", re.M)),
    ("DIFF:add_function",   re.compile(r"^\+\s*def\s+\w+", re.M)),
    ("DIFF:add_class",      re.compile(r"^\+\s*class\s+\w+", re.M)),
    ("DIFF:add_condition",  re.compile(r"^\+\s*(if |elif |else:)", re.M)),
    ("DIFF:add_try",        re.compile(r"^\+\s*(try:|except |finally:)", re.M)),
    ("DIFF:modify_function", re.compile(r"^[-+]\s*def\s+\w+", re.M)),
    ("DIFF:add_return",     re.compile(r"^\+\s*return\b", re.M)),
    ("DIFF:add_assert",     re.compile(r"^\+\s*assert\b", re.M)),
]

# ── File path normalisation ─────────────────────────────────────────

_PATH_RE = re.compile(
    r"""(?:^|[\s"'(])(/[^\s"')]+\.\w+)""",
    re.M,
)


def _classify_command(cmd: str) -> str:
    """Classify a bash command into one of ~15 categories."""
    for label, pat in _CMD_PATTERNS:
        if pat.search(cmd):
            return label
    return "other"


def _normalise_path(path: str) -> str:
    """Normalise a file path by keeping only last 2 components."""
    parts = path.strip().rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _extract_extension(path: str) -> Optional[str]:
    """Extract file extension from a path."""
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if len(ext) <= 6 and ext.isalnum():
            return ext
    return None


# ═══════════════════════════════════════════════════════════════════════
# Public key-extraction functions
# ═══════════════════════════════════════════════════════════════════════

def extract_action_keys(message: dict) -> Set[str]:
    """Extract ACTION/TOOL/CMD keys from an assistant message with tool_calls."""
    keys: Set[str] = set()
    tool_calls = message.get("tool_calls", [])
    if not tool_calls and message.get("function_call"):
        tool_calls = [{"function": message["function_call"]}]

    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        if name:
            keys.add(f"TOOL:{name}")

        args = func.get("arguments", "")
        if isinstance(args, str):
            args_str = args
        else:
            args_str = str(args)

        # Classify bash commands
        if "bash" in name.lower() or "execute" in name.lower():
            cmd = args_str
            cmd_class = _classify_command(cmd)
            keys.add(f"CMD:{cmd_class}")

        # Detect action types from arguments
        if "str_replace" in name.lower() or "edit" in name.lower():
            keys.add("ACTION:edit")
        elif "create" in name.lower() or "write" in name.lower():
            keys.add("ACTION:create")
        elif "view" in name.lower() or "read" in name.lower():
            keys.add("ACTION:view")
        elif "search" in name.lower() or "grep" in name.lower():
            keys.add("ACTION:search")

    return keys


def extract_observation_keys(message: dict) -> Set[str]:
    """Extract OBS keys from tool/user response messages."""
    keys: Set[str] = set()
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        content = str(content)

    for label, pat in _OBS_ERROR_PATTERNS:
        if pat.search(content):
            keys.add(label)

    for label, pat in _OBS_OUTCOME_PATTERNS:
        if pat.search(content):
            keys.add(label)

    return keys


def extract_file_keys(message: dict) -> Set[str]:
    """Extract FILE_EXT and FILE_PATH keys from file operations."""
    keys: Set[str] = set()
    tool_calls = message.get("tool_calls", [])
    if not tool_calls and message.get("function_call"):
        tool_calls = [{"function": message["function_call"]}]

    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "")
        if isinstance(args, dict):
            args_str = str(args)
        else:
            args_str = str(args)

        for match in _PATH_RE.finditer(args_str):
            path = match.group(1)
            ext = _extract_extension(path)
            if ext:
                keys.add(f"FILE_EXT:{ext}")
            norm = _normalise_path(path)
            if norm:
                keys.add(f"FILE_PATH:{norm}")

        # Also check common argument keys
        if isinstance(args, dict):
            for key in ("path", "file_path", "file", "filename"):
                val = args.get(key, "")
                if val and isinstance(val, str):
                    ext = _extract_extension(val)
                    if ext:
                        keys.add(f"FILE_EXT:{ext}")
                    norm = _normalise_path(val)
                    if norm:
                        keys.add(f"FILE_PATH:{norm}")

    return keys


def extract_diff_keys(message: dict) -> Set[str]:
    """Extract DIFF keys from edit/patch operations."""
    keys: Set[str] = set()
    tool_calls = message.get("tool_calls", [])
    if not tool_calls and message.get("function_call"):
        tool_calls = [{"function": message["function_call"]}]

    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "")
        if isinstance(args, dict):
            # str_replace_editor-style: look at new_str / old_str
            new_str = args.get("new_str", "") or args.get("replacement", "")
            old_str = args.get("old_str", "") or args.get("original", "")
            diff_text = f"+{new_str}\n-{old_str}"
        else:
            diff_text = str(args)

        for label, pat in _DIFF_PATTERNS:
            if pat.search(diff_text):
                keys.add(label)

    return keys


def extract_slice_keys(
    action_msg: dict,
    obs_msg: dict,
    progress: float,
) -> Set[str]:
    """Combine all key extractors for one action-observation pair + PHASE."""
    keys: Set[str] = set()
    keys |= extract_action_keys(action_msg)
    keys |= extract_observation_keys(obs_msg)
    keys |= extract_file_keys(action_msg)
    keys |= extract_diff_keys(action_msg)

    # Add phase key
    if progress < 1.0 / 3.0:
        keys.add("PHASE:early")
    elif progress < 2.0 / 3.0:
        keys.add("PHASE:mid")
    else:
        keys.add("PHASE:late")

    return keys


# ═══════════════════════════════════════════════════════════════════════
# IDF weighting and distance computation
# ═══════════════════════════════════════════════════════════════════════

def build_idf_weights(all_key_sets: List[Set[str]]) -> Dict[str, float]:
    """Compute IDF weights: idf(k) = log((1 + |V|) / (1 + df(k)))."""
    n = len(all_key_sets)
    if n == 0:
        return {}

    df: Dict[str, int] = {}
    for ks in all_key_sets:
        for k in ks:
            df[k] = df.get(k, 0) + 1

    idf: Dict[str, float] = {}
    for k, freq in df.items():
        idf[k] = math.log((1.0 + n) / (1.0 + freq))
    return idf


def weighted_jaccard(
    keys_i: Set[str],
    keys_j: Set[str],
    idf: Dict[str, float],
) -> float:
    """IDF-weighted Jaccard: sum(idf for intersection) / sum(idf for union)."""
    union = keys_i | keys_j
    if not union:
        return 0.0
    inter = keys_i & keys_j
    w_inter = sum(idf.get(k, 1.0) for k in inter)
    w_union = sum(idf.get(k, 1.0) for k in union)
    if w_union <= 0:
        return 0.0
    return w_inter / w_union


def compute_pairwise_distances(
    key_sets: List[Set[str]],
    idf: Dict[str, float],
) -> np.ndarray:
    """Compute full pairwise distance matrix: 1 - weighted_jaccard."""
    n = len(key_sets)
    dist = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            sim = weighted_jaccard(key_sets[i], key_sets[j], idf)
            d = 1.0 - sim
            dist[i, j] = d
            dist[j, i] = d
    return dist


def compute_knn(
    distances: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (knn_indices, knn_dists) arrays from distance matrix.

    For each node, finds the k nearest neighbours (excluding self).
    """
    n = distances.shape[0]
    actual_k = min(k, n - 1)
    if actual_k <= 0:
        return np.zeros((n, 0), dtype=np.int32), np.zeros((n, 0), dtype=np.float32)

    knn_indices = np.zeros((n, actual_k), dtype=np.int32)
    knn_dists = np.zeros((n, actual_k), dtype=np.float32)

    for i in range(n):
        row = distances[i].copy()
        row[i] = np.inf  # exclude self
        idx = np.argpartition(row, actual_k)[:actual_k]
        idx = idx[np.argsort(row[idx])]
        knn_indices[i] = idx
        knn_dists[i] = row[idx]

    return knn_indices, knn_dists
