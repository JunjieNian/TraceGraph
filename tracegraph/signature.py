"""IDF-weighted Jaccard distance over symbolic key sets.

The offline pipeline builds key sets in
``scripts/pipeline/extract_signatures.py``; this module supplies the shared
IDF weighting, distance, and kNN helpers, plus the observation-key
extractor that the live SWE detector applies to raw tool output.

Runtime observation keys:
    OBS:{pattern}    — exception names and test/traceback/success patterns
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Tuple

import numpy as np


# ── Observation patterns ─────────────────────────────────────────────

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


def extract_observation_keys(message: dict) -> Set[str]:
    """Extract OBS keys from a tool/user response message."""
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
