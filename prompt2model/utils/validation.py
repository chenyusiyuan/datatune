"""Validation helpers for LLM-generated samples.

This module provides:
- Strict JSON extraction for ``{\"input\": ..., \"output\": ...}`` pairs.
- Simple Chinese language detection.
- Length and sensitive-word filters.
- N-gram / ROUGE-L-style novelty checks against existing texts.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, List, Sequence, Tuple

from prompt2model.utils import get_formatted_logger
from prompt2model.utils.parse_responses import extract_content, find_and_parse_json

logger = get_formatted_logger("Validation")


def parse_strict_io_json(response: Any) -> dict[str, str] | None:
    """Parse a strict ``{\"input\": ..., \"output\": ...}`` object from an LLM response.

    The function uses the existing JSON extraction helper but enforces:
    - Both fields must be present and non-empty strings.
    - Extra top-level keys are ignored.
    """
    try:
        parsed = find_and_parse_json(response, ["input", "output"], [])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse JSON from response: %s", exc)
        return None

    if parsed is None:
        return None

    inp = parsed.get("input")
    out = parsed.get("output")
    if not isinstance(inp, str) or not isinstance(out, str):
        return None
    inp = inp.strip()
    out = out.strip()
    if not inp or not out:
        return None

    return {"input": inp, "output": out}


def is_chinese_text(text: str, threshold: float = 0.5) -> bool:
    """Heuristic check whether text is predominantly Chinese."""
    if not text:
        return False
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ratio = chinese_chars / max(1, len(text))
    return ratio >= threshold


def length_ok(text: str, min_len: int, max_len: int) -> bool:
    """Check if length of text (in characters) is within [min_len, max_len]."""
    n = len(text or "")
    return min_len <= n <= max_len


def contains_sensitive(text: str, keywords: Iterable[str]) -> bool:
    """Return True if text contains any sensitive keyword (case-insensitive)."""
    lowered = (text or "").lower()
    for kw in keywords:
        if kw and str(kw).lower() in lowered:
            return True
    return False


def _lcs_length(a: str, b: str) -> int:
    """Compute length of the Longest Common Subsequence between two strings."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    # Simple DP; fine for moderate lengths typical of answers.
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l_f1(a: str, b: str) -> float:
    """Compute a simple ROUGE-L F1 score between two strings."""
    a = a or ""
    b = b or ""
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    prec = lcs / len(b)
    rec = lcs / len(a)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def passes_ngram_novelty(
    text: str,
    existing_texts: Sequence[str],
    rougeL_threshold: float,
) -> bool:
    """Check that text is sufficiently novel vs. existing texts.

    Currently implemented as ROUGE-L F1 max similarity < threshold.
    """
    if not existing_texts:
        return True
    max_sim = 0.0
    for prev in existing_texts:
        sim = rouge_l_f1(text, prev)
        max_sim = max(max_sim, sim)
        if max_sim >= rougeL_threshold:
            return False
    return True


def extract_judge_index(response: Any) -> int:
    """Extract an integer index from a judge model response, or -1 on failure."""
    raw = extract_content(response)
    m = re.search(r"-?\d+", raw)
    if not m:
        return -1
    try:
        return int(m.group(0))
    except ValueError:
        return -1


__all__ = [
    "parse_strict_io_json",
    "is_chinese_text",
    "length_ok",
    "contains_sensitive",
    "rouge_l_f1",
    "passes_ngram_novelty",
    "extract_judge_index",
]

