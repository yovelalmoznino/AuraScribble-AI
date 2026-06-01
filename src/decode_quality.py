"""Detect degenerate decode outputs (template collapse), not just repetition loops."""

from __future__ import annotations

import re


def is_template_collapse(pred: str, truth: str | None = None, mode: str | None = None) -> bool:
    """
    True when pred is a known failure pattern (the and / single ה / \\frac{1} on everything).
    Used for sanity checks and export gates — stricter than CER alone.
    """
    p = pred.strip()
    if not p:
        return True

    pl = p.lower()
    truth_len = len((truth or "").strip())
    mode_key = (mode or "auto").lower()

    if "the the" in pl or pl.count("the the") >= 1:
        return True
    if "\\times \\times" in p:
        return True

    # English IAM template: short "the and..." on long truth
    if mode_key in ("text", "english", "auto") or (truth and truth_len > 15):
        if pl.startswith("the and") and (truth_len == 0 or len(p) < max(16, int(truth_len * 0.4))):
            return True
        if re.match(r"^the and[\w.]*\.?$", pl, re.I) and len(p) < 28:
            return True
        if pl.startswith("the ") and len(p) < 12 and truth_len > 20:
            return True

    # Hebrew: one–two letters when truth is a word or sentence
    if mode_key in ("hebrew", "mixed") or (truth and any("\u0590" <= c <= "\u05FF" for c in truth)):
        if truth_len >= 5 and len(p) <= 2 and any("\u0590" <= c <= "\u05FF" for c in p):
            return True

    # Math: default \\frac{1} on non-trivial truth
    if mode_key == "math" or (truth and ("\\" in truth or "^" in truth or "=" in truth)):
        compact_p = re.sub(r"\s+", "", p)
        compact_t = re.sub(r"\s+", "", truth or "")
        if truth_len > 8 and compact_p in ("\\frac{1}", "\\frac{1}") and compact_p != compact_t:
            return True
        if truth_len > 12 and compact_p.startswith("\\frac{1}") and len(compact_p) <= 8:
            return True

    words = pl.split()
    if len(words) >= 5 and len(set(words)) <= 2 and "the" in words:
        return True

    return False


def passes_export_gate(
    *,
    cer_mean: float,
    val_cer: float,
    collapse_count: int,
    sample_count: int,
    max_collapse_rate: float = 0.08,
    max_cer_mean: float = 0.35,
    max_val_cer: float = 0.5,
) -> tuple[bool, str]:
    if sample_count <= 0:
        return False, "no samples"
    rate = collapse_count / sample_count
    if val_cer > max_val_cer:
        return False, f"val_cer {val_cer:.3f} > {max_val_cer}"
    if cer_mean > max_cer_mean:
        return False, f"cer_mean {cer_mean:.3f} > {max_cer_mean}"
    if rate > max_collapse_rate:
        return False, f"collapse rate {rate:.1%} > {max_collapse_rate:.0%}"
    return True, "ok"
