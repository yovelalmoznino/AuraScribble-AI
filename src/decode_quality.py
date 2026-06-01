"""Detect degenerate decode outputs (template collapse), not just repetition loops."""

from __future__ import annotations

import re

# Greedy-decode junk seen on IAM when the model ignores strokes.
_EN_TEMPLATE_RE = re.compile(
    r"^(wan\s+)?ther[\w.]*\.?$|^the\s+and[\w.]*\.?$|^whe\s+tind$|^wac\{1\}$",
    re.I,
)
_MATH_JUNK_RE = re.compile(r"^-1\}?\)?$|^\\frac\{1\}$|^\{1\}$")


def _overlap_ratio(pred: str, truth: str) -> float:
    if not truth:
        return 0.0
    ps = set(pred.lower())
    ts = set(truth.lower())
    if not ts:
        return 0.0
    return len(ps & ts) / len(ts)


def is_template_collapse(pred: str, truth: str | None = None, mode: str | None = None) -> bool:
    """
    True when pred is a known failure pattern or implausibly short vs long truth.
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

    # English / IAM: template phrases or tiny pred on long truth
    if mode_key in ("text", "english", "auto") or (truth and truth_len > 15):
        if _EN_TEMPLATE_RE.match(pl.strip()):
            return True
        if pl.startswith("the and") and (truth_len == 0 or len(p) < max(16, int(truth_len * 0.4))):
            return True
        if pl.startswith("the ") and len(p) < 12 and truth_len > 20:
            return True
        if truth_len > 30 and len(p) < 22 and _overlap_ratio(p, truth or "") < 0.35:
            return True

    # Hebrew: one–two letters when truth is a word or sentence
    if mode_key in ("hebrew", "mixed") or (truth and any("\u0590" <= c <= "\u05FF" for c in truth)):
        if truth_len >= 5 and len(p) <= 2 and any("\u0590" <= c <= "\u05FF" for c in p):
            return True

    # Math: fragments like -1} on real expressions
    if mode_key == "math" or (truth and ("\\" in truth or "^" in truth or "=" in truth)):
        compact_p = re.sub(r"\s+", "", p)
        compact_t = re.sub(r"\s+", "", truth or "")
        if _MATH_JUNK_RE.match(compact_p) and truth_len > 6:
            return True
        if truth_len > 8 and compact_p in ("\\frac{1}",) and compact_p != compact_t:
            return True
        if truth_len > 10 and len(compact_p) <= 5 and ("=" in compact_t or "\\frac" in compact_t):
            return True

    words = pl.split()
    if len(words) >= 5 and len(set(words)) <= 2 and "the" in words:
        return True

    return False


def batch_template_collapse_rate(predictions: list[str], truths: list[str], modes: list[str]) -> float:
    if not predictions:
        return 0.0
    n = sum(
        1
        for p, t, m in zip(predictions, truths, modes)
        if is_template_collapse(p, t, m)
    )
    return n / len(predictions)


def passes_export_gate(
    *,
    cer_mean: float,
    val_cer: float,
    collapse_count: int,
    sample_count: int,
    max_collapse_rate: float = 0.08,
    max_cer_mean: float = 0.35,
    max_val_cer: float = 0.5,
    per_mode_val_cer: dict[str, float] | None = None,
    per_mode_collapse: dict[str, tuple[int, int]] | None = None,
    max_per_mode_val_cer: float = 0.5,
    max_per_mode_collapse_rate: float = 0.08,
    require_modes: tuple[str, ...] = ("english", "hebrew", "math"),
) -> tuple[bool, str]:
    """
    Global gate plus optional per-mode gates.

    per_mode_val_cer:  {mode: val_cer} per language/content mode.
    per_mode_collapse: {mode: (collapse_count, sample_count)} per mode.

    When per-mode data is supplied, EVERY mode listed in require_modes (that has
    data) must individually satisfy val_cer <= max_per_mode_val_cer and
    collapse_rate <= max_per_mode_collapse_rate. This stops a model that looks
    fine on the (English-dominated) global average from shipping while a minority
    mode like hebrew is silently collapsing.
    """
    if sample_count <= 0:
        return False, "no samples"
    rate = collapse_count / sample_count
    if val_cer > max_val_cer:
        return False, f"val_cer {val_cer:.3f} > {max_val_cer}"
    if cer_mean > max_cer_mean:
        return False, f"cer_mean {cer_mean:.3f} > {max_cer_mean}"
    if rate > max_collapse_rate:
        return False, f"collapse rate {rate:.1%} > {max_collapse_rate:.0%}"

    per_mode_val_cer = {(k or "").lower(): v for k, v in (per_mode_val_cer or {}).items()}
    per_mode_collapse = {(k or "").lower(): v for k, v in (per_mode_collapse or {}).items()}

    for mode in require_modes:
        mode = mode.lower()
        if mode in per_mode_val_cer:
            mode_cer = per_mode_val_cer[mode]
            if mode_cer > max_per_mode_val_cer:
                return False, f"{mode} val_cer {mode_cer:.3f} > {max_per_mode_val_cer}"
        if mode in per_mode_collapse:
            c_count, c_total = per_mode_collapse[mode]
            if c_total > 0:
                mode_rate = c_count / c_total
                if mode_rate > max_per_mode_collapse_rate:
                    return False, (
                        f"{mode} collapse {mode_rate:.1%} > {max_per_mode_collapse_rate:.0%}"
                    )

    return True, "ok"
