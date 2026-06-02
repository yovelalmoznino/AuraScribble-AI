from __future__ import annotations

import re


def cer(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    dp = [[0] * (len(truth) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(truth) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(truth) + 1):
            cost = 0 if pred[i - 1] == truth[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(1, len(truth))


def wer(pred: str, truth: str) -> float:
    p_words = pred.split()
    t_words = truth.split()
    if not t_words:
        return 0.0 if not p_words else 1.0
    dp = [[0] * (len(t_words) + 1) for _ in range(len(p_words) + 1)]
    for i in range(len(p_words) + 1):
        dp[i][0] = i
    for j in range(len(t_words) + 1):
        dp[0][j] = j
    for i in range(1, len(p_words) + 1):
        for j in range(1, len(t_words) + 1):
            cost = 0 if p_words[i - 1] == t_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(1, len(t_words))


def normalize_math_expression(text: str) -> str:
    compact = re.sub(r"\s+", "", text or "")
    compact = compact.replace("\\left", "").replace("\\right", "")
    compact = compact.replace("−", "-").replace("×", "*").replace("÷", "/")
    return compact


def math_symbol_f1(pred: str, truth: str) -> float:
    symbols = set("+-=*/^()[]{}<>≤≥≠≈×÷")
    p = [c for c in pred if c in symbols]
    t = [c for c in truth if c in symbols]
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    overlap = sum(min(p.count(ch), t.count(ch)) for ch in set(p) | set(t))
    precision = overlap / max(1, len(p))
    recall = overlap / max(1, len(t))
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def expression_exact_match(pred: str, truth: str) -> float:
    p = pred.replace(" ", "")
    t = truth.replace(" ", "")
    return 1.0 if p == t else 0.0


def expression_normalized_exact_match(pred: str, truth: str) -> float:
    p = normalize_math_expression(pred)
    t = normalize_math_expression(truth)
    return 1.0 if p == t else 0.0
