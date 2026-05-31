from __future__ import annotations

import argparse
import random
from pathlib import Path

from dataset import HandwritingSample, write_manifest
from generate_synthetic_hebrew import _jitter, _normalize_canvas, _path_to_points, _pick_font, render_sentence

# Templates: (label, category)
_MIXED_TEMPLATES: list[tuple[str, str]] = [
    ("פתרון: $x^2=4$", "he+math"),
    ("הערך של $x$ הוא 5", "he+math"),
    ("נגדיר $f(x)=ax+b$", "he+math"),
    ("המילה hello נכונה", "he+en"),
    ("התשובה answer היא כן", "he+en"),
    ("שלום world", "he+en"),
    ("Find $x$ when $y=2$", "en+math"),
    ("Let $f(x)=\\frac{a}{b}$", "en+math"),
    ("The value of $n$ is 3", "en+math"),
    ("Given $n$ מספרים, sum them", "he+en+math"),
    ("Let $x$ be מספר שלם", "he+en+math"),
    ("The answer is $42$ בעברית", "he+en+math"),
    ("הפתרון $x=5$ נכון", "he+en+math"),
]


def _render_mixed_line(text: str, rng: random.Random, max_points: int = 260) -> list[list[float]]:
    """Render mixed script line: Hebrew segments + Latin + inline math symbols."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    font_he = _pick_font()
    font_en = FontProperties(family="DejaVu Sans")
    x_cursor = 0.0
    gap = 18.0
    all_strokes: list[list[list[float]]] = []

    parts: list[tuple[str, str]] = []
    buf = ""
    in_math = False
    for ch in text:
        if ch == "$":
            if buf:
                parts.append((buf, "math" if in_math else "text"))
                buf = ""
            in_math = not in_math
            continue
        buf += ch
    if buf:
        parts.append((buf, "math" if in_math else "text"))

    for segment, kind in parts:
        if not segment.strip():
            continue
        if kind == "math":
            prop = font_en
            size = rng.uniform(16, 24)
        elif any("\u0590" <= c <= "\u05FF" for c in segment):
            prop = font_he
            size = rng.uniform(20, 30)
        else:
            prop = font_en
            size = rng.uniform(18, 28)
        path = TextPath((x_cursor, 0), segment, size=size, prop=prop)
        pts = _path_to_points(path, max_points=max_points // max(1, len(parts)))
        if pts:
            xs = [p[0] for p in pts]
            span = max(xs) - min(xs) if xs else 0
            shifted = [[p[0], p[1], p[2]] for p in pts]
            all_strokes.append(shifted)
            x_cursor += span + gap

    flat: list[list[float]] = []
    t = 0.0
    for stroke in all_strokes:
        for i, (x, y, _) in enumerate(stroke):
            flat.append([x, y, t])
            t += 2.0 if i == 0 else 1.0

    if len(flat) < 8:
        return render_sentence(text.replace("$", ""), font_prop=font_he, font_size=22, max_points=max_points, rng=rng)

    flat = _normalize_canvas(flat)
    flat = _jitter(flat, rng)
    return [[float(p[0]), float(p[1]), float(p[2])] for p in flat]


def generate_mixed_samples(
    *,
    per_template: int,
    seed: int,
) -> list[HandwritingSample]:
    rng = random.Random(seed)
    samples: list[HandwritingSample] = []
    for label, _cat in _MIXED_TEMPLATES:
        for _ in range(per_template):
            pts = _render_mixed_line(label, rng)
            if len(pts) < 8:
                continue
            samples.append(HandwritingSample(points=pts, text=label, mode="mixed"))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic mixed Hebrew/English/Math JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_mixed/mixed_synthetic.jsonl"),
    )
    parser.add_argument("--per-template", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    samples = generate_mixed_samples(per_template=args.per_template, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, samples)
    print(f"Wrote {len(samples)} mixed samples -> {args.output}")


if __name__ == "__main__":
    main()
