"""
Generate synthetic Hebrew/English/Math mixed handwriting JSONL.

Rewritten to use the current skeleton-based API of generate_synthetic_hebrew
(render_to_points + _jitter + discover_fonts). Each mixed template is rendered
as a single line; DejaVu Sans covers Latin, Hebrew, and most math symbols, so
one render per template is enough.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from dataset import HandwritingSample, write_manifest
from generate_synthetic_hebrew import (
    _jitter,
    apply_final_hebrew_letters,
    discover_fonts,
    render_to_points,
    to_visual_rtl,
)


# Templates: (label, category). Category is informational only — we tag mode="mixed".
_MIXED_TEMPLATES: list[tuple[str, str]] = [
    # he + math
    ("פתרון: $x^2=4$", "he+math"),
    ("הערך של $x$ הוא 5", "he+math"),
    ("נגדיר $f(x)=ax+b$", "he+math"),
    ("הפתרון $x=5$ נכון", "he+math"),
    ("נתון $a+b=10$", "he+math"),
    ("חשב את $2+3$", "he+math"),
    ("הוכח כי $n>0$", "he+math"),
    ("מצא את $y$ כאשר $x=1$", "he+math"),
    ("הנגזרת היא $f'(x)=2x$", "he+math"),
    ("האינטגרל של $x$ הוא $x^2/2$", "he+math"),
    ("הסכום $\\sum_{i=1}^{n} i$", "he+math"),
    ("הזווית היא $90$ מעלות", "he+math"),
    ("הקוטר שווה ל $2r$", "he+math"),
    ("שטח המעגל $\\pi r^2$", "he+math"),
    ("הפונקציה $f(x)=x^2$", "he+math"),
    ("המשוואה $ax+b=c$", "he+math"),
    ("ענה $3+4=7$", "he+math"),
    ("הציון $95$ נכון", "he+math"),
    ("$2024$ הייתה שנה טובה", "he+math"),
    # he + en
    ("המילה hello נכונה", "he+en"),
    ("התשובה answer היא כן", "he+en"),
    ("שלום world", "he+en"),
    ("השם שלי Alice", "he+en"),
    ("המחשב Windows מהיר", "he+en"),
    ("הטלפון iPhone חדש", "he+en"),
    ("הספר Harry Potter מעניין", "he+en"),
    ("הוא שלח לי email בבוקר", "he+en"),
    ("פגשתי את David בקפה", "he+en"),
    ("הקובץ נקרא data.json", "he+en"),
    ("פתחתי את Chrome מהר", "he+en"),
    ("מחשב Mac חזק מאוד", "he+en"),
    ("הקוד כתוב ב Python", "he+en"),
    ("המודל TensorFlow רץ", "he+en"),
    # en + math
    ("Find $x$ when $y=2$", "en+math"),
    ("Let $f(x)=\\frac{a}{b}$", "en+math"),
    ("The value of $n$ is 3", "en+math"),
    ("Solve $x^2 + 2x = 0$", "en+math"),
    ("Given $a > 0$, find $a$", "en+math"),
    ("Prove that $n \\geq 1$", "en+math"),
    ("Compute $\\sum_{i=1}^{10} i$", "en+math"),
    ("Let $A = \\{1,2,3\\}$", "en+math"),
    ("Define $g(x) = x^3$", "en+math"),
    ("The slope is $m=2$", "en+math"),
    ("If $x=5$ then $y=10$", "en+math"),
    ("Area is $\\pi r^2$", "en+math"),
    ("Probability $P(A) = 0.5$", "en+math"),
    ("$x=5$", "en+math"),
    ("$y \\neq 0$", "en+math"),
    # he + en + math
    ("Given $n$ מספרים, sum them", "he+en+math"),
    ("Let $x$ be מספר שלם", "he+en+math"),
    ("The answer is $42$ בעברית", "he+en+math"),
    ("Solve $x^2=4$ בכיתה", "he+en+math"),
    ("Define function $f(x)$ פשוטה", "he+en+math"),
    ("Find $y$ if $x=3$ במשוואה", "he+en+math"),
    # short pure
    ("Hello world", "en"),
    ("Good morning", "en"),
    ("Thank you", "en"),
    ("OK", "en"),
    ("yes", "en"),
    ("no", "en"),
]


def _has_hebrew(text: str) -> bool:
    return any("֐" <= c <= "׿" for c in text)


def _prepare_visual(text: str) -> str:
    """If the label contains any Hebrew, reverse the whole line so the PIL renderer
    (which is left-to-right only) draws the Hebrew runs in visual order. The logical
    label kept in the JSONL stays unchanged."""
    if not _has_hebrew(text):
        return text
    logical = apply_final_hebrew_letters(text)
    return to_visual_rtl(logical)


def generate_mixed_samples(
    *,
    per_template: int,
    seed: int,
    font_paths: list[str],
    max_points: int = 260,
    font_size_range: tuple[int, int] = (22, 32),
) -> list[HandwritingSample]:
    rng = random.Random(seed)
    samples: list[HandwritingSample] = []
    skipped = 0
    for label, _cat in _MIXED_TEMPLATES:
        visual = _prepare_visual(label)
        for _ in range(per_template):
            font_path = rng.choice(font_paths)
            fs = rng.randint(*font_size_range)
            try:
                pts = render_to_points(
                    visual, font_path=font_path, font_size=fs, max_points=max_points
                )
            except Exception:
                pts = []
            if len(pts) < 8:
                skipped += 1
                continue
            pts = _jitter(pts, rng)
            samples.append(
                HandwritingSample(
                    points=[[float(p[0]), float(p[1]), float(p[2])] for p in pts],
                    text=label,
                    mode="mixed",
                )
            )
    if skipped:
        print(f"Skipped {skipped} empty/short renders", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic mixed Hebrew/English/Math JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_mixed/mixed_synthetic.jsonl"),
    )
    parser.add_argument("--per-template", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--font", default=None, help="Explicit TTF font path")
    parser.add_argument("--max-points", type=int, default=260)
    args = parser.parse_args()

    font_paths = discover_fonts(args.font)
    if not font_paths:
        raise SystemExit(
            "No Hebrew-capable TTF found. Install fonts-noto-core / fonts-dejavu, or pass --font."
        )
    print(f"Using {len(font_paths)} font(s): {font_paths[:3]}{'...' if len(font_paths) > 3 else ''}")

    samples = generate_mixed_samples(
        per_template=args.per_template,
        seed=args.seed,
        font_paths=font_paths,
        max_points=args.max_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, samples)
    print(f"Wrote {len(samples)} mixed samples -> {args.output}")


if __name__ == "__main__":
    main()
