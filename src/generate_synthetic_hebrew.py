"""
Generate synthetic Hebrew handwriting JSONL for training.

Renders sentence outlines with a Hebrew-capable font, samples pen轨迹,
adds jitter/scale variants. Not a substitute for real ink — but boosts
Hebrew token coverage when IAM/CROHME dominate the manifest.

Usage:
  pip install matplotlib
  python src/generate_synthetic_hebrew.py \\
    --output data/raw/synthetic_hebrew/hebrew_synthetic.jsonl \\
    --variants 4

Kaggle (after setup cell):
  !python src/generate_synthetic_hebrew.py \\
    --output {DATA_RAW}/synthetic_hebrew/hebrew_synthetic.jsonl \\
    --variants 5 --sentences configs/hebrew_sentences.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

# Matplotlib only required for this script
import matplotlib

matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from dataset import HandwritingSample, write_manifest

# MOVETO=1, LINETO=2, CLOSEPOLY=79
_MOVETO = MplPath.MOVETO
_LINETO = MplPath.LINETO
_CLOSEPOLY = MplPath.CLOSEPOLY

_HEBREW_FONTS = (
    "Segoe UI",
    "Arial",
    "David",
    "Nirmala UI",
    "Noto Sans Hebrew",
    "Rubik",
    "DejaVu Sans",
)


def _pick_font() -> FontProperties:
    for name in _HEBREW_FONTS:
        try:
            path = findfont(FontProperties(family=name), fallback_to_default=False)
            if path and "dejavu" not in path.lower() or name == "DejaVu Sans":
                return FontProperties(family=name)
        except Exception:
            continue
    return FontProperties()


def _path_to_points(path: MplPath, max_points: int) -> list[list[float]]:
    verts = path.vertices
    codes = path.codes
    strokes: list[list[list[float]]] = []
    current: list[list[float]] = []

    for (x, y), code in zip(verts, codes):
        if code == _MOVETO:
            if len(current) >= 2:
                strokes.append(current)
            current = [[float(x), float(-y), 1.0]]
        elif code == _LINETO:
            current.append([float(x), float(-y), 0.0])
        elif code == _CLOSEPOLY:
            if len(current) >= 2:
                strokes.append(current)
            current = []

    if len(current) >= 2:
        strokes.append(current)

    flat: list[list[float]] = []
    t = 0.0
    for stroke in strokes:
        for i, pt in enumerate(stroke):
            flat.append([pt[0], pt[1], t])
            t += 2.0 if i == 0 else 1.0

    if len(flat) < 4:
        return []

    if len(flat) > max_points:
        idx = np.linspace(0, len(flat) - 1, max_points, dtype=int)
        flat = [flat[i] for i in idx]
        flat[0][2] = 1.0
    return flat


def _normalize_canvas(points: list[list[float]], width: float = 900.0, height: float = 400.0) -> list[list[float]]:
    if not points:
        return points
    arr = np.asarray(points, dtype=np.float32)
    xs, ys = arr[:, 0], arr[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    margin = 40.0
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    out: list[list[float]] = []
    t = 0.0
    for x, y, pen in points:
        nx = (x - cx) * scale + width / 2
        ny = (y - cy) * scale + height / 2
        out.append([round(nx, 2), round(ny, 2), t])
        t += 2.0 if pen >= 0.5 else 1.0
    return out


def _jitter(points: list[list[float]], rng: random.Random) -> list[list[float]]:
    if not points:
        return points
    scale = rng.uniform(0.85, 1.15)
    noise = rng.uniform(0.5, 2.5)
    dx = rng.uniform(-15, 15)
    dy = rng.uniform(-10, 10)
    out: list[list[float]] = []
    t = 0.0
    for x, y, pen in points:
        nx = x * scale + dx + rng.gauss(0, noise)
        ny = y * scale + dy + rng.gauss(0, noise)
        pen_out = 1.0 if pen >= 0.5 and len(out) == 0 else (1.0 if pen >= 0.5 and out and rng.random() < 0.08 else 0.0)
        if pen_out >= 0.5 and out and (abs(nx - out[-1][0]) + abs(ny - out[-1][1])) < 3:
            pen_out = 0.0
        out.append([round(nx, 2), round(ny, 2), t])
        t += 2.0 if pen_out >= 0.5 else 1.0
    return out


def render_sentence(
    text: str,
    *,
    font_prop: FontProperties,
    font_size: float,
    max_points: int,
    rng: random.Random,
) -> list[list[float]]:
    path = TextPath((0, 0), text, size=font_size, prop=font_prop)
    pts = _path_to_points(path, max_points=max_points)
    pts = _normalize_canvas(pts)
    pts = _jitter(pts, rng)
    return pts


def load_sentences(path: Path) -> list[str]:
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"No sentences in {path}")
    return lines


def generate_samples(
    sentences: list[str],
    *,
    variants: int,
    max_points: int,
    seed: int,
    font_size_range: tuple[float, float],
) -> list[HandwritingSample]:
    rng = random.Random(seed)
    font_prop = _pick_font()
    print(f"Using font: {font_prop.get_name()}", flush=True)

    samples: list[HandwritingSample] = []
    skipped = 0
    for text in sentences:
        for _ in range(variants):
            fs = rng.uniform(*font_size_range)
            pts = render_sentence(
                text,
                font_prop=font_prop,
                font_size=fs,
                max_points=max_points,
                rng=rng,
            )
            if len(pts) < 8:
                skipped += 1
                continue
            pts_py = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
            samples.append(
                HandwritingSample(points=pts_py, text=text, mode="hebrew")
            )

    if skipped:
        print(f"Skipped {skipped} empty/short renders", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Hebrew JSONL for training")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_hebrew/hebrew_synthetic.jsonl"),
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path("configs/hebrew_sentences.txt"),
    )
    parser.add_argument("--variants", type=int, default=4, help="Augmented traces per sentence")
    parser.add_argument("--max-points", type=int, default=220)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--font-size-min", type=float, default=18.0)
    parser.add_argument("--font-size-max", type=float, default=32.0)
    args = parser.parse_args()

    sentences = load_sentences(args.sentences)
    samples = generate_samples(
        sentences,
        variants=args.variants,
        max_points=args.max_points,
        seed=args.seed,
        font_size_range=(args.font_size_min, args.font_size_max),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, samples)
    print(f"Wrote {len(samples)} synthetic Hebrew samples -> {args.output}", flush=True)
    print(f"  ({len(sentences)} sentences x {args.variants} variants)", flush=True)


if __name__ == "__main__":
    main()
