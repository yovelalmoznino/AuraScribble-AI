"""
Generate synthetic Hebrew handwriting JSONL for training.

Renders each sentence with a Hebrew-capable TrueType font, then SKELETONIZES the
raster (Zhang-Suen thinning) and traces the centerline into ordered pen strokes.
This produces single-stroke-width, pen-like trajectories - much closer to real
ink than the old font-OUTLINE approach, which traced the contour of every glyph.

Hebrew is rendered in visual right-to-left order (PIL has no BiDi engine) and the
skeleton is traced right-to-left, so synthetic strokes match how Hebrew is
actually written. The text label stays in logical order.

Not a substitute for real ink, but a much better token-coverage booster than
before, especially when IAM English / CROHME math dominate the manifest.

Usage:
  pip install pillow numpy
  python src/generate_synthetic_hebrew.py \\
    --output data/raw/synthetic_hebrew/hebrew_synthetic.jsonl \\
    --variants 6

Kaggle (after setup cell):
  !python src/generate_synthetic_hebrew.py \\
    --output {DATA_RAW}/synthetic_hebrew/hebrew_synthetic.jsonl \\
    --variants 6 --sentences configs/hebrew_sentences.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import HandwritingSample, write_manifest
from skeletonize import skeleton_to_strokes, to_binary


# Candidate Hebrew-capable fonts (Windows + Linux/Kaggle). DejaVuSans (shipped
# with matplotlib) is the reliable cross-platform fallback - it has Hebrew.
_FONT_FILENAMES = (
    "david.ttf", "davidbd.ttf", "frank.ttf", "gisha.ttf", "mriam.ttf",
    "rod.ttf", "arial.ttf", "segoeui.ttf", "tahoma.ttf",
    "NotoSansHebrew-Regular.ttf", "NotoSansHebrew_Condensed-Regular.ttf",
    "DejaVuSans.ttf", "FreeSans.ttf", "Arial.ttf",
)

_FONT_DIRS = (
    r"C:\Windows\Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    "/kaggle/input",
)

_HEBREW_BLOCK = ("\u0590", "\u05FF")
_FINAL_FORMS = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}


def _is_hebrew(ch: str) -> bool:
    return _HEBREW_BLOCK[0] <= ch <= _HEBREW_BLOCK[1]


def _font_has_hebrew(path: str, size: int = 40) -> bool:
    try:
        font = ImageFont.truetype(path, size)
        img = Image.new("L", (size * 2, size * 2), 255)
        ImageDraw.Draw(img).text((4, 4), "א", font=font, fill=0)
        return int((np.asarray(img) < 128).sum()) > 5
    except Exception:
        return False


def discover_fonts(explicit: str | None = None) -> list[str]:
    """Return a list of usable Hebrew TTF paths (cross-platform)."""
    found: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        if path and path not in seen and Path(path).exists() and _font_has_hebrew(path):
            seen.add(path)
            found.append(path)

    if explicit:
        add(explicit)

    for d in _FONT_DIRS:
        base = Path(d)
        if not base.exists():
            continue
        for name in _FONT_FILENAMES:
            # direct hit
            add(str(base / name))
        # shallow recursive search for the known filenames (Linux nests fonts)
        try:
            for p in base.rglob("*.ttf"):
                if p.name in _FONT_FILENAMES:
                    add(str(p))
        except Exception:
            continue

    # matplotlib's bundled DejaVuSans as a last resort
    if not found:
        try:
            import matplotlib  # noqa

            add(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
        except Exception:
            pass

    return found


def apply_final_hebrew_letters(text: str) -> str:
    """Use sofit letters at the end of Hebrew words (basic heuristic)."""
    parts: list[str] = []
    word: list[str] = []

    def flush() -> None:
        if word:
            w = "".join(word)
            if len(w) > 1 and w[-1] in _FINAL_FORMS:
                w = w[:-1] + _FINAL_FORMS[w[-1]]
            parts.append(w)
            word.clear()

    for ch in text:
        if _is_hebrew(ch):
            word.append(ch)
        else:
            flush()
            parts.append(ch)
    flush()
    return "".join(parts)


def to_visual_rtl(text: str) -> str:
    """
    Poor-man's BiDi: render Hebrew right-to-left while keeping embedded
    Latin/digit runs left-to-right. Good enough for synthetic data; prefers the
    python-bidi library when it is installed.
    """
    try:
        from bidi.algorithm import get_display  # type: ignore

        return get_display(text)
    except Exception:
        pass

    rev = text[::-1]
    out: list[str] = []
    i = 0
    n = len(rev)
    while i < n:
        c = rev[i]
        if c.isascii() and c.isalnum():
            j = i
            while j < n and rev[j].isascii() and rev[j].isalnum():
                j += 1
            out.append(rev[i:j][::-1])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _jitter(points: list[list[float]], rng: random.Random) -> list[list[float]]:
    if not points:
        return points
    scale = rng.uniform(0.9, 1.12)
    dx = rng.uniform(-12, 12)
    dy = rng.uniform(-8, 8)
    noise = rng.uniform(0.4, 1.6)
    out: list[list[float]] = []
    for x, y, t in points:
        nx = x * scale + dx + rng.gauss(0, noise)
        ny = y * scale + dy + rng.gauss(0, noise)
        out.append([round(nx, 2), round(ny, 2), float(t)])
    return out


def render_to_points(
    visual_text: str,
    *,
    font_path: str,
    font_size: int,
    max_points: int,
) -> list[list[float]]:
    font = ImageFont.truetype(font_path, font_size)
    # measure
    tmp = Image.new("L", (8, 8), 255)
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), visual_text, font=font)
    w = max(8, bbox[2] - bbox[0] + font_size)
    h = max(8, bbox[3] - bbox[1] + font_size)
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((font_size // 2 - bbox[0], font_size // 4 - bbox[1]),
                             visual_text, font=font, fill=0)
    binary = to_binary(np.asarray(img))
    return skeleton_to_strokes(binary, max_points=max_points, rtl=True)


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
    font_paths: list[str],
    font_size_range: tuple[int, int],
) -> list[HandwritingSample]:
    rng = random.Random(seed)
    samples: list[HandwritingSample] = []
    skipped = 0

    for raw in sentences:
        logical = apply_final_hebrew_letters(raw)
        visual = to_visual_rtl(logical)
        for _ in range(variants):
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
                    text=logical,
                    mode="hebrew",
                )
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
    parser.add_argument("--variants", type=int, default=6, help="Augmented traces per sentence")
    parser.add_argument("--max-points", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--font", type=str, default=None, help="Explicit TTF path (optional)")
    parser.add_argument("--font-size-min", type=int, default=40)
    parser.add_argument("--font-size-max", type=int, default=72)
    args = parser.parse_args()

    font_paths = discover_fonts(args.font)
    if not font_paths:
        raise SystemExit(
            "No Hebrew-capable TTF font found. Pass --font <path-to-ttf> "
            "(e.g. a Noto Sans Hebrew or DejaVuSans.ttf)."
        )
    print(f"Using {len(font_paths)} font(s): {[Path(p).name for p in font_paths]}", flush=True)

    sentences = load_sentences(args.sentences)
    samples = generate_samples(
        sentences,
        variants=args.variants,
        max_points=args.max_points,
        seed=args.seed,
        font_paths=font_paths,
        font_size_range=(args.font_size_min, args.font_size_max),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, samples)
    print(f"Wrote {len(samples)} synthetic Hebrew samples -> {args.output}", flush=True)
    print(f"  ({len(sentences)} sentences x up to {args.variants} variants)", flush=True)


if __name__ == "__main__":
    main()
