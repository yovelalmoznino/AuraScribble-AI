"""Build curriculum manifests: short, medium, IAM-long, and full train."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import HandwritingSample, read_manifest, write_manifest


def _text_len(sample: HandwritingSample) -> int:
    return len(sample.text.strip())


def _dedup_key(sample: HandwritingSample) -> tuple:
    """Stable identity for a sample: mode + text + a cheap point signature."""
    pts = sample.points or []
    first = tuple(pts[0]) if pts else ()
    last = tuple(pts[-1]) if pts else ()
    return ((sample.mode or "auto").lower(), sample.text.strip(), len(pts), first, last)


def _is_single_line(sample: HandwritingSample) -> bool:
    text = sample.text or ""
    return "\n" not in text and "\r" not in text


def _clean_samples(samples: list[HandwritingSample]) -> list[HandwritingSample]:
    """Drop multi-line samples and exact duplicates (keep first occurrence)."""
    seen: set[tuple] = set()
    out: list[HandwritingSample] = []
    dropped_multiline = 0
    dropped_dup = 0
    for s in samples:
        if not _is_single_line(s):
            dropped_multiline += 1
            continue
        key = _dedup_key(s)
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        out.append(s)
    print(
        f"cleaned: kept {len(out)} (dropped {dropped_multiline} multi-line, {dropped_dup} duplicates)"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="output/train.jsonl")
    parser.add_argument("--short-out", default="output/train_short.jsonl")
    parser.add_argument("--medium-out", default="output/train_medium.jsonl")
    parser.add_argument("--iam-out", default="output/train_iam_long.jsonl")
    parser.add_argument("--max-chars-short", type=int, default=32)
    parser.add_argument("--medium-min-chars", type=int, default=33)
    parser.add_argument("--medium-max-chars", type=int, default=72)
    parser.add_argument("--iam-min-chars", type=int, default=24)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip dedup + single-line filtering (not recommended).",
    )
    parser.add_argument(
        "--rewrite-train",
        action="store_true",
        help="Also overwrite --train with the cleaned (deduped, single-line) samples.",
    )
    args = parser.parse_args()

    samples = read_manifest(Path(args.train))
    if not args.no_clean:
        samples = _clean_samples(samples)
        if args.rewrite_train:
            write_manifest(Path(args.train), samples)
            print(f"rewrote cleaned full train -> {args.train}")
    short = [s for s in samples if _text_len(s) <= args.max_chars_short]
    medium = [
        s
        for s in samples
        if args.medium_min_chars <= _text_len(s) <= args.medium_max_chars
    ]
    iam_long = [
        s
        for s in samples
        if _text_len(s) >= args.iam_min_chars
        and (s.mode or "").lower() in ("text", "english")
    ]

    write_manifest(Path(args.short_out), short)
    write_manifest(Path(args.medium_out), medium)
    write_manifest(Path(args.iam_out), iam_long)

    print(
        f"short (<={args.max_chars_short}): {len(short)} -> {args.short_out}\n"
        f"medium ({args.medium_min_chars}-{args.medium_max_chars}): {len(medium)} -> {args.medium_out}\n"
        f"iam_long (>={args.iam_min_chars}, text/english): {len(iam_long)} -> {args.iam_out}\n"
        f"full train: {len(samples)}"
    )


if __name__ == "__main__":
    main()
