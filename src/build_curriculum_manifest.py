"""Build curriculum manifests: short, medium, IAM-long, and full train."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import HandwritingSample, read_manifest, write_manifest


def _text_len(sample: HandwritingSample) -> int:
    return len(sample.text.strip())


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
    args = parser.parse_args()

    samples = read_manifest(Path(args.train))
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
