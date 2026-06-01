"""Split train manifest into short-line (curriculum phase 1) and full train."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import read_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="output/train.jsonl")
    parser.add_argument("--short-out", default="output/train_short.jsonl")
    parser.add_argument("--max-chars", type=int, default=32)
    args = parser.parse_args()

    train_path = Path(args.train)
    samples = read_manifest(train_path)
    short = [s for s in samples if len(s.text.strip()) <= args.max_chars]
    write_manifest(Path(args.short_out), short)
    print(f"Wrote {len(short)} / {len(samples)} samples to {args.short_out} (max_chars={args.max_chars})")


if __name__ == "__main__":
    main()
