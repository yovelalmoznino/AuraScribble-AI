from __future__ import annotations

import argparse
import random
from pathlib import Path

from dataset import HandwritingSample, read_manifest, write_manifest


def split_manifest(
    source: Path,
    train_out: Path,
    val_out: Path,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    samples = read_manifest(source)
    if not samples:
        raise ValueError(f"No samples found in {source}")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_count = max(1, int(len(samples) * val_ratio))
    val_indices = set(indices[:val_count])

    train_samples: list[HandwritingSample] = []
    val_samples: list[HandwritingSample] = []
    for i, sample in enumerate(samples):
        if i in val_indices:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(train_out, train_samples)
    write_manifest(val_out, val_samples)
    return len(train_samples), len(val_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSONL manifest into train/val sets.")
    parser.add_argument("--source", default="data/processed/all.jsonl", help="Source manifest path")
    parser.add_argument("--train-out", default="data/processed/train.jsonl")
    parser.add_argument("--val-out", default="data/processed/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        # Fallback: use train.jsonl as source if all.jsonl is missing
        fallback = Path("data/processed/train.jsonl")
        if fallback.exists():
            print(f"Source {source} not found; using {fallback}")
            source = fallback
        else:
            raise FileNotFoundError(f"Neither {args.source} nor {fallback} exists")

    train_n, val_n = split_manifest(
        source,
        Path(args.train_out),
        Path(args.val_out),
        args.val_ratio,
        args.seed,
    )
    print(f"Wrote {train_n} train + {val_n} val samples")
    print(f"  train: {args.train_out}")
    print(f"  val:   {args.val_out}")


if __name__ == "__main__":
    main()
