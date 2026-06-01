from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from dataset import HandwritingSample, read_manifest, write_manifest


def _mode_key(sample: HandwritingSample) -> str:
    return (sample.mode or "auto").lower()


def _jitter_points(points: list[list[float]], rng: random.Random) -> list[list[float]]:
    """Small spatial/time jitter so oversampled duplicates are not identical."""
    if not points:
        return [[0.0, 0.0, 0.0]]
    x_shift = rng.uniform(-1.2, 1.2)
    y_shift = rng.uniform(-1.2, 1.2)
    time_scale = rng.uniform(0.94, 1.06)
    out: list[list[float]] = []
    for p in points:
        if len(p) < 3:
            continue
        out.append(
            [
                float(p[0]) + x_shift + rng.uniform(-0.2, 0.2),
                float(p[1]) + y_shift + rng.uniform(-0.2, 0.2),
                float(p[2]) * time_scale,
            ]
        )
    return out if out else [[0.0, 0.0, 0.0]]


def stratified_split(
    samples: list[HandwritingSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[HandwritingSample], list[HandwritingSample]]:
    """
    Split per-mode so every mode is represented in val proportionally.
    A plain random split can leave a rare mode (e.g. hebrew) nearly absent
    from val, making val_<mode>_cer meaningless.
    """
    rng = random.Random(seed)
    by_mode: dict[str, list[HandwritingSample]] = defaultdict(list)
    for s in samples:
        by_mode[_mode_key(s)].append(s)

    train_samples: list[HandwritingSample] = []
    val_samples: list[HandwritingSample] = []
    for mode in sorted(by_mode.keys()):
        group = by_mode[mode][:]
        rng.shuffle(group)
        if len(group) <= 1:
            # Too few to spare for val; keep in train.
            train_samples.extend(group)
            continue
        val_count = max(1, int(len(group) * val_ratio))
        val_count = min(val_count, len(group) - 1)
        val_samples.extend(group[:val_count])
        train_samples.extend(group[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def balance_modes(
    samples: list[HandwritingSample],
    per_mode_target: int,
    max_oversample: float,
    seed: int,
) -> list[HandwritingSample]:
    """
    Balance the (train) set across modes:
      - majority modes are randomly downsampled to per_mode_target
      - minority modes are oversampled (with jitter) up to
        min(per_mode_target, count * max_oversample)

    The max_oversample cap prevents a tiny mode (e.g. 293 hebrew) from being
    duplicated dozens of times into pure overfit. The real fix for a tiny mode
    is more raw data; this just stops one mode from drowning the others.
    """
    rng = random.Random(seed)
    by_mode: dict[str, list[HandwritingSample]] = defaultdict(list)
    for s in samples:
        by_mode[_mode_key(s)].append(s)

    balanced: list[HandwritingSample] = []
    for mode in sorted(by_mode.keys()):
        group = by_mode[mode][:]
        rng.shuffle(group)
        count = len(group)
        if count == 0:
            continue
        cap = min(per_mode_target, int(count * max_oversample))
        if count >= cap:
            balanced.extend(group[:cap])
            continue
        # Oversample up to cap with jitter on the duplicates.
        out = list(group)
        i = 0
        while len(out) < cap:
            base = group[i % count]
            out.append(
                HandwritingSample(
                    points=_jitter_points(base.points, rng),
                    text=base.text,
                    mode=base.mode,
                )
            )
            i += 1
        balanced.extend(out)

    rng.shuffle(balanced)
    return balanced


def _print_mode_counts(label: str, samples: list[HandwritingSample]) -> None:
    counts = Counter(_mode_key(s) for s in samples)
    pretty = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {label} ({len(samples)}): {pretty}")


def split_manifest(
    source: Path,
    train_out: Path,
    val_out: Path,
    val_ratio: float,
    seed: int,
    *,
    stratify: bool = True,
    balance: bool = False,
    per_mode_target: int = 6000,
    max_oversample: float = 8.0,
) -> tuple[int, int]:
    samples = read_manifest(source)
    if not samples:
        raise ValueError(f"No samples found in {source}")

    _print_mode_counts("source", samples)

    if stratify:
        train_samples, val_samples = stratified_split(samples, val_ratio, seed)
    else:
        rng = random.Random(seed)
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        val_count = max(1, int(len(samples) * val_ratio))
        val_set = set(indices[:val_count])
        train_samples = [s for i, s in enumerate(samples) if i not in val_set]
        val_samples = [s for i, s in enumerate(samples) if i in val_set]

    if balance:
        train_samples = balance_modes(
            train_samples,
            per_mode_target=per_mode_target,
            max_oversample=max_oversample,
            seed=seed + 7,
        )

    _print_mode_counts("train", train_samples)
    _print_mode_counts("val", val_samples)

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
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Use a plain random split instead of per-mode stratified split.",
    )
    parser.add_argument(
        "--balance-modes",
        action="store_true",
        help="Balance the train set across modes (downsample majority, oversample minority).",
    )
    parser.add_argument(
        "--per-mode-target",
        type=int,
        default=6000,
        help="Target number of train samples per mode when --balance-modes is set.",
    )
    parser.add_argument(
        "--max-oversample",
        type=float,
        default=8.0,
        help="Max duplication factor for a minority mode when balancing.",
    )
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
        stratify=not args.no_stratify,
        balance=args.balance_modes,
        per_mode_target=args.per_mode_target,
        max_oversample=args.max_oversample,
    )
    print(f"Wrote {train_n} train + {val_n} val samples")
    print(f"  train: {args.train_out}")
    print(f"  val:   {args.val_out}")


if __name__ == "__main__":
    main()
