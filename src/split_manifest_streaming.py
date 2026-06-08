"""Streaming variant of split_manifest — RAM-friendly for Colab free (12 GB).

The original split_manifest.py reads the entire manifest into a Python list,
which blows up to 10+ GB RAM for a 1.5 GB file (Python dict/list overhead).
This streaming version makes TWO passes over the file:

  Pass 1: scan only `mode` from each line (cheap) and decide which line index
          goes to train vs val. With balance-modes, also decide which lines
          get pruned/kept per mode.

  Pass 2: re-open the file, copy lines verbatim to train.jsonl or val.jsonl
          based on the decisions from Pass 1.

We never hold stroke data in memory. Peak RAM is bounded by the per-line-index
metadata (~ N × small ints) ≈ 30 MB for 300K samples.

Same CLI as src/split_manifest.py, drop-in replacement.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--val-out", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-modes", action="store_true")
    parser.add_argument("--per-mode-target", type=int, default=15000)
    parser.add_argument("--max-oversample", type=float, default=8.0)
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Plain random split rather than per-mode stratified split.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ===== Pass 1: scan modes only =====
    print(f"Pass 1: scanning modes from {args.source}...", flush=True)
    line_modes: list[str] = []
    with args.source.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                line_modes.append("")
                continue
            try:
                # Cheap: locate "mode" key without full parse when possible.
                m_start = line.find('"mode"')
                if m_start >= 0:
                    colon = line.find(":", m_start)
                    q1 = line.find('"', colon)
                    q2 = line.find('"', q1 + 1)
                    if q1 >= 0 and q2 > q1:
                        line_modes.append(line[q1 + 1:q2])
                        continue
                # Fallback: full JSON parse
                obj = json.loads(line)
                line_modes.append(str(obj.get("mode", "")))
            except Exception:
                line_modes.append("")
            if (i + 1) % 50000 == 0:
                print(f"  scanned {i + 1} lines, current mode={line_modes[-1]}", flush=True)

    total = len(line_modes)
    print(f"  total lines: {total}", flush=True)

    # ===== Compute per-mode counts and source histogram =====
    from collections import Counter
    source_counts = Counter(line_modes)
    print(f"  source counts: {dict(source_counts)}", flush=True)

    # ===== Pass 1.5: decide train / val membership =====
    print("Computing split decisions...", flush=True)
    train_set: set[int] = set()
    val_set: set[int] = set()

    if args.no_stratify:
        all_idx = list(range(total))
        rng.shuffle(all_idx)
        n_val = int(round(total * args.val_ratio))
        val_set.update(all_idx[:n_val])
        train_set.update(all_idx[n_val:])
    else:
        # Stratified split: ~val_ratio of each mode goes to val.
        per_mode_indices: dict[str, list[int]] = {}
        for i, m in enumerate(line_modes):
            per_mode_indices.setdefault(m, []).append(i)
        for mode, idxs in per_mode_indices.items():
            local_rng = random.Random(args.seed + hash(mode) % 10000)
            local_rng.shuffle(idxs)
            n_val_m = int(round(len(idxs) * args.val_ratio))
            val_set.update(idxs[:n_val_m])
            train_set.update(idxs[n_val_m:])

    print(f"  pre-balance train={len(train_set)}, val={len(val_set)}", flush=True)

    # ===== Apply balance-modes to train set =====
    if args.balance_modes:
        train_by_mode: dict[str, list[int]] = {}
        for i in train_set:
            train_by_mode.setdefault(line_modes[i], []).append(i)
        new_train: set[int] = set()
        for mode, idxs in train_by_mode.items():
            count = len(idxs)
            cap = min(args.per_mode_target, int(count * args.max_oversample))
            if count >= cap:
                # Downsample
                rng.shuffle(idxs)
                new_train.update(idxs[:cap])
            else:
                # Keep all + oversample by duplicating indices.
                rng.shuffle(idxs)
                while len(new_train) < cap and idxs:
                    # We only add unique indices; for true oversampling we'd
                    # need a separate write step. Streaming split keeps just
                    # the unique survivors; if you need exact oversampling,
                    # use the non-streaming variant.
                    new_train.update(idxs[:cap])
                    break
        train_set = new_train

    # ===== Compute per-mode summary =====
    train_counts = Counter(line_modes[i] for i in train_set)
    val_counts = Counter(line_modes[i] for i in val_set)
    print(f"  source ({total}): "
          + ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())),
          flush=True)
    print(f"  train ({len(train_set)}): "
          + ", ".join(f"{k}={v}" for k, v in sorted(train_counts.items())),
          flush=True)
    print(f"  val ({len(val_set)}): "
          + ", ".join(f"{k}={v}" for k, v in sorted(val_counts.items())),
          flush=True)

    # ===== Pass 2: stream-write =====
    print(f"Pass 2: streaming output to {args.train_out} / {args.val_out}", flush=True)
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_val = 0
    with args.source.open("r", encoding="utf-8") as fin, \
            args.train_out.open("w", encoding="utf-8") as ftrain, \
            args.val_out.open("w", encoding="utf-8") as fval:
        for i, line in enumerate(fin):
            if i in train_set:
                ftrain.write(line if line.endswith("\n") else line + "\n")
                n_train += 1
            elif i in val_set:
                fval.write(line if line.endswith("\n") else line + "\n")
                n_val += 1
            if (i + 1) % 50000 == 0:
                print(f"  written {i + 1}/{total} lines (train={n_train}, val={n_val})", flush=True)

    print(f"Wrote {n_train} train + {n_val} val samples", flush=True)
    print(f"  train: {args.train_out}", flush=True)
    print(f"  val:   {args.val_out}", flush=True)


if __name__ == "__main__":
    main()
