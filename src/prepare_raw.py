from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import (
    HandwritingSample,
    load_crohme_dataset,
    load_iam_online_dataset,
    read_firebase_corrections,
    read_manifest,
    write_manifest,
)


def _discover_jsonl(raw_root: Path) -> list[HandwritingSample]:
    samples: list[HandwritingSample] = []
    for jsonl in sorted(raw_root.rglob("*.jsonl")):
        try:
            loaded = read_manifest(jsonl)
            print(f"  jsonl: {jsonl.relative_to(raw_root)} ({len(loaded)} samples)")
            samples.extend(loaded)
        except Exception as exc:
            print(f"  skip jsonl {jsonl}: {exc}")
    return samples


def _is_correction_json(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    has_points = "points" in data or "strokesJson" in data
    has_text = "truth" in data or "correctedText" in data
    return has_points and has_text


def _discover_json(raw_root: Path) -> list[HandwritingSample]:
    samples: list[HandwritingSample] = []
    # Firebase mirror paths (app uploads to training_data/new/{userId}/*.json)
    for rel in (
        "corrections",
        "training_data/new",
        "training_data/processed",
        "firebase",
    ):
        folder = raw_root / rel
        if folder.exists():
            loaded = read_firebase_corrections(folder)
            if loaded:
                print(f"  firebase corrections: {rel} ({len(loaded)} samples)")
                samples.extend(loaded)
    # Any other correction-shaped JSON trees under raw/
    if not samples:
        loaded = read_firebase_corrections(raw_root)
        if loaded:
            print(f"  firebase corrections: raw root ({len(loaded)} samples)")
            samples.extend(loaded)
    return samples


def _discover_crohme(raw_root: Path) -> list[HandwritingSample]:
    samples: list[HandwritingSample] = []
    seen: set[Path] = set()

    for name in ("crohme", "CROHME", "math"):
        folder = raw_root / name
        if folder.exists():
            seen.add(folder.resolve())

    for inkml in raw_root.rglob("*.inkml"):
        seen.add(inkml.parent.resolve())

    for folder in sorted(seen):
        loaded = load_crohme_dataset(folder)
        if loaded:
            try:
                rel = folder.relative_to(raw_root)
            except ValueError:
                rel = folder
            print(f"  inkml: {rel} ({len(loaded)} samples)")
            samples.extend(loaded)
    return samples


def _discover_iam(raw_root: Path) -> list[HandwritingSample]:
    samples: list[HandwritingSample] = []
    for name in ("iam", "IAM", "iam-online", "english", "text"):
        folder = raw_root / name
        if not folder.exists():
            continue
        loaded = load_iam_online_dataset(folder)
        if loaded:
            print(f"  iam: {folder.relative_to(raw_root)} ({len(loaded)} samples)")
            samples.extend(loaded)
    return samples


def prepare_raw(
    raw_dir: str | Path = "data/raw",
    output: str | Path = "data/processed/all.jsonl",
    *,
    append_processed: bool = True,
) -> int:
    """
    Scan data/raw and merge supported sources into one JSONL manifest.

    Supported:
      - **/*.jsonl
      - **/*.json (Firebase / app corrections)
      - crohme/ or **/*.inkml (CROHME math)
      - iam/ (IAM-OnDB)
    """
    raw_root = Path(raw_dir)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        print(f"[prepare_raw] Raw dir missing: {raw_root}")
        return 0

    print(f"[prepare_raw] Scanning {raw_root} ...")
    all_samples: list[HandwritingSample] = []
    all_samples.extend(_discover_jsonl(raw_root))
    all_samples.extend(_discover_json(raw_root))
    all_samples.extend(_discover_crohme(raw_root))
    all_samples.extend(_discover_iam(raw_root))

    if append_processed and out_path.exists():
        existing = read_manifest(out_path)
        print(f"  merge existing: {out_path} ({len(existing)} samples)")
        all_samples.extend(existing)

    seen_keys: set[str] = set()
    unique: list[HandwritingSample] = []
    for s in all_samples:
        first = tuple(s.points[0]) if s.points else ()
        key = f"{s.mode}|{s.text}|{len(s.points)}|{first}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(s)

    write_manifest(out_path, unique)
    print(f"[prepare_raw] Wrote {len(unique)} samples -> {out_path}")
    return len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all.jsonl from data/raw/")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="data/processed/all.jsonl")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    count = prepare_raw(args.raw, args.output, append_processed=not args.no_merge_existing)
    if count == 0:
        raise SystemExit("No samples found in data/raw.")


if __name__ == "__main__":
    main()
