from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from dataset import (
    HandwritingSample,
    load_crohme_dataset,
    load_iam_online_dataset,
    read_manifest,
    write_manifest,
)


HEBREW_SEED_PHRASES = [
    "שלום עולם",
    "מה שלומך היום",
    "בוקר טוב",
    "ערב טוב",
    "אני לומד כתיבה",
    "המודל צריך יותר דוגמאות",
    "למידת מכונה בעברית",
    "זיהוי כתב יד בזמן אמת",
    "נתונים מאוזנים משפרים דיוק",
    "היום מזג האוויר נעים",
    "תודה רבה",
    "אני אוהב מתמטיקה",
    "בדיקות איכות חשובות",
    "כל הכבוד לצוות",
    "אפשר להמשיך להתאמן",
    "כתיבה רציפה על מסך",
    "המערכת תומכת בעברית",
    "שיפור הדרגתי לאורך זמן",
]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_mode(samples: list[HandwritingSample], mode: str) -> list[HandwritingSample]:
    return [HandwritingSample(points=s.points, text=s.text, mode=mode) for s in samples if s.points and s.text]


def _resolve_hebrew_manifest(data_cfg: dict) -> Path | None:
    candidates = [
        data_cfg.get("hebrew_manifest"),
        data_cfg.get("hebrew_manifest_path"),
        "data/raw/hebrew_manifest.jsonl",
        "data/raw/hebrew.jsonl",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    return None


def _char_stroke(
    rng: random.Random,
    x0: float,
    y0: float,
    width: float,
    height: float,
    points_per_char: int = 6,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in range(points_per_char):
        p = i / max(1, points_per_char - 1)
        # A simple wavy right-to-left stroke segment.
        x = x0 - (width * p) + rng.uniform(-0.3, 0.3)
        y = y0 + (height * (0.35 + 0.25 * ((-1) ** i))) + rng.uniform(-0.25, 0.25)
        pts.append((x, y))
    return pts


def _simulate_online_points(text: str, rng: random.Random) -> list[list[float]]:
    points: list[list[float]] = []
    cursor_x = 0.0
    baseline_y = 0.0
    t = 0.0

    for ch in text:
        if ch.isspace():
            cursor_x -= rng.uniform(2.8, 4.2)
            t += 1.0
            continue

        char_w = rng.uniform(1.8, 3.0)
        char_h = rng.uniform(1.8, 3.4)
        stroke = _char_stroke(
            rng=rng,
            x0=cursor_x,
            y0=baseline_y + rng.uniform(-0.5, 0.5),
            width=char_w,
            height=char_h,
            points_per_char=rng.randint(5, 8),
        )
        for x, y in stroke:
            points.append([x, y, t])
            t += 1.0

        # Pen-up gap between characters.
        t += 1.0
        cursor_x -= char_w + rng.uniform(0.8, 1.6)

    if not points:
        return [[0.0, 0.0, 0.0]]
    return points


def generate_synthetic_hebrew_samples(count: int, seed: int) -> list[HandwritingSample]:
    rng = random.Random(seed)
    synthetic: list[HandwritingSample] = []
    for _ in range(max(0, count)):
        base = rng.choice(HEBREW_SEED_PHRASES)
        if rng.random() < 0.35:
            extra = rng.choice(HEBREW_SEED_PHRASES)
            text = f"{base} {extra.split()[0]}"
        else:
            text = base

        points = _simulate_online_points(text, rng)
        synthetic.append(HandwritingSample(points=points, text=text, mode="synthetic"))
    return synthetic


def _jitter_points(points: list[list[float]], rng: random.Random) -> list[list[float]]:
    if not points:
        return [[0.0, 0.0, 0.0]]
    x_shift = rng.uniform(-1.2, 1.2)
    y_shift = rng.uniform(-1.2, 1.2)
    time_scale = rng.uniform(0.92, 1.08)
    jittered: list[list[float]] = []
    for p in points:
        if len(p) < 3:
            continue
        jittered.append(
            [
                float(p[0]) + x_shift + rng.uniform(-0.15, 0.15),
                float(p[1]) + y_shift + rng.uniform(-0.15, 0.15),
                float(p[2]) * time_scale,
            ]
        )
    return jittered if jittered else [[0.0, 0.0, 0.0]]


def oversample_hebrew_samples(
    hebrew_samples: list[HandwritingSample],
    multiplier: int,
    seed: int,
) -> list[HandwritingSample]:
    if not hebrew_samples or multiplier <= 1:
        return hebrew_samples
    rng = random.Random(seed)
    out: list[HandwritingSample] = []
    for _ in range(multiplier):
        for sample in hebrew_samples:
            out.append(
                HandwritingSample(
                    points=_jitter_points(sample.points, rng),
                    text=sample.text,
                    mode="hebrew",
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess handwriting datasets into train/val manifests.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config YAML.")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    seed = int(training_cfg.get("seed", 1337))
    synthetic_count = int(data_cfg.get("synthetic_count", 20000))

    iam_path = Path(str(data_cfg.get("iam_online_path", "")))
    crohme_path = Path(str(data_cfg.get("crohme_path", "")))
    hebrew_manifest_path = _resolve_hebrew_manifest(data_cfg)

    english_samples = _normalize_mode(load_iam_online_dataset(iam_path), mode="english") if iam_path.exists() else []
    math_samples = _normalize_mode(load_crohme_dataset(crohme_path), mode="math") if crohme_path.exists() else []
    hebrew_samples = (
        _normalize_mode(read_manifest(hebrew_manifest_path), mode="hebrew")
        if hebrew_manifest_path is not None
        else []
    )
    original_hebrew_count = len(hebrew_samples)
    hebrew_samples = oversample_hebrew_samples(
        hebrew_samples=hebrew_samples,
        multiplier=20,
        seed=seed + 31,
    )
    synthetic_samples = generate_synthetic_hebrew_samples(synthetic_count, seed=seed + 17)

    print(f"Loaded English samples: {len(english_samples)} from {iam_path}")
    print(f"Loaded Math samples: {len(math_samples)} from {crohme_path}")
    if hebrew_manifest_path is None:
        print("Loaded Hebrew samples: 0 (no Hebrew manifest found in config/default paths)")
    else:
        print(f"Loaded Hebrew samples: {original_hebrew_count} from {hebrew_manifest_path}")
    print(
        f"Original Hebrew samples: {original_hebrew_count} | "
        f"After Oversampling (x20): {len(hebrew_samples)}"
    )
    print(f"Generated Synthetic Hebrew samples: {len(synthetic_samples)}")

    all_samples = english_samples + math_samples + hebrew_samples + synthetic_samples
    if not all_samples:
        raise RuntimeError("No samples available after preprocessing.")

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    split_idx = max(1, int(len(all_samples) * 0.9))
    split_idx = min(split_idx, len(all_samples) - 1) if len(all_samples) > 1 else 1
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_manifest = Path(str(data_cfg.get("train_manifest", "data/processed/train.jsonl")))
    val_manifest = Path(str(data_cfg.get("val_manifest", "data/processed/val.jsonl")))
    write_manifest(train_manifest, train_samples)
    write_manifest(val_manifest, val_samples)

    print(f"Saved train manifest: {train_manifest} ({len(train_samples)} samples)")
    print(f"Saved val manifest: {val_manifest} ({len(val_samples)} samples)")


if __name__ == "__main__":
    main()
