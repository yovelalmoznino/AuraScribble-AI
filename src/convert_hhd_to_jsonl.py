from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import HandwritingSample, write_manifest

# Map folder names (Latin transliteration) to Hebrew letters when needed.
_FOLDER_TO_HEBREW = {
    "alef": "א", "bet": "ב", "gimel": "ג", "dalet": "ד", "he": "ה", "vav": "ו",
    "zayin": "ז", "het": "ח", "tet": "ט", "yod": "י", "kaf": "כ", "lamed": "ל",
    "mem": "מ", "nun": "נ", "samekh": "ס", "ayin": "ע", "pe": "פ", "tsadi": "צ",
    "qof": "ק", "resh": "ר", "shin": "ש", "tav": "ת",
}


def _label_from_path(path: Path) -> str:
    name = path.parent.name.strip()
    if len(name) == 1 and "\u0590" <= name <= "\u05FF":
        return name
    if name.lower() in _FOLDER_TO_HEBREW:
        return _FOLDER_TO_HEBREW[name.lower()]
    if name.upper() in _FOLDER_TO_HEBREW:
        return _FOLDER_TO_HEBREW[name.upper()]
    return name[:1] if name else "?"


def _skeleton_to_points(img: Image.Image, max_points: int = 80) -> list[list[float]]:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    arr = (arr < 128).astype(np.uint8)
    ys, xs = np.where(arr > 0)
    if len(xs) < 4:
        return []
    order = np.lexsort((xs, ys))
    xs, ys = xs[order], ys[order]
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points, dtype=int)
        xs, ys = xs[idx], ys[idx]
    points: list[list[float]] = []
    t = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        points.append([float(x), float(y), t])
        t += 2.0 if i == 0 else 1.0
    return points


def _label_from_hf(label: str) -> str:
    """HF labels look like '1א' (class id + letter)."""
    for ch in str(label):
        if "\u0590" <= ch <= "\u05FF":
            return ch
    text = str(label).strip()
    if text in _FOLDER_TO_HEBREW:
        return _FOLDER_TO_HEBREW[text]
    return text[:1] if text else "?"


def convert_hhd_parquet(root: Path) -> list[HandwritingSample]:
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        return []

    samples: list[HandwritingSample] = []
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return []

    for pf in parquet_files:
        try:
            ds = load_dataset("parquet", data_files=str(pf), split="train")
        except Exception as exc:
            print(f"Skip parquet {pf.name}: {exc}")
            continue
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                try:
                    img = Image.open(img)
                except Exception:
                    continue
            label = _label_from_hf(row.get("label", ""))
            if not label or label == "?":
                continue
            pts = _skeleton_to_points(img)
            if len(pts) < 4:
                continue
            samples.append(HandwritingSample(points=pts, text=label, mode="hebrew"))
    return samples


def convert_hhd_folder(root: Path) -> list[HandwritingSample]:
    samples: list[HandwritingSample] = []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in exts:
            continue
        label = _label_from_path(img_path)
        if not label or label == "?":
            continue
        try:
            img = Image.open(img_path)
        except Exception:
            continue
        pts = _skeleton_to_points(img)
        if len(pts) < 4:
            continue
        samples.append(HandwritingSample(points=pts, text=label, mode="hebrew"))
    return samples


def extract_rar_if_needed(root: Path) -> Path:
    rar_files = list(root.glob("*.rar")) + list(root.glob("**/*.rar"))
    if not rar_files:
        return root
    try:
        import rarfile  # type: ignore
    except ImportError:
        print("Install rarfile or extract HHD manually into data/raw/hhd/")
        return root
    extract_dir = root / "extracted"
    extract_dir.mkdir(exist_ok=True)
    for rar in rar_files:
        with rarfile.RarFile(rar) as rf:
            rf.extractall(extract_dir)
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HHD character images to pseudo-stroke JSONL")
    parser.add_argument("--input", type=Path, default=Path("data/raw/hhd"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/hhd/hhd_strokes.jsonl"))
    args = parser.parse_args()

    root = extract_rar_if_needed(args.input)
    samples = convert_hhd_parquet(root)
    if not samples:
        samples = convert_hhd_folder(root)
    if not samples and (root / "TRAIN").exists():
        samples = convert_hhd_folder(root / "TRAIN")
    if not samples:
        samples = convert_hhd_parquet(root / "data") if (root / "data").exists() else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, samples)
    print(f"Wrote {len(samples)} HHD samples -> {args.output}")


if __name__ == "__main__":
    main()
