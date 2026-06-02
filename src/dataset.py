from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random
import xml.etree.ElementTree as ET

import numpy as np

@dataclass
class HandwritingSample:
    points: list[list[float]]
    text: str
    mode: str = "auto"


def read_manifest(path: str | Path) -> list[HandwritingSample]:
    samples: list[HandwritingSample] = []
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[read_manifest] Skipping malformed JSON at {manifest}:{line_no} ({exc})")
                continue

            points = row.get("points", [])
            text = row.get("text", row.get("truth", ""))
            mode = row.get("mode", row.get("category", "auto"))
            if not isinstance(points, list) or not isinstance(text, str):
                print(f"[read_manifest] Skipping invalid schema at {manifest}:{line_no}")
                continue

            samples.append(
                HandwritingSample(
                    points=points,
                    text=text,
                    mode=mode if isinstance(mode, str) else "auto",
                )
            )
    return samples


def write_manifest(path: str | Path, samples: Iterable[HandwritingSample]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps({"points": s.points, "text": s.text, "mode": s.mode}, ensure_ascii=False)
                + "\n"
            )


def subsample_points(points: list[list[float]], min_dist: float = 5.0) -> list[list[float]]:
    """
    Drop pen points closer than min_dist px — matches Android buildInputFeature().
    Preserves the third coordinate (timestamp / pen flag) when present.
    """
    if len(points) < 2:
        return list(points)
    out: list[list[float]] = [list(points[0])]
    lx, ly = float(points[0][0]), float(points[0][1])
    for pt in points[1:]:
        x, y = float(pt[0]), float(pt[1])
        if np.hypot(x - lx, y - ly) > min_dist:
            out.append(list(pt))
            lx, ly = x, y
    if len(out) < 2 and len(points) >= 2:
        out.append(list(points[-1]))
    return out


def _canonical_scale(arr: np.ndarray) -> np.ndarray:
    """
    Translate to origin and scale so the writing HEIGHT is ~100 units.

    Datasets arrive in wildly different coordinate scales (IAM online ~thousands
    of px, CROHME ~tens of units, Hebrew ~hundreds). Without this, a fixed-pixel
    subsample threshold keeps every IAM point but collapses a CROHME formula to a
    handful of points. Normalizing to a canonical height makes all modes share one
    scale so resampling keeps comparable stroke detail everywhere.

    NOTE: Android `buildInputFeature()` must apply the SAME normalization before
    inference, or the deployed model will see out-of-distribution inputs.
    """
    xy = arr[:, :2].astype(np.float32)
    xmin = float(xy[:, 0].min())
    ymin = float(xy[:, 1].min())
    h = float(xy[:, 1].max() - ymin)
    w = float(xy[:, 0].max() - xmin)
    # For a single text line width >> height; fall back to width/8 if flat.
    span = max(h, w / 8.0, 1e-3)
    scale = 100.0 / span
    out = arr.astype(np.float32).copy()
    out[:, 0] = (xy[:, 0] - xmin) * scale
    out[:, 1] = (xy[:, 1] - ymin) * scale
    return out


def _resample_arclength(arr: np.ndarray, target_points: int = 200, min_dist: float = 1.5) -> np.ndarray:
    """
    Drop points so the kept set is ~target_points, spaced by arc length.

    Uses the sample's own path length to pick the spacing, so dense samples are
    thinned and sparse ones are preserved — consistent density across modes and
    safely under max_seq_len.
    """
    if len(arr) < 2:
        return arr
    diffs = np.diff(arr[:, :2], axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    path_len = float(seg.sum())
    step = max(min_dist, path_len / float(max(1, target_points)))
    kept = [arr[0]]
    acc = 0.0
    for i in range(1, len(arr)):
        acc += float(seg[i - 1])
        if acc >= step:
            kept.append(arr[i])
            acc = 0.0
    if len(kept) < 2:
        kept.append(arr[-1])
    return np.asarray(kept, dtype=np.float32)


def points_to_relative_features(points: list[list[float]]) -> np.ndarray:
    """
    Convert absolute points into relative (dx, dy, pen_state) features.

    Pipeline: canonical scale (height -> ~100) -> arc-length resample to a stable
    point count -> relative deltas with robust per-sample normalization. This
    makes English/Hebrew/Math share one coordinate scale and density.

    pen_state:
      - 1.0 at sequence start or a large discontinuity (stroke boundary)
      - 0.0 for normal pen-down continuation
    """
    if not points:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    arr = _canonical_scale(arr)
    arr = _resample_arclength(arr, target_points=200, min_dist=1.5)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    x = arr[:, 0]
    y = arr[:, 1]
    dx = np.diff(x, prepend=x[:1])
    dy = np.diff(y, prepend=y[:1])
    # Identify likely stroke starts by larger-than-usual jump.
    jump = np.sqrt(dx * dx + dy * dy)
    threshold = max(0.2, float(np.percentile(jump, 90)) * 1.5)
    pen_state = np.zeros_like(dx, dtype=np.float32)
    pen_state[0] = 1.0
    pen_state[jump > threshold] = 1.0

    feats = np.stack([dx, dy, pen_state], axis=1).astype(np.float32)
    # Robust normalization for deltas only; keep pen_state binary.
    scale = np.percentile(np.abs(feats[:, :2]), 95)
    if scale > 1e-6:
        feats[:, :2] = np.clip(feats[:, :2] / scale, -5.0, 5.0)
    return feats


def maybe_augment_relative_features(features: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled or len(features) < 2:
        return features
    out = features.copy()
    # Add small noise to delta channels only.
    out[:, 0] += np.random.normal(0.0, 0.015, size=len(out))
    out[:, 1] += np.random.normal(0.0, 0.02, size=len(out))
    # Random local time warp by duplicating/removing a few points.
    if random.random() < 0.2 and len(out) > 6:
        i = random.randint(1, len(out) - 2)
        out = np.delete(out, i, axis=0)
    if random.random() < 0.2 and len(out) > 4:
        i = random.randint(1, len(out) - 2)
        out = np.insert(out, i, out[i], axis=0)
    out[:, 2] = (out[:, 2] > 0.5).astype(np.float32)
    return out


def load_crohme_dataset(path: str | Path) -> list[HandwritingSample]:
    """
    Placeholder loader skeleton for CROHME (online math handwriting).

    Expected output:
      list[HandwritingSample(points=[[x,y,t],...], text="<latex>", mode="math")]
    """
    root = Path(path)
    if not root.exists():
        return []

    samples: list[HandwritingSample] = []
    inkml_files = sorted(root.rglob("*.inkml"))
    for inkml in inkml_files:
        parsed = _parse_inkml_file(inkml)
        if parsed is not None:
            samples.append(parsed)
    return samples


def load_iam_online_dataset(path: str | Path) -> list[HandwritingSample]:
    """
    Loader for IAM-OnDB (online English handwriting), in priority order:
      1) JSONL manifests already prepared (fast path)
      2) IAM-OnDB jsonised (DeepWriting / ETH AIT) *-segmented[-nonorm].json files
         with real pen trajectories (word_stroke) and transcriptions (word_ascii)
      3) generic XML stroke files + sidecar text files (true Point/Stroke trajectories)

    The legacy IAM offline form-metadata path (cmp bounding boxes -> fake zigzag
    pseudo-strokes) is DISABLED by default because it caused English template
    collapse. Set env IAM_ALLOW_FAKE_BBOX=1 to re-enable it as a last resort.

    Output: list[HandwritingSample(points=[[x,y,t],...], text="<transcript>", mode="english")]
    """
    root = Path(path)
    if not root.exists():
        return []

    samples: list[HandwritingSample] = []

    # Preferred fast path: JSONL manifests already prepared.
    for jsonl in sorted(root.rglob("*.jsonl")):
        try:
            samples.extend(read_manifest(jsonl))
        except Exception:
            continue
    if samples:
        return samples

    # Path 2: IAM-OnDB jsonised (real online trajectories).
    json_files = sorted(root.rglob("*-segmented-nonorm.json")) + sorted(
        root.rglob("*-segmented.json")
    )
    # Dedup per record: prefer the "jsready" variant when both exist for a record.
    by_record: dict[str, Path] = {}
    for jf in json_files:
        record = _iamondb_record_key(jf)
        existing = by_record.get(record)
        if existing is None:
            by_record[record] = jf
        elif "jsready" in jf.name and "jsready" not in existing.name:
            by_record[record] = jf
    for jf in sorted(by_record.values()):
        samples.extend(_parse_iamondb_json(jf))
    if samples:
        return samples

    xml_files = sorted(root.rglob("*.xml"))

    # Path 3: generic XML stroke files and sidecar text files by stem.
    for xml_file in xml_files:
        points = _parse_generic_stroke_xml(xml_file)
        if not points:
            continue
        text = _find_sidecar_text(xml_file)
        if not text:
            continue
        samples.append(HandwritingSample(points=points, text=text, mode="english"))
    if samples:
        return samples

    # Disabled fallback: IAM offline form metadata (cmp boxes -> fake strokes).
    # Opt-in only; this is the known cause of English template collapse.
    if os.environ.get("IAM_ALLOW_FAKE_BBOX") == "1":
        for xml_file in xml_files:
            form_lines = _parse_iam_form_xml(xml_file)
            if form_lines:
                samples.extend(form_lines)

    return samples


def _iamondb_record_key(json_path: Path) -> str:
    """
    Map a jsonised IAM-OnDB filename to a stable record key so that the
    'jsready' and plain variants of the same record dedup to one entry.

    e.g. 'a01-020z-jsready-segmented-nonorm.json' -> 'a01-020z'
         'c01-038z-segmented-nonorm.json'         -> 'c01-038z'
    """
    stem = json_path.name
    for suffix in ("-jsready-segmented-nonorm.json", "-segmented-nonorm.json",
                   "-jsready-segmented.json", "-segmented.json"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return json_path.stem


def _parse_iamondb_json(json_path: Path) -> list[HandwritingSample]:
    """
    Parse one IAM-OnDB jsonised (DeepWriting / ETH AIT) file into line-level samples.

    Each file holds multiple line samples keyed 'sample0', 'sample1', ... . Each
    sample provides:
      - word_ascii:  the line transcription
      - word_stroke: ordered points [{ "x", "y", "ts", "ev" }, ...]
                     ev in {0: pen-down, 1: writing, 2: pen-up}

    Points are emitted as [x, y, t] where t is a monotonic counter (consistent
    with the other parsers in this module), with an extra step inserted at each
    pen-up / pen-down boundary so multi-stroke lines keep their stroke gaps.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    out: list[HandwritingSample] = []
    for key in sorted(data.keys()):
        if not key.startswith("sample"):
            continue
        sample = data.get(key)
        if not isinstance(sample, dict):
            continue
        text = str(sample.get("word_ascii") or "").strip()
        if not text:
            continue
        word_stroke = sample.get("word_stroke")
        if not isinstance(word_stroke, list) or not word_stroke:
            continue

        points: list[list[float]] = []
        t = 0.0
        prev_ev: int | None = None
        for pt in word_stroke:
            if not isinstance(pt, dict):
                continue
            try:
                x = float(pt["x"])
                y = float(pt["y"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                ev = int(pt.get("ev", 1))
            except (TypeError, ValueError):
                ev = 1
            # Insert a pen-up gap on a new stroke (pen-down after a previous point)
            # or right after a pen-up event.
            if points and (ev == 0 or prev_ev == 2):
                t += 1.0
            points.append([x, y, t])
            t += 1.0
            prev_ev = ev

        if not points:
            continue
        out.append(HandwritingSample(points=points, text=text, mode="english"))

    return out


def _parse_inkml_file(path: Path) -> HandwritingSample | None:
    try:
        tree = ET.parse(path)
    except Exception:
        return None
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}", 1)[0] + "}"

    traces = root.findall(f".//{ns}trace")
    points: list[list[float]] = []
    t = 0.0
    for trace in traces:
        text = (trace.text or "").strip()
        if not text:
            continue
        # InkML trace format: "x y, x y, ..."
        chunks = [c.strip() for c in text.replace("\n", " ").split(",") if c.strip()]
        trace_points: list[tuple[float, float]] = []
        for chunk in chunks:
            vals = [v for v in chunk.split() if v]
            if len(vals) < 2:
                continue
            try:
                x = float(vals[0])
                y = float(vals[1])
            except ValueError:
                continue
            trace_points.append((x, y))
        if not trace_points:
            continue
        # mark new stroke with repeated start time step but separate segment
        for idx, (x, y) in enumerate(trace_points):
            points.append([x, y, t + idx])
        t += len(trace_points) + 1.0

    if not points:
        return None

    # CROHME ground truth commonly in annotation type="truth"
    truth = ""
    for ann in root.findall(f".//{ns}annotation"):
        ann_type = (ann.attrib.get("type") or "").lower()
        if ann_type in ("truth", "latex", "normalizedlabel", "label"):
            truth = (ann.text or "").strip()
            if truth:
                break
    if not truth:
        # fallback: first non-empty annotation text
        for ann in root.findall(f".//{ns}annotation"):
            text = (ann.text or "").strip()
            if text:
                truth = text
                break
    if not truth:
        # Fallback: reconstruct from symbol-level traceGroup annotations.
        token_groups = []
        for grp in root.findall(f".//{ns}traceGroup"):
            ann = grp.find(f"{ns}annotation")
            if ann is None:
                continue
            ann_type = (ann.attrib.get("type") or "").lower()
            text = (ann.text or "").strip()
            if ann_type == "truth" and text and text != "Closest Strk":
                token_groups.append(text)
        if token_groups:
            truth = "".join(token_groups)

    truth = _normalize_crohme_truth(truth)
    if not truth:
        return None

    return HandwritingSample(points=points, text=truth, mode="math")


def _parse_generic_stroke_xml(path: Path) -> list[list[float]]:
    try:
        tree = ET.parse(path)
    except Exception:
        return []
    root = tree.getroot()
    points: list[list[float]] = []
    t = 0.0

    # Accept common tags: Point, point, Stroke, trace-like nodes with x/y attrs.
    for stroke_like in root.findall(".//Stroke") + root.findall(".//stroke") + root.findall(".//trace"):
        local_points: list[tuple[float, float]] = []
        children = list(stroke_like)
        if children:
            for p in children:
                x_attr = p.attrib.get("x") or p.attrib.get("X")
                y_attr = p.attrib.get("y") or p.attrib.get("Y")
                if x_attr is None or y_attr is None:
                    continue
                try:
                    local_points.append((float(x_attr), float(y_attr)))
                except ValueError:
                    continue
        else:
            # inline text fallback "x y, x y"
            txt = (stroke_like.text or "").strip()
            if txt:
                chunks = [c.strip() for c in txt.split(",") if c.strip()]
                for c in chunks:
                    vals = c.split()
                    if len(vals) < 2:
                        continue
                    try:
                        local_points.append((float(vals[0]), float(vals[1])))
                    except ValueError:
                        continue
        for idx, (x, y) in enumerate(local_points):
            points.append([x, y, t + idx])
        if local_points:
            t += len(local_points) + 1.0

    # absolute fallback: all Point nodes
    if not points:
        for p in root.findall(".//Point") + root.findall(".//point"):
            x_attr = p.attrib.get("x") or p.attrib.get("X")
            y_attr = p.attrib.get("y") or p.attrib.get("Y")
            if x_attr is None or y_attr is None:
                continue
            try:
                points.append([float(x_attr), float(y_attr), t])
                t += 1.0
            except ValueError:
                continue
    return points


def _find_sidecar_text(xml_file: Path) -> str:
    # Try sibling txt/label files with same stem.
    candidates = [
        xml_file.with_suffix(".txt"),
        xml_file.with_suffix(".lab"),
        xml_file.with_suffix(".label"),
    ]
    for cand in candidates:
        if cand.exists():
            text = cand.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
    return ""


def _normalize_crohme_truth(text: str) -> str:
    if not text:
        return ""
    # Normalize common CROHME label quirks while preserving LaTeX-like tokens.
    out = text.strip()
    out = out.replace("−", "-").replace("—", "-").replace("∗", "*")
    out = out.replace("\u00a0", " ").replace("\t", " ")
    out = " ".join(out.split())
    return out


def _parse_iam_form_xml(path: Path) -> list[HandwritingSample]:
    """
    Parse IAM OFFLINE form XML (line-level transcripts + cmp bounding boxes) into
    pseudo online traces (4-point zigzag per box).

    WARNING: these are FAKE strokes, not real pen trajectories. They cause English
    template collapse, so load_iam_online_dataset only calls this when the env flag
    IAM_ALLOW_FAKE_BBOX=1 is set. Prefer real online data (IAM-OnDB jsonised).
    """
    try:
        tree = ET.parse(path)
    except Exception:
        return []
    root = tree.getroot()
    if root.tag.lower() != "form":
        return []

    handwritten = root.find(".//handwritten-part")
    if handwritten is None:
        return []

    out: list[HandwritingSample] = []
    for line in handwritten.findall("line"):
        text = (line.attrib.get("text") or "").strip()
        if not text:
            continue
        points: list[list[float]] = []
        t = 0.0
        words = line.findall("word")
        if not words:
            continue
        for word in words:
            cmps = word.findall("cmp")
            for cmp_node in cmps:
                try:
                    x = float(cmp_node.attrib.get("x", "0"))
                    y = float(cmp_node.attrib.get("y", "0"))
                    w = float(cmp_node.attrib.get("width", "1"))
                    h = float(cmp_node.attrib.get("height", "1"))
                except ValueError:
                    continue
                # Build a tiny pseudo-stroke within each component box.
                local = [
                    (x, y + h * 0.5),
                    (x + w * 0.35, y + h * 0.2),
                    (x + w * 0.7, y + h * 0.8),
                    (x + w, y + h * 0.5),
                ]
                for px, py in local:
                    points.append([px, py, t])
                    t += 1.0
                # pen-up delimiter
                t += 1.0
        if points:
            out.append(HandwritingSample(points=points, text=text, mode="text"))
    return out


# הוסיפי את זה לסוף הקובץ dataset.py

def read_firebase_corrections(path: str | Path) -> list[HandwritingSample]:
    """
    Reads correction JSON files uploaded by the app to Firebase Storage.

    Expected object shape (see TrainingDataSyncWorker.buildPayload):
      {"truth": "...", "prediction": "...", "points": [[x,y,t], ...], "mode": "hebrew"}

    Supports nested folders, e.g. training_data/new/{userId}/*.json
    (Firebase layout) or a flat folder of *.json after manual download.
    """
    samples: list[HandwritingSample] = []
    dir_path = Path(path)

    if not dir_path.exists():
        print(f"[read_firebase_corrections] Directory {path} does not exist.")
        return []

    if dir_path.is_file() and dir_path.suffix.lower() == ".json":
        json_files = [dir_path]
    else:
        json_files = sorted(dir_path.rglob("*.json"))

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # קלוד שומר את הנקודות בשדה strokesJson כמחרוזת או points כמערך
            raw_points = data.get("points")
            if not raw_points and "strokesJson" in data:
                raw_points = json.loads(data["strokesJson"])
            
            text = (data.get("truth") or data.get("correctedText") or "").strip()
            mode = data.get("mode", "correction")
            if not isinstance(mode, str):
                mode = "correction"

            if raw_points and text:
                samples.append(
                    HandwritingSample(
                        points=raw_points,
                        text=text,
                        mode=mode,
                    )
                )
        except Exception as e:
            print(f"[read_firebase_corrections] Error reading {json_file}: {e}")
            
    print(f"Successfully loaded {len(samples)} correction samples.")
    return samples