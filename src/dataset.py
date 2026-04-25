from __future__ import annotations

import json
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


def points_to_relative_features(points: list[list[float]]) -> np.ndarray:
    """
    Convert absolute points into relative (dx, dy, pen_state) features.

    pen_state:
      - 1.0 at sequence start or a large discontinuity (stroke boundary)
      - 0.0 for normal pen-down continuation
    """
    if not points:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
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
    Placeholder loader skeleton for IAM-OnDB (online English handwriting).

    Expected output:
      list[HandwritingSample(points=[[x,y,t],...], text="<transcript>", mode="text")]
    """
    root = Path(path)
    if not root.exists():
        return []

    # IAM-OnDB packaging varies; this loader supports:
    # 1) direct JSONL manifests in the folder
    # 2) paired XML stroke files + sidecar text files
    samples: list[HandwritingSample] = []

    # Preferred fast path: JSONL manifests already prepared.
    for jsonl in sorted(root.rglob("*.jsonl")):
        try:
            samples.extend(read_manifest(jsonl))
        except Exception:
            continue
    if samples:
        return samples

    xml_files = sorted(root.rglob("*.xml"))
    # Fallback path A: IAM form metadata XML with line text + cmp boxes.
    for xml_file in xml_files:
        form_lines = _parse_iam_form_xml(xml_file)
        if form_lines:
            samples.extend(form_lines)
    if samples:
        return samples

    # Fallback path B: generic XML stroke files and sidecar text files by stem.
    for xml_file in xml_files:
        points = _parse_generic_stroke_xml(xml_file)
        if not points:
            continue
        text = _find_sidecar_text(xml_file)
        if not text:
            continue
        samples.append(HandwritingSample(points=points, text=text, mode="text"))
    return samples


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
    Parse IAM form XML (line-level transcripts + cmp bounding boxes) into pseudo online traces.

    This is a pragmatic bridge when true pen trajectories are unavailable in the raw folder.
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
    מתאם לקריאת הנתונים המגיעים מהאפליקציה (AuraScribble).
    קורא קבצי JSON בפורמט שבו כל קובץ הוא דגימה אחת.
    """
    samples: list[HandwritingSample] = []
    dir_path = Path(path)
    
    if not dir_path.exists():
        print(f"[read_firebase_corrections] Directory {path} does not exist.")
        return []

    for json_file in dir_path.glob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # קלוד שומר את הנקודות בשדה strokesJson כמחרוזת או points כמערך
            raw_points = data.get("points")
            if not raw_points and "strokesJson" in data:
                raw_points = json.loads(data["strokesJson"])
            
            text = data.get("correctedText", "")
            
            if raw_points and text:
                samples.append(
                    HandwritingSample(
                        points=raw_points,
                        text=text,
                        mode="correction"
                    )
                )
        except Exception as e:
            print(f"[read_firebase_corrections] Error reading {json_file}: {e}")
            
    print(f"Successfully loaded {len(samples)} correction samples.")
    return samples