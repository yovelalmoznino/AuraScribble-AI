# AuraScribble — Kaggle Notebook v3.1 (Rebuild)

Copy each cell into a **new** Kaggle notebook. Order: **1 → 18**.

**Quick start (עברית):** see [`KAGGLE_V3_RUN.md`](KAGGLE_V3_RUN.md)

**v3.1 changes (after `the and` template collapse):**
- Curriculum **4 steps:** short → **medium** → full → **IAM-long only**
- Stricter **`decode_quality`** (flags `the and`, single `ה`, default `\frac{1}`)
- Phase 2b: **patience 12**, lr **2e-5**; Phase 2c: IAM-only, lr **1e-5**
- Training logs **`val template_collapse`** rate each epoch

## Kaggle settings

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x1** |
| Dataset | Upload fresh `handwriting_training_bundle.zip` from repo |
| Secret (optional) | `FIREBASE_SERVICE_ACCOUNT_JSON` |

## ZIP layout

```
handwriting-model/
  src/              # must include model_transformer.py (parallel decoder forward)
  configs/          # train_kaggle_v3.yaml, train_kaggle_rebuild.yaml, vocab.txt
  data/raw/         # iam/, crohme/, optional hhd/
```

**Do not** include old `checkpoint_best.pt` in the ZIP.

---

## Cell 1 — Markdown

```markdown
# AuraScribble handwriting v3
- Curriculum: short lines → full IAM/hebrew/math
- Early stopping on val_cer
- Upload to Firebase only if eval CER ≤ 0.35
```

---

## Cell 2 — Setup

```python
from pathlib import Path
import os, sys, shutil

INPUT = Path("/kaggle/input")
INPUT_ROOT = None
for name in ("configs/train_kaggle_v3.yaml", "configs/train_kaggle_rebuild.yaml", "configs/train.yaml"):
    for cfg in INPUT.rglob(name):
        INPUT_ROOT = cfg.parent.parent
        break
    if INPUT_ROOT:
        break
assert INPUT_ROOT, f"handwriting-model not found under {list(INPUT.iterdir())}"

WORK = Path("/kaggle/working/handwriting-model")
WORK.mkdir(parents=True, exist_ok=True)
for folder in ("src", "configs", "scripts"):
    s, d = INPUT_ROOT / folder, WORK / folder
    if s.exists():
        if d.exists(): shutil.rmtree(d)
        shutil.copytree(s, d)
        print("Copied", folder)

os.chdir(WORK)
sys.path.insert(0, str(WORK / "src"))

DATA_RAW = WORK / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
input_raw = INPUT_ROOT / "data" / "raw"
if input_raw.exists():
    for item in input_raw.iterdir():
        dest = DATA_RAW / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        print("Copied data/raw/", item.name)

OUTPUT = WORK / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
(WORK / "models").mkdir(exist_ok=True)
print("INPUT:", INPUT_ROOT)
print("v3 config:", (WORK / "configs/train_kaggle_v3.yaml").exists())
```

---

## Cells 3–8

Same as before: GPU packages, HHD, synthetic Hebrew, synthetic mixed, prepare_raw, build_vocab, split train/val.

(Copy cells 3–9 from your previous notebook if unchanged.)

---

## Cell 9 — Split train / val

```python
!python src/split_manifest.py \
    --source {OUTPUT / "all.jsonl"} \
    --train-out {OUTPUT / "train.jsonl"} \
    --val-out {OUTPUT / "val.jsonl"} \
    --val-ratio 0.1 \
    --seed 1337
```

---

## Cell 10 — Oversample (×2) + English boost

```python
import random
from dataset import read_manifest, write_manifest, HandwritingSample

def is_priority(s: HandwritingSample) -> bool:
    m = (s.mode or "").lower()
    if m in ("hebrew", "mixed", "correction"):
        return True
    if any("\u0590" <= c <= "\u05FF" for c in s.text):
        return "$" in s.text or any("a" <= c.lower() <= "z" for c in s.text if c.isascii())
    return False

def is_long_english(s: HandwritingSample) -> bool:
    m = (s.mode or "").lower()
    return m in ("english", "text") and len(s.text.strip()) > 12

train_path = OUTPUT / "train.jsonl"
train = read_manifest(train_path)
priority = [s for s in train if is_priority(s)]
english_long = [s for s in train if is_long_english(s)]
rest = [s for s in train if s not in priority and s not in english_long]
train_boosted = rest + priority * 2 + english_long * 2
random.Random(1337).shuffle(train_boosted)
write_manifest(train_path, train_boosted)
print(f"Train: {len(train_boosted)} | priority x2: {len(priority)} | english/text x2: {len(english_long)}")
```

---

## Cell 10b — Curriculum manifests (short + medium + IAM-long)

```python
!python src/build_curriculum_manifest.py \
    --train {OUTPUT / "train.jsonl"} \
    --short-out {OUTPUT / "train_short.jsonl"} \
    --medium-out {OUTPUT / "train_medium.jsonl"} \
    --iam-out {OUTPUT / "train_iam_long.jsonl"} \
    --max-chars-short 32 \
    --medium-min-chars 33 \
    --medium-max-chars 72 \
    --iam-min-chars 24
```

---

## Cell 11 — Base config (v3)

```python
import shutil, yaml
from pathlib import Path

for src_name in ("train_kaggle_v3.yaml", "train_kaggle_rebuild.yaml"):
    src = WORK / "configs" / src_name
    if src.exists():
        cfg_src = src
        break
else:
    raise FileNotFoundError("Missing train_kaggle_v3.yaml in ZIP")

cfg_dst = WORK / "configs" / "train_kaggle.yaml"
shutil.copy2(cfg_src, cfg_dst)
cfg = yaml.safe_load(cfg_dst.read_text(encoding="utf-8"))

cfg["output_dir"] = str(OUTPUT)
cfg["val_manifest"] = str(OUTPUT / "val.jsonl")
cfg["model_path"] = str(OUTPUT / "checkpoint_best.pt")
cfg["resume_from_checkpoint"] = False

# v3 defaults (safe to tweak)
cfg["epochs"] = 30
cfg["learning_rate"] = 5e-5
cfg["batch_size"] = 24
cfg["dropout"] = 0.2
cfg["weight_decay"] = 0.01
cfg["label_smoothing"] = 0.1
cfg["val_max_samples"] = 300
cfg["early_stop_patience"] = 6
cfg["early_stop_min_delta"] = 0.01
cfg["decode_repetition_penalty"] = 2.5

cfg_dst.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("epochs:", cfg["epochs"], "early_stop:", cfg["early_stop_patience"])
print("val_max_samples:", cfg["val_max_samples"])
```

---

## Cell 12 — Train phase 1 (short lines)

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
p1 = cfg.get("curriculum_phase1") or {}
cfg["train_manifest"] = str(OUTPUT / "train_short.jsonl")
cfg["max_tgt_len"] = int(p1.get("max_tgt_len", 64))
cfg["epochs"] = int(p1.get("epochs", 18))
cfg["resume_from_checkpoint"] = False
Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("Phase 1:", cfg["train_manifest"], "max_tgt_len:", cfg["max_tgt_len"], "epochs:", cfg["epochs"])

import os
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Phase 1 failed exit={rc}"
```

---

## Cell 12b — Train phase 2a (medium lines, resume)

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
p2a = cfg.get("curriculum_phase2a") or {}
cfg["train_manifest"] = str(OUTPUT / "train_medium.jsonl")
cfg["max_tgt_len"] = int(p2a.get("max_tgt_len", 96))
cfg["epochs"] = int(p2a.get("epochs", 16))
cfg["learning_rate"] = float(p2a.get("learning_rate", 3e-5))
cfg["early_stop_patience"] = int(p2a.get("early_stop_patience", 8))
cfg["resume_from_checkpoint"] = True
cfg["model_path"] = str(OUTPUT / "checkpoint_best.pt")
Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("Phase 2a: medium", cfg["train_manifest"])

import os
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Phase 2a failed exit={rc}"
```

---

## Cell 12c — Train phase 2b (full train, resume)

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
p2b = cfg.get("curriculum_phase2b") or {}
cfg["train_manifest"] = str(OUTPUT / "train.jsonl")
cfg["max_tgt_len"] = int(p2b.get("max_tgt_len", 128))
cfg["epochs"] = int(p2b.get("epochs", 35))
cfg["learning_rate"] = float(p2b.get("learning_rate", 2e-5))
cfg["early_stop_patience"] = int(p2b.get("early_stop_patience", 12))
cfg["resume_from_checkpoint"] = True
cfg["model_path"] = str(OUTPUT / "checkpoint_best.pt")
Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("Phase 2b: full train", cfg["early_stop_patience"], "patience")

import os
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Phase 2b failed exit={rc}"
```

---

## Cell 12d — Train phase 2c (IAM / English long only, resume)

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
p2c = cfg.get("curriculum_phase2c") or {}
cfg["train_manifest"] = str(OUTPUT / "train_iam_long.jsonl")
cfg["max_tgt_len"] = int(p2c.get("max_tgt_len", 128))
cfg["epochs"] = int(p2c.get("epochs", 20))
cfg["learning_rate"] = float(p2c.get("learning_rate", 1e-5))
cfg["early_stop_patience"] = int(p2c.get("early_stop_patience", 10))
cfg["resume_from_checkpoint"] = True
cfg["model_path"] = str(OUTPUT / "checkpoint_best.pt")
Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("Phase 2c: IAM long only")

import os
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Phase 2c failed exit={rc}"
```

---

## Cell 13 — Sanity (strict template collapse check)

```python
import random
import torch, yaml
from pathlib import Path
from dataset import read_manifest
from decode import greedy_decode
from decode_quality import is_template_collapse, passes_export_gate
from metrics import cer
from model_factory import build_model
from tokenizer import CharTokenizer

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(OUTPUT / "checkpoint_best.pt", map_location=device, weights_only=False)
tok = CharTokenizer(cfg["vocab_path"])
if "vocab" in ckpt:
    tok.vocab = ckpt["vocab"]
    tok.stoi = {t: i for i, t in enumerate(tok.vocab)}

model = build_model(cfg, len(tok)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

rep_pen = float(cfg.get("decode_repetition_penalty", 2.5))
val = read_manifest(cfg["val_manifest"])
random.seed(42)

ok, bad = 0, 0
for s in random.sample(val, min(24, len(val))):
    pred = greedy_decode(
        model, tok, s.points, device,
        max_seq_len=cfg["max_seq_len"],
        mode=s.mode,
        repetition_penalty=rep_pen,
    )
    c = cer(pred, s.text)
    collapsed = is_template_collapse(pred, s.text, s.mode)
    tag = "COLLAPSE" if collapsed else "OK"
    if collapsed:
        bad += 1
    else:
        ok += 1
    print(f"[{tag}] [{s.mode}] CER={c:.2f}")
    print(f"  truth: {s.text[:70]!r}")
    print(f"  pred:  {pred[:70]!r}\n")

val_cer = float(ckpt.get("val_cer", 99))
print(f"Summary: OK={ok} COLLAPSE={bad} | epoch={ckpt.get('epoch')} val_cer={val_cer}")
ready, reason = passes_export_gate(
    cer_mean=val_cer,
    val_cer=val_cer,
    collapse_count=bad,
    sample_count=ok + bad,
)
print("Export gate:", ready, "-", reason)
print("(Also need eval_report cer_mean <= 0.35 in cell 15)")
```

---

## Cell 14 — Evaluate + export ONNX

```python
from pathlib import Path
from dataset import read_manifest, write_manifest

val_small = OUTPUT / "val_300.jsonl"
write_manifest(val_small, read_manifest(OUTPUT / "val.jsonl")[:300])

!python src/predict.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --manifest {val_small} \
    --output {OUTPUT / "predictions.jsonl"}

!python src/evaluate.py \
    --manifest {val_small} \
    --predictions {OUTPUT / "predictions.jsonl"} \
    --report {OUTPUT / "eval_report.json"}

!python src/export_onnx.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --trace-time 128 \
    --trace-tokens 128
```

---

## Cell 15 — Firebase upload (if CER OK)

```python
import json, os
from pathlib import Path

report = json.loads((OUTPUT / "eval_report.json").read_text(encoding="utf-8"))
cer_mean = float(report.get("cer_mean", 999))
print("cer_mean:", cer_mean, "cer_by_mode:", report.get("cer_by_mode"))

if cer_mean <= 0.35:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret(
            "FIREBASE_SERVICE_ACCOUNT_JSON"
        )
    except Exception:
        pass
    !python src/upload_firebase.py \
        --local output/model.onnx \
        --vocab configs/vocab.txt \
        --bucket aurascribblr.firebasestorage.app \
        --remote models/latest_handwriting.onnx
    print("Uploaded to Firebase")
else:
    print("SKIP Firebase — CER too high (need <= 0.35)")
```

---

## Cell 16 — Download bundle

```python
import zipfile, shutil
from pathlib import Path

bundle = Path("/kaggle/working/handwriting_bundle")
bundle.mkdir(exist_ok=True)
for src, name in [
    ("output/model.onnx", "model.onnx"),
    ("output/model.int8.onnx", "model.int8.onnx"),
    ("output/checkpoint_best.pt", "checkpoint_best.pt"),
    ("output/eval_report.json", "eval_report.json"),
    ("configs/vocab.txt", "vocab.txt"),
]:
    p = WORK / src
    if p.exists():
        shutil.copy2(p, bundle / name)
        print("OK", name)

zip_path = Path("/kaggle/working/handwriting_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in bundle.iterdir():
        zf.write(f, f.name)
print("Download:", zip_path)
```

---

## Local pack (before upload)

```powershell
cd tools\handwriting-model
python src\pack_kaggle_zip.py --output handwriting_training_bundle.zip
```

Re-upload the ZIP to Kaggle, then paste cells into a **new** notebook.

---

## When to export

| Check | Target |
|-------|--------|
| `val_cer` (best checkpoint) | **< 0.5** (ideal < 0.35) |
| `val_text` | **< 1.0** (not ~3.0) |
| Cell 13 `COLLAPSE` count | **≤ 2** / 24 (`the and`, `ה`, `\frac{1}`) |
| `val template_collapse` in train log | trending **down** |
| `eval_report.json` cer_mean | **≤ 0.35** for Firebase |

**v3.1 curriculum:** phase 1 short → 2a medium → 2b full → 2c IAM-long. Do not skip 2a/2c.

If `val_cer` still > 1.0 after 2c — add more IAM to ZIP before another run.
