# AuraScribble — Kaggle Rebuild Notebook (Transformer + Mixed + LaTeX)

Copy each cell into a **new** Kaggle notebook. Order: **1 → 16**.

## Kaggle settings

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x1** |
| Dataset | Upload `handwriting_training_bundle.zip` |
| Secret (optional) | `FIREBASE_SERVICE_ACCOUNT_JSON` |

## ZIP layout

```
handwriting-model/
  src/
  configs/          # vocab.txt, train_kaggle_rebuild.yaml, hebrew_sentences.txt
  scripts/
  data/raw/         # iam/, crohme/, optional hhd/
```

**Do not** include old `checkpoint_best.pt` in the ZIP.

---

## Cell 1 — Markdown

```markdown
# AuraScribble handwriting rebuild
- StrokeTransformer from scratch
- Hebrew (final letters) + English (case) + Math LaTeX + **mixed**
- Upload to Firebase if val CER is good
```

---

## Cell 2 — Setup

```python
from pathlib import Path
import os, sys, shutil

INPUT = Path("/kaggle/input")
INPUT_ROOT = None
for cfg in INPUT.rglob("configs/train_kaggle_rebuild.yaml"):
    INPUT_ROOT = cfg.parent.parent
    break
if INPUT_ROOT is None:
    for cfg in INPUT.rglob("configs/train.yaml"):
        INPUT_ROOT = cfg.parent.parent
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
            print("Copied data/raw/", item.name)
        else:
            shutil.copy2(item, dest)
            print("Copied data/raw/", item.name)

OUTPUT = WORK / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
(WORK / "models").mkdir(exist_ok=True)
print("INPUT:", INPUT_ROOT)
print("data/raw:", DATA_RAW.exists())
print("FRESH — no checkpoint from ZIP")
```

---

## Cell 3 — GPU + packages

```python
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
!pip install -q PyYAML tqdm onnx onnxruntime onnxscript google-cloud-storage matplotlib pillow scikit-image
```

---

## Cell 4 — HHD convert (auto-download if missing)

```python
from pathlib import Path

hhd = DATA_RAW / "hhd"
hhd.mkdir(parents=True, exist_ok=True)
strokes_jsonl = DATA_RAW / "hhd" / "hhd_strokes.jsonl"

if not any(hhd.rglob("*.png")) and not any(hhd.rglob("*.jpg")) and not any(hhd.rglob("*.parquet")):
    print("No local HHD — downloading from HuggingFace (~40 MB)...")
    !pip install -q huggingface_hub datasets pyarrow
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="sivan22/hebrew-handwritten-dataset",
        repo_type="dataset",
        local_dir=str(hhd),
    )
    print("Downloaded HHD to", hhd)

if any(hhd.rglob("*.parquet")) or any(hhd.rglob("*.png")) or any(hhd.rglob("*.jpg")):
    !pip install -q datasets pyarrow
    !python src/convert_hhd_to_jsonl.py --input {hhd} --output {strokes_jsonl}
    import json
    from pathlib import Path
    n = sum(1 for _ in open(strokes_jsonl, encoding="utf-8")) if Path(strokes_jsonl).exists() else 0
    print(f"HHD strokes JSONL lines: {n}")
else:
    print("Skip HHD — no parquet/images found after download")
```

---

## Cell 5 — Synthetic Hebrew (final letters)

```python
!python src/generate_synthetic_hebrew.py \
    --output {DATA_RAW}/synthetic_hebrew/hebrew_synthetic.jsonl \
    --sentences configs/hebrew_sentences.txt \
    --variants 10
```

---

## Cell 6 — Synthetic mixed (he+en+math)

```python
!python src/generate_synthetic_mixed.py \
    --output {DATA_RAW}/synthetic_mixed/mixed_synthetic.jsonl \
    --per-template 10
```

---

## Cell 7 — Prepare data + Firebase corrections

```python
from pathlib import Path
import os
from collections import Counter
from dataset import read_manifest, read_firebase_corrections, write_manifest

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret(
        "FIREBASE_SERVICE_ACCOUNT_JSON"
    )
    print("Firebase secret OK")
except Exception as e:
    print("Firebase secret skipped:", e)

FB_CORR = WORK / "data" / "firebase_corrections"
FB_CORR.mkdir(parents=True, exist_ok=True)
if os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON"):
    !python src/download_firebase_corrections.py \
        --local-dir {FB_CORR} \
        --bucket aurascribblr.firebasestorage.app \
        --prefix training_data/new/
else:
    print("No Firebase — ZIP only")

all_path = OUTPUT / "all.jsonl"
!python src/prepare_raw.py --raw {DATA_RAW} --output {all_path} --no-merge-existing

if any(FB_CORR.rglob("*.json")):
    merged = read_manifest(all_path) + read_firebase_corrections(FB_CORR)
    write_manifest(all_path, merged)

samples = read_manifest(all_path)
modes = Counter((s.mode or "auto").lower() for s in samples)
print(f"Total: {len(samples)}")
for k, v in modes.most_common():
    print(f"  {k}: {v}")
assert len(samples) > 200, "Too few samples — add iam/ + crohme/ to ZIP"
```

---

## Cell 8 — Build vocab

```python
!python src/build_vocab.py --manifest {OUTPUT / "all.jsonl"} --output configs/vocab.txt
```

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

## Cell 10 — Oversample mixed + Hebrew + corrections

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

train_path = OUTPUT / "train.jsonl"
train = read_manifest(train_path)
priority = [s for s in train if is_priority(s)]
rest = [s for s in train if s not in priority]
train_boosted = rest + priority * 3
random.Random(1337).shuffle(train_boosted)
write_manifest(train_path, train_boosted)
print(f"Train: {len(train_boosted)} (boosted {len(priority)} x3)")
```

---

## Cell 11 — Config

```python
import shutil, yaml
from pathlib import Path

cfg_src = WORK / "configs" / "train_kaggle_rebuild.yaml"
cfg_dst = WORK / "configs" / "train_kaggle.yaml"
shutil.copy2(cfg_src, cfg_dst)
cfg = yaml.safe_load(cfg_dst.read_text(encoding="utf-8"))
cfg["output_dir"] = str(OUTPUT)
cfg["train_manifest"] = str(OUTPUT / "train.jsonl")
cfg["val_manifest"] = str(OUTPUT / "val.jsonl")
cfg["model_path"] = str(WORK / "models" / "checkpoint_best.pt")
cfg["resume_from_checkpoint"] = False
# Retrain overrides (see train_kaggle_rebuild.yaml in repo)
cfg["epochs"] = 40
cfg["learning_rate"] = 5e-5
cfg["batch_size"] = 24
cfg["dropout"] = 0.2
cfg["val_max_samples"] = None
cfg_dst.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("model_type:", cfg.get("model_type"))
print("epochs:", cfg.get("epochs"), "batch:", cfg.get("batch_size"), "lr:", cfg.get("learning_rate"))
```

---

## Cell 12 — Train (run once)

```python
import os
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Training failed exit={rc}"
```

---

## Cell 13 — Sanity (mixed + math)

```python
import torch, yaml
from pathlib import Path
from dataset import read_manifest
from decode import greedy_decode
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

checks = [
    s for s in read_manifest(cfg["val_manifest"])
    if (s.mode or "").lower() in ("mixed", "math", "hebrew")
][:12]
for s in checks:
    pred = greedy_decode(model, tok, s.points, device, max_seq_len=cfg["max_seq_len"], mode=s.mode)
    print(f"[{s.mode}] truth={s.text[:60]!r}")
    print(f"       pred ={pred[:60]!r}  cer={cer(pred, s.text):.3f}")
    print("---")
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
    print("SKIP Firebase — CER too high")
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

## Quality targets

| mode | CER target |
|------|------------|
| english | < 0.20 |
| hebrew | < 0.30 |
| math | < 0.25 |
| mixed | < 0.35 |
| mean | < 0.30 |
