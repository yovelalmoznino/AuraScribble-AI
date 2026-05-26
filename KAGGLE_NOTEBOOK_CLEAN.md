# AuraScribble — מחברת Kaggle מלאה (נקייה)

מחברת חדשה מאפס. העתיקי **תא-תא** (Code) לפי הסדר.

---

## לפני שמתחילים

### Kaggle

| הגדרה | ערך |
|--------|-----|
| **Settings → Accelerator** | GPU T4 (או P100) |
| **Dataset** | ZIP (מבנה למטה) |
| **Secret** (אופציונלי) | `FIREBASE_SERVICE_ACCOUNT_JSON` = מפתח Service Account |

### מה שם ב-ZIP

```
handwriting-model/
  src/          ← כל הקוד (כולל generate_synthetic_hebrew.py)
  configs/      ← vocab.txt, train.yaml, train_kaggle_fresh.yaml, hebrew_sentences.txt
  scripts/
  data/raw/     ← iam, crohme, hebrew_source.jsonl, ...
```

**אל תכללי** `models/checkpoint_best.pt` ישן ב-ZIP אם את מאמנת מאפס.

### venv במחשב (Windows)

```powershell
cd C:\Users\yalmo\Desktop\Notebbok\tools\handwriting-model

# פעם ראשונה בלבד:
python -m venv .venv

# בכל פעם שפותחים טרמינל:
.\.venv\Scripts\activate

# אמור להופיע (.venv) בתחילת השורה
pip install -r requirements.txt

# יציאה:
deactivate
```

---

## סדר תאים

`1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14`

---

## תא 1 — Markdown

```markdown
# AuraScribble Handwriting — Kaggle (clean)
- Fresh training (no old checkpoint)
- Synthetic Hebrew + IAM + CROHME + Firebase corrections
- `python -u` for live logs
- Firebase upload only if CER is good enough
```

---

## תא 2 — Setup

```python
from pathlib import Path
import os
import sys
import shutil

INPUT = Path("/kaggle/input")

# === עדכני לשם ה-Dataset שלך ב-Kaggle ===
INPUT_ROOT = INPUT / "datasets/yovelalmoznino/aurascribblemodel/handwriting-model/handwriting-model"
if not INPUT_ROOT.exists():
    for cfg in INPUT.rglob("configs/train.yaml"):
        INPUT_ROOT = cfg.parent.parent
        print("Auto-detected:", INPUT_ROOT)
        break

assert INPUT_ROOT.exists(), f"handwriting-model not found. Inputs: {[p.name for p in INPUT.iterdir()]}"

WORK = Path("/kaggle/working/handwriting-model")
WORK.mkdir(parents=True, exist_ok=True)

for folder in ("src", "configs", "scripts"):
    src_dir, dst_dir = INPUT_ROOT / folder, WORK / folder
    if src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print("Copied", folder)

os.chdir(WORK)
sys.path.insert(0, str(WORK / "src"))

DATA_RAW = INPUT_ROOT / "data" / "raw"
OUTPUT = WORK / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
(WORK / "models").mkdir(exist_ok=True)

print("INPUT (read-only data):", INPUT_ROOT)
print("WORK (code + output):  ", WORK)
print("data/raw exists:", DATA_RAW.exists())
print("FRESH — not loading checkpoint from ZIP")
```

---

## תא 3 — GPU + חבילות

```python
import torch

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("Enable GPU: Settings → Accelerator → GPU")

!pip install -q PyYAML tqdm onnx onnxruntime onnxscript google-cloud-storage matplotlib
```

---

## תא 4 — עברית סינטטית

```python
!python src/generate_synthetic_hebrew.py \
    --output {DATA_RAW}/synthetic_hebrew/hebrew_synthetic.jsonl \
    --sentences configs/hebrew_sentences.txt \
    --variants 8
```

---

## תא 5 — נתונים (raw + Firebase)

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
    # בלי --checkpoint (אימון מאפס!)
    !python src/download_firebase_corrections.py \
        --local-dir {FB_CORR} \
        --bucket aurascribblr.firebasestorage.app \
        --prefix training_data/new/
else:
    print("No Firebase — using ZIP data only")

all_path = OUTPUT / "all.jsonl"
!python src/prepare_raw.py --raw {DATA_RAW} --output {all_path} --no-merge-existing

if any(FB_CORR.rglob("*.json")):
    merged = read_manifest(all_path) + read_firebase_corrections(FB_CORR)
    write_manifest(all_path, merged)
    print("Merged Firebase corrections")

samples = read_manifest(all_path)
modes = Counter((s.mode or "auto").lower() for s in samples)
print(f"Total samples: {len(samples)}")
for k, v in modes.most_common():
    print(f"  {k}: {v}")
assert len(samples) > 500, "Too few samples — check ZIP data/raw"
```

---

## תא 6 — split train / val

```python
!python src/split_manifest.py \
    --source {OUTPUT / "all.jsonl"} \
    --train-out {OUTPUT / "train.jsonl"} \
    --val-out {OUTPUT / "val.jsonl"} \
    --val-ratio 0.1 \
    --seed 1337
```

---

## תא 7 — oversample עברית + תיקונים

```python
import random
from dataset import read_manifest, write_manifest, HandwritingSample

def has_hebrew(s: HandwritingSample) -> bool:
    return any("\u0590" <= c <= "\u05FF" for c in s.text)

train_path = OUTPUT / "train.jsonl"
train = read_manifest(train_path)
priority = [
    s for s in train
    if s.mode.lower() in ("hebrew", "correction") or has_hebrew(s)
]
rest = [s for s in train if s not in priority]
boost = 5
train_boosted = rest + priority * boost
random.Random(1337).shuffle(train_boosted)
write_manifest(train_path, train_boosted)
print(f"Train size: {len(train_boosted)} (boosted {len(priority)} priority x{boost})")
```

---

## תא 8 — קונפיג אימון

```python
from pathlib import Path
import shutil
import yaml

cfg_src = WORK / "configs" / "train_kaggle_fresh.yaml"
cfg_dst = WORK / "configs" / "train_kaggle.yaml"

if cfg_src.exists():
    shutil.copy2(cfg_src, cfg_dst)
    cfg = yaml.safe_load(cfg_dst.read_text(encoding="utf-8"))
else:
    base = yaml.safe_load((WORK / "configs" / "train.yaml").read_text(encoding="utf-8"))
    cfg = base
    cfg["resume_from_checkpoint"] = False
    cfg["epochs"] = 12
    cfg["batch_size"] = 48
    cfg["learning_rate"] = 1e-4
    cfg["val_every_epochs"] = 1
    cfg["val_max_samples"] = 200
    cfg["log_every_batches"] = 15
    cfg["mode_loss_weights"] = {"hebrew": 4.0, "correction": 5.0, "text": 2.0}

cfg["output_dir"] = str(OUTPUT)
cfg["train_manifest"] = str(OUTPUT / "train.jsonl")
cfg["val_manifest"] = str(OUTPUT / "val.jsonl")
cfg["model_path"] = str(WORK / "models" / "checkpoint_best.pt")
cfg["resume_from_checkpoint"] = False

base_full = yaml.safe_load((WORK / "configs" / "train.yaml").read_text(encoding="utf-8"))
for key in ("export", "firebase"):
    if key in base_full:
        cfg[key] = base_full[key]

cfg_dst.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("resume_from_checkpoint:", cfg["resume_from_checkpoint"])
print("epochs:", cfg["epochs"], "batch:", cfg["batch_size"], "lr:", cfg["learning_rate"])
```

---

## תא 9 — אימון (הריצי פעם אחת בלבד!)

```python
import os

print(">>> python -u src/train.py --config configs/train_kaggle.yaml")
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Training failed, exit code {rc}"
```

צפי: ~2.5 דקות לכל epoch (train) + ~5 דקות validation.  
12 epochs ≈ **~3 שעות**. תראי `epoch N batch 15/342` כל כמה דקות.

**מטרה:** `val_cer` < **0.5** (טוב: < 0.3).

---

## תא 10 — sanity (10 דוגמאות)

```python
import torch
import yaml
from pathlib import Path
from dataset import read_manifest
from decode import greedy_decode
from metrics import cer
from model import HandwritingSeq2SeqModel
from tokenizer import CharTokenizer

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(OUTPUT / "checkpoint_best.pt", map_location=device, weights_only=False)
tok = CharTokenizer(cfg["vocab_path"])
if "vocab" in ckpt:
    tok.vocab = ckpt["vocab"]
    tok.stoi = {t: i for i, t in enumerate(tok.vocab)}

model = HandwritingSeq2SeqModel(
    input_dim=cfg["input_dim"],
    hidden=cfg["hidden_dim"],
    layers=cfg["num_layers"],
    dropout=cfg["dropout"],
    vocab_size=len(tok),
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

for s in read_manifest(cfg["val_manifest"])[:10]:
    pred = greedy_decode(model, tok, s.points, device, max_seq_len=cfg["max_seq_len"])
    print(f"truth={s.text[:50]!r}")
    print(f"pred ={pred[:50]!r}  cer={cer(pred, s.text):.3f}")
    print("---")
```

---

## תא 11 — predict + evaluate (300 דוגמאות)

```python
import json
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

print(json.dumps(json.loads((OUTPUT / "eval_report.json").read_text(encoding="utf-8")), indent=2))
```

---

## תא 12 — export ONNX

```python
!python src/export_onnx.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --trace-time 128 \
    --trace-tokens 96
```

---

## תא 13 — Firebase (רק אם CER טוב)

```python
import json
import os
from pathlib import Path

report = json.loads((OUTPUT / "eval_report.json").read_text(encoding="utf-8"))
cer_mean = float(report.get("cer_mean", 999))
print("cer_mean on val_300:", cer_mean)

if cer_mean > 0.6:
    raise SystemExit(
        f"cer_mean={cer_mean:.3f} — NOT uploading. Collect more Hebrew / train longer."
    )

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

print("Uploaded models/latest_handwriting.onnx + vocab")
```

---

## תא 14 — ZIP להורדה

```python
from pathlib import Path
import json
import zipfile
import shutil

bundle = Path("/kaggle/working/handwriting_bundle")
bundle.mkdir(exist_ok=True)

for src, name in [
    ("output/model.onnx", "model.onnx"),
    ("output/model.int8.onnx", "model.int8.onnx"),
    ("output/checkpoint_best.pt", "checkpoint_best.pt"),
    ("output/eval_report.json", "eval_report.json"),
    ("output/training_log.jsonl", "training_log.jsonl"),
    ("configs/vocab.txt", "vocab.txt"),
]:
    p = WORK / src
    if p.exists():
        shutil.copy2(p, bundle / name)
        print("OK", name)

zip_path = Path("/kaggle/working/handwriting_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in bundle.iterdir():
        if f.is_file():
            zf.write(f, f.name)

print("Download:", zip_path)

log = OUTPUT / "training_log.jsonl"
if log.exists():
    print("\n=== Training log (last 5 epochs) ===")
    for line in log.read_text(encoding="utf-8").strip().splitlines()[-5:]:
        print(json.loads(line))
```

---

## טבלת פתרון בעיות

| בעיה | פתרון |
|------|--------|
| `handwriting-model not found` | עדכני `INPUT_ROOT` בתא 2 לשם ה-dataset |
| אין פלט באימון | ודאי `python -u` בתא 9; GPU מופעל |
| אימון כפול (שורות כפולות) | הריצי תא 9 **פעם אחת** |
| `KeyError: export` | ודאי `train_kaggle_fresh.yaml` ב-ZIP (תא 8) |
| `cer_mean` גבוה | עוד עברית (תא 4 + איסוף), אל תעלי Firebase |
| OOM | בתא 8: `batch_size: 32` |

## זמן משוער (T4)

| שלב | זמן |
|-----|-----|
| נתונים + synthetic | ~3 דקות |
| 12 epochs | ~3 שעות |
| predict/eval/export | ~20 דקות |
