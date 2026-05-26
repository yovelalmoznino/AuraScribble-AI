# Kaggle — אימון מאפס (מחברת חדשה)

מחליפה את הריצה הקודמת. **לא** טוענים `checkpoint_best.pt` מה-ZIP.

## לפני הרצה

| הגדרה | ערך |
|--------|-----|
| Accelerator | **GPU T4** |
| Dataset | ZIP עם `handwriting-model/` + `data/raw` |
| Secret (אופציונלי) | `FIREBASE_SERVICE_ACCOUNT_JSON` |

## סדר תאים

`1 → 2 → 4 → 5 → 6 → 6b → 7 → 8 → 9 → 10 → 11 → 12`

---

### תא 1 — Markdown

```markdown
# AuraScribble — Fresh training (scratch)
- No old checkpoint from ZIP
- Unbuffered logs (`python -u`)
- Upload to Firebase only if val_cer < 0.6
```

---

### תא 2 — Setup

```python
from pathlib import Path
import os
import sys
import shutil

INPUT = Path("/kaggle/input")
INPUT_ROOT = INPUT / "datasets/yovelalmoznino/aurascribblemodel/handwriting-model/handwriting-model"
if not INPUT_ROOT.exists():
    for cfg in INPUT.rglob("configs/train.yaml"):
        INPUT_ROOT = cfg.parent.parent
        print("Auto-detected:", INPUT_ROOT)
        break
assert INPUT_ROOT.exists(), f"Not found. Inputs: {[p.name for p in INPUT.iterdir()]}"

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

# FRESH: do NOT copy checkpoint from ZIP
print("FRESH training — skipping ZIP checkpoint")
print("WORK:", WORK)
print("data/raw:", DATA_RAW.exists())
```

---

### תא 4 — GPU + packages

```python
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
!pip install -q PyYAML tqdm onnx onnxruntime onnxscript google-cloud-storage
```

---

### תא 5a — עברית סינטטית (מומלץ)

```python
!pip install -q matplotlib
!python src/generate_synthetic_hebrew.py \
    --output {DATA_RAW}/synthetic_hebrew/hebrew_synthetic.jsonl \
    --sentences configs/hebrew_sentences.txt \
    --variants 6
```

~70 משפטים × 6 ≈ **420** דגימות `mode=hebrew` (קווי מתאר מגופן + רעש — לא תחליף כתב אמיתי).

---

## תא 5 — נתונים

```python
from pathlib import Path
import os

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
        --prefix training_data/new/ \
        --checkpoint
else:
    print("No Firebase — ZIP only")

all_path = OUTPUT / "all.jsonl"
!python src/prepare_raw.py --raw {DATA_RAW} --output {all_path} --no-merge-existing

if any(FB_CORR.rglob("*.json")):
    from dataset import read_firebase_corrections, read_manifest, write_manifest
    merged = read_manifest(all_path) + read_firebase_corrections(FB_CORR)
    write_manifest(all_path, merged)
    print("Merged Firebase corrections")

# סטטיסטיקה לפי mode
from collections import Counter
from dataset import read_manifest
samples = read_manifest(all_path)
modes = Counter((s.mode or "auto").lower() for s in samples)
print(f"Total: {len(samples)}")
for k, v in modes.most_common():
    print(f"  {k}: {v}")
assert len(samples) > 100, "Too few samples"
```

---

### תא 6 — split

```python
!python src/split_manifest.py \
    --source {OUTPUT / "all.jsonl"} \
    --train-out {OUTPUT / "train.jsonl"} \
    --val-out {OUTPUT / "val.jsonl"} \
    --val-ratio 0.1 \
    --seed 1337
```

---

### תא 6b — oversample עברית + תיקונים (מומלץ)

```python
import random
from dataset import read_manifest, write_manifest, HandwritingSample

def has_hebrew(s: HandwritingSample) -> bool:
    return any("\u0590" <= c <= "\u05FF" for c in s.text)

train_path = OUTPUT / "train.jsonl"
train = read_manifest(train_path)
priority = [s for s in train if s.mode.lower() in ("hebrew", "correction") or has_hebrew(s)]
rest = [s for s in train if s not in priority]
boost = 5  # כפול עברית/תיקונים
train_boosted = rest + priority * boost
random.Random(1337).shuffle(train_boosted)
write_manifest(train_path, train_boosted)
print(f"Train after boost: {len(train_boosted)} (priority x{boost}: {len(priority)} unique)")
```

---

### תא 7 — קונפיג (מאפס)

```python
from pathlib import Path
import shutil

# העתק קונפיג fresh מה-repo (אחרי copytree) או בנה ידנית:
cfg_src = WORK / "configs" / "train_kaggle_fresh.yaml"
cfg_dst = WORK / "configs" / "train_kaggle.yaml"
if cfg_src.exists():
    shutil.copy2(cfg_src, cfg_dst)
else:
    # fallback inline
    import yaml
    cfg = yaml.safe_load((WORK / "configs" / "train.yaml").read_text(encoding="utf-8"))
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
    Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

# נתיבים
import yaml
cfg = yaml.safe_load(cfg_dst.read_text(encoding="utf-8"))
cfg["output_dir"] = str(OUTPUT)
cfg["train_manifest"] = str(OUTPUT / "train.jsonl")
cfg["val_manifest"] = str(OUTPUT / "val.jsonl")
cfg["model_path"] = str(WORK / "models" / "checkpoint_best.pt")
# שמור גם export/firebase ל-export_onnx (חובה)
base = yaml.safe_load((WORK / "configs" / "train.yaml").read_text(encoding="utf-8"))
for key in ("export", "firebase", "model"):
    if key in base and key not in cfg:
        cfg[key] = base[key]
cfg_dst.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("resume_from_checkpoint:", cfg.get("resume_from_checkpoint"))
print("epochs:", cfg.get("epochs"), "batch:", cfg.get("batch_size"))
```

---

### תא 8 — אימון (חובה `python -u`)

```python
import os
print(">>> python -u src/train.py --config configs/train_kaggle.yaml")
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Training failed with code {rc}"
```

צפי: כל ~2–4 דקות שורת `epoch N batch ...`. בסוף כל epoch: `val_cer`.  
**מטרה:** `val_cer` יורד מתחת **0.5** (אידיאלי < 0.3). אם נשאר ~1.0 — משהו עדיין שבור.

---

### תא 9 — sanity (10 דוגמאות)

```python
import json
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
    input_dim=cfg["input_dim"], hidden=cfg["hidden_dim"],
    layers=cfg["num_layers"], dropout=cfg["dropout"], vocab_size=len(tok),
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

val = read_manifest(cfg["val_manifest"])[:10]
for s in val:
    pred = greedy_decode(model, tok, s.points, device, max_seq_len=cfg["max_seq_len"])
    print(f"truth={s.text[:40]!r}")
    print(f"pred ={pred[:40]!r}  cer={cer(pred, s.text):.3f}")
    print("---")
```

---

### תא 10 — predict (מוגבל)

```python
!python src/predict.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --manifest {OUTPUT / "val.jsonl"} \
    --output {OUTPUT / "predictions.jsonl"} \
    --limit 300
```

---

### תא 11 — evaluate

```python
import json
from pathlib import Path
!python src/evaluate.py \
    --manifest {OUTPUT / "val.jsonl"} \
    --predictions {OUTPUT / "predictions.jsonl"} \
    --report {OUTPUT / "eval_report.json"}
print(json.dumps(json.loads((OUTPUT / "eval_report.json").read_text()), indent=2, ensure_ascii=False))
```

---

### תא 12 — export ONNX

```python
!python src/export_onnx.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --trace-time 128 \
    --trace-tokens 96
```

---

### תא 13 — Firebase (רק אם CER סביר)

```python
import json
import os
from pathlib import Path

report = json.loads((OUTPUT / "eval_report.json").read_text(encoding="utf-8"))
cer_mean = float(report.get("cer_mean", 999))
log = OUTPUT / "training_log.jsonl"
if log.exists():
    last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    print("Last epoch log:", last)

if cer_mean > 0.6:
    raise SystemExit(
        f"cer_mean={cer_mean:.3f} too high — NOT uploading to Firebase. "
        "Fix training or run more epochs."
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
print("Uploaded — test app with new OTA model")
```

---

## מה היה שבור בריצה הקודמת (לוג Kaggle)

1. **הורדת checkpoint מה-ZIP** → המשך אימון ממשקולות ישנות (`val_cer≈1.06`).
2. **רק 167 דגימות עברית** מול ~13k IAM + 3900 math → המודל לא למד עברית.
3. **5 epochs + lr=2e-5** — מעט מדי ל-fine-tune; מאפס צריך lr גבוה יותר (`1e-4`) ו-12 epochs.
4. **פלט מבופר** — נראה שהכל קרה בבת אחת (תיקון: `python -u`).

## ציפיות זמן (T4)

| שלב | זמן |
|-----|-----|
| 12 epochs train | ~3–4 שעות |
| predict 300 | ~15 דקות |
| export + upload | ~2 דקות |
