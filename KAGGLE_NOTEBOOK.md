# AuraScribble — מחברת Kaggle מלאה

**לפני שמתחילים:** Settings → Accelerator → **GPU**  
**Dataset:** ZIP אחד (קוד + `data/raw`) — **לא מעלים מחדש** כשמשנים קוד  
**Secret (אופציונלי):** `FIREBASE_SERVICE_ACCOUNT_JSON` = מפתח Service Account (לא `google-services.json`)

**סדר הרצה:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → (12) → (13)

---

## תא 1 — Markdown

```markdown
# AuraScribble — Handwriting Training

- ZIP ב-input: קוד + data (לא נוגעים)
- קוד נערך ב-working עם `%%writefile`
- פלט: `output/` + העלאה ל-Firebase OTA
```

---

## תא 2 — Setup (העתק קוד ל-working, נתיבים)

```python
from pathlib import Path
import os
import sys
import shutil

INPUT = Path("/kaggle/input")

# === עדכן אם שם ה-dataset שונה ===
INPUT_ROOT = INPUT / "notebbok" / "tools" / "handwriting-model"
if not INPUT_ROOT.exists():
    for cfg in INPUT.rglob("configs/train.yaml"):
        INPUT_ROOT = cfg.parent.parent
        print("Auto-detected:", INPUT_ROOT)
        break

assert INPUT_ROOT.exists(), f"לא נמצא handwriting-model. Inputs: {[p.name for p in INPUT.iterdir()]}"

WORK = Path("/kaggle/working/handwriting-model")
WORK.mkdir(parents=True, exist_ok=True)

# העתק רק קוד (קטן) — לעריכה עם %%writefile
for folder in ("src", "configs", "scripts"):
    src_dir, dst_dir = INPUT_ROOT / folder, WORK / folder
    if src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print("Copied", folder)

os.chdir(WORK)
sys.path.insert(0, str(WORK / "src"))

# נתיבים — data נשארת ב-ZIP (input)
DATA_RAW = INPUT_ROOT / "data" / "raw"
MODELS = INPUT_ROOT / "models"
OUTPUT = WORK / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

(WORK / "models").mkdir(exist_ok=True)
if (MODELS / "checkpoint_best.pt").exists():
    shutil.copy2(MODELS / "checkpoint_best.pt", WORK / "models" / "checkpoint_best.pt")

print("INPUT (data):", INPUT_ROOT)
print("WORK (code): ", WORK)
print("data/raw:    ", DATA_RAW.exists())
print()
print("לעריכת קוד: תא חדש עם")
print("%%writefile /kaggle/working/handwriting-model/src/שם_קובץ.py")
```

---

## תא 3 — (אופציונלי) עריכת קוד — `%%writefile`

> הרץ תא זה **רק** כשאת משנה קובץ. הדבקי **את כל הקובץ** מתחת לשורה הראשונה.

```python
%%writefile /kaggle/working/handwriting-model/src/train.py

# הדבקי כאן את כל תוכן train.py מהמחשב
# ...
```

אחרי `%%writefile` → הרצי מחדש מתא 7.

---

## תא 4 — GPU + חבילות

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!pip install -q PyYAML tqdm onnx onnxruntime onnxscript google-cloud-storage
```

---

## תא 5 — נתונים: `data/raw` (ZIP) + Firebase → `all.jsonl`

```python
from pathlib import Path
import os
import shutil

# --- Firebase Secret (Service Account JSON) ---
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret(
        "FIREBASE_SERVICE_ACCOUNT_JSON"
    )
    print("Firebase secret OK")
except Exception as e:
    print("Firebase secret skipped:", e)

# --- הורדת תיקונים מ-Firebase ---
FB_CORR = WORK / "data" / "firebase_corrections"
FB_CORR.mkdir(parents=True, exist_ok=True)

if os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    !python src/download_firebase_corrections.py \
        --local-dir {FB_CORR} \
        --bucket aurascribblr.firebasestorage.app \
        --prefix training_data/new/ \
        --checkpoint
else:
    print("No Firebase credentials — using ZIP data only")

# --- בניית all.jsonl מ-data/raw ב-ZIP ---
raw_path = str(DATA_RAW)
all_path = str(OUTPUT / "all.jsonl")
!python src/prepare_raw.py --raw {raw_path} --output {all_path} --no-merge-existing

# --- מיזוג תיקוני Firebase (אם הורדו) ---
if any(FB_CORR.rglob("*.json")):
    from dataset import read_firebase_corrections, read_manifest, write_manifest
    merged = read_manifest(all_path) + read_firebase_corrections(FB_CORR)
    write_manifest(all_path, merged)
    print("Merged Firebase corrections")

n = sum(1 for line in open(all_path, encoding="utf-8") if line.strip())
print(f"all.jsonl: {n} samples")
assert n > 0, "אין דגימות — בדקי data/raw ב-ZIP או Firebase Secret"

print("checkpoint:", (WORK / "models" / "checkpoint_best.pt").exists())
```

---

## תא 6 — קונפיג GPU

```python
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/train.yaml").read_text(encoding="utf-8"))
cfg["batch_size"] = 64          # 128 אם יש VRAM; 32 אם OOM
cfg["epochs"] = 5               # Kaggle איטי — 5 סבבים + checkpoint לרוב מספיק
cfg["learning_rate"] = 2e-5
cfg["num_workers"] = 0          # 0 ב-Kaggle לרוב יותר יציב מ-2
cfg["val_every_epochs"] = 5     # validation רק בסבב 5 (חוסך המון זמן)
cfg["val_max_samples"] = 150    # greedy_decode על 150 דוגמאות, לא כל val
cfg["log_every_batches"] = 20   # הדפסה כל 20 batches (Kaggle לא נראה תקוע)
cfg["output_dir"] = str(OUTPUT)
cfg["train_manifest"] = str(OUTPUT / "train.jsonl")
cfg["val_manifest"] = str(OUTPUT / "val.jsonl")
cfg["model_path"] = str(WORK / "models" / "checkpoint_best.pt")

Path("configs/train_kaggle.yaml").write_text(
    yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
    encoding="utf-8",
)
print(f"Config: {cfg['epochs']} epochs, batch={cfg['batch_size']}")
```

---

## תא 7 — split train / val

```python
src = str(OUTPUT / "all.jsonl")
train_out = str(OUTPUT / "train.jsonl")
val_out = str(OUTPUT / "val.jsonl")
!python src/split_manifest.py --source {src} --train-out {train_out} --val-out {val_out} --val-ratio 0.1 --seed 1337
```

---

## תא 8 — אימון

```python
import os

print(">>> python -u src/train.py --config configs/train_kaggle.yaml")
# -u = unbuffered stdout (חשוב ב-Kaggle — אחרת אין פלט עד סוף epoch)
assert os.system("python -u src/train.py --config configs/train_kaggle.yaml") == 0, "Training failed"
```

---

## תא 9 — predict

```python
!python src/predict.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --manifest {OUTPUT / "val.jsonl"} \
    --output {OUTPUT / "predictions.jsonl"}
```

---

## תא 10 — evaluate

```python
from pathlib import Path
import json

!python src/evaluate.py \
    --manifest {OUTPUT / "val.jsonl"} \
    --predictions {OUTPUT / "predictions.jsonl"} \
    --report {OUTPUT / "eval_report.json"}

report = json.loads(Path(OUTPUT / "eval_report.json").read_text(encoding="utf-8"))
print(json.dumps(report, indent=2, ensure_ascii=False))
```

---

## תא 11 — export ONNX

```python
!python src/export_onnx.py \
    --config configs/train_kaggle.yaml \
    --checkpoint output/checkpoint_best.pt \
    --trace-time 128 \
    --trace-tokens 96
```

---

## תא 12 — ZIP להורדה

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
    ("output/export_summary.json", "export_summary.json"),
    ("configs/vocab.txt", "vocab.txt"),
]:
    p = WORK / src
    if p.exists():
        shutil.copy(p, bundle / name)
        print("OK", name)

zip_path = Path("/kaggle/working/handwriting_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in bundle.iterdir():
        if f.is_file():
            zf.write(f, f.name)

print("Download from Output:", zip_path)

log = OUTPUT / "training_log.jsonl"
if log.exists():
    print("\n=== Last epochs ===")
    for line in log.read_text(encoding="utf-8").strip().splitlines()[-5:]:
        r = json.loads(line)
        print(f"epoch {r['epoch']}: loss={r.get('train_loss')} val_cer={r.get('val_cer')}")
```

---

## תא 13 — גרף (אופציונלי)

```python
from pathlib import Path
import json
import matplotlib.pyplot as plt

log = OUTPUT / "training_log.jsonl"
if not log.exists():
    print("No training log")
else:
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    epochs = [r["epoch"] for r in rows]
    losses = [r.get("train_loss") for r in rows]
    val_cers = [r.get("val_cer") for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, losses, "b-o", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_cers, "r-s", label="val_cer")
    ax2.set_ylabel("val_cer", color="r")
    plt.title("Training progress")
    fig.tight_layout()
    plt.show()
```

---

## תא 14 — העלאת מודל ל-Firebase (OTA)

```python
import os
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret(
        "FIREBASE_SERVICE_ACCOUNT_JSON"
    )
    print("Firebase secret OK")
except Exception as e:
    print("No secret:", e)

assert (WORK / "output" / "model.onnx").exists(), "Run export (cell 11) first"

!python src/upload_firebase.py \
    --local output/model.onnx \
    --vocab configs/vocab.txt \
    --bucket aurascribblr.firebasestorage.app \
    --remote models/latest_handwriting.onnx

print("Done. האפליקציה תוריד OTA מ-models/latest_handwriting.onnx")
```

---

## עזרה מהירה

| בעיה | פתרון |
|------|--------|
| `לא נמצא handwriting-model` | בתא 2 עדכן `INPUT_ROOT` לנתיב ב-ZIP |
| `missing token_uri, client_email` | Secret שגוי — צריך **Service Account key**, לא `google-services.json` |
| `0 samples` | בדקי `data/raw` ב-ZIP או Firebase Secret |
| OOM | בתא 6: `batch_size = 32` |
| 40+ דקות בלי פלט | Settings → **GPU**; תא 8 עם `python -u`; בדקי `Using device: cuda` |
| תקוע בלי `Epoch 1` | גללי למעלה — אם `Using device: cpu` האימון איטי מאוד (~שעה/epoch) |
| שינית קוד | `%%writefile` בתא 3 → הרצי מחדש מתא 8 |
| שינית data | העלי ZIP dataset מחדש (לא קשור לקוד) |

## מתי מעלים ZIP מחדש?

| שינוי | ZIP חדש? |
|--------|-----------|
| קוד Python | **לא** — `%%writefile` |
| `data/raw` | **כן** |
| checkpoint ב-ZIP | **כן** |
