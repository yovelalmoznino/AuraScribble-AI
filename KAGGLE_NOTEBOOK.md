# Kaggle — מחברת אימון מלאה

העתק כל בלוק `# %%` לתא נפרד במחברת Kaggle.  
הפעל **GPU** (Settings → Accelerator → GPU T4 x2 או P100).

## לפני שמתחילים — Datasets ב-Kaggle

הוסף ל-Notebook (Add Data):

| Dataset | תוכן מומלץ |
|---------|------------|
| **Repo** | כל הפרויקט (או לפחות `tools/handwriting-model/`) |
| **Data** (אופציונלי) | `data/processed/all.jsonl` + תיקיית `data/corrections/` |
| **Checkpoint** (אופציונלי) | `models/checkpoint_best.pt` ל-fine-tune |

בתא 2 עדכן את `REPO_ROOT` לנתיב האמיתי תחת `/kaggle/input/...`.

---

## תא 1 — כותרת (Markdown)

```markdown
# AuraScribble — Handwriting Training (Kaggle GPU)

Pipeline: split → train → predict → evaluate → export ONNX

**פלט:** הורד מ-Output את `handwriting_bundle.zip` (מכיל `model.onnx`, checkpoint, דוחות).
```

---

## תא 2 — הגדרות נתיבים (ערוך כאן)

```python
from pathlib import Path
import os
import sys

# === ערוך לפי שמות ה-Datasets שלך ב-Kaggle ===
REPO_INPUT = Path("/kaggle/input")  # שורש כל ה-inputs

# דוגמה: אם ה-repo הוא dataset בשם "notebbok"
REPO_ROOT = REPO_INPUT / "notebbok" / "tools" / "handwriting-model"

# אם כל handwriting-model הוא dataset נפרד בשם "aurascribble-handwriting":
# REPO_ROOT = REPO_INPUT / "aurascribble-handwriting"

# נתונים חיצוניים (אם לא בתוך ה-repo)
DATA_INPUT = REPO_INPUT / "handwriting-data"  # אופציונלי — מחק/שנה אם לא קיים
CHECKPOINT_INPUT = REPO_INPUT / "handwriting-checkpoint"  # אופציונלי

# עבודה (פלט להורדה)
WORK = Path("/kaggle/working/handwriting_run")
WORK.mkdir(parents=True, exist_ok=True)

assert REPO_ROOT.exists(), f"Repo not found: {REPO_ROOT}\nList input: {list(REPO_INPUT.iterdir())}"
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

print("CWD:", os.getcwd())
print("Repo OK:", REPO_ROOT)
```

---

## תא 3 — GPU + התקנת חבילות

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!pip install -q PyYAML tqdm onnx onnxruntime onnxscript
```

---

## תא 4 — הכנת נתונים ו-checkpoint

```python
import shutil
import yaml

# העתקת נתונים מ-dataset חיצוני (אם יש)
all_jsonl = Path("data/processed/all.jsonl")
all_jsonl.parent.mkdir(parents=True, exist_ok=True)

if DATA_INPUT.exists():
    src_all = DATA_INPUT / "all.jsonl"
    if src_all.exists():
        shutil.copy(src_all, all_jsonl)
        print("Copied all.jsonl from DATA_INPUT")
    src_processed = DATA_INPUT / "processed" / "all.jsonl"
    if src_processed.exists():
        shutil.copy(src_processed, all_jsonl)
        print("Copied processed/all.jsonl from DATA_INPUT")

if not all_jsonl.exists():
    train_only = Path("data/processed/train.jsonl")
    if train_only.exists():
        shutil.copy(train_only, all_jsonl)
        print("Using train.jsonl as all.jsonl for split")
    else:
        raise FileNotFoundError(
            "No data/processed/all.jsonl or train.jsonl. "
            "Add a Kaggle dataset with training JSONL."
        )

# Checkpoint התחלתי (fine-tune)
ckpt_dst = Path("models/checkpoint_best.pt")
ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
if CHECKPOINT_INPUT.exists():
    for name in ("checkpoint_best.pt", "models/checkpoint_best.pt"):
        src = CHECKPOINT_INPUT / name
        if src.exists():
            shutil.copy(src, ckpt_dst)
            print("Copied starter checkpoint from", src)
            break
elif not ckpt_dst.exists():
    print("WARNING: No models/checkpoint_best.pt — training from scratch")

# תיקונים מ-Firebase (אם הורדת ל-dataset)
corr_src = DATA_INPUT / "corrections" if DATA_INPUT.exists() else None
corr_dst = Path("data/corrections")
if corr_src and corr_src.exists():
    if corr_dst.exists():
        shutil.rmtree(corr_dst)
    shutil.copytree(corr_src, corr_dst)
    print("Copied corrections:", len(list(corr_dst.glob("*.json"))), "files")
```

---

## תא 5 — קונפיג ל-Kaggle GPU

```python
cfg_path = Path("configs/train.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

# התאמה ל-GPU — אימון "מסיבי"
cfg["batch_size"] = 64          # 32 אם OOM
cfg["epochs"] = 20              # הגדל ל-30+ אם יש זמן
cfg["learning_rate"] = 2e-5
cfg["num_workers"] = 2
cfg["output_dir"] = "output"

kaggle_cfg = Path("configs/train_kaggle.yaml")
kaggle_cfg.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8")
print("Wrote", kaggle_cfg)
print("epochs:", cfg["epochs"], "| batch_size:", cfg["batch_size"])
```

---

## תא 6 — פיצול train / val

```python
!python src/split_manifest.py \
  --source data/processed/all.jsonl \
  --train-out data/processed/train.jsonl \
  --val-out data/processed/val.jsonl \
  --val-ratio 0.1 \
  --seed 1337
```

---

## תא 7 — אימון (השלב הכבד)

```python
import os
corrections_arg = ""
if Path("data/corrections").exists() and any(Path("data/corrections").glob("*.json")):
    corrections_arg = "--corrections_dir data/corrections"

cmd = f"python src/train.py --config configs/train_kaggle.yaml {corrections_arg}"
print("Running:", cmd)
os.system(cmd)
```

---

## תא 8 — חיזוי על validation

```python
!python src/predict.py \
  --config configs/train_kaggle.yaml \
  --checkpoint output/checkpoint_best.pt \
  --manifest data/processed/val.jsonl \
  --output output/predictions.jsonl
```

---

## תא 9 — הערכה (CER)

```python
!python src/evaluate.py \
  --manifest data/processed/val.jsonl \
  --predictions output/predictions.jsonl \
  --report output/eval_report.json

import json
report = json.loads(Path("output/eval_report.json").read_text(encoding="utf-8"))
print("=== Eval report ===")
print(json.dumps(report, indent=2, ensure_ascii=False))
```

---

## תא 10 — ייצוא ONNX

```python
!python src/export_onnx.py \
  --config configs/train_kaggle.yaml \
  --checkpoint output/checkpoint_best.pt \
  --trace-time 128 \
  --trace-tokens 96
```

---

## תא 11 — אריזה להורדה + תצוגת לוג אימון

```python
import json
import zipfile

bundle_dir = Path("/kaggle/working/handwriting_bundle")
bundle_dir.mkdir(parents=True, exist_ok=True)

files_to_copy = [
    ("output/model.onnx", "model.onnx"),
    ("output/model.int8.onnx", "model.int8.onnx"),
    ("output/checkpoint_best.pt", "checkpoint_best.pt"),
    ("output/eval_report.json", "eval_report.json"),
    ("output/training_log.jsonl", "training_log.jsonl"),
    ("output/export_summary.json", "export_summary.json"),
    ("configs/vocab.txt", "vocab.txt"),
]

for src, name in files_to_copy:
    p = Path(src)
    if p.exists():
        shutil.copy(p, bundle_dir / name)
        print("OK", name)

zip_path = Path("/kaggle/working/handwriting_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in bundle_dir.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(bundle_dir))

print("\nDownload from Output:", zip_path)

# סיכום אימון
log_path = Path("output/training_log.jsonl")
if log_path.exists():
    print("\n=== Training log (last 5 epochs) ===")
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines[-5:]:
        row = json.loads(line)
        print(f"epoch {row.get('epoch')}: loss={row.get('train_loss')} val_cer={row.get('val_cer')}")
```

---

## תא 12 — (אופציונלי) גרף val_cer

```python
import json
import matplotlib.pyplot as plt

log_path = Path("output/training_log.jsonl")
if not log_path.exists():
    print("No training log")
else:
    epochs, val_cers, losses = [], [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        epochs.append(r["epoch"])
        losses.append(r.get("train_loss"))
        if "val_cer" in r:
            val_cers.append(r["val_cer"])
        else:
            val_cers.append(None)

    fig, ax1 = plt.subplots(figsize=(8, 4))
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

## אחרי Kaggle — פריסה

1. הורד `handwriting_bundle.zip` מ-Output (אם לא העלית ישירות — ראה תא 13)  
2. **אוטומטי:** הרץ תא 13 עם Kaggle Secret `FIREBASE_SERVICE_ACCOUNT_JSON`  
3. **ידני:** העלה `model.onnx` ל-Firebase Console → `models/latest_handwriting.onnx`  
4. אם `vocab.txt` השתנה — עדכן גם ב-assets של האפליקציה  

---

## תא 13 — העלאה אוטומטית ל-Firebase

1. Firebase Console → Service accounts → Generate private key  
2. Kaggle Notebook → **Add-ons → Secrets** → שם: `FIREBASE_SERVICE_ACCOUNT_JSON` → הדבק את JSON המלא  

```python
import os
!pip install -q google-cloud-storage

try:
    from kaggle_secrets import UserSecretsClient
    sa_json = UserSecretsClient().get_secret("FIREBASE_SERVICE_ACCOUNT_JSON")
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = sa_json
    print("Loaded Firebase credentials from Kaggle secret")
except Exception as e:
    print("No Kaggle secret — set FIREBASE_SERVICE_ACCOUNT_JSON manually or skip upload:", e)

!python src/upload_firebase.py \
  --local output/model.onnx \
  --vocab configs/vocab.txt \
  --bucket aurascribblr.firebasestorage.app \
  --remote models/latest_handwriting.onnx
```

---

## פתרון בעיות ב-Kaggle

| בעיה | פתרון |
|------|--------|
| `CUDA out of memory` | בתא 5: `batch_size = 32` או `16` |
| `Repo not found` | תקן `REPO_ROOT` בתא 2 |
| אימון איטי | ודא GPU מופעל, לא CPU |
| `No module named dataset` | ודא `os.chdir(REPO_ROOT)` ו-`sys.path` בתא 2 |
| Session timeout | שמור checkpoint ל-Output; המשך מ-`output/checkpoint_best.pt` |
