# Kaggle — מחברת אימון מלאה (כולל `data/raw`)

הפעל **GPU**. הוסף Dataset עם ה-repo (כולל `data/raw/`).

---

## תא 1 — Markdown

```markdown
# AuraScribble — Handwriting Training (Kaggle GPU)

**Pipeline:** raw → all.jsonl → split → train → predict → evaluate → export → Firebase OTA
```

---

## תא 2 — נתיבים

```python
from pathlib import Path
import os
import sys

REPO_INPUT = Path("/kaggle/input")
REPO_ROOT = REPO_INPUT / "notebbok" / "tools" / "handwriting-model"
DATA_INPUT = REPO_INPUT / "handwriting-data"  # אופציונלי

if not REPO_ROOT.exists():
    for train_py in REPO_INPUT.rglob("src/train.py"):
        c = train_py.parent.parent
        if (c / "configs" / "train.yaml").exists():
            REPO_ROOT = c
            print("Auto-detected:", REPO_ROOT)
            break

assert REPO_ROOT.exists(), f"Repo not found. Inputs: {[p.name for p in REPO_INPUT.iterdir()]}"

os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))
print("CWD:", os.getcwd())
print("data/raw exists:", Path("data/raw").exists())
```

---

## תא 3 — GPU + pip

```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
!pip install -q PyYAML tqdm onnx onnxruntime onnxscript google-cloud-storage
```

---

## תא 4 — Firebase תיקונים + `data/raw` → all.jsonl + checkpoint

**Secret:** Add-ons → Secrets → `FIREBASE_SERVICE_ACCOUNT_JSON`

```python
from pathlib import Path
import os
import shutil

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret(
        "FIREBASE_SERVICE_ACCOUNT_JSON"
    )
    print("Firebase secret loaded")
except Exception as e:
    print("No Kaggle secret:", e)

if os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    !python src/download_firebase_corrections.py \
      --local-dir data/raw/training_data/new \
      --bucket aurascribblr.firebasestorage.app \
      --prefix training_data/new/ \
      --checkpoint
else:
    print("Skipped Firebase download — add FIREBASE_SERVICE_ACCOUNT_JSON")

if DATA_INPUT.exists():
    dst = Path("data/raw")
    dst.mkdir(parents=True, exist_ok=True)
    src = DATA_INPUT / "raw" if (DATA_INPUT / "raw").exists() else DATA_INPUT
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print("Merged DATA_INPUT into data/raw")

!python src/prepare_raw.py --raw data/raw --output data/processed/all.jsonl --no-merge-existing

all_jsonl = Path("data/processed/all.jsonl")
n = sum(1 for line in open(all_jsonl, encoding="utf-8") if line.strip())
print(f"all.jsonl: {n} samples")
assert n > 0, "No samples — check Firebase secret or data/raw/"

CHECKPOINT_INPUT = REPO_INPUT / "handwriting-checkpoint"
ckpt = Path("models/checkpoint_best.pt")
if not ckpt.exists() and CHECKPOINT_INPUT.exists():
    for src in [CHECKPOINT_INPUT / "checkpoint_best.pt", CHECKPOINT_INPUT / "models/checkpoint_best.pt"]:
        if src.exists():
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, ckpt)
            print("Checkpoint from dataset:", src)
print("checkpoint:", "OK" if ckpt.exists() else "from scratch")
```

---

## תא 5 — קונפיג GPU

```python
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/train.yaml").read_text(encoding="utf-8"))
cfg["batch_size"] = 64
cfg["epochs"] = 20
cfg["learning_rate"] = 2e-5
cfg["num_workers"] = 2
Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print(cfg["epochs"], "epochs,", cfg["batch_size"], "batch")
```

---

## תא 6 — split

```python
!python src/split_manifest.py --source data/processed/all.jsonl \
  --train-out data/processed/train.jsonl --val-out data/processed/val.jsonl --val-ratio 0.1
```

---

## תא 7 — train

```python
from pathlib import Path
import os

extra = ""
if Path("data/corrections").exists() and any(Path("data/corrections").glob("*.json")):
    extra = "--corrections_dir data/corrections"
elif Path("data/raw/corrections").exists():
    extra = "--corrections_dir data/raw/corrections"

cmd = f"python src/train.py --config configs/train_kaggle.yaml {extra}"
print(">>>", cmd)
assert os.system(cmd) == 0, "Training failed"
```

---

## תא 8 — predict

```python
!python src/predict.py --config configs/train_kaggle.yaml \
  --checkpoint output/checkpoint_best.pt --manifest data/processed/val.jsonl \
  --output output/predictions.jsonl
```

---

## תא 9 — evaluate

```python
from pathlib import Path
import json

!python src/evaluate.py --manifest data/processed/val.jsonl \
  --predictions output/predictions.jsonl --report output/eval_report.json

print(json.dumps(json.loads(Path("output/eval_report.json").read_text()), indent=2))
```

---

## תא 10 — export ONNX

```python
!python src/export_onnx.py --config configs/train_kaggle.yaml \
  --checkpoint output/checkpoint_best.pt --trace-time 128 --trace-tokens 96
```

---

## תא 11 — ZIP

```python
from pathlib import Path
import json, zipfile, shutil

bundle = Path("/kaggle/working/handwriting_bundle")
bundle.mkdir(exist_ok=True)
for src, name in [
    ("output/model.onnx", "model.onnx"),
    ("output/checkpoint_best.pt", "checkpoint_best.pt"),
    ("output/eval_report.json", "eval_report.json"),
    ("output/training_log.jsonl", "training_log.jsonl"),
    ("configs/vocab.txt", "vocab.txt"),
]:
    p = Path(src)
    if p.exists():
        shutil.copy(p, bundle / name)
        print("OK", name)

zip_path = Path("/kaggle/working/handwriting_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in bundle.iterdir():
        z.write(f, f.name)
print("Download:", zip_path)
```

---

## תא 12 — גרף (אופציונלי)

```python
from pathlib import Path
import json, matplotlib.pyplot as plt

rows = [json.loads(l) for l in Path("output/training_log.jsonl").read_text().splitlines() if l.strip()]
e = [r["epoch"] for r in rows]
plt.plot(e, [r["train_loss"] for r in rows], label="loss")
plt.plot(e, [r.get("val_cer") for r in rows], label="val_cer")
plt.legend(); plt.xlabel("epoch"); plt.show()
```

---

## תא 13 — Firebase upload

```python
import os
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret("FIREBASE_SERVICE_ACCOUNT_JSON")
    print("Firebase secret OK")
except Exception as e:
    print("No secret:", e)

!python src/upload_firebase.py --local output/model.onnx --vocab configs/vocab.txt \
  --bucket aurascribblr.firebasestorage.app --remote models/latest_handwriting.onnx
```

---

## מבנה `data/raw/` הנתמך

```
data/raw/
├── *.jsonl              # דגימות מוכנות
├── corrections/*.json   # תיקוני אפליקציה
├── crohme/**/*.inkml    # מתמטיקה (CROHME)
├── iam/                 # אנגלית (IAM-OnDB)
└── math/                # חלופה ל-crohme
```
