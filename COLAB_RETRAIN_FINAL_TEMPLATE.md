# Colab Retrain Final Template

Use this template when opening a **new Colab notebook**.  
Goal: one stable loop that only needs more training data each run.

## Cell 1 - Setup

```python
import os
from pathlib import Path

ROOT = Path("/content/handwriting-model")
os.chdir(ROOT)
print("cwd:", Path.cwd())
!python -V
!pip -q install -r requirements.txt
```

## Cell 2 - Optional split refresh

```bash
!python -u src/split_data.py \
  --input data/processed/all.jsonl \
  --train-out data/processed/train.jsonl \
  --val-out data/processed/val.jsonl \
  --test-out data/processed/test.jsonl \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --balance-modes
```

## Cell 3 - One-command retrain + gate + export (+ optional upload)

```bash
!python -u src/retrain_release_pipeline.py \
  --work-dir . \
  --config configs/train.yaml \
  --checkpoint output/checkpoint_best.pt \
  --manifest data/processed/val.jsonl \
  --summary output/retrain_release_summary.json
```

To upload automatically when gate passes:

```bash
!python -u src/retrain_release_pipeline.py \
  --work-dir . \
  --config configs/train.yaml \
  --checkpoint output/checkpoint_best.pt \
  --manifest data/processed/val.jsonl \
  --summary output/retrain_release_summary.json \
  --upload
```

## Cell 4 - Verify summary

```python
import json
from pathlib import Path

s = json.loads(Path("output/retrain_release_summary.json").read_text(encoding="utf-8"))
print("status:", s["status"])
print("gate_ok:", s["gate_ok"], "| reason:", s["gate_reason"])
print("cer_mean:", s["metrics"]["cer_mean"])
print("wer_mean:", s["metrics"]["wer_mean"])
print("cer_by_mode:", s["metrics"]["cer_by_mode"])
print("math_norm_exact:", s["metrics"]["math_expression_normalized_exact_mean"])
print("uploaded:", s["artifacts"]["uploaded"])
```

## Cell 5 - Hard-check Firebase vocab count (raw)

```python
from pathlib import Path
from src.upload_firebase import _resolve_credentials, DEFAULT_BUCKET
from google.cloud import storage

creds = _resolve_credentials(None)
client = storage.Client(credentials=creds, project=creds.project_id)
out = Path("/tmp/latest_vocab.txt")
client.bucket(DEFAULT_BUCKET).blob("models/latest_vocab.txt").download_to_filename(str(out))
raw = out.read_text(encoding="utf-8").splitlines()
print("remote_raw_count:", len(raw))
print("has_space_token:", " " in raw)
```

