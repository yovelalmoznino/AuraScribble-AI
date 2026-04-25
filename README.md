# Handwriting Model Workspace

This workspace contains the offline training/export/evaluation pipeline for the
Android on-device handwriting recognizer (math + Hebrew + English).

## Layout

- `data/`
  - `raw/` source datasets
  - `processed/` normalized line-level samples
- `configs/`
  - `train.yaml` training hyperparameters
  - `vocab.txt` unified token vocabulary
- `src/`
  - `dataset.py` ingestion and normalization
  - `train.py` baseline sequence training entrypoint
  - `export_onnx.py` ONNX export and quantization
  - `evaluate.py` CER/WER and math-symbol metrics
- `scripts/`
  - `prepare_data.ps1`
  - `train.ps1`
  - `export_onnx.ps1`
  - `evaluate.ps1`

## Quick Start

1. Create python env and install dependencies:
   - `python -m venv .venv`
   - `.\\.venv\\Scripts\\activate`
   - `pip install -r requirements.txt`
2. Prepare data:
   - `pwsh ./scripts/prepare_data.ps1`
3. Train baseline:
   - `pwsh ./scripts/train.ps1`
4. Export ONNX:
   - `pwsh ./scripts/export_onnx.ps1`
5. Evaluate:
   - `pwsh ./scripts/evaluate.ps1`

## Real Dataset Ingestion

The training script can load real data from:

- `data.train_manifest` / `data.val_manifest` (JSONL format)
- `data.crohme_path` (CROHME InkML tree)
- `data.iam_online_path` (IAM-OnDB prepared folder)

### Expected sample schema (JSONL)

Each line:

```json
{"points": [[x, y, t], [x, y, t], ...], "text": "target transcript", "mode": "math|text|auto"}
```

### Notes

- CROHME loader parses `.inkml` traces and reads LaTeX from annotation fields (`truth`, `latex`, etc.).
- IAM loader supports:
  - direct `.jsonl` manifests (preferred)
  - fallback XML strokes + sidecar transcript files (`.txt`, `.lab`, `.label`) by same stem.

## Android Integration Output

Export artifacts should be copied to:

- `app/src/main/assets/models/handwriting/handwriting_v1.onnx`
- `app/src/main/assets/models/handwriting/vocab.txt`

