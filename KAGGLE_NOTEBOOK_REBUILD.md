# AuraScribble — Kaggle Notebook v3.1 (Rebuild)

Copy each cell into a **new** Kaggle notebook. Order: **1 → 18**.

**Quick start (עברית):** see [`KAGGLE_V3_RUN.md`](KAGGLE_V3_RUN.md)

**v3.1 changes (after `the and` template collapse):**
- Curriculum **4 steps:** short → **medium** → full → **IAM-long only**
- Stricter **`decode_quality`** (flags `the and`, single `ה`, default `\frac{1}`)
- Phase 2b: **patience 12**, lr **2e-5**; Phase 2c: IAM-only, lr **1e-5**
- Training logs **`val template_collapse`** rate each epoch

**v3.2 data changes (real ink + balance):**
- English now uses **real IAM-OnDB online strokes** (jsonised), not fake bbox zigzags
- **Balanced, stratified split** (`--balance-modes`) replaces the manual ×2 oversample (old Cell 10 removed)
- Synthetic Hebrew is now **centerline (skeleton), RTL, multi-font** (was mirrored font outlines)
- **HHD** converts via skeletonization; Firebase **corrections** auto-ingested (Cells 4–5)
- Curriculum manifests are **deduped + single-line**; export gate is now **per-mode** (Cell 13)

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
# AuraScribble handwriting v4 (Hybrid CTC + AR)
- Architecture: Conv+Transformer encoder, CTC head + AR decoder head
- Curriculum: short lines → medium → full IAM/hebrew/math
- Early stopping on val_cer (with larger validation sample)
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

## Cell 3 — GPU packages

```python
# Torch is preinstalled on Kaggle GPU images. These cover data prep:
!pip -q install pillow numpy python-bidi datasets google-cloud-storage onnx onnxruntime 2>/dev/null
import PIL, numpy as np
print("pillow", PIL.__version__, "numpy", np.__version__)
```

`python-bidi` improves synthetic Hebrew RTL shaping (the generator falls back to a
built-in heuristic if it is missing). `datasets` is only needed for HHD parquet.

---

## Cell 4 — Firebase corrections (REAL user ink — optional but best for Hebrew)

```python
import os
# Needs the FIREBASE_SERVICE_ACCOUNT_JSON Kaggle secret.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = UserSecretsClient().get_secret(
        "FIREBASE_SERVICE_ACCOUNT_JSON"
    )
    rc = os.system(
        "python -u src/download_firebase_corrections.py "
        f"--local-dir {DATA_RAW / 'training_data' / 'new'} --include-processed"
    )
    print("corrections download rc:", rc)
except Exception as e:
    print("Skipping Firebase corrections (no secret / offline):", e)
```

`prepare_raw` auto-ingests every `*.json` under `data/raw/` via
`read_firebase_corrections`, so the downloaded files need no extra conversion.
These are the only **real** Hebrew sentences/words from the app — the highest-value
Hebrew data you have.

---

## Cell 5 — HHD (real Hebrew letters → centerline strokes)

```python
import os
# Put the HHD dataset under data/raw/hhd/ (parquet, image folders, or a .rar).
hhd_dir = DATA_RAW / "hhd"
if hhd_dir.exists() and any(hhd_dir.iterdir()):
    rc = os.system(
        "python -u src/convert_hhd_to_jsonl.py "
        f"--input {hhd_dir} --output {hhd_dir / 'hhd_strokes.jsonl'}"
    )
    print("HHD convert rc:", rc)
else:
    print("No HHD data at", hhd_dir, "- skipping (add it as a Kaggle dataset to use).")
```

Now uses **Zhang-Suen skeletonization** (centerline), not a raster scan — so HHD
letters look like pen strokes. Output JSONL is auto-ingested by `prepare_raw`.

---

## Cell 6 — Synthetic Hebrew (centerline, RTL, multi-font)

```python
import os
rc = os.system(
    "python -u src/generate_synthetic_hebrew.py "
    f"--output {DATA_RAW / 'synthetic_hebrew' / 'hebrew_synthetic.jsonl'} "
    "--sentences configs/hebrew_sentences.txt --variants 8"
)
assert rc == 0, f"synthetic hebrew failed exit={rc}"
```

Rewritten: PIL raster → skeleton centerline → RTL trace (was font outlines,
visually mirrored). 127 sentences × 8 variants. If it prints "No Hebrew-capable
TTF font found", add `--font /path/to/NotoSansHebrew-Regular.ttf`.

---

## Cell 7 — prepare_raw (merge everything → all.jsonl)

```python
import os
rc = os.system(
    "python -u src/prepare_raw.py "
    f"--raw {DATA_RAW} --output {OUTPUT / 'all.jsonl'} --no-merge-existing"
)
assert rc == 0, f"prepare_raw failed exit={rc}"
# Expect lines like: iam (~11242), crohme (~3900), hebrew jsonl + corrections.
```

---

## Cell 8 — build_vocab

```python
!python src/build_vocab.py \
    --manifest {OUTPUT / "all.jsonl"} \
    --output configs/vocab.txt
```

---

## Cell 9 — Split train / val (stratified + balanced per-mode)

```python
!python src/split_manifest.py \
    --source {OUTPUT / "all.jsonl"} \
    --train-out {OUTPUT / "train.jsonl"} \
    --val-out {OUTPUT / "val.jsonl"} \
    --val-ratio 0.1 --seed 1337 \
    --balance-modes --per-mode-target 6000 --max-oversample 8
```

`--balance-modes` downsamples the dominant mode (English) and oversamples minority
modes (Hebrew/math) with jitter, so English no longer drowns Hebrew. The split is
**stratified**, so every mode is represented in val (real `val_hebrew` / `val_math`).
Tune `--per-mode-target` (e.g. raise once you have more real Hebrew).

> The old "Cell 10 — Oversample ×2 + English boost" is **removed** — `--balance-modes`
> replaces it. Do not run a manual oversample on top of it.

---

## Cell 10b — Curriculum manifests (cleaned: dedup + single-line)

```python
!python src/build_curriculum_manifest.py \
    --train {OUTPUT / "train.jsonl"} \
    --short-out {OUTPUT / "train_short.jsonl"} \
    --medium-out {OUTPUT / "train_medium.jsonl"} \
    --iam-out {OUTPUT / "train_iam_long.jsonl"} \
    --max-chars-short 32 \
    --medium-min-chars 33 \
    --medium-max-chars 72 \
    --iam-min-chars 24 \
    --rewrite-train
```

`--rewrite-train` overwrites `train.jsonl` with the cleaned (deduped, single-line)
set so Phase 2b trains on clean data too.

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
cfg["model_type"] = "hybrid"         # new v4 architecture
cfg["epochs"] = 30
cfg["learning_rate"] = 2e-4          # linear warmup before cosine decay
cfg["warmup_frac"] = 0.06
cfg["lr_min_frac"] = 0.1             # cosine decays to 10% of peak, not 0 (keep learning late)
cfg["batch_size"] = 24
cfg["dropout"] = 0.1                 # lowered 0.2->0.1: model was UNDERfitting, not overfitting
cfg["weight_decay"] = 0.01
cfg["label_smoothing"] = 0.05        # lowered 0.1->0.05: 0.1 added a ~0.83 nat loss floor
cfg["decoder_layers"] = 4            # raised 2->4: decoder was the capacity bottleneck
cfg["ctc_loss_weight"] = 0.3         # hybrid loss (normalized): keep CTC as stabilizer, not dominant
cfg["ar_loss_weight"] = 0.7          # prioritize AR objective after CTC normalization
cfg["decode_strategy"] = "joint"     # evaluate/infer with AR+CTC candidate fusion
cfg["decode_ctc_weight"] = 0.55
cfg["decode_ar_weight"] = 0.45
cfg["val_max_samples"] = 1000
cfg["early_stop_patience"] = 15      # raised 6->15: greedy val_cer is noisy, don't stop early
cfg["early_stop_min_delta"] = 0.003  # lowered so small real gains don't trip early-stop
cfg["decode_repetition_penalty"] = 2.5

cfg_dst.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("model_type:", cfg["model_type"], "| epochs:", cfg["epochs"], "| early_stop:", cfg["early_stop_patience"])
print("val_max_samples:", cfg["val_max_samples"], "| ctc/ar loss:", cfg["ctc_loss_weight"], cfg["ar_loss_weight"])
print("decode:", cfg["decode_strategy"], "| decode ctc/ar:", cfg["decode_ctc_weight"], cfg["decode_ar_weight"])
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
cfg["learning_rate"] = float(p1.get("learning_rate", 2e-4))  # phase 1 lr (was inheriting 5e-5)
cfg["early_stop_patience"] = max(15, int(p1.get("early_stop_patience", 15)))
cfg["resume_from_checkpoint"] = False
Path("configs/train_kaggle.yaml").write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
print("Phase 1:", cfg["train_manifest"], "max_tgt_len:", cfg["max_tgt_len"], "epochs:", cfg["epochs"])

import os
rc = os.system("python -u src/train.py --config configs/train_kaggle.yaml")
assert rc == 0, f"Phase 1 failed exit={rc}"
```

---

## Cell 11b — (Optional) One-click autopilot training

If you prefer a fully automatic run (no manual stop/edit/restart loops), run this
cell instead of Cells 12/12b/12c/12d. It will:
- run phase 1 -> 2a -> 2b -> 2c in sequence,
- automatically retry phase 2a once with safer fallback settings if needed,
- tune decode weights automatically from final eval per-mode CER,
- optionally export/upload to Firebase only if quality gate passes,
- write a summary report to `output/autopilot_report.json`.

```python
import os

# Assumes data prep + split + curriculum files already exist from previous cells.
cmd = (
    "python -u src/autopilot_train.py "
    "--config configs/train_kaggle.yaml "
    "--report output/autopilot_report.json "
    "--export"
)
rc = os.system(cmd)
assert rc == 0, f"Autopilot failed exit={rc}"
```

After completion, open:
- `output/autopilot_report.json`
- `output/training_log.jsonl`
- `output/eval_report_autopilot.json`

If you want upload to Firebase at the end (only when gate passes), run:

```python
import os
cmd = (
    "python -u src/autopilot_train.py "
    "--config configs/train_kaggle.yaml "
    "--report output/autopilot_report.json "
    "--export --upload"
)
rc = os.system(cmd)
assert rc == 0, f"Autopilot+upload failed exit={rc}"
```

> For Kaggle/Colab: provide credentials via `FIREBASE_SERVICE_ACCOUNT_JSON` secret/env
> or `--firebase-credentials /path/to/service-account.json`.

---

## Cell 12b — Train phase 2a (medium lines, resume)

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/train_kaggle.yaml").read_text(encoding="utf-8"))
p2a = cfg.get("curriculum_phase2a") or {}
cfg["train_manifest"] = str(OUTPUT / "train_medium.jsonl")
cfg["max_tgt_len"] = int(p2a.get("max_tgt_len", 96))
cfg["epochs"] = max(24, int(p2a.get("epochs", 24)))
cfg["learning_rate"] = float(p2a.get("learning_rate", 3e-5))
cfg["early_stop_patience"] = max(20, int(p2a.get("early_stop_patience", 20)))
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
cfg["early_stop_patience"] = max(15, int(p2b.get("early_stop_patience", 15)))
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
cfg["early_stop_patience"] = max(15, int(p2c.get("early_stop_patience", 15)))
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

# Evaluate a stratified sample so every mode gets a fair per-mode check.
from collections import defaultdict
by_mode_samples = defaultdict(list)
for s in val:
    by_mode_samples[(s.mode or "auto").lower()].append(s)

picked = []
for m, items in by_mode_samples.items():
    random.shuffle(items)
    picked.extend(items[: min(20, len(items))])

ok, bad = 0, 0
mode_cer = defaultdict(list)          # mode -> [cer, ...]
mode_collapse = defaultdict(lambda: [0, 0])  # mode -> [collapse_count, total]
for s in picked:
    m = (s.mode or "auto").lower()
    pred = greedy_decode(
        model, tok, s.points, device,
        max_seq_len=cfg["max_seq_len"],
        mode=s.mode,
        repetition_penalty=rep_pen,
    )
    c = cer(pred, s.text)
    collapsed = is_template_collapse(pred, s.text, s.mode)
    mode_cer[m].append(c)
    mode_collapse[m][1] += 1
    if collapsed:
        mode_collapse[m][0] += 1
        bad += 1
    else:
        ok += 1
    print(f"[{'COLLAPSE' if collapsed else 'OK'}] [{m}] CER={c:.2f}  truth={s.text[:50]!r}  pred={pred[:50]!r}")

per_mode_val_cer = {m: (sum(v) / len(v)) for m, v in mode_cer.items() if v}
per_mode_collapse = {m: (cc, tt) for m, (cc, tt) in mode_collapse.items()}
print("\nPer-mode val_cer:", {m: round(v, 3) for m, v in per_mode_val_cer.items()})
print("Per-mode collapse:", {m: f"{cc}/{tt}" for m, (cc, tt) in per_mode_collapse.items()})

val_cer = float(ckpt.get("val_cer", 99))
print(f"Summary: OK={ok} COLLAPSE={bad} | epoch={ckpt.get('epoch')} val_cer={val_cer}")
ready, reason = passes_export_gate(
    cer_mean=val_cer,
    val_cer=val_cer,
    collapse_count=bad,
    sample_count=ok + bad,
    per_mode_val_cer=per_mode_val_cer,
    per_mode_collapse=per_mode_collapse,
    require_modes=("english", "hebrew", "math"),
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
