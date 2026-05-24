Set-Location $PSScriptRoot\..

$ErrorActionPreference = "Stop"

Write-Host "=== 0/7 Prepare data/raw -> all.jsonl ==="
if (Test-Path "data/raw") {
    python "src/prepare_raw.py" --raw "data/raw" --output "data/processed/all.jsonl" --no-merge-existing
} else {
    Write-Host "No data/raw folder — using existing data/processed/all.jsonl"
}

Write-Host "=== 1/7 Split train/val ==="
python "src/split_manifest.py" --source "data/processed/all.jsonl" --train-out "data/processed/train.jsonl" --val-out "data/processed/val.jsonl"

Write-Host "=== 2/7 Train ==="
python "src/train.py" --config "configs/train.yaml"

Write-Host "=== 3/7 Predict on val ==="
python "src/predict.py" --config "configs/train.yaml" --checkpoint "output/checkpoint_best.pt" --manifest "data/processed/val.jsonl" --output "output/predictions.jsonl"

Write-Host "=== 4/7 Evaluate ==="
python "src/evaluate.py" --manifest "data/processed/val.jsonl" --predictions "output/predictions.jsonl" --report "output/eval_report.json"

Write-Host "=== 5/7 Export ONNX ==="
python "src/export_onnx.py" --config "configs/train.yaml" --checkpoint "output/checkpoint_best.pt"

Write-Host "=== 6/7 Upload to Firebase (optional) ==="
if ($env:GOOGLE_APPLICATION_CREDENTIALS -or $env:FIREBASE_SERVICE_ACCOUNT_JSON -or (Test-Path "configs/firebase_service_account.json")) {
    python "src/upload_firebase.py" --local "output/model.onnx" --vocab "configs/vocab.txt"
} else {
    Write-Host "Skipped Firebase upload — set GOOGLE_APPLICATION_CREDENTIALS or configs/firebase_service_account.json"
}

Write-Host "Done."
