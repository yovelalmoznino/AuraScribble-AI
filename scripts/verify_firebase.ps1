# Download Firebase ONNX + vocab and run validation (CER on val.jsonl).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path ".\.venv\Scripts\activate.ps1")) {
    Write-Host "Create venv first: python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt"
    exit 1
}
.\.venv\Scripts\activate
python src/verify_firebase_model.py @args
