Set-Location $PSScriptRoot\..
python "src/prepare_raw.py" --raw "data/raw" --output "data/processed/all.jsonl"
