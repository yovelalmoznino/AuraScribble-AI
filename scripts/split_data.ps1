Set-Location $PSScriptRoot\..
python "src/split_manifest.py" --source "data/processed/all.jsonl" --train-out "data/processed/train.jsonl" --val-out "data/processed/val.jsonl" --val-ratio 0.1 --seed 1337
