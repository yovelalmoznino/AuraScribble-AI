Set-Location $PSScriptRoot\..
python "src/predict.py" --config "configs/train.yaml" --checkpoint "output/checkpoint_best.pt" --manifest "data/processed/val.jsonl" --output "output/predictions.jsonl"
