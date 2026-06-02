Set-Location $PSScriptRoot\..

$local = "output/model.onnx"
if (-not (Test-Path $local)) {
    Write-Error "Missing $local — run export_onnx.ps1 first"
}

$vocab = if (Test-Path "output/vocab.from_checkpoint.txt") { "output/vocab.from_checkpoint.txt" } else { "configs/vocab.txt" }
python src/upload_firebase.py --local $local --vocab $vocab
