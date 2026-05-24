Set-Location $PSScriptRoot\..

$local = "output/model.onnx"
if (-not (Test-Path $local)) {
    Write-Error "Missing $local — run export_onnx.ps1 first"
}

python src/upload_firebase.py --local $local --vocab configs/vocab.txt
