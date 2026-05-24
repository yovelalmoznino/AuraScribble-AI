Set-Location $PSScriptRoot\..

$assets = "..\..\app\src\main\assets\models\handwriting"
New-Item -ItemType Directory -Force -Path $assets | Out-Null

Copy-Item -Force "output\model.onnx" "$assets\handwriting_v1.onnx"
Copy-Item -Force "configs\vocab.txt" "$assets\vocab.txt"

Write-Host "Copied model.onnx -> handwriting_v1.onnx"
Write-Host "Copied vocab.txt -> assets"
