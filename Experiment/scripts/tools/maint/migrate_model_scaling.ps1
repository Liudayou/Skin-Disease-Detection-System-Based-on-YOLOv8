$sourcePath = "D:\SkinSystem\Experiment\runs\detect\model_scaling"
$targetBase = "D:\SkinSystem\Experiment\weights\model_scaling"

if (-not (Test-Path $targetBase)) {
    New-Item -ItemType Directory -Force -Path $targetBase | Out-Null
}

$bestWeights = Get-ChildItem -Path $sourcePath -Recurse -Filter "best.pt"

$movedCount = 0
foreach ($file in $bestWeights) {
    $runName = $file.Directory.Parent.Name
    $targetDir = Join-Path $targetBase $runName
    
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }
    
    $targetFile = Join-Path $targetDir "best.pt"
    Copy-Item -Path $file.FullName -Destination $targetFile -Force
    $movedCount++
}

Write-Host "Successfully migrated $movedCount best weights!"
