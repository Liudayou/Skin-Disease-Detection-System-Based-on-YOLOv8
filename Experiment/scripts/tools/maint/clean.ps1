$deletedCount = 0
do {
    $emptyDirs = Get-ChildItem -Path "D:\SkinSystem\Experiment\runs" -Recurse -Directory | Where-Object { $_.GetFileSystemInfos().Count -eq 0 }
    if ($emptyDirs) {
        foreach ($dir in $emptyDirs) {
            Remove-Item -LiteralPath $dir.FullName -Force 
            $deletedCount++
        }
    }
} while ($emptyDirs)
Write-Host "Total empty directories removed: $deletedCount"
