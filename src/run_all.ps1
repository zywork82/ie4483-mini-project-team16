<#
Runs the Person 2 experiments in order and saves ALL console output to a log file.
Put this file in the repo root. Run it from the repo root with the venv active.

  powershell -ExecutionPolicy Bypass -File run_all.ps1 -Tier 1                  # 9 runs
  powershell -ExecutionPolicy Bypass -File run_all.ps1 -Tier 2                  # 12 runs
  powershell -ExecutionPolicy Bypass -File run_all.ps1 -Tier 3                  # 9 runs
  powershell -ExecutionPolicy Bypass -File run_all.ps1 -Tier 4 -Models resnet18,resnet50   # 6 runs, FULL data

  Add -DryRun to only print the commands without running them.
  Add -Models resnet18   to run just one model.

Tier 1: baseline, 3 seeds per model         (lr 0.001, batch 32)
Tier 2: learning rate + batch size          (seed 42)
Tier 3: augmentation + fine-tuning          (seed 42)
Tier 4: full data (20,000 train images), seeds 42/43/44, no augmentation
#>
param(
    [int]$Tier = 1,
    [string[]]$Models = @("resnet18", "resnet50", "vgg16"),
    [switch]$DryRun
)

New-Item -ItemType Directory -Force -Path results | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$logfile = "results\run_all_tier${Tier}_$stamp.txt"
Start-Transcript -Path $logfile | Out-Null

Write-Host "Log file: $logfile"
Write-Host ("Started: " + (Get-Date))
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
Write-Host "If cuda available is False, runs will be slow (CPU). Press Ctrl+C to stop."

$runs = New-Object System.Collections.ArrayList
function Add-Run($label, $m, $flags) {
    [void]$runs.Add([pscustomobject]@{ Id = "T$Tier-$m-$label"; Model = $m; Args = $flags })
}

$base = "--subset 2000 --epochs 5 --batch-size 32 --who P2"

foreach ($m in $Models) {
    switch ($Tier) {
        1 {
            foreach ($s in 42, 1, 2) { Add-Run "base-s$s" $m "$base --lr 0.001 --seed $s" }
        }
        2 {
            Add-Run "lr0.01"   $m "$base --lr 0.01 --seed 42"
            Add-Run "lr0.0001" $m "$base --lr 0.0001 --seed 42"
            Add-Run "bs16"     $m "--subset 2000 --epochs 5 --batch-size 16 --who P2 --lr 0.001 --seed 42"
            Add-Run "bs64"     $m "--subset 2000 --epochs 5 --batch-size 64 --who P2 --lr 0.001 --seed 42"
        }
        3 {
            Add-Run "augment"        $m "$base --lr 0.001 --augment --seed 42"
            Add-Run "unfreeze"       $m "$base --lr 0.0001 --unfreeze --seed 42"
            Add-Run "unfreeze-aug"   $m "$base --lr 0.0001 --unfreeze --augment --seed 42"
        }
        4 {
            foreach ($s in 42, 43, 44) {
                Add-Run "full-s$s" $m "--epochs 5 --batch-size 32 --who P2 --lr 0.001 --seed $s"
            }
        }
        default { Write-Host "Tier must be 1, 2, 3 or 4"; Stop-Transcript | Out-Null; exit 1 }
    }
}

$failed = @()
$i = 0
foreach ($r in $runs) {
    $i++
    $cmd = "python src/train.py --model $($r.Model) $($r.Args)"
    Write-Host ""
    Write-Host "=== [$i/$($runs.Count)] $($r.Id) ===" -ForegroundColor Cyan
    Write-Host $cmd
    if ($DryRun) { continue }

    $t0 = Get-Date
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        $failed += $r.Id
        Write-Host "FAILED: $($r.Id)" -ForegroundColor Red
    }
    $el = (Get-Date) - $t0
    Write-Host ("Wall time for this run: " + $el.ToString("hh\:mm\:ss"))
}

Write-Host ""
Write-Host "=== Finished $($runs.Count) runs ===" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host ("FAILED runs: " + ($failed -join ", ")) -ForegroundColor Red
    Write-Host "Common fix for vgg16 out-of-memory: rerun that run by hand with --batch-size 16"
}
Write-Host "Results: python src/summarize_results.py   |   timing: results\timing_log.csv"
Write-Host "Full console log saved to: $logfile"
Stop-Transcript | Out-Null