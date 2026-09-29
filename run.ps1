# AntiGravSim: Automated Pipeline
# 1. Train -> 2. Evaluate -> 3. Plot Altitude -> 4. Plot Acceleration -> 5. Render Video
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path $PSScriptRoot

$py = "$PSScriptRoot\.venv\Scripts\python.exe"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AntiGravSim: Neural Gravity Cancellation Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Train
Write-Host "`n[1/5] Training PPO Agent (60,000 steps)..." -ForegroundColor Yellow
& $py train.py --timesteps 60000 --n-envs 4
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed!" -ForegroundColor Red; exit 1 }

# 2. Evaluate
Write-Host "`n[2/5] Running Quantitative Benchmark..." -ForegroundColor Yellow
& $py evaluate.py --episodes 5 --save-data

# 3. Plot Altitude
Write-Host "`n[3/5] Generating Altitude & Trajectory Plot..." -ForegroundColor Yellow
& $py visuals/plot_altitude.py

# 4. Plot Acceleration
Write-Host "`n[4/5] Generating Force & Net Acceleration Plot..." -ForegroundColor Yellow
& $py visuals/plot_acceleration.py

# 5. Render Video / GIF
Write-Host "`n[5/5] Generating Telemetry Animation (GIF & MP4)..." -ForegroundColor Yellow
& $py visuals/create_animation.py

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Pipeline Complete! Results saved in ./outputs/" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
