param(
  [string]$Start = "2020-01-01T00:00:00",
  [string]$End = "2026-02-17T18:00:00",
  [int]$StepHours = 6,
  [int]$InputTimeHours = 6,
  [int]$LeadTimeHours = 6,
  [int]$MunicipalityLimit = 900,
  [int]$StateTuningTrials = 8,
  [int]$MunicipioTuningTrials = 6,
  [double]$MaxMissingRatio = 1.0,
  [switch]$ResumeTraining
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
  if ($ResumeTraining) {
    Write-Host "[1/4] Resume ativo: pulando download MERRA-2."
  }
  else {
    Write-Host "[1/4] Download MERRA-2 (HF publico)..."
    docker compose run --rm trainer python scripts/download_merra2.py `
      --source hf-public `
      --start $Start `
      --end $End `
      --step-hours $StepHours `
      --input-time-hours $InputTimeHours `
      --lead-time-hours $LeadTimeHours `
      --allow-missing `
      --max-missing-ratio $MaxMissingRatio
  }

  Write-Host "[2/4] Treino state (modo adaptativo + fine tuning)..."
  if ($ResumeTraining) {
    powershell -ExecutionPolicy Bypass -File .\scripts\train_auto.ps1 -Scope state -Mode auto -Device cpu -FineTune -TuningTrials $StateTuningTrials -Resume -ReuseFeatures
  }
  else {
    powershell -ExecutionPolicy Bypass -File .\scripts\train_auto.ps1 -Scope state -Mode auto -Device cpu -FineTune -TuningTrials $StateTuningTrials
  }

  Write-Host "[3/4] Treino municipio (modo adaptativo + fine tuning)..."
  if ($ResumeTraining) {
    powershell -ExecutionPolicy Bypass -File .\scripts\train_auto.ps1 -Scope municipio -Mode auto -Device cpu -MunicipalityLimit $MunicipalityLimit -FineTune -TuningTrials $MunicipioTuningTrials -Resume -ReuseFeatures
  }
  else {
    powershell -ExecutionPolicy Bypass -File .\scripts\train_auto.ps1 -Scope municipio -Mode auto -Device cpu -MunicipalityLimit $MunicipalityLimit -FineTune -TuningTrials $MunicipioTuningTrials
  }

  Write-Host "[4/4] Verificacao API..."
  $stateCount = (Invoke-RestMethod -Uri "http://localhost:8000/api/forecast/latest?scope=state&variable=T2M" -TimeoutSec 20).count
  $muniCount = (Invoke-RestMethod -Uri "http://localhost:8000/api/forecast/latest?scope=municipio&variable=T2M" -TimeoutSec 20).count
  Write-Host "OK -> state: $stateCount | municipio: $muniCount"
}
finally {
  Pop-Location
}
