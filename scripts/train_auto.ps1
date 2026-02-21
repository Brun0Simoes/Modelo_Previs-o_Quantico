param(
  [ValidateSet("state", "municipio")]
  [string]$Scope = "state",
  [ValidateSet("auto", "persistence", "prithvi")]
  [string]$Mode = "auto",
  [ValidateSet("cpu", "cuda")]
  [string]$Device = "cpu",
  [string]$Uf = "",
  [int]$MunicipalityLimit = 900,
  [int]$MaxSamples = 0,
  [switch]$FineTune,
  [int]$TuningTrials = 0,
  [switch]$Resume,
  [switch]$ReuseFeatures,
  [switch]$ForcePrithvi
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
  $args = @(
    "compose", "run", "--rm",
    "-e", "DEVICE=$Device",
    "trainer",
    "python", "scripts/train_quantum.py",
    "--config", "configs/train.yaml",
    "--scope", $Scope,
    "--mode", $Mode
  )

  if ($Scope -eq "municipio") {
    if ($Uf) {
      $args += @("--uf", $Uf)
    }
    if ($MunicipalityLimit -gt 0) {
      $args += @("--municipality-limit", "$MunicipalityLimit")
    }
  }

  if ($MaxSamples -gt 0) {
    $args += @("--max-samples", "$MaxSamples")
  }

  if ($ForcePrithvi) {
    $args += @("--force-prithvi")
  }
  
  if ($FineTune) {
    $args += @("--fine-tune")
  }
  if ($TuningTrials -gt 0) {
    $args += @("--tuning-trials", "$TuningTrials")
  }
  if ($Resume) {
    $args += @("--resume")
  }
  if ($ReuseFeatures) {
    $args += @("--reuse-features")
  }

  Write-Host "Iniciando treino adaptativo..."
  Write-Host ("Comando: docker " + ($args -join " "))
  & docker @args
}
finally {
  Pop-Location
}
