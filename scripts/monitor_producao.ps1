param(
  [string]$Start = "2020-01-01T00:00:00",
  [string]$End = "2026-02-17T18:00:00",
  [int]$InputTimeHours = 6,
  [int]$LeadTimeHours = 6,
  [int]$ExpectedMerraFiles = 0,
  [switch]$FollowLogs
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
  $trainerName = docker ps --format "{{.Names}}" | Select-String "trainer-run" | Select-Object -First 1
  $apiName = docker ps --format "{{.Names}}" | Select-String "api-1" | Select-Object -First 1

  $expected = $ExpectedMerraFiles
  if ($expected -le 0) {
    $startTs = [datetime]::Parse($Start).AddHours(-1 * $InputTimeHours)
    $endTs = [datetime]::Parse($End).AddHours($LeadTimeHours)
    $days = [int]([math]::Floor(($endTs.Date - $startTs.Date).TotalDays) + 1)
    if ($days -lt 1) { $days = 1 }
    $expected = $days * 2
  }

  Write-Host "=== Containers ==="
  docker ps --format "table {{.Names}}`t{{.Status}}`t{{.RunningFor}}"

  $root = (Get-ChildItem .\data\prithvi_input\merra2\*.nc -ErrorAction SilentlyContinue | Measure-Object).Count
  $nested = (Get-ChildItem .\data\prithvi_input\merra2\merra-2\*.nc -ErrorAction SilentlyContinue | Measure-Object).Count
  $total = $root + $nested
  $pct = if ($expected -gt 0) { [math]::Round(($total / $expected) * 100.0, 1) } else { 0.0 }

  Write-Host ""
  Write-Host "=== Download MERRA-2 ==="
  Write-Host "Intervalo alvo: $Start -> $End"
  Write-Host "Arquivos raiz: $root"
  Write-Host "Arquivos subpasta merra-2: $nested"
  Write-Host "Total detectado: $total / $expected ($pct`%)"

  Write-Host ""
  Write-Host "=== Checkpoints ==="
  Get-ChildItem .\data\models\*.pt -ErrorAction SilentlyContinue |
    Select-Object Name, LastWriteTime, Length |
    Format-Table -AutoSize

  if ($FollowLogs -and $trainerName) {
    Write-Host ""
    Write-Host "=== Logs (follow) ==="
    docker logs -f $trainerName
  }
}
finally {
  Pop-Location
}
