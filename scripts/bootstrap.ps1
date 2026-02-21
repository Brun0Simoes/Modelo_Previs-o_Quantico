param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$preferredDrive = "E"

$drive = ([System.IO.Path]::GetPathRoot($projectRoot)).TrimEnd(':','\')
if ($drive -ne $preferredDrive) {
  throw "Projeto deve ficar no drive $preferredDrive`: encontrado em $drive`:"
}

$folders = @(
  "data\\raw_merra",
  "data\\prithvi_input\\merra2",
  "data\\prithvi_input\\climatology",
  "data\\features",
  "data\\models",
  "data\\outputs",
  "data\\reference",
  "cache\\huggingface",
  "logs"
)

foreach ($folder in $folders) {
  New-Item -ItemType Directory -Path (Join-Path $projectRoot $folder) -Force | Out-Null
}

$envPath = Join-Path $projectRoot ".env"
$examplePath = Join-Path $projectRoot ".env.example"
if (-not (Test-Path $envPath)) {
  Copy-Item $examplePath $envPath
}

Write-Host "Projeto inicializado em: $projectRoot"
Write-Host "Pastas de dados/cache criadas no drive E:."
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Root,Used,Free | Format-Table -AutoSize
