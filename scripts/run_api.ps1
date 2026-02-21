param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
  docker compose up api
}
finally {
  Pop-Location
}
