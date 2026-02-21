param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
  if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
  }

  docker info | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine nao esta acessivel. Abra o Docker Desktop e tente novamente."
  }

  docker compose build api
  if ($LASTEXITCODE -ne 0) {
    throw "Falha no build do container api."
  }

  docker compose run --rm api python -m pip check
  if ($LASTEXITCODE -ne 0) {
    throw "pip check encontrou conflitos."
  }

  docker compose run --rm api python -c "import torch, pennylane, fastapi; print('torch', torch.__version__); print('pennylane', pennylane.__version__); print('fastapi', fastapi.__version__)"
  if ($LASTEXITCODE -ne 0) {
    throw "Falha no import das bibliotecas principais."
  }

  Write-Host "Dependencias validadas no container."
}
finally {
  Pop-Location
}
