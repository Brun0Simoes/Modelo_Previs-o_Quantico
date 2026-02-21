# Changelog

## [1.0.0] - 2026-02-21

### Added
- Pipeline completo de previsao meteo para Brasil com base Prithvi-WxC e correcao QML.
- Treino adaptativo por memoria (`auto`, `prithvi`, `persistence`).
- Fine tuning com busca de hiperparametros, validacao temporal, early stopping e resume.
- Dashboards operacionais:
  - painel principal no estilo DSAT
  - previsao 24h por regiao e cidade
  - snap por cidade e data (00:00-23:00)
  - comparativo previsto x real hora a hora
- Endpoints FastAPI para status de modelos e consumo operacional.
- Documentacao tecnico-cientifica completa no `README.md` com galeria de dashboards.

### Changed
- `README.md` reorganizado para publicacao cientifica e producao.
- Escopo ajustado removendo referencia explicita ao caminho local do drive.

### Notes
- Dados pesados e checkpoints grandes permanecem fora do repositorio via `.gitignore`.
- Reproducao e operacao suportadas por scripts em `scripts/`.
