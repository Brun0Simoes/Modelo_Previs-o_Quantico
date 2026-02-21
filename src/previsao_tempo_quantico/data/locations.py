from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import requests

MUNICIPIOS_URL = "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/master/csv/municipios.csv"
ESTADOS_IBGE_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"


def ensure_location_catalog(reference_dir: Path, force_refresh: bool = False) -> tuple[Path, Path]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    municipios_path = reference_dir / "municipios.csv"
    estados_path = reference_dir / "estados.csv"

    if force_refresh or not municipios_path.exists():
        response = requests.get(MUNICIPIOS_URL, timeout=120)
        response.raise_for_status()
        municipios_path.write_text(response.text, encoding="utf-8")

    if force_refresh or not estados_path.exists():
        response = requests.get(ESTADOS_IBGE_URL, timeout=120)
        response.raise_for_status()
        estados = pd.DataFrame(response.json())[["id", "sigla", "nome"]]
        estados = estados.rename(columns={"id": "codigo_uf", "sigla": "uf", "nome": "estado_nome"})
        estados.to_csv(estados_path, index=False)

    return municipios_path, estados_path


def _load_catalog(reference_dir: Path) -> pd.DataFrame:
    municipios_path, estados_path = ensure_location_catalog(reference_dir)
    municipios = pd.read_csv(municipios_path)
    estados = pd.read_csv(estados_path)

    municipios = municipios.merge(estados, how="left", on="codigo_uf")
    municipios = municipios.rename(
        columns={
            "codigo_ibge": "region_id",
            "nome": "region_name",
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    municipios["scope"] = "municipio"
    municipios["region_id"] = municipios["region_id"].astype(str)
    return municipios


def _to_state_catalog(municipios: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        municipios.groupby(["uf", "estado_nome"], as_index=False)
        .agg(lat=("lat", "mean"), lon=("lon", "mean"), n_municipios=("region_id", "count"))
        .sort_values("uf")
    )
    grouped["region_id"] = grouped["uf"]
    grouped["region_name"] = grouped["estado_nome"]
    grouped["scope"] = "state"
    return grouped[["region_id", "region_name", "uf", "lat", "lon", "scope", "n_municipios"]]


def load_regions(
    reference_dir: Path,
    scope: Literal["state", "municipio"] = "state",
    uf: str | None = None,
    municipality_limit: int | None = None,
) -> pd.DataFrame:
    municipios = _load_catalog(reference_dir)

    if uf:
        municipios = municipios.loc[municipios["uf"].str.upper() == uf.upper()].copy()

    if scope == "state":
        return _to_state_catalog(municipios)

    if municipality_limit and municipality_limit > 0:
        municipios = municipios.sort_values(["capital", "region_name"], ascending=[False, True]).head(municipality_limit)

    return municipios[["region_id", "region_name", "uf", "lat", "lon", "scope", "capital"]].reset_index(drop=True)
