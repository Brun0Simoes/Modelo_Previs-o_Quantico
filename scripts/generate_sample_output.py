from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from previsao_tempo_quantico.data import ensure_location_catalog, load_regions, save_forecast_snapshot
from previsao_tempo_quantico.settings import get_settings


VARIABLES = ["T2M", "PRECTOT", "U10M", "V10M"]


def _fake_signal(lat: np.ndarray, lon: np.ndarray, variable: str) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    if variable == "T2M":
        return 293.0 + 9.0 * np.sin(lat_r) + 2.0 * np.cos(lon_r)
    if variable == "PRECTOT":
        return np.abs(0.0002 + 0.0008 * np.sin(lat_r * 2.0) * np.cos(lon_r * 1.5))
    if variable == "U10M":
        return 1.5 + 6.0 * np.cos(lat_r) * np.sin(lon_r)
    if variable == "V10M":
        return -1.5 + 6.0 * np.sin(lat_r) * np.cos(lon_r)
    return np.zeros_like(lat)


def main() -> None:
    settings = get_settings()
    ensure_location_catalog(settings.reference_dir)

    source_time = datetime.now(tz=timezone.utc)

    for scope in ("state", "municipio"):
        regions = load_regions(
            reference_dir=settings.reference_dir,
            scope=scope,
            municipality_limit=1200 if scope == "municipio" else None,
        )

        lat = regions["lat"].to_numpy(dtype=float)
        lon = regions["lon"].to_numpy(dtype=float)

        for variable in VARIABLES:
            payload = regions[["region_id", "region_name", "uf", "lat", "lon"]].copy()
            payload["value"] = _fake_signal(lat, lon, variable)
            save_forecast_snapshot(
                output_dir=settings.output_dir,
                scope=scope,
                variable=variable,
                forecast=payload,
                model_name="demo-sintetico",
                source_timestamp=source_time,
            )

    print("Arquivos de exemplo gerados em data/outputs")


if __name__ == "__main__":
    main()
