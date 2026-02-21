from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from previsao_tempo_quantico.models.prithvi_runner import PrithviForecastEngine


def _align_lat_lon(lats: np.ndarray, lons: np.ndarray, lat_size: int, lon_size: int) -> tuple[np.ndarray, np.ndarray]:
    trim_lats = lats[:lat_size]
    trim_lons = lons[:lon_size]
    return trim_lats, trim_lons


def _nearest_indexes(regions: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    region_lats = regions["lat"].to_numpy(dtype=np.float64)
    region_lons = regions["lon"].to_numpy(dtype=np.float64)

    if lons.min() >= 0:
        region_lons = np.mod(region_lons, 360.0)

    lat_idx = np.abs(lats[:, None] - region_lats[None, :]).argmin(axis=0)
    lon_idx = np.abs(lons[:, None] - region_lons[None, :]).argmin(axis=0)
    return lat_idx, lon_idx


def build_training_frame(
    engine: PrithviForecastEngine,
    regions: pd.DataFrame,
    variables: Iterable[str],
    start: str,
    end: str,
    input_time_hours: int,
    lead_time_hours: int,
    max_samples: int | None = None,
) -> pd.DataFrame:
    variables = [var.upper() for var in variables]
    missing = [var for var in variables if var not in engine.surface_indexes]
    if missing:
        raise ValueError(f"Variaveis nao suportadas pelo modelo: {missing}")

    frames: list[pd.DataFrame] = []
    lat_idx_cache: np.ndarray | None = None
    lon_idx_cache: np.ndarray | None = None

    for sample in engine.iter_rollouts(
        start=start,
        end=end,
        input_time_hours=input_time_hours,
        lead_time_hours=lead_time_hours,
        max_samples=max_samples,
    ):
        pred = sample.prediction[0].numpy()
        target = sample.target[0].numpy()

        lat_size = pred.shape[-2]
        lon_size = pred.shape[-1]

        lats, lons = _align_lat_lon(sample.lats, sample.lons, lat_size, lon_size)

        if lat_idx_cache is None or lon_idx_cache is None:
            lat_idx_cache, lon_idx_cache = _nearest_indexes(regions, lats, lons)

        base = regions[["region_id", "region_name", "uf", "lat", "lon", "scope"]].copy()
        ts = pd.Timestamp(sample.timestamp)
        base["timestamp"] = ts
        base["sin_hod"] = np.sin(2.0 * np.pi * ts.hour / 24.0)
        base["cos_hod"] = np.cos(2.0 * np.pi * ts.hour / 24.0)
        base["sin_doy"] = np.sin(2.0 * np.pi * ts.dayofyear / 366.0)
        base["cos_doy"] = np.cos(2.0 * np.pi * ts.dayofyear / 366.0)

        for variable in variables:
            channel = engine.surface_indexes[variable]
            pred_field = pred[channel, :lat_size, :lon_size]
            target_field = target[channel, :lat_size, :lon_size]
            base[f"pred_{variable}"] = pred_field[lat_idx_cache, lon_idx_cache]
            base[f"target_{variable}"] = target_field[lat_idx_cache, lon_idx_cache]

        frames.append(base)

    if not frames:
        raise RuntimeError("Nenhuma amostra foi gerada. Verifique os dados de entrada.")

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data
