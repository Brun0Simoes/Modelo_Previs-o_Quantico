from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def _decode_merra_time(values: np.ndarray, day: pd.Timestamp) -> pd.DatetimeIndex:
    if np.issubdtype(values.dtype, np.integer):
        # Alguns arquivos trazem minutos desde o inicio do proprio dia.
        base = np.datetime64(pd.Timestamp(day).normalize().to_datetime64())
        ts = base + values.astype("timedelta64[m]")
        return pd.to_datetime(ts, utc=True)
    return pd.to_datetime(values, utc=True, errors="coerce")


def _nearest_indexes(regions: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    region_lats = regions["lat"].to_numpy(dtype=np.float64)
    region_lons = regions["lon"].to_numpy(dtype=np.float64)

    if lons.min() >= 0:
        region_lons = np.mod(region_lons, 360.0)

    lat_idx = np.abs(lats[:, None] - region_lats[None, :]).argmin(axis=0)
    lon_idx = np.abs(lons[:, None] - region_lons[None, :]).argmin(axis=0)
    return lat_idx, lon_idx


def build_training_frame_persistence(
    merra_input_dir: Path,
    regions: pd.DataFrame,
    variables: list[str],
    start: str,
    end: str,
    step_hours: int,
    lead_time_hours: int,
    max_samples: int | None = None,
) -> pd.DataFrame:
    variables = [var.upper() for var in variables]
    init_times = pd.date_range(start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC"), freq=f"{step_hours}h")

    @lru_cache(maxsize=64)
    def _load_day(day: pd.Timestamp):
        file = Path(merra_input_dir) / f"MERRA2_sfc_{day.strftime('%Y%m%d')}.nc"
        if not file.exists():
            return None, None
        try:
            ds = xr.load_dataset(file, engine="h5netcdf")
            times = _decode_merra_time(ds["time"].values, day=day)
            return ds, times
        except Exception:
            return None, None

    # Usa primeira data disponivel para definir grade.
    ds0 = None
    for ts in init_times:
        ds_candidate, _ = _load_day(ts)
        if ds_candidate is not None:
            ds0 = ds_candidate
            break
    if ds0 is None:
        raise RuntimeError("Nenhum arquivo MERRA-2 disponivel para montar o dataset no intervalo informado.")
    lats = ds0["lat"].values
    lons = ds0["lon"].values
    lat_idx, lon_idx = _nearest_indexes(regions, lats, lons)

    frames: list[pd.DataFrame] = []
    skipped_missing_file = 0
    skipped_missing_time = 0
    used_samples = 0

    total_steps = len(init_times) if max_samples is None else min(max_samples, len(init_times))
    iter_times = init_times[:total_steps]

    for init_time in tqdm(
        iter_times,
        total=total_steps,
        desc="Extraindo features persistencia",
        unit="amostra",
    ):

        target_time = init_time + pd.Timedelta(hours=lead_time_hours)

        ds_init, times_init = _load_day(init_time)
        ds_target, times_target = _load_day(target_time)
        if ds_init is None or ds_target is None or times_init is None or times_target is None:
            skipped_missing_file += 1
            continue

        init_match = np.where(times_init == init_time)[0]
        target_match = np.where(times_target == target_time)[0]

        if len(init_match) == 0 or len(target_match) == 0:
            skipped_missing_time += 1
            continue

        init_idx = int(init_match[0])
        target_idx = int(target_match[0])

        base = regions[["region_id", "region_name", "uf", "lat", "lon", "scope"]].copy()
        ts = pd.Timestamp(init_time)
        base["timestamp"] = ts
        base["sin_hod"] = np.sin(2.0 * np.pi * ts.hour / 24.0)
        base["cos_hod"] = np.cos(2.0 * np.pi * ts.hour / 24.0)
        base["sin_doy"] = np.sin(2.0 * np.pi * ts.dayofyear / 366.0)
        base["cos_doy"] = np.cos(2.0 * np.pi * ts.dayofyear / 366.0)

        for var in variables:
            if var in ds_init.variables and var in ds_target.variables:
                pred_field = ds_init[var].isel(time=init_idx).values
                target_field = ds_target[var].isel(time=target_idx).values
                base[f"pred_{var}"] = pred_field[lat_idx, lon_idx]
                base[f"target_{var}"] = target_field[lat_idx, lon_idx]
            else:
                base[f"pred_{var}"] = np.nan
                base[f"target_{var}"] = np.nan

        frames.append(base)
        used_samples += 1

    if not frames:
        raise RuntimeError("Nao foi possivel montar dataset de treino no modo persistence.")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    print(
        "[persistence] amostras usadas: "
        f"{used_samples}/{total_steps} | "
        f"puladas por falta de arquivo: {skipped_missing_file} | "
        f"puladas por timestamp ausente: {skipped_missing_time}"
    )
    return out
