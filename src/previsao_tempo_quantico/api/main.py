from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr

from previsao_tempo_quantico.data import (
    ensure_location_catalog,
    list_available_forecasts,
    load_latest_forecast,
    load_regions,
)
from previsao_tempo_quantico.models import QuantumResidualRegressor
from previsao_tempo_quantico.settings import get_settings

settings = get_settings()
ensure_location_catalog(settings.reference_dir)

app = FastAPI(
    title="Previsao Tempo Brasil - Prithvi + QML",
    version="0.2.0",
    description="API de previsao com base Prithvi-WxC e correcao quantica por regiao.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ui_dir = Path(__file__).resolve().parents[1] / "ui"
app.mount("/static", StaticFiles(directory=ui_dir), name="static")

BRAZIL_STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
)

UF_TO_MACROREGION: dict[str, str] = {
    "AC": "Norte",
    "AL": "Nordeste",
    "AP": "Norte",
    "AM": "Norte",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "DF": "Centro-Oeste",
    "ES": "Sudeste",
    "GO": "Centro-Oeste",
    "MA": "Nordeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "MG": "Sudeste",
    "PA": "Norte",
    "PB": "Nordeste",
    "PR": "Sul",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RJ": "Sudeste",
    "RN": "Nordeste",
    "RS": "Sul",
    "RO": "Norte",
    "RR": "Norte",
    "SC": "Sul",
    "SP": "Sudeste",
    "SE": "Nordeste",
    "TO": "Norte",
}


def _macroregion_for_uf(uf: str) -> str:
    return UF_TO_MACROREGION.get(str(uf).upper(), "Desconhecida")


def _safe_float(value: object, ndigits: int = 3) -> float | None:
    try:
        casted = float(value)
    except Exception:
        return None
    if not np.isfinite(casted):
        return None
    return round(casted, ndigits)


def _to_temp_c(temp_raw: object) -> float | None:
    value = _safe_float(temp_raw, ndigits=4)
    if value is None:
        return None
    if value > 130.0:
        return value - 273.15
    return value


def _to_rain_mm_h(rain_raw: object) -> float | None:
    value = _safe_float(rain_raw, ndigits=8)
    if value is None:
        return None
    rain_mmh = max(0.0, value * 3600.0)
    return round(rain_mmh, 3)


def _climate_label(temp_c: float | None, rain_mm_h: float | None) -> str:
    if rain_mm_h is not None:
        if rain_mm_h >= 8.0:
            return "chuva forte"
        if rain_mm_h >= 2.0:
            return "chuva moderada"
        if rain_mm_h >= 0.2:
            return "chuva fraca"
    if temp_c is None:
        return "indefinido"
    if temp_c >= 32.0:
        return "muito quente"
    if temp_c <= 18.0:
        return "frio"
    return "tempo estavel"


def _rain_probability_pct(rain_mm_h: float | None) -> float | None:
    if rain_mm_h is None:
        return None
    rain = max(0.0, float(rain_mm_h))
    prob = 100.0 * (1.0 - float(np.exp(-rain / 1.5)))
    return _safe_float(min(max(prob, 0.0), 100.0), ndigits=1)


def _interp_numeric_anchors(anchors: dict[int, float | None], hour: int) -> float | None:
    valid = sorted((h, v) for h, v in anchors.items() if v is not None and np.isfinite(v))
    if not valid:
        return None

    if hour <= valid[0][0]:
        return float(valid[0][1])
    if hour >= valid[-1][0]:
        return float(valid[-1][1])

    left_h = valid[0][0]
    left_v = float(valid[0][1])
    right_h = valid[-1][0]
    right_v = float(valid[-1][1])
    for idx in range(1, len(valid)):
        prev_h, prev_v = valid[idx - 1]
        next_h, next_v = valid[idx]
        if prev_h <= hour <= next_h:
            left_h, left_v = prev_h, float(prev_v)
            right_h, right_v = next_h, float(next_v)
            break

    if right_h == left_h:
        return left_v
    ratio = (hour - left_h) / (right_h - left_h)
    return left_v + ratio * (right_v - left_v)


def _build_hourly_from_forecasts(forecasts: list[dict], source_time: pd.Timestamp) -> list[dict]:
    if not forecasts:
        return []

    temp_anchors: dict[int, float | None] = {}
    rain_anchors: dict[int, float | None] = {}
    for row in forecasts:
        hour = int(row.get("horizon_hours", 0))
        temp_anchors[hour] = _safe_float(row.get("temperature_c"), ndigits=4)
        rain_anchors[hour] = _safe_float(row.get("rain_mm_h"), ndigits=4)

    if 6 in temp_anchors and 12 in temp_anchors and 0 not in temp_anchors:
        t6, t12 = temp_anchors[6], temp_anchors[12]
        if t6 is not None and t12 is not None:
            temp_anchors[0] = (2.0 * float(t6)) - float(t12)
    if 6 in rain_anchors and 12 in rain_anchors and 0 not in rain_anchors:
        r6, r12 = rain_anchors[6], rain_anchors[12]
        if r6 is not None and r12 is not None:
            rain_anchors[0] = max(0.0, (2.0 * float(r6)) - float(r12))

    if 0 not in temp_anchors and 6 in temp_anchors:
        temp_anchors[0] = temp_anchors[6]
    if 0 not in rain_anchors and 6 in rain_anchors:
        rain_anchors[0] = rain_anchors[6]

    rows: list[dict] = []
    for hour in range(0, 24):
        ts = source_time + pd.Timedelta(hours=hour)
        temp_c = _interp_numeric_anchors(temp_anchors, hour)
        rain_mmh = _interp_numeric_anchors(rain_anchors, hour)
        rain_mmh = max(0.0, rain_mmh) if rain_mmh is not None else None
        rows.append(
            {
                "hour_offset": int(hour),
                "hour_of_day": int(hour),
                "hour_label": f"{hour:02d}:00",
                "timestamp": ts.isoformat(),
                "temperature_c": _safe_float(temp_c),
                "rain_mm_h": _safe_float(rain_mmh),
                "rain_probability_pct": _rain_probability_pct(rain_mmh),
                "condition": _climate_label(temp_c, rain_mmh),
            }
        )
    return rows


def _normalize_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower().strip()


def _metric_bundle(predicted: pd.Series, observed: pd.Series) -> dict[str, float | None]:
    pred_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=np.float64)
    obs_values = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(pred_values) & np.isfinite(obs_values)
    if not mask.any():
        return {"rmse": None, "mae": None, "n": 0}

    diff = pred_values[mask] - obs_values[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae, "n": int(mask.sum())}


def _decode_merra_time(values: np.ndarray, day: pd.Timestamp) -> pd.DatetimeIndex:
    if np.issubdtype(values.dtype, np.integer):
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


def _latest_merra_surface_file() -> Path | None:
    candidates = sorted(settings.merra_input_dir.glob("MERRA2_sfc_*.nc"))
    if not candidates:
        candidates = sorted(settings.merra_input_dir.glob("MERRA2_sfc_*.nc4"))
    if not candidates:
        return None
    return candidates[-1]


def _list_merra_surface_files() -> list[Path]:
    files = list(settings.merra_input_dir.glob("MERRA2_sfc_*.nc"))
    files.extend(settings.merra_input_dir.glob("MERRA2_sfc_*.nc4"))
    return sorted(files)


def _merra_surface_file_for_day(day: pd.Timestamp) -> Path | None:
    day_code = pd.Timestamp(day).strftime("%Y%m%d")
    for ext in (".nc", ".nc4"):
        candidate = settings.merra_input_dir / f"MERRA2_sfc_{day_code}{ext}"
        if candidate.exists():
            return candidate
    return None


def _merra_surface_day_bounds() -> tuple[str, str] | None:
    files = _list_merra_surface_files()
    if not files:
        return None
    days = sorted({str(_day_from_merra_filename(path).date()) for path in files})
    if not days:
        return None
    return days[0], days[-1]


def _day_from_merra_filename(path: Path) -> pd.Timestamp:
    match = re.search(r"(\d{8})", path.stem)
    if not match:
        raise ValueError(f"Nao foi possivel extrair data de {path.name}")
    return pd.Timestamp(match.group(1), tz="UTC")


def _quantum_correct_frame_with_payload(frame: pd.DataFrame, payload: dict, device: str = "cpu") -> pd.DataFrame:
    input_columns = payload["input_columns"]
    baseline_columns = payload["baseline_columns"]
    target_columns = payload["target_columns"]

    scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)

    model = QuantumResidualRegressor(
        input_dim=len(input_columns),
        output_dim=len(target_columns),
        n_qubits=int(payload["n_qubits"]),
        n_layers=int(payload["n_layers"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["model_state"])

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()

    for col in input_columns + baseline_columns:
        if col not in frame.columns:
            frame[col] = 0.0

    features = frame[input_columns].to_numpy(dtype=np.float32)
    features_scaled = (features - scaler_mean) / np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    baseline = frame[baseline_columns].to_numpy(dtype=np.float32)

    with torch.no_grad():
        x_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=torch_device)
        b_tensor = torch.tensor(baseline, dtype=torch.float32, device=torch_device)
        corrected = model(x_tensor, b_tensor).cpu().numpy()

    out = frame.copy()
    for idx, target_col in enumerate(target_columns):
        variable = target_col.replace("target_", "")
        out[f"corrected_{variable}"] = corrected[:, idx]
    return out


@lru_cache(maxsize=96)
def _load_feature_rows_cached(
    scope: str,
    variable: str,
    source_timestamp: str,
    uf: str,
    feature_path_str: str,
    feature_mtime: float,
) -> list[dict]:
    del feature_mtime
    feature_path = Path(feature_path_str)
    if not feature_path.exists():
        return []

    variable = variable.upper()
    pred_col = f"pred_{variable}"
    target_col = f"target_{variable}"
    required_cols = [
        "region_id",
        "region_name",
        "uf",
        "lat",
        "lon",
        "timestamp",
        pred_col,
        target_col,
    ]

    header = pd.read_csv(feature_path, nrows=0)
    available = set(header.columns.tolist())
    if not set(required_cols).issubset(available):
        return []

    target_ts = _normalize_timestamp(source_timestamp)
    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        feature_path,
        usecols=required_cols,
        chunksize=250_000,
        dtype={"region_id": "string", "region_name": "string", "uf": "string"},
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        match = chunk.loc[chunk["timestamp"] == target_ts].copy()
        if match.empty:
            continue
        if uf:
            match = match.loc[match["uf"].str.upper() == uf.upper()]
            if match.empty:
                continue
        frames.append(match)

    if not frames:
        return []

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns={pred_col: "baseline_value", target_col: "observed_value"})
    data["region_id"] = data["region_id"].astype(str)
    data = data.drop_duplicates(subset=["region_id"], keep="last")
    cols = ["region_id", "region_name", "uf", "lat", "lon", "baseline_value", "observed_value"]
    return data[cols].to_dict(orient="records")


def _ensure_states_geojson(reference_dir: Path) -> Path:
    target = reference_dir / "brazil_states.geojson"
    if target.exists():
        return target
    response = requests.get(BRAZIL_STATES_GEOJSON_URL, timeout=120)
    response.raise_for_status()
    target.write_text(response.text, encoding="utf-8")
    return target


@lru_cache(maxsize=8)
def _load_states_geojson_cached(path_str: str, mtime: float) -> dict:
    del mtime
    path = Path(path_str)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value <= 0 or value > 24:
            continue
        horizons.append(value)
    if not horizons:
        horizons = [6, 12, 18, 24]
    return sorted(set(horizons))


@lru_cache(maxsize=24)
def _build_next24_cached(
    scope: str,
    uf: str,
    municipality_limit: int,
    horizons_key: str,
    current_time_key: str,
    model_path_str: str,
    model_mtime: float,
    merra_path_str: str,
    merra_mtime: float,
) -> dict:
    del model_mtime
    del merra_mtime
    model_path = Path(model_path_str)
    merra_path = Path(merra_path_str)
    horizons = [int(v) for v in horizons_key.split(",") if v.strip()]

    regions = load_regions(
        reference_dir=settings.reference_dir,
        scope=scope,  # type: ignore[arg-type]
        uf=(uf or None),
        municipality_limit=municipality_limit if scope == "municipio" else None,
    )
    if regions.empty:
        return {
            "scope": scope,
            "horizons_hours": horizons,
            "source_time": "",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "count": 0,
            "items": [],
            "summaries": [],
            "state_horizon_values": {},
            "units": {"temperature": "C", "rainfall": "mm/h", "rain_probability": "%"},
        }

    model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
    baseline_columns = [str(c) for c in model_payload.get("baseline_columns", [])]
    input_columns = [str(c) for c in model_payload.get("input_columns", [])]

    required_vars = sorted(
        {
            col.replace("pred_", "")
            for col in baseline_columns
            if col.startswith("pred_")
        }
    )
    if "T2M" not in required_vars:
        required_vars.append("T2M")
    if "PRECTOT" not in required_vars:
        required_vars.append("PRECTOT")

    day = _day_from_merra_filename(merra_path)
    ds = xr.load_dataset(merra_path, engine="h5netcdf")
    times = _decode_merra_time(ds["time"].values, day=day)
    valid_idx = np.where(~times.isna())[0]
    if len(valid_idx) == 0:
        raise RuntimeError("Arquivo MERRA-2 sem timestamps validos.")

    last_idx = int(valid_idx[-1])
    data_source_time = pd.Timestamp(times[last_idx]).tz_convert("UTC")
    source_time = _normalize_timestamp(current_time_key).floor("D")

    lats = ds["lat"].values
    lons = ds["lon"].values
    lat_idx, lon_idx = _nearest_indexes(regions, lats, lons)

    base_values: dict[str, np.ndarray] = {}
    for var in required_vars:
        if var in ds.variables:
            raw = ds[var].isel(time=last_idx).values
            base_values[var] = raw[lat_idx, lon_idx].astype(np.float32)
        else:
            base_values[var] = np.full(len(regions), np.nan, dtype=np.float32)

    records: list[dict] = []
    for row in regions[["region_id", "region_name", "uf", "lat", "lon"]].to_dict(orient="records"):
        records.append(
            {
                "region_id": str(row["region_id"]),
                "region_name": row["region_name"],
                "uf": row["uf"],
                "lat": _safe_float(row["lat"], ndigits=6),
                "lon": _safe_float(row["lon"], ndigits=6),
                "macroregion": _macroregion_for_uf(str(row["uf"])),
                "forecasts": [],
            }
        )

    base_frame = regions[["region_id", "region_name", "uf", "lat", "lon"]].copy()
    base_frame["scope"] = scope

    for horizon in horizons:
        frame = base_frame.copy()
        ts = source_time + pd.Timedelta(hours=horizon)
        frame["timestamp"] = ts
        frame["sin_hod"] = np.sin(2.0 * np.pi * ts.hour / 24.0)
        frame["cos_hod"] = np.cos(2.0 * np.pi * ts.hour / 24.0)
        frame["sin_doy"] = np.sin(2.0 * np.pi * ts.dayofyear / 366.0)
        frame["cos_doy"] = np.cos(2.0 * np.pi * ts.dayofyear / 366.0)

        for var in required_vars:
            frame[f"pred_{var}"] = base_values[var]

        for col in input_columns + baseline_columns:
            if col not in frame.columns:
                frame[col] = 0.0

        corrected = _quantum_correct_frame_with_payload(frame=frame, payload=model_payload, device="cpu")

        corr_t2m = corrected["corrected_T2M"] if "corrected_T2M" in corrected.columns else frame.get("pred_T2M")
        corr_pr = (
            corrected["corrected_PRECTOT"] if "corrected_PRECTOT" in corrected.columns else frame.get("pred_PRECTOT")
        )
        base_t2m = frame["pred_T2M"] if "pred_T2M" in frame.columns else pd.Series(np.nan, index=frame.index)
        base_pr = frame["pred_PRECTOT"] if "pred_PRECTOT" in frame.columns else pd.Series(np.nan, index=frame.index)

        for idx, record in enumerate(records):
            temp_c = _to_temp_c(corr_t2m.iloc[idx] if corr_t2m is not None else np.nan)
            rain_mmh = _to_rain_mm_h(corr_pr.iloc[idx] if corr_pr is not None else np.nan)
            baseline_temp_c = _to_temp_c(base_t2m.iloc[idx])
            baseline_rain_mmh = _to_rain_mm_h(base_pr.iloc[idx])
            record["forecasts"].append(
                {
                    "horizon_hours": int(horizon),
                    "temperature_c": _safe_float(temp_c),
                    "rain_mm_h": _safe_float(rain_mmh),
                    "rain_probability_pct": _rain_probability_pct(rain_mmh),
                    "baseline_temperature_c": _safe_float(baseline_temp_c),
                    "baseline_rain_mm_h": _safe_float(baseline_rain_mmh),
                    "condition": _climate_label(temp_c, rain_mmh),
                }
            )

    for rec in records:
        rec["hourly"] = _build_hourly_from_forecasts(rec.get("forecasts", []), source_time=source_time)

    summaries: list[dict] = []
    macroregions = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    for macro in macroregions:
        macro_records = [r for r in records if r["macroregion"] == macro]
        if not macro_records:
            continue
        horizon_rows: list[dict] = []
        for horizon in horizons:
            temps = []
            rains = []
            probs = []
            for rec in macro_records:
                found = next((f for f in rec["forecasts"] if f["horizon_hours"] == horizon), None)
                if not found:
                    continue
                if found["temperature_c"] is not None:
                    temps.append(float(found["temperature_c"]))
                if found["rain_mm_h"] is not None:
                    rains.append(float(found["rain_mm_h"]))
                if found.get("rain_probability_pct") is not None:
                    probs.append(float(found["rain_probability_pct"]))
            horizon_rows.append(
                {
                    "horizon_hours": int(horizon),
                    "temperature_c_mean": _safe_float(np.mean(temps) if temps else np.nan),
                    "rain_mm_h_mean": _safe_float(np.mean(rains) if rains else np.nan),
                    "rain_probability_pct_mean": _safe_float(np.mean(probs) if probs else np.nan),
                }
            )

        temp_24 = [r["temperature_c_mean"] for r in horizon_rows if r["temperature_c_mean"] is not None]
        rain_24 = [r["rain_mm_h_mean"] for r in horizon_rows if r["rain_mm_h_mean"] is not None]
        rain_prob_24 = [r["rain_probability_pct_mean"] for r in horizon_rows if r["rain_probability_pct_mean"] is not None]
        summaries.append(
            {
                "macroregion": macro,
                "temperature_c_24h_mean": _safe_float(np.mean(temp_24) if temp_24 else np.nan),
                "rain_mm_h_24h_mean": _safe_float(np.mean(rain_24) if rain_24 else np.nan),
                "rain_probability_pct_24h_mean": _safe_float(np.mean(rain_prob_24) if rain_prob_24 else np.nan),
                "horizons": horizon_rows,
            }
        )

    state_horizon_values: dict[str, dict[str, dict[str, float | None]]] = {}
    if scope == "state":
        for rec in records:
            uf_key = str(rec["uf"]).upper()
            state_horizon_values.setdefault(uf_key, {})
            for fc in rec["forecasts"]:
                state_horizon_values[uf_key][str(fc["horizon_hours"])] = {
                    "temperature_c": fc["temperature_c"],
                    "rain_mm_h": fc["rain_mm_h"],
                    "rain_probability_pct": fc.get("rain_probability_pct"),
                }
    else:
        accumulator: dict[str, dict[int, dict[str, list[float]]]] = {}
        for rec in records:
            uf_key = str(rec["uf"]).upper()
            accumulator.setdefault(uf_key, {})
            for fc in rec["forecasts"]:
                horizon = int(fc["horizon_hours"])
                accumulator[uf_key].setdefault(horizon, {"temperature_c": [], "rain_mm_h": [], "rain_probability_pct": []})
                if fc["temperature_c"] is not None:
                    accumulator[uf_key][horizon]["temperature_c"].append(float(fc["temperature_c"]))
                if fc["rain_mm_h"] is not None:
                    accumulator[uf_key][horizon]["rain_mm_h"].append(float(fc["rain_mm_h"]))
                if fc.get("rain_probability_pct") is not None:
                    accumulator[uf_key][horizon]["rain_probability_pct"].append(float(fc["rain_probability_pct"]))
        for uf_key, by_horizon in accumulator.items():
            state_horizon_values[uf_key] = {}
            for horizon, vectors in by_horizon.items():
                state_horizon_values[uf_key][str(horizon)] = {
                    "temperature_c": _safe_float(np.mean(vectors["temperature_c"]) if vectors["temperature_c"] else np.nan),
                    "rain_mm_h": _safe_float(np.mean(vectors["rain_mm_h"]) if vectors["rain_mm_h"] else np.nan),
                    "rain_probability_pct": _safe_float(
                        np.mean(vectors["rain_probability_pct"]) if vectors["rain_probability_pct"] else np.nan
                    ),
                }

    return {
        "scope": scope,
        "horizons_hours": horizons,
        "source_time": source_time.isoformat(),
        "data_source_time": data_source_time.isoformat(),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "count": len(records),
        "items": records,
        "summaries": summaries,
        "state_horizon_values": state_horizon_values,
        "units": {"temperature": "C", "rainfall": "mm/h", "rain_probability": "%"},
    }


def _find_region_record(records: list[dict], scope: str, region: str, uf: str = "") -> dict | None:
    if not records:
        return None

    scoped = list(records)
    uf_norm = (uf or "").upper().strip()
    if uf_norm:
        scoped = [rec for rec in scoped if str(rec.get("uf", "")).upper() == uf_norm]
        if not scoped:
            return None

    raw = (region or "").strip()
    if not raw:
        return scoped[0] if scoped else None
    region_norm = _normalize_text(raw)

    if scope == "state" and len(raw) == 2:
        direct = next((rec for rec in scoped if str(rec.get("uf", "")).upper() == raw.upper()), None)
        if direct:
            return direct

    direct_id = next((rec for rec in scoped if str(rec.get("region_id", "")) == raw), None)
    if direct_id:
        return direct_id

    exact_name = next(
        (rec for rec in scoped if _normalize_text(str(rec.get("region_name", ""))) == region_norm),
        None,
    )
    if exact_name:
        return exact_name

    fuzzy = next(
        (rec for rec in scoped if region_norm in _normalize_text(str(rec.get("region_name", "")))),
        None,
    )
    return fuzzy


def _build_observed_hourly_for_point(
    merra_path: Path,
    source_day: pd.Timestamp,
    lat: float,
    lon: float,
) -> tuple[list[dict], str]:
    day = _day_from_merra_filename(merra_path)
    ds = xr.load_dataset(merra_path, engine="h5netcdf")
    try:
        times = _decode_merra_time(ds["time"].values, day=day)
        valid_idx = np.where(~times.isna())[0]
        observed_source_time = ""
        if len(valid_idx) > 0:
            observed_source_time = pd.Timestamp(times[int(valid_idx[-1])]).tz_convert("UTC").isoformat()

        point_df = pd.DataFrame([{"lat": float(lat), "lon": float(lon)}])
        lat_idx_arr, lon_idx_arr = _nearest_indexes(point_df, ds["lat"].values, ds["lon"].values)
        lat_idx = int(lat_idx_arr[0])
        lon_idx = int(lon_idx_arr[0])

        temp_anchors: dict[int, float | None] = {}
        rain_anchors: dict[int, float | None] = {}

        for idx in valid_idx:
            ts = pd.Timestamp(times[int(idx)]).tz_convert("UTC")
            if str(ts.date()) != str(pd.Timestamp(source_day).date()):
                continue
            hour = int(ts.hour)
            if "T2M" in ds.variables:
                temp_raw = ds["T2M"].isel(time=int(idx)).values[lat_idx, lon_idx]
                temp_anchors[hour] = _to_temp_c(temp_raw)
            if "PRECTOT" in ds.variables:
                rain_raw = ds["PRECTOT"].isel(time=int(idx)).values[lat_idx, lon_idx]
                rain_anchors[hour] = _to_rain_mm_h(rain_raw)

        rows: list[dict] = []
        day_start = pd.Timestamp(source_day).tz_convert("UTC").floor("D")
        for hour in range(0, 24):
            ts = day_start + pd.Timedelta(hours=hour)
            temp_c = _interp_numeric_anchors(temp_anchors, hour)
            rain_mmh = _interp_numeric_anchors(rain_anchors, hour)
            rain_mmh = max(0.0, float(rain_mmh)) if rain_mmh is not None else None
            rows.append(
                {
                    "hour_of_day": int(hour),
                    "hour_label": f"{hour:02d}:00",
                    "timestamp": ts.isoformat(),
                    "temperature_c": _safe_float(temp_c),
                    "rain_mm_h": _safe_float(rain_mmh),
                    "rain_probability_pct": _rain_probability_pct(rain_mmh),
                    "condition": _climate_label(temp_c, rain_mmh),
                }
            )
        return rows, observed_source_time
    finally:
        ds.close()


def _comparison_metric_bundle(rows: list[dict], pred_key: str, obs_key: str) -> dict[str, float | None]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rmse": None, "mae": None, "n": 0}
    return _metric_bundle(frame[pred_key], frame[obs_key])


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(ui_dir / "index.html")


@app.get("/dashboard/hourly", include_in_schema=False)
def hourly_dashboard() -> FileResponse:
    return FileResponse(ui_dir / "hourly.html")


@app.get("/dashboard/snap", include_in_schema=False)
def snap_dashboard() -> FileResponse:
    return FileResponse(ui_dir / "snap.html")


@app.get("/dashboard/compare", include_in_schema=False)
def compare_dashboard() -> FileResponse:
    return FileResponse(ui_dir / "compare.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_root": str(settings.data_root),
        "preferred_drive": settings.preferred_drive,
    }


@app.get("/api/geo/states")
def geo_states() -> dict:
    path = _ensure_states_geojson(settings.reference_dir)
    return _load_states_geojson_cached(str(path), path.stat().st_mtime)


@app.get("/api/regions")
def regions(
    scope: Literal["state", "municipio"] = Query(default="state"),
    uf: str | None = Query(default=None),
    limit: int = Query(default=900, ge=1, le=5570),
) -> dict:
    items = load_regions(
        reference_dir=settings.reference_dir,
        scope=scope,
        uf=uf,
        municipality_limit=limit if scope == "municipio" else None,
    )
    return {"scope": scope, "count": len(items), "items": items.to_dict(orient="records")}


@app.get("/api/forecast/latest")
def latest_forecast(
    scope: Literal["state", "municipio"] = Query(default="state"),
    variable: str = Query(default="T2M"),
    uf: str | None = Query(default=None),
) -> dict:
    payload = load_latest_forecast(settings.output_dir, scope, variable)
    if uf:
        payload["items"] = [
            item for item in payload["items"] if str(item.get("uf", "")).upper() == uf.upper()
        ]
        payload["count"] = len(payload["items"])
    return payload


@app.get("/api/forecast/files")
def forecast_files() -> dict:
    files = list_available_forecasts(settings.output_dir)
    return {"count": len(files), "files": files}


@app.get("/api/forecast/compare24h/meta")
def compare24h_meta(
    scope: Literal["state", "municipio"] = Query(default="state"),
    uf: str | None = Query(default=None),
    municipality_limit: int = Query(default=1200, ge=50, le=5570),
) -> dict:
    bounds = _merra_surface_day_bounds()
    today = str(pd.Timestamp.now(tz="UTC").date())
    if bounds is None:
        return {
            "scope": scope,
            "min_date": "",
            "max_observed_date": "",
            "max_selectable_date": today,
            "recommended_date": "",
            "count": 0,
            "regions": [],
            "error": "Nenhum arquivo MERRA2_sfc encontrado em data/prithvi_input/merra2.",
        }

    min_date, max_observed_date = bounds
    recommended_date = min(today, max_observed_date)
    regions_df = load_regions(
        reference_dir=settings.reference_dir,
        scope=scope,
        uf=uf,
        municipality_limit=municipality_limit if scope == "municipio" else None,
    )
    fields = ["region_id", "region_name", "uf", "lat", "lon"]
    regions = regions_df[fields].to_dict(orient="records") if not regions_df.empty else []
    return {
        "scope": scope,
        "min_date": min_date,
        "max_observed_date": max_observed_date,
        "max_selectable_date": today,
        "recommended_date": recommended_date,
        "count": len(regions),
        "regions": regions,
    }


@app.get("/api/forecast/compare24h")
def compare24h_forecast(
    date: str = Query(..., description="Data no formato YYYY-MM-DD"),
    region: str = Query(..., min_length=1, description="UF/cidade/codigo da regiao"),
    scope: Literal["state", "municipio"] = Query(default="state"),
    uf: str | None = Query(default=None),
    municipality_limit: int = Query(default=5570, ge=50, le=5570),
    horizons: str = Query(default="6,12,18,24"),
) -> dict:
    try:
        selected_day = _normalize_timestamp(date).floor("D")
    except Exception:
        return {
            "scope": scope,
            "date": date,
            "count": 0,
            "rows": [],
            "error": "Data invalida. Use formato YYYY-MM-DD.",
        }

    today = pd.Timestamp.now(tz="UTC").floor("D")
    if selected_day > today:
        return {
            "scope": scope,
            "date": str(selected_day.date()),
            "count": 0,
            "rows": [],
            "error": "Comparacao com dado real so e permitida ate a data atual.",
        }

    bounds = _merra_surface_day_bounds()
    merra_path = _merra_surface_file_for_day(selected_day)
    if merra_path is None:
        return {
            "scope": scope,
            "date": str(selected_day.date()),
            "count": 0,
            "rows": [],
            "available_range": {"min_date": bounds[0], "max_date": bounds[1]} if bounds else None,
            "error": "Sem observacao MERRA-2 para a data selecionada.",
        }

    model_path = settings.model_dir / f"quantum_{scope}.pt"
    if not model_path.exists():
        return {
            "scope": scope,
            "date": str(selected_day.date()),
            "count": 0,
            "rows": [],
            "error": f"Modelo nao encontrado: {model_path.name}",
        }

    parsed_horizons = _parse_horizons(horizons)
    payload = _build_next24_cached(
        scope=scope,
        uf=(uf or "").upper(),
        municipality_limit=municipality_limit,
        horizons_key=",".join(str(h) for h in parsed_horizons),
        current_time_key=selected_day.isoformat(),
        model_path_str=str(model_path),
        model_mtime=model_path.stat().st_mtime,
        merra_path_str=str(merra_path),
        merra_mtime=merra_path.stat().st_mtime,
    )

    selected = _find_region_record(payload.get("items") or [], scope=scope, region=region, uf=(uf or ""))
    if selected is None:
        return {
            "scope": scope,
            "date": str(selected_day.date()),
            "count": 0,
            "rows": [],
            "error": "Regiao nao encontrada para os filtros informados.",
        }

    lat = pd.to_numeric(selected.get("lat"), errors="coerce")
    lon = pd.to_numeric(selected.get("lon"), errors="coerce")
    if not np.isfinite(lat) or not np.isfinite(lon):
        return {
            "scope": scope,
            "date": str(selected_day.date()),
            "count": 0,
            "rows": [],
            "error": "Latitude/longitude da regiao invalida.",
        }

    predicted_rows = list(selected.get("hourly") or [])
    observed_rows, observed_source_time = _build_observed_hourly_for_point(
        merra_path=merra_path,
        source_day=selected_day,
        lat=float(lat),
        lon=float(lon),
    )

    pred_by_hour = {
        int(row.get("hour_of_day", row.get("hour_offset", -1))): row
        for row in predicted_rows
        if row.get("hour_of_day") is not None or row.get("hour_offset") is not None
    }
    obs_by_hour = {int(row.get("hour_of_day", -1)): row for row in observed_rows if row.get("hour_of_day") is not None}

    rows: list[dict] = []
    for hour in range(0, 24):
        pred = pred_by_hour.get(hour, {})
        obs = obs_by_hour.get(hour, {})

        pred_temp = pred.get("temperature_c")
        obs_temp = obs.get("temperature_c")
        pred_prob = pred.get("rain_probability_pct")
        obs_prob = obs.get("rain_probability_pct")
        pred_rain = pred.get("rain_mm_h")
        obs_rain = obs.get("rain_mm_h")

        temp_error = _safe_float(float(pred_temp) - float(obs_temp)) if pred_temp is not None and obs_temp is not None else None
        prob_error = _safe_float(float(pred_prob) - float(obs_prob)) if pred_prob is not None and obs_prob is not None else None
        rain_error = _safe_float(float(pred_rain) - float(obs_rain)) if pred_rain is not None and obs_rain is not None else None

        rows.append(
            {
                "hour_of_day": hour,
                "hour_label": f"{hour:02d}:00",
                "timestamp": (selected_day + pd.Timedelta(hours=hour)).isoformat(),
                "pred_temperature_c": pred_temp,
                "obs_temperature_c": obs_temp,
                "error_temperature_c": temp_error,
                "pred_rain_probability_pct": pred_prob,
                "obs_rain_probability_pct": obs_prob,
                "error_rain_probability_pct": prob_error,
                "pred_rain_mm_h": pred_rain,
                "obs_rain_mm_h": obs_rain,
                "error_rain_mm_h": rain_error,
                "pred_condition": pred.get("condition"),
                "obs_condition": obs.get("condition"),
            }
        )

    temp_metrics = _comparison_metric_bundle(rows, "pred_temperature_c", "obs_temperature_c")
    rain_prob_metrics = _comparison_metric_bundle(rows, "pred_rain_probability_pct", "obs_rain_probability_pct")
    rain_mm_metrics = _comparison_metric_bundle(rows, "pred_rain_mm_h", "obs_rain_mm_h")
    coverage_pct = (100.0 * temp_metrics["n"] / 24.0) if temp_metrics["n"] else 0.0

    return {
        "scope": scope,
        "date": str(selected_day.date()),
        "region": {
            "region_id": selected.get("region_id"),
            "region_name": selected.get("region_name"),
            "uf": selected.get("uf"),
            "lat": selected.get("lat"),
            "lon": selected.get("lon"),
        },
        "source_time": payload.get("source_time"),
        "data_source_time": payload.get("data_source_time"),
        "observed_source_time": observed_source_time,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "count": len(rows),
        "metrics": {
            "temperature_rmse": _safe_float(temp_metrics["rmse"]),
            "temperature_mae": _safe_float(temp_metrics["mae"]),
            "rain_probability_rmse": _safe_float(rain_prob_metrics["rmse"]),
            "rain_probability_mae": _safe_float(rain_prob_metrics["mae"]),
            "rain_mm_h_rmse": _safe_float(rain_mm_metrics["rmse"]),
            "rain_mm_h_mae": _safe_float(rain_mm_metrics["mae"]),
            "coverage_pct": _safe_float(coverage_pct, ndigits=2),
            "samples": temp_metrics["n"],
        },
        "rows": rows,
        "units": {"temperature": "C", "rainfall": "mm/h", "rain_probability": "%"},
    }


@app.get("/api/forecast/next24h")
def next24_forecast(
    scope: Literal["state", "municipio"] = Query(default="state"),
    uf: str | None = Query(default=None),
    municipality_limit: int = Query(default=1200, ge=50, le=5570),
    horizons: str = Query(default="6,12,18,24"),
    date: str | None = Query(default=None, description="Data de referencia no formato YYYY-MM-DD"),
) -> dict:
    model_path = settings.model_dir / f"quantum_{scope}.pt"
    if not model_path.exists():
        return {
            "scope": scope,
            "horizons_hours": _parse_horizons(horizons),
            "source_time": "",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "count": 0,
            "items": [],
            "summaries": [],
            "state_horizon_values": {},
            "units": {"temperature": "C", "rainfall": "mm/h", "rain_probability": "%"},
            "error": f"Modelo nao encontrado: {model_path.name}",
        }

    merra_path = _latest_merra_surface_file()
    if merra_path is None:
        return {
            "scope": scope,
            "horizons_hours": _parse_horizons(horizons),
            "source_time": "",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "count": 0,
            "items": [],
            "summaries": [],
            "state_horizon_values": {},
            "units": {"temperature": "C", "rainfall": "mm/h", "rain_probability": "%"},
            "error": "Nenhum arquivo MERRA2_sfc encontrado em data/prithvi_input/merra2.",
        }

    parsed_horizons = _parse_horizons(horizons)
    target_day = pd.Timestamp.now(tz="UTC").floor("D")
    if date:
        try:
            target_day = _normalize_timestamp(date).floor("D")
        except Exception:
            return {
                "scope": scope,
                "horizons_hours": parsed_horizons,
                "source_time": "",
                "generated_at": pd.Timestamp.utcnow().isoformat(),
                "count": 0,
                "items": [],
                "summaries": [],
                "state_horizon_values": {},
                "units": {"temperature": "C", "rainfall": "mm/h", "rain_probability": "%"},
                "error": "Data invalida. Use formato YYYY-MM-DD.",
            }

    current_time_key = target_day.isoformat()
    return _build_next24_cached(
        scope=scope,
        uf=(uf or "").upper(),
        municipality_limit=municipality_limit,
        horizons_key=",".join(str(h) for h in parsed_horizons),
        current_time_key=current_time_key,
        model_path_str=str(model_path),
        model_mtime=model_path.stat().st_mtime,
        merra_path_str=str(merra_path),
        merra_mtime=merra_path.stat().st_mtime,
    )


@app.get("/api/forecast/snap")
def city_snap_forecast(
    city: str = Query(..., min_length=2),
    date: str | None = Query(default=None, description="Data no formato YYYY-MM-DD"),
    uf: str | None = Query(default=None),
    municipality_limit: int = Query(default=5570, ge=50, le=5570),
) -> dict:
    model_path = settings.model_dir / "quantum_municipio.pt"
    if not model_path.exists():
        return {
            "city": city,
            "date": date or "",
            "count": 0,
            "hourly": [],
            "error": f"Modelo nao encontrado: {model_path.name}",
        }

    merra_path = _latest_merra_surface_file()
    if merra_path is None:
        return {
            "city": city,
            "date": date or "",
            "count": 0,
            "hourly": [],
            "error": "Nenhum arquivo MERRA2_sfc encontrado em data/prithvi_input/merra2.",
        }

    selected_day = pd.Timestamp.now(tz="UTC").floor("D")
    if date:
        try:
            selected_day = _normalize_timestamp(date).floor("D")
        except Exception:
            return {
                "city": city,
                "uf": (uf or "").upper(),
                "date": date,
                "count": 0,
                "hourly": [],
                "error": "Data invalida. Use formato YYYY-MM-DD.",
            }

    payload = _build_next24_cached(
        scope="municipio",
        uf=(uf or "").upper(),
        municipality_limit=municipality_limit,
        horizons_key="6,12,18,24",
        current_time_key=selected_day.isoformat(),
        model_path_str=str(model_path),
        model_mtime=model_path.stat().st_mtime,
        merra_path_str=str(merra_path),
        merra_mtime=merra_path.stat().st_mtime,
    )

    items = payload.get("items") or []
    term = _normalize_text(city)
    matches = [item for item in items if term in _normalize_text(item.get("region_name", ""))]
    if not matches:
        return {
            "city": city,
            "date": date or "",
            "count": 0,
            "hourly": [],
            "error": "Cidade nao encontrada no horizonte atual.",
        }

    exact = next((it for it in matches if _normalize_text(it.get("region_name", "")) == term), None)
    selected = exact or matches[0]
    hourly = list(selected.get("hourly") or [])
    selected_date = str(selected_day.date())
    available_dates = [selected_date]
    return {
        "city": selected.get("region_name", city),
        "uf": selected.get("uf", ""),
        "region_id": selected.get("region_id", ""),
        "lat": selected.get("lat"),
        "lon": selected.get("lon"),
        "date": selected_date,
        "source_time": payload.get("source_time", ""),
        "data_source_time": payload.get("data_source_time", ""),
        "generated_at": payload.get("generated_at", ""),
        "available_dates": available_dates,
        "count": len(hourly),
        "hourly": hourly,
        "forecasts": selected.get("forecasts", []),
    }


@app.get("/api/forecast/compare/latest")
def compare_latest_forecast(
    scope: Literal["state", "municipio"] = Query(default="state"),
    variable: str = Query(default="T2M"),
    uf: str | None = Query(default=None),
) -> dict:
    variable = variable.upper()
    payload = load_latest_forecast(settings.output_dir, scope, variable)

    if uf:
        payload["items"] = [
            item for item in payload["items"] if str(item.get("uf", "")).upper() == uf.upper()
        ]
        payload["count"] = len(payload["items"])

    forecast_df = pd.DataFrame(payload.get("items") or [])
    if forecast_df.empty:
        return {
            "scope": scope,
            "variable": variable,
            "model_name": payload.get("model_name", ""),
            "source_timestamp": payload.get("source_timestamp", ""),
            "generated_at": payload.get("generated_at", ""),
            "count": 0,
            "matched_count": 0,
            "coverage_pct": 0.0,
            "metrics": {
                "rmse_corrected": None,
                "mae_corrected": None,
                "rmse_baseline": None,
                "mae_baseline": None,
            },
            "items": [],
        }

    forecast_df["region_id"] = forecast_df["region_id"].astype(str)
    forecast_df["corrected_value"] = pd.to_numeric(forecast_df["value"], errors="coerce")
    forecast_df = forecast_df.drop(columns=["value"], errors="ignore")

    source_timestamp = str(payload.get("source_timestamp") or "")
    feature_rows: list[dict] = []
    feature_path = settings.feature_dir / f"features_{scope}.csv"
    if source_timestamp and feature_path.exists():
        feature_rows = _load_feature_rows_cached(
            scope=scope,
            variable=variable,
            source_timestamp=source_timestamp,
            uf=(uf or ""),
            feature_path_str=str(feature_path),
            feature_mtime=feature_path.stat().st_mtime,
        )

    if feature_rows:
        feature_df = pd.DataFrame(feature_rows)
        feature_df["region_id"] = feature_df["region_id"].astype(str)
        merged = forecast_df.merge(feature_df, how="left", on="region_id", suffixes=("", "_feature"))
        for col in ("region_name", "uf", "lat", "lon"):
            alt = f"{col}_feature"
            if alt in merged.columns:
                merged[col] = merged[col].fillna(merged[alt])
                merged = merged.drop(columns=[alt])
    else:
        merged = forecast_df.copy()
        merged["baseline_value"] = np.nan
        merged["observed_value"] = np.nan

    for col in ("lat", "lon", "corrected_value", "baseline_value", "observed_value"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["corrected_error"] = merged["corrected_value"] - merged["observed_value"]
    merged["baseline_error"] = merged["baseline_value"] - merged["observed_value"]
    merged["abs_corrected_error"] = np.abs(merged["corrected_error"])
    merged["abs_baseline_error"] = np.abs(merged["baseline_error"])

    corrected_metrics = _metric_bundle(merged["corrected_value"], merged["observed_value"])
    baseline_metrics = _metric_bundle(merged["baseline_value"], merged["observed_value"])
    matched_count = int(pd.to_numeric(merged["observed_value"], errors="coerce").notna().sum())
    coverage_pct = (matched_count / len(merged)) * 100.0 if len(merged) > 0 else 0.0

    fields = [
        "region_id",
        "region_name",
        "uf",
        "lat",
        "lon",
        "corrected_value",
        "baseline_value",
        "observed_value",
        "corrected_error",
        "baseline_error",
        "abs_corrected_error",
        "abs_baseline_error",
    ]
    items = merged[fields].where(pd.notna(merged[fields]), None).to_dict(orient="records")

    return {
        "scope": scope,
        "variable": variable,
        "model_name": payload.get("model_name", ""),
        "source_timestamp": payload.get("source_timestamp", ""),
        "generated_at": payload.get("generated_at", ""),
        "count": int(len(items)),
        "matched_count": matched_count,
        "coverage_pct": round(coverage_pct, 2),
        "metrics": {
            "rmse_corrected": corrected_metrics["rmse"],
            "mae_corrected": corrected_metrics["mae"],
            "rmse_baseline": baseline_metrics["rmse"],
            "mae_baseline": baseline_metrics["mae"],
        },
        "items": items,
    }


@app.get("/api/model/status")
def model_status() -> dict:
    model_files = sorted(settings.model_dir.glob("*.pt"))
    models = []
    for model_file in model_files:
        details = {
            "file": model_file.name,
            "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
        }
        try:
            payload = torch.load(model_file, map_location="cpu", weights_only=False)
            details["train_loss"] = payload.get("train_loss")
            details["val_loss"] = payload.get("val_loss")
            details["n_qubits"] = payload.get("n_qubits")
            metrics = payload.get("metrics") or {}
            details["val_rmse"] = metrics.get("rmse")
            details["val_mae"] = metrics.get("mae")
            details["hidden_dim"] = payload.get("hidden_dim")
            details["n_layers"] = payload.get("n_layers")
            tuning = payload.get("tuning") or {}
            details["fine_tuned"] = bool(tuning.get("enabled", False))
            details["best_trial"] = tuning.get("best_trial")
            details["trials"] = tuning.get("trials")
        except Exception:
            details["train_loss"] = None
            details["val_loss"] = None
            details["n_qubits"] = None
            details["val_rmse"] = None
            details["val_mae"] = None
            details["hidden_dim"] = None
            details["n_layers"] = None
            details["fine_tuned"] = None
            details["best_trial"] = None
            details["trials"] = None
        models.append(details)

    return {"count": len(models), "models": models}
