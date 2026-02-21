from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def _forecast_filename(scope: str, variable: str) -> str:
    safe_scope = scope.lower().strip()
    safe_variable = variable.upper().strip()
    return f"latest_{safe_scope}_{safe_variable}.json"


def save_forecast_snapshot(
    output_dir: Path,
    scope: str,
    variable: str,
    forecast: pd.DataFrame,
    model_name: str,
    source_timestamp: datetime,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _forecast_filename(scope, variable)

    payload = {
        "scope": scope,
        "variable": variable.upper(),
        "model_name": model_name,
        "source_timestamp": source_timestamp.astimezone(timezone.utc).isoformat(),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "count": int(len(forecast)),
        "items": forecast.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def load_latest_forecast(output_dir: Path, scope: str, variable: str) -> dict:
    path = output_dir / _forecast_filename(scope, variable)
    if not path.exists():
        return {
            "scope": scope,
            "variable": variable.upper(),
            "model_name": "",
            "source_timestamp": "",
            "generated_at": "",
            "count": 0,
            "items": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def list_available_forecasts(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(path.name for path in output_dir.glob("latest_*.json"))
