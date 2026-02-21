from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from previsao_tempo_quantico.data import (
    ensure_location_catalog,
    load_regions,
    save_forecast_snapshot,
)
from previsao_tempo_quantico.models import PrithviForecastEngine
from previsao_tempo_quantico.settings import get_settings
from previsao_tempo_quantico.training import (
    apply_quantum_correction,
    build_training_frame,
    build_training_frame_persistence,
    train_quantum_model,
    tune_quantum_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino local do corretor quantico para previsao do tempo")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--scope", choices=["state", "municipio"], default=None)
    parser.add_argument("--uf", default=None)
    parser.add_argument("--municipality-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mode", choices=["auto", "prithvi", "persistence"], default="auto")
    parser.add_argument("--force-prithvi", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--tuning-trials", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--no-temporal-validation", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Arquivo de configuracao invalido")
    return data


def _available_ram_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        lines = meminfo.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = float(parts[1])
                    return kb / (1024.0 * 1024.0)
    return None


def _memory_snapshot() -> tuple[float | None, float | None, float | None]:
    ram_gb = _available_ram_gb()
    vram_free_gb: float | None = None
    vram_total_gb: float | None = None

    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        vram_free_gb = free_b / (1024.0**3)
        vram_total_gb = total_b / (1024.0**3)

    return ram_gb, vram_free_gb, vram_total_gb


def _can_run_prithvi(
    ram_gb: float | None,
    vram_free_gb: float | None,
    vram_total_gb: float | None,
) -> bool:
    # Regra conservadora para evitar OOM do checkpoint de 2.3B parametros.
    if (
        vram_total_gb is not None
        and vram_free_gb is not None
        and vram_total_gb >= 24.0
        and vram_free_gb >= 18.0
    ):
        return True
    if ram_gb is not None and ram_gb >= 64.0:
        return True
    return False


def _resolve_mode(
    requested_mode: str,
    force_prithvi: bool,
    ram_gb: float | None,
    vram_free_gb: float | None,
    vram_total_gb: float | None,
) -> str:
    can_prithvi = _can_run_prithvi(ram_gb, vram_free_gb, vram_total_gb)

    if requested_mode == "auto":
        return "prithvi" if can_prithvi else "persistence"

    if requested_mode == "prithvi" and not can_prithvi and not force_prithvi:
        print(
            "Memoria insuficiente para modo prithvi no hardware atual. "
            "Alternando automaticamente para modo persistence."
        )
        return "persistence"

    return requested_mode


def main() -> None:
    args = parse_args()
    settings = get_settings()

    config = load_config(Path(args.config))

    scope = args.scope or config["project"]["scope"]
    uf = args.uf if args.uf is not None else config["project"].get("uf")
    municipality_limit = (
        args.municipality_limit
        if args.municipality_limit is not None
        else config["project"].get("municipality_limit", 900)
    )

    ensure_location_catalog(settings.reference_dir)
    regions = load_regions(
        reference_dir=settings.reference_dir,
        scope=scope,
        uf=uf,
        municipality_limit=municipality_limit if scope == "municipio" else None,
    )

    if regions.empty:
        raise RuntimeError("Nenhuma regiao encontrada para o escopo solicitado")

    variables = [v.upper() for v in config["outputs"]["variables"]]
    training_cfg = config.get("training", {})
    tuning_cfg = config.get("fine_tuning", {})
    fine_tune_enabled = bool(args.fine_tune or tuning_cfg.get("enabled", False))
    tuning_trials = int(args.tuning_trials if args.tuning_trials is not None else tuning_cfg.get("trials", 8))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else training_cfg.get("val_fraction", 0.2))
    temporal_validation = not args.no_temporal_validation
    if not args.no_temporal_validation:
        temporal_validation = bool(training_cfg.get("temporal_validation", True))
    early_stopping_patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else training_cfg.get("early_stopping_patience", 5)
    )
    min_epochs = int(args.min_epochs if args.min_epochs is not None else training_cfg.get("min_epochs", 5))
    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 42))
    search_space = tuning_cfg.get("search_space")
    if not isinstance(search_space, dict):
        search_space = None

    ram_gb, vram_free_gb, vram_total_gb = _memory_snapshot()
    effective_mode = _resolve_mode(
        requested_mode=args.mode,
        force_prithvi=args.force_prithvi,
        ram_gb=ram_gb,
        vram_free_gb=vram_free_gb,
        vram_total_gb=vram_total_gb,
    )

    print(f"Escopo: {scope} | Regioes: {len(regions)}")
    print(
        f"Memoria detectada -> RAM livre: {ram_gb if ram_gb is not None else 'n/d'} GB | "
        f"VRAM livre: {vram_free_gb if vram_free_gb is not None else 'n/d'} GB | "
        f"VRAM total: {vram_total_gb if vram_total_gb is not None else 'n/d'} GB"
    )
    print(f"Modo solicitado: {args.mode} | Modo efetivo: {effective_mode}")
    print(
        "Treino: "
        f"fine_tune={'sim' if fine_tune_enabled else 'nao'} | "
        f"trials={tuning_trials if fine_tune_enabled else 1} | "
        f"val_fraction={val_fraction:.2f} | temporal_validation={'sim' if temporal_validation else 'nao'}"
    )
    if args.resume:
        print("Resume: ativo")

    feature_path = settings.feature_dir / f"features_{scope}.csv"
    checkpoint = settings.model_dir / f"quantum_{scope}.pt"
    resume_path = settings.model_dir / f"quantum_{scope}.resume.pt"
    reuse_features = bool(args.reuse_features or args.resume)

    if effective_mode == "prithvi":
        model_name = "Prithvi-WxC-1.0-2300M-rollout + QuantumResidual"
    else:
        model_name = "PersistenceBaseline(MERRA-2) + QuantumResidual"

    if reuse_features and feature_path.exists():
        print("[1/3] Reutilizando features salvas...")
        frame = pd.read_csv(feature_path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        print(f"Features carregadas de: {feature_path} | linhas: {len(frame)}")
    else:
        print("[1/3] Extraindo features...")
        if effective_mode == "prithvi":
            print("Inicializando Prithvi-WxC rollout...")
            engine = PrithviForecastEngine(
                data_dir=settings.data_root,
                merra_input_dir=settings.merra_input_dir,
                climatology_dir=settings.merra_climatology_dir,
                model_name="large_rollout",
                device=settings.device,
            )

            available_variables = [var for var in variables if var in engine.surface_indexes]
            missing_variables = [var for var in variables if var not in engine.surface_indexes]
            if missing_variables:
                print(f"Aviso: variaveis nao suportadas pelo checkpoint atual e ignoradas: {missing_variables}")
            if not available_variables:
                raise RuntimeError("Nenhuma variavel solicitada e suportada pelo modelo.")
            variables = available_variables

            frame = build_training_frame(
                engine=engine,
                regions=regions,
                variables=variables,
                start=config["merra2"]["start"],
                end=config["merra2"]["end"],
                input_time_hours=int(config["merra2"]["input_time_hours"]),
                lead_time_hours=int(config["merra2"]["lead_time_hours"]),
                max_samples=args.max_samples,
            )
        else:
            print("Inicializando baseline de persistencia com MERRA-2 (modo leve)...")
            frame = build_training_frame_persistence(
                merra_input_dir=settings.merra_input_dir,
                regions=regions,
                variables=variables,
                start=config["merra2"]["start"],
                end=config["merra2"]["end"],
                step_hours=int(config["merra2"]["step_hours"]),
                lead_time_hours=int(config["merra2"]["lead_time_hours"]),
                max_samples=args.max_samples,
            )

        frame.to_csv(feature_path, index=False)
        print(f"Features salvas em: {feature_path}")

    if not args.skip_train:
        print("[2/3] Treinando corretor quantico...")
        train_kwargs = dict(
            frame=frame,
            variables=variables,
            output_path=checkpoint,
            epochs=int(training_cfg["epochs"]),
            batch_size=int(training_cfg["batch_size"]),
            learning_rate=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
            hidden_dim=int(training_cfg["hidden_dim"]),
            n_qubits=int(training_cfg["n_qubits"]),
            n_layers=int(training_cfg["n_layers"]),
            device=settings.device,
            val_fraction=val_fraction,
            temporal_validation=temporal_validation,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            resume=args.resume,
            resume_path=resume_path,
        )
        if fine_tune_enabled:
            result = tune_quantum_model(
                **train_kwargs,
                tuning_trials=tuning_trials,
                search_space=search_space,
            )
            model_name = f"{model_name} + FineTuned"
        else:
            result = train_quantum_model(**train_kwargs)
        print(f"Modelo salvo em: {result.checkpoint_path}")
        print(f"Train loss: {result.train_loss:.6f} | Val loss: {result.val_loss:.6f}")
        if result.metrics:
            print(
                f"Metricas val -> RMSE: {result.metrics.get('rmse', 0.0):.6f} | "
                f"MAE: {result.metrics.get('mae', 0.0):.6f}"
            )
        if result.hyperparameters:
            print(f"Hiperparametros finais: {result.hyperparameters}")
    elif not checkpoint.exists():
        raise RuntimeError(f"--skip-train informado, mas checkpoint nao existe: {checkpoint}")

    corrected = apply_quantum_correction(
        frame=frame,
        checkpoint_path=checkpoint,
        device=settings.device,
    )

    latest_timestamp = pd.Timestamp(corrected["timestamp"].max())
    latest_frame = corrected.loc[corrected["timestamp"] == latest_timestamp].copy()
    source_timestamp = latest_timestamp.to_pydatetime()
    if source_timestamp.tzinfo is None:
        source_timestamp = source_timestamp.replace(tzinfo=timezone.utc)

    for variable in variables:
        payload = latest_frame[["region_id", "region_name", "uf", "lat", "lon"]].copy()
        payload["value"] = latest_frame[f"corrected_{variable}"]
        save_forecast_snapshot(
            output_dir=settings.output_dir,
            scope=scope,
            variable=variable,
            forecast=payload,
            model_name=model_name,
            source_timestamp=source_timestamp,
        )

    print("[3/3] Previsoes exportadas.")
    print("Previsoes corrigidas salvas em data/outputs")


if __name__ == "__main__":
    main()
