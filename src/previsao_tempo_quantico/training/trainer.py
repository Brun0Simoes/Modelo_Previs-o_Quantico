from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from previsao_tempo_quantico.models.quantum_residual import QuantumResidualRegressor


@dataclass(slots=True)
class TrainingResult:
    checkpoint_path: Path
    train_loss: float
    val_loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, float | int] = field(default_factory=dict)


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


def _adaptive_batch_size(requested_batch_size: int, torch_device: torch.device) -> int:
    effective = max(1, int(requested_batch_size))

    if torch_device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024.0**3)
        total_gb = total_bytes / (1024.0**3)

        if free_gb < 2.0:
            effective = min(effective, 4)
        elif free_gb < 4.0:
            effective = min(effective, 8)
        elif free_gb < 8.0:
            effective = min(effective, 16)

        print(
            f"[mem] GPU detectada: livre {free_gb:.2f} GB / total {total_gb:.2f} GB | batch={effective}"
        )
        return effective

    ram_gb = _available_ram_gb()
    if ram_gb is not None:
        if ram_gb < 4.0:
            effective = min(effective, 8)
        elif ram_gb < 8.0:
            effective = min(effective, 16)
        elif ram_gb < 16.0:
            effective = min(effective, 32)
        print(f"[mem] RAM disponivel: {ram_gb:.2f} GB | batch={effective}")
    else:
        print(f"[mem] RAM nao detectada | batch={effective}")

    return effective


def _feature_columns(variables: list[str]) -> tuple[list[str], list[str], list[str]]:
    variables = [var.upper() for var in variables]
    input_columns = [
        "lat",
        "lon",
        "sin_hod",
        "cos_hod",
        "sin_doy",
        "cos_doy",
    ] + [f"pred_{var}" for var in variables]
    baseline_columns = [f"pred_{var}" for var in variables]
    target_columns = [f"target_{var}" for var in variables]
    return input_columns, baseline_columns, target_columns


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_frame(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    if not required_columns:
        return frame.copy()
    subset = frame.loc[:, required_columns].replace([np.inf, -np.inf], np.nan)
    valid_mask = subset.notna().all(axis=1)
    clean = frame.loc[valid_mask].copy()
    if clean.empty:
        raise RuntimeError("Nenhuma linha valida apos limpeza de NaN/Inf.")
    return clean


def _temporal_split_indices(
    frame: pd.DataFrame,
    val_fraction: float,
    temporal_validation: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if temporal_validation and "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        unique_ts = np.sort(ts.dropna().unique())
        if len(unique_ts) >= 2:
            split_idx = int(len(unique_ts) * (1.0 - val_fraction))
            split_idx = max(1, min(split_idx, len(unique_ts) - 1))
            train_ts = set(unique_ts[:split_idx])
            val_ts = set(unique_ts[split_idx:])
            train_idx = np.flatnonzero(ts.isin(train_ts).to_numpy())
            val_idx = np.flatnonzero(ts.isin(val_ts).to_numpy())
            if len(train_idx) > 0 and len(val_idx) > 0:
                return train_idx, val_idx

    all_idx = np.arange(len(frame))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def _compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(torch.mean(torch.abs(err)).item())
    rmse = float(torch.sqrt(torch.mean(err**2)).item())
    return {"mae": mae, "rmse": rmse}


def _to_cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _optimizer_state_to_cpu(optimizer_state: dict) -> dict:
    state = optimizer_state.get("state", {})
    param_groups = optimizer_state.get("param_groups", [])
    out_state: dict = {}

    for key, value in state.items():
        item = {}
        for sub_key, sub_value in value.items():
            if torch.is_tensor(sub_value):
                item[sub_key] = sub_value.detach().cpu()
            else:
                item[sub_key] = sub_value
        out_state[key] = item

    return {"state": out_state, "param_groups": param_groups}


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _load_resume_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _run_training(
    model: QuantumResidualRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    torch_device: torch.device,
    patience: int,
    min_epochs: int,
    show_batch_progress: bool,
    epoch_desc: str,
    resume_state: dict | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> tuple[dict[str, torch.Tensor], float, float, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = _to_cpu_state_dict(model.state_dict())
    best_train = float("inf")
    best_val = float("inf")
    best_metrics: dict[str, float] = {"mae": float("inf"), "rmse": float("inf")}
    wait = 0
    start_epoch = 1

    if resume_state:
        resume_model = resume_state.get("model_state")
        if isinstance(resume_model, dict):
            model.load_state_dict(resume_model, strict=True)

        resume_opt = resume_state.get("optimizer_state")
        if isinstance(resume_opt, dict):
            optimizer.load_state_dict(resume_opt)

        resume_best = resume_state.get("best_state")
        if isinstance(resume_best, dict):
            best_state = resume_best

        best_train = float(resume_state.get("best_train", best_train))
        best_val = float(resume_state.get("best_val", best_val))
        loaded_metrics = resume_state.get("best_metrics")
        if isinstance(loaded_metrics, dict):
            best_metrics = {
                "mae": float(loaded_metrics.get("mae", best_metrics["mae"])),
                "rmse": float(loaded_metrics.get("rmse", best_metrics["rmse"])),
            }
        wait = int(resume_state.get("wait", wait))
        start_epoch = int(resume_state.get("epoch_completed", 0)) + 1

    if start_epoch > epochs:
        return best_state, best_train, best_val, best_metrics

    epoch_bar = tqdm(
        range(start_epoch, epochs + 1),
        total=epochs,
        initial=max(0, start_epoch - 1),
        desc=epoch_desc,
        unit="epoca",
    )
    for epoch in epoch_bar:
        model.train()
        train_losses: list[float] = []

        train_iter = (
            tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoca {epoch}/{epochs} treino",
                unit="lote",
                leave=False,
            )
            if show_batch_progress
            else train_loader
        )

        for x_batch, baseline_batch, y_batch in train_iter:
            x_batch = x_batch.to(torch_device)
            baseline_batch = baseline_batch.to(torch_device)
            y_batch = y_batch.to(torch_device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_batch, baseline_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        preds: list[torch.Tensor] = []
        trues: list[torch.Tensor] = []

        with torch.no_grad():
            val_iter = (
                tqdm(
                    val_loader,
                    total=len(val_loader),
                    desc=f"Epoca {epoch}/{epochs} valid",
                    unit="lote",
                    leave=False,
                )
                if show_batch_progress
                else val_loader
            )
            for x_batch, baseline_batch, y_batch in val_iter:
                x_batch = x_batch.to(torch_device)
                baseline_batch = baseline_batch.to(torch_device)
                y_batch = y_batch.to(torch_device)
                y_hat = model(x_batch, baseline_batch)
                loss = criterion(y_hat, y_batch)
                val_losses.append(float(loss.detach().cpu().item()))
                preds.append(y_hat.detach().cpu())
                trues.append(y_batch.detach().cpu())

        cur_train = float(np.mean(train_losses)) if train_losses else 0.0
        cur_val = float(np.mean(val_losses)) if val_losses else 0.0
        cur_metrics = (
            _compute_metrics(torch.cat(trues, dim=0), torch.cat(preds, dim=0))
            if preds and trues
            else {"mae": 0.0, "rmse": 0.0}
        )

        improved = cur_val < (best_val - 1e-8)
        if improved:
            best_val = cur_val
            best_train = cur_train
            best_metrics = cur_metrics
            best_state = _to_cpu_state_dict(model.state_dict())
            wait = 0
        else:
            wait += 1

        progress_pct = (epoch / epochs) * 100.0
        epoch_bar.set_postfix(
            train=f"{cur_train:.4f}",
            val=f"{cur_val:.4f}",
            best=f"{best_val:.4f}",
            rmse=f"{cur_metrics['rmse']:.4f}",
            done=f"{progress_pct:.1f}%",
        )

        if epoch_callback:
            epoch_callback(
                {
                    "epoch_completed": int(epoch),
                    "model_state": _to_cpu_state_dict(model.state_dict()),
                    "optimizer_state": _optimizer_state_to_cpu(optimizer.state_dict()),
                    "best_state": best_state,
                    "best_train": float(best_train),
                    "best_val": float(best_val),
                    "best_metrics": best_metrics,
                    "wait": int(wait),
                }
            )

        if epoch >= max(1, min_epochs) and wait >= max(1, patience):
            break

    return best_state, best_train, best_val, best_metrics


def _fit_model(
    frame: pd.DataFrame,
    variables: list[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    n_qubits: int,
    n_layers: int,
    device: str,
    val_fraction: float,
    temporal_validation: bool,
    seed: int,
    early_stopping_patience: int,
    min_epochs: int,
    show_batch_progress: bool,
    epoch_desc: str,
    resume_state: dict | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> dict:
    input_columns, baseline_columns, target_columns = _feature_columns(variables)
    required_columns = input_columns + baseline_columns + target_columns
    frame_clean = _sanitize_frame(frame, required_columns)

    if len(frame_clean) < 32:
        raise RuntimeError("Conjunto de treino muito pequeno. Minimo recomendado: 32 amostras.")

    scaler = StandardScaler()
    features = scaler.fit_transform(frame_clean[input_columns].to_numpy(dtype=np.float32))
    baseline = frame_clean[baseline_columns].to_numpy(dtype=np.float32)
    targets = frame_clean[target_columns].to_numpy(dtype=np.float32)

    x_tensor = torch.tensor(features, dtype=torch.float32)
    baseline_tensor = torch.tensor(baseline, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)

    train_idx, val_idx = _temporal_split_indices(
        frame=frame_clean,
        val_fraction=val_fraction,
        temporal_validation=temporal_validation,
        seed=seed,
    )

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Falha no split treino/validacao.")

    train_set = TensorDataset(x_tensor[train_idx], baseline_tensor[train_idx], y_tensor[train_idx])
    val_set = TensorDataset(x_tensor[val_idx], baseline_tensor[val_idx], y_tensor[val_idx])

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    effective_batch_size = _adaptive_batch_size(batch_size, torch_device)

    train_loader = DataLoader(train_set, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=effective_batch_size, shuffle=False)

    model = QuantumResidualRegressor(
        input_dim=x_tensor.shape[1],
        output_dim=y_tensor.shape[1],
        n_qubits=n_qubits,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    ).to(torch_device)

    best_state, best_train, best_val, best_metrics = _run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        torch_device=torch_device,
        patience=early_stopping_patience,
        min_epochs=min_epochs,
        show_batch_progress=show_batch_progress,
        epoch_desc=epoch_desc,
        resume_state=resume_state,
        epoch_callback=epoch_callback,
    )

    return {
        "best_state": best_state,
        "train_loss": best_train,
        "val_loss": best_val,
        "metrics": best_metrics,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_columns": input_columns,
        "baseline_columns": baseline_columns,
        "target_columns": target_columns,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "total": int(len(frame_clean)),
        },
    }


def _save_checkpoint(
    output_path: Path,
    fit_payload: dict,
    n_qubits: int,
    n_layers: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    extra: dict | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": fit_payload["best_state"],
        "input_columns": fit_payload["input_columns"],
        "baseline_columns": fit_payload["baseline_columns"],
        "target_columns": fit_payload["target_columns"],
        "scaler_mean": fit_payload["scaler_mean"],
        "scaler_scale": fit_payload["scaler_scale"],
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_loss": fit_payload["train_loss"],
        "val_loss": fit_payload["val_loss"],
        "metrics": fit_payload["metrics"],
        "split_sizes": fit_payload["split_sizes"],
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)


def train_quantum_model(
    frame: pd.DataFrame,
    variables: list[str],
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 16,
    n_qubits: int = 6,
    n_layers: int = 3,
    device: str = "cpu",
    val_fraction: float = 0.2,
    temporal_validation: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 5,
    min_epochs: int = 5,
    resume: bool = False,
    resume_path: Path | None = None,
) -> TrainingResult:
    _set_seed(seed)
    resume_file = resume_path or output_path.with_suffix(".resume.pt")

    resume_state: dict | None = None
    if resume:
        payload = _load_resume_payload(resume_file)
        if payload and payload.get("kind") == "train_resume":
            resume_state = payload.get("state") if isinstance(payload.get("state"), dict) else None
            if resume_state:
                done_epoch = int(resume_state.get("epoch_completed", 0))
                print(f"[resume] retomando treino simples no epoch {done_epoch + 1}.")

    def _on_epoch(state: dict) -> None:
        _atomic_torch_save(
            resume_file,
            {
                "kind": "train_resume",
                "schema": 1,
                "state": state,
                "config": {
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "hidden_dim": int(hidden_dim),
                    "n_qubits": int(n_qubits),
                    "n_layers": int(n_layers),
                    "seed": int(seed),
                },
            },
        )

    fit_payload = _fit_model(
        frame=frame,
        variables=variables,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        device=device,
        val_fraction=val_fraction,
        temporal_validation=temporal_validation,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        min_epochs=min_epochs,
        show_batch_progress=True,
        epoch_desc="Treino quantico",
        resume_state=resume_state,
        epoch_callback=_on_epoch,
    )

    _save_checkpoint(
        output_path=output_path,
        fit_payload=fit_payload,
        n_qubits=n_qubits,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
    )

    if resume_file.exists():
        try:
            resume_file.unlink()
        except Exception:
            pass

    return TrainingResult(
        checkpoint_path=output_path,
        train_loss=float(fit_payload["train_loss"]),
        val_loss=float(fit_payload["val_loss"]),
        metrics=fit_payload["metrics"],
        hyperparameters={
            "hidden_dim": hidden_dim,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
        },
    )


def tune_quantum_model(
    frame: pd.DataFrame,
    variables: list[str],
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 16,
    n_qubits: int = 6,
    n_layers: int = 3,
    device: str = "cpu",
    val_fraction: float = 0.2,
    temporal_validation: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 5,
    min_epochs: int = 5,
    tuning_trials: int = 8,
    search_space: dict[str, list[float | int]] | None = None,
    resume: bool = False,
    resume_path: Path | None = None,
) -> TrainingResult:
    _set_seed(seed)
    search = search_space or {
        "hidden_dim": [hidden_dim, 24, 32],
        "n_qubits": [max(2, n_qubits - 2), n_qubits, n_qubits + 2],
        "n_layers": [max(1, n_layers - 1), n_layers, n_layers + 1],
        "learning_rate": [learning_rate / 2.0, learning_rate, learning_rate * 2.0],
        "weight_decay": [weight_decay / 2.0, weight_decay, weight_decay * 5.0],
    }

    resume_file = resume_path or output_path.with_suffix(".resume.pt")

    trials = max(1, int(tuning_trials))
    sampled_params: list[dict[str, float | int]] = [
        {
            "hidden_dim": int(hidden_dim),
            "n_qubits": int(n_qubits),
            "n_layers": int(n_layers),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        }
    ]
    seen = {
        (
            int(hidden_dim),
            int(n_qubits),
            int(n_layers),
            float(learning_rate),
            float(weight_decay),
        )
    }

    all_combos = []
    for hidden, qubits, layers, lr, wd in product(
        search["hidden_dim"],
        search["n_qubits"],
        search["n_layers"],
        search["learning_rate"],
        search["weight_decay"],
    ):
        candidate = {
            "hidden_dim": int(hidden),
            "n_qubits": int(max(2, qubits)),
            "n_layers": int(max(1, layers)),
            "learning_rate": float(lr),
            "weight_decay": float(max(0.0, wd)),
        }
        all_combos.append(candidate)

    random.shuffle(all_combos)
    for candidate in all_combos:
        if len(sampled_params) >= trials:
            break
        key = (
            candidate["hidden_dim"],
            candidate["n_qubits"],
            candidate["n_layers"],
            candidate["learning_rate"],
            candidate["weight_decay"],
        )
        if key in seen:
            continue
        seen.add(key)
        sampled_params.append(candidate)

    best_trial_idx = -1
    best_payload: dict | None = None
    best_val = float("inf")
    best_params: dict[str, float | int] = sampled_params[0]
    trial_history: list[dict] = []
    next_trial_idx = 1
    current_trial_resume: dict | None = None

    if resume:
        payload = _load_resume_payload(resume_file)
        if payload and payload.get("kind") == "tune_resume":
            saved_sampled = payload.get("sampled_params")
            if isinstance(saved_sampled, list) and saved_sampled:
                sampled_params = saved_sampled

            saved_history = payload.get("trial_history")
            if isinstance(saved_history, list):
                trial_history = saved_history

            best_val = float(payload.get("best_val", best_val))
            best_trial_idx = int(payload.get("best_trial_idx", best_trial_idx))

            saved_best_payload = payload.get("best_payload")
            if isinstance(saved_best_payload, dict):
                best_payload = saved_best_payload

            saved_best_params = payload.get("best_params")
            if isinstance(saved_best_params, dict):
                best_params = saved_best_params

            next_trial_idx = max(1, int(payload.get("next_trial_idx", 1)))

            saved_current = payload.get("current_trial")
            if isinstance(saved_current, dict):
                current_trial_resume = saved_current

            print(f"[resume] retomando fine tuning no trial {next_trial_idx}/{len(sampled_params)}.")

    def _save_tune_resume(next_idx: int, current_trial: dict | None) -> None:
        _atomic_torch_save(
            resume_file,
            {
                "kind": "tune_resume",
                "schema": 1,
                "sampled_params": sampled_params,
                "trial_history": trial_history,
                "best_val": float(best_val),
                "best_trial_idx": int(best_trial_idx),
                "best_params": best_params,
                "best_payload": best_payload,
                "next_trial_idx": int(next_idx),
                "current_trial": current_trial,
                "config": {
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "seed": int(seed),
                },
            },
        )

    remaining_trials = [
        (idx, trial)
        for idx, trial in enumerate(sampled_params, start=1)
        if idx >= next_trial_idx
    ]

    trial_bar = tqdm(
        remaining_trials,
        total=len(sampled_params),
        initial=min(len(sampled_params), max(0, next_trial_idx - 1)),
        desc="Fine tuning",
        unit="trial",
    )
    for idx, trial in trial_bar:
        epoch_desc = (
            f"Treino trial {idx}/{len(sampled_params)} "
            f"[q={trial['n_qubits']},l={trial['n_layers']},h={trial['hidden_dim']}]"
        )

        resume_state = None
        if current_trial_resume and int(current_trial_resume.get("idx", -1)) == idx:
            state = current_trial_resume.get("state")
            if isinstance(state, dict):
                resume_state = state
                print(
                    f"[resume] trial {idx}: retomando no epoch "
                    f"{int(resume_state.get('epoch_completed', 0)) + 1}."
                )

        def _on_trial_epoch(state: dict, trial_idx: int = idx, trial_cfg: dict = trial) -> None:
            _save_tune_resume(
                next_idx=trial_idx,
                current_trial={
                    "idx": int(trial_idx),
                    "trial": trial_cfg,
                    "state": state,
                },
            )

        fit_payload = _fit_model(
            frame=frame,
            variables=variables,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=float(trial["learning_rate"]),
            weight_decay=float(trial["weight_decay"]),
            hidden_dim=int(trial["hidden_dim"]),
            n_qubits=int(trial["n_qubits"]),
            n_layers=int(trial["n_layers"]),
            device=device,
            val_fraction=val_fraction,
            temporal_validation=temporal_validation,
            seed=seed + idx,
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            show_batch_progress=False,
            epoch_desc=epoch_desc,
            resume_state=resume_state,
            epoch_callback=_on_trial_epoch,
        )

        val_loss = float(fit_payload["val_loss"])
        trial_metrics = {
            "trial": idx,
            "val_loss": val_loss,
            "train_loss": float(fit_payload["train_loss"]),
            "rmse": float(fit_payload["metrics"]["rmse"]),
            "mae": float(fit_payload["metrics"]["mae"]),
            **trial,
        }
        trial_history.append(trial_metrics)

        if val_loss < best_val:
            best_val = val_loss
            best_payload = fit_payload
            best_trial_idx = idx
            best_params = trial

        _save_tune_resume(next_idx=idx + 1, current_trial=None)
        current_trial_resume = None
        trial_bar.set_postfix(best=f"{best_val:.4f}", trial=f"{idx}/{len(sampled_params)}")

    if best_payload is None:
        raise RuntimeError("Fine tuning nao produziu nenhum resultado.")

    _save_checkpoint(
        output_path=output_path,
        fit_payload=best_payload,
        n_qubits=int(best_params["n_qubits"]),
        n_layers=int(best_params["n_layers"]),
        hidden_dim=int(best_params["hidden_dim"]),
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(best_params["weight_decay"]),
        epochs=epochs,
        batch_size=batch_size,
        extra={
            "tuning": {
                "enabled": True,
                "trials": len(sampled_params),
                "best_trial": best_trial_idx,
                "history": trial_history,
            }
        },
    )

    if resume_file.exists():
        try:
            resume_file.unlink()
        except Exception:
            pass

    return TrainingResult(
        checkpoint_path=output_path,
        train_loss=float(best_payload["train_loss"]),
        val_loss=float(best_payload["val_loss"]),
        metrics=best_payload["metrics"],
        hyperparameters={
            "hidden_dim": int(best_params["hidden_dim"]),
            "n_qubits": int(best_params["n_qubits"]),
            "n_layers": int(best_params["n_layers"]),
            "learning_rate": float(best_params["learning_rate"]),
            "weight_decay": float(best_params["weight_decay"]),
            "batch_size": batch_size,
            "epochs": epochs,
            "best_trial": best_trial_idx,
            "trials": len(sampled_params),
        },
    )


def apply_quantum_correction(
    frame: pd.DataFrame,
    checkpoint_path: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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

    features = frame[input_columns].to_numpy(dtype=np.float32)
    features_scaled = (features - scaler_mean) / np.where(scaler_scale == 0.0, 1.0, scaler_scale)

    baseline = frame[baseline_columns].to_numpy(dtype=np.float32)

    with torch.no_grad():
        x_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=torch_device)
        b_tensor = torch.tensor(baseline, dtype=torch.float32, device=torch_device)
        corrected = model(x_tensor, b_tensor).cpu().numpy()

    out = frame.copy()
    for i, column in enumerate(target_columns):
        variable = column.replace("target_", "")
        out[f"corrected_{variable}"] = corrected[:, i]

    return out
