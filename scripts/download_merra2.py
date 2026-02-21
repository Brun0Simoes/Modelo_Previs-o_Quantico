from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PrithviWxC.download import get_prithvi_wxc_climatology, get_prithvi_wxc_input
from previsao_tempo_quantico.settings import get_settings

HF_REPO_MERRA = "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download e preparacao de MERRA-2 para Prithvi-WxC")
    parser.add_argument("--start", required=True, help="Inicio ISO, ex: 2020-01-02T00:00:00")
    parser.add_argument("--end", required=True, help="Fim ISO, ex: 2020-01-05T00:00:00")
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--input-time-hours", type=int, default=6)
    parser.add_argument("--lead-time-hours", type=int, default=6)
    parser.add_argument(
        "--source",
        choices=["hf-public", "earthdata"],
        default="hf-public",
        help="Fonte dos dados MERRA-2: hf-public (sem credencial) ou earthdata (com credencial NASA).",
    )
    parser.add_argument(
        "--hf-repo",
        default=HF_REPO_MERRA,
        help="Repositorio Hugging Face com os arquivos MERRA-2 pre-processados.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Permite continuar mesmo com alguns arquivos faltando no intervalo.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.05,
        help="Percentual maximo de arquivos faltando permitido quando --allow-missing estiver ativo.",
    )
    return parser.parse_args()


def _all_steps_for_init(ts: pd.Timestamp, input_time_hours: int, lead_time_hours: int) -> list[np.datetime64]:
    init = np.datetime64(ts.to_datetime64())
    input_delta = np.timedelta64(input_time_hours, "h")
    input_times = [init - input_delta, init]
    output_times = init + np.arange(
        input_time_hours,
        lead_time_hours + 1,
        input_time_hours,
    ).astype("timedelta64[h]")
    return input_times + list(output_times)


def _download_from_hf_public(
    init_times: pd.DatetimeIndex,
    input_time_hours: int,
    lead_time_hours: int,
    hf_repo: str,
    allow_missing: bool = False,
    max_missing_ratio: float = 0.05,
) -> None:
    settings = get_settings()

    all_steps: list[np.datetime64] = []
    for ts in init_times:
        all_steps.extend(
            _all_steps_for_init(
                ts=ts,
                input_time_hours=input_time_hours,
                lead_time_hours=lead_time_hours,
            )
        )

    unique_days = sorted({pd.Timestamp(step).strftime("%Y%m%d") for step in all_steps})

    print(f"[HF] Repo: {hf_repo}")
    print(f"[HF] Dias necessarios: {len(unique_days)}")

    missing: list[str] = []
    downloaded = 0
    skipped = 0
    expected_files = [
        f"{prefix}_{day}.nc"
        for day in unique_days
        for prefix in ("MERRA2_sfc", "MERRA_pres")
    ]

    progress = tqdm(expected_files, total=len(expected_files), desc="Download MERRA-2 (HF)", unit="arquivo")
    for filename in progress:
        target = settings.merra_input_dir / filename
        nested = settings.merra_input_dir / "merra-2" / filename
        if target.exists() or nested.exists():
            skipped += 1
            progress.set_postfix(downloaded=downloaded, skipped=skipped, missing=len(missing))
            continue
        try:
            hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                subfolder="merra-2",
                local_dir=settings.merra_input_dir,
            )
            downloaded += 1
        except Exception:
            missing.append(filename)
        progress.set_postfix(downloaded=downloaded, skipped=skipped, missing=len(missing))

    total_expected = len(unique_days) * 2
    missing_unique = sorted(set(missing))
    missing_ratio = (len(missing_unique) / total_expected) if total_expected else 0.0

    print(
        f"[HF] Arquivos esperados: {total_expected} | baixados: {downloaded} | pulados: {skipped} | "
        f"faltantes unicos: {len(missing_unique)} ({missing_ratio:.2%})"
    )

    if missing_unique:
        if (not allow_missing) or (missing_ratio > max_missing_ratio):
            missing_text = ", ".join(missing_unique[:20])
            suffix = " ..." if len(missing_unique) > 20 else ""
            raise RuntimeError(
                "Arquivos MERRA-2 indisponiveis no Hugging Face para este intervalo. "
                f"Faltantes: {missing_text}{suffix}. "
                f"Taxa faltante: {missing_ratio:.2%} (limite: {max_missing_ratio:.2%})."
            )
        print(
            f"[HF] Aviso: seguindo com faltantes dentro do limite permitido ({max_missing_ratio:.2%})."
        )

    # O HF cria a subpasta "merra-2"; o dataloader espera arquivos diretos em "merra2".
    nested_dir = settings.merra_input_dir / "merra-2"
    if nested_dir.exists():
        for file in nested_dir.glob("*.nc"):
            target = settings.merra_input_dir / file.name
            if not target.exists():
                file.replace(target)

    get_prithvi_wxc_climatology(
        time_steps=all_steps,
        climatology_dir=settings.merra_input_dir.parent,
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if end < start:
        raise ValueError("--end deve ser maior ou igual a --start")

    init_times = pd.date_range(start=start, end=end, freq=f"{args.step_hours}h")
    if len(init_times) == 0:
        raise RuntimeError("Intervalo sem timestamps validos")

    print(f"Total de inicializacoes: {len(init_times)}")
    print(f"Saida pre-processada em: {settings.merra_input_dir}")

    if args.source == "hf-public":
        _download_from_hf_public(
            init_times=init_times,
            input_time_hours=args.input_time_hours,
            lead_time_hours=args.lead_time_hours,
            hf_repo=args.hf_repo,
            allow_missing=args.allow_missing,
            max_missing_ratio=args.max_missing_ratio,
        )
    else:
        for ts in init_times:
            init_np = np.datetime64(ts.to_datetime64())
            print(f"[EarthData] Preparando inicializacao {ts.isoformat()}")
            get_prithvi_wxc_input(
                time=init_np,
                input_time_step=args.input_time_hours,
                lead_time=args.lead_time_hours,
                input_data_dir=settings.merra_input_dir,
                download_dir=settings.raw_merra_dir,
            )

    print("Download e preparacao concluidos.")


if __name__ == "__main__":
    main()
