from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from PrithviWxC.configs import get_model_config
from PrithviWxC.dataloaders.merra2 import input_scalers, output_scalers, static_input_scalers
from PrithviWxC.dataloaders.merra2_rollout import Merra2RolloutDataset, preproc
from PrithviWxC.download import download_model_weights, get_prithvi_wxc_scaling_factors
from PrithviWxC.model import PrithviWxC
from PrithviWxC.rollout import rollout_iter


@dataclass(slots=True)
class RolloutSample:
    timestamp: datetime
    prediction: torch.Tensor
    target: torch.Tensor
    lats: np.ndarray
    lons: np.ndarray


class PrithviForecastEngine:
    def __init__(
        self,
        data_dir: Path,
        merra_input_dir: Path,
        climatology_dir: Path,
        model_name: str = "large_rollout",
        device: str = "cuda",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.merra_input_dir = Path(merra_input_dir)
        self.climatology_dir = Path(climatology_dir)
        self.model_name = model_name

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.use_fp16 = self.device.type == "cuda"

        self.padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}

        self.config = get_model_config(model_name, self.data_dir)
        self.model = self._load_model_memory_efficient()
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval().to(self.device)

        self.surface_indexes = {
            name: idx for idx, name in enumerate(self.config.surface_vars)
        }

    def _load_model_memory_efficient(self) -> PrithviWxC:
        cfg = self.config
        args = asdict(cfg)

        surface_vars = args.pop("surface_vars")
        static_surface_vars = args.pop("static_surface_vars")
        vertical_vars = args.pop("vertical_vars")
        levels = args.pop("levels")

        get_prithvi_wxc_scaling_factors(self.data_dir)
        surf_in_scal_path = self.data_dir / "climatology" / "musigma_surface.nc"
        vert_in_scal_path = self.data_dir / "climatology" / "musigma_vertical.nc"
        surf_out_scal_path = self.data_dir / "climatology" / "anomaly_variance_surface.nc"
        vert_out_scal_path = self.data_dir / "climatology" / "anomaly_variance_vertical.nc"

        in_mu, in_sig = input_scalers(
            surface_vars,
            vertical_vars,
            levels,
            surf_in_scal_path,
            vert_in_scal_path,
        )
        output_sig = output_scalers(
            surface_vars,
            vertical_vars,
            levels,
            surf_out_scal_path,
            vert_out_scal_path,
        )
        static_mu, static_sig = static_input_scalers(
            surf_in_scal_path,
            static_surface_vars,
        )

        args["input_scalers_mu"] = in_mu
        args["input_scalers_sigma"] = in_sig
        args["static_input_scalers_mu"] = static_mu
        args["static_input_scalers_sigma"] = static_sig
        args["static_input_scalers_epsilon"] = 0.0
        args["output_scalers"] = output_sig ** 0.5
        args["mask_ratio_inputs"] = 0.0
        args["mask_ratio_targets"] = 0.0
        args["checkpoint_encoder"] = list(range(2 * args["n_blocks_encoder"] + 1))
        args["checkpoint_decoder"] = list(range(2 * args["n_blocks_decoder"] + 1))

        model = PrithviWxC(**args)

        weights_path = download_model_weights(self.model_name, self.data_dir)

        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)

        del state_dict
        del checkpoint
        gc.collect()

        return model

    def build_dataset(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        input_time_hours: int,
        lead_time_hours: int,
    ) -> Merra2RolloutDataset:
        dataset = Merra2RolloutDataset(
            time_range=(start, end),
            input_time=-abs(input_time_hours),
            lead_time=lead_time_hours,
            data_path_surface=self.merra_input_dir,
            data_path_vertical=self.merra_input_dir,
            climatology_path_surface=self.climatology_dir,
            climatology_path_vertical=self.climatology_dir,
            surface_vars=self.config.surface_vars,
            static_surface_vars=self.config.static_surface_vars,
            vertical_vars=self.config.vertical_vars,
            levels=self.config.levels,
            positional_encoding=self.config.positional_encoding,
        )
        if len(dataset) == 0:
            raise RuntimeError("Nao ha amostras validas. Verifique intervalo e dados MERRA-2.")
        return dataset

    def iter_rollouts(
        self,
        start: str,
        end: str,
        input_time_hours: int,
        lead_time_hours: int,
        max_samples: int | None = None,
        show_progress: bool = True,
    ) -> Iterator[RolloutSample]:
        dataset = self.build_dataset(start, end, input_time_hours, lead_time_hours)

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: preproc(batch, self.padding),
        )

        total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        iterator = loader
        if show_progress:
            iterator = tqdm(
                loader,
                total=total,
                desc="Extraindo features Prithvi",
                unit="amostra",
            )

        for index, batch in enumerate(iterator):
            if max_samples is not None and index >= max_samples:
                break

            with torch.no_grad():
                for key, value in list(batch.items()):
                    if isinstance(value, torch.Tensor):
                        tensor = value.to(self.device)
                        if self.use_fp16 and tensor.is_floating_point():
                            tensor = tensor.to(dtype=torch.float16)
                        batch[key] = tensor

                prediction = rollout_iter(dataset.nsteps, self.model, batch)
                target = batch["ys"][:, dataset.nsteps - 1]

            sample_meta = dataset.samples[index][0]
            timestamp = pd.Timestamp(sample_meta[0]).to_pydatetime()

            prediction_cpu = prediction.detach().to("cpu")
            target_cpu = target.detach().to("cpu")

            lats = np.asarray(dataset.lats)
            lons = np.asarray(dataset.lons)

            yield RolloutSample(
                timestamp=timestamp,
                prediction=prediction_cpu,
                target=target_cpu,
                lats=lats,
                lons=lons,
            )
