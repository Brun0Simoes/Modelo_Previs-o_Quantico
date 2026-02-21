from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    data_root: Path
    model_cache_dir: Path
    hf_home: Path
    preferred_drive: str
    api_host: str
    api_port: int
    device: str

    @property
    def merra_input_dir(self) -> Path:
        return self.data_root / "prithvi_input" / "merra2"

    @property
    def merra_climatology_dir(self) -> Path:
        return self.data_root / "prithvi_input" / "climatology"

    @property
    def raw_merra_dir(self) -> Path:
        return self.data_root / "raw_merra"

    @property
    def feature_dir(self) -> Path:
        return self.data_root / "features"

    @property
    def model_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def output_dir(self) -> Path:
        return self.data_root / "outputs"

    @property
    def reference_dir(self) -> Path:
        return self.data_root / "reference"

    def ensure_directories(self) -> None:
        folders = [
            self.data_root,
            self.model_cache_dir,
            self.hf_home,
            self.raw_merra_dir,
            self.merra_input_dir,
            self.merra_climatology_dir,
            self.feature_dir,
            self.model_dir,
            self.output_dir,
            self.reference_dir,
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    def validate_storage_drive(self) -> None:
        if os.name != "nt":
            return

        drive = self.data_root.drive.rstrip(":").upper()
        expected = self.preferred_drive.rstrip(":").upper()
        if expected and drive and drive != expected:
            raise RuntimeError(
                f"Dados estao em {self.data_root} (drive {drive}), mas o esperado e {expected}."
            )


def get_settings() -> AppSettings:
    default_root = Path(__file__).resolve().parents[2]
    project_root_env = os.getenv("PROJECT_ROOT")
    project_root = Path(project_root_env) if project_root_env else default_root

    if os.name == "nt" and not project_root.exists():
        project_root = default_root

    data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
    model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", project_root / "cache" / "huggingface"))
    hf_home = Path(os.getenv("HF_HOME", model_cache_dir))

    settings = AppSettings(
        project_root=project_root,
        data_root=data_root,
        model_cache_dir=model_cache_dir,
        hf_home=hf_home,
        preferred_drive=os.getenv("PREFERRED_DRIVE", "E:"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        device=os.getenv("DEVICE", "cuda"),
    )

    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(settings.model_cache_dir / "hub"))

    settings.ensure_directories()
    settings.validate_storage_drive()
    return settings
