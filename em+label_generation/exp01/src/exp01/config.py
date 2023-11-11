from typing import Dict, Literal, Optional, Tuple, Union, Sequence

from denoising_diffusion_pytorch import GaussianDiffusion, SimpleDataset, Unet, ZarrDataset, CellMapDatasets3Das2D
from pydantic import BaseModel, Field


class GaussianDiffusionConfig(BaseModel):
    timesteps: int = 1000
    sampling_timesteps: Optional[int] = None
    objective: str = "pred_v"
    beta_schedule: str = "sigmoid"
    ddim_sampling_eta: float = 0.0
    auto_normalize: bool = True
    offset_noise_strength: float = 0.0
    min_snr_loss_weight: bool = False
    min_snr_gamma: float = 5.0

    def get_constructor(self):
        return GaussianDiffusion


class TrainingConfig(BaseModel):
    train_batch_size: int
    train_lr: float
    train_num_steps: int
    gradient_accumulate_every: int
    ema_decay: float
    amp: bool
    calculate_fid: bool


class UnetConfig(BaseModel):
    dim: int
    channels: int
    dim_mults: Tuple[int, ...]

    def get_constructor(self):
        return Unet


class SimpleDataConfig(BaseModel):
    data_type: Literal["jpg", "jpeg", "png", "tiff"]
    data_dir: str

    def get_constructor(self):
        return SimpleDataset


class ZarrDataConfig(BaseModel):
    data_type: Literal["zarr"]
    data_dir: str

    def get_constructor(self):
        return ZarrDataset


class CellMapDatasets3Das2DConfig(BaseModel):
    data_type: Literal["cellmap3das2d"]
    data_paths: Sequence[str]
    class_list: Sequence[str]
    scale: Dict[Literal["x", "y", "z"], int]
    augment_horizontal_flip: bool = True
    augment_vertical_flip: bool = True
    annotation_paths: Union[None, Sequence[Union[str, None]]] = None
    crop_lists: Union[None, Sequence[Union[None, Sequence[str]]]] = None
    raw_datasets: Union[None, Sequence[str]] = None

    def get_constructor(self):
        return CellMapDatasets3Das2D


class TrackingConfig(BaseModel):
    experiment_name: str
    tracking_uri: str
    continue_run_id: Optional[str] = None
    run_name: Optional[str] = None


class ExperimentConfig(BaseModel):
    image_size: int
    architecture: UnetConfig  # turn this into union to add more architectures
    diffusion: GaussianDiffusionConfig  # turn this into union to add more architectures
    data: Union[SimpleDataConfig, ZarrDataConfig, CellMapDatasets3Das2DConfig] = Field(..., discriminator="data_type")
    training: TrainingConfig
    tracking: TrackingConfig
