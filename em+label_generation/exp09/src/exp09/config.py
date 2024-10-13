from typing import Dict, Literal, Optional, Sequence, Tuple, Union

from denoising_diffusion_pytorch import (
    CellMapDataset3Das2D,
    CellMapDatasets3Das2D,
    GaussianDiffusion,
    ProcessOptions,
    ProcessOptionsNames,
    SampleExporter,
    SimpleDataset,
    Unet,
    ZarrDataset,
    RawChannelOptions,
    LabelRepresentation
)
from pydantic import BaseModel, Field


class DataInfo(BaseModel):
    contrast: Sequence[Union[int, float]]
    crop_group: str
    crops: Sequence[str]
    raw: Optional[str] = None

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
    channel_weights: Union[None, Sequence[float]] = None

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
    dataloader_nworkers: int = 88
    persistent_workers: bool = True
    prefetch_factor: int = 2
    save_and_sample_every: int = 1000


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


class CellMapDataset3Das2DConfig(BaseModel):
    data_type: Literal["cellmap3das2d_single"]
    dataname: str
    datainfo: DataInfo
    class_list: Sequence[str]
    scale: Dict[Literal["x", "y", "z"], int]
    augment_horizontal_flip: bool = True
    augment_vertical_flip: bool = True
    dask_workers: int = 0
    pre_load: bool = False
    contrast_adjust: bool = True
    raw_channel: RawChannelOptions = "append"
    label_representation: LabelRepresentation = "binary"
    random_crop: bool = True

    def get_constructor(self):
        return CellMapDataset3Das2D


class CellMapDatasets3Das2DConfig(BaseModel):
    data_type: Literal["cellmap3das2d"]
    data_config: str
    class_list: Sequence[str]
    scale: Dict[Literal["x", "y", "z"], int]
    augment_horizontal_flip: bool = True
    augment_vertical_flip: bool = True
    dask_workers: int = 0
    pre_load: bool = False
    contrast_adjust: bool = True
    raw_channel: RawChannelOptions = "append"
    label_representation: LabelRepresentation = "binary"
    random_crop: bool = True

    def get_constructor(self):
        return CellMapDatasets3Das2D


class TrackingConfig(BaseModel):
    experiment_name: str
    tracking_uri: str
    continue_run_id: Optional[str] = None
    run_name: Optional[str] = None


class SampleExporterConfig(BaseModel):
    channel_assignment: Dict[str, Tuple[Tuple[int, int, int], Sequence[Union[None, ProcessOptionsNames]]]]
    sample_digits: int = 5
    file_format: Literal[".zarr", ".png"] = ".zarr"
    sample_batch_size: int = 1
    colors: Optional[Sequence[Union[Tuple[int, int, int], Sequence[Tuple[float, float, float]]]]] = None
    threshold: int = 0

    def get_constructor(self):
        return SampleExporter


class ExperimentConfig(BaseModel):
    image_size: int
    architecture: UnetConfig  # turn this into union to add more architectures
    diffusion: GaussianDiffusionConfig  # turn this into union to add more architectures
    data: Union[SimpleDataConfig, ZarrDataConfig, CellMapDatasets3Das2DConfig, CellMapDataset3Das2DConfig] = Field(
        ..., discriminator="data_type"
    )
    training: TrainingConfig
    tracking: TrackingConfig
    exporter: SampleExporterConfig


class InferenceConfig(BaseModel):
    exporter: SampleExporterConfig
    checkpoint: int
    eval_batch_size: int
    num_samples: int
    experiment_run_id: str
