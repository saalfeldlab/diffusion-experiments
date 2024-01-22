from typing import Dict, Literal, Optional, Tuple, Union, Sequence

from denoising_diffusion_pytorch import (
    GaussianDiffusion,
    SimpleDataset,
    Unet,
    ZarrDataset,
    CellMapDatasets3Das2D,
    CellMapDataset3Das2D,
    PreProcessOptions,
    InferenceSaver,
)
from pydantic import BaseModel, Field, validator


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
    data_paths: Sequence[str]
    class_list: Sequence[str]
    scale: Dict[Literal["x", "y", "z"], int]
    augment_horizontal_flip: bool = True
    augment_vertical_flip: bool = True
    allow_single_class_crops: Union[None, Sequence[Union[str, None]]] = None
    annotation_path: Optional[str] = None
    crop_list: Optional[Sequence[str]] = None
    raw_dataset: Optional[str] = None
    dask_workers: int = 0
    pre_load: bool = False
    contrast_adjust: bool = True
    include_raw: bool = True

    def get_constructor(self):
        return CellMapDataset3Das2D


class CellMapDatasets3Das2DConfig(BaseModel):
    data_type: Literal["cellmap3das2d"]
    data_paths: Sequence[str]
    class_list: Sequence[str]
    scale: Dict[Literal["x", "y", "z"], int]
    augment_horizontal_flip: bool = True
    augment_vertical_flip: bool = True
    annotation_paths: Union[None, Sequence[Union[str, None]]] = None
    allow_single_class_crops: Union[None, Sequence[Union[str, None]]] = None
    crop_lists: Union[None, Sequence[Union[None, Sequence[str]]]] = None
    raw_datasets: Union[None, Sequence[None, str]] = None
    dask_workers: int = 0
    pre_load: bool = False
    contrast_adjust: bool = True
    include_raw: bool = True

    def get_constructor(self):
        return CellMapDatasets3Das2D


class TrackingConfig(BaseModel):
    experiment_name: str
    tracking_uri: str
    continue_run_id: Optional[str] = None
    run_name: Optional[str] = None


class InferenceSaverConfig(BaseModel):
    channel_assignment: Dict[str, Tuple[Tuple[int, int, int], Sequence[Union[None, PreProcessOptions]]]]
    sample_digits: int = 5
    
    def get_constructor(self):
        return InferenceSaver
    
    @validator("channel_assignment", pre=True, always=True)
    def convert_enum_from_str(cls, value):
        if isinstance(value, dict):
            for key, dict_value in value.items():
                if (
                    isinstance(dict_value, (list, tuple))
                    and len(dict_value) == 2
                    and isinstance(dict_value[1], (list, tuple))
                ):
                    processed_options = []
                    print(dict_value[1])
                    for option in dict_value[1]:
                        if isinstance(option, str):
                            try:
                                processed_options.append(PreProcessOptions[option])
                            except KeyError:
                                raise ValueError(f"Invalid preprocess option: {option}"
                                                 )
                        else:
                            processed_options.append(option)
                    value[key] = (dict_value[0], processed_options)
        return value

class ExperimentConfig(BaseModel):
    image_size: int
    architecture: UnetConfig  # turn this into union to add more architectures
    diffusion: GaussianDiffusionConfig  # turn this into union to add more architectures
    data: Union[SimpleDataConfig, ZarrDataConfig, CellMapDatasets3Das2DConfig, CellMapDataset3Das2DConfig] = Field(
        ..., discriminator="data_type"
    )
    training: TrainingConfig
    tracking: TrackingConfig
    inference_saver: InferenceSaverConfig
