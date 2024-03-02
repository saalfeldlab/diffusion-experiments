from typing import Dict, Literal, Optional, Sequence, Tuple, Union

from denoising_diffusion_pytorch import (
    CellMapDataset3Das2D,
    CellMapDatasets3Das2D,
    BaselineSegmentation,
    BaselineSegmentationTrainer,
    LabelRepresentation,
    ProcessOptions,
    ProcessOptionsNames,
    SegmentationActivationNames,
    SegmentationMetrics,
    SegmentationMetricsNames,
    SampleExporter,
    SimpleDataset,
    Unet,
    ZarrDataset,
    RawChannelOptions,
)
from pydantic import BaseModel, Field


class BaselineSegmentationConfig(BaseModel):
    loss_fn: SegmentationMetricsNames = "CROSS_ENTROPY"
    activation: Union[None,SegmentationActivationNames] = None

    def get_constructor(self):
        return BaselineSegmentation


class TrainingConfig(BaseModel):
    train_batch_size: int
    validation_criteria: Union[None, Sequence[SegmentationMetricsNames]] = ["CROSS_ENTROPY"]
    validation_batch_size: int
    gradient_accumulate_every: int
    train_lr: float
    train_num_steps: int
    dataloader_nworkers: int = 88
    persistent_workers: bool
    prefetch_factor: int
    save_and_sample_every: int


class UnetConfig(BaseModel):
    dim: int
    channels: int
    dim_mults: Tuple[int, ...]
    out_dim: int

    def get_constructor(self):
        return Unet


class CellMapDataset3Das2DConfig(BaseModel):
    data_type: Literal["cellmap3das2d_single"]
    data_path: str
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
    raw_channel: RawChannelOptions = "first"
    label_representation: LabelRepresentation = "class_ids"
    random_crop: bool = True

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
    raw_datasets: Union[None, Sequence[Union[None, str]]] = None
    dask_workers: int = 0
    pre_load: bool = False
    contrast_adjust: bool = True
    raw_channel: RawChannelOptions = "first"
    label_representation: LabelRepresentation = "class_ids"
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
    segmentation: BaselineSegmentationConfig
    training_data: Union[CellMapDatasets3Das2DConfig, CellMapDataset3Das2DConfig] = Field(
        ..., discriminator="data_type"
    )
    validation_data: Union[CellMapDatasets3Das2DConfig, CellMapDataset3Das2DConfig] = Field(
        ..., discriminator="data_type"
    )
    training: TrainingConfig
    tracking: TrackingConfig
    loader_exporter: SampleExporterConfig
    prediction_exporter: SampleExporterConfig


class InferenceConfig(BaseModel):
    exporter: SampleExporterConfig
    checkpoint: int
    eval_batch_size: int
    num_samples: int
    experiment_run_id: str
