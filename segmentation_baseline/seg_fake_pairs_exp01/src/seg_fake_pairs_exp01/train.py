import os
from urllib.parse import urlparse, urlunparse

import mlflow
import yaml
from denoising_diffusion_pytorch import (
    Unet,
    CellMapDataset3Das2D,
    CellMapDatasets3Das2D,
    BaselineSegmentation,
    BaselineSegmentationTrainer,
    SampleExporter,
)

from seg_fake_pairs_exp01.config import (
    ExperimentConfig,
    TrackingConfig,
)
from seg_fake_pairs_exp01.utility import flatten_dict, get_repo_and_commit_cwd
import warnings

warnings.filterwarnings("ignore", module="pydantic_ome_ngff")  # line104


def track(config: TrackingConfig):
    parsed_uri = urlparse(config.tracking_uri)
    if parsed_uri.scheme == "file":
        print(os.path.exists(parsed_uri.path))  # this is just accessing it for autofs
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    if config.continue_run_id is None:
        with mlflow.start_run(run_name=config.run_name) as run:
            return run.info.run_id
    else:
        #    with mlflow.start_run(run_id = config.continue_run_id) as run:
        return config.continue_run_id


def run():
    # Load configuration from YAML file
    with open("experiment_config.yaml") as config_file:
        yaml_data = yaml.safe_load(config_file)
    config = ExperimentConfig(**yaml_data)

    run_id = track(config.tracking)
    if config.tracking.continue_run_id is None:
        config.tracking.continue_run_id = run_id
    with mlflow.start_run(run_id=run_id):
        repo, commit_hash = get_repo_and_commit_cwd()
        mlflow.log_param("repo", repo)
        mlflow.log_param("commit", commit_hash)
        mlflow.log_params(flatten_dict(config.dict()))
        mlflow.pytorch.autolog()
        architecture = config.architecture.get_constructor()(**config.architecture.dict())

        segmentation = config.segmentation.get_constructor()(
            architecture, image_size=config.image_size, **config.segmentation.dict()
        )
        prediction_exporter = config.prediction_exporter.get_constructor()(**config.prediction_exporter.dict())
        loader_exporter = config.loader_exporter.get_constructor()(**config.loader_exporter.dict())
        training_data_args = config.training_data.dict()
        del training_data_args["data_type"]
        training_data_args["image_size"] = config.image_size
        training_dataset = config.training_data.get_constructor()(**training_data_args)

        validation_data_args = config.validation_data.dict()
        del validation_data_args["data_type"]
        validation_data_args["image_size"] = config.image_size
        validation_dataset = config.validation_data.get_constructor()(**validation_data_args)
        if isinstance(validation_dataset, CellMapDatasets3Das2D):
            crop_list = []
            for ds in validation_dataset.datasets:
                crop_list.append([c.crop_name for c in ds.crops])
            mlflow.log_param("validation_crop_list", crop_list)
        elif isinstance(validation_dataset, CellMapDataset3Das2D):
            crop_list = [c.crop_name for c in validation_dataset.crops]
            mlflow.log_param("validation_crop_list", crop_list)
        if isinstance(training_dataset, CellMapDatasets3Das2D):
            crop_list = []
            for ds in training_dataset.datasets:
                crop_list.append([c.crop_name for c in ds.crops])
            mlflow.log_param("training_crop_list", crop_list)
        elif isinstance(training_dataset, CellMapDataset3Das2D):
            crop_list = [c.crop_name for c in training_dataset.crops]
            mlflow.log_param("training_crop_list", crop_list)
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        if parsed_artifact_uri.scheme != "file":
            raise NotImplementedError(
                f"Using a {parsed_artifact_uri.scheme} connection to save artifacts is not implemented"
            )
        trainer = BaselineSegmentationTrainer(
            segmentation,
            training_dataset,
            loader_exporter,
            prediction_exporter,
            results_folder=os.path.join(parsed_artifact_uri.path, "checkpoints"),
            validation_dataset=validation_dataset,
            **config.training.dict(),
        )
        trainer.train()


if __name__ == "__main__":
    run()
