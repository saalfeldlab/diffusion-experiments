import os
import warnings
from urllib.parse import urlparse, urlunparse

import mlflow
import yaml
from denoising_diffusion_pytorch import (
    CellMapDataset3Das2D,
    CellMapDatasets3Das2D,
    GaussianDiffusion,
    Trainer,
    Unet,
)

from exp11.config import ExperimentConfig, TrackingConfig
from exp11.utility import flatten_dict, get_repo_and_commit_cwd

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
        mlflow.log_params(flatten_dict(config.model_dump()))
        mlflow.pytorch.autolog()
        architecture = config.architecture.get_constructor()(
            **config.architecture.model_dump()
        )

        diffusion = config.diffusion.get_constructor()(
            architecture, image_size=config.image_size, **config.diffusion.model_dump()
        )

        data_args = config.data.model_dump()
        del data_args["data_type"]
        data_args["image_size"] = config.image_size
        dataset = config.data.get_constructor()(**data_args)
        if config.exporter.annotations is None:
            config.exporter.annotations = dataset.sample_requests
        sample_exporter = config.exporter.get_constructor()(
            **config.exporter.model_dump()
        )
        if isinstance(dataset, CellMapDatasets3Das2D):
            crop_list = []
            for ds in dataset.datasets:
                crop_list.append([c.crop_name for c in ds.crops])
            mlflow.log_param("crop_list", crop_list)
        elif isinstance(dataset, CellMapDataset3Das2D):
            crop_list = [c.crop_name for c in dataset.crops]
            mlflow.log_param("crop_list", crop_list)
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        if parsed_artifact_uri.scheme != "file":
            msg = f"Using a {parsed_artifact_uri.scheme} connection to save artifacts is not implemented"
            raise NotImplementedError(msg)
        trainer = Trainer(
            diffusion,
            dataset,
            sample_exporter,
            results_folder=os.path.join(parsed_artifact_uri.path, "checkpoints"),
            **config.training.model_dump(),
        )
        trainer.train()


if __name__ == "__main__":
    run()
