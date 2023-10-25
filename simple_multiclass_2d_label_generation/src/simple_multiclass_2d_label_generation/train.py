import os
from urllib.parse import urlparse, urlunparse

import mlflow
import yaml
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

from simple_multiclass_2d_label_generation.config import (
    ExperimentConfig,
    TrackingConfig,
)
from simple_multiclass_2d_label_generation.utility import flatten_dict


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
    with mlflow.start_run(run_id=run_id):

        mlflow.log_params(flatten_dict(config.model_dump()))
        mlflow.pytorch.autolog()
        architecture = config.architecture.get_constructor()(
            **config.architecture.model_dump()
        )

        diffusion = config.diffusion.get_constructor()(
            architecture, image_size=config.image_size, **config.diffusion.model_dump()
        )
        dataset = config.data.get_constructor()(config.data.data_dir, config.image_size)
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        if parsed_artifact_uri.scheme != "file":
            raise NotImplementedError(
                f"Using a {parsed_artifact_uri.scheme} connection to save artifacts is not implemented"
            )
        trainer = Trainer(
            diffusion,
            dataset,
            results_folder=parsed_artifact_uri.path,
            **config.training.model_dump(),
        )
        trainer.train()


if __name__ == "__main__":
    run()
