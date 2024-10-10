from exp09.utility import unflatten_dict
import click
import yaml
import os
import mlflow
from config import InferenceConfig, ExperimentConfig
from urllib.parse import urlparse
from denoising_diffusion_pytorch.denoising_diffusion import Trainer


@click.command("Running inference with given configuration file")
@click.argument("inference_yaml", type=click.Path(exists=True))
def run(inference_yaml):
    with open(inference_yaml) as inference_config_file:
        yaml_data = yaml.safe_load(inference_config_file)
    inference_config = InferenceConfig(**yaml_data)
    with open("experiment_config.yaml") as experiment_config_file:
        yaml_data = yaml.safe_load(experiment_config_file)
    experiment_config = ExperimentConfig(**yaml_data)
    mlflow.set_tracking_uri(experiment_config.tracking.tracking_uri)
    experiment_run_id = inference_config.experiment_run_id
    with mlflow.start_run(run_id=experiment_run_id):
        architecture = experiment_config.architecture.get_constructor()(**experiment_config.architecture.dict())
        diffusion = experiment_config.diffusion.get_constructor()(
            architecture, image_size=experiment_config.image_size, **experiment_config.diffusion.dict()
        )
        data_args = experiment_config.data.dict()
        del data_args["data_type"]
        data_args["pre_load"] = False
        data_args["image_size"] = experiment_config.image_size
        dataset = experiment_config.data.get_constructor()(**data_args)
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        if parsed_artifact_uri.scheme != "file":
            raise NotImplementedError(
                f"Using a {parsed_artifact_uri.scheme} connection to save artifacts is not implemented"
            )
        dummy_exporter = experiment_config.exporter.get_constructor()(**experiment_config.exporter.dict())
        trainer = Trainer(
            diffusion,
            dataset,
            dummy_exporter,
            results_folder=os.path.join(parsed_artifact_uri.path, "checkpoints"),
            **experiment_config.training.dict(),
        )
        trainer.load(inference_config.checkpoint)
    inference_exporter = inference_config.exporter.get_constructor()(**inference_config.exporter.dict())
    save_path = os.path.join(
        parsed_artifact_uri.path, "checkpoints", f"ckpt_{inference_config.checkpoint:0{trainer.milestone_digits}d}"
    )
    num_saved_samples = 0
    while num_saved_samples < inference_config.num_samples:
        samples = trainer.sampler.sample(inference_config.eval_batch_size)
        num_saved_samples += inference_exporter.save_sample(save_path, samples)


if __name__ == "__main__":
    run()
