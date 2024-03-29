from seg_exp01.utility import unflatten_dict
import click
import yaml
import os
import mlflow
from config import InferenceConfig, ExperimentConfig
from urllib.parse import urlparse
from denoising_diffusion_pytorch.baseline_segmentation import BaselineSegmentationPredictor

# import torch

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
        segmentation = experiment_config.segmentation.get_constructor()(
            architecture, image_size=experiment_config.image_size, **experiment_config.segmentation.dict()
        )
        training_data_args = inference_config.data.dict()
        del training_data_args["data_type"]
        # print(training_data_args)
        # folder = training_data_args["folder"]
        # del training_data_args["folder"]
        dataset = inference_config.data.get_constructor()(**training_data_args, convert_image_to="L")
        exporter = inference_config.exporter.get_constructor()(**inference_config.exporter.dict())
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        predictor = BaselineSegmentationPredictor(
            segmentation,
            dataset,
            exporter,
            results_folder = os.path.join(parsed_artifact_uri.path, "checkpoints"),
            **inference_config.predictor.dict()
        )
        # predictor.model.inference_model.eval()
        # with torch.inference_mode():
        #     for data in predictor.dl:
        #         prediction = predictor.model.inference_model(data)
        #         print(prediction.size())
        # print(len(dataset))
        predictor.inference()

if __name__ == "__main__":
    run()
