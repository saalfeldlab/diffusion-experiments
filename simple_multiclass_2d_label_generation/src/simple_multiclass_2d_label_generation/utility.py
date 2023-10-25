import logging
import subprocess
from typing import Dict

import yaml
import git 

from simple_multiclass_2d_label_generation.config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_dict(nested_dict, parent_key="", separator=".", max_depth=1):
    flat_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, Dict) and max_depth > 0:
            flat_dict.update(flatten_dict(value, new_key, separator, max_depth - 1))
        else:
            flat_dict[new_key] = value
    return flat_dict


def start_ui():
    with open("experiment_config.yaml") as config_file:
        yaml_data = yaml.safe_load(config_file)
    config = ExperimentConfig(**yaml_data)
    command = ["mlflow", "ui", "--backend-store-uri", config.tracking.tracking_uri]
    subprocess.run(command)

def get_commit_hash_cwd():
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.commit.hexsha
    return commit_hash

