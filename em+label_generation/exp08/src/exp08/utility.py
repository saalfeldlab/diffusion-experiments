import logging
import subprocess
from typing import Dict, Optional, Tuple

import yaml
import git

from exp08.config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_dict(nested_dict: Dict, parent_key: str = "", separator: str = ".", max_depth: int = 1) -> Dict:
    flat_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, Dict) and max_depth > 0:
            flat_dict.update(flatten_dict(value, new_key, separator, max_depth - 1))
        else:
            flat_dict[new_key] = value
    return flat_dict


def unflatten_dict(flat_dict: Dict, separator: str = ".") -> Dict:
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(separator)
        iterator_dict = nested_dict
        for nesting_key in keys[:-1]:
            if nesting_key not in iterator_dict:
                iterator_dict[nesting_key] = dict()
            iterator_dict = iterator_dict[nesting_key]
        iterator_dict[keys[-1]] = value
    return nested_dict


def start_ui() -> None:
    with open("experiment_config.yaml") as config_file:
        yaml_data = yaml.safe_load(config_file)
    config = ExperimentConfig(**yaml_data)
    command = ["mlflow", "ui", "--backend-store-uri", config.tracking.tracking_uri]
    subprocess.run(command)


def get_repo_and_commit_cwd(remote_name: Optional[str] = "origin") -> Tuple[str, str]:
    repo = git.Repo(search_parent_directories=True)
    if len(repo.remotes) == 0 or remote_name is None:
        remote = None
    else:
        remote = next((r for r in repo.remotes if r.name == remote_name), None)
    if remote is None:
        repo_name = repo.working_dir
    else:
        repo_name = remote.url

    commit_hash = repo.head.commit.hexsha
    return repo_name, commit_hash
