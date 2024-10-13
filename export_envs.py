import logging
import os
import subprocess
from pathlib import Path

from git import Repo

logger = logging.getLogger(__name__)


def get_changed_pyproject_tomls(repo_path="."):
    """
    Get a list of staged pyproject.toml files that have been added, copied, modified, or renamed.
    """
    repo = Repo(repo_path)
    diff_files = [item.a_path for item in repo.index.diff("HEAD")]
    changed_files = [f for f in diff_files if f.endswith("pyproject.toml")]
    return changed_files


def regenerate_and_export_default_env(project_dir):
    currdir = os.getcwd()
    os.chdir(project_dir)
    (project_dir / "condaenvs").mkdir(exist_ok=True)
    subprocess.run(["hatch", "env", "prune"], check=True)
    subprocess.run(["hatch", "env", "create"], check=True)
    # env_path = subprocess.run(
    #     ["hatch", "env", "find"], capture_output=True, text=True, check=True
    # ).stdout.strip()
    env_file = project_dir / "condaenvs" / "default.txt"
    subprocess.run(
        ["hatch", "-e", "default", "run", "pip", "freeze", ">", str(env_file)],
        check=True,
    )
    os.chdir(currdir)


def check_for_yaml_changes(project_dir, repo_path="."):
    repo = Repo(repo_path)
    yaml_file = str(
        (project_dir / "condaenvs" / "default.txt").relative_to(
            Path(repo_path).absolute()
        )
    )
    modified = False
    if repo.is_dirty():
        if yaml_file in [item.a_path for item in repo.index.diff(None)]:
            modified = True
    if yaml_file in repo.untracked_files:
        modified = True
    return modified


def main():
    changed_pyproject_files = get_changed_pyproject_tomls()
    if not changed_pyproject_files:
        logger.info(
            "No changes to pyproject.toml files. No environments to be exported"
        )
        return

    changed_dirs = []
    for pyproject_file in changed_pyproject_files:
        project_dir = (Path(".").absolute() / pyproject_file).parent
        changed_dirs.append(project_dir)
    for project_dir in changed_dirs:
        regenerate_and_export_default_env(project_dir)
    for project_dir in changed_dirs:
        if check_for_yaml_changes(project_dir):
            logger.error(
                "YAML files have changed. Please stage the changes and commit again."
            )
            exit(1)


if __name__ == "__main__":
    main()
