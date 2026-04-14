import os
from pathlib import Path

PROJECT_ROOT = Path("/scratch3/workspace/seoyunjeong_umass_edu-plantclef/PlantCLEF2026")


def get_data_dir() -> str:
    """
    Get the data directory for the PlantCLEF2026 project on the UMass HPC.
    """
    return str(PROJECT_ROOT / "data")


def get_scratch_data_dir() -> str:
    """
    Get the scratch data directory. Points to the same location as get_data_dir()
    since the project lives on scratch3 already.
    """
    return str(PROJECT_ROOT / "data")


def get_home_dir():
    """Get the home directory for the current user."""
    return Path(os.path.expanduser("~"))


def get_class_mappings_file() -> str:
    """
    Get the path to class_mapping.txt for the DINOv2 model.
    """
    return str(PROJECT_ROOT / "pretrained_models" / "class_mapping.txt")


if __name__ == "__main__":
    print(get_class_mappings_file())
