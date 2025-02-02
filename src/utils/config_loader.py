import json
import yaml
from pathlib import Path
from typing import Dict

def load_json_config(filepath: str) -> Dict:
    """Loads a configuration file in JSON format."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file {filepath} not found.")

    with open(filepath, "r") as f:
        return json.load(f)

def load_yaml_config(filepath: str) -> Dict:
    """Loads a configuration file in YAML format."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file {filepath} not found.")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)
