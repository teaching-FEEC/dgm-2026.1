"""Load YAML experiment configs and derive output paths."""

from pathlib import Path
from types import SimpleNamespace

import yaml
import time


def _to_namespace(obj):
    """Recursively convert nested dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(config_path, project_root):
    """Load an experiment YAML and inject derived output paths.

    Adds two derived attributes based on cfg.experiment_name:
      - cfg.checkpoint_path: project_root / "models" / "{experiment_name}.pt"
      - cfg.results_dir:     project_root / "results" / "{experiment_name}"
    The results dir is created if it does not exist.

    Args:
        config_path: Path to YAML config file.
        project_root: Path to project root (used to anchor derived paths).

    Returns:
        SimpleNamespace with the YAML contents and derived paths.
    """
    config_path = Path(config_path)
    project_root = Path(project_root)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if "experiment_name" not in raw:
        raise ValueError(f"Config {config_path} is missing required 'experiment_name'")

    exp_name = raw["experiment_name"]
    current_time = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{exp_name}_{current_time}"
    raw["checkpoint_path"] = project_root / "models" / f"{exp_name}.pt"
    raw["results_dir"] = project_root / "results" / exp_name
    raw["results_dir"].mkdir(parents=True, exist_ok=True)

    return _to_namespace(raw)
