"""Load the project's YAML config into a plain Python dict.

Kept deliberately tiny: one function, one job. Every future script calls
`load_config()` instead of hard-coding paths or hyperparameters, so a run's
copy of the config (saved into its results folder) is a complete record of
what produced it.
"""

from pathlib import Path

import yaml

# The repo root is three levels up from this file:
# src/solaruq/utils/config.py -> src/solaruq/utils -> src/solaruq -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """Read a YAML config file and return it as a dict.

    Parameters
    ----------
    path:
        Path to a YAML file. Defaults to configs/config.yaml at the repo root.

    Returns
    -------
    dict
        The parsed config. Nested keys (e.g. config["forecast"]["horizon_steps"])
        map directly onto the structure of the YAML file.
    """
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)
