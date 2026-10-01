"""Project configuration loading."""
from pathlib import Path
from functools import lru_cache
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
