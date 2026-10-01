"""Load raw project tables. This is the only processing module that reads files."""
from pathlib import Path
import pandas as pd

RAW_TABLES = [
    "subjects", "teachers", "topics", "classes", "students", "class_schedule",
    "curriculum", "assessments", "scores", "interventions", "intervention_participants",
]


def load_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    missing = [name for name in RAW_TABLES if not (raw_dir / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw tables: {', '.join(missing)}")
    return {name: pd.read_csv(raw_dir / f"{name}.csv") for name in RAW_TABLES}
