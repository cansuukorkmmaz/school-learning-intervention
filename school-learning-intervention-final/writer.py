"""
writer.py -- the ONLY place in the generator that touches the disk.

Generation (generators.py) returns DataFrames; this module decides where they
go. The ground-truth answer key is written to a different folder on purpose,
so the analysis code can never accidentally read it from data/raw/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# The tables the *system* is allowed to read.
RAW_TABLES = [
    "subjects", "teachers", "topics", "classes", "students", "class_schedule",
    "curriculum", "assessments", "scores", "interventions", "intervention_participants",
]


def write_tables(tables: dict[str, pd.DataFrame], raw_dir: Path, truth_dir: Path) -> list[Path]:
    """Write the 11 raw CSVs to raw_dir and ground_truth.csv to truth_dir."""
    raw_dir, truth_dir = Path(raw_dir), Path(truth_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in RAW_TABLES:
        path = raw_dir / f"{name}.csv"
        tables[name].to_csv(path, index=False)      # dates are written as YYYY-MM-DD
        written.append(path)

    truth_path = truth_dir / "ground_truth.csv"
    tables["ground_truth"].to_csv(truth_path, index=False)
    written.append(truth_path)
    return written
