"""Convert validated relational tables into analysis-ready observations."""
from datetime import date
import numpy as np
import pandas as pd


def build_observations(tables: dict[str, pd.DataFrame], as_of_date: str | date, half_life_days: float = 42) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of_date)
    scores = tables["scores"].copy()
    assessments = tables["assessments"][["assessment_id", "class_id", "date"]].copy().rename(columns={"class_id": "assessment_class_id"})
    students = tables["students"][["student_id", "class_id"]].copy()

    observations = scores.merge(assessments, on="assessment_id", how="left", validate="many_to_one")
    observations = observations.merge(students, on="student_id", how="left", validate="many_to_one")
    observations["class_id"] = observations["class_id"].astype(str)
    observations["date"] = pd.to_datetime(observations["date"], errors="raise")
    observations = observations[observations["date"] <= as_of].copy()
    observations["age_days"] = (as_of - observations["date"]).dt.days.astype(int)
    observations["recency_weight"] = np.power(0.5, observations["age_days"] / half_life_days)
    return observations[["student_id", "class_id", "assessment_id", "topic_id", "date", "score_pct", "age_days", "recency_weight"]].sort_values(["student_id", "date", "topic_id"]).reset_index(drop=True)
