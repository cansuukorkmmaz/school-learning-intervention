import pandas as pd

from src.config import load_config
from src.pipeline import run_pipeline
from src.processing.loaders import load_raw_tables
from src.processing.validate import validate_tables
from src.processing.preprocess import build_observations


def test_validation_accepts_generated_data():
    tables = load_raw_tables("data/raw")
    report = validate_tables(tables)
    assert report.valid


def test_as_of_date_filters_future_observations():
    tables = load_raw_tables("data/raw")
    observations = build_observations(tables, "2025-11-01", load_config()["analysis"]["half_life_days"])
    assert observations["date"].max() <= pd.Timestamp("2025-11-01")


def test_gap_detector_finds_planted_trigonometry_gaps():
    result = run_pipeline("2025-12-19", "data")
    trig = result["gap_table"]
    found = set(trig.loc[(trig.topic_id == "TRIG") & (trig.category == "SPECIFIC_GAP"), "student_id"])
    expected = {"S003", "S007", "S011", "S014", "S018", "S023", "S028", "S031", "S036"}
    assert expected <= found


def test_cross_class_matcher_exercises_capacity_and_schedule_branches():
    result = run_pipeline("2025-12-19", "data")
    counts = result["matches"].match_status.value_counts().to_dict()
    assert counts.get("SUGGESTED", 0) == 4
    assert counts.get("NO_CAPACITY", 0) == 1
    assert counts.get("NO_MATCH_SCHEDULE", 0) == 4


def test_outcome_analysis_has_comparison_group():
    result = run_pipeline("2025-12-19", "data")
    summary = result["outcome_summary"].iloc[0]
    assert summary["participant_n"] == 8
    assert summary["comparison_n"] >= 5
    assert summary["difference_in_differences"] > 0
