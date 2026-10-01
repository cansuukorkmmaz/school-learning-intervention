"""End-to-end orchestration for the school learning intervention system."""
from pathlib import Path
import pandas as pd

from .config import load_config
from .processing.loaders import load_raw_tables
from .processing.validate import validate_tables, raise_if_invalid
from .processing.preprocess import build_observations
from .analysis.gap_detection import detect_gaps
from .analysis.trends import compute_trends
from .analysis.class_insights import build_class_topic_summary
from .recommendation.grouping import build_support_groups
from .recommendation.matching import match_cross_class_students
from .recommendation.recommend import build_recommendations
from .feedback.outcomes import evaluate_intervention
from .config import PROJECT_ROOT


def run_pipeline(as_of_date: str | None = None, data_dir: str | Path = "data", config: dict | None = None) -> dict[str, object]:
    config = config or load_config()
    data_dir = Path(data_dir)
    tables = load_raw_tables(data_dir / "raw")
    report = validate_tables(tables)
    raise_if_invalid(report)
    if as_of_date is None:
        as_of_date = tables["assessments"]["date"].max()
    observations = build_observations(tables, as_of_date, config["analysis"]["half_life_days"])
    gap_table = detect_gaps(observations, tables["classes"], tables["topics"], config)
    topic_trends, overall_trends = compute_trends(observations, config)
    gap_table = gap_table.merge(topic_trends[["student_id", "topic_id", "trend_change", "trend_label"]], on=["student_id", "topic_id"], how="left", suffixes=("_initial", ""))
    declining = gap_table.trend_label.eq("declining")
    gap_table.loc[declining, "priority"] *= 1.25
    class_summary = build_class_topic_summary(gap_table)
    groups = build_support_groups(gap_table, config)
    matches = match_cross_class_students(gap_table, tables["students"], tables["classes"], tables["curriculum"], tables["class_schedule"], as_of_date, config)
    recommendations = build_recommendations(gap_table, groups, class_summary, matches, config)

    outcomes = pd.DataFrame()
    comparison = pd.DataFrame()
    outcome_summary = pd.DataFrame()
    if not tables["interventions"].empty and not tables["intervention_participants"].empty:
        intervention = tables["interventions"].iloc[0].copy()
        intervention.start_date = pd.Timestamp(intervention.start_date)
        intervention.end_date = pd.Timestamp(intervention.end_date)
        participants = tables["intervention_participants"][tables["intervention_participants"].intervention_id == intervention.intervention_id].copy()
        start_obs = build_observations(tables, intervention.start_date - pd.Timedelta(days=1), config["analysis"]["half_life_days"])
        start_gap = detect_gaps(start_obs, tables["classes"], tables["topics"], config)
        start_topic_trends, _ = compute_trends(start_obs, config)
        if not start_gap.empty:
            start_gap = start_gap.merge(start_topic_trends[["student_id", "topic_id", "trend_change", "trend_label"]], on=["student_id", "topic_id"], how="left", suffixes=("_initial", ""))
        outcomes, comparison, outcome_summary = evaluate_intervention(intervention, participants, observations, start_gap)

    result = {"tables": tables, "validation": report, "observations": observations, "gap_table": gap_table, "topic_trends": topic_trends, "overall_trends": overall_trends, "class_summary": class_summary, "support_groups": groups, "matches": matches, "recommendations": recommendations, "outcomes": outcomes, "comparison": comparison, "outcome_summary": outcome_summary}
    return result
