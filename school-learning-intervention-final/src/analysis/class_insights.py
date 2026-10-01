"""Class/topic-level descriptive insights."""
import pandas as pd


def build_class_topic_summary(gap_table: pd.DataFrame) -> pd.DataFrame:
    if gap_table.empty:
        return pd.DataFrame()
    summary = gap_table.groupby(["class_id", "topic_id"], as_index=False).agg(
        class_median=("topic_score", "median"),
        student_count=("student_id", "nunique"),
        specific_gap_count=("category", lambda s: int((s == "SPECIFIC_GAP").sum())),
        class_wide_count=("category", lambda s: int((s == "CLASS_WIDE").sum())),
    )
    summary["gap_share"] = (summary["specific_gap_count"] + summary["class_wide_count"]) / summary["student_count"]
    return summary
