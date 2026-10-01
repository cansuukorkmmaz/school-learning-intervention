"""Trend analysis at topic and student levels."""
import numpy as np
import pandas as pd


def _fit_change(frame: pd.DataFrame, value_col: str = "score_pct") -> float:
    if len(frame) < 2:
        return 0.0
    x = frame["week_index"].to_numpy(dtype=float)
    y = frame[value_col].to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * (x.max() - x.min()))


def compute_trends(observations: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = config["analysis"]["trend_change_threshold"]
    min_topic = config["analysis"]["min_topic_trend_observations"]
    min_overall = config["analysis"]["min_overall_trend_assessments"]
    obs = observations.copy()
    obs["week_index"] = ((obs["date"] - obs["date"].min()).dt.days // 7).astype(int)

    topic_rows = []
    for (student_id, topic_id), group in obs.groupby(["student_id", "topic_id"]):
        change = _fit_change(group)
        if len(group) < min_topic:
            label = "insufficient_data"
        elif change >= threshold:
            label = "improving"
        elif change <= -threshold:
            label = "declining"
        else:
            label = "stable"
        topic_rows.append({"student_id": student_id, "topic_id": topic_id, "trend_change": change, "trend_label": label, "trend_n": len(group)})

    assessment_mean = obs.groupby(["student_id", "assessment_id", "date"], as_index=False)["score_pct"].mean()
    overall_rows = []
    for student_id, group in assessment_mean.groupby("student_id"):
        group = group.sort_values("date").copy()
        group["week_index"] = ((group["date"] - obs["date"].min()).dt.days // 7).astype(int)
        change = _fit_change(group)
        if len(group) < min_overall:
            label = "insufficient_data"
        elif change >= threshold:
            label = "improving"
        elif change <= -threshold:
            label = "declining"
        else:
            label = "stable"
        overall_rows.append({"student_id": student_id, "overall_trend_change": change, "overall_trend_label": label, "assessment_count": len(group)})
    return pd.DataFrame(topic_rows), pd.DataFrame(overall_rows)
