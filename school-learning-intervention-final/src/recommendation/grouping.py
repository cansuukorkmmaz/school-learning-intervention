"""Group students who have the same actionable learning need."""
import pandas as pd


def build_support_groups(gap_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    rec = config["recommendation"]
    actionable = gap_table[(gap_table.category == "SPECIFIC_GAP") & (gap_table.confidence.isin(["medium", "high"])) & (~gap_table.trend_label.eq("improving"))].copy()
    if actionable.empty:
        return pd.DataFrame(columns=["group_id", "topic_id", "grade_level", "student_ids", "student_count", "average_score"])
    rows = []
    for (topic, grade), group in actionable.groupby(["topic_id", "grade_level"] if "grade_level" in actionable else ["topic_id"]):
        students = sorted(group.student_id.unique())
        if len(students) < rec["min_group_size"]:
            continue
        for start in range(0, len(students), rec["max_group_size"]):
            chunk = students[start:start + rec["max_group_size"]]
            if len(chunk) < rec["min_group_size"]:
                continue
            subset = group[group.student_id.isin(chunk)]
            rows.append({"group_id": f"GRP_{topic}_{start // rec['max_group_size'] + 1:02d}", "topic_id": topic, "grade_level": grade if isinstance(grade, int) else None, "student_ids": chunk, "student_count": len(chunk), "average_score": round(float(subset.topic_score.mean()), 1)})
    return pd.DataFrame(rows)
