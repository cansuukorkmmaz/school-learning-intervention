"""Translate analysis results into explainable, human-reviewable recommendations."""
import pandas as pd


def build_recommendations(gap_table, support_groups, class_summary, matches, config):
    rec = []
    actionable = gap_table[(gap_table.category == "SPECIFIC_GAP") & (gap_table.confidence.isin(["medium", "high"])) & (~gap_table.trend_label.eq("improving"))]
    for row in actionable.itertuples(index=False):
        rec.append({"recommendation_id": f"R1_{row.student_id}_{row.topic_id}", "recommendation_type": "TARGETED_PRACTICE", "student_id": row.student_id, "topic_id": row.topic_id, "group_id": None, "host_class_id": None, "status": "SUGGESTED", "rationale": f"{row.topic_id} score {row.topic_score:.1f} is {row.personal_gap:.1f} points below the student's topic baseline ({row.personal_baseline:.1f}), with {row.n_observations} observations."})

    for group in support_groups.itertuples(index=False):
        for student_id in group.student_ids:
            rec.append({"recommendation_id": f"R2_{group.group_id}_{student_id}", "recommendation_type": "SUPPORT_SESSION", "student_id": student_id, "topic_id": group.topic_id, "group_id": group.group_id, "host_class_id": None, "status": "SUGGESTED", "rationale": f"{group.student_count} students share an actionable {group.topic_id} gap; group average is {group.average_score:.1f}."})

    for row in matches.itertuples(index=False):
        if row.match_status == "SUGGESTED":
            rec.append({"recommendation_id": f"R3_{row.student_id}_{row.topic_id}", "recommendation_type": "CROSS_CLASS_ATTENDANCE", "student_id": row.student_id, "topic_id": row.topic_id, "group_id": None, "host_class_id": row.host_class_id, "status": "SUGGESTED", "rationale": row.reason})

    broad = gap_table[gap_table.category == "BROAD_DIFFICULTY"].drop_duplicates("student_id")
    for row in broad.itertuples(index=False):
        student_topics = gap_table[gap_table.student_id == row.student_id]
        rec.append({"recommendation_id": f"R4_{row.student_id}", "recommendation_type": "TEACHER_REVIEW", "student_id": row.student_id, "topic_id": None, "group_id": None, "host_class_id": None, "status": "SUGGESTED", "rationale": f"Low performance across {len(student_topics)} assessed topics; topic-specific intervention may not address the broader pattern."})

    for row in class_summary.itertuples(index=False):
        if row.gap_share >= config["recommendation"]["class_wide_share"] or row.class_median < config["recommendation"]["class_median_threshold"]:
            rec.append({"recommendation_id": f"R5_{row.class_id}_{row.topic_id}", "recommendation_type": "CLASS_WIDE_INSIGHT", "student_id": None, "topic_id": row.topic_id, "group_id": None, "host_class_id": row.class_id, "status": "INFORMATIONAL", "rationale": f"Class median is {row.class_median:.1f}; {row.gap_share:.0%} of analyzed students show a gap-related flag."})

    monitored = gap_table[(gap_table.category == "WATCH") | (gap_table.confidence == "low")]
    for row in monitored.itertuples(index=False):
        rec.append({"recommendation_id": f"R6_{row.student_id}_{row.topic_id}", "recommendation_type": "MONITOR", "student_id": row.student_id, "topic_id": row.topic_id, "group_id": None, "host_class_id": None, "status": "INFORMATIONAL", "rationale": f"Monitor {row.topic_id}: category={row.category}, confidence={row.confidence}, observations={row.n_observations}."})
    return pd.DataFrame(rec)
