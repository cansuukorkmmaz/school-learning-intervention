"""Before/after intervention analysis with an explicitly non-causal comparison."""
import pandas as pd


def _window_scores(observations, student_id, topic_id, start, end):
    data = observations[(observations.student_id == student_id) & (observations.topic_id == topic_id)].sort_values("date")
    before = data[data.date < pd.Timestamp(start)].tail(2)
    after = data[data.date > pd.Timestamp(end)].head(2)
    return before.score_pct.mean() if not before.empty else None, after.score_pct.mean() if not after.empty else None, len(before), len(after)


def evaluate_intervention(intervention, participants, observations, gap_table_at_start):
    topic = intervention.topic_id
    rows = []
    for row in participants.itertuples(index=False):
        before, after, n_before, n_after = _window_scores(observations, row.student_id, topic, intervention.start_date, intervention.end_date)
        delta = after - before if before is not None and after is not None else None
        if delta is None:
            label = "not_evaluable"
        elif n_before < 2 or n_after < 2:
            label = "low_confidence"
        elif delta >= 10:
            label = "improved"
        elif delta <= -10:
            label = "declined"
        else:
            label = "no_clear_change"
        rows.append({"student_id": row.student_id, "topic_id": topic, "before_score": before, "after_score": after, "delta": delta, "label": label, "n_before": n_before, "n_after": n_after, "attended": row.attended})

    outcomes = pd.DataFrame(rows)
    comparison_ids = set(gap_table_at_start[(gap_table_at_start.category == "SPECIFIC_GAP") & (gap_table_at_start.topic_id == topic)].student_id) - set(participants.student_id)
    comparison_rows = []
    for student_id in sorted(comparison_ids):
        before, after, n_before, n_after = _window_scores(observations, student_id, topic, intervention.start_date, intervention.end_date)
        if before is not None and after is not None:
            comparison_rows.append({"student_id": student_id, "before_score": before, "after_score": after, "delta": after - before, "n_before": n_before, "n_after": n_after})
    comparison = pd.DataFrame(comparison_rows)

    attended = outcomes[outcomes.attended == True]
    participant_delta = attended.delta.mean() if not attended.empty else None
    comparison_delta = comparison.delta.mean() if not comparison.empty else None
    difference = participant_delta - comparison_delta if participant_delta is not None and comparison_delta is not None else None
    before_diff = abs(attended.before_score.mean() - comparison.before_score.mean()) if not attended.empty and not comparison.empty else None
    summary = pd.DataFrame([{"intervention_id": intervention.intervention_id, "topic_id": topic, "participant_n": len(attended), "comparison_n": len(comparison), "participant_mean_delta": participant_delta, "comparison_mean_delta": comparison_delta, "difference_in_differences": difference, "before_group_difference": before_diff, "comparability_warning": bool(before_diff is not None and before_diff > 10)}])
    return outcomes, comparison, summary
