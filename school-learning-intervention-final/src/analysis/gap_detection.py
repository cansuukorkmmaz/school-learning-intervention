"""Explainable learning-gap detection."""
import numpy as np
import pandas as pd


def _weighted_topic_scores(observations: pd.DataFrame) -> pd.DataFrame:
    grouped = observations.groupby(["student_id", "class_id", "topic_id"], as_index=False)
    result = grouped.apply(lambda g: pd.Series({
        "topic_score": np.average(g["score_pct"], weights=g["recency_weight"]),
        "n_observations": g["assessment_id"].nunique(),
        "score_std": g["score_pct"].std(ddof=0) if len(g) > 1 else 0.0,
    }), include_groups=False).reset_index(drop=True)
    return result


def _add_baselines(topic_scores: pd.DataFrame, classes: pd.DataFrame, config: dict) -> pd.DataFrame:
    min_class_n = config["min_class_n"]
    rows = []
    for row in topic_scores.itertuples(index=False):
        others = topic_scores[(topic_scores.student_id == row.student_id) & (topic_scores.topic_id != row.topic_id)]
        baseline = float(others.topic_score.median()) if len(others) >= config["min_other_topics"] else np.nan
        class_topic = topic_scores[(topic_scores.class_id == row.class_id) & (topic_scores.topic_id == row.topic_id)]
        if len(class_topic) >= min_class_n:
            class_median = float(class_topic.topic_score.median())
            class_n = len(class_topic)
        else:
            grade = int(classes.loc[classes.class_id == row.class_id, "grade_level"].iloc[0])
            grade_topic = topic_scores.merge(classes[["class_id", "grade_level"]], on="class_id")
            grade_topic = grade_topic[(grade_topic.grade_level == grade) & (grade_topic.topic_id == row.topic_id)]
            class_median = float(grade_topic.topic_score.median()) if not grade_topic.empty else np.nan
            class_n = len(grade_topic)
        rows.append({**row._asdict(), "personal_baseline": baseline, "personal_gap": baseline - row.topic_score if pd.notna(baseline) else np.nan, "class_median": class_median, "class_gap": class_median - row.topic_score if pd.notna(class_median) else np.nan, "class_n": class_n})
    return pd.DataFrame(rows)


def detect_gaps(observations: pd.DataFrame, classes: pd.DataFrame, topics: pd.DataFrame, config: dict) -> pd.DataFrame:
    analysis = config["analysis"]
    topic_scores = _weighted_topic_scores(observations)
    if topic_scores.empty:
        return pd.DataFrame()
    table = _add_baselines(topic_scores, classes, analysis)
    table = table.merge(topics[["topic_id", "grade_level"]], on="topic_id", how="left", validate="many_to_one")

    topic_counts = table.groupby("student_id")["topic_id"].nunique()
    low_counts = table.assign(is_low=table.topic_score < analysis["abs_threshold"]).groupby("student_id")["is_low"].sum()
    assessed_counts = table.groupby("student_id")["topic_id"].nunique()
    broad_students = set(low_counts[(assessed_counts >= analysis["broad_min_topics"]) & ((low_counts / assessed_counts) >= analysis["broad_share"])].index)

    def classify(row):
        n, score = int(row.n_observations), float(row.topic_score)
        if n == 0:
            return "NO_DATA"
        if pd.isna(row.personal_baseline):
            return "NEEDS_MORE_DATA"
        if score >= analysis["abs_threshold"]:
            return "WATCH" if score < 60 and row.trend_label == "declining" else "NO_GAP"
        if row.student_id in broad_students:
            return "BROAD_DIFFICULTY"
        if n == 1:
            return "WATCH"
        if row.personal_gap >= analysis["rel_gap_min"]:
            return "SPECIFIC_GAP" if row.class_gap >= analysis["class_gap_min"] else "CLASS_WIDE"
        return "WATCH"

    table["trend_label"] = "stable"
    table["category"] = table.apply(classify, axis=1)
    table["severity"] = np.where(table.category == "SPECIFIC_GAP", pd.cut(table.personal_gap, bins=[-np.inf, 25, 35, np.inf], labels=["mild", "moderate", "significant"], right=False), "")
    table["severity"] = table["severity"].astype(str)
    severity_weight = {"mild": 1, "moderate": 2, "significant": 3}
    table["confidence_score"] = np.minimum(table.n_observations, 3) / 3
    table["confidence"] = table.n_observations.map({1: "low", 2: "medium"}).fillna("high")
    table.loc[table.score_std > analysis["noise_sd_cutoff"], "confidence"] = table.loc[table.score_std > analysis["noise_sd_cutoff"], "confidence"].map({"high": "medium", "medium": "low", "low": "low"})
    table["priority"] = table.apply(lambda r: severity_weight.get(r.severity, 0) * r.confidence_score, axis=1)
    return table.sort_values(["priority", "student_id", "topic_id"], ascending=[False, True, True]).reset_index(drop=True)
