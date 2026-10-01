"""Streamlit interface for the school learning intervention prototype."""
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


st.set_page_config(
    page_title="Learning Intervention System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_pipeline(as_of: str):
    return run_pipeline(as_of_date=as_of, data_dir=PROJECT_ROOT / "data")


def format_topic(topic: str) -> str:
    labels = {
        "ALG": "Algebra",
        "FUNC": "Functions",
        "GEO": "Geometry",
        "TRIG": "Trigonometry",
        "DERIV": "Derivatives",
    }
    return labels.get(topic, topic)


def format_category(value: str) -> str:
    return value.replace("_", " ").title()


st.title("Learning Intervention System")
st.caption("Explainable, data-driven support for student learning decisions")

with st.sidebar:
    st.header("Analysis settings")
    as_of = str(st.date_input("Analysis date", value=date(2025, 12, 19)))
    st.caption("Only observations dated on or before the selected date are used.")

try:
    result = get_pipeline(as_of)
except Exception as exc:
    st.error(f"The analysis pipeline could not be loaded: {exc}")
    st.stop()

students = result["tables"]["students"]
gap = result["gap_table"].copy()
recs = result["recommendations"].copy()
matches = result["matches"].copy()
groups = result["support_groups"].copy()
outcome_summary = result["outcome_summary"].copy()

specific = gap[gap["category"] == "SPECIFIC_GAP"]
monitor = gap[(gap["category"] == "WATCH") | (gap["confidence"] == "low")]
suggested_matches = matches[matches["match_status"] == "SUGGESTED"] if not matches.empty else matches

metric_cols = st.columns(4)
metric_cols[0].metric("Students", int(students.student_id.nunique()))
metric_cols[1].metric("Specific gaps", int(len(specific)))
metric_cols[2].metric("Support groups", int(len(groups)))
metric_cols[3].metric("Cross-class matches", int(len(suggested_matches)))

st.divider()

tab_overview, tab_students, tab_interventions, tab_outcomes = st.tabs(
    ["Overview", "Student analysis", "Interventions", "Outcomes"]
)

with tab_overview:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Learning-gap distribution")
        category_counts = gap["category"].value_counts().rename_axis("category").reset_index(name="count")
        category_counts["category"] = category_counts["category"].map(format_category)
        st.bar_chart(category_counts.set_index("category"))

    with right:
        st.subheader("Priority gaps")
        priority_cols = [
            "student_id", "class_id", "topic_id", "topic_score", "personal_gap",
            "severity", "confidence", "priority",
        ]
        priority = specific.sort_values("priority", ascending=False)[priority_cols].head(10).copy()
        priority["topic_id"] = priority["topic_id"].map(format_topic)
        priority = priority.rename(columns={
            "student_id": "Student",
            "class_id": "Class",
            "topic_id": "Topic",
            "topic_score": "Score",
            "personal_gap": "Gap",
            "severity": "Severity",
            "confidence": "Confidence",
            "priority": "Priority",
        })
        st.dataframe(priority, use_container_width=True, hide_index=True)

    st.subheader("Class-level signals")
    class_summary = result["class_summary"].copy()
    if not class_summary.empty:
        class_summary["topic_id"] = class_summary["topic_id"].map(format_topic)
        st.dataframe(class_summary, use_container_width=True, hide_index=True)

    with st.expander("How recommendations are produced"):
        st.markdown(
            "The pipeline combines recency-weighted topic performance, a student's own baseline, "
            "class context, observation count and trend signals. Recommendations are suggestions only; "
            "teacher approval is required before an intervention is treated as approved."
        )

with tab_students:
    st.subheader("Student-level analysis")
    student_ids = sorted(students["student_id"].unique())
    selected_student = st.selectbox("Student", student_ids)

    student_gap = gap[gap["student_id"] == selected_student].copy().sort_values("priority", ascending=False)
    student_obs = result["observations"][result["observations"]["student_id"] == selected_student].copy()
    student_recs = recs[recs["student_id"] == selected_student].copy()

    student_info = students[students["student_id"] == selected_student].iloc[0]
    st.caption(f"Home class: {student_info['class_id']}")

    if not student_gap.empty:
        chart_data = student_gap[["topic_id", "topic_score", "personal_baseline"]].copy()
        chart_data["topic_id"] = chart_data["topic_id"].map(format_topic)
        chart_data = chart_data.set_index("topic_id")
        chart_data.columns = ["Current score", "Personal baseline"]
        st.bar_chart(chart_data)

        detail_cols = [
            "topic_id", "topic_score", "personal_baseline", "personal_gap",
            "class_gap", "n_observations", "trend_label", "confidence", "category",
        ]
        detail = student_gap[detail_cols].copy()
        detail["topic_id"] = detail["topic_id"].map(format_topic)
        st.dataframe(detail, use_container_width=True, hide_index=True)

    st.subheader("Why the system flagged this student")
    flagged = student_gap[student_gap["category"] == "SPECIFIC_GAP"]
    if flagged.empty:
        st.info("No specific learning gap is currently flagged for this student.")
    else:
        for row in flagged.itertuples(index=False):
            st.markdown(
                f"**{format_topic(row.topic_id)}:** score {row.topic_score:.1f}, "
                f"personal baseline {row.personal_baseline:.1f}, "
                f"gap {row.personal_gap:.1f} points, {int(row.n_observations)} observations, "
                f"confidence {row.confidence}."
            )

    if not student_recs.empty:
        st.subheader("Current recommendations")
        display = student_recs[["recommendation_type", "topic_id", "status", "rationale"]].copy()
        display["topic_id"] = display["topic_id"].map(lambda x: format_topic(x) if pd.notna(x) else "-")
        st.dataframe(display, use_container_width=True, hide_index=True)

with tab_interventions:
    st.subheader("Recommended interventions")

    type_labels = {
        "TARGETED_PRACTICE": "Targeted practice",
        "SUPPORT_SESSION": "Support session",
        "CROSS_CLASS_ATTENDANCE": "Cross-class attendance",
        "TEACHER_REVIEW": "Teacher review",
        "CLASS_WIDE_INSIGHT": "Class-wide insight",
        "MONITOR": "Monitor",
    }

    if recs.empty:
        st.info("No recommendations for the selected analysis date.")
    else:
        display = recs.copy()
        display["recommendation_type"] = display["recommendation_type"].map(type_labels).fillna(display["recommendation_type"])
        display["topic_id"] = display["topic_id"].map(lambda x: format_topic(x) if pd.notna(x) else "-")
        display = display.rename(columns={
            "recommendation_id": "ID",
            "recommendation_type": "Type",
            "student_id": "Student",
            "topic_id": "Topic",
            "group_id": "Group",
            "host_class_id": "Host class",
            "status": "Status",
            "rationale": "Rationale",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Cross-class matching")
    if matches.empty:
        st.info("No cross-class matches were generated.")
    else:
        match_display = matches.copy()
        match_display["topic_id"] = match_display["topic_id"].map(format_topic)
        st.dataframe(match_display, use_container_width=True, hide_index=True)

    st.caption("A cross-class suggestion is never an automatic approval. Teacher decisions remain part of the intervention process.")

with tab_outcomes:
    st.subheader("Intervention outcomes")
    if outcome_summary.empty:
        st.info("No completed intervention is available for the selected analysis date.")
    else:
        row = outcome_summary.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Participant change", f"{row.participant_mean_delta:+.1f}")
        c2.metric("Comparison change", f"{row.comparison_mean_delta:+.1f}")
        c3.metric("Difference", f"{row.difference_in_differences:+.1f}")

        st.dataframe(outcome_summary, use_container_width=True, hide_index=True)

        outcomes = result["outcomes"].copy()
        if not outcomes.empty:
            st.subheader("Student-level outcome changes")
            st.dataframe(outcomes, use_container_width=True, hide_index=True)

        st.caption(
            "These are descriptive changes, not causal estimates. The participants were not randomly assigned, "
            "so the comparison does not establish that the intervention caused the observed difference."
        )

with st.expander("Methodology and safeguards"):
    st.markdown(
        "**Data:** synthetic and anonymized.  "
        "\n\n**Decision logic:** explainable rules based on recency-weighted performance, personal baseline, "
        "class context, observation count and trends.  "
        "\n\n**Human oversight:** recommendations remain `SUGGESTED` until reviewed. The system does not rank teachers "
        "or automatically approve student movement between classes.  "
        "\n\n**Evaluation:** the hidden ground-truth dataset is used only by tests/evaluation and is not part of the "
        "recommendation path."
    )
