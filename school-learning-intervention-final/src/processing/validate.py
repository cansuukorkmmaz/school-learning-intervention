"""Structural and relational validation for raw school data."""
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    @property
    def valid(self) -> bool:
        return not self.errors


def _require_columns(tables, required, report):
    for table, columns in required.items():
        missing = set(columns) - set(tables[table].columns)
        if missing:
            report.errors.append(f"{table}: missing columns {sorted(missing)}")


def validate_tables(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    required = {
        "students": ["student_id", "class_id"],
        "classes": ["class_id", "grade_level", "subject_id", "teacher_id", "capacity"],
        "topics": ["topic_id", "subject_id", "grade_level"],
        "assessments": ["assessment_id", "class_id", "date"],
        "scores": ["student_id", "assessment_id", "topic_id", "score_pct"],
        "curriculum": ["class_id", "topic_id", "start_date", "end_date"],
    }
    _require_columns(tables, required, report)
    if report.errors:
        return report

    students, classes, topics = tables["students"], tables["classes"], tables["topics"]
    assessments, scores, curriculum = tables["assessments"], tables["scores"], tables["curriculum"]

    for name in ["students", "classes", "topics", "assessments"]:
        if tables[name].iloc[:, 0].duplicated().any():
            report.errors.append(f"{name}: duplicate primary IDs detected")

    if scores["score_pct"].isna().any():
        report.errors.append("scores: score_pct contains missing values")
    elif ((scores["score_pct"] < 0) | (scores["score_pct"] > 100)).any():
        report.errors.append("scores: score_pct must be between 0 and 100")

    key = ["student_id", "assessment_id", "topic_id"]
    if scores.duplicated(key).any():
        report.errors.append("scores: duplicate (student_id, assessment_id, topic_id) rows")

    student_ids, class_ids, topic_ids = set(students.student_id), set(classes.class_id), set(topics.topic_id)
    assessment_ids = set(assessments.assessment_id)
    for column, allowed in [("student_id", student_ids), ("assessment_id", assessment_ids), ("topic_id", topic_ids)]:
        unknown = set(scores[column].dropna()) - allowed
        if unknown:
            report.errors.append(f"scores: unknown {column} values {sorted(unknown)}")

    for column, allowed in [("class_id", class_ids), ("topic_id", topic_ids)]:
        unknown = set(curriculum[column].dropna()) - allowed
        if unknown:
            report.errors.append(f"curriculum: unknown {column} values {sorted(unknown)}")

    parsed_assessment_dates = pd.to_datetime(assessments["date"], errors="coerce")
    if parsed_assessment_dates.isna().any():
        report.errors.append("assessments: unparsable date values")

    parsed_start = pd.to_datetime(curriculum["start_date"], errors="coerce")
    parsed_end = pd.to_datetime(curriculum["end_date"], errors="coerce")
    if parsed_start.isna().any() or parsed_end.isna().any():
        report.errors.append("curriculum: unparsable date values")
    elif (parsed_start > parsed_end).any():
        report.errors.append("curriculum: start_date must be on or before end_date")

    if set(students.class_id) - class_ids:
        report.errors.append("students: unknown class_id values")

    assessment_class = assessments.set_index("assessment_id")["class_id"]
    score_classes = scores["assessment_id"].map(assessment_class)
    student_classes = scores["student_id"].map(students.set_index("student_id")["class_id"])
    if (score_classes != student_classes).any():
        report.errors.append("scores: student does not belong to the assessment class")

    # Relational checks that are useful but not necessarily fatal.
    assessment_counts = scores.groupby("student_id")["assessment_id"].nunique()
    low_count = int((assessment_counts < 3).sum())
    if low_count:
        report.warnings.append(f"{low_count} students have fewer than 3 assessments")

    return report


def raise_if_invalid(report: ValidationReport) -> None:
    if report.errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(report.errors))
