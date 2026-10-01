"""
scenarios.py -- the *description* of the synthetic world.

Everything in this file is plain data: no randomness, no file I/O, no pandas.
Two kinds of things live here:

  1. The school itself (teachers, topics, classes, timetable, curriculum,
     assessment plan).
  2. The PLANTED PATTERNS (which student gets which learning gap, trend, ...)
     and the numeric parameters of the score model.

Keeping this separate from the generation code means you can change the
world (add a class, add a topic, move an exam) without touching any logic.
"""
from __future__ import annotations

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Reproducibility and calendar
# ---------------------------------------------------------------------------
SEED = 42
TERM_START = date(2025, 9, 8)   # Monday of week 1
AS_OF = date(2025, 12, 19)      # "today" for the live analysis (Friday, week 15)


def week_start(week: int) -> date:
    """Monday of the given school week (week 1 = TERM_START)."""
    return TERM_START + timedelta(weeks=week - 1)


def week_end(week: int) -> date:
    """Friday of the given school week."""
    return week_start(week) + timedelta(days=4)


def week_of(d: date) -> int:
    """School week number (1-based) that a date falls in."""
    return (d - TERM_START).days // 7 + 1


# ---------------------------------------------------------------------------
# Parameters of the score model:
#   score = ability + topic_effect + planted_effect + trend_effect + noise
# ---------------------------------------------------------------------------
ABILITY_MEAN = 68.0
ABILITY_SD = 10.0
ABILITY_MIN = 55.0        # "typical" students are truncated to [55, 88] so that
ABILITY_MAX = 88.0        # unplanted students rarely look like they have a gap
TOPIC_EFFECT_SD = 5.0     # persistent per (student, topic) strength/weakness
NOISE_SD = 6.0            # assessment-to-assessment randomness
TREND_CENTER_WEEK = 9     # trend effect is 0 at this week (keeps mean ability intact)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
SUBJECTS = [("MATH", "Mathematics")]

TEACHERS = [
    ("T01", "Ms. Demir"),
    ("T02", "Mr. Aksoy"),
    ("T03", "Mr. Kaya"),
]

# (topic_id, topic_name, subject_id, grade_level)
TOPICS = [
    ("ALG", "Algebra", "MATH", 10),
    ("FUNC", "Functions", "MATH", 10),
    ("GEO", "Geometry", "MATH", 10),
    ("TRIG", "Trigonometry", "MATH", 10),
    ("DERIV", "Derivatives", "MATH", 10),
]

# 10B and 10C deliberately share the same timetable slots (parallel classes),
# so the matcher will meet a schedule clash. 10A has different slots.
CLASSES = [
    {"class_id": "10A", "class_name": "Grade 10A Mathematics", "grade_level": 10,
     "subject_id": "MATH", "teacher_id": "T01", "capacity": 24, "n_students": 20,
     "slots": [("Mon", 2), ("Wed", 4), ("Fri", 1)]},
    {"class_id": "10B", "class_name": "Grade 10B Mathematics", "grade_level": 10,
     "subject_id": "MATH", "teacher_id": "T02", "capacity": 24, "n_students": 20,
     "slots": [("Tue", 3), ("Thu", 2), ("Fri", 4)]},
    {"class_id": "10C", "class_name": "Grade 10C Mathematics", "grade_level": 10,
     "subject_id": "MATH", "teacher_id": "T03", "capacity": 24, "n_students": 20,
     "slots": [("Tue", 3), ("Thu", 2), ("Fri", 4)]},
]

# ---------------------------------------------------------------------------
# Curriculum: (topic_id, first_week, last_week). Dates are derived from weeks.
# 10A/10B finished Trigonometry and are now on Derivatives.
# 10C runs in a different order and is CURRENTLY teaching Trigonometry
# (weeks 13-18), which makes it the host for cross-class matching.
# ---------------------------------------------------------------------------
CURRICULUM_PLAN = {
    "10A": [("ALG", 1, 3), ("FUNC", 4, 6), ("GEO", 7, 9), ("TRIG", 10, 12), ("DERIV", 13, 15)],
    "10B": [("ALG", 1, 3), ("FUNC", 4, 6), ("GEO", 7, 9), ("TRIG", 10, 12), ("DERIV", 13, 15)],
    "10C": [("ALG", 1, 4), ("GEO", 5, 8), ("FUNC", 9, 12), ("TRIG", 13, 18), ("DERIV", 19, 22)],
}

# ---------------------------------------------------------------------------
# Assessment plan per class: (name, type, week, [topics tested]).
# Assessment ids are "<class_id>-<sequence>", e.g. "10A-04" (the 10A midterm).
# The generator REFUSES to produce a score for a topic the class has not
# started by the assessment date, so a typo here fails loudly.
# ---------------------------------------------------------------------------
_AB_ASSESSMENTS = [
    ("Quiz 1", "quiz", 3, ["ALG"]),
    ("Quiz 2", "quiz", 5, ["ALG", "FUNC"]),
    ("Quiz 3", "quiz", 6, ["FUNC"]),
    ("Midterm", "midterm", 9, ["ALG", "FUNC", "GEO"]),
    ("Quiz 4", "quiz", 12, ["FUNC", "GEO", "TRIG"]),
    ("Quiz 5", "quiz", 13, ["FUNC", "TRIG", "DERIV"]),
    ("Mock Exam", "mock", 15, ["ALG", "FUNC", "GEO", "TRIG", "DERIV"]),
]

ASSESSMENT_PLAN = {
    "10A": _AB_ASSESSMENTS,
    "10B": _AB_ASSESSMENTS,
    "10C": [
        ("Quiz 1", "quiz", 3, ["ALG"]),
        ("Quiz 2", "quiz", 6, ["ALG", "GEO"]),
        ("Midterm", "midterm", 9, ["ALG", "GEO", "FUNC"]),
        ("Quiz 3", "quiz", 11, ["GEO", "FUNC"]),
        ("Quiz 4", "quiz", 14, ["FUNC", "TRIG"]),
        ("Mock Exam", "mock", 15, ["ALG", "GEO", "FUNC", "TRIG"]),
    ],
}

# ---------------------------------------------------------------------------
# The completed intervention: a Functions support session, weeks 10-11.
# Timing matters: at the day before it starts (2025-11-09) the students already
# have Algebra, Functions and Geometry scores, so a baseline can be computed.
# ---------------------------------------------------------------------------
INTERVENTION_ID = "INT_FUNC_01"
INTERVENTION_TOPIC = "FUNC"
INTERVENTION_START = week_start(10)   # Monday 2025-11-10
INTERVENTION_END = week_end(11)       # Friday 2025-11-21

# ---------------------------------------------------------------------------
# PLANTED SCENARIOS (student ids are explicit so tests are deterministic)
# ---------------------------------------------------------------------------
# 1. Trigonometry gap: 5 students in 10A, 4 in 10B. Trig is 30-38 pts below own level.
TRIG_GAP_STUDENTS = ["S003", "S007", "S011", "S014", "S018",
                     "S023", "S028", "S031", "S036"]
TRIG_GAP_DROP = (32.0, 40.0)

# A planted gap must actually land in "low" territory (< 50). If a student with
# ability 88 lost 30 points they would still score 58 and the "gap" would be
# invisible, so students who receive a topic gap draw their general ability
# from this narrower band: gap topic ~ 20-38, other topics ~ 55-75.
GAP_STUDENT_ABILITY = (60.0, 70.0)

# 2. Functions gap + support session participants (attended) ...
FUNC_PARTICIPANTS = ["S002", "S005", "S009", "S016", "S022", "S026", "S033", "S038"]
FUNC_NON_RESPONDERS = ["S009", "S038"]        # attended, but barely improved
# ... and the comparison group: same gap, did not attend.
FUNC_COMPARISON = ["S004", "S012", "S018", "S025", "S030", "S039"]
FUNC_NO_SHOW = ["S039"]                       # approved for the session, did not turn up
FUNC_GAP_DROP = (32.0, 40.0)                  # Functions gap before/during the session
RESPONDER_BOOST = (14.0, 24.0)                # extra points after the session
NON_RESPONDER_BOOST = (-2.0, 4.0)
COMPARISON_DRIFT = (2.0, 6.0)                 # modest improvement without help

# 3. Generally low performance (broad difficulty): low ability on every topic.
BROAD_STUDENTS = ["S006", "S027", "S045", "S052"]
BROAD_ABILITY = (38.0, 45.0)

# 4./5. Trends: +/- 1.5 points per week on every topic.
IMPROVING_STUDENTS = ["S008", "S013", "S029", "S034"]
IMPROVING_ABILITY = (60.0, 66.0)
IMPROVING_SLOPE = 1.5
DECLINING_STUDENTS = ["S001", "S017", "S024", "S035"]
DECLINING_ABILITY = (72.0, 78.0)
DECLINING_SLOPE = -1.5

# 6. Class-wide difficulty: the whole of 10B does badly on Derivatives.
CLASS_WIDE_CLASS = "10B"
CLASS_WIDE_TOPIC = "DERIV"
CLASS_WIDE_DROP = 22.0

# 7. Single low observation: a low Derivatives score, but the student missed
#    Quiz 5, so only ONE Derivatives observation exists (must stay low-confidence).
SINGLE_LOW_TOPIC = "DERIV"
SINGLE_LOW_DROP = 28.0
SINGLE_LOW_ABILITY = (60.0, 70.0)        # so the one Derivatives score is really < 50
SINGLE_LOW_OBS = {"S010": "10A-06", "S019": "10A-06"}   # student -> assessment missed

# 8. Missing attendance: absent for one assessment => NO score rows (never 0).
MISSING_ATTENDANCE = {"S015": "10A-04", "S021": "10B-06", "S044": "10C-03"}
