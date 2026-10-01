"""
generators.py -- pure generation logic.

Every function here takes plain inputs and returns pandas DataFrames (or small
helper objects). Nothing in this file reads or writes files; that is writer.py's
job. This means you can call generate_all() from a test or a notebook and get
the whole dataset in memory.

Generation order (matches the 12 steps we agreed on):
   1 students   2 teachers   3 subjects   4 topics   5 classes   6 schedule
   7 curriculum 8 assessments 9 planted scenarios 10 scores
  11 interventions   12 ground truth
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from . import scenarios as sc


# ===========================================================================
# Steps 1-8: the "school" tables (deterministic, no randomness)
# ===========================================================================
def make_students() -> pd.DataFrame:
    """Step 1. Sequential anonymous ids S001.., assigned class by class."""
    rows, n = [], 1
    for cls in sc.CLASSES:
        for _ in range(cls["n_students"]):
            rows.append({"student_id": f"S{n:03d}", "class_id": cls["class_id"]})
            n += 1
    return pd.DataFrame(rows)


def make_teachers() -> pd.DataFrame:
    """Step 2."""
    return pd.DataFrame(sc.TEACHERS, columns=["teacher_id", "teacher_name"])


def make_subjects() -> pd.DataFrame:
    """Step 3."""
    return pd.DataFrame(sc.SUBJECTS, columns=["subject_id", "subject_name"])


def make_topics() -> pd.DataFrame:
    """Step 4."""
    return pd.DataFrame(sc.TOPICS, columns=["topic_id", "topic_name", "subject_id", "grade_level"])


def make_classes() -> pd.DataFrame:
    """Step 5."""
    cols = ["class_id", "class_name", "grade_level", "subject_id", "teacher_id", "capacity"]
    return pd.DataFrame([{c: cls[c] for c in cols} for cls in sc.CLASSES])


def make_class_schedule() -> pd.DataFrame:
    """Step 6. One row per weekly lesson slot."""
    rows = [{"class_id": cls["class_id"], "day": day, "period": period}
            for cls in sc.CLASSES for day, period in cls["slots"]]
    return pd.DataFrame(rows)


def make_curriculum() -> pd.DataFrame:
    """Step 7. Teaching window per (class, topic): Monday of first week to Friday of last week."""
    rows = [{"class_id": cid, "topic_id": tid,
             "start_date": sc.week_start(w1), "end_date": sc.week_end(w2)}
            for cid, plan in sc.CURRICULUM_PLAN.items() for tid, w1, w2 in plan]
    return pd.DataFrame(rows)


def make_assessments() -> pd.DataFrame:
    """Step 8. One row per assessment; a Friday in the planned week."""
    rows = []
    for cid, plan in sc.ASSESSMENT_PLAN.items():
        for seq, (name, kind, week, _topics) in enumerate(plan, start=1):
            rows.append({"assessment_id": f"{cid}-{seq:02d}", "class_id": cid,
                         "assessment_name": name, "assessment_type": kind,
                         "date": sc.week_end(week)})
    return pd.DataFrame(rows)


def assessment_topic_map() -> dict[str, list[str]]:
    """Which topics each assessment tests (internal; scores.csv carries it implicitly)."""
    return {f"{cid}-{seq:02d}": list(topics)
            for cid, plan in sc.ASSESSMENT_PLAN.items()
            for seq, (_n, _k, _w, topics) in enumerate(plan, start=1)}


# ===========================================================================
# Step 9: planted scenarios
# ===========================================================================
@dataclass(frozen=True)
class Rule:
    """A planted effect on ONE topic, optionally limited to a date range."""
    topic_id: str
    effect: float                      # points added to the score (negative = worse)
    start: Optional[date] = None       # applies to assessments on/after this date
    end: Optional[date] = None         # applies to assessments on/before this date

    def applies(self, topic_id: str, on: date) -> bool:
        if topic_id != self.topic_id:
            return False
        if self.start is not None and on < self.start:
            return False
        if self.end is not None and on > self.end:
            return False
        return True


@dataclass
class ScenarioPlan:
    """Everything the score generator needs to know about planted patterns."""
    rules: dict = field(default_factory=lambda: defaultdict(list))  # student -> [Rule]
    ability_range: dict = field(default_factory=dict)               # student -> (lo, hi)
    slopes: dict = field(default_factory=dict)                      # student -> points/week
    absences: set = field(default_factory=set)                      # {(student, assessment)}
    truth: list = field(default_factory=list)                       # rows for ground_truth


def build_scenario_plan(students: pd.DataFrame, rng: np.random.Generator) -> ScenarioPlan:
    """Turn the scenario constants into concrete per-student rules.

    Random draws (e.g. how big each student's gap is) happen here, once, in a
    fixed order, so the same seed always plants the same gaps.
    """
    plan = ScenarioPlan()

    def note(sid, scenario, topic="", signal="", notes=""):
        plan.truth.append({"student_id": sid, "scenario": scenario, "topic_id": topic,
                           "expected_signal": signal, "notes": notes})

    during_or_before = sc.INTERVENTION_END
    after = sc.INTERVENTION_END + timedelta(days=1)

    # 1. Trigonometry gap ---------------------------------------------------
    for sid in sc.TRIG_GAP_STUDENTS:
        drop = rng.uniform(*sc.TRIG_GAP_DROP)
        plan.ability_range[sid] = sc.GAP_STUDENT_ABILITY
        plan.rules[sid].append(Rule("TRIG", -drop))
        note(sid, "TRIG_GAP", "TRIG", "SPECIFIC_GAP", f"Trig planted {drop:.0f} pts below own level")

    # 2. Functions gap: participants (some respond, some do not) ---------------
    for sid in sc.FUNC_PARTICIPANTS:
        drop = rng.uniform(*sc.FUNC_GAP_DROP)
        responder = sid not in sc.FUNC_NON_RESPONDERS
        boost = rng.uniform(*(sc.RESPONDER_BOOST if responder else sc.NON_RESPONDER_BOOST))
        plan.ability_range[sid] = sc.GAP_STUDENT_ABILITY
        plan.rules[sid] += [Rule("FUNC", -drop, end=during_or_before),
                            Rule("FUNC", -drop + boost, start=after)]
        note(sid, "FUNC_PARTICIPANT_RESPONDER" if responder else "FUNC_PARTICIPANT_NONRESPONDER",
             "FUNC", "IMPROVED" if responder else "NO_CLEAR_CHANGE",
             f"gap -{drop:.0f}, boost {boost:+.0f} after session")

    # 2b. Functions gap: comparison group (same gap, small natural drift) ------
    for sid in sc.FUNC_COMPARISON:
        drop = rng.uniform(*sc.FUNC_GAP_DROP)
        drift = rng.uniform(*sc.COMPARISON_DRIFT)
        plan.ability_range[sid] = sc.GAP_STUDENT_ABILITY
        plan.rules[sid] += [Rule("FUNC", -drop, end=during_or_before),
                            Rule("FUNC", -drop + drift, start=after)]
        extra = "; approved but did not attend" if sid in sc.FUNC_NO_SHOW else ""
        note(sid, "FUNC_COMPARISON", "FUNC", "NO_CLEAR_CHANGE",
             f"gap -{drop:.0f}, drift {drift:+.0f} without session{extra}")

    # 3. Broad difficulty: low ability everywhere ------------------------------
    for sid in sc.BROAD_STUDENTS:
        plan.ability_range[sid] = sc.BROAD_ABILITY
        note(sid, "BROAD_DIFFICULTY", "", "BROAD_DIFFICULTY", "ability 38-45 on every topic")

    # 4./5. Trends --------------------------------------------------------------
    for sid in sc.IMPROVING_STUDENTS:
        plan.ability_range[sid] = sc.IMPROVING_ABILITY
        plan.slopes[sid] = sc.IMPROVING_SLOPE
        note(sid, "IMPROVING_TREND", "", "IMPROVING", f"{sc.IMPROVING_SLOPE:+.1f} pts/week")
    for sid in sc.DECLINING_STUDENTS:
        plan.ability_range[sid] = sc.DECLINING_ABILITY
        plan.slopes[sid] = sc.DECLINING_SLOPE
        note(sid, "DECLINING_TREND", "", "DECLINING", f"{sc.DECLINING_SLOPE:+.1f} pts/week")

    # 6. Class-wide difficulty ---------------------------------------------------
    in_class = students.loc[students["class_id"] == sc.CLASS_WIDE_CLASS, "student_id"]
    for sid in in_class:
        plan.rules[sid].append(Rule(sc.CLASS_WIDE_TOPIC, -sc.CLASS_WIDE_DROP))
        note(sid, "CLASS_WIDE_DERIV", sc.CLASS_WIDE_TOPIC, "CLASS_WIDE",
             f"whole class {sc.CLASS_WIDE_DROP:.0f} pts lower on {sc.CLASS_WIDE_TOPIC}")

    # 7. Single low observation ---------------------------------------------------
    for sid, missed in sc.SINGLE_LOW_OBS.items():
        plan.ability_range[sid] = sc.SINGLE_LOW_ABILITY
        plan.rules[sid].append(Rule(sc.SINGLE_LOW_TOPIC, -sc.SINGLE_LOW_DROP))
        plan.absences.add((sid, missed))
        note(sid, "SINGLE_LOW_OBS", sc.SINGLE_LOW_TOPIC, "WATCH",
             f"only one {sc.SINGLE_LOW_TOPIC} score (missed {missed}); must not be a confident gap")

    # 8. Missing attendance ---------------------------------------------------------
    for sid, missed in sc.MISSING_ATTENDANCE.items():
        plan.absences.add((sid, missed))
        note(sid, "MISSING_ATTENDANCE", "", "NO_SCORE_ROW", f"absent for {missed}; no row, never 0")

    return plan


# ===========================================================================
# Step 10: the score model
#   score = ability + topic_effect + planted_effect + trend_effect + noise
# ===========================================================================
def draw_abilities(students: pd.DataFrame, plan: ScenarioPlan, rng: np.random.Generator) -> dict:
    """General level of each student. Scenario students may override the range."""
    out = {}
    for sid in students["student_id"]:
        if sid in plan.ability_range:
            out[sid] = float(rng.uniform(*plan.ability_range[sid]))
        else:
            out[sid] = float(np.clip(rng.normal(sc.ABILITY_MEAN, sc.ABILITY_SD),
                                     sc.ABILITY_MIN, sc.ABILITY_MAX))
    return out


def draw_topic_effects(students: pd.DataFrame, topics: pd.DataFrame,
                       rng: np.random.Generator) -> dict:
    """Persistent strengths/weaknesses: one number per (student, topic)."""
    return {(sid, tid): float(rng.normal(0.0, sc.TOPIC_EFFECT_SD))
            for sid in students["student_id"] for tid in topics["topic_id"]}


def generate_scores(students, assessments, abilities, topic_effects,
                    plan: ScenarioPlan, rng: np.random.Generator) -> pd.DataFrame:
    """One row per (student, assessment, topic). Absent => no row."""
    topic_map = assessment_topic_map()
    by_class = students.groupby("class_id")["student_id"].apply(list).to_dict()
    rows = []
    for a in assessments.sort_values(["date", "assessment_id"]).itertuples(index=False):
        week = sc.week_of(a.date)
        for sid in by_class[a.class_id]:
            if (sid, a.assessment_id) in plan.absences:
                continue                                    # absent => no row (NOT zero)
            trend = plan.slopes.get(sid, 0.0) * (week - sc.TREND_CENTER_WEEK)
            for tid in topic_map[a.assessment_id]:
                planted = sum(r.effect for r in plan.rules.get(sid, []) if r.applies(tid, a.date))
                noise = rng.normal(0.0, sc.NOISE_SD)
                raw = abilities[sid] + topic_effects[(sid, tid)] + planted + trend + noise
                rows.append({"student_id": sid, "assessment_id": a.assessment_id,
                             "topic_id": tid, "score_pct": round(float(np.clip(raw, 0, 100)), 1)})
    return pd.DataFrame(rows)


# ===========================================================================
# Step 11: intervention data
# ===========================================================================
def make_interventions() -> pd.DataFrame:
    n_part = sum(1 for s in sc.FUNC_PARTICIPANTS)
    return pd.DataFrame([{
        "intervention_id": sc.INTERVENTION_ID,
        "intervention_type": "SUPPORT_SESSION",
        "topic_id": sc.INTERVENTION_TOPIC,
        "host_class_id": None,                       # not cross-class
        "start_date": sc.INTERVENTION_START,
        "end_date": sc.INTERVENTION_END,
        "status": "COMPLETED",
        "rationale": (f"{n_part + len(sc.FUNC_NO_SHOW)} Grade 10 students showed a Functions gap "
                      f"as of {sc.INTERVENTION_START - timedelta(days=1)}; "
                      f"support session approved by home teachers."),
    }])


def make_intervention_participants() -> pd.DataFrame:
    """Approved students. `attended` is what defines a real participant later."""
    rows = [{"intervention_id": sc.INTERVENTION_ID, "student_id": sid,
             "home_teacher_decision": "APPROVED",
             "target_teacher_decision": "NOT_REQUIRED",   # no host class in a support session
             "attended": True} for sid in sc.FUNC_PARTICIPANTS]
    rows += [{"intervention_id": sc.INTERVENTION_ID, "student_id": sid,
              "home_teacher_decision": "APPROVED",
              "target_teacher_decision": "NOT_REQUIRED",
              "attended": False} for sid in sc.FUNC_NO_SHOW]
    return pd.DataFrame(rows)


# ===========================================================================
# Step 12: ground truth (for tests / evaluation ONLY)
# ===========================================================================
def make_ground_truth(plan: ScenarioPlan) -> pd.DataFrame:
    cols = ["student_id", "scenario", "topic_id", "expected_signal", "notes"]
    return pd.DataFrame(plan.truth, columns=cols).sort_values(["scenario", "student_id"],
                                                             ignore_index=True)


# ===========================================================================
# Self-check: "do not generate impossible data"
# (This is the generator checking ITSELF. Step 2's validate.py will later check
#  any incoming data independently, because real data is not produced by us.)
# ===========================================================================
def check_invariants(tables: dict, plan: Optional[ScenarioPlan] = None) -> None:
    scores, students = tables["scores"], tables["students"]
    assessments, curriculum = tables["assessments"], tables["curriculum"]

    if scores[["student_id", "assessment_id", "topic_id"]].duplicated().any():
        raise ValueError("duplicate (student, assessment, topic) rows")
    if not scores["score_pct"].between(0, 100).all():
        raise ValueError("score outside 0-100")

    m = scores.merge(assessments[["assessment_id", "class_id", "date"]], on="assessment_id")
    m = m.merge(students.rename(columns={"class_id": "home_class_id"}), on="student_id")
    if (m["class_id"] != m["home_class_id"]).any():
        raise ValueError("student has a score for an assessment of another class")

    m = m.merge(curriculum, on=["class_id", "topic_id"], how="left")
    if m["start_date"].isna().any():
        raise ValueError("score for a topic that is not in the class curriculum")
    if (m["date"] < m["start_date"]).any():
        raise ValueError("score for a topic the class had not started yet")
    if (assessments["date"] > sc.AS_OF).any():
        raise ValueError("assessment dated after AS_OF")

    if plan is not None:
        have = set(zip(scores["student_id"], scores["assessment_id"]))
        for pair in plan.absences:
            if pair in have:
                raise ValueError(f"absent student {pair} still has score rows")

    known = set(students["student_id"])
    if not set(tables["ground_truth"]["student_id"]) <= known:
        raise ValueError("ground_truth mentions unknown students")
    if not set(tables["intervention_participants"]["student_id"]) <= known:
        raise ValueError("participants mention unknown students")


# ===========================================================================
# Orchestration
# ===========================================================================
def generate_all(seed: int = sc.SEED) -> dict[str, pd.DataFrame]:
    """Build every table in memory. Same seed => identical output.

    Four independent random streams (scenarios, abilities, topic effects,
    noise) are spawned from the one seed, so changing how one component
    consumes randomness does not silently reshuffle the others.
    """
    r_scn, r_abil, r_topic, r_noise = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))

    students = make_students()
    topics = make_topics()
    assessments = make_assessments()

    plan = build_scenario_plan(students, r_scn)
    abilities = draw_abilities(students, plan, r_abil)
    topic_effects = draw_topic_effects(students, topics, r_topic)
    scores = generate_scores(students, assessments, abilities, topic_effects, plan, r_noise)

    tables = {
        "subjects": make_subjects(),
        "teachers": make_teachers(),
        "topics": topics,
        "classes": make_classes(),
        "students": students,
        "class_schedule": make_class_schedule(),
        "curriculum": make_curriculum(),
        "assessments": assessments,
        "scores": scores,
        "interventions": make_interventions(),
        "intervention_participants": make_intervention_participants(),
        "ground_truth": make_ground_truth(plan),
    }
    check_invariants(tables, plan)
    return tables
