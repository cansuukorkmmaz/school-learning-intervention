"""
Sanity tests for the synthetic data generator (Step 1 verification).

These check that the generator (a) is reproducible, (b) never produces
impossible data, and (c) really planted the patterns we intended.
They deliberately use simple pandas, NOT the future gap-detection code.

Run from the project root:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.data_generation import scenarios as sc
from src.data_generation.generators import generate_all


@pytest.fixture(scope="module")
def tables():
    return generate_all(seed=sc.SEED)


def _obs(tables):
    """scores joined with assessment date and class."""
    return tables["scores"].merge(tables["assessments"], on="assessment_id")


def test_same_seed_gives_identical_data(tables):
    again = generate_all(seed=sc.SEED)
    for name, df in tables.items():
        pd.testing.assert_frame_equal(df, again[name])


def test_different_seed_changes_scores(tables):
    other = generate_all(seed=sc.SEED + 1)
    assert not tables["scores"]["score_pct"].equals(other["scores"]["score_pct"])


def test_no_score_before_topic_is_taught(tables):
    m = _obs(tables).merge(tables["curriculum"], on=["class_id", "topic_id"])
    assert (m["date"] >= m["start_date"]).all()


def test_10c_has_no_trig_before_it_starts_teaching_it(tables):
    o = _obs(tables)
    trig_10c = o[(o.class_id == "10C") & (o.topic_id == "TRIG")]
    assert not trig_10c.empty
    assert trig_10c["date"].min() >= sc.week_start(13)


def test_absence_means_no_rows_not_zero(tables):
    scores = tables["scores"]
    for sid, aid in {**sc.MISSING_ATTENDANCE, **sc.SINGLE_LOW_OBS}.items():
        assert scores[(scores.student_id == sid) & (scores.assessment_id == aid)].empty
    assert (scores["score_pct"] > 0).all()          # nobody was silently given a 0


def test_trig_gap_students_are_far_below_their_own_other_topics(tables):
    s = tables["scores"]
    for sid in sc.TRIG_GAP_STUDENTS:
        mine = s[s.student_id == sid]
        trig = mine[mine.topic_id == "TRIG"]["score_pct"].mean()
        other = mine[mine.topic_id != "TRIG"]["score_pct"].mean()
        assert other - trig > 20, sid


def test_broad_students_are_low_everywhere(tables):
    s = tables["scores"]
    for sid in sc.BROAD_STUDENTS:
        assert s[s.student_id == sid]["score_pct"].mean() < 55, sid


def test_trend_students_have_the_planted_direction(tables):
    o = _obs(tables)
    o["week"] = o["date"].map(sc.week_of)
    per_assessment = o.groupby(["student_id", "week"])["score_pct"].mean().reset_index()

    def slope(sid):
        d = per_assessment[per_assessment.student_id == sid]
        return float(np.polyfit(d["week"], d["score_pct"], 1)[0])   # points per week

    assert all(slope(s) > 0.5 for s in sc.IMPROVING_STUDENTS)
    assert all(slope(s) < -0.5 for s in sc.DECLINING_STUDENTS)


def test_single_low_students_have_exactly_one_deriv_score(tables):
    s = tables["scores"]
    for sid in sc.SINGLE_LOW_OBS:
        assert len(s[(s.student_id == sid) & (s.topic_id == "DERIV")]) == 1


def _delta(tables, students):
    """before = mean of last 2 FUNC scores before start; after = first 2 after end."""
    o = _obs(tables)
    o = o[o.topic_id == "FUNC"]
    out = []
    for sid in students:
        d = o[o.student_id == sid].sort_values("date")
        before = d[d.date < sc.INTERVENTION_START].tail(2)["score_pct"].mean()
        after = d[d.date > sc.INTERVENTION_END].head(2)["score_pct"].mean()
        out.append(after - before)
    return pd.Series(out)


def test_intervention_participants_improved_more_than_comparison(tables):
    part = tables["intervention_participants"]
    attended = part[part.attended]["student_id"].tolist()
    assert sorted(attended) == sorted(sc.FUNC_PARTICIPANTS)
    d_p, d_c = _delta(tables, attended).mean(), _delta(tables, sc.FUNC_COMPARISON).mean()
    assert d_p > d_c + 5, (d_p, d_c)
    assert 0 <= d_c <= 12


def test_class_wide_derivatives_gap_in_10b(tables):
    o = _obs(tables)
    d = o[o.topic_id == "DERIV"].groupby("class_id")["score_pct"].median()
    assert d["10B"] < 52 and d["10A"] > 60


# --- added after the first run exposed gaps that were not actually "low" -------
def test_planted_trig_gap_is_really_low_not_just_relatively_low(tables):
    s = tables["scores"]
    for sid in sc.TRIG_GAP_STUDENTS:
        trig = s[(s.student_id == sid) & (s.topic_id == "TRIG")]["score_pct"].mean()
        assert trig < 50, (sid, trig)


def test_functions_gap_students_are_low_before_the_session(tables):
    o = _obs(tables)
    o = o[(o.topic_id == "FUNC") & (o.date < sc.INTERVENTION_START)]
    for sid in sc.FUNC_PARTICIPANTS + sc.FUNC_COMPARISON:
        before = o[o.student_id == sid].tail(2)["score_pct"].mean()
        assert before < 50, (sid, before)


def test_single_low_observation_is_low(tables):
    s = tables["scores"]
    for sid in sc.SINGLE_LOW_OBS:
        row = s[(s.student_id == sid) & (s.topic_id == "DERIV")]
        assert row["score_pct"].iloc[0] < 50, sid
