"""Cross-class lesson matching. Suggestions never become approvals here."""
import pandas as pd


def match_cross_class_students(gap_table, students, classes, curriculum, schedules, as_of_date, config):
    rec = config["recommendation"]
    as_of = pd.Timestamp(as_of_date)
    students_by_id = students.set_index("student_id")["class_id"].to_dict()
    class_rows = classes.set_index("class_id").to_dict("index")
    schedule_map = {cid: set(zip(g.day, g.period)) for cid, g in schedules.groupby("class_id")}
    approved_seats = {}
    rows = []

    candidates = gap_table[(gap_table.category == "SPECIFIC_GAP") & (gap_table.confidence.isin(["medium", "high"])) & (gap_table.severity.isin(["moderate", "significant"])) ].copy()
    for row in candidates.itertuples(index=False):
        home = students_by_id[row.student_id]
        home_slots = schedule_map.get(home, set())
        candidates_for_student = []
        for host, meta in class_rows.items():
            if host == home:
                continue
            cur = curriculum[(curriculum.class_id == host) & (curriculum.topic_id == row.topic_id)]
            if cur.empty:
                candidates_for_student.append((host, "NO_MATCH_TOPIC", "Host class is not currently teaching this topic.")); continue
            cur = cur.iloc[0]
            start, end = pd.Timestamp(cur.start_date), pd.Timestamp(cur.end_date)
            if not (start <= as_of <= end):
                candidates_for_student.append((host, "NO_MATCH_TOPIC", "Host class is not currently teaching this topic.")); continue
            if abs(int(meta["grade_level"]) - int(row.grade_level)) > 1:
                candidates_for_student.append((host, "NO_MATCH_LEVEL", "Host class is outside the supported grade range.")); continue
            days_remaining = (end - as_of).days
            if days_remaining < rec["min_days_remaining"]:
                candidates_for_student.append((host, "UNIT_ENDING_SOON", f"Only {days_remaining} days remain in the unit.")); continue
            compatible = sorted(schedule_map.get(host, set()) - home_slots)
            if not compatible:
                candidates_for_student.append((host, "NO_MATCH_SCHEDULE", "Host lessons overlap with the student's home-class timetable.")); continue
            enrolled = int((students.class_id == host).sum())
            seats = int(meta["capacity"]) - enrolled - approved_seats.get(host, 0)
            if seats <= 0:
                candidates_for_student.append((host, "NO_CAPACITY", "No seats remain in the host class.")); continue
            candidates_for_student.append((host, "SUGGESTED", f"Compatible slots: {', '.join(f'{d} P{p}' for d,p in compatible)}."))

        suggested = [x for x in candidates_for_student if x[1] == "SUGGESTED"]
        if suggested:
            host, status, reason = sorted(suggested, key=lambda x: x[0])[0]
            approved_seats[host] = approved_seats.get(host, 0) + 1
            meta = class_rows[host]
            slots = sorted(schedule_map[host] - home_slots)
            rows.append({"student_id": row.student_id, "topic_id": row.topic_id, "host_class_id": host, "host_teacher_id": meta["teacher_id"], "compatible_slots": "; ".join(f"{d} P{p}" for d,p in slots), "seats_available": int(meta["capacity"]) - int((students.class_id == host).sum()) - approved_seats[host] + 1, "match_status": status, "reason": reason})
        else:
            priority = {"NO_CAPACITY": 0, "NO_MATCH_SCHEDULE": 1, "UNIT_ENDING_SOON": 2, "NO_MATCH_LEVEL": 3, "NO_MATCH_TOPIC": 4}
            host, status, reason = sorted(candidates_for_student, key=lambda item: priority.get(item[1], 99))[0] if candidates_for_student else (None, "NO_MATCH_TOPIC", "No compatible host class found.")
            rows.append({"student_id": row.student_id, "topic_id": row.topic_id, "host_class_id": host, "host_teacher_id": class_rows[host]["teacher_id"] if host else None, "compatible_slots": "", "seats_available": 0, "match_status": status, "reason": reason})
    return pd.DataFrame(rows)
