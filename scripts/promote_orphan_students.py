# scripts/promote_orphan_students.py
# Dry-run seeder for orphan student users -> create students + seed enrollments
# Usage:
#   python scripts/promote_orphan_students.py                  (dry run)
#   python scripts/promote_orphan_students.py --limit 200      (dry run, first 200)
#   python scripts/promote_orphan_students.py --commit         (apply)
#   python scripts/promote_orphan_students.py --commit --limit 500

from __future__ import annotations
import os, sys, argparse, random, math
from datetime import datetime
from bson import ObjectId

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db import col  # your helper

PROGRAM_CODE = "BSED-ENGLISH"
PROGRAM_NAME = "BSED English"
UNIVERSITY = {"university_name": "Sample University", "university_short": "SU"}
COLLEGE = {"college_name": "College of Education"}

# Academic calendar to use (2 sem + optional summer=3 supported by your code)
SCHOOL_YEARS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"]
TERMS_ORDER = [1, 2, 3]  # 1=Sem1, 2=Sem2, 3=Summer (some charts will simply show S3, which is OK)

# ------------- helpers -------------
def _nemail(e: str) -> str:
    return (e or "").strip().lower()

def _remark_for(grade):
    if grade is None:
        return "INC"
    try:
        g = float(grade)
    except Exception:
        return "INC"
    if g >= 75:
        return "PASSED"
    # ~10% chance to mark “DROP” instead of FAILED for realism
    return "DROP" if random.random() < 0.10 else "FAILED"

def _grade_random():
    """
    Weighted random:
      - 75–95 common
      - 60–74 sometimes
      - None (INC) occasionally
    """
    r = random.random()
    if r < 0.05:
        return None  # INC
    if r < 0.15:
        return random.randint(60, 74)
    return random.randint(75, 95)

def _ensure_counters():
    col("counters").update_one({"_id": "student_no"}, {"$setOnInsert": {"seq": 0}}, upsert=True)

def _next_student_no():
    """
    Use atomic counter for student_no -> "S00001"
    Also returns the plain integer for student_id if you want to reuse.
    """
    _ensure_counters()
    doc = col("counters").find_one_and_update(
        {"_id": "student_no"},
        {"$inc": {"seq": 1}},
        return_document=True,
        upsert=True,
    )
    n = int(doc["seq"])
    return f"S{n:05d}", n

def _curriculum_index():
    """
    Map (yearLevel, semester) -> [ {code,title,units}, ... ]
    from curriculum.courseCode == PROGRAM_CODE
    """
    cur = col("curriculum").find_one({"courseCode": PROGRAM_CODE})
    mapping = {}
    if not cur:
        return mapping
    for s in cur.get("subjects", []):
        yl = int(s.get("yearLevel", 0) or 0)
        sem = int(s.get("semester", 0) or 0)
        code = s.get("subjectCode")
        title = s.get("subjectName")
        units = int(s.get("units", 3) or 3)
        if not yl or not sem or not code:
            continue
        mapping.setdefault((yl, sem), []).append({"code": code, "title": title, "units": units})
    return mapping

def _make_terms_path(start_sy: str, start_year_level: int, n_terms: int, include_summer=True):
    """
    Produce a list of (school_year, semester, year_level_for_that_term) of length n_terms,
    starting from given school_year and academic yearLevel.
    We advance year_level on S2->next-year S1; Summer keeps same year_level.
    """
    out = []
    sy_list = list(SCHOOL_YEARS)
    if start_sy not in sy_list:
        sy_list = SCHOOL_YEARS
        start_sy = SCHOOL_YEARS[0]

    sy_idx = sy_list.index(start_sy)
    yl = max(1, min(4, int(start_year_level)))

    # term sequence starting at S1 typically; but allow starting S1 or S2 randomly
    sem_seq = [1, 2] + ([3] if include_summer else [])
    sem_pos = 0  # always start at S1 for consistency

    while len(out) < n_terms and sy_idx < len(sy_list):
        sem = sem_seq[sem_pos]
        out.append((sy_list[sy_idx], sem, yl))
        sem_pos += 1
        if sem_pos >= len(sem_seq):
            sem_pos = 0
            # after S2 we increase year level; summer does not auto-advance level
            yl = min(4, yl + 1)
            sy_idx += 1

    return out

def _upsert_student(email: str, name: str) -> dict:
    """
    Ensure a student doc exists for email; create if missing.
    Also ensure non-null student_id (to avoid unique index on null).
    """
    email = _nemail(email)
    s = col("students").find_one({"email": email})
    if s:
        # normalize shape (safely set program and student_id if missing)
        updates = {}
        if not s.get("student_no"):
            sn, _ = _next_student_no()
            updates["student_no"] = sn
            updates["student_id"] = sn
        if not s.get("program") or not s["program"].get("program_code"):
            updates["program"] = {"program_code": PROGRAM_CODE, "program_name": PROGRAM_NAME}
        if updates:
            updates["updated_at"] = datetime.utcnow()
            col("students").update_one({"_id": s["_id"]}, {"$set": updates})
            s.update(updates)
        return s

    # create new
    sn, _n = _next_student_no()
    doc = {
        "student_no": sn,
        "student_id": sn,   # keep non-null to satisfy any legacy unique index
        "email": email,
        "name": name or email,
        "base_year_level": random.randint(1, 3),
        "program": {"program_code": PROGRAM_CODE, "program_name": PROGRAM_NAME},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    col("students").insert_one(doc)
    return doc

def _enrollment_key(doc: dict) -> dict:
    return {
        "student.student_no": doc["student"]["student_no"],
        "term.school_year": doc["term"]["school_year"],
        "term.semester": doc["term"]["semester"],
        "subject.code": doc["subject"]["code"],
    }

def _upsert_enrollment(e: dict, commit: bool):
    if not commit:
        return 1  # pretend we wrote
    res = col("enrollments").update_one(_enrollment_key(e), {"$set": e, "$setOnInsert": {"created_at": datetime.utcnow()}}, upsert=True)
    return 1 if (res.upserted_id or res.modified_count >= 0) else 0

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Promote orphan student users into students + seed enrollments")
    ap.add_argument("--limit", type=int, default=200, help="Max orphan users to process (default 200)")
    ap.add_argument("--commit", action="store_true", help="Write changes. Omit for dry-run.")
    ap.add_argument("--min-terms", type=int, default=2, help="Min terms to create per student")
    ap.add_argument("--max-terms", type=int, default=6, help="Max terms to create per student")
    ap.add_argument("--include-summer", action="store_true", help="Include summer (semester=3) terms in path")
    args = ap.parse_args()

    # Build curriculum index
    cmap = _curriculum_index()
    if not cmap:
        print("WARNING: No curriculum found for BSED-ENGLISH. I will still create students; "
              "enrollments will be skipped.")
    # Collect emails that already have enrollments
    present = set(e for e in col("enrollments").distinct("student.email") if e)
    # Find student users that are not in enrollments (orphans)
    q = {"role": "student", "active": True}
    orphan = []
    for u in col("users").find(q, {"email": 1, "name": 1}).limit(100000):
        em = _nemail(u.get("email"))
        if em and em not in present:
            orphan.append({"email": em, "name": u.get("name") or em})
        if len(orphan) >= args.limit:
            break

    print(f"Found orphan student users to process: {len(orphan)} (limit={args.limit})")

    wrote_students = 0
    wrote_enrollments = 0
    for u in orphan:
        stu = _upsert_student(u["email"], u["name"])
        wrote_students += 1

        # create 2–6 consecutive terms starting from a random SY and their base_year_level
        start_sy = random.choice(SCHOOL_YEARS[:-1])  # avoid last SY for longer paths
        n_terms = random.randint(max(1, args.min_terms), max(args.min_terms, args.max_terms))
        terms = _make_terms_path(start_sy, stu.get("base_year_level", 1), n_terms, include_summer=args.include_summer)

        for (sy, sem, yl) in terms:
            # curriculum subjects for that year level and sem (only 1/2 have entries; summer will be empty)
            pool = list(cmap.get((yl, sem), []))
            if not pool:
                # For summer: optionally create 0–2 generic electives
                if sem == 3 and args.include_summer:
                    for i in range(random.randint(0, 2)):
                        grade = _grade_random()
                        e = {
                            "enrollment_no": None,
                            "term": {"school_year": sy, "semester": 3},
                            "student": {
                                "student_no": stu["student_no"],
                                "name": stu.get("name"),
                                "email": stu.get("email"),
                                "base_year_level": stu.get("base_year_level"),
                            },
                            "program": {"program_code": PROGRAM_CODE, "program_name": PROGRAM_NAME},
                            "college": COLLEGE,
                            "university": UNIVERSITY,
                            "teacher": {"email": "", "name": ""},
                            "subject": {"code": f"ENGSUM-{i+1}", "title": "Summer Elective", "units": 2, "year_level": yl, "semester": 3},
                            "grade": grade,
                            "remark": _remark_for(grade),
                            "updated_at": datetime.utcnow(),
                        }
                        wrote_enrollments += _upsert_enrollment(e, args.commit)
                continue

            # take between 4–8 subjects from curriculum pool for that term (or all if fewer)
            take = min(len(pool), random.randint(4, 8))
            subjects = random.sample(pool, take) if take < len(pool) else pool
            for s in subjects:
                grade = _grade_random()
                e = {
                    "enrollment_no": None,
                    "term": {"school_year": sy, "semester": sem},
                    "student": {
                        "student_no": stu["student_no"],
                        "name": stu.get("name"),
                        "email": stu.get("email"),
                        "base_year_level": stu.get("base_year_level"),
                    },
                    "program": {"program_code": PROGRAM_CODE, "program_name": PROGRAM_NAME},
                    "college": COLLEGE,
                    "university": UNIVERSITY,
                    "teacher": {"email": "", "name": ""},  # will be backfilled
                    "subject": {
                        "code": s["code"],
                        "title": s["title"],
                        "units": s.get("units", 3),
                        "year_level": yl,
                        "semester": sem,
                    },
                    "grade": grade,
                    "remark": _remark_for(grade),
                    "updated_at": datetime.utcnow(),
                }
                wrote_enrollments += _upsert_enrollment(e, args.commit)

    print("\n=== SUMMARY ===")
    print(f"students processed: {wrote_students}")
    print(f"enrollments {'to write' if not args.commit else 'written'}: {wrote_enrollments}")
    if not args.commit:
        print("\n(DRY RUN) No database changes made. Re-run with --commit to apply.")
    else:
        print("\nDone.")
        print("Next recommended steps:")
        print("  1) python scripts/backfill_teacher_from_offerings.py --commit")
        print("  2) python scripts/link_user_ids_from_email.py --commit")
