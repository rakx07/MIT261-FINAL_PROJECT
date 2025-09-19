# scripts/seed_enrollments_from_curriculum.py
# Robust seeder: offerings schema tolerant, curriculum -> enrollments
import sys, os, argparse, random
from datetime import datetime, timezone

# allow "from db import col"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db import col  # noqa

# ------------------------------
# Helpers to normalize documents
# ------------------------------

def _get(obj, *paths, default=None):
    """
    Safe getter: _get(doc, ("term","school_year"), ("school_year",)) -> first match
    """
    for path in paths:
        if not path:
            continue
        cur = obj
        ok = True
        for p in path if isinstance(path, (list, tuple)) else (path,):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return cur
    return default


def _norm_offering(off: dict):
    """
    Return (school_year:str, semester:int, subject_code:str) from an offering doc,
    tolerant of multiple schemas.
    """
    sy = _get(off, ("school_year",), ("term", "school_year"))
    sem = _get(off, ("semester",), ("term", "semester"))
    code = _get(off, ("subject_code",), ("subject", "code"), ("subjectCode",))
    if sem is not None:
        try:
            sem = int(sem)
        except Exception:
            sem = None
    return sy, sem, code


def _norm_teacher_from_off(off: dict) -> dict:
    """
    Produce a teacher block:
    { email, name, teacher_id }
    from various possible fields inside an offering.
    """
    t = off.get("teacher") or {}
    email = (_get(off, ("teacher_email",)) or _get(t, ("email",)) or "").strip().lower()
    name = _get(off, ("teacher_name",)) or _get(t, ("name",))
    tid = _get(off, ("teacher_id",), ("teacherId",), ("teacher", "id"))
    # normalize teacher_id like "T0011" -> "0011"
    if isinstance(tid, str) and tid:
        tid_norm = tid.lstrip("Tt")
    else:
        tid_norm = tid

    # If we still have no email/name but we do have a teacher_id, look up in teachers
    if (not email or not name) and tid_norm:
        u = col("teachers").find_one({"teacher_id": str(tid_norm)}, {"email": 1, "name": 1})
        if u:
            email = email or (u.get("email") or "").lower()
            name = name or u.get("name")

    return {
        "email": email,
        "name": name,
        "teacher_id": str(tid_norm) if tid_norm is not None else None,
    }


def _teacher_index_from_offerings() -> dict:
    """
    Build index:  (sy, sem, subject_code) -> teacher dict
    Handles both old/new schemas of offerings.
    """
    idx = {}
    fields = {
        "school_year": 1,
        "semester": 1,
        "subject_code": 1,
        "subject.code": 1,
        "subjectCode": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "teacher.email": 1,
        "teacher.name": 1,
        "teacher.id": 1,
        "teacher_id": 1,
        "teacher_email": 1,
        "teacher_name": 1,
    }
    for off in col("offerings").find({}, fields):
        sy, sem, code = _norm_offering(off)
        if not (sy and sem and code):
            continue
        idx[(sy, sem, code)] = _norm_teacher_from_off(off)
    return idx


def _curriculum_map(course_code: str):
    """
    Return dict:  (yearLevel, semester) -> [ {code, title, units} ... ]
    Handles both your curriculum variants.
    """
    cur = col("curriculum").find_one({"courseCode": course_code}) \
          or col("curriculum").find_one({"program_code": course_code})
    if not cur:
        raise RuntimeError(f"Curriculum not found for courseCode={course_code}")

    out = {}
    for s in cur.get("subjects", []):
        yl = _get(s, ("yearLevel",))
        sem = _get(s, ("semester",))
        code = _get(s, ("subjectCode",), ("code",))
        name = _get(s, ("subjectName",), ("title",))
        units = _get(s, ("units",))
        if yl is None or sem is None or not code:
            continue
        try:
            yl = int(yl)
            sem = int(sem)
        except Exception:
            continue
        out.setdefault((yl, sem), []).append({
            "code": code,
            "title": name or code,
            "units": units if isinstance(units, (int, float)) else None
        })
    return out, cur


def _advance_sy(start_sy: str, years: int) -> str:
    """
    '2020-2021', years=1 -> '2021-2022'
    """
    try:
        a, b = start_sy.split("-")
        a, b = int(a), int(b)
        a += years
        b += years
        return f"{a}-{b}"
    except Exception:
        return start_sy


def _grade_tuple():
    """
    Create a random (grade, remark) with some INC/DROPPED probability.
    Filipino grading style 60-100 integers, pass>=75 by your convention.
    """
    r = random.random()
    if r < 0.07:
        return (None, "DROPPED")
    if r < 0.12:
        return (None, "INCOMPLETE")
    # graded
    g = int(random.normalvariate(85, 7))
    g = max(60, min(99, g))
    return (g, "PASSED" if g >= 75 else "FAILED")


# ------------------------------
# Main seeding logic
# ------------------------------

def seed_enrollments(course_code: str,
                     program_name_fallback: str = "",
                     max_years=4,
                     include_summer=True,
                     limit_students=None,
                     dry_run=True):
    """
    Create enrollments from curriculum for students in given course/program.
    - Uses offerings teacher index when possible.
    - Upserts enrollments by deterministic enrollment_no.
    """
    # teacher index by (sy,sem,subj)
    t_idx = _teacher_index_from_offerings()

    # curriculum map
    cmap, cur_doc = _curriculum_map(course_code)
    program_name = cur_doc.get("courseName") or cur_doc.get("program_name") or program_name_fallback or course_code

    # pick students
    q = {"$or": [
        {"program.program_code": course_code},
        {"program_code": course_code}
    ]}
    fields = {"student_no": 1, "name": 1, "email": 1,
              "program.program_code": 1, "program.program_name": 1,
              "base_year_level": 1, "current_year_level": 1, "entry_sy": 1}
    cur = col("students").find(q, fields).limit(limit_students or 10**9)

    enrollments = col("enrollments")

    scanned = 0
    to_upsert = 0

    for stu in cur:
        scanned += 1
        student_no = stu.get("student_no") or str(stu.get("_id"))
        entry_sy = stu.get("entry_sy") or "2020-2021"
        cur_level = int(stu.get("current_year_level") or 1)
        years_to_seed = min(max_years, cur_level)  # seed up to current progress

        for y in range(1, years_to_seed + 1):
            sy = _advance_sy(entry_sy, y - 1)
            for sem in (1, 2, 3) if include_summer else (1, 2):
                subjects = cmap.get((y, sem)) or []
                for subj in subjects:
                    code = subj["code"]
                    title = subj["title"]
                    units = subj.get("units")

                    # find teacher from offerings
                    tch = t_idx.get((sy, sem, code), {"email": "", "name": "", "teacher_id": None})

                    grade, remark = _grade_tuple()

                    # deterministic enrollment number avoids duplicates
                    eno = f"E{student_no}-{sy.replace('-','')}-S{sem}-{code}"

                    doc = {
                        "enrollment_no": eno,
                        "term": {"school_year": sy, "semester": sem},
                        "student": {
                            "student_no": student_no,
                            "name": stu.get("name"),
                            "email": stu.get("email"),
                            "base_year_level": int(stu.get("base_year_level") or 1),
                        },
                        "program": {
                            "program_code": course_code,
                            "program_name": program_name,
                            "years": max_years,
                            "semesters_per_year": 3 if include_summer else 2,
                        },
                        "subject": {"code": code, "title": title, "units": units},
                        "teacher": tch,
                        "grade": grade,
                        "remark": remark,
                        "prerequisites_met": True,
                        "updated_at": datetime.now(timezone.utc),
                    }

                    to_upsert += 1
                    if not dry_run:
                        enrollments.update_one({"enrollment_no": eno}, {"$set": doc}, upsert=True)

    print(
        f"seed_enrollments_from_curriculum: students_scanned={scanned}, "
        f"rows_{'would_upsert' if dry_run else 'upserted'}={to_upsert}"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Seed enrollments from curriculum (schema-tolerant).")
    ap.add_argument("--course", default="BSED-ENGLISH", help="course/program code")
    ap.add_argument("--max-years", type=int, default=4)
    ap.add_argument("--include-summer", action="store_true", help="include S3 (summer) terms")
    ap.add_argument("--limit-students", type=int, default=None)
    ap.add_argument("--commit", action="store_true", help="write to DB (omit for dry run)")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_enrollments(
        course_code=args.course,
        max_years=args.max_years,
        include_summer=args.include_summer,
        limit_students=args.limit_students,
        dry_run=not args.commit,
    )


if __name__ == "__main__":
    main()
