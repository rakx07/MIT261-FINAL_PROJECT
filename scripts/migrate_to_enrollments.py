# scripts/migrate_to_enrollments.py
# --------------------------------
# Migrate old `grades` and optionally `grades_ingested` into canonical `enrollments`.

# Make project root importable
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import argparse
import re
from db import col

# --- CONFIG ---
SEMESTER_MAP = {
    1: {"school_year": "2023-2024", "semester": 1},
    2: {"school_year": "2023-2024", "semester": 2},
}
PROGRAM_CODE = "BSED-ENGLISH"
PROGRAM_NAME = "BSED English"
DEFAULT_TERM_FOR_INGESTED = 1  # put grades_ingested into S1 unless you know the real term

def norm_prog():
    return {"program_code": PROGRAM_CODE, "program_name": PROGRAM_NAME}

def remark_for(grade):
    if grade is None:
        return "INC"
    try:
        return "PASSED" if float(grade) >= 75 else "FAILED"
    except Exception:
        return "INC"

def subject_units(code):
    s = col("subjects").find_one({"_id": code}, {"Units": 1, "Description": 1})
    if not s:
        return 3, ""  # fallback
    return int(s.get("Units", 3)), s.get("Description", "")

def student_lookup_by_id(numeric_id):
    s = col("students").find_one({"_id": numeric_id})
    if not s:
        return None
    return {
        "student_no": s.get("student_id") or f"STU-{numeric_id}",
        "name": s.get("Name", ""),
        "year_level": s.get("YearLevel"),
        "program_code": PROGRAM_CODE,
    }

def student_lookup_by_name(name):
    if not name:
        return None
    s = col("students").find_one({"Name": name})
    if s:
        return {
            "student_no": s.get("student_id") or f"STU-{s.get('_id')}",
            "name": s.get("Name", ""),
            "year_level": s.get("YearLevel"),
            "program_code": PROGRAM_CODE,
        }
    s = col("students").find_one({"Name": re.compile(f"^{re.escape(name)}$", re.I)})
    if s:
        return {
            "student_no": s.get("student_id") or f"STU-{s.get('_id')}",
            "name": s.get("Name", ""),
            "year_level": s.get("YearLevel"),
            "program_code": PROGRAM_CODE,
        }
    return None

def upsert_enrollment(doc):
    key = {
        "student.student_no": doc["student"]["student_no"],
        "subject.code": doc["subject"]["code"],
        "term.school_year": doc["term"]["school_year"],
        "term.semester": doc["term"]["semester"],
    }
    col("enrollments").update_one(
        key,
        {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
    )

def migrate_from_grades(limit=None, progress_every=500):
    rows_written = 0
    docs_seen = 0
    errors = 0
    cursor = col("grades").find({}).batch_size(200)
    print("[grades] migrating...", flush=True)

    for g in cursor:
        docs_seen += 1
        try:
            student = student_lookup_by_id(g.get("StudentID"))
            term = SEMESTER_MAP.get(g.get("SemesterID"))
            if not (student and term):
                continue

            codes    = g.get("SubjectCodes", []) or []
            grades   = g.get("Grades", []) or []
            teachers = g.get("Teachers", []) or []

            m = max(len(codes), len(grades), len(teachers))
            for i in range(m):
                code   = codes[i] if i < len(codes) else None
                grade  = grades[i] if i < len(grades) else None
                tname  = teachers[i] if i < len(teachers) else ""
                if not code:
                    continue
                units, title = subject_units(code)
                doc = {
                    "term": term,
                    "student": {"student_no": student["student_no"], "name": student["name"]},
                    "program": norm_prog(),
                    "teacher": {"email": "", "name": tname or ""},
                    "subject": {"code": code, "title": title, "units": units},
                    "grade": grade,
                    "remark": remark_for(grade),
                    "updated_at": datetime.utcnow(),
                }
                upsert_enrollment(doc)
                rows_written += 1
        except Exception as e:
            errors += 1
            if errors <= 10:  # don’t spam; show first 10
                print(f"[grades] error on doc {_id_safe(g)}: {e}", flush=True)

        if progress_every and docs_seen % progress_every == 0:
            print(f"[grades] processed {docs_seen} grade-docs; wrote ~{rows_written} rows...", flush=True)

        if limit and docs_seen >= limit:
            break

    print(f"[grades] done. processed_docs={docs_seen}, rows_written={rows_written}, errors={errors}", flush=True)
    return rows_written

def migrate_from_grades_ingested_by_name(limit=None, progress_every=1000):
    if col("grades_ingested").estimated_document_count() == 0:
        print("[grades_ingested] none found; skipping.")
        return 0

    fields = {"StudentID": 1, "SubjectCode": 1, "Grade": 1, "Remark": 1}
    rows_written = 0
    docs_seen = 0
    errors = 0
    cursor = col("grades_ingested").find({}, fields).batch_size(500)
    print("[grades_ingested] migrating (name-matched)...", flush=True)

    for r in cursor:
        docs_seen += 1
        try:
            student = student_lookup_by_name(r.get("StudentID"))
            if not student:
                continue
            code  = r.get("SubjectCode")
            grade = r.get("Grade")
            if not code:
                continue

            units, title = subject_units(code)
            term = SEMESTER_MAP.get(DEFAULT_TERM_FOR_INGESTED)
            doc = {
                "term": term,
                "student": {"student_no": student["student_no"], "name": student["name"]},
                "program": norm_prog(),
                "teacher": {"email": "", "name": ""},
                "subject": {"code": code, "title": title, "units": units},
                "grade": grade,
                "remark": r.get("Remark") or remark_for(grade),
                "updated_at": datetime.utcnow(),
            }
            upsert_enrollment(doc)
            rows_written += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"[ingested] error on doc {_id_safe(r)}: {e}", flush=True)

        if progress_every and docs_seen % progress_every == 0:
            print(f"[ingested] processed {docs_seen}; wrote ~{rows_written} rows...", flush=True)

        if limit and docs_seen >= limit:
            break

    print(f"[grades_ingested] done. processed_docs={docs_seen}, rows_written={rows_written}, errors={errors}", flush=True)
    return rows_written

def add_indexes():
    print("[enrollments] ensuring indexes...", flush=True)
    col("enrollments").create_index([("student.student_no", 1)])
    col("enrollments").create_index([("term.school_year", 1), ("term.semester", 1)])
    col("enrollments").create_index([("subject.code", 1)])
    col("enrollments").create_index([("program.program_code", 1)])
    col("enrollments").create_index([("teacher.email", 1)])
    print("[enrollments] indexes done.", flush=True)

def _id_safe(d):
    try:
        return d.get("_id")
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Migrate grades → enrollments")
    parser.add_argument("--limit", type=int, default=None, help="Only process N docs from each source (for testing)")
    parser.add_argument("--ingested", dest="ingested", action="store_true", help="Also migrate grades_ingested by student name")
    parser.add_argument("--no-ingested", dest="ingested", action="store_false", help="Skip grades_ingested (default)")
    parser.set_defaults(ingested=False)
    args = parser.parse_args()

    print("Starting migration...", flush=True)
    n1 = migrate_from_grades(limit=args.limit)
    n2 = 0
    if args.ingested:
        n2 = migrate_from_grades_ingested_by_name(limit=args.limit)
    add_indexes()
    total = n1 + n2
    print(f"Done. wrote_rows={total}. Verify data in `enrollments` and then archive `grades`/`grades_ingested`.", flush=True)

if __name__ == "__main__":
    main()
