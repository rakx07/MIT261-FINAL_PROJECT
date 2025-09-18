# scripts/check_deans.py

# --- make project root importable (so we can import db.py) ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db import col

print("=== Top 10 overall (all terms combined) ===")
pipe = [
    {"$match": {"grade": {"$ne": None}, "program.program_code": "BSED-ENGLISH"}},
    {"$group": {
        "_id": "$student.student_no",
        "name": {"$first": "$student.name"},
        "gpa": {"$avg": "$grade"},
        "subjects": {"$sum": 1},
    }},
    {"$sort": {"gpa": -1}},
    {"$limit": 10},
]
for r in col("enrollments").aggregate(pipe):
    print(f"{r['name']:<25} GPA={r['gpa']:.2f} subjects={r['subjects']}")

print("\n=== Top 10 per term (Dean's list is usually per term) ===")
pipe_term = [
    {"$match": {"grade": {"$ne": None}, "program.program_code": "BSED-ENGLISH"}},
    {"$group": {
        "_id": {"student": "$student.student_no",
                "sy": "$term.school_year", "sem": "$term.semester"},
        "name": {"$first": "$student.name"},
        "gpa": {"$avg": "$grade"},
        "subjects": {"$sum": 1},
    }},
    {"$sort": {"gpa": -1}},
    {"$limit": 10},
]
for r in col("enrollments").aggregate(pipe_term):
    k = r["_id"]
    print(f"{r['name']:<25} {k['sy']} S{k['sem']}  GPA={r['gpa']:.2f}")

for thr in (90, 88, 85):
    cur = col("enrollments").aggregate([
        {"$match": {"grade": {"$ne": None}, "program.program_code": "BSED-ENGLISH"}} ,
        {"$group": {
            "_id": {"student": "$student.student_no",
                    "sy": "$term.school_year", "sem": "$term.semester"},
            "gpa": {"$avg": "$grade"}
        }},
        {"$match": {"gpa": {"$gte": thr}}},
        {"$count": "n"}
    ])
    n = next(cur, {}).get("n", 0)
    print(f"\nDean's List candidates with GPA ≥ {thr}: {n}")
