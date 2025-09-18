# make project root importable
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db import col
from datetime import datetime

DOMAIN = "students.su.edu"  # change if you like

def synth_email(student_no: str) -> str:
    # STU-15 → stu-15@students.su.edu
    return f"{(student_no or '').lower()}@{DOMAIN}"

def run():
    students = list(col("students").find({}, {"student_id": 1, "Name": 1, "email": 1, "Email": 1}))
    updated_students = 0
    for s in students:
        s_no = s.get("student_id") or f"STU-{s.get('_id')}"
        email = s.get("email") or s.get("Email") or synth_email(s_no)
        if email != s.get("email") or email != s.get("Email"):
            col("students").update_one({"_id": s["_id"]}, {"$set": {"email": email, "Email": email}})
            updated_students += 1

        # push into enrollments
        col("enrollments").update_many(
            {"student.student_no": s_no},
            {"$set": {"student.email": email, "updated_at": datetime.utcnow()}}
        )
    print(f"Students updated: {updated_students}")

if __name__ == "__main__":
    run()
    print("Done backfilling student emails.")
