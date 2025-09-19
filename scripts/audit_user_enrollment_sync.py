# scripts/audit_user_enrollment_sync.py
# --- path bootstrap so `from db import col` works when run as a script ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# -------------------------------------------------------------------------

from datetime import datetime, timezone
from collections import defaultdict
from db import col

def _nemail(e):
    return (e or "").strip().lower()

def main():
    users = col("users")
    enroll = col("enrollments")

    stu_emails = set(_nemail(e) for e in enroll.distinct("student.email") if e)
    tch_emails = set(_nemail(e) for e in enroll.distinct("teacher.email") if e)

    U_all = list(users.find({}, {"email":1, "role":1, "active":1, "name":1}))
    u_students = [u for u in U_all if (u.get("role","").lower()=="student")]
    u_teachers = [u for u in U_all if (u.get("role","").lower() in ("teacher","faculty"))]

    u_stu_emails = set(_nemail(u.get("email")) for u in u_students if u.get("email"))
    u_tch_emails = set(_nemail(u.get("email")) for u in u_teachers if u.get("email"))

    orphan_students = [u for u in u_students if _nemail(u.get("email")) not in stu_emails]
    orphan_teachers = [u for u in u_teachers if _nemail(u.get("email")) not in tch_emails]

    missing_student_accounts = sorted(list(stu_emails - u_stu_emails))
    missing_teacher_accounts = sorted(list(tch_emails - u_tch_emails))

    buckets = defaultdict(list)
    for u in U_all:
        buckets[_nemail(u.get("email"))].append(u)
    duplicates = {e:rows for e,rows in buckets.items() if e and len(rows)>1}

    print("==== AUDIT SUMMARY ====")
    print(f"users total: {len(U_all)}")
    print(f"enrollments distinct student emails: {len(stu_emails)}")
    print(f"enrollments distinct teacher emails: {len(tch_emails)}")
    print()
    print(f"orphan student users (role=student, no enrollments): {len(orphan_students)}")
    print(f"orphan teacher users (role=teacher/faculty, no enrollments): {len(orphan_teachers)}")
    print(f"missing student accounts (present in enrollments only): {len(missing_student_accounts)}")
    print(f"missing teacher accounts (present in enrollments only): {len(missing_teacher_accounts)}")
    print(f"duplicate user emails: {len(duplicates)}")
    print()

    import csv
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def dump_csv(path, rows, fields):
        if not rows: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k:r.get(k) for k in fields})
        print("wrote:", path)

    dump_csv(f"orphan_students_{now}.csv",
             orphan_students, ["_id","email","name","active","role"])
    dump_csv(f"orphan_teachers_{now}.csv",
             orphan_teachers, ["_id","email","name","active","role"])

    if missing_student_accounts:
        with open(f"missing_student_accounts_{now}.txt","w",encoding="utf-8") as f:
            f.write("\n".join(missing_student_accounts))
        print("wrote: missing_student_accounts_*.txt")

    if missing_teacher_accounts:
        with open(f"missing_teacher_accounts_{now}.txt","w",encoding="utf-8") as f:
            f.write("\n".join(missing_teacher_accounts))
        print("wrote: missing_teacher_accounts_*.txt")

    if duplicates:
        with open(f"duplicate_user_emails_{now}.txt","w",encoding="utf-8") as f:
            for e, rows in duplicates.items():
                f.write(f"{e} => {[str(r.get('_id')) for r in rows]}\n")
        print("wrote: duplicate_user_emails_*.txt")

if __name__ == "__main__":
    main()
