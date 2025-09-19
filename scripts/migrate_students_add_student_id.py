# scripts/migrate_students_add_student_id.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db import col
from pymongo.errors import OperationFailure

def main():
    c = col("students")

    # Backfill student_id from student_no (or _id if needed)
    missing = {"$or": [{"student_id": {"$exists": False}}, {"student_id": None}]}
    bulk = []
    for d in c.find(missing, {"_id": 1, "student_no": 1}):
        sid = d.get("student_no") or str(d["_id"])
        bulk.append(
            {"updateOne": {"filter": {"_id": d["_id"]}, "update": {"$set": {"student_id": sid}}}}
        )
        if len(bulk) >= 1000:
            c.bulk_write(bulk)
            bulk = []
    if bulk:
        c.bulk_write(bulk)

    # Recreate unique index on student_id as sparse (allows many docs missing the field in the future)
    try:
        c.drop_index("student_id_1")
    except OperationFailure:
        pass
    c.create_index("student_id", unique=True, sparse=True)

    print("Migration completed: student_id backfilled, index recreated (unique + sparse).")

if __name__ == "__main__":
    main()
