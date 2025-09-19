# scripts/deactivate_orphans.py
# --- path bootstrap ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ----------------------

import argparse
from datetime import datetime, timezone
from db import col

def _n(e): return (e or "").strip().lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["student","teacher"], required=True)
    ap.add_argument("--delete", action="store_true", help="Physically delete (use with care).")
    ap.add_argument("--commit", action="store_true", help="Apply changes. Default is dry-run.")
    args = ap.parse_args()

    users = col("users")
    enroll = col("enrollments")

    if args.role=="student":
        E = set(_n(e) for e in enroll.distinct("student.email") if e)
        role_q = {"role":"student"}
    else:
        E = set(_n(e) for e in enroll.distinct("teacher.email") if e)
        role_q = {"role":{"$in":["teacher","faculty"]}}

    orphans = list(users.find(
        {"$and":[role_q, {"email":{"$exists":True}}]},
        {"email":1,"name":1,"active":1,"role":1}))
    orphans = [u for u in orphans if _n(u.get("email")) not in E]

    print(f"Found orphan {args.role} users: {len(orphans)}")
    if not args.commit:
        print("(dry-run) No changes written. Add --commit to apply.")
        return

    if args.delete:
        ids = [u["_id"] for u in orphans]
        if ids:
            res = users.delete_many({"_id":{"$in":ids}})
            print("deleted:", res.deleted_count)
        return

    now = datetime.now(timezone.utc)
    updates = 0
    for u in orphans:
        res = users.update_one(
            {"_id": u["_id"]},
            {"$set":{
                "active": False,
                "deactivated_reason":"orphan_no_enrollments",
                "updated_at": now
            }})
        updates += res.modified_count
    print("deactivated:", updates)

if __name__ == "__main__":
    main()
