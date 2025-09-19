# scripts/link_user_ids_from_email.py
"""
Link users._id into enrollments.student.user_id and enrollments.teacher.user_id
based on matching emails.

Usage (from project root or scripts/ — this file bootstraps sys.path):
  Dry-run:
    python scripts/link_user_ids_from_email.py
    python scripts/link_user_ids_from_email.py --dry-run
  Commit:
    python scripts/link_user_ids_from_email.py --commit
  Also create helpful indexes (recommended once):
    python scripts/link_user_ids_from_email.py --commit --ensure-indexes
"""
from __future__ import annotations
import os, sys, argparse
from typing import Dict, Iterable, Tuple

# --- bootstrap project root so "from db import col" works even if run from /scripts
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import col  # noqa: E402


def _norm_email(e) -> str:
    return (e or "").strip().lower()


def _build_user_map() -> Dict[str, object]:
    """
    email(str)->_id(ObjectId)
    Only includes users with a non-empty email.
    """
    m: Dict[str, object] = {}
    for u in col("users").find({"email": {"$exists": True, "$ne": ""}}, {"email": 1}):
        em = _norm_email(u.get("email"))
        if em:
            m[em] = u["_id"]
    return m


def _distinct_nonempty(coll: str, field: str) -> list[str]:
    vals = col(coll).distinct(field)
    out = []
    for v in vals:
        em = _norm_email(v)
        if em:
            out.append(em)
    return out


def _ensure_indexes():
    col("users").create_index("email", unique=True)
    col("enrollments").create_index("student.email")
    col("enrollments").create_index("teacher.email")
    col("enrollments").create_index("student.user_id")
    col("enrollments").create_index("teacher.user_id")


def _link_side(side: str, emails: Iterable[str], user_map: Dict[str, object], commit: bool) -> Tuple[int, int]:
    """
    side: 'student' or 'teacher'
    Returns (matched_docs_to_update, actually_updated_docs_if_commit)
    """
    matched = 0
    updated = 0
    for em in emails:
        uid = user_map.get(em)
        if not uid:
            continue  # there is no user for this email

        q_missing = {
            f"{side}.email": em,
            "$or": [
                {f"{side}.user_id": {"$exists": False}},
                {f"{side}.user_id": None},
            ],
        }
        n_miss = col("enrollments").count_documents(q_missing)
        matched += n_miss
        if commit and n_miss:
            res = col("enrollments").update_many(q_missing, {"$set": {f"{side}.user_id": uid}})
            updated += res.modified_count or 0
    return matched, updated


def main():
    ap = argparse.ArgumentParser(description="Backfill user_id into enrollments from users by matching emails.")
    ap.add_argument("--commit", action="store_true", help="Actually write updates. Default: dry-run.")
    ap.add_argument("--dry-run", action="store_true", help="Synonym for not passing --commit.")
    ap.add_argument("--ensure-indexes", action="store_true", help="Create helpful indexes (safe to run multiple times).")
    args = ap.parse_args()
    commit = bool(args.commit and not args.dry_run)

    if args.ensure_indexes:
        _ensure_indexes()
        print("Indexes ensured.")

    user_map = _build_user_map()
    print(f"users mapped (email→_id): {len(user_map)}")

    stu_emails = _distinct_nonempty("enrollments", "student.email")
    tch_emails = _distinct_nonempty("enrollments", "teacher.email")
    print(f"distinct student emails in enrollments: {len(stu_emails)}")
    print(f"distinct teacher emails in enrollments: {len(tch_emails)}")

    s_match, s_upd = _link_side("student", stu_emails, user_map, commit)
    t_match, t_upd = _link_side("teacher", tch_emails, user_map, commit)

    print("\nSummary:")
    print(f"  student.user_id to set (missing): {s_match}")
    print(f"  teacher.user_id to set (missing): {t_match}")
    if commit:
        print(f"  student.user_id updated: {s_upd}")
        print(f"  teacher.user_id updated: {t_upd}")
    else:
        print("  (dry-run; pass --commit to apply changes)")


if __name__ == "__main__":
    main()
