# scripts/sync_users_from_enrollments.py
"""
Create missing Users for anyone who appears in enrollments (students/teachers).

Usage (from project root):
    python scripts/sync_users_from_enrollments.py --dry-run
    python scripts/sync_users_from_enrollments.py --commit
    python scripts/sync_users_from_enrollments.py --who students --commit
    python scripts/sync_users_from_enrollments.py --who teachers --commit

You can also run it from inside the scripts/ directory; this file will
bootstrap the project root onto sys.path automatically.
"""
from __future__ import annotations
import os, sys, csv, argparse, secrets, string
from typing import Dict, List, Tuple

# --- bootstrap project root so "from db import col" works even when run from /scripts
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import col  # noqa: E402
from utils.auth import get_user, create_user  # noqa: E402

def _norm_email(e: str | None) -> str:
    return (e or "").strip().lower()

def _gen_temp_pw(n: int = 12) -> str:
    n = max(8, n)
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    # ensure complexity
    seed = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*"),
    ]
    seed += [secrets.choice(alphabet) for _ in range(n - len(seed))]
    for i in range(len(seed) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        seed[i], seed[j] = seed[j], seed[i]
    return "".join(seed)

def _collect_distinct_people(prefix: str) -> List[Tuple[str, str]]:
    """
    Return list of (email, name) distinct from enrollments.<prefix>.
    prefix is 'student' or 'teacher'.
    """
    pipeline = [
        {"$match": {f"{prefix}.email": {"$exists": True, "$ne": ""}}},
        {"$group": {
            "_id": f"${prefix}.email",
            "name": {"$first": f"${prefix}.name"},
        }},
        {"$sort": {"_id": 1}}
    ]
    out: List[Tuple[str, str]] = []
    for r in col("enrollments").aggregate(pipeline, allowDiskUse=True):
        em = _norm_email(r.get("_id"))
        nm = (r.get("name") or "").strip()
        if em:
            out.append((em, nm or em))
    return out

def _already_have_user(emails: List[str]) -> Dict[str, bool]:
    have: Dict[str, bool] = {}
    if not emails:
        return have
    # fetch in batches
    B = 500
    for i in range(0, len(emails), B):
        batch = emails[i:i+B]
        for u in col("users").find({"email": {"$in": batch}}, {"email": 1}):
            have[_norm_email(u.get("email"))] = True
    return have

def sync_people(prefix: str, role: str, commit: bool) -> List[Dict[str, str]]:
    """
    prefix: 'student' or 'teacher'
    role:   'student' or 'teacher' (user role)
    Returns list of created accounts (email, name, role, temp_password).
    """
    people = _collect_distinct_people(prefix)
    emails = [_norm_email(e) for e, _ in people]
    have = _already_have_user(emails)

    created: List[Dict[str, str]] = []
    for em, name in people:
        if have.get(em):
            continue
        temp = _gen_temp_pw()
        if commit:
            # utils.auth.create_user() is idempotent; it returns existing user if email already present.
            create_user(email=em, name=name, role=role, password=temp, must_change_password=True, active=True)
        created.append({"email": em, "name": name, "role": role, "temp_password": temp})
    return created

def write_csv(rows: List[Dict[str, str]], path: str) -> None:
    if not rows:
        return
    fields = ["email", "name", "role", "temp_password"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Create missing Users for anyone present in enrollments.")
    ap.add_argument("--commit", action="store_true", help="Actually write to DB. Omit for dry run.")
    ap.add_argument("--who", choices=["students", "teachers", "both"], default="both",
                    help="Which group to sync. Default: both.")
    ap.add_argument("--out", default="created_users.csv", help="CSV output for created temp passwords.")
    args = ap.parse_args()

    do_students = args.who in ("students", "both")
    do_teachers = args.who in ("teachers", "both")

    all_created: List[Dict[str, str]] = []

    if do_students:
        c = sync_people("student", "student", commit=args.commit)
        print(f"students: to_create={len(c)}{' (committed)' if args.commit else ' (dry-run)'}")
        all_created.extend(c)
    if do_teachers:
        c = sync_people("teacher", "teacher", commit=args.commit)
        print(f"teachers: to_create={len(c)}{' (committed)' if args.commit else ' (dry-run)'}")
        all_created.extend(c)

    if all_created:
        write_csv(all_created, args.out)
        print(f"Wrote {len(all_created)} new accounts to {args.out}")
    else:
        print("Nothing to create.")

if __name__ == "__main__":
    main()
