# scripts/backfill_teacher_from_offerings.py
# Map offerings like "ENGX-Y2S1-06" -> catalog codes like "ENG2106"
# Then backfill enrollments.teacher.{email,name} when missing.
#
# Run:
#   python scripts/backfill_teacher_from_offerings.py              # dry run
#   python scripts/backfill_teacher_from_offerings.py --commit     # write changes
#   python scripts/backfill_teacher_from_offerings.py --commit --limit 500

from __future__ import annotations
import os, sys, re, argparse
from typing import Dict, Tuple, Optional

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db import col


# ---------- helpers ----------
CID_RE = re.compile(r"^([A-Z]{3,})X-Y(\d)S(\d)-(\d{2})$")  # e.g. ENGX-Y2S1-06

def to_catalog_code(curr_code: str | None) -> Optional[str]:
    """
    "ENGX-Y2S1-06" -> "ENG2106"
    "ITX-Y1S2-03"  -> "ITX1203"
    Returns None if not recognized.
    """
    if not curr_code:
        return None
    m = CID_RE.match(curr_code.strip().upper())
    if not m:
        return None
    prefix, year, sem, nn = m.groups()
    return f"{prefix}{year}{sem}{nn}"

def norm_tid(tid: str | None) -> str:
    if not tid:
        return ""
    x = str(tid).strip().upper()
    if x.startswith("T"):
        x = x[1:]
    return x.zfill(4)

def build_program_map() -> Dict[str, str]:
    """program_id -> program_code"""
    out: Dict[str, str] = {}
    for p in col("programs").find({}, {"program_id": 1, "code": 1}):
        if p.get("program_id") and p.get("code"):
            out[p["program_id"]] = p["code"]
    return out

def build_teacher_map() -> Dict[str, Dict[str, str]]:
    """
    teacher_id -> {email, name}. If email missing, synthesize "0007@su.edu".
    """
    out: Dict[str, Dict[str, str]] = {}
    for t in col("teachers").find({}, {"teacher_id": 1, "email": 1, "name": 1}):
        tid = norm_tid(t.get("teacher_id"))
        email = (t.get("email") or "").strip().lower()
        if not email and tid:
            email = f"{tid}@su.edu"
        out[tid] = {"email": email, "name": t.get("name") or ""}
    return out


# ---------- build offerings index ----------
def build_offerings_index():
    """
    Build index keyed by (school_year, semester, program_code, CATALOG_subject_code)
    to {email, name} for the teacher on that offering.
    """
    pid2code = build_program_map()
    tmap     = build_teacher_map()

    idx: Dict[Tuple[str, int, str, str], Dict[str, str]] = {}

    cur = col("offerings").find(
        {},
        {
            "school_year": 1,
            "semester": 1,
            "program_id": 1,
            "subject_code": 1,
            "teacher_id": 1,
        },
    )

    for o in cur:
        sy  = str(o.get("school_year") or "")
        sem = int(o.get("semester") or 0)
        pid = o.get("program_id")
        subj_curr = o.get("subject_code")
        cat = to_catalog_code(subj_curr)  # <-- key step
        if not (sy and sem and pid and cat):
            continue

        prog_code = pid2code.get(pid)
        if not prog_code:
            continue

        tid = norm_tid(o.get("teacher_id"))
        tdoc = tmap.get(tid)
        if not tdoc:
            continue

        idx[(sy, sem, prog_code, cat)] = {
            "email": (tdoc.get("email") or "").strip().lower(),
            "name":  tdoc.get("name") or "",
        }

    return idx


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="apply updates")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    idx = build_offerings_index()
    print("Building offerings index…")
    print(f"index size: {len(idx)}")

    # enrollments that need teacher
    q = {
        "$or": [
            {"teacher": {"$exists": False}},
            {"teacher.email": {"$exists": False}},
            {"teacher.email": ""},
        ]
    }
    proj = {
        "term.school_year": 1,
        "term.semester": 1,
        "program.program_code": 1,
        "subject.code": 1,
        "teacher.email": 1,
        "teacher.name": 1,
    }

    scanned = matched = updated = skipped = 0
    updates = []

    for e in col("enrollments").find(q, proj):
        scanned += 1
        term = e.get("term") or {}
        prog = e.get("program") or {}
        sy   = str(term.get("school_year") or "")
        sem  = int(term.get("semester") or 0)
        pcd  = str(prog.get("program_code") or "")
        sc   = str((e.get("subject") or {}).get("code") or "")
        key  = (sy, sem, pcd, sc)

        info = idx.get(key)
        if not info:
            skipped += 1
            continue

        em = (info.get("email") or "").strip().lower()
        nm = info.get("name") or ""
        if not em:
            skipped += 1
            continue

        matched += 1
        updates.append({"_id": e["_id"], "teacher": {"email": em, "name": nm}})

    print(f"scanned={scanned}, matched={matched}, would_update={len(updates)}, skipped={skipped}")

    if not args.commit or not updates:
        return

    lim = args.limit or len(updates)
    u = 0
    for up in updates[:lim]:
        res = col("enrollments").update_one({"_id": up["_id"]}, {"$set": {"teacher": up["teacher"]}})
        u += res.modified_count
    print(f"updated={u}")

if __name__ == "__main__":
    main()
