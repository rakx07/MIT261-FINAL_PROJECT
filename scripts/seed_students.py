# scripts/seed_students.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from datetime import datetime, timezone
from db import col

PROGRAM = {"program_code": "BSED-ENGLISH", "program_name": "BSED English"}
ENTRY_SYS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"]

def rand_name():
    first = ["Alex","Morgan","Riley","Jamie","Evelyn","Sofia","Liam","Noah","Logan","Ava","Lucas","Mia","Ethan"]
    last  = ["Santos","Ramos","Domingo","Ortega","Bautista","Reyes","Navarro","Garcia","Cruz","Querubin","Castillo"]
    return f"{random.choice(first)} {random.choice(last)}"

def main(n_students=400, three_year_program=False):
    students = col("students")
    for i in range(n_students):
        entry_sy = random.choice(ENTRY_SYS)
        max_year = 3 if three_year_program else 4
        cur_year = random.randint(1, max_year)

        sid = f"S{entry_sy.split('-')[0]}-{i:04d}"               # student_no
        email = f"{sid.lower()}@students.su.edu"

        doc = {
            "_id": sid,                          # OK if you prefer ObjectId; this keeps it readable
            "student_no": sid,
            "student_id": sid,                    # <<< IMPORTANT: make it unique (matches unique index)
            "name": rand_name(),
            "email": email,
            "program": dict(PROGRAM),
            "base_year_level": 1,
            "current_year_level": cur_year,
            "entry_sy": entry_sy,
            "expected_grad_sy": ENTRY_SYS[min(ENTRY_SYS.index(entry_sy)+max_year-1, len(ENTRY_SYS)-1)],
            "active": True,
            "created_at": datetime.now(timezone.utc),
        }

        students.update_one({"_id": sid}, {"$set": doc}, upsert=True)

    print("seed_students: done.")

if __name__ == "__main__":
    main()
