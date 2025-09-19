# scripts/seed_offerings.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from db import col

YEARS = ["2020-2021","2021-2022","2022-2023","2023-2024"]
PROGRAM_ID = "P-ENGED"
PROGRAM_CODE = "BSED-ENGLISH"

def list_teachers():
    t = list(col("teachers").find({}, {"teacher_id":1,"email":1,"name":1}))
    if not t:
        for i in range(1, 16):
            col("teachers").insert_one({
                "teacher_id": f"T{1000+i:04d}",
                "email": f"{i:04d}@su.edu",
                "name": f"Teacher {i}"
            })
        t = list(col("teachers").find({}, {"teacher_id":1,"email":1,"name":1}))
    return t

def subjects_from_curriculum():
    cur = col("curriculum").find_one({"courseCode":"BSED-ENGLISH"})
    if not cur:
        raise SystemExit("No curriculum found for BSED-ENGLISH")
    return cur["subjects"]

def main(sections=("11-1","11-2","21-1","21-2","31-1","31-2","41-1","41-2")):
    teachers = list_teachers()
    teacher_ids = [t["teacher_id"] for t in teachers]
    subs = subjects_from_curriculum()
    offerings = col("offerings")

    for sy in YEARS:
        for sem in (1, 2, 3):  # include summer
            pool = [s for s in subs if int(s["semester"]) == sem]
            if sem == 3 and not pool:
                pool = random.sample(subs, k=min(3, len(subs)))
                for s in pool: s["semester"] = 3

            for s in pool:
                year_level = int(s["yearLevel"])
                for section in sections:
                    if not section.startswith(f"{year_level}"):
                        continue
                    teacher_id = random.choice(teacher_ids)
                    oid = f"O{sy.replace('-','')}-{s['subjectCode']}-S{sem}-{section}"
                    doc = {
                        "_id": oid, "offering_id": oid,
                        "school_year": sy, "semester": sem,
                        "subject_code": s["subjectCode"],
                        "program_id": PROGRAM_ID, "program_code": PROGRAM_CODE,
                        "year_level": year_level, "section": section,
                        "teacher_id": teacher_id,
                        "units": int(s.get("units") or s.get("lec", 0) + s.get("lab", 0)),
                    }
                    offerings.update_one({"_id": oid}, {"$set": doc}, upsert=True)
    print("offerings upserted.")

if __name__ == "__main__":
    main()
