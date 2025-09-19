# scripts/seed_semesters.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db import col

YEARS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"]

def make_doc(sy: str, sem: int):
    return {
        "_id": f"{sy}_S{sem}",
        "school_year": sy,
        "semester": sem,
        "label": f"{sy} S{sem}",
        "type": "summer" if sem == 3 else "regular",
    }

def main():
    docs = []
    for sy in YEARS:
        for sem in (1, 2, 3):  # S3 = summer
            docs.append(make_doc(sy, sem))
    c = col("semesters")
    for d in docs:
        c.update_one({"_id": d["_id"]}, {"$set": d}, upsert=True)
    print(f"upserted {len(docs)} semester docs")

if __name__ == "__main__":
    main()
