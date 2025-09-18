import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db import col
from datetime import datetime

def run():
    # 1) Remove unused legacy blocks
    to_unset = {
        "class": "",
        "college": "",
        "university": "",
        "previous_program": "",
    }
    res = col("enrollments").update_many({}, {"$unset": to_unset})
    print("unset legacy blocks:", res.modified_count)

    # 2) Normalize program to BSED-ENGLISH when student is in that course
    #    (You pasted that all students are BSED-ENGLISH.)
    res = col("enrollments").update_many(
        {},
        {"$set": {
            "program.program_code": "BSED-ENGLISH",
            "program.program_name": "BSED English",
            "updated_at": datetime.utcnow()
        }}
    )
    print("normalized program:", res.modified_count)

    # 3) Optional: keep only fields we use in reports — if some docs still have noisy keys
    #    you can $unset them similarly here.

if __name__ == "__main__":
    run()
    print("Cleanup done.")
