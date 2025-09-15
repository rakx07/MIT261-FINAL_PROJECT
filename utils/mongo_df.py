# utils/mongo_df.py
from bson import ObjectId
import pandas as pd

def docs_to_df(docs, drop_fields=None):
    drop_fields = set(drop_fields or [])
    rows = []
    for d in docs:
        row = {}
        for k, v in d.items():
            if k in drop_fields:
                continue
            row[k] = str(v) if isinstance(v, ObjectId) else v
        rows.append(row)
    return pd.DataFrame(rows)
