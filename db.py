import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "mit261")

_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGODB_URI, retryWrites=True)
        _db = _client[DB_NAME]
    return _db

def col(name: str):
    return get_db()[name]
