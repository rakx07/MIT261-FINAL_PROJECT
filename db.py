import os
from pymongo import MongoClient
from dotenv import load_dotenv
import streamlit as st

# Load local .env if it exists
load_dotenv()

# Try Streamlit secrets first (on cloud), fallback to .env (local)
MONGODB_URI = (
    st.secrets["MONGODB_URI"]
    if "MONGODB_URI" in st.secrets
    else os.getenv("MONGODB_URI")
)

DB_NAME = (
    st.secrets["DB_NAME"]
    if "DB_NAME" in st.secrets
    else os.getenv("DB_NAME", "mit261")
)

_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(
            MONGODB_URI,
            retryWrites=True,
            serverSelectionTimeoutMS=8000,
            appname="MIT261_STREAMLIT",
        )
        # Fail fast if Atlas/URI/network is misconfigured
        _client.admin.command("ping")
        _db = _client[DB_NAME]
    return _db

def col(name: str):
    return get_db()[name]
