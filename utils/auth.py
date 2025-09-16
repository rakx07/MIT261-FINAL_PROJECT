# utils/auth.py
from __future__ import annotations
import secrets, string, re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

import bcrypt
from bson import ObjectId
from db import col

# ---------------- core helpers ----------------
def _users():
    return col("users")

def _now():
    return datetime.now(tz=timezone.utc)

def _nemail(e: str) -> str:
    return (e or "").strip().lower()

def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def _check_pw(pw: str, stored) -> bool:
    """Accept string or bytes hashes."""
    if not stored:
        return False
    try:
        if isinstance(stored, bytes):
            return bcrypt.checkpw(pw.encode("utf-8"), stored)
        if isinstance(stored, str):
            return bcrypt.checkpw(pw.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        pass
    return False

def _public(u: dict) -> dict:
    if not u: return {}
    out = dict(u)
    out.pop("password_hash", None)
    out.pop("reset_token", None)
    if isinstance(out.get("_id"), ObjectId):
        out["_id"] = str(out["_id"])
    return out

def _ensure_indexes():
    _users().create_index("email", unique=True)
    _users().create_index([("role", 1), ("active", 1)])

# ---------------- CRUD ----------------
def get_user(email: str) -> Optional[dict]:
    return _users().find_one({"email": _nemail(email)})

def list_users(filter_: Optional[dict] = None, fields: Optional[dict] = None) -> List[dict]:
    fields = fields or {"password_hash": 0, "reset_token": 0}
    return [_public(d) for d in _users().find(filter_ or {}, fields)]

def create_user(email: str, name: str, role: str, password: str,
                active: bool = True, must_change_password: bool = False, **extra) -> dict:
    _ensure_indexes()
    email = _nemail(email)
    if (u := get_user(email)):
        return _public(u)
    doc = {
        "email": email, "name": name or email, "role": role,
        "active": bool(active), "must_change_password": bool(must_change_password),
        "password_hash": _hash(password),
        "created_at": _now(), "updated_at": _now(), **(extra or {})
    }
    _users().insert_one(doc)
    return _public(doc)

def set_password(email: str, new_password: str, clear_reset: bool = True) -> bool:
    upd = {
        "password_hash": _hash(new_password),
        "must_change_password": False,
        "password_changed_at": _now(),
        "updated_at": _now()
    }
    if clear_reset:
        res = _users().update_one({"email": _nemail(email)}, {"$set": upd, "$unset": {"reset_token": ""}})
    else:
        res = _users().update_one({"email": _nemail(email)}, {"$set": upd})
    return res.modified_count == 1

def verify_login(email: str, password: str) -> Optional[dict]:
    u = get_user(email)
    if not u or not u.get("active", True):
        return None
    if not _check_pw(password, u.get("password_hash")):
        return None
    _users().update_one({"_id": u["_id"]}, {"$set": {"last_login_at": _now()}})
    return u

def require_password_change(email: str, flag: bool = True) -> bool:
    return _users().update_one({"email": _nemail(email)},
                               {"$set": {"must_change_password": bool(flag), "updated_at": _now()}}).modified_count == 1

def set_role(email: str, role: str) -> bool:
    return _users().update_one({"email": _nemail(email)},
                               {"$set": {"role": role, "updated_at": _now()}}).modified_count == 1

def deactivate_user(email: str) -> bool:
    return _users().update_one({"email": _nemail(email)},
                               {"$set": {"active": False, "updated_at": _now()}}).modified_count == 1

def ensure_default_admin(email: str, password: str = "Admin@1234", reset_password: bool = True) -> dict:
    """Create or repair admin. If exists and reset_password=True, set new password + force change."""
    _ensure_indexes()
    email = _nemail(email)
    u = get_user(email)
    if not u:
        return create_user(email=email, name="Administrator", role="admin",
                           password=password, must_change_password=True)
    updates = {"role": "admin", "active": True, "updated_at": _now()}
    if reset_password:
        updates.update({
            "password_hash": _hash(password),
            "must_change_password": True,
            "password_changed_at": _now(),
        })
    _users().update_one({"_id": u["_id"]}, {"$set": updates})
    u.update(updates)
    return _public(u)

# ---------------- temp password generator ----------------
def _gen_temp_pw(n: int = 12) -> str:
    """Guaranteed complexity: lower, upper, digit, symbol + random rest."""
    n = max(8, n)
    pools = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*"),
    ]
    rest_len = n - len(pools)
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    pools += [secrets.choice(alphabet) for _ in range(rest_len)]
    # Fisher–Yates shuffle
    for i in range(len(pools)-1, 0, -1):
        j = secrets.randbelow(i+1)
        pools[i], pools[j] = pools[j], pools[i]
    return "".join(pools)

# ---------------- imports (teachers/students) ----------------
def import_from_collection(source: str, role: str, email_field: str, name_field: str,
                           must_change_password: bool = True) -> Tuple[List[Dict[str, str]], int, int]:
    created, scanned, inserted = [], 0, 0
    for d in col(source).find({}, {email_field: 1, name_field: 1}):
        scanned += 1
        email = _nemail(d.get(email_field, ""))
        name  = d.get(name_field) or email
        if not email or get_user(email):
            continue
        temp = _gen_temp_pw()
        _users().insert_one({
            "email": email, "name": name, "role": role, "active": True,
            "password_hash": _hash(temp),
            "must_change_password": bool(must_change_password),
            "source": f"import_{source}",
            "created_at": _now(), "updated_at": _now()
        })
        created.append({"email": email, "name": name, "role": role, "temp_password": temp})
        inserted += 1
    return created, scanned, inserted

# ---------------- optional reset tokens ----------------
def start_password_reset(email: str) -> Optional[str]:
    u = get_user(email)
    if not u: return None
    token = secrets.token_urlsafe(24)
    _users().update_one({"_id": u["_id"]},
                        {"$set": {"reset_token": token, "updated_at": _now(), "must_change_password": True}})
    return token

def complete_password_reset(email: str, token: str, new_password: str) -> bool:
    u = get_user(email)
    if not u or token != u.get("reset_token"):
        return False
    return set_password(email, new_password, clear_reset=True)

# ---------------- admin: issue a fresh temp (show once) ----------------
def issue_temp_password(email: str, n: int = 12) -> Optional[Dict[str, str]]:
    """Generate a new temp password for an existing account and force change on next login."""
    u = get_user(email)
    if not u:
        return None
    temp = _gen_temp_pw(n)
    _users().update_one(
        {"_id": u["_id"]},
        {"$set": {
            "password_hash": _hash(temp),
            "must_change_password": True,
            "password_changed_at": _now(),
            "updated_at": _now()
        }}
    )
    return {"email": u["email"], "name": u.get("name", u["email"]), "role": u.get("role", ""), "temp_password": temp}
