# app.py
# ─────────────────────────────────────────────────────────────────────────────
# MIT261 Student Analytics — Entry / Login
# - Single place that sets page config
# - Seeds demo accounts on demand
# - Verifies bcrypt password
# - Redirects to role page with st.switch_page()
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
from datetime import datetime
import bcrypt
from pymongo.errors import PyMongoError

# Our tiny DB helper (already in your repo)
from db import col

st.set_page_config(
    page_title="MIT261 Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide default Streamlit chrome a bit
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .small-note {opacity: .7; font-size: 0.9rem;}
      .tight {padding-top: .35rem; padding-bottom: .35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────────── Helpers ──────────────────────────────────────
def _b(x):  # bytes helper
    return x if isinstance(x, (bytes, bytearray)) else str(x).encode("utf-8")


def hash_pw(plain: str) -> str:
    return bcrypt.hashpw(_b(plain), bcrypt.gensalt()).decode("utf-8")


def verify_pw(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_b(plain), _b(hashed))
    except Exception:
        return False


def get_user_by_email(email: str):
    if not email:
        return None
    return col("users").find_one({"email": email.strip().lower()})


def upsert_user(email: str, name: str, role: str, password_plain: str):
    now = datetime.utcnow()
    email_n = email.strip().lower()
    doc = {
        "email": email_n,
        "name": name,
        "role": role,
        "password": hash_pw(password_plain),
        "active": True,
        "updated_at": now,
    }
    col("users").update_one(
        {"email": email_n},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


def create_sample_accounts():
    # Registrar / Faculty / Student / Admin
    upsert_user("registrar@su.edu", "Registrar", "registrar", "reg123")
    upsert_user("0005@su.edu", "Riley Santiago", "faculty", "teach123")
    upsert_user("s00001@students.su.edu", "Morgan Querubin", "student", "stud123")
    upsert_user("admin@su.edu", "Site Admin", "admin", "admin123")


def reset_demo_accounts():
    col("users").delete_many(
        {"email": {"$in": ["registrar@su.edu", "0005@su.edu", "s00001@students.su.edu", "admin@su.edu"]}}
    )
    create_sample_accounts()


def go_to_role_home(role: str):
    role = (role or "").lower()
    # These are your existing pages
    if role == "registrar":
        st.switch_page("pages/1_Registrar.py")
    elif role == "faculty":
        st.switch_page("pages/2_Faculty.py")
    elif role == "student":
        st.switch_page("pages/3_Student.py")
    elif role == "admin":
        st.switch_page("pages/9_Admin_Users.py")
    else:
        st.warning("Unknown role; ask an admin to update your account.")


# ───────────────────────────── UI ───────────────────────────────────────────
def login_view():
    st.markdown("### 🎓 MIT261 Student Analytics")
    st.markdown("<div class='small-note'>Sign in</div>", unsafe_allow_html=True)

    with st.container():
        email = st.text_input("Email", key="login_email", value="", placeholder="you@su.edu")
        pw_col1, pw_col2 = st.columns([1, 14])
        with pw_col1:
            st.write("")  # spacer
        password = st.text_input("Password", type="password", key="login_pw")

        b1, b2 = st.columns([1, 1])
        with b1:
            do_login = st.button("Login", use_container_width=True)
        with b2:
            do_seed = st.button("Create sample accounts", use_container_width=True)

        # Optional tiny reset button
        if st.button("Reset demo accounts", type="secondary"):
            reset_demo_accounts()
            st.success("Demo accounts reset. Try registrar@su.edu / reg123")
            st.stop()

    if do_seed:
        try:
            create_sample_accounts()
            st.success("Sample accounts created. Try registrar@su.edu / reg123")
        except PyMongoError as e:
            st.error(f"Failed creating users: {e}")
        st.stop()

    if do_login:
        user = get_user_by_email(email)
        if not user or not user.get("password") or not verify_pw(password, user["password"]):
            st.error("Invalid credentials.")
            st.stop()

        if not user.get("active", True):
            st.error("Account is disabled. Contact an admin.")
            st.stop()

        # Save to session and route
        st.session_state.user = {
            "email": user["email"],
            "name": user.get("name", ""),
            "role": user.get("role", ""),
        }
        go_to_role_home(user.get("role", ""))


def auto_route_if_logged_in():
    u = st.session_state.get("user")
    if u and u.get("role"):
        go_to_role_home(u["role"])


# ───────────────────────────── Entry ────────────────────────────────────────
if __name__ == "__main__":
    auto_route_if_logged_in()
    login_view()
