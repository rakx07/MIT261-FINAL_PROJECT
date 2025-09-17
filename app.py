# app.py
import streamlit as st
from utils import auth

st.set_page_config(page_title="MIT261 Student Analytics", page_icon="🎓", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# Header & helpers
# ───────────────────────────────────────────────────────────────────────────────

def header():
    st.markdown("## 🎓 MIT261 Student Analytics")
    u = st.session_state.get("user")
    if u:
        st.caption(f"Logged in as **{u['email']}** · role: **{u['role']}**")
        if st.button("Log out", type="secondary"):
            st.session_state.pop("user", None)
            st.session_state.pop("pending_user", None)
            st.rerun()

def show_change_password():
    u = st.session_state.get("pending_user")
    st.info("You must set a new password to continue.")
    p1 = st.text_input("New password", type="password")
    p2 = st.text_input("Confirm new password", type="password")
    if st.button("Update password"):
        if not p1 or p1 != p2:
            st.error("Passwords do not match.")
            return
        if auth.set_password(u["email"], p1):
            st.session_state["user"] = auth._public(auth.get_user(u["email"]))
            st.session_state.pop("pending_user", None)
            st.success("Password updated.")
            st.rerun()
        else:
            st.error("Failed to update password.")

def show_login():
    st.subheader("Sign in")
    email = st.text_input("Email")
    pw    = st.text_input("Password", type="password")
    c1, c2 = st.columns([1,1])

    if c1.button("Login", use_container_width=True):
        u = auth.verify_login(email, pw)
        if not u:
            st.error("Invalid email/password or inactive account.")
        elif u.get("must_change_password"):
            st.session_state["pending_user"] = auth._public(u)
            st.rerun()
        else:
            st.session_state["user"] = auth._public(u)
            st.success("Welcome!")
            st.rerun()

    if c2.button("Create default admin", use_container_width=True):
        a = auth.ensure_default_admin("admin@su.edu", "Admin@1234", reset_password=True)
        st.success(
            f"Admin ensured: {a['email']} / temp password **Admin@1234** "
            "(must change on first login)."
        )

# ───────────────────────────────────────────────────────────────────────────────
# Entry
# ───────────────────────────────────────────────────────────────────────────────
header()

# Pending password change?
if "pending_user" in st.session_state and not st.session_state.get("user"):
    show_change_password()
    st.stop()

# Not logged in yet → show login
if not st.session_state.get("user"):
    show_login()
    st.stop()

# Logged in
u = st.session_state["user"]

# ───────────────────────────────────────────────────────────────────────────────
# Role-based navigation (Streamlit 1.37+)
# Only show the pages allowed for this role.
# Keep page-level guards in each page as a second layer of protection.
# ───────────────────────────────────────────────────────────────────────────────
pages = []

# Registrar & Admin → Registrar dashboard
if u["role"] in ("registrar", "admin"):
    pages.append(st.Page("pages/1_Registrar.py", title="Registrar", icon="📊"))

# Teacher, Registrar & Admin → Faculty dashboard
if u["role"] in ("teacher", "registrar", "admin"):
    pages.append(st.Page("pages/2_Faculty.py", title="Faculty", icon="🧑‍🏫"))

# Everyone → Student dashboard
pages.append(st.Page("pages/3_Student.py", title="Student", icon="🎒"))

# Admin tools
if u["role"] == "admin":
    pages.append(st.Page("pages/9_Admin_Users.py", title="Admin Users", icon="🛠️"))

nav = st.navigation({"app": pages})

# Optional: if the user is a student, direct them to Student page on first load
try:
    if u["role"] == "student":
        st.switch_page("pages/3_Student.py")
except Exception:
    pass

# Render the selected page
nav.run()
