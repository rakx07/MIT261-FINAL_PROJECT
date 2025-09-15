# pages/9_Admin_Users.py
import streamlit as st
import pandas as pd
from utils import auth
from utils.mongo_df import docs_to_df

st.set_page_config(page_title="Admin: Users", page_icon="🛠️", layout="wide")

u = st.session_state.get("user")
if not u or u.get("role") != "admin":
    st.error("Admins only.")
    st.stop()

st.title("🛠️ Admin: Users")
tabs = st.tabs(["Create User", "Import from Teachers", "Import from Students", "All Users"])

with tabs[0]:
    st.subheader("Create User")
    email = st.text_input("Email")
    name  = st.text_input("Name")
    role  = st.selectbox("Role", ["student", "teacher", "registrar", "admin"], index=1)
    pw    = st.text_input("Password", type="password")
    mustc = st.checkbox("Must change password on first login", value=True)
    if st.button("Create"):
        if not email or not pw:
            st.error("Email and password are required.")
        else:
            user = auth.create_user(email, name, role, pw, must_change_password=mustc)
            st.success(f"Created (or exists): {user['email']}")

with tabs[1]:
    st.subheader("Import from Teachers")
    if st.button("Create teacher accounts not in Users"):
        created, scanned, inserted = auth.import_from_collection(
            "teachers", role="teacher", email_field="email", name_field="name"
        )
        st.success(f"Scanned {scanned}. Created {inserted}.")
        if created:
            df = pd.DataFrame(created)
            st.warning("Copy or download these temporary passwords now—they are NOT stored in plaintext.")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode(),
                               "new_teacher_accounts.csv", "text/csv")

with tabs[2]:
    st.subheader("Import from Students")
    # Adjust if your students collection uses different fields
    if st.button("Create student accounts not in Users"):
        created, scanned, inserted = auth.import_from_collection(
            "students", role="student", email_field="Email", name_field="Name"
        )
        st.success(f"Scanned {scanned}. Created {inserted}.")
        if created:
            df = pd.DataFrame(created)
            st.warning("Copy or download these temporary passwords now—they are NOT stored in plaintext.")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode(),
                               "new_student_accounts.csv", "text/csv")

with tabs[3]:
    st.subheader("All Users")
    docs = auth.list_users()
    st.dataframe(docs_to_df(docs), use_container_width=True)
    st.caption("Passwords are never shown. Temp passwords appear ONLY at import time.")
