# pages/9_Admin_Users.py
import streamlit as st
import pandas as pd
from utils import auth
from utils.mongo_df import docs_to_df

st.set_page_config(page_title="Admin: Users", page_icon="🛠️", layout="wide")

# --------- access control ---------
u = st.session_state.get("user")
if not u or u.get("role") != "admin":
    st.error("Admins only.")
    st.stop()

# --------- session store for created/issued temps (persists across reruns) ---------
if "last_created_accounts" not in st.session_state:
    st.session_state["last_created_accounts"] = []

st.title("🛠️ Admin: Users")
tabs = st.tabs(["Create User", "Import from Teachers", "Import from Students", "All Users"])

# =========== Tab 0: Create User ===========
with tabs[0]:
    st.subheader("Create User")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        email = st.text_input("Email")
        name  = st.text_input("Name")
    with col2:
        role  = st.selectbox("Role", ["student", "teacher", "registrar", "admin"], index=1)
        mustc = st.checkbox("Must change password on first login", value=True)
    with col3:
        pw    = st.text_input("Password (leave blank to auto-generate)", type="password")

    if st.button("Create", type="primary"):
        if not email:
            st.error("Email is required.")
        else:
            temp_pw = pw or auth._gen_temp_pw()  # generate once on UI side
            auth.create_user(email, name, role, temp_pw, must_change_password=mustc)
            st.session_state["last_created_accounts"] = [{
                "email": email, "name": name or email, "role": role, "temp_password": temp_pw
            }]
            st.success(f"User created (or existed): {email}. Share the temp password below.")

    if st.session_state["last_created_accounts"]:
        df = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now—they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "new_accounts.csv",
            "text/csv",
            key="dl_single",
        )

# =========== Tab 1: Import from Teachers ===========
with tabs[1]:
    st.subheader("Import from Teachers")
    if st.button("Create teacher accounts not in Users", help="Scans the 'teachers' collection"):
        created, scanned, inserted = auth.import_from_collection(
            "teachers", role="teacher", email_field="email", name_field="name"
        )
        st.success(f"Scanned {scanned}. Created {inserted}.")
        st.session_state["last_created_accounts"] = created

    if st.session_state["last_created_accounts"]:
        df = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now—they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "new_teacher_accounts.csv",
            "text/csv",
            key="dl_teachers",
        )

# =========== Tab 2: Import from Students ===========
with tabs[2]:
    st.subheader("Import from Students")
    if st.button("Create student accounts not in Users", help="Scans the 'students' collection"):
        created, scanned, inserted = auth.import_from_collection(
            "students", role="student", email_field="Email", name_field="Name"
        )
        st.success(f"Scanned {scanned}. Created {inserted}.")
        st.session_state["last_created_accounts"] = created

    if st.session_state["last_created_accounts"]:
        df = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now—they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "new_student_accounts.csv",
            "text/csv",
            key="dl_students",
        )

# =========== Tab 3: All Users (with reset tool) ===========
with tabs[3]:
    st.subheader("All Users")
    docs = auth.list_users()
    st.dataframe(docs_to_df(docs), use_container_width=True)
    st.caption("Passwords are never shown. Temp passwords appear ONLY at creation/import/reset time.")

    st.markdown("---")
    st.markdown("### 🔁 Issue a new temporary password")
    all_emails = [d.get("email") for d in docs]
    c1, c2 = st.columns([2,1])
    with c1:
        email_to_reset = st.selectbox("Select user", all_emails, index=0 if all_emails else None)
    with c2:
        temp_len = st.number_input("Length", min_value=8, max_value=32, value=12, step=1)

    if st.button("Issue Temp Password", disabled=not all_emails):
        row = auth.issue_temp_password(email_to_reset, n=int(temp_len))
        if not row:
            st.error("User not found.")
        else:
            st.session_state["last_created_accounts"] = [row]
            st.success(f"Issued a new temporary password for {row['email']}. Share it below.")

    if st.session_state["last_created_accounts"]:
        df = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now—they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "issued_temp_passwords.csv",
            "text/csv",
            key="dl_reset",
        )
