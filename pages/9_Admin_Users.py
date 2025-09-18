# pages/9_Admin_Users.py

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import auth
from utils.mongo_df import docs_to_df
from utils.auth import require_role

# ---- auth gate (admins only) ----
user = require_role("admin")

st.title("🛠️ Admin: Users")

# keep a place to show any newly generated temp passwords
if "last_created_accounts" not in st.session_state:
    st.session_state["last_created_accounts"] = []

tabs = st.tabs(
    [
        "Create User",
        "Import from Teachers",
        "Import from Students",
        "Directory / Export / Reset",
    ]
)

# -------------------------------
# 1) Create User
# -------------------------------
with tabs[0]:
    st.subheader("Create User")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        email = st.text_input("Email")
        name = st.text_input("Name")
    with col2:
        role = st.selectbox("Role", ["student", "teacher", "registrar", "admin"], index=1)
        mustc = st.checkbox("Must change password on first login", value=True)
    with col3:
        pw = st.text_input("Password (leave blank to auto-generate)", type="password")

    if st.button("Create", type="primary"):
        if not email:
            st.error("Email is required.")
        else:
            temp_pw = pw or auth._gen_temp_pw()
            auth.create_user(email, name, role, temp_pw, must_change_password=mustc)
            st.session_state["last_created_accounts"] = [
                {"email": email, "name": name or email, "role": role, "temp_password": temp_pw}
            ]
            st.success(f"User created (or existed): {email}. Share the temp password below.")

    if st.session_state["last_created_accounts"]:
        df = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now — they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "new_accounts.csv",
            "text/csv",
            key="dl_single",
        )

# -------------------------------
# 2) Import from Teachers
# -------------------------------
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
        st.warning("Copy or download these temporary passwords now — they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "new_teacher_accounts.csv",
            "text/csv",
            key="dl_teachers",
        )

# -------------------------------
# 3) Import from Students
# -------------------------------
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
        st.warning("Copy or download these temporary passwords now — they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "new_student_accounts.csv",
            "text/csv",
            key="dl_students",
        )

# -------------------------------
# 4) Directory / Export / Reset
# -------------------------------
with tabs[3]:
    st.subheader("User Directory (filter, export, bulk temp reset)")

    # ---- filters
    roles_all = ["student", "teacher", "registrar", "admin"]
    f1, f2, f3, f4 = st.columns([1.4, 1.1, 1.2, 2])

    with f1:
        pick_roles = st.multiselect("Role(s)", roles_all, default=roles_all)
    with f2:
        status_pick = st.selectbox("Status", ["Any", "Active only", "Inactive only"], index=0)
    with f3:
        must_change = st.selectbox("Must change PW", ["Any", "Yes", "No"], index=0)
    with f4:
        q = st.text_input("Search (email or name contains)", placeholder="e.g. @su.edu or 'Sofia'")

    # Build a Mongo-ish filter for list_users(); any fuzzy text is applied in-memory.
    db_filter = {}
    if pick_roles and len(pick_roles) < len(roles_all):
        db_filter["role"] = {"$in": pick_roles}
    if status_pick != "Any":
        db_filter["active"] = status_pick == "Active only"
    if must_change != "Any":
        db_filter["must_change_password"] = must_change == "Yes"

    docs = auth.list_users(db_filter)  # password fields are already excluded by utils.auth
    df = docs_to_df(docs)

    # in-memory text search
    if q:
        qlow = q.lower()
        mask = df["email"].str.lower().str.contains(qlow, na=False) | df["name"].str.lower().str.contains(qlow, na=False)
        df = df[mask]

    if df.empty:
        st.info("No users match your filters.")
        st.stop()

    # show a concise view + selection checkboxes for bulk operations
    show_cols = [c for c in ["email", "name", "role", "active", "must_change_password", "last_login_at", "created_at"] if c in df.columns]
    view = df[show_cols].copy()
    view.insert(0, "select", False)

    edited = st.data_editor(
        view,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "select": st.column_config.CheckboxColumn("select", help="Select rows for bulk temp reset"),
            "active": st.column_config.CheckboxColumn("active", disabled=True),
            "must_change_password": st.column_config.CheckboxColumn("must_change_password", disabled=True),
        },
    )

    # ---- Export buttons
    exp_left, exp_right = st.columns([1, 1])
    with exp_left:
        st.download_button(
            "⬇️ Download CSV (filtered)",
            edited.drop(columns=["select"]).to_csv(index=False).encode("utf-8"),
            "users_filtered.csv",
            "text/csv",
        )
    with exp_right:
        st.download_button(
            "⬇️ Download emails (one per line)",
            "\n".join(edited["email"].tolist()).encode("utf-8"),
            "emails_filtered.txt",
            "text/plain",
        )

    st.markdown("---")

    # ---- Bulk temp password reset
    picks = edited[edited["select"]]["email"].tolist()
    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.metric("Selected", len(picks))
    with r2:
        temp_len = st.number_input("Temp length", min_value=8, max_value=32, value=12, step=1)
    with r3:
        bulk = st.button("🔁 Issue NEW temp passwords to selected users", disabled=(len(picks) == 0), type="primary")

    if bulk:
        out = []
        for em in picks:
            row = auth.issue_temp_password(em, n=int(temp_len))
            # row → {email, name, role, temp_password}
            if row:
                out.append(row)
        if not out:
            st.warning("No temp passwords issued.")
        else:
            st.session_state["last_created_accounts"] = out
            st.success(f"Issued {len(out)} temporary password(s). Share them securely below.")

    # Show any newly issued temps (also from other tabs)
    if st.session_state["last_created_accounts"]:
        df_new = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now — they are NOT stored in plaintext.")
        st.dataframe(df_new, use_container_width=True)
        st.download_button(
            "Download issued temps (CSV)",
            df_new.to_csv(index=False).encode("utf-8"),
            "issued_temp_passwords.csv",
            "text/csv",
            key="dl_reset_bulk",
        )

    st.caption("Note: existing passwords are never shown anywhere. Only newly created or newly issued temp passwords appear here.")
