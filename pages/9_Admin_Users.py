# pages/9_Admin_Users.py

from __future__ import annotations

import pandas as pd
import streamlit as st

from db import col
from utils import auth
from utils.mongo_df import docs_to_df
from utils.auth import require_role

from utils.auth import current_user  # already available in your auth helpers

def _user_header(u: dict | None):
    if not u:
        return
    st.markdown(
        f"""
        <div style="margin-top:-8px;margin-bottom:10px;padding:10px 12px;
             border:1px solid rgba(0,0,0,.06); border-radius:10px;
             background:linear-gradient(180deg,#0b1220 0%,#0e1729 100%);
             color:#e6edff;">
          <div style="font-size:14px;opacity:.85">Signed in as</div>
          <div style="font-size:16px;font-weight:700;">{u.get('name','')}</div>
          <div style="font-size:13px;opacity:.75;">{u.get('email','')}</div>
          <div style="margin-top:6px;font-size:12px;display:inline-block;
               padding:2px 6px;border:1px solid rgba(255,255,255,.12);
               border-radius:6px;letter-spacing:.4px;">
            {(u.get('role','') or '').upper()}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )





# ---- auth gate (admins only) ----
user = require_role("admin")

st.title("🛠️ Admin: Users")
try:
    u = user  # often set by user = require_role("admin")
except NameError:
    u = current_user()
_user_header(u)




# keep a place to show any newly generated temp passwords
if "last_created_accounts" not in st.session_state:
    st.session_state["last_created_accounts"] = []

tabs = st.tabs(
    [
        "Create User",
        "Import from Teachers",
        "Import from Students",
        "Sync from Enrollments",   # ⬅️ NEW
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
# 4) Sync from Enrollments (NEW)
# -------------------------------
with tabs[3]:
    st.subheader("Sync accounts from the enrollments collection")

    def _distinct_from_enrollments(path_email: str, path_name: str) -> list[dict]:
        """Return [{'email':..., 'name':...}, ...] from enrollments.<path> (lower-cased emails, deduped)."""
        pipe = [
            {"$match": {path_email: {"$exists": True, "$ne": ""}}},
            {"$group": {"_id": f"${path_email}", "name": {"$last": f"${path_name}"}}},
            {"$sort": {"_id": 1}},
        ]
        seen, out = set(), []
        for r in col("enrollments").aggregate(pipe):
            em = (r.get("_id") or "").strip().lower()
            nm = r.get("name") or em
            if em and em not in seen:
                seen.add(em)
                out.append({"email": em, "name": nm})
        return out

    # gather all emails already present in Users (any role)
    existing = {u["email"].strip().lower() for u in auth.list_users({}, {"email": 1}) if u.get("email")}
    st.caption(f"Existing users in DB: **{len(existing)}**")

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Students (from enrollments.student.email)**")
        studs = _distinct_from_enrollments("student.email", "student.name")
        studs_missing = [d for d in studs if d["email"] not in existing]
        st.write(f"Found in enrollments: {len(studs)} · Missing in Users: {len(studs_missing)}")
        if studs_missing:
            dfm = pd.DataFrame(studs_missing)
            st.dataframe(dfm, use_container_width=True, hide_index=True, height=240)
            if st.button("Create missing student accounts", type="primary", key="mk_students"):
                created = []
                for d in studs_missing:
                    temp = auth._gen_temp_pw()
                    auth.create_user(d["email"], d["name"], "student", temp, must_change_password=True)
                    created.append({**d, "role": "student", "temp_password": temp})
                st.session_state["last_created_accounts"] = created
                st.success(f"Created {len(created)} student account(s). See temps below.")
        else:
            st.info("No missing student accounts.")

    with colR:
        st.markdown("**Teachers (from enrollments.teacher.email)**")
        teach = _distinct_from_enrollments("teacher.email", "teacher.name")
        teach_missing = [d for d in teach if d["email"] not in existing]
        st.write(f"Found in enrollments: {len(teach)} · Missing in Users: {len(teach_missing)}")
        if teach_missing:
            dft = pd.DataFrame(teach_missing)
            st.dataframe(dft, use_container_width=True, hide_index=True, height=240)
            if st.button("Create missing teacher accounts", type="primary", key="mk_teachers_enr"):
                created = []
                for d in teach_missing:
                    temp = auth._gen_temp_pw()
                    auth.create_user(d["email"], d["name"], "teacher", temp, must_change_password=True)
                    created.append({**d, "role": "teacher", "temp_password": temp})
                st.session_state["last_created_accounts"] = created
                st.success(f"Created {len(created)} teacher account(s). See temps below.")
        else:
            st.info("No missing teacher accounts.")

    if st.session_state["last_created_accounts"]:
        df = pd.DataFrame(st.session_state["last_created_accounts"])
        st.warning("Copy or download these temporary passwords now — they are NOT stored in plaintext.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download newly-created accounts (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "new_accounts_from_enrollments.csv",
            "text/csv",
            key="dl_from_enrollments",
        )

# -------------------------------
# 5) Directory / Export / Reset
# -------------------------------
with tabs[4]:
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
                                                                                                                                                                                                                                                                                                                                                                           