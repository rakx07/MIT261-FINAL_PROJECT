# pages/2_Faculty.py
from __future__ import annotations
import math
from typing import List, Optional, Tuple, Set

import pandas as pd
import streamlit as st

from db import col
from utils.auth import current_user

from utils.auth import require_role
# teachers see only their scope; registrar/admin can use the teacher filter
user = require_role("teacher", "registrar", "admin")
# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _term_label(sy: str | None, sem: int | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem or 0)
    except Exception:
        s = 0
    return f"{sy} S{s}" if s else sy


def _term_sort_key(label: str) -> tuple[int, int]:
    """Sort "2023-2024 S1", "2023-2024 S2", "2024-2025 S1" properly."""
    if not isinstance(label, str) or " S" not in label:
        return (0, 0)
    sy, s = label.split(" S", 1)
    try:
        start_year = int(sy.split("-")[0])
    except Exception:
        start_year = 0
    try:
        sem = int(s)
    except Exception:
        sem = 0
    return (start_year, sem)


def _to_num_grade(x) -> float | None:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Teacher lists & scope resolution (with offerings fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _teacher_id_by_email(email: str) -> str | None:
    if not email:
        return None
    t = col("teachers").find_one({"email": email}, {"teacher_id": 1})
    return t.get("teacher_id") if t else None


def _offering_keys_for_teacher(email: str) -> Set[tuple]:
    """
    Build {(school_year, semester, subject_code)} a teacher handles via `offerings`.
    offerings doc assumed to have: { teacher_id, subject_code, school_year, semester }
    """
    tid = _teacher_id_by_email(email)
    if not tid:
        return set()
    keys: Set[tuple] = set()
    for o in col("offerings").find({"teacher_id": tid},
                                   {"subject_code": 1, "school_year": 1, "semester": 1}):
        keys.add((o.get("school_year"), o.get("semester"), o.get("subject_code")))
    return keys


def list_teacher_emails_from_enrollments() -> List[Tuple[str, str]]:
    """[(name, email)] discovered from enrollments (if present)."""
    pipe = [
        {"$match": {"teacher.email": {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": "$teacher.email", "name": {"$first": "$teacher.name"}}},
        {"$sort": {"_id": 1}},
    ]
    out = []
    for r in col("enrollments").aggregate(pipe):
        em = (r.get("_id") or "").strip().lower()
        nm = r.get("name") or ""
        if em:
            out.append((nm or em, em))
    return out


def list_all_teachers() -> List[Tuple[str, str]]:
    """[(name, email)] from `teachers` collection (for admin/registrar picker)."""
    out: List[Tuple[str, str]] = []
    for t in col("teachers").find({}, {"name": 1, "email": 1}).sort("email", 1):
        em = (t.get("email") or "").strip().lower()
        nm = t.get("name") or em
        if em:
            out.append((nm, em))
    return out


def scope_df_to_teacher(df: pd.DataFrame, teacher_email: str) -> tuple[pd.DataFrame, str]:
    """
    Try to scope by teacher:
      1) If df has teacher_email (from enrollments), use it.
      2) Else fallback to offerings: match (school_year, semester, subject_code).
    Returns (scoped_df, how).
    """
    target = (teacher_email or "").strip().lower()
    if not target:
        return df.iloc[0:0], "no_email"

    # Approach 1: enrollments already have teacher_email
    if "teacher_email" in df.columns:
        dfe = df[df["teacher_email"].str.lower() == target]
        if len(dfe):
            return dfe, "enrollments_teacher_email"

    # Approach 2: offerings fallback based on term + subject
    keys = _offering_keys_for_teacher(target)
    if not keys:
        return df.iloc[0:0], "no_offerings"

    mask = df.apply(
        lambda r: (r.get("term_school_year"), r.get("term_semester"), r.get("subject_code")) in keys,
        axis=1,
    )
    return df[mask], "offerings_fallback"


# ──────────────────────────────────────────────────────────────────────────────
# Data loader
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_enrollments_df_all() -> pd.DataFrame:
    """
    Load *all* enrollments and flatten into a DataFrame.
    We scope later (so offerings fallback can work even if teacher.email is absent).
    """
    proj = {
        "grade": 1, "remark": 1,
        "term.school_year": 1, "term.semester": 1,
        "student.name": 1, "student.student_no": 1,
        "subject.code": 1, "subject.title": 1,
        "teacher.email": 1, "teacher.name": 1,
        "program.program_code": 1,
    }
    rows = list(col("enrollments").find({}, proj))

    if not rows:
        return pd.DataFrame(columns=[
            "student_no", "student_name",
            "subject_code", "subject_title",
            "grade", "remark",
            "term_label", "term_school_year", "term_semester",
            "teacher_email", "teacher_name",
            "program_code",
        ])

    def flat(e):
        term = e.get("term") or {}
        stu  = e.get("student") or {}
        sub  = e.get("subject") or {}
        tch  = e.get("teacher") or {}
        prog = e.get("program") or {}
        sy   = term.get("school_year")
        sem  = term.get("semester")
        return {
            "student_no": stu.get("student_no"),
            "student_name": stu.get("name"),
            "subject_code": sub.get("code"),
            "subject_title": sub.get("title"),
            "grade": _to_num_grade(e.get("grade")),
            "remark": e.get("remark"),
            "term_label": _term_label(sy, sem),
            "term_school_year": sy,
            "term_semester": sem,
            "teacher_email": (tch.get("email") or "").strip().lower(),
            "teacher_name": tch.get("name"),
            "program_code": prog.get("program_code"),
        }

    df = pd.DataFrame([flat(r) for r in rows])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

def main():
    u = current_user() or {}
    role = (u.get("role") or "").lower()
    user_email = (u.get("email") or "").strip().lower()

    st.title("🏫 Faculty Dashboard")
    st.caption("Teacher scope applies automatically for faculty. Registrars/Admins can filter by teacher.")

    df = load_enrollments_df_all()

    # ── Choose the teacher context
    teacher_email_to_scope: Optional[str] = None
    if role in ("faculty", "teacher"):
        teacher_email_to_scope = user_email
        st.caption(f"Signed in as **{user_email}** (faculty) — auto-scoped.")
    else:
        # registrars/admins: pick a teacher (prefer teachers collection;
        # if none, gracefully fallback to those seen in enrollments)
        all_teachers = list_all_teachers()
        if not all_teachers:
            all_teachers = list_teacher_emails_from_enrollments()

        if all_teachers:
            display = [f"{nm}  ({em})" for nm, em in all_teachers]
            picked = st.selectbox("Filter by teacher (Registrar/Admin)", options=["(All)"] + display, index=0)
            if picked != "(All)":
                idx = display.index(picked)
                teacher_email_to_scope = all_teachers[idx][1]
        else:
            st.info("No teachers found. Showing **all enrollments** for the selected filters.")

    # ── Scope to teacher if one is set
    if teacher_email_to_scope:
        df, how = scope_df_to_teacher(df, teacher_email_to_scope)
        if how == "enrollments_teacher_email":
            st.caption("Scoped by teacher email present in enrollments.")
        elif how == "offerings_fallback":
            st.caption("Scoped by offerings mapping (since enrollments lack teacher email).")
        elif how == "no_offerings":
            st.info("No offerings found for this teacher in the selected filters.")
        elif how == "no_email":
            st.info("No teacher email provided.")
    else:
        st.info("No teacher filter selected; showing **all enrollments** instead.")

    # ── Quick filters
    c1, c2, c3 = st.columns(3)
    with c1:
        terms = sorted(df["term_label"].dropna().unique(), key=_term_sort_key)
        sel_terms = st.multiselect("Term(s)", options=terms, default=terms)
    with c2:
        subjects = sorted(df["subject_code"].dropna().unique())
        sel_subjects = st.multiselect("Subject(s)", options=subjects, default=subjects)
    with c3:
        progs = sorted(df["program_code"].dropna().unique())
        sel_progs = st.multiselect("Program(s)", options=progs, default=progs)

    if sel_terms:
        df = df[df["term_label"].isin(sel_terms)]
    if sel_subjects:
        df = df[df["subject_code"].isin(sel_subjects)]
    if sel_progs:
        df = df[df["program_code"].isin(sel_progs)]

    # ──────────────────────────────────────────────────────────────────────────
    # 1) Class Grade Distribution (Histogram)
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("1) Class Grade Distribution (Histogram)")
    graded = df.dropna(subset=["grade"])
    if graded.empty:
        st.info("No graded entries found for this scope.")
    else:
        # 60–100 in 5-pt bins
        bins = list(range(60, 101, 5))
        hist = pd.cut(graded["grade"], bins=bins, right=True, include_lowest=True).value_counts().sort_index()
        chart_df = pd.DataFrame({"range": hist.index.astype(str), "count": hist.values}).set_index("range")
        st.bar_chart(chart_df)

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Student Progress Tracker (Avg by Term)
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("2) Student Progress Tracker (Avg by Term)")
    if graded.empty:
        st.info("No data to compute term averages.")
    else:
        g = (
            graded.groupby("term_label", as_index=False)["grade"]
            .mean()
            .rename(columns={"grade": "avg_grade"})
        )
        g = g.sort_values(by="term_label", key=lambda s: s.map(_term_sort_key)).set_index("term_label")
        st.line_chart(g)

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Subject Difficulty Heatmap (Fail %)
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("3) Subject Difficulty Heatmap (Fail %)")
    if graded.empty:
        st.info("No data for fail rates.")
    else:
        tmp = graded.copy()
        tmp["is_fail"] = tmp["grade"] < 75
        fail = (
            tmp.groupby("subject_code", as_index=False)
            .agg(total=("grade", "size"), fails=("is_fail", "sum"))
        )
        fail["fail_rate_%"] = (fail["fails"] / fail["total"] * 100).round(2)
        if fail.empty:
            st.info("No subjects to display.")
        else:
            st.dataframe(fail.sort_values("fail_rate_%", ascending=False),
                         use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Intervention Candidates (latest term)
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("4) Intervention Candidates")
    if graded.empty:
        st.info("No graded entries.")
    else:
        latest_term = None
        if not graded["term_label"].isna().all() and graded["term_label"].dropna().size:
            latest_term = sorted(graded["term_label"].dropna().unique(), key=_term_sort_key)[-1]
        cur = graded if latest_term is None else graded[graded["term_label"] == latest_term]
        risk = cur[cur["grade"] < 75].copy().sort_values(["student_name", "grade"])
        if risk.empty:
            st.success("No at-risk students in the latest term.")
        else:
            show = risk[["student_no", "student_name", "subject_code", "grade", "term_label"]].rename(
                columns={
                    "student_no": "Student No",
                    "student_name": "Student",
                    "subject_code": "Subject",
                    "grade": "Grade",
                    "term_label": "Term",
                }
            )
            st.dataframe(show, use_container_width=True,
                         height=min(500, 35 + 28 * len(show)))

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Grade Submission Status
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("5) Grade Submission Status")
    df_status = df.copy()  # keep ungraded rows for this
    if df_status.empty:
        st.info("No enrollments to summarize.")
    else:
        status = (
            df_status.groupby("subject_code")
            .agg(total=("grade", "size"), graded=("grade", lambda s: s.notna().sum()))
            .reset_index()
        )
        status["completion_%"] = (status["graded"] / status["total"] * 100).round(1)
        status = status.rename(columns={
            "subject_code": "Subject",
            "total": "Total Enrollments",
            "graded": "Graded Count",
            "completion_%": "Completion %",
        }).sort_values(["Completion %", "Subject"], ascending=[True, True])
        st.dataframe(status, use_container_width=True)


if __name__ == "__main__":
    main()
