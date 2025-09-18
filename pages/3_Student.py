# pages/3_Student.py

from __future__ import annotations
import math
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

from db import col
from utils.auth import require_role, get_current_user


# ----------------------------
# Helpers
# ----------------------------

def _term_label(sy: str | None, sem: int | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem or 0)
    except Exception:
        s = 0
    return f"{sy} S{s}" if s else sy


def _term_sort_key(label: str) -> tuple[int, int]:
    """
    Sort "2023-2024 S1" < "2023-2024 S2" < "2024-2025 S1" properly.
    """
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
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _flatten(e: dict) -> dict:
    term = e.get("term") or {}
    stu = e.get("student") or {}
    sub = e.get("subject") or {}
    prog = e.get("program") or {}
    tch = e.get("teacher") or {}
    return {
        "student_no": stu.get("student_no"),
        "student_name": stu.get("name"),
        "student_email": (stu.get("email") or "").strip().lower(),
        "subject_code": sub.get("code"),
        "subject_title": sub.get("title"),
        "units": sub.get("units"),
        "grade": _to_num_grade(e.get("grade")),
        "remark": e.get("remark"),
        "term_label": _term_label(term.get("school_year"), term.get("semester")),
        "program_code": prog.get("program_code"),
        "teacher_name": tch.get("name"),
        "teacher_email": (tch.get("email") or "").strip().lower(),
    }


@st.cache_data(show_spinner=False)
def _load_enrollments_by_email(email: str) -> pd.DataFrame:
    email = (email or "").strip().lower()
    rows = list(
        col("enrollments").find(
            {"student.email": email},
            {
                "term.school_year": 1,
                "term.semester": 1,
                "student.student_no": 1,
                "student.name": 1,
                "student.email": 1,
                "subject.code": 1,
                "subject.title": 1,
                "subject.units": 1,
                "program.program_code": 1,
                "teacher.name": 1,
                "teacher.email": 1,
                "grade": 1,
                "remark": 1,
            },
        )
    )
    return pd.DataFrame([_flatten(r) for r in rows]) if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_enrollments_by_studentno(student_no: str) -> pd.DataFrame:
    rows = list(
        col("enrollments").find(
            {"student.student_no": student_no},
            {
                "term.school_year": 1,
                "term.semester": 1,
                "student.student_no": 1,
                "student.name": 1,
                "student.email": 1,
                "subject.code": 1,
                "subject.title": 1,
                "subject.units": 1,
                "program.program_code": 1,
                "teacher.name": 1,
                "teacher.email": 1,
                "grade": 1,
                "remark": 1,
            },
        )
    )
    return pd.DataFrame([_flatten(r) for r in rows]) if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _list_students_for_picker() -> List[Tuple[str, str, str]]:
    """
    For registrar/admin/teacher view: return a sorted list of students present
    in enrollments as (label, student_no, email) where label = "Name (Sxxxxx)".
    """
    pipe = [
        {"$group": {
            "_id": "$student.student_no",
            "name": {"$first": "$student.name"},
            "email": {"$first": "$student.email"}
        }},
        {"$sort": {"_id": 1}}
    ]
    out = []
    for r in col("enrollments").aggregate(pipe):
        sno = r.get("_id")
        nm = r.get("name") or ""
        em = (r.get("email") or "").strip().lower()
        if sno:
            out.append((f"{nm} ({sno})", sno, em))
    return out


def _gpa(s: pd.Series) -> float | None:
    s = s.dropna()
    if s.empty:
        return None
    return float(s.mean())


# ----------------------------
# Page
# ----------------------------

def main():
    # ---- Option A: single guard (no custom guard function) ----
    user = require_role("student", "teacher", "registrar", "admin")
    role = (user.get("role") or "").lower()

    st.title("🎓 Student Dashboard")

    # Determine scope: student sees self; others can pick a student.
    df: pd.DataFrame

    if role == "student":
        target_email = (user.get("email") or "").strip().lower()
        df = _load_enrollments_by_email(target_email)

        # Soft fallback by name if email isn’t present in enrollments
        if df.empty:
            nm = user.get("name") or ""
            alt = list(
                col("enrollments").find(
                    {"student.name": nm},
                    {"student.email": 1, "student.student_no": 1},
                    limit=1,
                )
            )
            if alt:
                sno = alt[0].get("student", {}).get("student_no")
                if sno:
                    df = _load_enrollments_by_studentno(sno)

        st.caption("Scope: your enrollments")
    else:
        choices = _list_students_for_picker()
        if not choices:
            st.info("No students found in enrollments.")
            return
        labels = [c[0] for c in choices]
        picked = st.selectbox("View student", labels, index=0)
        idx = labels.index(picked)
        _label, student_no, email = choices[idx]
        # Prefer email when available (usually more reliable after backfill)
        df = _load_enrollments_by_email(email) if email else _load_enrollments_by_studentno(student_no)
        st.caption(f"Scope: {picked}")

    if df.empty:
        st.warning("No enrollments found.")
        return

    # ----------------------------
    # Filters
    # ----------------------------
    cols = st.columns(3)
    with cols[0]:
        terms = sorted(df["term_label"].dropna().unique(), key=_term_sort_key)
        sel_terms = st.multiselect("Term(s)", options=terms, default=terms)
    with cols[1]:
        subjects = sorted(df["subject_code"].dropna().unique())
        sel_subjects = st.multiselect("Subject(s)", options=subjects, default=subjects)
    with cols[2]:
        dept = sorted(df["program_code"].dropna().unique())
        sel_prog = st.multiselect("Program(s)", options=dept, default=dept)

    if sel_terms:
        df = df[df["term_label"].isin(sel_terms)]
    if sel_subjects:
        df = df[df["subject_code"].isin(sel_subjects)]
    if sel_prog:
        df = df[df["program_code"].isin(sel_prog)]

    # ----------------------------
    # Summary metrics
    # ----------------------------
    graded = df.dropna(subset=["grade"]).copy()
    overall_gpa = _gpa(graded["grade"]) if not graded.empty else None

    latest_term = None
    if not graded.empty and not graded["term_label"].isna().all():
        latest_term = (
            sorted(graded["term_label"].dropna().unique(), key=_term_sort_key)[-1]
            if graded["term_label"].dropna().size
            else None
        )
    term_gpa = _gpa(graded[graded["term_label"] == latest_term]["grade"]) if latest_term else None

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Overall GPA", f"{overall_gpa:.2f}" if overall_gpa is not None else "—")
    with m2:
        st.metric("Latest Term", latest_term or "—")
    with m3:
        st.metric("Latest Term GPA", f"{term_gpa:.2f}" if term_gpa is not None else "—")

    # ----------------------------
    # 1) Term GPA trend
    # ----------------------------
    st.subheader("1) GPA Trend by Term")
    if graded.empty:
        st.info("No graded entries yet.")
    else:
        g = (
            graded.groupby("term_label", as_index=False)["grade"]
            .mean()
            .rename(columns={"grade": "gpa"})
        )
        g = g.sort_values(by="term_label", key=lambda s: s.map(_term_sort_key)).set_index("term_label")
        st.line_chart(g)

    # ----------------------------
    # 2) Transcript / Enrollments
    # ----------------------------
    st.subheader("2) Transcript")
    show = df.copy()
    show = show.sort_values(["term_label", "subject_code"], key=lambda s: s.map(_term_sort_key) if s.name == "term_label" else s)
    show = show.rename(
        columns={
            "term_label": "Term",
            "subject_code": "Subject",
            "subject_title": "Title",
            "units": "Units",
            "grade": "Grade",
            "remark": "Remark",
            "teacher_name": "Teacher",
        }
    )
    st.dataframe(
        show[["Term", "Subject", "Title", "Units", "Grade", "Remark", "Teacher"]],
        use_container_width=True,
        height=min(520, 38 + 28 * len(show)),
    )

    # ----------------------------
    # 3) Incomplete / Failed
    # ----------------------------
    st.subheader("3) Incomplete / Failed")
    flags = df[(df["remark"].str.upper().eq("INC")) | (df["grade"].fillna(100) < 75)]
    if flags.empty:
        st.success("No incomplete or failed subjects in the selected scope.")
    else:
        view = flags.copy().rename(columns={"term_label": "Term", "subject_code": "Subject", "subject_title": "Title"})
        st.dataframe(view[["Term", "Subject", "Title", "Grade", "Remark"]], use_container_width=True)

    # ----------------------------
    # 4) Units Summary
    # ----------------------------
    st.subheader("4) Units Summary")
    with_units = df.copy()
    with_units["Units"] = pd.to_numeric(with_units["units"], errors="coerce")
    sum_units = with_units.groupby("term_label", as_index=False)["Units"].sum().sort_values(
        by="term_label", key=lambda s: s.map(_term_sort_key)
    )
    st.dataframe(sum_units.rename(columns={"term_label": "Term"}), use_container_width=True)


if __name__ == "__main__":
    main()
