# pages/1_Registrar.py
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from db import col
from utils.auth import require_role

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _term_label(sy: str | None, sem: int | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem or 0)
    except Exception:
        s = 0
    return f"{sy} S{s}" if s else sy


def _parse_term_label(label: str) -> Tuple[Optional[str], Optional[int]]:
    if not label or " S" not in label:
        return None, None
    sy, s = label.split(" S", 1)
    try:
        return sy, int(s)
    except Exception:
        return sy, None


def _term_sort_key(label: str) -> tuple[int, int]:
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


def _sort_key_for_series_of_term_labels(s: pd.Series) -> pd.Series:
    return s.map(lambda x: _term_sort_key(x or ""))


def _to_regex(s: str | None) -> Optional[re.Pattern]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return re.compile(s, re.IGNORECASE)
    except Exception:
        return None


def _empty_state(msg: str = "No data for the current filters."):
    st.info(msg)


@st.cache_data(show_spinner=False, ttl=300)
def _all_term_labels() -> List[str]:
    """
    Pull every term label we can find:
      1) Prefer 'semesters' collection (seeded).
      2) Fallback to distinct terms in 'enrollments'.
    Returned list is sorted chronologically.
    """
    labels: set[str] = set()

    # from semesters (preferred)
    for d in col("semesters").find({}, {"_id": 0, "school_year": 1, "semester": 1}):
        lbl = _term_label(d.get("school_year"), d.get("semester"))
        if lbl and " S" in lbl:
            labels.add(lbl)

    # fallback/enrichment: from enrollments
    pipe = [
        {"$group": {"_id": {"sy": "$term.school_year", "sem": "$term.semester"}}},
    ]
    for g in col("enrollments").aggregate(pipe):
        v = g.get("_id") or {}
        lbl = _term_label(v.get("sy"), v.get("sem"))
        if lbl and " S" in lbl:
            labels.add(lbl)

    out = sorted(labels, key=_term_sort_key)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=True, ttl=300)
def load_enrollments_df(
    selected_terms: Tuple[str, ...],
    subject_regex_str: str,
    dept_regex_str: str,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with:
    student_no, student_name, program, program_code, program_name,
    subject_code, subject_title, department, units,
    grade, remark, teacher_name, teacher_email,
    term_label, school_year, semester, base_year, section
    """
    fields = {
        "_id": 0,
        "student.student_no": 1,
        "student.name": 1,
        "student.base_year_level": 1,
        "program.program_code": 1,
        "program.program_name": 1,
        "subject.code": 1,
        "subject.title": 1,
        "subject.department": 1,   # may be missing
        "subject.units": 1,        # may be missing
        "grade": 1,
        "remark": 1,
        "teacher.name": 1,
        "teacher.email": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "section": 1,              # may be missing
    }

    mongo_filter: Dict[str, Any] = {}
    if selected_terms:
        ors = []
        for t in selected_terms:
            sy, sem = _parse_term_label(t)
            if sy and sem is not None:
                ors.append({"term.school_year": sy, "term.semester": sem})
        if ors:
            mongo_filter["$or"] = ors

    rows = list(col("enrollments").find(mongo_filter, fields))
    if not rows:
        return pd.DataFrame(
            columns=[
                "student_no","student_name","program","program_code","program_name",
                "subject_code","subject_title","department","units",
                "grade","remark","teacher_name","teacher_email",
                "term_label","school_year","semester","base_year","section",
            ]
        )

    df = pd.json_normalize(rows)
    ren = {
        "student.student_no": "student_no",
        "student.name": "student_name",
        "student.base_year_level": "base_year",
        "program.program_code": "program_code",
        "program.program_name": "program_name",
        "subject.code": "subject_code",
        "subject.title": "subject_title",
        "subject.department": "department",
        "subject.units": "units",
        "teacher.name": "teacher_name",
        "teacher.email": "teacher_email",
        "term.school_year": "school_year",
        "term.semester": "semester",
    }
    df = df.rename(columns=ren)

    # Ensure expected columns always exist (avoid KeyError on subset)
    for c in [
        "department","units","program_code","program_name",
        "base_year","section","teacher_name","teacher_email",
        "remark","grade","subject_code","subject_title",
        "school_year","semester",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    # Compose helpers
    df["program"] = df.apply(
        lambda r: r.get("program_name") or r.get("program_code") or "(Unknown)", axis=1
    )
    df["term_label"] = df.apply(
        lambda r: _term_label(r.get("school_year"), r.get("semester")), axis=1
    )

    # Types
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).astype(float)
    df["base_year"] = pd.to_numeric(df["base_year"], errors="coerce")

    # Optional regex filters (only if column exists)
    subj_re = _to_regex(subject_regex_str)
    if subj_re is not None and "subject_code" in df.columns:
        df = df[df["subject_code"].astype(str).str.contains(subj_re)]

    dept_re = _to_regex(dept_regex_str)
    if dept_re is not None and "department" in df.columns:
        df = df[df["department"].astype(str).str.contains(dept_re)]

    # Final sort
    if "term_label" in df.columns:
        df = df.sort_values(
            by=["term_label", "subject_code", "student_no"],
            key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
        )

    # Safe subset (columns guaranteed above)
    keep = [
        "student_no","student_name","program","program_code","program_name",
        "subject_code","subject_title","department","units",
        "grade","remark","teacher_name","teacher_email",
        "term_label","school_year","semester","base_year","section",
    ]
    return df[keep]



# ──────────────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────────────

def render_probation(df: pd.DataFrame, max_gpa: float, fail_pct_threshold: float = 30.0, pass_mark: float = 75.0):
    """Academic Probation list with requested layout."""
    title = "⚠️ Academic Probation"
    st.subheader(f"{title}")
    st.caption(f"Criteria: GPA < {int(max_gpa)}% or ≥{int(fail_pct_threshold)}% fails")

    if df.empty:
        _empty_state()
        return

    # Per-student aggregates for the current filter scope
    agg = df.groupby(["student_no", "student_name"], as_index=False).agg(
        GPA=("grade", "mean"),
        Units=("units", "sum"),
        High=("grade", "max"),
        Low=("grade", "min"),
        Total=("grade", "size"),
        Fails=("grade", lambda s: (s < pass_mark).sum()),
    )
    agg["Fail_%"] = (agg["Fails"] / agg["Total"] * 100.0).round(0)
    agg["GPA"] = agg["GPA"].round(0)
    agg["Units"] = agg["Units"].fillna(0).round(0).astype(int)

    # Attach simple program code & year
    prog_map = (
        df.groupby(["student_no"], as_index=False)
          .agg(program_code=("program_code", "first"),
               program=("program", "first"),
               base_year=("base_year", "first"))
    )
    out = agg.merge(prog_map, on="student_no", how="left")
    out["Prog"] = out["program_code"].fillna(out["program"]).fillna("")
    out["Yr"] = pd.to_numeric(out["base_year"], errors="coerce").fillna("").astype(object)

    mask = (out["GPA"] < float(max_gpa)) | (out["Fail_%"] >= float(fail_pct_threshold))
    out = out.loc[mask].copy().sort_values(["GPA", "Low", "High"]).reset_index(drop=True)
    out.index = out.index + 1  # 1-based row #
    out.rename_axis("#", inplace=True)

    view = out[["student_no", "student_name", "Prog", "Yr", "GPA", "Units", "High", "Low"]].rename(
        columns={"student_no": "ID", "student_name": "Name"}
    )
    st.dataframe(view, use_container_width=True)


def render_failed_students(df: pd.DataFrame, pass_mark: float = 75.0):
    """Failed students report (by subject) including teacher + term columns and CSV download."""
    st.subheader("📉 Failed Students (by Subject)")

    if df.empty:
        _empty_state("No enrollments in scope.")
        return

    # Subject picker from current scope
    codes = sorted(df["subject_code"].dropna().unique())
    chosen = st.multiselect("Subject(s)", options=codes, default=codes, key="fail_subjects")
    if chosen:
        df = df[df["subject_code"].isin(chosen)]

    failed = df[df["grade"] < pass_mark].copy()
    if failed.empty:
        _empty_state("No failed grades within the current filters.")
        return

    show = failed[
        [
            "student_no","student_name","program",
            "subject_code","subject_title","grade","remark",
            "teacher_name","teacher_email","section",
            "term_label","school_year","semester",
        ]
    ].rename(
        columns={
            "student_no":"ID",
            "student_name":"Student",
            "program":"Program",
            "subject_code":"Subject",
            "subject_title":"Title",
            "teacher_name":"Teacher",
            "teacher_email":"Teacher Email",
            "term_label":"Term",
            "school_year":"SY",
            "semester":"Sem",
            "section":"Section",
        }
    )

    st.dataframe(show, use_container_width=True, height=min(560, 40 + 28 * len(show)))

    # CSV download
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="failed_students.csv", mime="text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum Progress & Advising
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def _curriculum_courses() -> List[Tuple[str, str]]:
    """Returns list of (courseCode, courseName) from 'curriculum' collection."""
    out: List[Tuple[str, str]] = []
    for d in col("curriculum").find({}, {"_id": 0, "courseCode": 1, "courseName": 1}).sort("courseCode", 1):
        code = d.get("courseCode") or ""
        name = d.get("courseName") or code
        if code:
            out.append((code, name))
    return out


def _load_curriculum(course_code: str) -> Optional[dict]:
    return col("curriculum").find_one({"courseCode": course_code})


def _find_student_any(identifier: str) -> Optional[dict]:
    """Find student by student_no or email (students collection), fallback to users(role=student)."""
    ident = (identifier or "").strip().lower()
    if not ident:
        return None

    s = col("students").find_one(
        {"$or": [{"student_no": ident.upper()}, {"email": ident}]},
        {"_id": 0, "student_no": 1, "name": 1, "email": 1, "program.program_code": 1, "program.program_name": 1,
         "base_year_level": 1, "curriculum_year": 1}
    )
    if s:
        return s

    u = col("users").find_one({"role": "student", "email": ident}, {"_id": 0, "email": 1, "name": 1})
    return u


def _enrollment_grade_map(student_no: str | None, email: str | None) -> Dict[str, float]:
    """Map subject_code -> best/last numeric grade for this student across all time."""
    if not student_no and not email:
        return {}
    q = {"$or": [{"student.student_no": student_no}, {"student.email": email}]}
    rows = col("enrollments").find(q, {"_id": 0, "subject.code": 1, "grade": 1})
    out: Dict[str, float] = {}
    for r in rows:
        code = (r.get("subject") or {}).get("code")
        g = r.get("grade")
        try:
            gv = float(g)
        except Exception:
            continue
        if code:
            out[code] = gv  # overwrite → keeps the last seen grade
    return out


def render_curriculum_advising():
    st.subheader("📘 Curriculum Progress & Advising")

    courses = _curriculum_courses()
    if not courses:
        _empty_state("No curriculum records found.")
        return

    code_options = [c for c, _ in courses]
    labels = [f"{c} — {n}" for c, n in courses]
    if "BSIT" in code_options:
        default_idx = code_options.index("BSIT")
    else:
        default_idx = 0

    colA, colB = st.columns([1, 1])
    with colA:
        picked = st.selectbox("Select a Course Code:", options=list(range(len(labels))), index=default_idx,
                              format_func=lambda i: labels[i])
        course_code = code_options[picked]
    with colB:
        student_query = st.text_input("Enter Student ID or Email:")

    if not student_query:
        st.caption("Tip: enter a student number like `S00001` or an email.")
        return

    stu = _find_student_any(student_query)
    if not stu:
        _empty_state("Student not found.")
        return

    curr = _load_curriculum(course_code)
    if not curr:
        _empty_state("Curriculum record not found for this course.")
        return

    # Gather grades by subject
    subj_grade = _enrollment_grade_map(stu.get("student_no"), stu.get("email"))

    # Student Info card
    box = st.container(border=True)
    with box:
        st.markdown("**Student Information**")
        lines = []
        nm = stu.get("name") or "(Unknown)"
        sid = stu.get("student_no") or "—"
        prog = curr.get("courseName") or course_code
        yr = stu.get("base_year_level") or "—"
        cy = stu.get("curriculum_year") or curr.get("curriculumYear") or "—"
        lines.append(f"**Name:** {nm}")
        lines.append(f"**Student ID:** {sid}")
        lines.append(f"**Course:** {course_code} – {prog}")
        lines.append(f"**Year Level:** {yr}")
        lines.append(f"**Curriculum Year:** {cy}")
        st.write("\n\n".join(lines))

    # Build curriculum table grouped by year/semester
    subjects = curr.get("subjects") or []
    if not subjects:
        _empty_state("No subjects defined in this curriculum.")
        return

    # organize
    dfc = pd.DataFrame(subjects)
    # Normalize columns
    for c in ["lec", "lab", "units"]:
        dfc[c] = pd.to_numeric(dfc.get(c, 0), errors="coerce").fillna(0).astype(int)
    dfc["grade"] = dfc["subjectCode"].map(lambda c: subj_grade.get(c))

    # Group by academic year, show semesters 1/2/3 (3 = summer)
    for year in sorted(dfc["yearLevel"].dropna().unique()):
        st.markdown(f"### 🧭 Year {int(year)}")
        for sem in [1, 2, 3]:
            block = dfc[(dfc["yearLevel"] == year) & (dfc["semester"] == sem)]
            if block.empty:
                continue
            st.markdown(f"**First Semester**" if sem == 1 else ("**Second Semester**" if sem == 2 else "**Summer**"))
            show = block[
                ["subjectCode", "subjectName", "grade", "lec", "lab", "units", "prerequisites"]
            ].rename(
                columns={
                    "subjectCode": "Subject Code",
                    "subjectName": "Description",
                    "lec": "Lec Hours",
                    "lab": "Lab Hours",
                    "units": "Units",
                }
            )
            st.dataframe(show, use_container_width=True, height=min(420, 40 + 28 * len(show)))

# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

def main():
    require_role("registrar", "admin")

    st.title("Registrar Dashboard")

    # Optional logout on the right (won't crash if helper isn't present)
    try:
        from utils.auth import signout_button
        signout_button(align="right")
    except Exception:
        pass

    with st.expander("Filters", expanded=True):
        all_terms = _all_term_labels()
        # pick the most recent 4 as defaults (safe if <4 available)
        default_terms = tuple(all_terms[-4:]) if all_terms else tuple()
        sel_terms = st.multiselect("Term(s)", options=all_terms, default=default_terms, key="reg_terms")

        col1, col2 = st.columns(2)
        with col1:
            subj_pat = st.text_input("Subject code (exact or regex)", value="")
        with col2:
            dept_pat = st.text_input("Department (exact or regex)", value="")

        col3, col4 = st.columns(2)
        with col3:
            deans_min = st.number_input("Dean's List minimum GPA", min_value=75, max_value=99, value=90, step=1)
        with col4:
            probation_max = st.number_input("Probation maximum GPA", min_value=60, max_value=85, value=75, step=1)

    df = load_enrollments_df(tuple(sel_terms), subj_pat, dept_pat)

    st.markdown("## Student Academic Standing Report")
    c1, c2, c3 = st.columns(3)
    with c1:
        want_gpa = st.checkbox("GPA Reports", value=False)
    with c2:
        want_deans = st.checkbox("Dean's List", value=False)
    with c3:
        want_prob = st.checkbox("Probation", value=True)

    if st.button("Apply filters & generate", type="primary"):
        if want_prob:
            render_probation(df, max_gpa=float(probation_max))
        if want_deans:
            st.subheader("🏅 Dean’s List")
            _empty_state("Hook up here if you want a Dean's List table!")
        if want_gpa:
            st.subheader("📊 GPA Reports")
            _empty_state("Hook up here if you want GPA distributions!")

    st.markdown("---")
    render_failed_students(df)

    st.markdown("---")
    render_curriculum_advising()


if __name__ == "__main__":
    main()
