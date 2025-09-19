# pages/1_Registrar.py
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from db import col  # uses your existing db.py helper

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



from utils.auth import require_role
user = require_role("registrar", "admin")   # only registrar/admin may open
# -----------------------------
# Small helpers (robust parsing)
# -----------------------------
from utils.auth import require_role, render_logout_sidebar  # (left as-is; not called)

def _safe_str(x) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x) if x is not None else ""


def _term_label(sy: Optional[str], sem: Optional[int]) -> str:
    sy = _safe_str(sy)
    try:
        sem = int(sem) if sem is not None and _safe_str(sem) != "" else None
    except Exception:
        sem = None
    return f"{sy} S{sem}" if sy and sem is not None else ""


def _parse_term_label(label: object) -> Tuple[str, Optional[int]]:
    """
    Accepts any object; returns (school_year, semester) or ("", None).
    Tolerates NaN, floats, etc.
    """
    s = _safe_str(label)
    if not s or " S" not in s:
        return "", None
    try:
        parts = s.split(" S")
        sy = parts[0].strip()
        sem = int(parts[1].strip())
        return sy, sem
    except Exception:
        return "", None


def _sort_key_for_series_of_term_labels(s: pd.Series):
    """
    Converts '2023-2024 S1' -> (2023-2024, 1) tuple for proper ordering.
    """
    ss = s.astype(str)
    tuples = ss.map(lambda t: _parse_term_label(t))
    # map to sortable tuples: (sy_string, sem_int_or_0)
    return tuples.map(lambda xy: (xy[0], xy[1] if xy[1] is not None else 0))


# -----------------------------
# Data loading
# -----------------------------

@st.cache_data(show_spinner=False, ttl=300)
def load_distinct_terms() -> List[str]:
    fields = {"term.school_year": 1, "term.semester": 1, "_id": 0}
    terms = []
    for r in col("enrollments").find({}, fields):
        sy = r.get("term", {}).get("school_year")
        sem = r.get("term", {}).get("semester")
        lbl = _term_label(sy, sem)
        if lbl:
            terms.append(lbl)
    terms = sorted(list(set(terms)), key=lambda x: (_parse_term_label(x)[0], _parse_term_label(x)[1] or 0))
    return terms


def _to_regex(s: str) -> Optional[re.Pattern]:
    s = _safe_str(s).strip()
    if not s:
        return None
    try:
        return re.compile(s, re.I)
    except Exception:
        return None


@st.cache_data(show_spinner=True, ttl=300)
def load_enrollments_df(
    selected_terms: Tuple[str, ...],
    subject_regex_str: str,
    dept_regex_str: str,
) -> pd.DataFrame:
    """
    Pulls enrollments and returns a tidy DataFrame with robust types:
      columns: student_no, student_name, program, subject_code, subject_title,
      department, grade (numeric), raw_grade (original), remark, teacher_name, teacher_email,
      term_label, school_year, semester, section
    """
    fields = {
        "_id": 0,
        "student.student_no": 1,
        "student.name": 1,
        "program.program_code": 1,
        "program.program_name": 1,
        "subject.code": 1,
        "subject.title": 1,
        "subject.department": 1,     # if not present in docs it's fine (becomes NaN)
        "grade": 1,                # original grade text/number
        "remark": 1,               # INC/DROPPED/etc
        "teacher.name": 1,
        "teacher.email": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "section": 1,
    }

    # Build a naive filter for terms if provided (OR over each term)
    mongo_filter = {}
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
        return pd.DataFrame(columns=[
            "student_no","student_name","program",
            "subject_code","subject_title","department",
            "grade","raw_grade","remark",
            "teacher_name","teacher_email",
            "term_label","school_year","semester","section"
        ])

    df = pd.json_normalize(rows).rename(columns={
        "student.student_no": "student_no",
        "student.name": "student_name",
        "program.program_code": "program_code",
        "program.program_name": "program_name",
        "subject.code": "subject_code",
        "subject.title": "subject_title",
        "subject.department": "department",
        "teacher.name": "teacher_name",
        "teacher.email": "teacher_email",
        "term.school_year": "school_year",
        "term.semester": "semester",
    })

    # Program display string
    df["program"] = df.apply(
        lambda r: r.get("program_name") or r.get("program_code") or "(Unknown)",
        axis=1,
    )

    # Build term_label
    df["term_label"] = df.apply(lambda r: _term_label(r.get("school_year"), r.get("semester")), axis=1)

    # Keep the original grade text and also a numeric version
    df["raw_grade"] = df.get("grade")
    df["grade"] = pd.to_numeric(df["raw_grade"], errors="coerce")  # numeric, may be NaN for INC/etc

    # Optional filters
    subj_re = _to_regex(subject_regex_str)
    if subj_re is not None:
        df = df[df["subject_code"].astype(str).str.contains(subj_re)]

    dept_re = _to_regex(dept_regex_str)
    if dept_re is not None and "department" in df.columns:
        df = df[df["department"].astype(str).str.contains(dept_re)]

    # Sort consistently
    if "term_label" in df.columns:
        df = df.sort_values(
            by=["term_label", "student_no", "subject_code"],
            key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
        )

    # Final column order (include remark & raw_grade for later reports)
    wanted = [
        "student_no","student_name","program",
        "subject_code","subject_title","department",
        "grade","raw_grade","remark",
        "teacher_name","teacher_email",
        "term_label","school_year","semester","section",
    ]
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan
    return df[wanted]


# -----------------------------
# Renderers
# -----------------------------

def _empty_state(msg: str):
    st.info(msg)
    st.stop()


def render_gpa_reports(df: pd.DataFrame):
    st.subheader("GPA Reports")
    if df.empty:
        _empty_state("No rows for the current filters.")
    # Histogram
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df["grade"].astype(float), bins=20)
    ax.set_xlabel("GPA")
    ax.set_ylabel("Students")
    st.pyplot(fig, clear_figure=True)

    # basic stats per student (selected terms)
    out = (
        df.groupby(["student_no", "student_name"], as_index=False)
        .agg(GPA=("grade", "mean"), Subjects=("grade", "size"))
        .sort_values("GPA", ascending=False)
    )
    out["GPA"] = out["GPA"].round(2)
    st.dataframe(out, use_container_width=True)


def render_deans_list(df: pd.DataFrame, min_gpa: float):
    st.subheader("Student Academic Standing → Dean’s List")
    st.caption(f"Threshold: GPA ≥ {min_gpa}")

    if df.empty:
        _empty_state("No rows for the current filters.")

    # Per student per term
    per = (
        df.groupby(["student_no", "student_name", "term_label"], as_index=False)
        .agg(GPA=("grade", "mean"), Subjects=("grade", "size"))
    )
    per = per[per["GPA"] >= float(min_gpa)]
    # sort by term then GPA desc
    if not per.empty:
        per = per.sort_values(
            by=["term_label", "GPA"],
            ascending=[True, False],
            key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
        )
        per["GPA"] = per["GPA"].round(2)
    else:
        st.info("No students met the Dean’s List threshold.")
        return
    st.dataframe(per, use_container_width=True)


# -----------------------------
# PROBATION — updated to support adjustable fail-rate (%)
# -----------------------------
def render_probation(
    df: pd.DataFrame,
    max_gpa: float,
    fail_pct_threshold: float,
    pass_cutoff: float = 75.0,
):
    """
    Place a student on probation for a term if:
      - term GPA <= max_gpa  OR
      - fail percentage >= fail_pct_threshold
    where "fail" = numeric grade < pass_cutoff.
    """
    st.subheader("Student Academic Standing → Probation")
    st.caption(
        f"Criteria: GPA ≤ {max_gpa} or Fail % ≥ {fail_pct_threshold}% (fail < {pass_cutoff})."
    )

    if df.empty:
        _empty_state("No rows for the current filters.")

    d = df.copy()
    gnum = pd.to_numeric(d["grade"], errors="coerce")
    d["is_fail"] = (gnum < float(pass_cutoff)).fillna(False)  # NaN not counted as fail

    per = (
        d.groupby(["student_no", "student_name", "term_label"], as_index=False)
        .agg(
            GPA=("grade", "mean"),           # mean ignores NaN
            Subjects=("grade", "size"),      # count of rows
            Fails=("is_fail", "sum"),
        )
    )
    per["fail_pct"] = (per["Fails"] / per["Subjects"] * 100).round(2)

    res = per[(per["GPA"] <= float(max_gpa)) | (per["fail_pct"] >= float(fail_pct_threshold))].copy()
    if res.empty:
        st.info("No students matched the probation threshold.")
        return

    res["GPA"] = res["GPA"].round(2)
    res = res.sort_values(
        by=["term_label", "GPA", "fail_pct"],
        ascending=[True, True, False],
        key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
    )

    st.dataframe(
        res.rename(columns={"Subjects": "Subjects"}),
        use_container_width=True
    )


def render_subject_pass_fail(df: pd.DataFrame, pass_cutoff: float = 75.0):
    st.subheader("Subject Pass/Fail Distribution")
    if df.empty:
        _empty_state("No rows for the current filters.")
    d = df.copy()
    d["pass_flag"] = (pd.to_numeric(d["grade"], errors="coerce") >= pass_cutoff).fillna(False)
    agg = (
        d.groupby(["term_label", "subject_code", "subject_title"], as_index=False)
        .agg(pass_count=("pass_flag", "sum"), total=("pass_flag", "size"))
    )
    agg["fail_count"] = agg["total"] - agg["pass_count"]
    agg["pass_%"] = (agg["pass_count"] / agg["total"] * 100).round(1)
    agg["fail_%"] = (agg["fail_count"] / agg["total"] * 100).round(1)
    agg = agg.sort_values(by="term_label", key=_sort_key_for_series_of_term_labels)
    show = agg[
        ["subject_code", "subject_title", "term_label", "pass_count", "fail_count", "pass_%", "fail_%"]
    ].rename(columns={"subject_title": "Subject Name", "term_label": "Semester", "pass_%": "Pass %", "fail_%": "Fail %"})
    st.dataframe(show, use_container_width=True)


def render_enrollment_analysis(df: pd.DataFrame):
    """Enrollment Trend Analysis (term-over-term).

    Columns:
      - Semester: term label (e.g., '2023-2024 S1')
      - Total Enrollment: unique students in the term
      - New Enrollees: students whose first-ever term is this term
      - Dropouts: students in previous term who are not in this term
      - Retention Rate (%): retained_from_prev / prev_total * 100
    """
    st.subheader("Enrollment Trend Analysis")
    if df.empty:
        _empty_state("No rows for the current filters.")

    # unique (student, term) pairs
    stu_term = (
        df[["student_no", "term_label"]]
        .dropna()
        .drop_duplicates()
    )
    if stu_term.empty:
        st.info("No student/term combinations found.")
        return

    # sort terms using the helpers already in this page
    terms = sorted(
        stu_term["term_label"].unique().tolist(),
        key=lambda x: (_parse_term_label(x)[0], _parse_term_label(x)[1] or 0),
    )
    order_map = {t: i for i, t in enumerate(terms)}

    # first appearance (term) for each student
    tmp = stu_term.assign(order=stu_term["term_label"].map(order_map))
    first_by_student = (
        tmp.sort_values("order")
        .groupby("student_no", as_index=False)
        .first()[["student_no", "term_label"]]
    )
    first_term_map = dict(zip(first_by_student["student_no"], first_by_student["term_label"]))

    rows = []
    prev_set: set[str] = set()

    for idx, term in enumerate(terms):
        cur_set = set(stu_term.loc[stu_term["term_label"] == term, "student_no"])
        total = len(cur_set)

        # new = first-ever term equals current term
        new_count = sum(1 for s in cur_set if first_term_map.get(s) == term)

        # dropouts = previous term students who didn't show up now
        dropouts = len(prev_set - cur_set) if idx > 0 else 0

        # retention = stayed from previous term / previous term total
        if idx > 0 and len(prev_set) > 0:
            retained = len(prev_set & cur_set)
            retention = retained / len(prev_set) * 100.0
        else:
            retention = None  # no prior term to compare

        rows.append(
            {
                "Semester": term,
                "Total Enrollment": total,
                "New Enrollees": new_count,
                "Dropouts": dropouts,
                "Retention Rate (%)": (round(retention, 2) if retention is not None else None),
            }
        )

        prev_set = cur_set

    out = pd.DataFrame(rows)

    # pretty % display (first term -> "—")
    out["Retention Rate (%)"] = out["Retention Rate (%)"].map(
        lambda v: (f"{v:.0f}%" if pd.notna(v) else "—")
    )

    st.dataframe(out, use_container_width=True)


# (Your updated incomplete grades logic kept as before)
def render_incomplete_grades(df: pd.DataFrame):
    st.subheader("Incomplete Grades Report")

    if df.empty:
        _empty_state("No rows for the current filters.")

    INC_STAT = {"INC", "INCOMPLETE", "INCOMP"}
    DRP_STAT = {"DROP", "DROPPED", "DRP", "WITHDRAWN", "WD", "W"}

    remark = df.get("remark", pd.Series(dtype=object)).astype(str).str.upper()
    raw = df.get("raw_grade", pd.Series(dtype=object)).astype(str).str.upper()

    mask_inc = remark.isin(INC_STAT) | raw.isin(INC_STAT)
    mask_drp = remark.isin(DRP_STAT) | raw.isin(DRP_STAT)
    incomplete = df[mask_inc | mask_drp].copy()

    # light backfill for display
    incomplete["student_name"] = incomplete["student_name"].replace({None: "", "": np.nan}).fillna(incomplete["student_no"])
    incomplete["teacher_name"] = incomplete["teacher_name"].replace({None: "", "": np.nan}).fillna("None")

    if incomplete.empty:
        st.success("No INC or DROPPED grades in the current scope.")
        return

    show = incomplete[[
        "term_label","school_year","semester",
        "subject_code","subject_title",
        "student_no","student_name",
        "raw_grade","remark",
        "teacher_name","section","department"
    ]].rename(columns={
        "raw_grade": "grade_text"
    }).sort_values(
        by=["term_label","subject_code","student_name"],
        key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s
    )

    st.dataframe(show, use_container_width=True, height=min(600, 38 + 28 * len(show)))


def render_retention_dropout(df: pd.DataFrame):
    st.subheader("Retention and Dropout Rates (approx.)")
    if df.empty:
        _empty_state("No rows for the current filters.")
    terms = (
        df["term_label"]
        .dropna()
        .unique()
        .tolist()
    )
    terms = sorted(terms, key=lambda x: (_parse_term_label(x)[0], _parse_term_label(x)[1] or 0))
    data = []
    for i, t in enumerate(terms):
        cur = set(df.loc[df["term_label"] == t, "student_no"].dropna().unique().tolist())
        nxt = set()
        if i + 1 < len(terms):
            nxt = set(df.loc[df["term_label"] == terms[i + 1], "student_no"].dropna().unique().tolist())
        if cur:
            retain = len(cur & nxt) / len(cur) * 100.0
        else:
            retain = np.nan
        data.append({"term_label": t, "students": len(cur), "retention_to_next_term_%": round(retain, 2)})
    st.dataframe(pd.DataFrame(data), use_container_width=True)


def render_top_performers_per_program(df: pd.DataFrame, topn: int = 10):
    st.subheader("Top Performers per Program")
    if df.empty:
        _empty_state("No rows for the current filters.")
    per = (
        df.groupby(["program", "student_no", "student_name"], as_index=False)
        .agg(GPA=("grade", "mean"), Subjects=("grade", "size"))
        .sort_values(["program", "GPA"], ascending=[True, False])
    )
    per["GPA"] = per["GPA"].round(2)
    out = (
        per.groupby("program", as_index=False)
        .head(topn)
        .reset_index(drop=True)
    )
    st.dataframe(out, use_container_width=True)


# -----------------------------
# Failed Students (by Subject) — unchanged
# -----------------------------
def render_failed_students_by_subject(df: pd.DataFrame, passing: float = 75.0) -> None:
    st.subheader("Failed Students (by Subject)")
    st.caption("Includes teacher, section, and term info based on current filters.")
    if df.empty:
        _empty_state("No rows for the current filters.")

    d = df.copy()
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    fails = d[d["grade"] < float(passing)].copy()

    # backfill student/teacher names for display
    fails["student_name"] = fails["student_name"].replace({None: "", "": np.nan}).fillna(fails["student_no"])
    fails["teacher_name"] = fails["teacher_name"].replace({None: "", "": np.nan}).fillna("None")

    if fails.empty:
        st.success("No failing records in the current scope.")
        return

    cols = [
        "term_label",
        "school_year",
        "semester",
        "subject_code",
        "subject_title",
        "student_no",
        "student_name",
        "grade",
        "teacher_name",
        "section",
        "department",
    ]
    for c in cols:
        if c not in fails.columns:
            fails[c] = np.nan

    fails = fails[cols].sort_values(
        by=["term_label", "subject_code", "student_name"],
        key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
    )
    st.dataframe(fails, use_container_width=True, height=min(600, 38 + 28 * len(fails)))


# -----------------------------
# Curriculum Progress & Advising — unchanged
# -----------------------------
def _load_program_codes_from_curricula() -> List[str]:
    course_codes = set()
    for cname in ["curriculum", "curricula", "curriculums"]:
        try:
            for doc in col(cname).find({}, {"courseCode": 1, "_id": 0}):
                cc = _safe_str(doc.get("courseCode"))
                if cc:
                    course_codes.add(cc)
        except Exception:
            pass
    if course_codes:
        return sorted(course_codes)

    # fallback: distinct program.program_code in enrollments
    try:
        codes = col("enrollments").distinct("program.program_code")
        return sorted([_safe_str(x) for x in codes if _safe_str(x)])
    except Exception:
        return []


def _find_curriculum_doc(course_code: str) -> Optional[dict]:
    for cname in ["curriculum", "curricula", "curriculums"]:
        try:
            doc = col(cname).find_one({"courseCode": course_code})
            if doc:
                return doc
        except Exception:
            continue
    return None


def _load_student_master(email_or_id: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (student_no, name, email) by checking 'students' then enrollments.
    """
    key = _safe_str(email_or_id).strip()
    if not key:
        return None, None, None

    # try 'students' collection
    try:
        q = {"$or": [{"student_no": key}, {"email": key}]}
        sdoc = col("students").find_one(q)
        if sdoc:
            return _safe_str(sdoc.get("student_no")), _safe_str(sdoc.get("name")), _safe_str(sdoc.get("email"))
    except Exception:
        pass

    # fallback: look up by enrollments
    try:
        doc = col("enrollments").find_one(
            {"$or": [{"student.student_no": key}, {"student.email": key}]},
            {"student.student_no": 1, "student.name": 1, "student.email": 1, "_id": 0},
        )
        if doc:
            s = doc.get("student", {})
            return _safe_str(s.get("student_no")), _safe_str(s.get("name")), _safe_str(s.get("email"))
    except Exception:
        pass

    return None, None, None


def _load_all_enrollments_for_student(student_no: Optional[str], email: Optional[str]) -> pd.DataFrame:
    filt = {}
    if student_no:
        filt = {"student.student_no": student_no}
    elif email:
        filt = {"student.email": email}
    else:
        return pd.DataFrame()

    fields = {
        "_id": 0,
        "student.student_no": 1,
        "student.name": 1,
        "student.email": 1,
        "program.program_code": 1,
        "program.program_name": 1,
        "subject.code": 1,
        "subject.title": 1,
        "grade": 1,
        "remark": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "section": 1,
        "teacher.name": 1,
    }
    rows = list(col("enrollments").find(filt, fields))
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows).rename(
        columns={
            "student.student_no": "student_no",
            "student.name": "student_name",
            "student.email": "student_email",
            "program.program_code": "program_code",
            "program.program_name": "program_name",
            "subject.code": "subject_code",
            "subject.title": "subject_title",
            "term.school_year": "school_year",
            "term.semester": "semester",
            "teacher.name": "teacher_name",
        }
    )
    df["term_label"] = df.apply(lambda r: _term_label(r.get("school_year"), r.get("semester")), axis=1)
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    return df


def render_curriculum_progress_advising() -> None:
    st.subheader("Curriculum Progress & Advising")

    # UI
    course_codes = _load_program_codes_from_curricula()
    c1, c2 = st.columns([2, 2])
    with c1:
        sel_course = st.selectbox("Select a Course Code:", options=course_codes or ["(none found)"])
    with c2:
        student_key = st.text_input("Enter Student ID or Email:", value="")
    load_btn = st.button("Load Student Curriculum")

    if not load_btn:
        st.info("Pick a course and type a student ID/email, then click **Load Student Curriculum**.")
        return

    if not student_key.strip():
        st.warning("Please enter a student number or email.")
        return

    # Resolve student master and fetch enrollments across all terms
    student_no, student_name, student_email = _load_student_master(student_key)
    if not (student_no or student_email):
        st.error("Student not found in 'students' or 'enrollments'.")
        return

    stud_df = _load_all_enrollments_for_student(student_no, student_email)
    curri = _find_curriculum_doc(sel_course)

    # Student Info
    with st.container():
        st.markdown(
            f"""
            **Student Information**  
            • **Name:** {student_name or '—'}  
            • **Student No.:** {student_no or '—'}  
            • **Email:** {student_email or '—'}  
            • **Course:** {sel_course}
            """
        )

    # If no curriculum doc, show what the student has taken
    if not curri:
        st.warning("No curriculum document found for this course code. Showing taken subjects only.")
        if stud_df.empty:
            st.info("No enrollments for this student.")
            return

        # Make sure columns exist even if absent in DB
        must_cols = ["term_label", "subject_code", "subject_title", "grade", "remark", "teacher_name", "section"]
        for c in must_cols:
            if c not in stud_df.columns:
                stud_df[c] = np.nan

        st.dataframe(
            stud_df[must_cols].rename(
                columns={
                    "subject_code": "Code",
                    "subject_title": "Description",
                    "grade": "Grade",
                    "remark": "Remark",
                    "teacher_name": "Teacher",
                    "section": "Section",
                    "term_label": "Term",
                }
            ).sort_values(
                by=["Term", "Code"],
                key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "Term" else s,
            ),
            use_container_width=True,
            height=min(640, 38 + 28 * len(stud_df)),
        )
        return

    # Build curriculum DataFrame
    subs = curri.get("subjects", []) or []
    cur_df = pd.DataFrame(subs)
    if cur_df.empty:
        st.warning("Curriculum has no 'subjects' array.")
        return

    # Normalize column names
    ren = {
        "yearLevel": "yearLevel",
        "semester": "semester",
        "subjectCode": "subjectCode",
        "subjectName": "subjectName",
        "lec": "lec",
        "lab": "lab",
        "units": "units",
        "prerequisites": "prerequisites",
    }
    cur_df = cur_df.rename(columns=ren)
    for need in ["yearLevel", "semester", "subjectCode", "subjectName", "units", "prerequisites"]:
        if need not in cur_df.columns:
            cur_df[need] = np.nan

    # Ensure the student enrollments have all columns we want to show
    needed_from_enr = ["subject_code", "grade", "remark", "term_label", "teacher_name", "section"]
    for c in needed_from_enr:
        if c not in stud_df.columns:
            stud_df[c] = np.nan

    take = stud_df[needed_from_enr].rename(columns={"subject_code": "subjectCode"})
    merged = cur_df.merge(take, on="subjectCode", how="left")

    # Status
    def _status(row):
        g = row.get("grade")
        r = _safe_str(row.get("remark")).upper()
        if pd.notna(g):
            try:
                g = float(g)
                return "PASSED" if g >= 75 else "FAILED"
            except Exception:
                pass
        if r in {"INC", "INCOMPLETE", "INCOMP"}:
            return "INCOMPLETE"
        if pd.isna(g) and not r:
            return "NOT TAKEN"
        return r or "NOT TAKEN"

    merged["Status"] = merged.apply(_status, axis=1)

    # Pretty prerequisites
    def _prettify_prereq(v):
        if isinstance(v, (list, tuple)):
            return ", ".join([_safe_str(x) for x in v if _safe_str(x)])
        return _safe_str(v)

    merged["prerequisites"] = merged["prerequisites"].map(_prettify_prereq)

    # Display by Year → Semester
    for yr in sorted(merged["yearLevel"].dropna().astype(int).unique().tolist()):
        st.markdown(f"#### Year {yr}")
        for sem in sorted(merged.loc[merged["yearLevel"] == yr, "semester"].dropna().astype(int).unique().tolist()):
            st.markdown(f"**Semester {sem}**")
            view = merged[(merged["yearLevel"] == yr) & (merged["semester"] == sem)].copy()
            view = view[
                [
                    "subjectCode",
                    "subjectName",
                    "units",
                    "grade",
                    "Status",
                    "term_label",
                    "teacher_name",
                    "section",
                    "prerequisites",
                ]
            ].rename(
                columns={
                    "subjectCode": "Code",
                    "subjectName": "Description",
                    "units": "Units",
                    "grade": "Grade",
                    "term_label": "Term",
                    "teacher_name": "Teacher",
                    "section": "Section",
                    "prerequisites": "Prerequisites",
                }
            )
            st.dataframe(
                view,
                use_container_width=True,
                height=min(420, 38 + 28 * len(view)),
            )


# -----------------------------
# Page UI
# -----------------------------

def main():
    st.title("Registrar Dashboard")
    # right under st.title(...)
    try:
        u = user  # you already set: user = require_role("registrar", "admin")
    except NameError:
        u = current_user()
    _user_header(u)


    # --- Filters ---
    with st.expander("Filters", expanded=True):
        all_terms = load_distinct_terms()
        sel_terms = st.multiselect("Term(s)", options=all_terms, default=tuple(all_terms))

        c1, c2 = st.columns(2)
        with c1:
            subj_regex = st.text_input("Subject code (exact or regex)", value="")
        with c2:
            dept_regex = st.text_input("Department (exact or regex)", value="")

        c3, c4 = st.columns(2)
        with c3:
            deans_min = st.number_input("Dean's List minimum GPA", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        with c4:
            probation_max = st.number_input("Probation maximum GPA", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            # NEW: adjustable fail-rate (%) threshold for probation
            probation_fail_pct = st.number_input(
                "Probation minimum fail rate (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0
            )

    # --- Academic standing switches ---
    st.markdown("### Student Academic Standing Report")
    col_cb1, col_cb2, col_cb3 = st.columns([1, 1, 1])
    with col_cb1:
        show_gpa = st.checkbox("GPA Reports", value=False)
    with col_cb2:
        show_deans = st.checkbox("Dean's List", value=False)
    with col_cb3:
        show_prob = st.checkbox("Probation", value=False)

    gen1 = st.button("Apply filters & generate", type="primary")

    # --- Other Reports ---
    st.markdown("### Other Reports / Views")
    with st.expander("Choose one or more reports", expanded=False):
        r1 = st.checkbox("Subject Pass/Fail Distribution", value=False)
        r2 = st.checkbox("Enrollment Analysis", value=False)
        r3 = st.checkbox("Incomplete Grades Report", value=False)
        r4 = st.checkbox("Retention and Dropout Rates", value=False)
        r5 = st.checkbox("Top Performers per Program", value=False)
        r6 = st.checkbox("Failed Students (by Subject)", value=False)
        r7 = st.checkbox("Curriculum Progress & Advising", value=False)
        gen2 = st.button("Generate selected report(s)")

    # Load gating (unchanged)
    if not (gen1 or gen2 or r7):
        st.info("Select the reports you want, then click **Apply** / **Generate**.")
        return

    # Load enrollments only when needed
    if gen1 or gen2:
        df = load_enrollments_df(tuple(sel_terms), subj_regex, dept_regex)
    else:
        df = None  # r7-only path

    # --- Academic standing ---
    if gen1 and df is not None:
        if show_gpa:
            render_gpa_reports(df)
        if show_deans:
            render_deans_list(df, float(deans_min))
        if show_prob:
            # pass the adjustable fail-rate (%) threshold
            render_probation(df, float(probation_max), float(probation_fail_pct))

    # --- Other reports ---
    if gen2 and df is not None:
        if r1:
            render_subject_pass_fail(df)
        if r2:
            render_enrollment_analysis(df)
        if r3:
            render_incomplete_grades(df)
        if r4:
            render_retention_dropout(df)
        if r5:
            render_top_performers_per_program(df, topn=10)
        if r6:
            render_failed_students_by_subject(df)
        if r7:
            render_curriculum_progress_advising()

    # Also render the curriculum tool whenever its toggle is on,
    # even if "Generate" wasn't pressed.
    if r7 and not gen2:
        render_curriculum_progress_advising()


if __name__ == "__main__":
    main()
