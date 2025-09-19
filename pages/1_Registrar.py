# pages/1_Registrar.py
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from db import col  # uses your existing db.py helper

from utils.auth import require_role
user = require_role("registrar", "admin")   # only registrar/admin may open
# -----------------------------
# Small helpers (robust parsing)
# -----------------------------
from utils.auth import require_role, render_logout_sidebar  # add this import

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
        "subject.department": 1,
        "grade": 1,                # original grade text/number
        "remark": 1,               # INC/DROPPED/etc
        "teacher.name": 1,
        "teacher.email": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "section": 1,
    }

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

    # Display program
    df["program"] = df.apply(lambda r: r.get("program_name") or r.get("program_code") or "(Unknown)", axis=1)

    # Term label
    df["term_label"] = df.apply(lambda r: _term_label(r.get("school_year"), r.get("semester")), axis=1)

    # Keep original grade text and also a numeric version
    df["raw_grade"] = df.get("grade")
    df["grade"] = pd.to_numeric(df["raw_grade"], errors="coerce")  # numeric, may be NaN for INC/etc
    # IMPORTANT: do NOT drop NaNs — we need them for INC/DROPPED reports

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


def render_probation(
    df: pd.DataFrame,
    max_gpa: float,
    fail_pct_threshold: float,
    pass_cutoff: float = 75.0,
) -> None:
    """Students on probation if GPA <= max_gpa OR fail% >= fail_pct_threshold."""
    st.subheader("Student Academic Standing → Probation")
    st.caption(f"Thresholds: GPA ≤ {max_gpa}  •  Fail % ≥ {fail_pct_threshold}  (Fail = grade < {pass_cutoff})")

    if df.empty:
        _empty_state("No rows for the current filters.")

    d = df.copy()
    d["grade_num"] = pd.to_numeric(d.get("grade"), errors="coerce")
    d = d[~d["grade_num"].isna()]  # only numeric grades contribute to GPA/fail%

    if d.empty:
        st.info("No numeric grades to compute GPA / fail rates.")
        return

    # mark fails
    d["is_fail"] = d["grade_num"] < float(pass_cutoff)

    # per-student per-term stats
    agg = (
        d.groupby(["student_no", "student_name", "term_label"], as_index=False)
         .agg(
             GPA=("grade_num", "mean"),
             Subjects=("grade_num", "size"),
             Fails=("is_fail", "sum"),
         )
    )
    agg["Fail %"] = (agg["Fails"] / agg["Subjects"] * 100).round(2)

    # probation rule: GPA low OR fail% high
    prob = agg[(agg["GPA"] <= float(max_gpa)) | (agg["Fail %"] >= float(fail_pct_threshold))].copy()

    if prob.empty:
        st.info("No students matched the probation thresholds.")
        return

    prob["GPA"] = prob["GPA"].round(2)

    prob = prob.sort_values(
        by=["term_label", "Fail %", "GPA"],
        ascending=[True, False, True],
        key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
    )

    st.dataframe(prob, use_container_width=True)


def render_incomplete_grades(df: pd.DataFrame):
    """Incomplete / Dropped grades with student & teacher names back-filled."""
    st.subheader("Incomplete Grades Report")

    if df.empty:
        _empty_state("No rows for the current filters.")

    # Treat these as incomplete/dropped statuses
    INC_STAT = {"INC", "INCOMPLETE", "INCOMP"}
    DRP_STAT = {"DROP", "DROPPED", "DRP", "WITHDRAWN", "WD", "W"}

    # Ensure helper for blank checks
    def _is_blank(series: pd.Series) -> pd.Series:
        return series.isna() | (series.astype(str).str.strip() == "")

    # Normalize sources
    remark = df.get("remark", pd.Series(dtype=object)).astype(str).str.upper()
    raw    = df.get("raw_grade", pd.Series(dtype=object)).astype(str).str.upper()

    mask_inc = remark.isin(INC_STAT) | raw.isin(INC_STAT)
    mask_drp = remark.isin(DRP_STAT) | raw.isin(DRP_STAT)
    incomplete = df[mask_inc | mask_drp].copy()

    if incomplete.empty:
        st.success("No INC or DROPPED grades in the current scope.")
        return

    # ---------------------------
    # Back-fill missing student_name
    # ---------------------------
    if {"student_no", "student_name"}.issubset(df.columns):
        map_names = (
            df[["student_no", "student_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("student_no")["student_name"]
        )
        incomplete["student_name"] = incomplete["student_name"].fillna(
            incomplete["student_no"].map(map_names)
        )

    need_students = (
        incomplete.loc[_is_blank(incomplete["student_name"]), "student_no"]
        .dropna().unique().tolist()
    )
    if need_students:
        lookup = {}
        try:
            cur = col("enrollments").find(
                {"student.student_no": {"$in": need_students}},
                {"student.student_no": 1, "student.name": 1, "_id": 0},
            )
            for r in cur:
                s = r.get("student", {})
                sno, nm = s.get("student_no"), s.get("name")
                if sno and nm and sno not in lookup:
                    lookup[sno] = nm
        except Exception:
            lookup = {}
        if lookup:
            incomplete["student_name"] = incomplete["student_name"].fillna(
                incomplete["student_no"].map(lookup)
            )

    # ---------------------------
    # Back-fill missing teacher_name
    # ---------------------------
    # Step 1: map teacher_email -> teacher_name using current DF
    if {"teacher_email", "teacher_name"}.issubset(df.columns):
        email_to_name = (
            df.loc[~_is_blank(df["teacher_email"]) & ~_is_blank(df["teacher_name"]),
                   ["teacher_email", "teacher_name"]]
            .drop_duplicates()
            .set_index("teacher_email")["teacher_name"]
        )
        mask_missing_tname = _is_blank(incomplete["teacher_name"])
        if not email_to_name.empty and "teacher_email" in incomplete.columns:
            fills = incomplete.loc[mask_missing_tname, "teacher_email"].map(email_to_name)
            incomplete.loc[mask_missing_tname, "teacher_name"] = incomplete.loc[mask_missing_tname, "teacher_name"].fillna(fills)

    # Step 2: most common teacher per (term_label, subject_code, section) in current DF
    for c in ["term_label", "subject_code", "section", "teacher_name"]:
        if c not in df.columns:
            df[c] = np.nan
    known = df.loc[~_is_blank(df["teacher_name"])].copy()
    if not known.empty:
        try:
            key_cols = ["term_label", "subject_code", "section"]
            common_map = (
                known.groupby(key_cols)["teacher_name"]
                .agg(lambda s: s.value_counts().index[0])
            )
            mask_missing_tname = _is_blank(incomplete["teacher_name"])
            if mask_missing_tname.any():
                key_df = incomplete.loc[mask_missing_tname, key_cols]
                tuple_keys = list(map(tuple, key_df.values))
                mapped = pd.Series(tuple_keys).map(common_map.to_dict())
                incomplete.loc[mask_missing_tname, "teacher_name"] = incomplete.loc[mask_missing_tname, "teacher_name"].fillna(mapped.values)
        except Exception:
            pass

    # Step 3: one-shot DB lookup by missing teacher emails
    mask_missing_tname = _is_blank(incomplete["teacher_name"])
    if mask_missing_tname.any() and "teacher_email" in incomplete.columns:
        missing_emails = (
            incomplete.loc[mask_missing_tname, "teacher_email"]
            .dropna().astype(str).str.strip().unique().tolist()
        )
        if missing_emails:
            email_lookup = {}
            try:
                cur = col("enrollments").find(
                    {"teacher.email": {"$in": missing_emails}},
                    {"teacher.email": 1, "teacher.name": 1, "_id": 0},
                )
                for r in cur:
                    t = r.get("teacher", {})
                    em, nm = t.get("email"), t.get("name")
                    if em and nm and em not in email_lookup:
                        email_lookup[em] = nm
            except Exception:
                email_lookup = {}
            if email_lookup:
                fills = incomplete.loc[mask_missing_tname, "teacher_email"].map(email_lookup)
                incomplete.loc[mask_missing_tname, "teacher_name"] = incomplete.loc[mask_missing_tname, "teacher_name"].fillna(fills)

    # ---------------------------
    # Display
    # ---------------------------
    cols = [
        "term_label","school_year","semester",
        "subject_code","subject_title",
        "student_no","student_name",
        "raw_grade","remark",
        "teacher_name","section","department"
    ]
    for c in cols:
        if c not in incomplete.columns:
            incomplete[c] = np.nan

    show = (
        incomplete[cols]
        .rename(columns={"raw_grade": "grade_text"})
        .sort_values(
            by=["term_label","subject_code","student_name"],
            key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "term_label" else s,
        )
    )

    st.caption("Rows where grade/remark indicates INC or DROPPED. Names back-filled when possible.")
    st.dataframe(show, use_container_width=True, height=min(600, 38 + 28 * len(show)))


def render_enrollment_analysis(df: pd.DataFrame):
    st.subheader("Enrollment Analysis (rows per term & subject)")
    if df.empty:
        _empty_state("No rows for the current filters.")
    g = (
        df.groupby(["term_label", "subject_code"], as_index=False)
        .size()
        .sort_values(by="term_label", key=_sort_key_for_series_of_term_labels)
    )
    st.dataframe(g, use_container_width=True)


def render_retention_dropout(df: pd.DataFrame):
    st.subheader("Retention and Dropout Rates (approx.)")
    if df.empty:
        _empty_state("No rows for the current filters.")
    # Very rough proxy: count unique students per term, compare forward term participation
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
# NEW: Failed Students (by Subject)
# -----------------------------

def render_failed_students_by_subject(df: pd.DataFrame, passing: float = 75.0) -> None:
    """Show students who failed per subject. Backfills missing student & teacher names."""
    st.subheader("Failed Students (by Subject)")
    st.caption("Includes teacher, section, and term info based on current filters.")
    if df.empty:
        _empty_state("No rows for the current filters.")

    d = df.copy()
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    fails = d[d["grade"] < float(passing)].copy()

    if fails.empty:
        st.success("No failing records in the current scope.")
        return

    # ---------------------------
    # Back-fill missing student_name
    # ---------------------------
    if {"student_no", "student_name"}.issubset(d.columns):
        names_from_df = (
            d[["student_no", "student_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("student_no")["student_name"]
        )
        fails["student_name"] = fails["student_name"].fillna(
            fails["student_no"].map(names_from_df)
        )

    still_missing_students = (
        fails.loc[fails["student_name"].isna(), "student_no"]
        .dropna()
        .unique()
        .tolist()
    )
    if still_missing_students:
        lookup = {}
        try:
            cur = col("enrollments").find(
                {"student.student_no": {"$in": still_missing_students}},
                {"student.student_no": 1, "student.name": 1, "_id": 0},
            )
            for r in cur:
                s = r.get("student", {})
                sno = s.get("student_no")
                nm = s.get("name")
                if sno and nm and sno not in lookup:
                    lookup[sno] = nm
        except Exception:
            lookup = {}

        if lookup:
            fails["student_name"] = fails["student_name"].fillna(
                fails["student_no"].map(lookup)
            )

    # ---------------------------
    # Back-fill missing teacher_name
    # ---------------------------
    # Normalize missing teacher_name check
    def _is_blank(series: pd.Series) -> pd.Series:
        return series.isna() | (series.astype(str).str.strip() == "")

    # Step 1: use mapping from teacher_email -> teacher_name within current df
    if {"teacher_email", "teacher_name"}.issubset(d.columns):
        email_to_name = (
            d.loc[~_is_blank(d["teacher_email"]) & ~_is_blank(d["teacher_name"]),
                  ["teacher_email", "teacher_name"]]
            .drop_duplicates()
            .set_index("teacher_email")["teacher_name"]
        )
        mask_missing_tname = _is_blank(fails["teacher_name"])
        if not email_to_name.empty and "teacher_email" in fails.columns:
            fills = fails.loc[mask_missing_tname, "teacher_email"].map(email_to_name)
            fails.loc[mask_missing_tname, "teacher_name"] = fails.loc[mask_missing_tname, "teacher_name"].fillna(fills)

    # Step 2: most common teacher per (term_label, subject_code, section) from current df
    for col_needed in ["term_label", "subject_code", "section", "teacher_name"]:
        if col_needed not in d.columns:
            d[col_needed] = np.nan
    known = d.loc[~_is_blank(d["teacher_name"])].copy()
    if not known.empty:
        try:
            # most frequent teacher per key
            key_cols = ["term_label", "subject_code", "section"]
            common_map = (
                known.groupby(key_cols)["teacher_name"]
                .agg(lambda s: s.value_counts().index[0])
            )
            mask_missing_tname = _is_blank(fails["teacher_name"])
            if mask_missing_tname.any():
                key_df = fails.loc[mask_missing_tname, key_cols]
                tuple_keys = list(map(tuple, key_df.values))
                map_vals = pd.Series(tuple_keys).map(common_map.to_dict())
                fails.loc[mask_missing_tname, "teacher_name"] = fails.loc[mask_missing_tname, "teacher_name"].fillna(map_vals.values)
        except Exception:
            pass

    # Step 3: single DB lookup by missing teacher emails
    mask_missing_tname = _is_blank(fails["teacher_name"])
    if mask_missing_tname.any() and "teacher_email" in fails.columns:
        missing_emails = (
            fails.loc[mask_missing_tname, "teacher_email"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        if missing_emails:
            email_lookup = {}
            try:
                cur = col("enrollments").find(
                    {"teacher.email": {"$in": missing_emails}},
                    {"teacher.email": 1, "teacher.name": 1, "_id": 0},
                )
                for r in cur:
                    t = r.get("teacher", {})
                    em = t.get("email")
                    nm = t.get("name")
                    if em and nm and em not in email_lookup:
                        email_lookup[em] = nm
            except Exception:
                email_lookup = {}

            if email_lookup:
                fills = fails.loc[mask_missing_tname, "teacher_email"].map(email_lookup)
                fails.loc[mask_missing_tname, "teacher_name"] = fails.loc[mask_missing_tname, "teacher_name"].fillna(fills)

    # ---------------------------
    # Display
    # ---------------------------
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
# NEW: Curriculum Progress & Advising
# -----------------------------

def _load_program_codes_from_curricula() -> List[str]:
    """
    Try several likely collection names to fetch curriculum course codes.
    Falls back to program codes from enrollments if none found.
    """
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

    # --- NEW: back-fill missing name / number from enrollments ---
    # Some databases don’t store name/number in the quick lookup we did above.
    # If we have enrollments, extract the first non-null values and show them.
    if not student_name and not stud_df.empty and "student_name" in stud_df.columns:
        s = stud_df["student_name"].dropna()
        if not s.empty:
            student_name = str(s.iloc[0])
    if not student_no and not stud_df.empty and "student_no" in stud_df.columns:
        s = stud_df["student_no"].dropna()
        if not s.empty:
            student_no = str(s.iloc[0])
    # -------------------------------------------------------------

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
        # NEW toggles appended (no changes to previous ones)
        r6 = st.checkbox("Failed Students (by Subject)", value=False)
        r7 = st.checkbox("Curriculum Progress & Advising", value=False)
        gen2 = st.button("Generate selected report(s)")

    # ---- IMPORTANT: gating fix ----
    # Allow the page to proceed when the curriculum tool is toggled (r7),
    # even if Apply/Generate wasn't clicked. The curriculum tool loads on demand.
    if not (gen1 or gen2 or r7):
        st.info("Select the reports you want, then click **Apply** / **Generate**.")
        return

    # Load enrollments only when needed (curriculum tool doesn't need this df)
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
            render_probation(df, float(probation_max))

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
    # even if "Generate" wasn't pressed. It fetches data on its own when you click its button.
    if r7 and not gen2:
        render_curriculum_progress_advising()

def render_subject_pass_fail(df: pd.DataFrame) -> None:
    """Subject pass/fail distribution per subject and term (semester)."""
    st.subheader("Subject Pass/Fail Distribution")
    if df.empty:
        _empty_state("No rows for the current filters.")

    # Work on a copy; consider only numeric grades for pass/fail
    d = df.copy()
    d["grade_num"] = pd.to_numeric(d.get("grade"), errors="coerce")
    d = d[~d["grade_num"].isna()]  # ignore INC/Dropped/etc. for this view
    if d.empty:
        st.info("No numeric grades available for pass/fail distribution.")
        return

    # Pass if grade >= 75
    d["is_pass"] = d["grade_num"] >= 75

    # Aggregate by subject + term
    g = (
        d.groupby(["subject_code", "subject_title", "term_label"], as_index=False)
         .agg(
             **{
                 "Pass Count": ("is_pass", lambda s: int(s.sum())),
                 "Fail Count": ("is_pass", lambda s: int((~s).sum())),
             }
         )
    )
    g["Total"]   = g["Pass Count"] + g["Fail Count"]
    g["Pass %"]  = ((g["Pass Count"] / g["Total"]) * 100).round(0).fillna(0).astype(int)
    g["Fail %"]  = ((g["Fail Count"] / g["Total"]) * 100).round(0).fillna(0).astype(int)

    # Pretty rename/ordering to match your sample
    out = g.rename(
        columns={
            "subject_code": "Subject Code",
            "subject_title": "Subject Name",
            "term_label": "Semester",
        }
    )[
        ["Subject Code", "Subject Name", "Semester", "Pass Count", "Fail Count", "Pass %", "Fail %"]
    ]

    # Sort by semester then subject code (semester uses your term sort helper)
    out = out.sort_values(
        by=["Semester", "Subject Code"],
        key=lambda s: _sort_key_for_series_of_term_labels(s) if s.name == "Semester" else s,
    )

    # Render percent columns with % sign (purely cosmetic)
    out["Pass %"] = out["Pass %"].astype(str) + "%"
    out["Fail %"] = out["Fail %"].astype(str) + "%"

    st.dataframe(out, use_container_width=True)

if __name__ == "__main__":
    main()
