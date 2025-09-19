# pages/3_Student.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from db import col
from utils.auth import require_role, current_user


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def _nemail(e: str | None) -> str:
    return (e or "").strip().lower()


def _to_num_grade(x) -> float | None:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _term_label(sy: str | None, sem: int | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem or 0)
    except Exception:
        s = 0
    return f"{sy} S{s}" if s else sy


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


@st.cache_data(show_spinner=False)
def load_term_catalog() -> list[tuple[str, str, int]]:
    """
    Read the full school-year/semester catalog (including Summer S3) from `semesters`.
    Returns a sorted list of (label, school_year, semester).
    """
    rows = list(col("semesters").find({}, {"school_year": 1, "semester": 1}))
    seen = set()
    out: list[tuple[str, str, int]] = []
    for r in rows:
        sy = r.get("school_year")
        sem = r.get("semester")
        if sy and sem is not None:
            label = _term_label(sy, sem)
            key = (label, sy, int(sem))
            if key not in seen:
                seen.add(key)
                out.append(key)
    out.sort(key=lambda t: _term_sort_key(t[0]))
    return out


@st.cache_data(show_spinner=False)
def curriculum_units_map() -> Dict[str, float]:
    """
    Build a subjectCode -> units map from `curriculums` documents.
    """
    mapping: Dict[str, float] = {}
    cur_docs = list(col("curriculums").find({}, {"subjects": 1}))
    for doc in cur_docs:
        for s in doc.get("subjects", []) or []:
            code = s.get("subjectCode") or s.get("subject_code")
            if not code:
                continue
            units = s.get("units")
            if units is None:
                # sometimes lec/lab are present
                units = float((s.get("lec") or 0)) + float((s.get("lab") or 0))
            try:
                mapping[str(code)] = float(units or 0)
            except Exception:
                mapping[str(code)] = 0.0
    return mapping


def _subject_units(sub: dict, units_map: Dict[str, float]) -> float:
    # prefer units inside enrollment.subject
    if sub:
        if sub.get("units") is not None:
            try:
                return float(sub.get("units") or 0)
            except Exception:
                pass
        # fallback lec + lab
        try:
            v = float(sub.get("lec") or 0) + float(sub.get("lab") or 0)
            if v:
                return v
        except Exception:
            pass
        # fallback to curriculum map
        code = sub.get("code") or sub.get("subjectCode")
        if code and code in units_map:
            return float(units_map[code])
    return 0.0


@st.cache_data(show_spinner=False)
def load_student_enrollments(email: Optional[str] = None, student_no: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single student's enrollments. Either `email` or `student_no` must be provided.
    """
    if not email and not student_no:
        return pd.DataFrame(columns=[
            "term_label", "subject_code", "subject_title", "units", "grade", "remark",
            "section", "teacher_name", "program_code"
        ])

    q = {}
    if email:
        q["student.email"] = _nemail(email)
    if student_no:
        q["student.student_no"] = student_no

    proj = {
        "grade": 1,
        "remark": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "subject.code": 1,
        "subject.title": 1,
        "subject.units": 1,
        "subject.lec": 1,
        "subject.lab": 1,
        "teacher.name": 1,
        "program.program_code": 1,
        "section": 1,
    }
    rows = list(col("enrollments").find(q, proj))

    u_map = curriculum_units_map()

    def flatten(e: dict) -> dict:
        term = e.get("term") or {}
        sub = e.get("subject") or {}
        prog = e.get("program") or {}
        return {
            "term_label": _term_label(term.get("school_year"), term.get("semester")),
            "subject_code": sub.get("code"),
            "subject_title": sub.get("title"),
            "units": _subject_units(sub, u_map),
            "grade": _to_num_grade(e.get("grade")),
            "remark": e.get("remark"),
            "section": e.get("section"),
            "teacher_name": (e.get("teacher") or {}).get("name"),
            "program_code": prog.get("program_code"),
        }

    return pd.DataFrame([flatten(r) for r in rows])


# ──────────────────────────────────────────
# Page
# ──────────────────────────────────────────

def main():
    # student can view self; registrar/admin can search
    user = require_role("student", "registrar", "admin")
    role = (user.get("role") or "").lower()

    st.title("🧑‍🎓 Student Dashboard")

    df = pd.DataFrame()
    picked_email = None
    picked_no = None

    if role == "student":
        picked_email = _nemail(user.get("email"))
        st.caption(f"Signed in as **{picked_email or user.get('email','(no email)')}**. Showing your records.")
        df = load_student_enrollments(email=picked_email)
    else:
        st.caption("Registrar/Admin: search a student by email or ID.")
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            picked_email = _nemail(st.text_input("Student email (preferred)", ""))
        with c2:
            picked_no = st.text_input("Student No.", "")
        with c3:
            if st.button("Load", use_container_width=True):
                st.session_state["_student_query"] = {"email": picked_email, "no": picked_no}

        q = st.session_state.get("_student_query", {})
        if q:
            df = load_student_enrollments(email=q.get("email") or None, student_no=q.get("no") or None)

    if df.empty:
        st.warning("No enrollments found for this student.")
        return

    # Global picker from semesters (all terms, including Summer)
    term_catalog = load_term_catalog()
    all_term_labels = [lbl for (lbl, _, _) in term_catalog]

    # Filters (do not change plots — they are computed from the filtered df)
    fcols = st.columns(3)
    with fcols[0]:
        sel_terms = st.multiselect("Term(s)", options=all_term_labels, default=all_term_labels)
    with fcols[1]:
        section_opts = sorted([s for s in df["section"].dropna().unique().tolist() if s not in (None, "", "—")])
        sel_sections = st.multiselect("Section(s)", options=section_opts, default=section_opts)
    with fcols[2]:
        subj_opts = sorted(df["subject_code"].dropna().unique().tolist())
        sel_subjects = st.multiselect("Subject(s)", options=subj_opts, default=subj_opts)

    dff = df.copy()
    if sel_terms:
        dff = dff[dff["term_label"].isin(sel_terms)]
    if sel_sections:
        dff = dff[dff["section"].isin(sel_sections)]
    if sel_subjects:
        dff = dff[dff["subject_code"].isin(sel_subjects)]

    # ── 1) Academic Transcript Viewer
    st.subheader("1) Academic Transcript Viewer  ↪")

    if dff.empty:
        st.info("No rows after filters.")
    else:
        for term, block in dff.sort_values("term_label", key=lambda s: s.map(_term_sort_key)).groupby("term_label"):
            # GPA for the term (weighted by units when available)
            graded = block.dropna(subset=["grade"])
            if not graded.empty:
                w = graded["units"].replace({np.nan: 0.0}).astype(float)
                g = graded["grade"].astype(float)
                denom = (w.where(w > 0, 1.0))  # avoid division by zero if units missing
                term_gpa = (g * denom).sum() / denom.sum()
                gpa_txt = f"{term_gpa:.2f}"
            else:
                gpa_txt = "—"

            units_sum = float(block["units"].fillna(0).sum())
            st.markdown(f"**{term} · GPA:** {gpa_txt} · **Units:** {units_sum:g}")

            show = (
                block[["subject_code", "subject_title", "units", "grade", "remark"]]
                .rename(columns={
                    "subject_code": "Subject",
                    "subject_title": "Description",
                    "units": "units",
                    "grade": "grade",
                    "remark": "remark",
                })
            )
            # make sure units display as ints if whole numbers
            show["units"] = show["units"].map(lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x)
            st.dataframe(show, use_container_width=True)

    # ── 2) Performance Trend Over Time (GPA by term)
    st.subheader("2) Performance Trend Over Time (Avg by Term)")
    graded_all = dff.dropna(subset=["grade"])
    if graded_all.empty:
        st.info("No graded data after filters.")
    else:
        g = (
            graded_all.groupby("term_label", as_index=False)["grade"]
            .mean()
            .rename(columns={"grade": "avg_grade"})
            .sort_values("term_label", key=lambda s: s.map(_term_sort_key))
            .set_index("term_label")
        )
        st.line_chart(g)

    # ── 3) Passed vs Failed Summary
    st.subheader("3) Passed vs Failed Summary")
    if graded_all.empty:
        st.info("No graded data to summarize.")
    else:
        pv = pd.Series(
            {
                "Passed": (graded_all["grade"] >= 75).sum(),
                "Failed": (graded_all["grade"] < 75).sum(),
                "Incomplete": (dff["remark"].fillna("").str.upper() == "INCOMPLETE").sum(),
                "Dropped": (dff["remark"].fillna("").str.upper() == "DROPPED").sum(),
            }
        )
        st.bar_chart(pv.to_frame("count"))  # simple, readable

    # ── 4) Subject Difficulty Ratings (fail % by subject for this student)
    st.subheader("4) Subject Difficulty Ratings (per-subject status)")
    if dff.empty:
        st.info("No data to compute subject statuses.")
    else:
        sstats = (
            dff.assign(
                status=np.select(
                    [
                        dff["remark"].fillna("").str.upper().eq("DROPPED"),
                        dff["remark"].fillna("").str.upper().eq("INCOMPLETE"),
                        dff["grade"].fillna(1000) < 75,
                        dff["grade"].notna() & (dff["grade"] >= 75),
                    ],
                    ["DROPPED", "INCOMPLETE", "FAILED", "PASSED"],
                    default="—",
                )
            )[["subject_code", "subject_title", "status"]]
            .rename(columns={"subject_code": "Subject", "subject_title": "Title", "status": "Status"})
        )
        st.dataframe(sstats, use_container_width=True)


if __name__ == "__main__":
    main()
