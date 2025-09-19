# pages/3_Student.py
from __future__ import annotations
import io
import math
from datetime import date
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st

from db import col
from utils.auth import get_current_user, require_role

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
from utils.auth import require_role, render_logout_sidebar  # add this import

PASSING_GRADE = 75  # align with your app-wide rule


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


def _to_num_grade(x) -> float | None:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _cumulative_gpa(df: pd.DataFrame) -> Optional[float]:
    """Units-weighted GPA across all rows with numeric grade + positive units."""
    if df.empty:
        return None
    g = pd.to_numeric(df["grade"], errors="coerce")
    u = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    mask = g.notna() & u.gt(0)
    if not mask.any():
        return None
    return round(float((g[mask] * u[mask]).sum() / u[mask].sum()), 2)


def _semester_gpa(block: pd.DataFrame) -> Optional[float]:
    if block.empty:
        return None
    g = pd.to_numeric(block["grade"], errors="coerce")
    u = pd.to_numeric(block["units"], errors="coerce").fillna(0)
    mask = g.notna() & u.gt(0)
    if not mask.any():
        return None
    return round(float((g[mask] * u[mask]).sum() / u[mask].sum()), 2)


def _grade_buckets(df: pd.DataFrame) -> Dict[str, int]:
    g = pd.to_numeric(df["grade"], errors="coerce")
    passed = int((g >= PASSING_GRADE).sum())
    failed = int((g.notna() & (g < PASSING_GRADE)).sum())
    incomplete = int(g.isna().sum())
    return {"Passed": passed, "Failed": failed, "In-Progress": incomplete}


# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _subject_units_map(codes: List[str]) -> Dict[str, int]:
    """
    Build {subject_code -> units} from the 'subjects' collection
    (fallback when enrollments.subject.units is missing).
    """
    if not codes:
        return {}
    q = {"_id": {"$in": list(set([c for c in codes if c]))}}
    proj = {"Units": 1}
    out: Dict[str, int] = {}
    for d in col("subjects").find(q, proj):
        code = d.get("_id")
        units = d.get("Units", 0)
        if code:
            try:
                out[code] = int(units or 0)
            except Exception:
                out[code] = 0
    return out


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_student_enrollments(
    student_email: Optional[str] = None,
    student_no: Optional[str] = None
) -> pd.DataFrame:
    """
    Pulls the student's enrollments and flattens to a DataFrame.
    Prefers email; falls back to student_no if provided.
    """
    q: Dict[str, Any] = {}
    if student_email:
        q = {"student.email": student_email.strip().lower()}
    elif student_no:
        q = {"student.student_no": student_no}

    proj = {
        "grade": 1,
        "remark": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "student.name": 1,
        "student.student_no": 1,
        "student.email": 1,
        "subject.code": 1,
        "subject.title": 1,
        "subject.units": 1,             # may be missing -> we'll fallback
        "program.program_code": 1,
        "offering.section": 1,          # preferred place for section
        "section": 1,                    # fallback if some docs keep it at root
    }

    rows = list(col("enrollments").find(q, proj))
    if not rows:
        return pd.DataFrame(columns=[
            "student_no", "student_email", "student_name",
            "subject_code", "subject_title", "program_code",
            "term_label", "grade", "remark", "units", "section"
        ])

    def flat(e):
        term = e.get("term") or {}
        stu = e.get("student") or {}
        sub = e.get("subject") or {}
        prog = e.get("program") or {}
        section = (e.get("offering") or {}).get("section") or e.get("section")
        return {
            "student_no": stu.get("student_no"),
            "student_email": (stu.get("email") or "").strip().lower(),
            "student_name": stu.get("name"),
            "subject_code": sub.get("code"),
            "subject_title": sub.get("title"),
            "program_code": prog.get("program_code"),
            "term_label": _term_label(term.get("school_year"), term.get("semester")),
            "grade": _to_num_grade(e.get("grade")),
            "remark": e.get("remark"),
            "units": sub.get("units", None),  # finalize below with subjects fallback
            "sy": term.get("school_year"),
            "sem": term.get("semester"),
            "section": section,
        }

    df = pd.DataFrame([flat(r) for r in rows])

    # fill missing/zero units from 'subjects' collection
    missing = df["units"].isna() | (pd.to_numeric(df["units"], errors="coerce").fillna(0) == 0)
    if missing.any():
        codes = df.loc[missing, "subject_code"].dropna().unique().tolist()
        umap = _subject_units_map(codes)
        df.loc[missing, "units"] = df.loc[missing, "subject_code"].map(lambda c: umap.get(c, 0))
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).astype(int)

    # stable order
    df["term_key"] = df["term_label"].map(_term_sort_key)
    df = df.sort_values(["term_key", "subject_code"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_cohort_stats(codes: List[str], sys: List[str], sems: List[int]) -> pd.DataFrame:
    """
    Fetch class/peer stats for the student's selected subjects/terms in one go.
    Returns avg grade and fail rate by (subject_code, term_label).
    """
    if not codes or not sys or not sems:
        return pd.DataFrame(columns=["subject_code", "term_label", "class_avg", "fail_rate_pct"])

    q = {
        "subject.code": {"$in": list(set([c for c in codes if c]))},
        "term.school_year": {"$in": list(set([s for s in sys if s]))},
        "term.semester": {"$in": list(set(sems))}
    }
    proj = {
        "subject.code": 1,
        "grade": 1,
        "term.school_year": 1,
        "term.semester": 1,
    }
    rows = list(col("enrollments").find(q, proj))
    if not rows:
        return pd.DataFrame(columns=["subject_code", "term_label", "class_avg", "fail_rate_pct"])

    def flat(r):
        t = r.get("term") or {}
        return {
            "subject_code": (r.get("subject") or {}).get("code"),
            "grade": _to_num_grade(r.get("grade")),
            "term_label": _term_label(t.get("school_year"), t.get("semester")),
        }

    df = pd.DataFrame([flat(r) for r in rows])
    if df.empty:
        return pd.DataFrame(columns=["subject_code", "term_label", "class_avg", "fail_rate_pct"])

    grp = df.groupby(["subject_code", "term_label"])
    out = grp.agg(
        class_avg=("grade", "mean"),
        fails=("grade", lambda s: (s.notna() & (s < PASSING_GRADE)).sum()),
        total=("grade", "size")
    ).reset_index()
    out["class_avg"] = out["class_avg"].round(2)
    out["fail_rate_pct"] = ((out["fails"] / out["total"]) * 100).round(1)
    return out[["subject_code", "term_label", "class_avg", "fail_rate_pct"]]


# ------------------------------------------------------------------
# Prospectus / PDF (same logic as your evaluation page)
# ------------------------------------------------------------------

def _prospectus_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return dict(overall_gpa=None, total_units_earned=0, passed_cnt=0, failed_cnt=0, inprog_cnt=0)
    d = df.copy()
    d["units"] = pd.to_numeric(d["units"], errors="coerce").fillna(0)
    g = pd.to_numeric(d["grade"], errors="coerce")
    passed = (g >= PASSING_GRADE).sum()
    failed = (g.notna() & (g < PASSING_GRADE)).sum()
    inprog = g.isna().sum()
    earned = int(d.loc[g >= PASSING_GRADE, "units"].sum())
    overall = _cumulative_gpa(d)
    return dict(overall_gpa=overall, total_units_earned=earned,
                passed_cnt=int(passed), failed_cnt=int(failed), inprog_cnt=int(inprog))


def _build_pdf(student: Dict[str, Any], per_sem: Dict[str, pd.DataFrame],
               gpa_points: List[Tuple[str, float]], summary: Dict[str, Any]) -> bytes:
    # light-weight table PDF using ReportLab (optional dependency).
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=36, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Student Prospectus — {student.get('student_name','')}", styles["Title"]))
    story.append(Paragraph(f"Student No: {student.get('student_no','—')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # summary
    story.append(Paragraph("Summary", styles["Heading2"]))
    sum_data = [
        ["Overall GPA", summary["overall_gpa"] if summary["overall_gpa"] is not None else "—"],
        ["Units Earned", summary["total_units_earned"]],
        [f"Passed (≥ {PASSING_GRADE})", summary["passed_cnt"]],
        [f"Failed (< {PASSING_GRADE})", summary["failed_cnt"]],
        ["In-Progress", summary["inprog_cnt"]],
    ]
    t = Table(sum_data, hAlign="LEFT", colWidths=[180, 200])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # per-term sections
    for sem, block in per_sem.items():
        story.append(Paragraph(sem, styles["Heading2"]))
        data = [["Subject Code", "Description", "Units", "Final Grade"]]
        for _, r in block.iterrows():
            data.append([
                r.get("subject_code", ""),
                r.get("subject_title", ""),
                int(r.get("units", 0) or 0),
                r.get("grade", ""),
            ])
        gpa = _semester_gpa(block)
        units = int(pd.to_numeric(block["units"], errors="coerce").fillna(0).sum())
        data.append(["", "Total Units / GPA", units, gpa if gpa is not None else "—"])

        tbl = Table(data, hAlign="LEFT", colWidths=[80, None, 50, 70])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26364a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef2ff")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # GPA trend table
    if gpa_points:
        story.append(Paragraph("GPA Trend", styles["Heading2"]))
        trend = [["Semester", "GPA"]] + [[k, v if v is not None else "—"] for k, v in gpa_points]
        from reportlab.platypus import Table
        t2 = Table(trend, hAlign="LEFT")
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t2)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ------------------------------------------------------------------
# Optional student finder (Registrar/Admin)
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _search_candidates(keyword: str, limit: int = 50) -> List[Dict[str, str]]:
    """
    Returns [{student_no, email, name}] from 'students' (preferred) and
    falls back to 'users' with role='student' if needed.
    """
    kw = (keyword or "").strip()
    if not kw:
        return []
    # exact email or student no match first
    students = col("students")
    hits = list(students.find(
        {"$or": [
            {"email": kw},
            {"student_no": kw},
            {"name": {"$regex": kw, "$options": "i"}}
        ]},
        {"student_no": 1, "email": 1, "name": 1}
    ).limit(limit))

    if not hits:
        # fallback to users collection
        users = col("users")
        hits = list(users.find(
            {"role": "student", "$or": [
                {"email": kw},
                {"name": {"$regex": kw, "$options": "i"}}
            ]},
            {"email": 1, "name": 1}
        ).limit(limit))
        # shape to student-like objects
        shaped = []
        for u in hits:
            shaped.append({"student_no": None, "email": u.get("email"), "name": u.get("name")})
        return shaped

    # normalize shape
    shaped = []
    for s in hits:
        shaped.append({"student_no": s.get("student_no"), "email": s.get("email"), "name": s.get("name")})
    return shaped


# ------------------------------------------------------------------
# UI / Page
# ------------------------------------------------------------------

def main():
    # Gate: student OR registrar/admin can access
    user = require_role("student", "registrar", "admin")

    st.title("🎒 Student Dashboard")

    # -------------------- student picker (admin/registrar) --------------------
    student_email: Optional[str] = None
    student_no: Optional[str] = None

    if (user.get("role") or "").lower() == "student":
        student_email = (user.get("email") or "").strip().lower()
        st.caption(f"Signed in as **{user.get('email','')}**. Showing your records.")
    else:
        st.caption("Registrar/Admin: search a student by email, student no., or name keyword.")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            kw = st.text_input("Find by email / student no. / name", value="")
        with c2:
            if st.button("Search"):
                st.session_state["_candidates"] = _search_candidates(kw, limit=50)
        with c3:
            # candidate picker
            cand = st.selectbox(
                "Matches",
                options=[f"{c.get('name','')} ({c.get('student_no') or '—'}) · {c.get('email','')}"
                         for c in st.session_state.get("_candidates", [])] or ["—"],
                index=0
            )
        with c4:
            load_btn = st.button("Load")

        # derive selection
        if st.session_state.get("_candidates") and cand != "—":
            idx = [f"{c.get('name','')} ({c.get('student_no') or '—'}) · {c.get('email','')}"
                   for c in st.session_state["_candidates"]].index(cand)
            choice = st.session_state["_candidates"][idx]
            student_email = (choice.get("email") or "").strip().lower()
            student_no = choice.get("student_no")
        elif kw and "@" in kw:
            student_email = kw.strip().lower()
        elif kw:
            student_no = kw.strip()

        if not (student_email or student_no):
            st.info("Search and pick a student to load the dashboard.")
            if not st.session_state.get("_loaded_any_student"):
                return
        elif load_btn:
            st.session_state["_loaded_any_student"] = True

    # -------------------- load records --------------------
    df = load_student_enrollments(student_email=student_email, student_no=student_no)
    if df.empty:
        st.warning("No enrollments found for this student.")
        return

    # -------------------- filters: terms / sections / subjects --------------------
    fcols = st.columns(3)
    with fcols[0]:
        term_opts = sorted(df["term_label"].dropna().unique().tolist(), key=_term_sort_key)
        sel_terms = st.multiselect("Term(s)", options=term_opts, default=term_opts)
    with fcols[1]:
        section_opts = sorted([s for s in df["section"].dropna().unique().tolist() if s not in (None, "", "—")])
        sel_sections = st.multiselect("Section(s)", options=section_opts, default=section_opts)
    with fcols[2]:
        subj_opts = sorted(df["subject_code"].dropna().unique().tolist())
        sel_subjects = st.multiselect("Subject(s)", options=subj_opts, default=subj_opts)

    if sel_terms:
        df = df[df["term_label"].isin(sel_terms)].copy()
    if sel_sections:
        df = df[df["section"].isin(sel_sections)].copy()
    if sel_subjects:
        df = df[df["subject_code"].isin(sel_subjects)].copy()

    # header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative GPA", _cumulative_gpa(df) or "—")
    gparts = _grade_buckets(df)
    c2.metric("Passed", gparts["Passed"])
    c3.metric("Failed", gparts["Failed"])
    c4.metric("In-Progress", gparts["In-Progress"])

    # ------------------------------------------------------------------
    # 1) Academic Transcript Viewer (interactive per-term tables)
    # ------------------------------------------------------------------
    st.subheader("1) Academic Transcript Viewer")
    for sem in sorted(df["term_label"].dropna().unique(), key=_term_sort_key):
        block = df[df["term_label"] == sem].copy()
        gpa = _semester_gpa(block)
        total_units = int(pd.to_numeric(block["units"], errors="coerce").fillna(0).sum())
        with st.expander(f"{sem}  ·  GPA: {gpa if gpa is not None else '—'}  ·  Units: {total_units}", expanded=True):
            show = block[["subject_code", "subject_title", "units", "grade", "remark", "section"]].rename(
                columns={
                    "subject_code": "Subject",
                    "subject_title": "Description",
                    "units": "Units",
                    "grade": "Final Grade",
                    "section": "Section",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 2) Performance Trend Over Time (GPA over semesters)
    # ------------------------------------------------------------------
    st.subheader("2) Performance Trend Over Time")
    if df["term_label"].nunique() >= 1:
        gpa_points = []
        for sem in sorted(df["term_label"].unique(), key=_term_sort_key):
            gpa_points.append((sem, _semester_gpa(df[df["term_label"] == sem])))
        chart_df = pd.DataFrame(gpa_points, columns=["Semester", "GPA"]).set_index("Semester").dropna()
        if not chart_df.empty:
            st.line_chart(chart_df)
        else:
            st.caption("No numeric GPA values yet to chart.")
    else:
        st.caption("No term data to plot.")

    # ------------------------------------------------------------------
    # 3) Subject Difficulty Ratings (fail-rate derived)
    # ------------------------------------------------------------------
    st.subheader("3) Subject Difficulty Ratings")
    cohort = load_cohort_stats(
        codes=df["subject_code"].dropna().unique().tolist(),
        sys=df["sy"].dropna().unique().tolist(),
        sems=[int(x) for x in df["sem"].dropna().unique().tolist()],
    )
    if cohort.empty:
        st.info("No cohort stats available for the selected filters.")
    else:
        merged = pd.merge(
            df[["subject_code", "subject_title", "term_label"]].drop_duplicates(),
            cohort,
            on=["subject_code", "term_label"],
            how="left",
        )

        def _rate(fr):
            if pd.isna(fr): return None
            if fr >= 60: return 5
            if fr >= 45: return 4
            if fr >= 30: return 3
            if fr >= 15: return 2
            return 1

        merged["difficulty_rating"] = merged["fail_rate_pct"].map(_rate)
        show = merged.rename(columns={
            "subject_code": "Subject",
            "subject_title": "Description",
            "class_avg": "Class Avg",
            "fail_rate_pct": "Fail %",
            "difficulty_rating": "Difficulty (1–5)"
        }).sort_values(["Fail %", "Subject"], ascending=[False, True])
        st.dataframe(show, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 4) Comparison with Class Average (pick a subject)
    # ------------------------------------------------------------------
    st.subheader("4) Comparison with Class Average")
    if df.empty:
        st.info("No data to compare.")
    else:
        subj_opts2 = sorted(df["subject_code"].dropna().unique())
        pick_subj = st.selectbox("Subject", options=subj_opts2)
        me = df[df["subject_code"] == pick_subj][["term_label", "grade"]].copy()
        me = me.groupby("term_label", as_index=False)["grade"].mean()
        me = me.sort_values("term_label", key=lambda s: s.map(_term_sort_key))

        csub = cohort[cohort["subject_code"] == pick_subj][["term_label", "class_avg"]].copy()
        csub = csub.sort_values("term_label", key=lambda s: s.map(_term_sort_key))

        comp = pd.merge(me, csub, on="term_label", how="outer").sort_values(
            "term_label", key=lambda s: s.map(_term_sort_key)
        )
        comp = comp.set_index("term_label")
        if comp[["grade", "class_avg"]].dropna(how="all").empty:
            st.caption("No comparable cohort stats for this subject in the selected filters.")
        else:
            st.bar_chart(comp.rename(columns={"grade": "My Grade", "class_avg": "Class Avg"}))

    # ------------------------------------------------------------------
    # 5) Passed vs Failed Summary
    # ------------------------------------------------------------------
    st.subheader("5) Passed vs Failed Summary")
    parts = _grade_buckets(df)
    part_df = pd.DataFrame.from_dict(parts, orient="index", columns=["count"])
    st.bar_chart(part_df)

    # ------------------------------------------------------------------
    # 6) Curriculum / Prospectus
    # ------------------------------------------------------------------
    st.subheader("6) Curriculum / Prospectus")
    per_sem: Dict[str, pd.DataFrame] = {}
    gpa_pts: List[Tuple[str, float]] = []
    for sem in sorted(df["term_label"].unique(), key=_term_sort_key):
        block = df[df["term_label"] == sem][["subject_code", "subject_title", "units", "grade"]].copy()
        per_sem[sem] = block
        gpa_pts.append((sem, _semester_gpa(block)))

    summary = _prospectus_summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall GPA", summary["overall_gpa"] if summary["overall_gpa"] is not None else "—")
    c2.metric("Units Earned", summary["total_units_earned"])
    c3.metric(f"Passed (≥ {PASSING_GRADE})", summary["passed_cnt"])
    c4.metric(f"Failed (< {PASSING_GRADE})", summary["failed_cnt"])
    c5.metric("In-Progress", summary["inprog_cnt"])

    with st.expander("View prospectus tables", expanded=False):
        for sem in sorted(per_sem.keys(), key=_term_sort_key):
            blk = per_sem[sem]
            gpa = _semester_gpa(blk)
            units = int(pd.to_numeric(blk["units"], errors="coerce").fillna(0).sum())
            st.markdown(f"**{sem}** · GPA: **{gpa if gpa is not None else '—'}** · Units: **{units}**")
            st.dataframe(
                blk.rename(columns={"subject_code": "Subject", "subject_title": "Description", "grade": "Final Grade"}),
                use_container_width=True, hide_index=True
            )

    student_stub = {
        "student_name": df["student_name"].iloc[0],
        "student_no": df["student_no"].iloc[0],
    }
    pdf_bytes = _build_pdf(student_stub, per_sem, gpa_pts, summary)
    st.download_button(
        label="Download Prospectus (PDF)",
        data=pdf_bytes if pdf_bytes else b"",
        file_name=f"prospectus_{student_stub['student_no']}_{date.today().isoformat()}.pdf",
        mime="application/pdf",
        disabled=(pdf_bytes is None or len(pdf_bytes) == 0),
    )

    st.caption(
        "Prospectus logic is aligned with the dedicated evaluation page "
        "(tables, per-term GPAs, summary, and a PDF export are built the same way)."
    )


if __name__ == "__main__":
    main()
