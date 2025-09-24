from __future__ import annotations

import io
import math
from datetime import date
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from db import col
from utils.auth import current_user, require_role

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

# ----------------------------
# General helpers
# ----------------------------

def _term_label(sy: str | None, sem: int | str | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem) if sem is not None and str(sem).strip() != "" else 0
    except Exception:
        s = 0
    if isinstance(sem, str) and sem and not sem.isdigit():
        # allows strings like "S1", "S2", "S3" (summer)
        return f"{sy} {sem}"
    return f"{sy} S{s}" if s else sy


def _term_sort_key(label: str) -> tuple[int, int]:
    """
    Sort like "2023-2024 S1" < "2023-2024 S2" < "2024-2025 S1".
    Falls back gracefully for odd labels.
    """
    if not isinstance(label, str) or " " not in label:
        return (0, 0)
    part_sy, part_s = label.split(" ", 1)
    try:
        y0 = int(part_sy.split("-")[0])
    except Exception:
        y0 = 0
    s = 0
    try:
        if part_s.startswith("S"):
            s = int(part_s[1:])
        elif part_s.isdigit():
            s = int(part_s)
    except Exception:
        s = 0
    return (y0, s)


def _to_num_grade(x) -> float | None:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

# ----------------------------
# Role / teacher helpers
# ----------------------------

def list_teacher_emails() -> List[Tuple[str, str]]:
    """
    Returns [(name, email), ...] from enrollments.teacher.* if present.
    """
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

# ----------------------------
# Curriculum / subject-units map
# ----------------------------

@st.cache_data(show_spinner=False)
def _build_subject_units_map() -> Dict[str, Dict[str, Any]]:
    """
    Build a {subject_code: {"units": int, "name": str}} map
    by scanning likely curriculum collections.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    candidates = ["curriculum", "curricula", "program_curricula", "prospectus", "curriculums"]
    for cname in candidates:
        try:
            c = col(cname)
        except Exception:
            continue
        try:
            for doc in c.find({}, {"subjects": 1}):
                subs = doc.get("subjects") or []
                if isinstance(subs, list):
                    for s in subs:
                        code = (s.get("subjectCode") or s.get("code") or s.get("subject_code") or "").strip()
                        if not code:
                            continue
                        units = s.get("units")
                        if units is None:
                            try:
                                # some schemas keep lec/lab
                                units = (s.get("lec") or 0) + (s.get("lab") or 0)
                            except Exception:
                                units = None
                        name = s.get("subjectName") or s.get("name") or s.get("title")
                        if code and code not in mapping:
                            mapping[code] = {"units": units, "name": name}
        except Exception:
            pass
    return mapping

# ----------------------------
# Enrollment → df loader
# ----------------------------

@st.cache_data(show_spinner=False)
def load_enrollments_df(student_email: Optional[str] = None,
                        student_no: Optional[str] = None,
                        restrict_teacher_email: Optional[str] = None) -> pd.DataFrame:
    """
    Pulls enrollments into a flattened DataFrame.
    You can restrict to a student (email or no) and/or to a teacher's classes.
    """
    q: Dict[str, Any] = {}
    if student_email:
        q["student.email"] = student_email.strip().lower()
    if student_no:
        q["student.student_no"] = student_no
    if restrict_teacher_email:
        q["teacher.email"] = restrict_teacher_email.strip().lower()

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
        "teacher.email": 1,
        "teacher.name": 1,
        "program.program_code": 1,
        "section": 1,
    }

    rows = list(col("enrollments").find(q, proj))
    if not rows:
        return pd.DataFrame(
            columns=[
                "student_no", "student_name", "student_email",
                "subject_code", "subject_title", "grade", "remark",
                "term_label", "teacher_email", "teacher_name",
                "program_code", "section"
            ]
        )

    def flatten(e):
        term = e.get("term") or {}
        stu = e.get("student") or {}
        sub = e.get("subject") or {}
        tch = e.get("teacher") or {}
        prog = e.get("program") or {}
        return {
            "student_no": stu.get("student_no"),
            "student_name": stu.get("name"),
            "student_email": (stu.get("email") or "").strip().lower(),
            "subject_code": sub.get("code"),
            "subject_title": sub.get("title"),
            "grade": _to_num_grade(e.get("grade")),
            "remark": e.get("remark"),
            "term_label": _term_label(term.get("school_year"), term.get("semester")),
            "teacher_email": (tch.get("email") or "").strip().lower(),
            "teacher_name": tch.get("name"),
            "program_code": prog.get("program_code"),
            "section": e.get("section"),
        }

    df = pd.DataFrame([flatten(r) for r in rows])
    return df

# ----------------------------
# Prospectus helpers (from your evaluation page)
# ----------------------------

PASSING_GRADE = 75

def _compute_semester_gpa(df_sem: pd.DataFrame) -> Optional[float]:
    if df_sem.empty:
        return None
    u = pd.to_numeric(df_sem["units"], errors="coerce").fillna(0)
    g = pd.to_numeric(df_sem["grade"], errors="coerce").fillna(0)
    total_units = u.sum()
    return round(float((g * u).sum() / total_units), 2) if total_units > 0 else None


def _compute_prospectus_summary(df_all: pd.DataFrame) -> Dict[str, Any]:
    if df_all.empty:
        return dict(overall_gpa=None, total_units_earned=0, passed_cnt=0, failed_cnt=0, inprog_cnt=0)
    df = df_all.copy()
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["grade_num"] = pd.to_numeric(df["grade"], errors="coerce")

    passed_mask = df["grade_num"].ge(PASSING_GRADE)
    failed_mask = df["grade_num"].lt(PASSING_GRADE)
    inprog_mask = df["grade_num"].isna()

    passed_cnt = int(passed_mask.sum())
    failed_cnt = int(failed_mask.sum())
    inprog_cnt = int(inprog_mask.sum())
    total_units_earned = int(df.loc[passed_mask, "units"].sum())

    g = df["grade_num"].fillna(0)
    u = df["units"]
    usable = df["grade_num"].notna() & u.gt(0)
    overall_gpa = round(float((g[usable] * u[usable]).sum() / u[usable].sum()), 2) if usable.any() else None

    return dict(
        overall_gpa=overall_gpa,
        total_units_earned=total_units_earned,
        passed_cnt=passed_cnt,
        failed_cnt=failed_cnt,
        inprog_cnt=inprog_cnt,
    )

def _build_pdf(student: Dict[str, Any],
               per_sem: Dict[str, pd.DataFrame],
               gpa_points: List[Tuple[str, Optional[float]]],
               summary: Dict[str, Any]) -> bytes:
    # Light-weight PDF via ReportLab if available
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

    story.append(Paragraph(f"Student Evaluation Sheet — {student.get('student_name','')}", styles["Title"]))
    story.append(Paragraph(f"Program: {student.get('program_code','—')}    Student No: {student.get('student_no','—')}", styles["Normal"]))
    story.append(Paragraph(f"Email: {student.get('student_email','—')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Summary
    story.append(Paragraph("Prospectus Summary", styles["Heading2"]))
    sum_data = [
        ["Overall GPA", summary["overall_gpa"] if summary["overall_gpa"] is not None else "—"],
        ["Total Units Earned", summary["total_units_earned"]],
        [f"Passed (≥ {PASSING_GRADE})", summary["passed_cnt"]],
        [f"Failed (< {PASSING_GRADE})", summary["failed_cnt"]],
        ["In-Progress / No Grade", summary["inprog_cnt"]],
    ]
    t_sum = Table(sum_data, hAlign="LEFT", colWidths=[180, 200])
    t_sum.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 14))

    # Per-semester tables
    for sem_label, df in per_sem.items():
        story.append(Paragraph(sem_label, styles["Heading2"]))
        data = [["Subject Code", "Description", "Units", "Final Grade", "Instructor"]]
        for _, r in df.iterrows():
            data.append([
                r.get("subject_code", ""),
                r.get("subject_title", ""),
                int((r.get("units", 0) or 0)),
                r.get("grade", ""),
                r.get("teacher_name", "") or r.get("teacher_email", ""),
            ])
        gpa = _compute_semester_gpa(df)
        total_units = int(pd.to_numeric(df["units"], errors="coerce").fillna(0).sum())
        data.append(["", "Total Units", total_units, gpa if gpa is not None else "—", ""])

        tbl = Table(data, hAlign="LEFT", colWidths=[80, None, 50, 70, 140])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26364a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (2, 1), (3, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef2ff")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # Trend
    if gpa_points:
        story.append(Paragraph("GPA Trend", styles["Heading2"]))
        trend = [["Semester", "GPA"]] + [[k, v if v is not None else "—"] for k, v in gpa_points]
        t2 = Table(trend, hAlign="LEFT")
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ]))
        story.append(t2)

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ----------------------------
# Prospectus builder from enrollments
# ----------------------------

def build_prospectus(df_enr: pd.DataFrame,
                     student_stub: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame],
                                                            List[Tuple[str, Optional[float]]],
                                                            Dict[str, Any],
                                                            pd.DataFrame]:
    """
    Convert the filtered enrollments of one student into a prospectus view.
    Adds units by looking up a curriculum mapping when available.
    """
    if df_enr.empty:
        return {}, [], student_stub, df_enr

    units_map = _build_subject_units_map()
    df = df_enr.copy()

    # Attach units & nicer titles if curriculum has them
    def _units_for(code: str | None) -> Optional[float]:
        if not code:
            return None
        hit = units_map.get(code)
        return hit.get("units") if hit else None

    def _title_for(code: str | None, fallback: str | None) -> str | None:
        if not code:
            return fallback
        hit = units_map.get(code)
        return (hit.get("name") or fallback) if hit else fallback

    df["units"] = df["subject_code"].map(lambda c: _units_for(c))
    df["subject_title"] = df.apply(lambda r: _title_for(r["subject_code"], r["subject_title"]), axis=1)

    # build order and group
    order_df = df[["term_label"]].drop_duplicates()
    order_df["sortkey"] = order_df["term_label"].map(_term_sort_key)
    order_df = order_df.sort_values("sortkey")

    per_sem: Dict[str, pd.DataFrame] = {}
    gpa_points: List[Tuple[str, Optional[float]]] = []

    for term in order_df["term_label"].tolist():
        block = df[df["term_label"] == term].copy()
        per_sem[term] = block[["subject_code", "subject_title", "units", "grade", "teacher_name", "teacher_email"]]
        gpa_points.append((term, _compute_semester_gpa(block)))

    return per_sem, gpa_points, student_stub, df

# ----------------------------
# Page
# ----------------------------

def main():
    u = current_user()
    role = (u.get("role") or "").lower()

    st.title("👨‍🎓 Student Dashboard")

    try:
        u = user  # if present
    except NameError:
        u = current_user()
    _user_header(u)

    if role in ("student",):
        st.caption(f"Signed in as {u.get('email','')}. Showing your records.")
    else:
        st.caption("Faculty mode: pick from your own classes by term and subject, then choose a student.")

    # --- Role filters / scope ---

    teacher_email: Optional[str] = None
    teachers = list_teacher_emails()

    if role in ("faculty", "teacher"):
        teacher_email = (u.get("email") or "").strip().lower()

    # --- Top filters (teacher scope first, then choose a student) ---

    # For faculty: terms and subjects they actually taught
    df_scope = load_enrollments_df(restrict_teacher_email=teacher_email) if teacher_email else load_enrollments_df()

    # Build options safely
    all_terms = sorted([t for t in df_scope["term_label"].dropna().unique()], key=_term_sort_key)
    default_terms = [t for t in all_terms[-3:]]  # last few by default

    st.markdown("**Term(s)**")
    sel_terms = st.multiselect("Term(s)",
                               options=all_terms,
                               default=[t for t in default_terms if t in all_terms],
                               key="student_terms_top")

    df_scope = df_scope[df_scope["term_label"].isin(sel_terms)] if sel_terms else df_scope

    subjects = sorted(df_scope["subject_code"].dropna().unique())
    st.markdown("**Subject(s)**")
    sel_subjects = st.multiselect("Subject(s)", options=subjects, default=subjects[:2], key="student_subjects_top")

    df_scope = df_scope[df_scope["subject_code"].isin(sel_subjects)] if sel_subjects else df_scope

    sections = sorted([s for s in df_scope["section"].dropna().unique()])
    st.markdown("**Section(s)**")
    sel_sections = st.multiselect("Section(s)", options=sections, default=sections, key="student_sections_top")
    if sel_sections:
        df_scope = df_scope[df_scope["section"].isin(sel_sections)]

    # --- Pick student ---

    student_label = None
    student_email = None
    student_no = None

    # If student is signed-in -> forced to own email
    if role == "student":
        student_email = (u.get("email") or "").strip().lower()
    else:
        # Registrar/Admin/Faculty can pick
        stu_opts = (
            df_scope[["student_name", "student_email", "student_no"]]
            .dropna(subset=["student_email"])
            .drop_duplicates()
        )
        if not stu_opts.empty:
            def _fmt(r):
                nm = r["student_name"] or ""
                em = r["student_email"] or ""
                no = r["student_no"] or ""
                return f"{nm} ({no}) — {em}".strip()

            labels = [_fmt(r) for _, r in stu_opts.iterrows()]
            picked = st.selectbox("Student", options=labels, index=0 if labels else None, key="student_pick")
            if picked:
                row = stu_opts.iloc[labels.index(picked)]
                student_label = picked
                student_email = row["student_email"]
                student_no = row["student_no"]
        else:
            st.info("No students found for the selected filters.")

    # --- Load enrollments for selected student (or current student) ---

    df = load_enrollments_df(student_email=student_email,
                             student_no=student_no,
                             restrict_teacher_email=teacher_email)

    # ----------------------------
    # 1) Class Grade Distribution
    # ----------------------------
    st.subheader("1) Class Grade Distribution (Histogram)")
    graded = df.dropna(subset=["grade"])
    if graded.empty:
        st.info("No graded entries found for this scope.")
    else:
        bins = list(range(60, 101, 5))
        hist = pd.cut(graded["grade"], bins=bins, right=True, include_lowest=True).value_counts().sort_index()
        chart_df = pd.DataFrame({"range": hist.index.astype(str), "count": hist.values}).set_index("range")
        st.bar_chart(chart_df)

    # ----------------------------
    # 2. Performance Trend Over Time
    # ----------------------------
    st.subheader("2. Performance Trend Over Time")

    if graded.empty:
        st.info("No data to compute term averages.")
    else:
        # Compute per-term average (Semester GPA)
        g = (
            graded.groupby("term_label", as_index=False)["grade"]
            .mean()
            .rename(columns={"grade": "Semester GPA"})
        )

        # Sort by academic term order and round like the mockup
        g = g.sort_values(by="term_label", key=lambda s: s.map(_term_sort_key))
        g["Semester GPA"] = g["Semester GPA"].round(2)

        # Show compact table like the screenshot (Semester | Semester GPA)
        tbl = g.rename(columns={"term_label": "Semester"})[["Semester", "Semester GPA"]]
        st.table(tbl)

        st.markdown(
            "<strong>Description:</strong> Represents GPA progression across semesters, "
            "ideal for a line chart visual.",
            unsafe_allow_html=True,
        )

        # Line chart for GPA trend
        chart_df = tbl.set_index("Semester")
        st.line_chart(chart_df)

        
    # ----------------------------
    # 3. Subject Difficulty Ratings  (scoped to student's section & term)
    # ----------------------------
    st.subheader("3. Subject Difficulty Ratings")
    st.caption("Shows where the student’s grade sits relative to classmates by subject (same section & same term).")

    if df.empty:
        st.info("No data for difficulty ratings.")
    else:
        # Student’s latest record per subject (grade + section + term)
        stu_grades = df.dropna(subset=["grade"]).copy()
        if stu_grades.empty:
            st.info("No student grades to compare.")
        else:
            stu_grades["sortkey"] = stu_grades["term_label"].map(_term_sort_key)
            stu_latest = (
                stu_grades.sort_values("sortkey")
                .groupby("subject_code", as_index=False)
                .tail(1)[["subject_code", "subject_title", "grade", "section", "term_label"]]
                .rename(columns={
                    "grade": "Your Grade (%)",
                    "section": "stu_section",
                    "term_label": "stu_term",
                })
            )

            # Class population: restrict strictly to the student's same section & term per subject
            pop = df_scope.dropna(subset=["grade"]).copy()
            if pop.empty or stu_latest.empty:
                st.info("No class population grades available for the student's section/term.")
            else:
                pop_match = pop.merge(
                    stu_latest[["subject_code", "stu_section", "stu_term"]],
                    on="subject_code",
                    how="inner",
                )
                pop_match = pop_match[
                    (pop_match["section"] == pop_match["stu_section"]) &
                    (pop_match["term_label"] == pop_match["stu_term"])
                ]

                if pop_match.empty:
                    st.info("No classmates found in the same section/term for these subjects.")
                else:
                    # Bucketize grades
                    def _bucket(v: float) -> str:
                        try:
                            g = float(v)
                        except Exception:
                            return "NA"
                        if g >= 90: return "90–100 (%)"
                        if g >= 80: return "80–89 (%)"
                        if g >= 70: return "70–79 (%)"
                        if g >= 60: return "60–69 (%)"
                        return "< 60 (%)"

                    pop_match["bucket"] = pop_match["grade"].map(_bucket)

                    # Counts per subject/bucket and totals
                    counts = (
                        pop_match.groupby(["subject_code", "subject_title", "bucket"], dropna=False)
                        .size()
                        .reset_index(name="n")
                    )
                    totals = (
                        pop_match.groupby(["subject_code", "subject_title"], dropna=False)
                        .size()
                        .reset_index(name="Total Students")
                    )

                    # Pivot to percentage columns
                    pivot = counts.pivot_table(
                        index=["subject_code", "subject_title"],
                        columns="bucket",
                        values="n",
                        fill_value=0,
                        aggfunc="sum",
                    ).reset_index()

                    for colname in ["90–100 (%)", "80–89 (%)", "70–79 (%)", "60–69 (%)", "< 60 (%)"]:
                        if colname not in pivot.columns:
                            pivot[colname] = 0

                    ratings = pivot.merge(totals, on=["subject_code", "subject_title"], how="left")
                    for colname in ["90–100 (%)", "80–89 (%)", "70–79 (%)", "60–69 (%)", "< 60 (%)"]:
                        ratings[colname] = ((ratings[colname] / ratings["Total Students"]) * 100).round(0).astype("Int64")

                    # Attach student's own grade (latest per subject)
                    ratings = ratings.merge(stu_latest[["subject_code", "Your Grade (%)"]], on="subject_code", how="left")

                    # Difficulty rule (based on share of "< 60 (%)")
                    def _difficulty(row):
                        lt60 = int(row.get("< 60 (%)") or 0)
                        if lt60 >= 20:
                            return "High"
                        if lt60 >= 10:
                            return "Medium"
                        return "Low"

                    ratings["Difficulty Level"] = ratings.apply(_difficulty, axis=1)

                    # Final display
                    show = ratings.rename(columns={
                        "subject_code": "Course Code",
                        "subject_title": "Course Name",
                    })[
                        ["Course Code", "Course Name", "Total Students", "Your Grade (%)",
                        "90–100 (%)", "80–89 (%)", "70–79 (%)", "60–69 (%)", "< 60 (%)",
                        "Difficulty Level"]
                    ]

                    # Order High → Medium → Low, then by "< 60 (%)" desc
                    lvl_order = pd.CategoricalDtype(categories=["High", "Medium", "Low"], ordered=True)
                    show["Difficulty Level"] = show["Difficulty Level"].astype(lvl_order)
                    show = show.sort_values(["Difficulty Level", "< 60 (%)"], ascending=[True, False]).reset_index(drop=True)

                    # Student header line like your mockup
                    header_left = (df["student_no"].dropna().iloc[0] if not df["student_no"].dropna().empty else "—")
                    header_right = (df["student_name"].dropna().iloc[0] if not df["student_name"].dropna().empty else "—")
                    st.markdown(f"**Student:** {header_left} – {header_right}")

                    st.dataframe(show, use_container_width=True)


    # ----------------------------
    # 4. Comparison with Class Average
    # ----------------------------
    st.subheader("4. Comparison with Class Average")

    if graded.empty:
        st.info("No graded entries available for comparison.")
    else:
        # Student’s latest record per subject (grade + section + term)
        stu_grades = graded.dropna(subset=["grade"]).copy()
        stu_grades["sortkey"] = stu_grades["term_label"].map(_term_sort_key)
        stu_latest = (
            stu_grades.sort_values("sortkey")
            .groupby("subject_code", as_index=False)
            .tail(1)[["subject_code", "subject_title", "grade", "section", "term_label"]]
            .rename(columns={
                "grade": "Your Grade (%)",
                "section": "stu_section",
                "term_label": "stu_term",
            })
        )

        # Class population: restrict strictly to the student's same section & term per subject
        pop = df_scope.dropna(subset=["grade"]).copy()
        if pop.empty or stu_latest.empty:
            st.info("No class population grades available for the student's section/term.")
        else:
            # Join to get, for each subject, the student's section & term to filter classmates
            pop_match = pop.merge(
                stu_latest[["subject_code", "stu_section", "stu_term"]],
                on="subject_code",
                how="inner",
            )

            # Keep only rows in the same section and same term as the student's latest record for that subject
            pop_match = pop_match[
                (pop_match["section"] == pop_match["stu_section"]) &
                (pop_match["term_label"] == pop_match["stu_term"])
            ]

            if pop_match.empty:
                st.info("No classmates found in the same section/term for comparison.")
            else:
                # Class average & ranking per subject (within student's section & term)
                stats = (
                    pop_match.groupby(["subject_code", "subject_title"], dropna=False)["grade"]
                    .agg(["mean", "count", list])
                    .reset_index()
                )
                stats = stats.rename(columns={"mean": "Class Average (%)", "count": "Total Students"})

                # Merge with student's own grade (latest per subject)
                merged = stats.merge(
                    stu_latest[["subject_code", "subject_title", "Your Grade (%)"]],
                    on=["subject_code", "subject_title"],
                    how="left",
                )

                # Rank calculation (within filtered classmates list)
                def _rank(row):
                    grades = sorted(row["list"], reverse=True)
                    yg = row["Your Grade (%)"]
                    try:
                        pos = grades.index(yg) + 1
                    except ValueError:
                        pos = None
                    return f"{pos} of {len(grades)}" if pos else "—"

                merged["Your Rank"] = merged.apply(_rank, axis=1)

                # Remarks
                def _remark(row):
                    yg = row["Your Grade (%)"]
                    ca = row["Class Average (%)"]
                    if pd.isna(yg) or pd.isna(ca):
                        return "—"
                    if yg > ca + 5:
                        return "Above class average—excellent standing"
                    elif yg < ca - 5:
                        return "Below class average—needs additional support"
                    else:
                        return "Slightly above average—solid performance"

                merged["Remark"] = merged.apply(_remark, axis=1)

                # Round percentages
                merged["Your Grade (%)"] = merged["Your Grade (%)"].round(0)
                merged["Class Average (%)"] = merged["Class Average (%)"].round(0)

                # Final table
                show = merged.rename(columns={
                    "subject_code": "Course Code",
                    "subject_title": "Course Name",
                })[
                    ["Course Code", "Course Name", "Total Students", "Your Grade (%)",
                    "Class Average (%)", "Your Rank", "Remark"]
                ]

                # Student header line like screenshot
                header_left = (df["student_no"].dropna().iloc[0] if not df["student_no"].dropna().empty else "—")
                header_right = (df["student_name"].dropna().iloc[0] if not df["student_name"].dropna().empty else "—")
                st.markdown(f"**Student:** {header_left} – {header_right}")

                st.dataframe(show, use_container_width=True)

                st.markdown(
                    "<strong>Description:</strong> Highlights how the student’s performance stacks up against peers.",
                    unsafe_allow_html=True,
                )


    # ----------------------------
    # 5) Grade Submission Status
    # ----------------------------
    st.subheader("5) Grade Submission Status")
    df_status = df.copy()
    if df_status.empty:
        st.info("No enrollments to summarize.")
    else:
        status = (
            df_status.groupby("subject_code")
            .agg(
                total=("grade", "size"),
                graded_cnt=("grade", lambda s: s.notna().sum()),
            )
            .reset_index()
        )
        status["completion_%"] = (status["graded_cnt"] / status["total"] * 100).round(1)
        status = status.sort_values(["completion_%", "subject_code"], ascending=[True, True])
        status = status.rename(
            columns={
                "subject_code": "Subject",
                "total": "Total Enrollments",
                "graded_cnt": "Graded Count",
                "completion_%": "Completion %",
            }
        )
        st.dataframe(status, use_container_width=True)

    # ----------------------------
    # 6) Prospectus / Curriculum Evaluation
    # ----------------------------
    st.subheader("6) Prospectus / Curriculum Evaluation")

    if role == "student":
        # Use the signed-in student's info
        student_stub = {
            "student_name": u.get("name") or "",
            "student_email": u.get("email") or "",
            "student_no": "",  # unknown from user record
            "program_code": df["program_code"].dropna().iloc[0] if not df.empty else "",
        }
    else:
        if not student_email:
            st.info("Select a student above to show the prospectus.")
            return
        # build a compact header from df
        student_stub = {
            "student_name": df["student_name"].dropna().iloc[0] if not df.empty else "",
            "student_email": student_email or "",
            "student_no": df["student_no"].dropna().iloc[0] if not df.empty else "",
            "program_code": df["program_code"].dropna().iloc[0] if not df.empty else "",
        }

    # Optional extra filters for the prospectus area (safe keys to avoid duplicates)
    with st.expander("Prospectus Filters", expanded=True):
        # Terms specifically for this student
        stu_terms = sorted([t for t in df["term_label"].dropna().unique()], key=_term_sort_key)
        default_stu_terms = stu_terms  # show all by default
        sel_terms_prosp = st.multiselect("Term(s)", options=stu_terms,
                                         default=[t for t in default_stu_terms if t in stu_terms],
                                         key="prospectus_terms")

        df_for_prosp = df[df["term_label"].isin(sel_terms_prosp)] if sel_terms_prosp else df

        # The user may also narrow on specific subjects/sections
        subj_opts = sorted([s for s in df_for_prosp["subject_code"].dropna().unique()])
        sel_subj_prosp = st.multiselect("Subject(s)", options=subj_opts, default=subj_opts, key="prospectus_subjects")

        sect_opts = sorted([s for s in df_for_prosp["section"].dropna().unique()])
        sel_sect_prosp = st.multiselect("Section(s)", options=sect_opts, default=sect_opts, key="prospectus_sections")

        if sel_subj_prosp:
            df_for_prosp = df_for_prosp[df_for_prosp["subject_code"].isin(sel_subj_prosp)]
        if sel_sect_prosp:
            df_for_prosp = df_for_prosp[df_for_prosp["section"].isin(sel_sect_prosp)]

    if df_for_prosp.empty:
        st.info("No enrollments to render for the selected prospectus filters.")
        return

    per_sem, gpa_points, student_hdr, df_curr = build_prospectus(df_for_prosp, student_stub)

    # Summary tiles
    summary = _compute_prospectus_summary(df_curr)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall GPA", summary["overall_gpa"] if summary["overall_gpa"] is not None else "—")
    c2.metric("Units Earned", summary["total_units_earned"])
    c3.metric(f"Passed (≥ {PASSING_GRADE})", summary["passed_cnt"])
    c4.metric(f"Failed (< {PASSING_GRADE})", summary["failed_cnt"])
    c5.metric("In-Progress", summary["inprog_cnt"])

    # Per-semester tables
    order = sorted(per_sem.keys(), key=_term_sort_key)
    for sem_label in order:
        block = per_sem[sem_label].copy()
        gpa = _compute_semester_gpa(block)
        total_units = int(pd.to_numeric(block["units"], errors="coerce").fillna(0).sum())
        with st.expander(sem_label, expanded=True):
            show = block.rename(columns={
                "subject_code": "Subject Code",
                "subject_title": "Description",
                "units": "Units",
                "grade": "Final Grade",
                "teacher_name": "Instructor",
            })
            # prefer teacher_name, but keep email if no name
            if "Instructor" in show and show["Instructor"].isna().all() and "teacher_email" in block:
                show["Instructor"] = block["teacher_email"]
            totals = pd.DataFrame([{
                "Subject Code": "", "Description": "Total Units",
                "Units": total_units, "Final Grade": gpa, "Instructor": ""
            }])
            st.dataframe(pd.concat([show, totals], ignore_index=True), use_container_width=True)
            st.markdown(
                f"**Semester GPA:** <span style='color:#1f5cff;font-weight:700'>{gpa if gpa is not None else '—'}</span>",
                unsafe_allow_html=True
            )

    # Trend chart
    st.markdown("**GPA Trend**")
    trend_df = pd.DataFrame(gpa_points, columns=["Semester", "GPA"]).set_index("Semester").dropna()
    if not trend_df.empty:
        st.line_chart(trend_df)
    else:
        st.caption("No numeric GPA values yet to chart.")

    # PDF download
    pdf_bytes = _build_pdf(student_hdr, per_sem, gpa_points, summary)
    st.download_button(
        "Download PDF",
        data=pdf_bytes if pdf_bytes else b"",
        file_name=f"evaluation_{(student_hdr.get('student_no') or 'student')}_{date.today().isoformat()}.pdf",
        mime="application/pdf",
        disabled=(pdf_bytes is None or len(pdf_bytes) == 0),
        key="prospectus_pdf_dl",
    )

if __name__ == "__main__":
    # Guard access: students can view, faculty/registrar/admin too.
    # If you want to restrict further, swap the roles here.
    require_role("student", "teacher", "faculty", "registrar", "admin")
    main()
