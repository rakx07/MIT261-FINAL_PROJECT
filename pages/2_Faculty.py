# pages/2_Faculty.py
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from db import col
from utils.auth import require_role, current_user

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


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def _term_label(sy: str | None, sem: int | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem or 0)
    except Exception:
        s = 0
    return f"{sy} S{s}" if s else sy


def _term_sort_key(label: str) -> tuple[int, int]:
    """Sort "2023-2024 S1", "2023-2024 S2", "2023-2024 S3" correctly."""
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


def _nemail(e: str | None) -> str:
    return (e or "").strip().lower()


@st.cache_data(show_spinner=False)
def load_term_catalog() -> list[tuple[str, str, int]]:
    """
    Read the full school-year/semester catalog (including Summer) from `semesters`.
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


def list_teacher_emails() -> List[Tuple[str, str]]:
    """
    Returns [(name, email), ...] from enrollments.teacher.*.
    """
    pipe = [
        {"$match": {"teacher.email": {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": "$teacher.email", "name": {"$first": "$teacher.name"}}},
        {"$sort": {"_id": 1}},
    ]
    out: list[tuple[str, str]] = []
    for r in col("enrollments").aggregate(pipe):
        em = _nemail(r.get("_id"))
        nm = r.get("name") or ""
        if em:
            out.append((nm or em, em))
    return out


@st.cache_data(show_spinner=False)
def load_enrollments_df(teacher_email: Optional[str]) -> pd.DataFrame:
    """
    Load enrollments as a flattened DataFrame.
    If `teacher_email` is provided, filter to that teacher; otherwise return all rows.
    """
    q = {}
    if teacher_email:
        q = {"teacher.email": teacher_email}

    proj = {
        "grade": 1,
        "remark": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "student.name": 1,
        "student.student_no": 1,
        "subject.code": 1,
        "subject.title": 1,
        "teacher.email": 1,
        "teacher.name": 1,
        "program.program_code": 1,
        "section": 1,
    }
    rows = list(col("enrollments").find(q, proj))

    def flatten(e: dict) -> dict:
        term = e.get("term") or {}
        stu = e.get("student") or {}
        sub = e.get("subject") or {}
        tch = e.get("teacher") or {}
        prog = e.get("program") or {}
        return {
            "student_no": stu.get("student_no"),
            "student_name": stu.get("name"),
            "subject_code": sub.get("code"),
            "subject_title": sub.get("title"),
            "grade": _to_num_grade(e.get("grade")),
            "remark": e.get("remark"),
            "term_label": _term_label(term.get("school_year"), term.get("semester")),
            "teacher_email": _nemail(tch.get("email")),
            "teacher_name": tch.get("name"),
            "program_code": prog.get("program_code"),
            "section": e.get("section"),
        }

    df = pd.DataFrame([flatten(r) for r in rows]) if rows else pd.DataFrame(
        columns=[
            "student_no", "student_name", "subject_code", "subject_title",
            "grade", "remark", "term_label", "teacher_email", "teacher_name",
            "program_code", "section",
        ]
    )
    return df


# ──────────────────────────────────────────
# Page
# ──────────────────────────────────────────

def main():
    # Auth — faculty, registrar, or admin
    user = require_role("faculty", "teacher", "registrar", "admin")

    st.title("🏫 Faculty Dashboard")
    st.caption("Teacher scope applies automatically for faculty. Registrars/Admins can filter by teacher.")
    # show who is signed in
    try:
        u = user  # if you used require_role(...) and stored it as `user`
    except NameError:
        from utils.auth import get_current_user  # if you already use this helper on this page
        u = get_current_user() or current_user()
    _user_header(u)

    teacher_email: Optional[str] = None
    teachers = list_teacher_emails()

    # Scope by role
    role = (user.get("role") or "").lower()
    if role in ("faculty", "teacher"):
        teacher_email = _nemail(user.get("email"))
    else:
        # Registrar/Admin can pick a teacher if present, else ALL
        if teachers:
            label_options = [f"{nm} ({em})" for nm, em in teachers]
            picked = st.selectbox("Filter by teacher (Registrar/Admin)", options=label_options)
            idx = label_options.index(picked)
            teacher_email = teachers[idx][1]
        else:
            st.info("No teacher emails found in enrollments; showing **all enrollments** instead.")

    df = load_enrollments_df(teacher_email)

    # Global term catalog for the picker (always show all terms)
    term_catalog = load_term_catalog()
    all_term_labels = [lbl for (lbl, _, _) in term_catalog]

    # Quick filters
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_terms = st.multiselect("Term(s)", options=all_term_labels, default=all_term_labels)
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

    # 1) Class Grade Distribution
    st.subheader("1) Class Grade Distribution")

    # Faculty Name field – show selected teacher if registrar/admin (FIX)
    faculty_display_name = user.get("name") or ""
    if role in ("registrar", "admin") and teacher_email:
        # lookup name from teachers list
        match = next((nm for nm, em in teachers if em == teacher_email), None)
        if match:
            faculty_display_name = match

    # Header fields (like your sample)
    hc1, hc2 = st.columns([1, 1.4])
    with hc1:
        st.text_input("Faculty Name:", value=faculty_display_name, key="faculty_name_display")
    with hc2:
        st.text_input(
            "Semester and School Year:",
            value=", ".join(sel_terms) if sel_terms else "",
            key="term_sy_display",
        )

    graded = df.dropna(subset=["grade"])
    if graded.empty:
        st.info("No graded entries found for this scope.")
    else:
        # ——— Build the subject-by-bin distribution table (percentages) ———
        bins = [0, 75, 80, 85, 90, 95, 100]
        labels = ["Below 75 (%)", "75–79 (%)", "80–84 (%)", "85–89 (%)", "90–94 (%)", "95–100 (%)"]

        tmp = graded.copy()
        tmp["Course Code"] = tmp["subject_code"].fillna("")
        tmp["Course Name"] = tmp["subject_title"].fillna("")
        tmp["bin"] = pd.cut(tmp["grade"], bins=bins, labels=labels, right=True, include_lowest=True)

        counts = (
            tmp.groupby(["Course Code", "Course Name", "bin"])
               .size()
               .unstack(fill_value=0)
        )

        for col in labels:
            if col not in counts.columns:
                counts[col] = 0

        counts = counts[labels]
        totals = counts.sum(axis=1)
        pct = (counts.div(totals.replace(0, 1), axis=0) * 100).round(0).astype("Int64").astype(str) + "%"

        pct["Total"] = totals.values
        pct = pct.reset_index()

        st.dataframe(
            pct[["Course Code", "Course Name"] + labels + ["Total"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Followed by: histogram**")
        hist_bins = list(range(60, 101, 5))
        hist_counts = pd.cut(
            graded["grade"], bins=hist_bins, right=True, include_lowest=True
        ).value_counts().sort_index()
        chart_df = pd.DataFrame(
            {"Range": hist_counts.index.astype(str), "Count": hist_counts.values}
        ).set_index("Range")
        st.bar_chart(chart_df)

    # 2) Student Progress Tracker  (REPLACED your old "Avg by term")
    st.subheader("2) Student Progress Tracker")
    st.caption("Shows longitudinal performance for individual students. Filtered by Subject or Course or YearLevel.")

    if graded.empty:
        st.info("No data available for student progress.")
    else:
        g = graded.copy()
        # GPA scale (0–4), like the screenshot
        g["gpa"] = (g["grade"].astype(float) / 100.0 * 4.0).clip(0, 4).round(2)

        # pick terms for columns: prefer selected terms; else latest 3
        terms_order = sel_terms if sel_terms else \
            sorted(g["term_label"].dropna().unique().tolist(), key=_term_sort_key)
        if len(terms_order) > 3:
            terms_order = terms_order[-3:]

        pivot = (
            g[g["term_label"].isin(terms_order)]
            .groupby(["student_no", "student_name", "term_label"])["gpa"]
            .mean()
            .reset_index()
            .pivot_table(index=["student_no", "student_name"],
                         columns="term_label", values="gpa", aggfunc="mean")
            .reindex(columns=terms_order)
        )

        # Trend like in image (↑ Improving / ↓ Needs Attention / → Stable High)
        def trend_text(row):
            vals = [v for v in row.tolist() if pd.notnull(v)]
            if len(vals) < 2:
                return "—"
            delta = vals[-1] - vals[0]
            if delta >= 0.10:
                return "↑ Improving"
            if delta <= -0.10:
                return "↓ Needs Attention"
            return "→ Stable High"

        trend = pivot.apply(trend_text, axis=1)
        pivot_display = pivot.copy().round(2)
        pivot_display.insert(0, "Student ID", [i[0] for i in pivot_display.index])
        pivot_display.insert(1, "Name", [i[1] for i in pivot_display.index])
        pivot_display["Overall Trend"] = trend.values
        pivot_display = pivot_display.reset_index(drop=True)

        st.dataframe(
            pivot_display[["Student ID", "Name"] + terms_order + ["Overall Trend"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Followed by: line graph / scatter chart.**")
        long_df = (
            pivot.reset_index()
                 .melt(id_vars=["student_no", "student_name"],
                       value_vars=terms_order, var_name="Term", value_name="GPA")
                 .dropna(subset=["GPA"])
                 .sort_values(["student_no", "Term"])
        )
        chart_wide = (
            long_df.pivot_table(index="Term", columns="student_name",
                                values="GPA", aggfunc="mean")
                   .reindex(terms_order)
        )
        st.line_chart(chart_wide)

    # 3) Subject Difficulty Heatmap (Fail %)
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
            st.dataframe(fail.sort_values("fail_rate_%", ascending=False), use_container_width=True)

    # 4) Intervention (latest term)
    st.subheader("4) Intervention Candidates")
    if graded.empty:
        st.info("No graded entries.")
    else:
        latest_term = None
        if not graded["term_label"].isna().all():
            uniq = graded["term_label"].dropna().unique().tolist()
            if uniq:
                latest_term = sorted(uniq, key=_term_sort_key)[-1]
        cur = graded if not latest_term else graded[graded["term_label"] == latest_term]
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
            st.dataframe(show, use_container_width=True, height=min(520, 35 + 28 * len(show)))

    # 5) Grade Submission Status (keep ungraded)
    st.subheader("5) Grade Submission Status")
    if df.empty:
        st.info("No enrollments to summarize.")
    else:
        status = (
            df.groupby("subject_code")
            .agg(
                total=("grade", "size"),
                graded=("grade", lambda s: s.notna().sum()),
            )
            .reset_index()
        )
        status["completion_%"] = (status["graded"] / status["total"] * 100).round(1)
        status = status.sort_values(["completion_%", "subject_code"], ascending=[True, True]).rename(
            columns={
                "subject_code": "Subject",
                "total": "Total Enrollments",
                "graded": "Graded Count",
                "completion_%": "Completion %",
            }
        )
        st.dataframe(status, use_container_width=True)


if __name__ == "__main__":
    main()
