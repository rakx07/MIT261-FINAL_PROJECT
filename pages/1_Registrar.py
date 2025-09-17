# pages/1_Registrar.py
import io
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from db import col

# ───────────────────────────────────────────────────────────────────────────────
# Access control
# ───────────────────────────────────────────────────────────────────────────────
def guard_role(*roles):
    u = st.session_state.get("user")
    if not u:
        st.error("Please log in from the home page."); st.stop()
    if roles and u.get("role") not in roles:
        st.warning(f"Access restricted to: {', '.join(roles)}"); st.stop()
    return u

user = guard_role("admin", "registrar")

st.title("📊 Registrar Dashboard")

# ───────────────────────────────────────────────────────────────────────────────
# Small helpers
# ───────────────────────────────────────────────────────────────────────────────
def df_download_button(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")

def fig_download_button(fig, filename, label="Download PNG"):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    st.download_button(label, buf.getvalue(), filename, "image/png")

def has_collection(name):
    try: return name in col(name).database.list_collection_names()
    except Exception: return False

def nonempty(name):
    try: return (col(name).estimated_document_count() or 0) > 0
    except Exception: return False

SRC = "grades_ingested" if (has_collection("grades_ingested") and nonempty("grades_ingested")) else "enrollments"
KEY_GRADE = "Grade" if SRC == "grades_ingested" else "grade"
KEY_SUBJ  = "SubjectCode" if SRC == "grades_ingested" else "subject.code"
KEY_REMARK= "Remark" if SRC == "grades_ingested" else "remark"

# term helpers
def term_label(row):
    if SRC == "grades_ingested":
        t = row.get("term", {})
        sy = t.get("school_year") or ""
        sm = t.get("semester") or ""
        return f"{sy} S{sm}" if sy or sm else ""
    else:
        t = row.get("term", {})
        return f"{t.get('school_year','')} S{t.get('semester','')}"

def universe_terms():
    try:
        if SRC == "grades_ingested":
            rows = col("grades_ingested").find({}, {"term.school_year":1,"term.semester":1}).limit(300000)
        else:
            rows = col("enrollments").find({}, {"term.school_year":1,"term.semester":1}).limit(300000)
        return sorted({term_label(r) for r in rows if term_label(r)})
    except Exception:
        return []

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar: controls
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Registrar Dashboard")

    # 1) Student Academic Standing section (single radio)
    with st.expander("Student Academic Standing Report", expanded=True):
        sas_choice = st.radio(
            "Choose view",
            ["GPA Reports", "Dean's List", "Probation"],
            index=["GPA Reports", "Dean's List", "Probation"].index(
                st.session_state.get("reg_sas_choice", "GPA Reports")
            ),
            label_visibility="collapsed",
        )
        st.session_state["reg_sas_choice"] = sas_choice

    # 2) Other reports section (checkboxes + Apply)
    with st.expander("Other Reports / Views", expanded=False):
        defaults = set(st.session_state.get("reg_other_sel", []))
        with st.form("reg_other_form"):
            c1 = st.checkbox("Subject Pass/Fail Distribution", value=("pass_fail" in defaults))
            c2 = st.checkbox("Enrollment Analysis", value=("enrollment" in defaults))
            c3 = st.checkbox("Incomplete Grades Report", value=("incomplete" in defaults))
            c4 = st.checkbox("Retention and Dropout Rates", value=("retention" in defaults))
            c5 = st.checkbox("Top Performers per Program", value=("top_perf" in defaults))
            applied = st.form_submit_button("Apply")
        if applied:
            chosen = []
            if c1: chosen.append("pass_fail")
            if c2: chosen.append("enrollment")
            if c3: chosen.append("incomplete")
            if c4: chosen.append("retention")
            if c5: chosen.append("top_perf")
            st.session_state["reg_other_sel"] = chosen
            st.success("Selections applied.")
            st.experimental_rerun()

# thresholds persist
deans_min = st.session_state.get("reg_deans_min", 90)
prob_max  = st.session_state.get("reg_prob_max", 75)

# ───────────────────────────────────────────────────────────────────────────────
# Filters (top of main area)
# ───────────────────────────────────────────────────────────────────────────────
with st.expander("Filters", expanded=False):
    term_opts = universe_terms()
    term_sel = st.multiselect("Term(s)", term_opts, default=term_opts)

    subj_filter = st.text_input("Subject code (exact or regex)", value=st.session_state.get("reg_subj", ""))
    dept_filter = st.text_input("Department (exact or regex)", value=st.session_state.get("reg_dept", ""))  # best-effort

    c1, c2 = st.columns(2)
    with c1:
        deans_min = st.number_input("Dean's List minimum GPA", min_value=0, max_value=100, value=int(deans_min), step=1)
    with c2:
        prob_max  = st.number_input("Probation maximum GPA", min_value=0, max_value=100, value=int(prob_max), step=1)

    # store
    st.session_state["reg_deans_min"] = deans_min
    st.session_state["reg_prob_max"]  = prob_max
    st.session_state["reg_subj"] = subj_filter
    st.session_state["reg_dept"] = dept_filter

# Common query filter from current filters
def q_from_filters():
    q = {}
    # term filter
    if term_sel:
        ors = []
        for t in term_sel:
            if " S" in t:
                sy, s = t.split(" S", 1)
                if SRC == "grades_ingested":
                    ors.append({"term.school_year": sy, "term.semester": int(s) if s.isdigit() else s})
                else:
                    ors.append({"term.school_year": sy, "term.semester": int(s) if s.isdigit() else s})
        if ors:
            q["$or"] = ors

    # subject & department filters (best effort)
    if subj_filter.strip():
        try:
            pat = re.compile(subj_filter.strip(), re.I)
            q[KEY_SUBJ] = {"$regex": pat}
        except re.error:
            q[KEY_SUBJ] = subj_filter.strip()

    if dept_filter.strip():
        # Try to match program_code or department-like field in enrollments
        if SRC == "enrollments":
            try:
                pat = re.compile(dept_filter.strip(), re.I)
                q["program.program_code"] = {"$regex": pat}
            except re.error:
                q["program.program_code"] = dept_filter.strip()
        # grades_ingested has no dept info normally; ignored
    return q

# ───────────────────────────────────────────────────────────────────────────────
# Computations
# ───────────────────────────────────────────────────────────────────────────────
def weighted_mean(grades, units):
    try:
        g = pd.Series(grades, dtype="float")
        u = pd.Series(units, dtype="float").fillna(0)
        if (u > 0).any():
            return (g * u).sum() / u.sum()
    except Exception:
        pass
    s = pd.Series(grades)
    return s.astype(float).mean() if not s.empty else float("nan")

def student_gpas():
    """Return DataFrame: student_id, GPA, Subjects (count)"""
    q = q_from_filters()
    if SRC == "grades_ingested":
        cur = col("grades_ingested").find(q, {"StudentID":1, KEY_GRADE:1, KEY_SUBJ:1})
        df = pd.DataFrame(list(cur))
        if df.empty:
            return pd.DataFrame(columns=["student_id","GPA","Subjects"])
        grp = df.groupby("StudentID")[KEY_GRADE].agg(["mean","count"]).reset_index()
        grp.columns = ["student_id","GPA","Subjects"]
        return grp
    else:
        cur = col("enrollments").find(q, {"student.student_no":1, "subject.units":1, KEY_GRADE:1})
        rows = [{"student_id": r.get("student",{}).get("student_no",""),
                 "grade": r.get(KEY_GRADE),
                 "units": (r.get("subject",{}).get("units") or 0)} for r in cur if r.get(KEY_GRADE) is not None]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["student_id","GPA","Subjects"])
        w = df.groupby("student_id").apply(lambda g: weighted_mean(g["grade"], g["units"])).reset_index(name="GPA")
        n = df.groupby("student_id")["grade"].count().reset_index(name="Subjects")
        out = pd.merge(w, n, on="student_id", how="left")
        return out

def program_map():
    """Best-effort map student_id -> program_code (from enrollments)."""
    m = {}
    try:
        cur = col("enrollments").find({}, {"student.student_no":1, "program.program_code":1}).limit(600000)
        for r in cur:
            sid = r.get("student",{}).get("student_no","")
            prog = r.get("program",{}).get("program_code","")
            if sid and prog:
                m[sid] = prog
    except Exception:
        pass
    return m

# ───────────────────────────────────────────────────────────────────────────────
# Report renderers
# ───────────────────────────────────────────────────────────────────────────────
def render_gpa_reports():
    st.subheader("Student Academic Standing → GPA Reports")
    df = student_gpas().sort_values("GPA", ascending=False)
    st.dataframe(df, use_container_width=True)
    if not df.empty:
        fig = plt.figure(); plt.hist(df["GPA"].dropna(), bins=30); plt.xlabel("GPA"); plt.ylabel("Count"); plt.title("GPA Distribution")
        st.pyplot(fig); fig_download_button(fig, "registrar_gpa_hist.png")
    st.caption("GPA is computed as weighted (when units exist), else simple mean of grades.")
    df_download_button(df, "registrar_gpa_reports.csv")

def render_deans_list():
    st.subheader("Student Academic Standing → Dean’s List")
    st.caption(f"Threshold: GPA ≥ {deans_min}")
    df = student_gpas()
    if df.empty:
        st.info("No data available.")
        return
    out = df[df["GPA"] >= float(deans_min)].copy()
    if out.empty:
        st.info("No students met the Dean’s List threshold. Try lowering the threshold or widening the term filter.")
        return
    out = out.sort_values(["GPA", "Subjects"], ascending=[False, False])
    st.dataframe(out, use_container_width=True)
    df_download_button(out, "registrar_deans_list.csv")

def render_probation():
    st.subheader("Student Academic Standing → Probation")
    st.caption(f"Threshold: GPA ≤ {prob_max}")
    df = student_gpas()
    if df.empty:
        st.info("No data available.")
        return
    out = df[df["GPA"] <= float(prob_max)].copy().sort_values(["GPA","Subjects"], ascending=[True, False])
    st.dataframe(out, use_container_width=True)
    df_download_button(out, "registrar_probation.csv")

def render_pass_fail():
    st.subheader("Subject Pass/Fail Distribution")
    q = q_from_filters()
    fields = {KEY_SUBJ:1, KEY_REMARK:1}
    cur = col(SRC).find(q, fields).limit(500000)
    rows = list(cur)
    if not rows:
        st.info("No rows found for current filters."); return
    df = pd.DataFrame(rows)
    df[KEY_REMARK] = df[KEY_REMARK].fillna("INC")
    summary = df.pivot_table(index=KEY_SUBJ, columns=KEY_REMARK, aggfunc="size", fill_value=0)
    st.dataframe(summary.reset_index().head(50), use_container_width=True)
    fig = plt.figure(); summary.sum().plot(kind="bar"); plt.title("Totals by Remark"); plt.xlabel("Remark"); plt.ylabel("Count")
    st.pyplot(fig)
    df_download_button(summary.reset_index(), "registrar_pass_fail_distribution.csv")
    fig_download_button(fig, "registrar_pass_fail_distribution.png")

def render_enrollment():
    st.subheader("Enrollment Analysis")
    q = q_from_filters()
    cur = col(SRC).find(q, {"term.school_year":1,"term.semester":1}).limit(700000)
    terms = pd.Series([term_label(r) for r in cur if term_label(r)])
    if terms.empty:
        st.info("No term data for current filters."); return
    trend = terms.value_counts().rename_axis("term").reset_index(name="count").sort_values("term")
    st.dataframe(trend, use_container_width=True)
    fig = plt.figure(); plt.plot(trend["term"], trend["count"], marker="o"); plt.xticks(rotation=45, ha="right")
    plt.title("Enrollment Trend by Term"); plt.xlabel("Term"); plt.ylabel("Count")
    st.pyplot(fig)
    df_download_button(trend, "registrar_enrollment_trend.csv")
    fig_download_button(fig, "registrar_enrollment_trend.png")

def render_incomplete():
    st.subheader("Incomplete Grades Report")
    q = q_from_filters()
    if SRC == "grades_ingested":
        cur = col("grades_ingested").find({"$and":[q, {"$or":[{KEY_REMARK:"INC"}, {KEY_GRADE: None}]}]},
                                          {"StudentID":1, KEY_SUBJ:1, "term":1}).limit(80000)
        df = pd.DataFrame([{"student_id": r.get("StudentID",""),
                            "subject_code": r.get(KEY_SUBJ,""),
                            "term": term_label(r)} for r in cur])
    else:
        cur = col("enrollments").find({"$and":[q, {KEY_REMARK:"INC"}]},
                                      {"student":1,"subject":1,"term":1}).limit(80000)
        df = pd.DataFrame([{"student_no": r.get("student",{}).get("student_no",""),
                            "subject_code": r.get("subject",{}).get("code",""),
                            "term": term_label(r)} for r in cur])
    if df.empty:
        st.info("No incomplete/missing grades found for current filters.")
        return
    st.dataframe(df.head(200), use_container_width=True)
    df_download_button(df, "registrar_incomplete_grades.csv")

def render_retention():
    st.subheader("Retention and Dropout Rates (Approx.)")
    q = q_from_filters()
    if SRC == "grades_ingested":
        cur = col("grades_ingested").find(q, {"StudentID":1, "term.school_year":1}).limit(900000)
        df = pd.DataFrame([{"student_id": r.get("StudentID",""),
                            "sy": r.get("term",{}).get("school_year","")} for r in cur])
    else:
        cur = col("enrollments").find(q, {"student.student_no":1, "term.school_year":1}).limit(900000)
        df = pd.DataFrame([{"student_id": r.get("student",{}).get("student_no",""),
                            "sy": r.get("term",{}).get("school_year","")} for r in cur])
    if df.empty:
        st.info("No data for retention view."); return
    cohort = df.dropna().groupby(["student_id","sy"]).size().reset_index(name="n")
    per_sy = cohort.groupby("sy")["student_id"].nunique().reset_index(name="unique_students").sort_values("sy")
    st.dataframe(per_sy, use_container_width=True)
    fig = plt.figure(); plt.plot(per_sy["sy"], per_sy["unique_students"], marker="o")
    plt.title("Unique Active Students by School Year"); plt.xlabel("School Year"); plt.ylabel("Students")
    st.pyplot(fig)
    df_download_button(per_sy, "registrar_retention_counts.csv")
    fig_download_button(fig, "registrar_retention_counts.png")

def render_top_performers():
    st.subheader("Top Performers per Program")
    top_n = st.number_input("Top N per program", min_value=1, max_value=100, value=10, step=1)
    gpas = student_gpas()
    if gpas.empty:
        st.info("No GPA data."); return
    pmap = program_map() if SRC == "enrollments" else {}
    gpas["program"] = gpas["student_id"].map(lambda x: pmap.get(x, "(Unknown)"))
    out = []
    for prog, block in gpas.groupby("program"):
        sub = block.sort_values(["GPA","Subjects"], ascending=[False, False]).head(int(top_n))
        out.append(sub)
    df = pd.concat(out).reset_index(drop=True) if out else pd.DataFrame(columns=gpas.columns)
    st.dataframe(df, use_container_width=True)
    df_download_button(df, "registrar_top_performers.csv")

# ───────────────────────────────────────────────────────────────────────────────
# Render selected views
# ───────────────────────────────────────────────────────────────────────────────
if sas_choice == "GPA Reports":
    render_gpa_reports()
elif sas_choice == "Dean's List":
    render_deans_list()
else:
    render_probation()

# Other reports selected via sidebar form
for key in st.session_state.get("reg_other_sel", []):
    st.markdown("---")
    if key == "pass_fail":   render_pass_fail()
    if key == "enrollment":  render_enrollment()
    if key == "incomplete":  render_incomplete()
    if key == "retention":   render_retention()
    if key == "top_perf":    render_top_performers()
