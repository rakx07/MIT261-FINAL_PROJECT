# pages/1_Registrar.py
from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from db import col

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _terms_options() -> List[Tuple[str, int]]:
    pipe = [
        {"$group": {"_id": {"sy": "$term.school_year", "sem": "$term.semester"}}},
        {"$project": {"_id": 0, "sy": "$_id.sy", "sem": "$_id.sem"}},
        {"$sort": {"sy": 1, "sem": 1}},
    ]
    rows = list(col("enrollments").aggregate(pipe))
    out = []
    for r in rows:
        sy, sem = r.get("sy"), r.get("sem")
        if sy and sem is not None:
            out.append((sy, int(sem)))
    return out

def _program_options() -> List[str]:
    pipe = [
        {"$group": {"_id": "$program.program_code"}},
        {"$project": {"_id": 0, "code": "$_id"}},
        {"$sort": {"code": 1}},
    ]
    codes = [r["code"] for r in col("enrollments").aggregate(pipe) if r.get("code")]
    # Keep BSED-ENGLISH first if it exists
    codes = sorted(codes, key=lambda c: (c != "BSED-ENGLISH", c))
    return codes

def _build_query(
    terms: List[Tuple[str, int]] | None,
    subj_regex: str | None,
    dept_regex: str | None,
    program_codes: List[str] | None,
) -> Dict[str, Any]:
    ands: List[Dict[str, Any]] = []
    if terms:
        ors = [{"term.school_year": sy, "term.semester": sem} for sy, sem in terms]
        ands.append({"$or": ors})
    if subj_regex:
        ands.append({"subject.code": {"$regex": subj_regex, "$options": "i"}})
    if dept_regex:
        ands.append({"subject.code": {"$regex": f"^{re.escape(dept_regex)}", "$options": "i"}})
    if program_codes:
        ands.append({"program.program_code": {"$in": program_codes}})
    return {"$and": ands} if ands else {}

def _load_enrollments_df(q: Dict[str, Any]) -> pd.DataFrame:
    fields = {
        "student.student_no": 1,
        "student.name": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "program.program_code": 1,
        "subject.code": 1,
        "subject.title": 1,
        "subject.units": 1,
        "subject.year_level": 1,
        "teacher.name": 1,
        "grade": 1,
        "_id": 0,
    }
    rows = list(col("enrollments").find(q, fields))
    if not rows:
        return pd.DataFrame(
            columns=[
                "student_no","student_name",
                "school_year","semester","program",
                "subject_code","subject_title","units","year_level",
                "teacher_name","grade",
            ]
        )
    df = pd.DataFrame(rows)
    # Flatten
    df["student_no"]   = df["student"].apply(lambda s: s.get("student_no") if isinstance(s, dict) else None)
    df["student_name"] = df["student"].apply(lambda s: s.get("name") if isinstance(s, dict) else None)
    df["school_year"]  = df["term"].apply(lambda t: t.get("school_year") if isinstance(t, dict) else None)
    df["semester"]     = df["term"].apply(lambda t: t.get("semester") if isinstance(t, dict) else None)
    df["program"]      = df["program"].apply(lambda p: p.get("program_code") if isinstance(p, dict) else None)
    df["subject_code"] = df["subject"].apply(lambda s: s.get("code") if isinstance(s, dict) else None)
    df["subject_title"]= df["subject"].apply(lambda s: s.get("title") if isinstance(s, dict) else None)
    df["units"]        = df["subject"].apply(lambda s: s.get("units") if isinstance(s, dict) else None)
    df["year_level"]   = df["subject"].apply(lambda s: s.get("year_level") if isinstance(s, dict) else None)
    df["teacher_name"] = df["teacher"].apply(lambda t: t.get("name") if isinstance(t, dict) else None)
    df = df.drop(columns=["student","term","program","subject","teacher"], errors="ignore")
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["semester"] = pd.to_numeric(df["semester"], errors="coerce")
    return df

# ---------------------------------------------------------------------
# Standing Reports
# ---------------------------------------------------------------------
def render_gpa_report(df: pd.DataFrame):
    st.subheader("GPA Reports")
    if df.empty:
        st.info("No rows matched your filter.")
        return
    g = (
        df.dropna(subset=["grade"])
          .groupby("student_no")
          .agg(student_id=("student_name","first"), GPA=("grade","mean"), Subjects=("grade","count"))
          .reset_index()
          .sort_values(["GPA","Subjects"], ascending=[False,False])
    )
    g["GPA"] = g["GPA"].round(2)
    st.dataframe(g, use_container_width=True)
    st.caption("GPA = mean of grades over the selected scope.")

def render_deans_list(df: pd.DataFrame, min_gpa: float):
    st.subheader("Student Academic Standing → Dean’s List")
    st.caption(f"Threshold: GPA ≥ {min_gpa}")
    if df.empty:
        st.info("No rows matched your filter.")
        return
    g = (
        df.dropna(subset=["grade"])
          .groupby(["student_no","school_year","semester"])
          .agg(student_id=("student_name","first"), GPA=("grade","mean"), Subjects=("grade","count"))
          .reset_index()
    )
    out = g[g["GPA"] >= float(min_gpa)].sort_values(["GPA","Subjects"], ascending=[False,False])
    out["GPA"] = out["GPA"].round(2)
    if out.empty:
        st.info("No students met the Dean’s List threshold. Try lowering the threshold or widening the term filter.")
    else:
        st.dataframe(out, use_container_width=True)

def render_probation(df: pd.DataFrame, max_gpa: float):
    st.subheader("Student Academic Standing → Probation")
    st.caption(f"Threshold: GPA ≤ {max_gpa}")
    if df.empty:
        st.info("No rows matched your filter.")
        return
    g = (
        df.dropna(subset=["grade"])
          .groupby(["student_no","school_year","semester"])
          .agg(student_id=("student_name","first"), GPA=("grade","mean"), Subjects=("grade","count"))
          .reset_index()
    )
    out = g[g["GPA"] <= float(max_gpa)].sort_values(["GPA","Subjects"], ascending=[True,False])
    out["GPA"] = out["GPA"].round(2)
    if out.empty:
        st.info("No students fell under the probation threshold for the selected scope.")
    else:
        st.dataframe(out, use_container_width=True)

# ---------------------------------------------------------------------
# Other Reports (old section but with Apply)
# ---------------------------------------------------------------------
def render_other_reports_panel() -> Dict[str,bool]:
    st.write("### Other Reports / Views")
    if "other_reports_sel" not in st.session_state:
        st.session_state.other_reports_sel = {"passfail":False,"enrollment":False,"incomplete":False,"retention":False,"toppers":False}
    with st.expander("Configure & run", expanded=False):
        with st.form("other_reports_form", clear_on_submit=False):
            s = st.session_state.other_reports_sel
            s["passfail"]   = st.checkbox("Subject Pass/Fail Distribution", value=s["passfail"])
            s["enrollment"] = st.checkbox("Enrollment Analysis", value=s["enrollment"])
            s["incomplete"] = st.checkbox("Incomplete Grades Report", value=s["incomplete"])
            s["retention"]  = st.checkbox("Retention and Dropout Rates", value=s["retention"])
            s["toppers"]    = st.checkbox("Top Performers per Program", value=s["toppers"])
            c1,c2,_ = st.columns([1,1,3])
            run = c1.form_submit_button("Apply")
            if c2.form_submit_button("Clear"):
                for k in s: s[k]=False
                run=True
        if run:
            st.session_state.other_reports_applied=True
            st.toast("Other reports updated.", icon="👍")
    return st.session_state.other_reports_sel if st.session_state.get("other_reports_applied") else {}

def _render_other_reports(df: pd.DataFrame, sel: Dict[str,bool]):
    if not sel:
        return
    if df.empty:
        st.info("No rows matched your filter.")
        return

    if sel.get("passfail"):
        st.markdown("#### Subject Pass/Fail Distribution")
        tmp = df.dropna(subset=["grade"]).copy()
        tmp["status"] = (tmp["grade"] >= 75).map({True:"PASSED", False:"FAILED"})
        g = (
            tmp.groupby(["subject_code","status"])
               .size().unstack(fill_value=0).reset_index()
               .sort_values("subject_code")
        )
        st.dataframe(g, use_container_width=True)

    if sel.get("enrollment"):
        st.markdown("#### Enrollment Analysis (by Subject)")
        g = (
            df.groupby(["school_year","semester","subject_code"])
              .agg(Students=("student_no","nunique")).reset_index()
              .sort_values(["school_year","semester","Students"], ascending=[True,True,False])
        )
        st.dataframe(g, use_container_width=True)

    if sel.get("incomplete"):
        st.markdown("#### Incomplete Grades Report")
        inc = df[df["grade"].isna()].copy()
        if inc.empty:
            st.info("No INC rows in this scope.")
        else:
            st.dataframe(inc[["student_no","student_name","school_year","semester","subject_code"]], use_container_width=True)

    if sel.get("retention"):
        st.markdown("#### Retention and Dropout Rates (naïve view)")
        g = df.groupby(["student_no","school_year","semester"]).size().rename("subjects").reset_index()
        nxt = g.copy()
        nxt["semester"] = nxt["semester"].astype(int)
        nxt["key"] = nxt["student_no"] + "|" + nxt["school_year"] + "|" + (nxt["semester"]-1).astype(str)
        g["key"]  = g["student_no"] + "|" + g["school_year"] + "|" + g["semester"].astype(str)
        cont = set(nxt["key"])
        g["continued_from_prev_sem"] = g["key"].isin(cont)
        rates = g.groupby(["school_year","semester"]).agg(Students=("student_no","nunique"), Continued=("continued_from_prev_sem","sum")).reset_index()
        rates["RetentionRate(%)"] = (rates["Continued"]/rates["Students"]*100).round(1)
        st.dataframe(rates, use_container_width=True)

    if sel.get("toppers"):
        st.markdown("#### Top Performers per Program")
        g = (
            df.dropna(subset=["grade"])
              .groupby(["program","student_no"])
              .agg(student_id=("student_name","first"), GPA=("grade","mean"), Subjects=("grade","count"))
              .reset_index()
        )
        g["GPA"] = g["GPA"].round(2)
        for prog, block in g.groupby("program"):
            st.markdown(f"**Program: {prog or '(Unknown)'}**")
            top = block.sort_values(["GPA","Subjects"], ascending=[False,False]).head(10)
            st.dataframe(top, use_container_width=True)

# ---------------------------------------------------------------------
# Student Performance Analytics (original 4)
# ---------------------------------------------------------------------
def render_student_performance_analytics(df: pd.DataFrame):
    st.write("### Student Performance Analytics")

    if df.empty:
        st.info("No rows matched your filter.")
        return

    with st.expander("Top Performers", expanded=False):
        style = st.radio("Ranking scope", ["Per semester", "Per school year", "Overall"], horizontal=True, key="spa_scope")
        if style == "Per semester":       keys = ["student_no","school_year","semester"]
        elif style == "Per school year":   keys = ["student_no","school_year"]
        else:                              keys = ["student_no"]

        g = (
            df.dropna(subset=["grade"])
              .groupby(keys)
              .agg(student_id=("student_name","first"),
                   program=("program","first"),
                   GPA=("grade","mean"),
                   Subjects=("grade","count"))
              .reset_index()
              .sort_values(["GPA","Subjects"], ascending=[False,False])
              .head(10)
        )
        g["GPA"] = g["GPA"].round(2)
        st.dataframe(g, use_container_width=True)

    with st.expander("Failing Students (>30% failed)", expanded=False):
        x = df.copy()
        x["failed"] = (x["grade"] < 75).astype(int)
        x["taken"]  = (~x["grade"].isna()).astype(int)
        r = x.groupby("student_no").agg(student_id=("student_name","first"), failed=("failed","sum"), taken=("taken","sum")).reset_index()
        r["rate"] = (r["failed"]/r["taken"]).fillna(0)
        bad = r[(r["taken"]>0) & (r["rate"]>0.30)].copy()
        bad["rate(%)"] = (bad["rate"]*100).round(1)
        bad = bad.sort_values(["rate","failed"], ascending=False)
        st.dataframe(bad[["student_no","student_id","failed","taken","rate(%)"]], use_container_width=True)

    with st.expander("Students with Grade Improvement", expanded=False):
        t = (
            df.dropna(subset=["grade"])
              .groupby(["student_no","school_year","semester"])
              .agg(student_id=("student_name","first"), GPA=("grade","mean"))
              .reset_index()
        )
        if t.empty:
            st.info("Not enough graded data.")
        else:
            term_key = t["school_year"].str.extract(r"(\d{4})").astype(float)[0] + t["semester"].astype(float)/10.0
            t = t.assign(term_key=term_key)
            first = t.sort_values("term_key").groupby("student_no").first()
            last  = t.sort_values("term_key").groupby("student_no").last()
            imp = (last["GPA"] - first["GPA"]).rename("delta").to_frame()
            imp["student_id"] = last["student_id"]
            imp["from"] = first["GPA"].round(2)
            imp["to"]   = last["GPA"].round(2)
            improved = imp[imp["delta"]>0].sort_values("delta", ascending=False).head(20).reset_index()
            improved["delta"] = improved["delta"].round(2)
            st.dataframe(improved[["student_no","student_id","from","to","delta"]], use_container_width=True)
            st.caption("Δ = last-term GPA − first-term GPA (within selected scope).")

    with st.expander("Distribution of Grades (60–69, 70–79, 80–89, 90–100)", expanded=False):
        vals = df["grade"].dropna()
        if vals.empty:
            st.info("No numeric grades to plot.")
        else:
            fig, ax = plt.subplots()
            ax.hist(vals, bins=[60,70,80,90,100])
            ax.set_xlabel("Grade"); ax.set_ylabel("Frequency"); ax.set_title("Histogram of Grades")
            st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------
# Advanced Analytics (NEW side menu)
# ---------------------------------------------------------------------
def _sidebar_advanced_menu() -> Dict[str, Any]:
    st.sidebar.markdown("**Advanced Analytics**")
    if "adv_cfg" not in st.session_state:
        st.session_state.adv_cfg = {
            # Subject & Teacher
            "hardest": False, "easiest": False, "avg_teacher": False, "fail_by_teacher_sem": False,
            # Course & Curriculum
            "trend_course": False, "load_intensity": False, "ge_vs_major": False,
            # Sem/AY
            "sem_worst": False, "sem_best": False, "variance": False,
            # Demographics
            "yl_dist": False, "count_per_course": False, "perf_by_yl": False,
            # Aux
            "top_n": 10, "major_regex": "ENG", "ge_regex": "^(?!ENG)",  # default: ENG=Major, non-ENG=GE
        }
    with st.sidebar.form("adv_form"):
        st.caption("Select analytics, then click Apply.")
        s = st.session_state.adv_cfg
        st.subheader("Subject & Teacher")
        s["hardest"] = st.checkbox("Hardest Subjects (highest fail rate)", value=s["hardest"])
        s["easiest"] = st.checkbox("Easiest Subjects (≥90 share)", value=s["easiest"])
        s["avg_teacher"] = st.checkbox("Average grades per teacher", value=s["avg_teacher"])
        s["fail_by_teacher_sem"] = st.checkbox("Teachers with most failures per semester", value=s["fail_by_teacher_sem"])

        st.subheader("Course & Curriculum")
        s["trend_course"] = st.checkbox("Grade trends per course (by school year)", value=s["trend_course"])
        s["load_intensity"] = st.checkbox("Subject load intensity (avg units per student)", value=s["load_intensity"])
        s["ge_vs_major"] = st.checkbox("GE vs Major performance (by regex)", value=s["ge_vs_major"])
        with st.expander("GE/Major regex (advanced)", expanded=False):
            s["major_regex"] = st.text_input("Major subject code matches (regex)", value=s["major_regex"])
            s["ge_regex"]    = st.text_input("GE subject code matches (regex)", value=s["ge_regex"])

        st.subheader("Semester & AY")
        s["sem_worst"] = st.checkbox("Semester(s) with lowest average grade", value=s["sem_worst"])
        s["sem_best"]  = st.checkbox("Best performing semester(s)", value=s["sem_best"])
        s["variance"]  = st.checkbox("Grade deviation across semesters (high variance subjects)", value=s["variance"])

        st.subheader("Demographics & Behavior")
        s["yl_dist"] = st.checkbox("Year level distribution", value=s["yl_dist"])
        s["count_per_course"] = st.checkbox("Student count per course (selected programs)", value=s["count_per_course"])
        s["perf_by_yl"] = st.checkbox("Performance by year level", value=s["perf_by_yl"])

        s["top_n"] = st.number_input("Top N (for lists/charts)", min_value=5, max_value=100, value=s["top_n"])

        cols = st.columns([1,1])
        run  = cols[0].form_submit_button("Apply")
        clear= cols[1].form_submit_button("Clear")
        if clear:
            for k in list(s.keys()):
                if isinstance(s[k], bool): s[k] = False
            run = True
    if run:
        st.session_state.adv_applied = True
        st.toast("Advanced analytics updated.", icon="📊")
    return st.session_state.adv_cfg if st.session_state.get("adv_applied") else {}

def _render_advanced(df: pd.DataFrame, cfg: Dict[str, Any]):
    if not cfg: return
    if df.empty:
        st.info("No rows matched your filter.")
        return
    N = int(cfg.get("top_n", 10))

    # ----- Subject & Teacher Analytics -----
    if cfg.get("hardest"):
        st.markdown("#### Hardest Subjects (highest failure %)")
        x = df.dropna(subset=["grade"]).copy()
        x["fail"] = (x["grade"] < 75).astype(int)
        t = x.groupby("subject_code").agg(Failed=("fail","sum"), Taken=("grade","count")).reset_index()
        t["Fail%(%)"] = (t["Failed"]/t["Taken"]*100).round(1)
        t = t[t["Taken"]>=10].sort_values("Fail%(%)", ascending=False).head(N)  # ignore tiny classes
        st.dataframe(t, use_container_width=True)

    if cfg.get("easiest"):
        st.markdown("#### Easiest Subjects (share of grades ≥ 90)")
        x = df.dropna(subset=["grade"]).copy()
        x["ge90"] = (x["grade"] >= 90).astype(int)
        t = x.groupby("subject_code").agg(AtLeast90=("ge90","sum"), Taken=("grade","count")).reset_index()
        t["Share≥90(%)"] = (t["AtLeast90"]/t["Taken"]*100).round(1)
        t = t[t["Taken"]>=10].sort_values("Share≥90(%)", ascending=False).head(N)
        st.dataframe(t, use_container_width=True)

    if cfg.get("avg_teacher"):
        st.markdown("#### Average Grades per Teacher")
        x = df.dropna(subset=["grade"]).copy()
        t = x.groupby("teacher_name").agg(GPA=("grade","mean"), Subjects=("grade","count")).reset_index()
        t["GPA"] = t["GPA"].round(2)
        t = t.sort_values(["GPA","Subjects"], ascending=[False,False]).head(N)
        st.dataframe(t, use_container_width=True)

    if cfg.get("fail_by_teacher_sem"):
        st.markdown("#### Teachers with Highest Failures per Semester")
        x = df.dropna(subset=["grade"]).copy()
        x["fail"] = (x["grade"] < 75).astype(int)
        t = x.groupby(["teacher_name","school_year","semester"]).agg(Failed=("fail","sum"), Taken=("grade","count")).reset_index()
        t["Fail%(%)"] = (t["Failed"]/t["Taken"]*100).round(1)
        t = t.sort_values(["Failed","Fail%(%)"], ascending=[False,False]).head(N)
        st.dataframe(t, use_container_width=True)

    # ----- Course & Curriculum Insights -----
    if cfg.get("trend_course"):
        st.markdown("#### Grade Trends per Course (by school year)")
        x = df.dropna(subset=["grade"]).copy()
        x["year_num"] = x["school_year"].str.extract(r"(\d{4})").astype(float)[0]
        t = x.groupby(["program","year_num"]).agg(GPA=("grade","mean")).reset_index()
        if t.empty:
            st.info("Not enough data.")
        else:
            for prog, block in t.groupby("program"):
                st.markdown(f"**{prog}**")
                fig, ax = plt.subplots()
                ax.plot(block["year_num"], block["GPA"])
                ax.set_xlabel("School year (start)"); ax.set_ylabel("Average grade"); ax.set_title(f"Trend — {prog}")
                st.pyplot(fig, clear_figure=True)

    if cfg.get("load_intensity"):
        st.markdown("#### Subject Load Intensity (average units per student per semester)")
        x = df.copy()
        by_stud_term = x.groupby(["program","student_no","school_year","semester"]).agg(Units=("units","sum")).reset_index()
        t = by_stud_term.groupby(["program","school_year","semester"]).agg(AvgUnits=("Units","mean"), Students=("student_no","nunique")).reset_index()
        t["AvgUnits"] = t["AvgUnits"].round(2)
        st.dataframe(t.sort_values(["program","school_year","semester"]), use_container_width=True)

    if cfg.get("ge_vs_major"):
        st.markdown("#### General Education vs Major Subjects Performance")
        major_re = re.compile(cfg.get("major_regex","ENG"), re.I)
        ge_re    = re.compile(cfg.get("ge_regex","^(?!ENG)"), re.I)
        x = df.dropna(subset=["grade"]).copy()
        def cat(code:str) -> str:
            if code and major_re.search(code): return "MAJOR"
            if code and ge_re.search(code):    return "GE"
            return "OTHER"
        x["Cat"] = x["subject_code"].apply(cat)
        t = x[x["Cat"].isin(["MAJOR","GE"])].groupby(["program","Cat"]).agg(GPA=("grade","mean"), Subjects=("grade","count")).reset_index()
        t["GPA"] = t["GPA"].round(2)
        st.dataframe(t, use_container_width=True)

    # ----- Semester & AY Analysis -----
    if cfg.get("sem_worst") or cfg.get("sem_best"):
        sem = (
            df.dropna(subset=["grade"])
              .groupby(["school_year","semester"])
              .agg(GPA=("grade","mean"), Subjects=("grade","count")).reset_index()
        )
        sem["GPA"] = sem["GPA"].round(2)
        if cfg.get("sem_worst"):
            st.markdown("#### Semester(s) with Lowest Average Grade")
            st.dataframe(sem.sort_values("GPA").head(N), use_container_width=True)
        if cfg.get("sem_best"):
            st.markdown("#### Best Performing Semester(s)")
            st.dataframe(sem.sort_values("GPA", ascending=False).head(N), use_container_width=True)

    if cfg.get("variance"):
        st.markdown("#### Grade Deviation Across Semesters (high variance subjects)")
        x = (
            df.dropna(subset=["grade"])
              .groupby(["subject_code","school_year","semester"])
              .agg(GPA=("grade","mean")).reset_index()
        )
        if x.empty:
            st.info("Not enough data.")
        else:
            v = x.groupby("subject_code")["GPA"].std(ddof=0).rename("StdDev").reset_index()
            v = v.sort_values("StdDev", ascending=False).head(N)
            v["StdDev"] = v["StdDev"].round(2)
            st.dataframe(v, use_container_width=True)

    # ----- Demographics & Behavior -----
    if cfg.get("yl_dist"):
        st.markdown("#### Year Level Distribution (unique students per level)")
        x = df.copy()
        t = x.groupby("year_level")["student_no"].nunique().rename("Students").reset_index()
        st.dataframe(t.sort_values("year_level"), use_container_width=True)

    if cfg.get("count_per_course"):
        st.markdown("#### Student Count per Course (selected programs)")
        t = df.groupby("program")["student_no"].nunique().rename("Students").reset_index()
        st.dataframe(t.sort_values("Students", ascending=False), use_container_width=True)

    if cfg.get("perf_by_yl"):
        st.markdown("#### Performance by Year Level (average grade)")
        x = df.dropna(subset=["grade"]).copy()
        t = x.groupby("year_level")["grade"].mean().rename("GPA").reset_index()
        t["GPA"] = t["GPA"].round(2)
        st.dataframe(t.sort_values("year_level"), use_container_width=True)

# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
def main():
    st.title("Registrar Dashboard")

    # ---------------- Filters ---------------
    with st.expander("Filters", expanded=True):
        # program selection
        programs = _program_options()
        default = ["BSED-ENGLISH"] if "BSED-ENGLISH" in programs else programs[:1]
        sel_programs = st.multiselect("Program(s)", programs, default=default, help="Focus BSED-ENGLISH or pick other courses.")

        # term selection
        all_terms = _terms_options()
        term_labels = [f"{sy} • S{sem}" for sy, sem in all_terms]
        sel_idx = st.multiselect("Term(s)", options=list(range(len(all_terms))), format_func=lambda i: term_labels[i])
        terms = [all_terms[i] for i in sel_idx] if sel_idx else None

        subj_regex = st.text_input("Subject code (exact or regex)", value="").strip() or None
        dept_regex = st.text_input("Department (exact or regex)", help="Use a subject code prefix e.g., 'ENG' or 'ITX'.").strip() or None

        c1, c2 = st.columns(2)
        min_deans = c1.number_input("Dean's List minimum GPA", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        max_prob  = c2.number_input("Probation maximum GPA",   min_value=0.0, max_value=100.0, value=75.0, step=1.0)

    # one query for the whole page
    q = _build_query(terms, subj_regex, dept_regex, sel_programs or None)
    df = _load_enrollments_df(q)

    # -------------- Sidebar menus --------------
    st.sidebar.markdown("## Registrar")
    standing = st.sidebar.radio("Student Academic Standing Report", ["GPA Reports","Dean's List","Probation"], index=1)
    other_sel = render_other_reports_panel()
    adv_cfg = _sidebar_advanced_menu()

    # -------------- Standing block --------------
    if standing == "GPA Reports":
        render_gpa_report(df)
    elif standing == "Dean's List":
        render_deans_list(df, min_deans)
    else:
        render_probation(df, max_prob)

    # -------------- Other Reports --------------
    _render_other_reports(df, other_sel)

    # -------------- Student Performance Analytics (original 4) --------------
    st.divider()
    render_student_performance_analytics(df)

    # -------------- Advanced Analytics (new) --------------
    st.divider()
    st.header("Advanced Analytics")
    _render_advanced(df, adv_cfg)

if __name__ == "__main__":
    main()
