# pages/3_Student.py
import io
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from db import col


def guard_role(*roles):
    u = st.session_state.get("user")
    if not u:
        st.error("Please log in from the home page."); st.stop()
    if roles and u.get("role") not in roles:
        st.warning(f"Access restricted to: {', '.join(roles)}"); st.stop()
    return u

user = guard_role("admin", "student", "registrar", "teacher")

st.title("🎒 Student Dashboard")

# ── helpers / source detection
def has_collection(name):
    try: return name in col(name).database.list_collection_names()
    except Exception: return False

def nonempty_count(name):
    try: return (col(name).estimated_document_count() or 0) > 0
    except Exception: return False

MODE = "ingested"
if has_collection("enrollments") and nonempty_count("enrollments"):
    MODE = "enrollments"
elif has_collection("grades") and nonempty_count("grades"):
    MODE = "grades"

def weighted_mean(grades, units):
    try:
        if units is not None and len(units) and pd.Series(units).notna().any():
            g = pd.Series(grades).astype(float)
            u = pd.Series(units).fillna(0).astype(float)
            if (u > 0).any():
                return (g * u).sum() / u.sum()
    except Exception:
        pass
    s = pd.Series(grades)
    if s.empty: return float("nan")
    return s.astype(float).mean()

# ───────────────────────────────────────────────────────────────────────────────
# Roster loading with role restrictions
# ───────────────────────────────────────────────────────────────────────────────
def roster_for_registrar_or_admin():
    out = []
    try:
        # users(role=student) first
        for u in col("users").find({"role":"student"}, {"email":1,"name":1}).sort("email", 1).limit(5000):
            out.append({"email": u.get("email",""), "name": u.get("name",""), "student_no": ""})
        # fallback to students
        if not out:
            for sdoc in col("students").find({}, {"Email":1,"email":1,"Name":1,"name":1,"student_no":1}).limit(10000):
                out.append({
                    "email": sdoc.get("Email") or sdoc.get("email") or "",
                    "name":  sdoc.get("Name")  or sdoc.get("name")  or "",
                    "student_no": sdoc.get("student_no") or ""
                })
    except Exception:
        pass
    return out

def roster_for_teacher(teacher_email: str):
    """
    Pull unique students taught by this teacher from enrollments.
    If enrollments aren’t available, returns empty (teacher view will show message).
    """
    out = []
    if not (has_collection("enrollments") and nonempty_count("enrollments")):
        return out
    seen = set()
    rows = col("enrollments").find({"teacher.email": teacher_email},
                                   {"student.email":1,"student.name":1,"student.student_no":1}).limit(300000)
    for r in rows:
        s = r.get("student",{})
        email = (s.get("email") or "").strip()
        name  = (s.get("name") or "").strip()
        sno   = (s.get("student_no") or "").strip()
        key = (email or sno or name).lower()
        if key and key not in seen:
            out.append({"email": email, "name": name, "student_no": sno})
            seen.add(key)
    return sorted(out, key=lambda x: (x["email"], x["name"], x["student_no"]))

# ───────────────────────────────────────────────────────────────────────────────
# Student selection with role logic
# ───────────────────────────────────────────────────────────────────────────────
if user["role"] == "student":
    chosen = {"email": user.get("email","").strip(), "name": "", "student_no": ""}
    st.caption(f"Active data source: **{MODE}** · Locked to: **{chosen['email']}**")

elif user["role"] == "teacher":
    t_email = user.get("email","").strip()
    roster = roster_for_teacher(t_email)
    if not roster:
        st.warning("No enrollments found for your classes. Ask the registrar to populate the enrollments collection.")
        st.stop()
    labels = [f"{r['email'] or r['student_no']} — {r['name']}" for r in roster]
    sel = st.selectbox("Select your student", labels, index=0)
    r = roster[labels.index(sel)]
    chosen = r
    st.caption(f"Active data source: **{MODE}** · Teacher scope: **{t_email}**")

else:  # registrar or admin
    st.info("Registrar/Admin: choose a student to view the reports below.")
    roster = roster_for_registrar_or_admin()
    labels = [f"{r['email']} — {r['name']}" if r["email"] else (r["name"] or r["student_no"]) for r in roster]
    sel = st.selectbox("Select student", labels, index=0 if labels else None, placeholder="email — name")
    manual = st.text_input("…or enter email or name manually", value="")
    if manual.strip():
        m = manual.strip()
        if "—" in m:  # clean pasted label
            m = m.split("—", 1)[0].strip()
        match = next((r for r in roster if r["email"] == m or r["name"] == m or r["student_no"] == m), None)
        chosen = match or {"email": m if "@" in m else "", "name": "" if "@" in m else m, "student_no": ""}
    else:
        chosen = roster[labels.index(sel)] if labels else {"email":"","name":"","student_no":""}
    st.caption(f"Active data source: **{MODE}**")

# ───────────────────────────────────────────────────────────────────────────────
# Term helpers
# ───────────────────────────────────────────────────────────────────────────────
def semester_lookup():
    lkp = {}
    if has_collection("semesters") and nonempty_count("semesters"):
        for s in col("semesters").find({}, {"_id":1,"school_year":1,"semester":1,"SchoolYear":1,"Semester":1,"label":1}).limit(100000):
            sid = s.get("_id")
            sy  = s.get("school_year") or s.get("SchoolYear")
            sem = s.get("semester") or s.get("Semester")
            label = s.get("label") or (f"{sy} S{sem}" if sy and sem else f"Sem {sid}")
            lkp[sid] = label; lkp[str(sid)] = label
    return lkp

SEM_LKP = semester_lookup()

def term_label(doc) -> str:
    if MODE == "enrollments":
        return f"{doc.get('term',{}).get('school_year','')} S{doc.get('term',{}).get('semester','')}".strip()
    elif MODE == "grades":
        sid = doc.get("SemesterID")
        return SEM_LKP.get(sid) or SEM_LKP.get(str(sid)) or (f"Sem {sid}" if sid is not None else "")
    else:
        return ""

# ───────────────────────────────────────────────────────────────────────────────
# Build query keys based on MODE
# ───────────────────────────────────────────────────────────────────────────────
def candidate_keys(ch):
    email = (ch.get("email") or "").strip().lower()
    name  = (ch.get("name") or "").strip()
    sno   = (ch.get("student_no") or "").strip()
    if MODE == "enrollments":
        return {"$or": [{"student.email": email}] if email else []} | \
               {"$or": ([{"student.student_no": sno}] if sno else [])} | \
               {"$or": ([{"student.name": name}] if name else [])}
    elif MODE == "grades":
        ors = []
        if email: ors.append({"StudentID": email})
        if sno:   ors.append({"StudentID": sno})
        if name:  ors.append({"StudentID": name})
        return {"$or": ors} if ors else {}
    else:
        ors = []
        if name:  ors.append({"StudentID": name})
        if email: ors.append({"StudentID": email})
        return {"$or": ors} if ors else {}

def find_terms_for_student():
    q = candidate_keys(chosen)
    if MODE == "enrollments" and q:
        rows = col("enrollments").find(q, {"term.school_year":1,"term.semester":1}).limit(200000)
        return sorted({f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}" for r in rows})
    elif MODE == "grades" and q:
        rows = col("grades").find(q, {"SemesterID":1}).limit(200000)
        return sorted({SEM_LKP.get(r.get("SemesterID")) or SEM_LKP.get(str(r.get("SemesterID"))) or f"Sem {r.get('SemesterID')}" for r in rows})
    return []

with st.expander("Optional filters (Term)"):
    terms = find_terms_for_student()
    term_sel = st.multiselect("Filter by term(s)", terms, default=terms)

def term_selected(doc) -> bool:
    if not term_sel: return True
    lbl = term_label(doc)
    return lbl in term_sel if lbl else False

# ───────────────────────────────────────────────────────────────────────────────
# Load transcript
# ───────────────────────────────────────────────────────────────────────────────
def load_transcript_df():
    q = candidate_keys(chosen)
    if not q:
        return pd.DataFrame(columns=["subject_code","semester","school_year","grade","remark"])

    if MODE == "enrollments":
        rows = list(col("enrollments").find(q, {"subject":1,"grade":1,"remark":1,"term":1}).limit(200000))
        rows = [r for r in rows if term_selected(r)]
        return pd.DataFrame([{
            "subject_code": r.get("subject",{}).get("code",""),
            "semester":     r.get("term",{}).get("semester",""),
            "school_year":  r.get("term",{}).get("school_year",""),
            "grade":        r.get("grade"),
            "remark":       r.get("remark") or ("PASSED" if (r.get("grade") or 0) >= 75 else "FAILED")
        } for r in rows])

    elif MODE == "grades":
        rows = list(col("grades").find(q, {"SubjectCodes":1,"Grades":1,"SemesterID":1}).limit(200000))
        out = []
        for r in rows:
            lbl = term_label(r)
            if term_sel and lbl not in term_sel: continue
            codes = r.get("SubjectCodes") or []
            grades = r.get("Grades") or []
            for c, g in zip(codes, grades):
                out.append({
                    "subject_code": c, "semester": lbl.split(" S")[-1] if " S" in lbl else "",
                    "school_year": lbl.split(" S")[0] if " S" in lbl else "",
                    "grade": g, "remark": ("PASSED" if (g or 0) >= 75 else "FAILED") if g is not None else "INC"
                })
        return pd.DataFrame(out)

    else:
        rows = list(col("grades_ingested").find({"$or":[{"StudentID": chosen.get("name","")}, {"StudentID": chosen.get("email","")}]},
                                                {"SubjectCode":1,"Grade":1,"Remark":1}).limit(200000))
        return pd.DataFrame([{
            "subject_code": r.get("SubjectCode",""),
            "semester": "", "school_year": "",
            "grade": r.get("Grade"),
            "remark": r.get("Remark") or ("PASSED" if (r.get("Grade") or 0) >= 75 else "FAILED")
        } for r in rows])

def df_download_button(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")

def fig_download_button(fig, filename, label="Download PNG"):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    st.download_button(label, buf.getvalue(), filename, "image/png")

df_all = load_transcript_df()
if df_all.empty:
    st.warning("No records found for this student (and term filter).")
    st.stop()

# 1) Transcript viewer
st.subheader("1) Academic Transcript Viewer")
st.dataframe(df_all.sort_values(["school_year","semester","subject_code"]), use_container_width=True, height=300)
df_download_button(df_all, "student_transcript.csv")

# 2) Performance Trend Over Time
st.subheader("2) Performance Trend Over Time")
tmp = df_all.dropna(subset=["grade"]).copy()
tmp["term"] = tmp["school_year"].astype(str) + " S" + tmp["semester"].astype(str)
trend = tmp.groupby("term", as_index=False)["grade"].mean().sort_values("term")
if not trend.empty:
    st.dataframe(trend, use_container_width=True)
    fig2 = plt.figure(); plt.plot(trend["term"], trend["grade"], marker="o")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Average Grade"); plt.title("Average Grade by Term")
    st.pyplot(fig2); df_download_button(trend, "student_trend.csv"); fig_download_button(fig2, "student_trend.png")
else:
    st.info("Not enough data to plot a trend.")

# 3) Subject Difficulty Ratings
st.subheader("3) Subject Difficulty Ratings")
df3 = df_all.copy(); df3["passed"] = df3["grade"].fillna(0) >= 75
by_subj = df3.groupby("subject_code")["passed"].mean().reset_index().rename(columns={"passed":"my_pass_rate"})
st.dataframe(by_subj.sort_values("my_pass_rate"), use_container_width=True)
df_download_button(by_subj, "student_subject_difficulty.csv")

# 4) Comparison with Class Average
st.subheader("4) Comparison with Class Average")
if MODE == "enrollments":
    all_rows = list(col("enrollments").find({"grade":{"$exists": True}}, {"subject.code":1,"grade":1}).limit(500000))
    df_all_class = pd.DataFrame([{"subject_code": r.get("subject",{}).get("code",""), "grade": r.get("grade")}
                                 for r in all_rows if r.get("grade") is not None])
elif MODE == "grades":
    all_rows = list(col("grades").find({}, {"SubjectCodes":1,"Grades":1}).limit(200000))
    pairs = []
    for r in all_rows:
        codes = r.get("SubjectCodes") or []
        grades = r.get("Grades") or []
        for c, g in zip(codes, grades):
            if g is not None: pairs.append({"subject_code": c, "grade": g})
    df_all_class = pd.DataFrame(pairs)
else:
    all_rows = list(col("grades_ingested").find({"Grade":{"$ne": None}}, {"SubjectCode":1,"Grade":1}).limit(500000))
    df_all_class = pd.DataFrame([{"subject_code": r.get("SubjectCode",""), "grade": r.get("Grade")} for r in all_rows])

my_avg = df_all.dropna(subset=["grade"]).groupby("subject_code")["grade"].mean().reset_index().rename(columns={"grade":"my_avg"})
class_avg = df_all_class.groupby("subject_code")["grade"].mean().reset_index().rename(columns={"grade":"class_avg"})
merged = pd.merge(my_avg, class_avg, on="subject_code", how="inner")
if not merged.empty:
    merged["delta"] = (merged["my_avg"] - merged["class_avg"]).round(2)
    st.dataframe(merged.sort_values("delta"), use_container_width=True)
    df_download_button(merged, "student_vs_class_avg.csv")
else:
    st.info("Not enough overlap to compare with class averages.")

# 5) Passed vs Failed Summary
st.subheader("5) Passed vs Failed Summary")
counts = df_all["remark"].fillna("INC").value_counts().reindex(["PASSED","FAILED","INC"], fill_value=0)
c1, c2 = st.columns(2)
with c1: st.dataframe(counts.rename("count"))
with c2:
    fig5 = plt.figure(); plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%"); st.pyplot(fig5)
    fig_download_button(fig5, "student_pass_fail.png")

# 6) Curriculum
st.subheader("6) Curriculum and Subject Viewer")
cur = None
try: cur = col("curriculum").find_one({}, {"subjects":1})
except Exception: cur = None
if cur and isinstance(cur.get("subjects"), list):
    all_codes = sorted({s.get("subjectCode") for s in cur["subjects"] if s.get("subjectCode")})
    completed = set(df_all.loc[df_all["grade"].fillna(0) >= 75, "subject_code"])
    remaining = [c for c in all_codes if c not in completed]
    c1, c2 = st.columns(2)
    with c1:
        dfc = pd.DataFrame({"subject_code": sorted(list(completed))})
        st.dataframe(dfc, use_container_width=True, height=250); df_download_button(dfc, "student_curriculum_completed.csv")
    with c2:
        dfr = pd.DataFrame({"subject_code": remaining})
        st.dataframe(dfr, use_container_width=True, height=250); df_download_button(dfr, "student_curriculum_remaining.csv")
else:
    st.info("No curriculum data detected.")
