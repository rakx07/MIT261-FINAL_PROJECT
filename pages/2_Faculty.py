# pages/2_Faculty.py
import io
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

user = guard_role("admin", "teacher", "registrar")

st.title("🧑‍🏫 Faculty Dashboard")
st.caption("Teacher scope applies automatically. Registrars/Admins can filter by teacher.")

def df_download_button(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")

def fig_download_button(fig, filename, label="Download PNG"):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    st.download_button(label, buf.getvalue(), filename, "image/png")

def has_collection(name): 
    try: return name in col(name).database.list_collection_names()
    except Exception: return False

USE_INGESTED = has_collection("grades_ingested") and col("grades_ingested").estimated_document_count() > 0

# ── Scope
teacher_email = None
if user["role"] == "teacher":
    teacher_email = user.get("email")

elif user["role"] in ("registrar", "admin"):
    # Build a list of teachers from enrollments (preferred)
    emails = []
    try:
        emails = sorted({r.get("teacher", {}).get("email","")
                         for r in col("enrollments").find({}, {"teacher.email":1}).limit(300000)
                         if r.get("teacher",{}).get("email")})
    except Exception:
        pass
    # fallback to users with role=teacher if enrollments lookup fails
    if not emails:
        try:
            emails = sorted({u.get("email","")
                             for u in col("users").find({"role":"teacher"}, {"email":1}).limit(5000)
                             if u.get("email")})
        except Exception:
            pass

    choice = st.selectbox("Filter by teacher (Registrar/Admin)", ["(All teachers)"] + emails, index=0)
    teacher_email = None if choice == "(All teachers)" else choice

# Build query with scope
def scoped_query(extra=None):
    extra = extra or {}
    if user["role"] == "teacher":
        return {"$and": [{"teacher.email": teacher_email}, extra] if extra else [{"teacher.email": teacher_email}]}
    if user["role"] in ("registrar", "admin") and teacher_email:
        return {"$and": [{"teacher.email": teacher_email}, extra] if extra else [{"teacher.email": teacher_email}]}
    return extra or {}

# 1) Class Grade Distribution (Histogram)
st.subheader("1) Class Grade Distribution (Histogram)")
q = scoped_query()
if USE_INGESTED:
    rows = list(col("grades_ingested").find(q, {"Grade":1}).limit(300000))
    df1 = pd.DataFrame([r for r in rows if r.get("Grade") is not None])
else:
    rows = list(col("enrollments").find(q, {"grade":1}).limit(300000))
    df1 = pd.DataFrame([{"Grade": r.get("grade")} for r in rows if r.get("grade") is not None])
if not df1.empty:
    fig1 = plt.figure(); plt.hist(df1["Grade"], bins=40); plt.xlabel("Grade"); plt.ylabel("Freq"); plt.title("Grade Distribution")
    st.pyplot(fig1); fig_download_button(fig1, "faculty_grade_histogram.png")
    st.text_area("Insight", value=f"{len(df1)} graded entries. Mean={df1['Grade'].mean():.2f}, Median={df1['Grade'].median():.2f}.", key="fac1")
else:
    st.info("No graded entries found for this scope.")

# 2) Student Progress Tracker (Avg by Term)
st.subheader("2) Student Progress Tracker (Avg by Term)")
if USE_INGESTED:
    rows = list(col("grades_ingested").find(q, {"Grade":1,"term":1}).limit(300000))
    df2 = pd.DataFrame([{"term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}",
                         "Grade": r.get("Grade")} for r in rows if r.get("Grade") is not None])
else:
    rows = list(col("enrollments").find(q, {"grade":1,"term":1}).limit(300000))
    df2 = pd.DataFrame([{"term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}",
                         "Grade": r.get("grade")} for r in rows if r.get("grade") is not None])
if not df2.empty:
    trend = df2.groupby("term", as_index=False).Grade.mean().sort_values("term")
    st.dataframe(trend, use_container_width=True)
    fig2 = plt.figure(); plt.plot(trend["term"], trend["Grade"], marker="o"); plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg Grade"); plt.title("Average Grade by Term")
    st.pyplot(fig2); df_download_button(trend, "faculty_progress_trend.csv"); fig_download_button(fig2, "faculty_progress_trend.png")
else:
    st.info("No data to compute term averages.")

# 3) Subject Difficulty Heatmap (Fail %)
st.subheader("3) Subject Difficulty Heatmap (Fail %)")
if USE_INGESTED:
    rows = list(col("grades_ingested").find(q, {"SubjectCode":1,"Remark":1,"term":1}).limit(300000))
    df3 = pd.DataFrame([{"Subject": r.get("SubjectCode",""),
                         "term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}",
                         "is_fail": 1 if r.get("Remark")=="FAILED" else 0} for r in rows])
else:
    rows = list(col("enrollments").find(q, {"subject.code":1,"remark":1,"term":1}).limit(300000))
    df3 = pd.DataFrame([{"Subject": r.get("subject",{}).get("code",""),
                         "term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}",
                         "is_fail": 1 if r.get("remark")=="FAILED" else 0} for r in rows])
if not df3.empty:
    grid = df3.groupby(["Subject","term"])["is_fail"].mean().reset_index()
    st.dataframe(grid.pivot(index="Subject", columns="term", values="is_fail").fillna(0).round(2), use_container_width=True)
    st.text_area("Insight", value="Higher values indicate higher failure rates.", key="fac3")
else:
    st.info("No data for fail rates.")

# 4) Intervention Candidates (grade <75 or INC)
st.subheader("4) Intervention Candidates")
if USE_INGESTED:
    q2 = scoped_query({"$or":[{"Grade":{"$lt":75}}, {"Remark":"INC"}]})
    rows = list(col("grades_ingested").find(q2, {"StudentID":1,"SubjectCode":1,"Grade":1,"Remark":1,"term":1}).limit(20000))
    df4 = pd.DataFrame([{"student_id": r.get("StudentID",""), "subject": r.get("SubjectCode",""),
                         "term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}",
                         "grade": r.get("Grade"), "remark": r.get("Remark")} for r in rows])
else:
    q2 = scoped_query({"$or":[{"grade":{"$lt":75}}, {"remark":"INC"}]})
    rows = list(col("enrollments").find(q2, {"student":1,"subject":1,"grade":1,"remark":1,"term":1}).limit(20000))
    df4 = pd.DataFrame([{"student": r.get("student",{}).get("student_no",""), "subject": r.get("subject",{}).get("code",""),
                         "term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}",
                         "grade": r.get("grade"), "remark": r.get("remark")} for r in rows])
st.dataframe(df4, use_container_width=True, height=300); df_download_button(df4, "faculty_intervention.csv")
st.text_area("Insight", value="Students at risk: grade < 75 or INC.", key="fac4")

# 5) Grade Submission Status
st.subheader("5) Grade Submission Status")
if USE_INGESTED:
    rows = list(col("grades_ingested").find(q, {"SubjectCode":1,"Grade":1}).limit(300000))
    df5 = pd.DataFrame(rows)
else:
    rows = list(col("enrollments").find(q, {"subject.code":1,"grade":1}).limit(300000))
    df5 = pd.DataFrame([{"SubjectCode": r.get("subject",{}).get("code",""), "Grade": r.get("grade")} for r in rows])
status = df5.groupby("SubjectCode")["Grade"].apply(lambda s: (s.notna().sum(), s.size)).reset_index()
status[["with_grade","total"]] = pd.DataFrame(status["Grade"].tolist(), index=status.index)
status["pct_with_grade"] = (status["with_grade"]/status["total"]*100).round(2)
status = status.drop(columns=["Grade"]).sort_values("pct_with_grade", ascending=False)
st.dataframe(status.head(50), use_container_width=True); df_download_button(status, "faculty_submission_status.csv")
st.text_area("Insight", value="Percent of records with grades per subject.", key="fac5")

# 6) Custom Query Builder
st.subheader("6) Custom Query Builder")
with st.form("custom_query"):
    subj = st.text_input("Subject Code (exact)", "")
    min_g = st.number_input("Min grade", 0, 100, 0)
    max_g = st.number_input("Max grade", 0, 100, 100)
    submit = st.form_submit_button("Run")
if submit:
    src = "grades_ingested" if USE_INGESTED else "enrollments"
    qf = scoped_query({})
    if subj.strip():
        qf["SubjectCode" if USE_INGESTED else "subject.code"] = subj.strip()
    key_grade = "Grade" if USE_INGESTED else "grade"
    qf[key_grade] = {"$gte": int(min_g), "$lte": int(max_g)}
    res = list(col(src).find(qf).limit(2000))
    dfr = pd.DataFrame(res); st.dataframe(dfr, use_container_width=True); df_download_button(dfr, "faculty_custom_query.csv")
    st.text_area("Insight", value="Ad-hoc filtered results.", key="fac6")

# 7) Students Grade Analytics (Per Teacher)
st.subheader("7) Students Grade Analytics (Per Teacher)")
if USE_INGESTED:
    rows = list(col("grades_ingested").find(q, {"teacher.email":1,"Grade":1}).limit(400000))
    df7 = pd.DataFrame([{"teacher": r.get("teacher",{}).get("email",""), "Grade": r.get("Grade")} for r in rows if r.get("Grade") is not None])
else:
    rows = list(col("enrollments").find(q, {"teacher.email":1,"grade":1}).limit(400000))
    df7 = pd.DataFrame([{"teacher": r.get("teacher",{}).get("email",""), "Grade": r.get("grade")} for r in rows if r.get("grade") is not None])
if not df7.empty:
    stats = df7.groupby("teacher")["Grade"].agg(["count","mean","median","min","max"]).round(2).reset_index()
    st.dataframe(stats, use_container_width=True); df_download_button(stats, "faculty_teacher_analytics.csv")
    st.text_area("Insight", value="Per-teacher grade stats (compare distributions).", key="fac7")
else:
    st.info("No graded entries found.")
