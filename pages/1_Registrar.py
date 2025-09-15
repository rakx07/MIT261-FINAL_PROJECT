import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from db import col

from utils import auth  # only for types; not needed if you already have guard_role in app
# from app import guard_role  # if guard_role is in app.py; or just paste it in a small common module

def guard_role(*roles):
    u = st.session_state.get("user")
    if not u:
        st.error("Please log in from the home page.")
        st.stop()
    if roles and u.get("role") not in roles:
        st.warning(f"Access restricted to: {', '.join(roles)}")
        st.stop()
    return u

user = guard_role("admin", "registrar")  # or "teacher", or "student"





st.set_page_config(page_title="Registrar Dashboard", page_icon="📊", layout="wide")

# Role guard
user = st.session_state.get("user")
if not user:
    st.error("Please login from the main page."); st.stop()
if user["role"] not in ("admin", "registrar"):
    st.warning("You need admin/registrar role to view this page."); st.stop()

st.title("📊 Registrar Dashboard")
st.caption("7 reports rendered from live data; use export buttons for your template.")

def df_download_button(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")

def fig_download_button(fig, filename, label="Download PNG"):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    st.download_button(label, buf.getvalue(), filename, "image/png")

def has_collection(name): 
    try: return name in col(name).database.list_collection_names()
    except Exception: return False

USE_INGESTED = has_collection("grades_ingested") and col("grades_ingested").estimated_document_count() > 0

# 1) Student Academic Standing Report
st.subheader("1) Student Academic Standing Report")
if USE_INGESTED:
    rows = list(col("grades_ingested").aggregate([
        {"$match": {"Grade": {"$ne": None}}},
        {"$group": {"_id": "$StudentID", "Average": {"$avg": "$Grade"}, "Subjects": {"$sum": 1}}},
        {"$sort": {"Average": -1}}, {"$limit": 50}
    ]))
    df1 = pd.DataFrame([{"student_id": r["_id"], "Average": round(r["Average"],2), "Subjects": r["Subjects"]} for r in rows])
else:
    rows = list(col("enrollments").aggregate([
        {"$match": {"grade": {"$exists": True}}},
        {"$group": {"_id": "$student.student_no", "name": {"$first": "$student.name"}, "Average": {"$avg": "$grade"}}},
        {"$sort": {"Average": -1}}, {"$limit": 50}
    ]))
    df1 = pd.DataFrame([{"student_id": r["_id"], "name": r.get("name",""), "Average": round(r["Average"],2)} for r in rows])
st.dataframe(df1, use_container_width=True)
df_download_button(df1, "registrar_academic_standing.csv")
st.text_area("Insight", value="Top students by GPA/average grade.", key="ins_reg1")

# 2) Subject Pass/Fail Distribution
st.subheader("2) Subject Pass/Fail Distribution")
if USE_INGESTED:
    rows = list(col("grades_ingested").find({}, {"SubjectCode":1, "Remark":1}).limit(500000))
    df2 = pd.DataFrame(rows); df2["Remark"] = df2["Remark"].fillna("INC")
    summary = df2.pivot_table(index="SubjectCode", columns="Remark", aggfunc="size", fill_value=0)
else:
    rows = list(col("enrollments").find({}, {"subject.code":1, "remark":1}).limit(200000))
    df2 = pd.DataFrame([{"SubjectCode": r.get("subject",{}).get("code",""), "Remark": r.get("remark","")} for r in rows])
    df2["Remark"] = df2["Remark"].fillna("INC"); summary = df2.pivot_table(index="SubjectCode", columns="Remark", aggfunc="size", fill_value=0)
st.dataframe(summary.reset_index().head(30), use_container_width=True)
fig2 = plt.figure(); summary.sum().plot(kind="bar"); plt.title("Totals by Remark"); plt.xlabel("Remark"); plt.ylabel("Count")
st.pyplot(fig2)
df_download_button(summary.reset_index(), "registrar_pass_fail_distribution.csv")
fig_download_button(fig2, "registrar_pass_fail_distribution.png")
st.text_area("Insight", value="Visual pass/fail/INC summary by subject.", key="ins_reg2")

# 3) Enrollment Trend Analysis
st.subheader("3) Enrollment Trend Analysis")
src = "grades_ingested" if USE_INGESTED else "enrollments"
rows = list(col(src).find({}, {"term.school_year":1,"term.semester":1}).limit(600000))
df3 = pd.DataFrame([{"term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}"} for r in rows])
trend = df3.value_counts("term").rename_axis("term").reset_index(name="count").sort_values("term")
st.dataframe(trend, use_container_width=True)
fig3 = plt.figure(); plt.plot(trend["term"], trend["count"], marker="o"); plt.xticks(rotation=45, ha="right")
plt.title("Enrollment Trend by Term"); plt.xlabel("Term"); plt.ylabel("Count")
st.pyplot(fig3)
df_download_button(trend, "registrar_enrollment_trend.csv")
fig_download_button(fig3, "registrar_enrollment_trend.png")
st.text_area("Insight", value="Semester-wise enrollment counts.", key="ins_reg3")

# 4) Incomplete Grades Report
st.subheader("4) Incomplete Grades Report")
if USE_INGESTED:
    rows = list(col("grades_ingested").find({"$or":[{"Remark":"INC"}, {"Grade": None}]}, {"StudentID":1,"SubjectCode":1,"term":1}).limit(50000))
    df4 = pd.DataFrame([{"student_id": r.get("StudentID",""), "subject_code": r.get("SubjectCode",""),
                         "term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}" } for r in rows])
else:
    rows = list(col("enrollments").find({"remark":"INC"}, {"student":1,"subject":1,"term":1}).limit(50000))
    df4 = pd.DataFrame([{"student_no": r.get("student",{}).get("student_no",""), "subject_code": r.get("subject",{}).get("code",""),
                         "term": f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}" } for r in rows])
st.dataframe(df4.head(200), use_container_width=True)
df_download_button(df4, "registrar_incomplete_grades.csv")
st.text_area("Insight", value="List of students with INC or missing grades.", key="ins_reg4")

# 5) Retention & Dropout (approximation via active students per year)
st.subheader("5) Retention and Dropout Rates (Approx.)")
rows = list(col(src).find({}, {"term.school_year":1, "student.student_no":1, "StudentID":1}).limit(800000))
if USE_INGESTED:
    df5 = pd.DataFrame([{"student_id": r.get("StudentID",""), "sy": r.get("term",{}).get("school_year","")} for r in rows])
else:
    df5 = pd.DataFrame([{"student_id": r.get("student",{}).get("student_no",""), "sy": r.get("term",{}).get("school_year","")} for r in rows])
cohort = df5.dropna().groupby(["student_id","sy"]).size().reset_index(name="n")
per_sy = cohort.groupby("sy")["student_id"].nunique().reset_index(name="unique_students").sort_values("sy")
st.dataframe(per_sy, use_container_width=True)
fig5 = plt.figure(); plt.plot(per_sy["sy"], per_sy["unique_students"], marker="o")
plt.title("Unique Active Students by School Year"); plt.xlabel("School Year"); plt.ylabel("Students")
st.pyplot(fig5)
df_download_button(per_sy, "registrar_retention_counts.csv")
fig_download_button(fig5, "registrar_retention_counts.png")
st.text_area("Insight", value="Approximate retention view via unique active students per school year.", key="ins_reg5")

# 6) Top Performers per Program (simplified overall top)
st.subheader("6) Top Performers per Program (Simplified Overall)")
if USE_INGESTED:
    rows = list(col("grades_ingested").aggregate([
        {"$match": {"Grade": {"$ne": None}}},
        {"$group": {"_id": "$StudentID", "Average": {"$avg": "$Grade"}, "N": {"$sum": 1}}},
        {"$sort": {"Average": -1}}, {"$limit": 30}
    ]))
    df6 = pd.DataFrame([{"student_id": r["_id"], "Average": round(r["Average"],2), "Subjects": r["N"]} for r in rows])
else:
    rows = list(col("enrollments").aggregate([
        {"$match": {"grade": {"$exists": True}}},
        {"$group": {"_id": "$student.student_no", "Average": {"$avg": "$grade"}}},
        {"$sort": {"Average": -1}}, {"$limit": 30}
    ]))
    df6 = pd.DataFrame([{"student_id": r["_id"], "Average": round(r["Average"],2)} for r in rows])
st.dataframe(df6, use_container_width=True)
df_download_button(df6, "registrar_top_performers.csv")
st.text_area("Insight", value="Overall top performers. Map to program via your student→program schema if available.", key="ins_reg6")

# 7) Curriculum Progress and Advising
st.subheader("7) Curriculum Progress and Advising")
cur = col("curriculum").find_one({}, {"subjects":1})
if cur and isinstance(cur.get("subjects"), list):
    all_codes = sorted({s.get("subjectCode") for s in cur["subjects"] if s.get("subjectCode")})
    student_id = st.text_input("StudentID / email (for preview):", value=df1.iloc[0]["student_id"] if not df1.empty else "")
    completed = set()
    if student_id:
        if USE_INGESTED:
            q = {"StudentID": student_id, "Grade": {"$ne": None}}
            rows = list(col("grades_ingested").find(q, {"SubjectCode":1,"Grade":1}).limit(5000))
            completed = {r.get("SubjectCode") for r in rows if r.get("Grade",0) >= 75}
        else:
            q = {"student.student_no": student_id, "grade": {"$exists": True}}
            rows = list(col("enrollments").find(q, {"subject.code":1,"grade":1}).limit(5000))
            completed = {r.get("subject",{}).get("code") for r in rows if r.get("grade",0) >= 75}
    remaining = [c for c in all_codes if c not in completed]
    c1, c2 = st.columns(2)
    with c1:
        dfc = pd.DataFrame({"Completed": sorted(list(completed))})
        st.dataframe(dfc, use_container_width=True, height=260)
        df_download_button(dfc, "registrar_curriculum_completed.csv")
    with c2:
        dfr = pd.DataFrame({"Remaining": remaining})
        st.dataframe(dfr, use_container_width=True, height=260)
        df_download_button(dfr, "registrar_curriculum_remaining.csv")
else:
    st.info("No curriculum doc found with a 'subjects' array.")
st.text_area("Insight", value="Completed vs remaining subjects for advising.", key="ins_reg7")
