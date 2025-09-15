# pages/3_Student.py
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






st.set_page_config(page_title="Student Dashboard", page_icon="🎒", layout="wide")

# ── Role guard
user = st.session_state.get("user")
if not user:
    st.error("Please login from the main page."); st.stop()
if user["role"] not in ("admin", "student", "registrar"):
    st.warning("You need student/registrar/admin role to view this page."); st.stop()

st.title("🎒 Student Dashboard")

# ── Helpers
def df_download_button(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")

def fig_download_button(fig, filename, label="Download PNG"):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    st.download_button(label, buf.getvalue(), filename, "image/png")

def has_collection(name):
    try: return name in col(name).database.list_collection_names()
    except Exception: return False

USE_INGESTED = has_collection("grades_ingested") and col("grades_ingested").estimated_document_count() > 0

# ───────────────────────────────────────────────────────────────────────────────
# Student selector:
# - student role → locked to own email
# - registrar/admin → can choose any student (from users.role=student, fallback to students.email)
# ───────────────────────────────────────────────────────────────────────────────
if user["role"] == "student":
    target_email = user.get("email")
    st.caption(f"Student email: **{target_email}**")
else:
    st.info("Registrar/Admin: choose a student to view the reports below.")
    # Try users (role=student) first
    options = list(col("users").find({"role": "student"}, {"email": 1, "name": 1}).sort("email", 1).limit(2000))
    if not options:
        # Fallback to students collection
        options = list(col("students").find({}, {"email": 1, "name": 1}).sort("email", 1).limit(2000))
    choices = [f"{o.get('email','')} — {o.get('name', '')}" for o in options if o.get("email")]
    preset = choices[0] if choices else ""
    selected = st.selectbox("Select student", choices, index=0 if choices else None, placeholder="email — name")
    manual = st.text_input("…or enter email manually", value="")
    target_email = (manual.strip() or (selected.split(" — ")[0] if selected else "")).lower()
    if not target_email:
        st.stop()

# Optional term filter for registrar/admin (narrow the report if desired)
with st.expander("Optional filters (Term)"):
    if USE_INGESTED:
        terms_rows = col("grades_ingested").find(
            {"StudentID": target_email}, {"term.school_year": 1, "term.semester": 1}
        )
    else:
        terms_rows = col("enrollments").find(
            {"student.email": target_email}, {"term.school_year": 1, "term.semester": 1}
        )
    terms = sorted({f"{r.get('term',{}).get('school_year','')} S{r.get('term',{}).get('semester','')}"
                    for r in terms_rows})
    term_sel = st.multiselect("Filter by term(s)", terms, default=terms)

def term_match(doc) -> bool:
    if not term_sel: 
        return True
    t = f"{doc.get('term',{}).get('school_year','')} S{doc.get('term',{}).get('semester','')}"
    return t in term_sel

# ───────────────────────────────────────────────────────────────────────────────
# Load transcript rows for the chosen student
# ───────────────────────────────────────────────────────────────────────────────
if USE_INGESTED:
    rows = list(col("grades_ingested").find(
        {"StudentID": target_email}, {"SubjectCode":1,"Grade":1,"Remark":1,"term":1}
    ).limit(200000))
    rows = [r for r in rows if term_match(r)]
    df_all = pd.DataFrame([{
        "subject_code": r.get("SubjectCode",""),
        "semester": r.get("term",{}).get("semester",""),
        "school_year": r.get("term",{}).get("school_year",""),
        "grade": r.get("Grade"),
        "remark": r.get("Remark") or ("PASSED" if (r.get("Grade") or 0) >= 75 else "FAILED")
    } for r in rows])
else:
    rows = list(col("enrollments").find(
        {"student.email": target_email}, {"subject":1,"grade":1,"remark":1,"term":1}
    ).limit(200000))
    rows = [r for r in rows if term_match(r)]
    df_all = pd.DataFrame([{
        "subject_code": r.get("subject",{}).get("code",""),
        "semester": r.get("term",{}).get("semester",""),
        "school_year": r.get("term",{}).get("school_year",""),
        "grade": r.get("grade"),
        "remark": r.get("remark") or ("PASSED" if (r.get("grade") or 0) >= 75 else "FAILED")
    } for r in rows])

if df_all.empty:
    st.warning("No records found for this student (and term filter).")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# 1) Academic Transcript Viewer
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("1) Academic Transcript Viewer")
st.dataframe(df_all.sort_values(["school_year","semester","subject_code"]),
             use_container_width=True, height=300)
df_download_button(df_all, "student_transcript.csv")

# ───────────────────────────────────────────────────────────────────────────────
# 2) Performance Trend Over Time
# ───────────────────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────────────────
# 3) Subject Difficulty Ratings (personal pass rate per subject)
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("3) Subject Difficulty Ratings")
df3 = df_all.copy(); df3["passed"] = df3["grade"].fillna(0) >= 75
by_subj = df3.groupby("subject_code")["passed"].mean().reset_index().rename(columns={"passed":"my_pass_rate"})
st.dataframe(by_subj.sort_values("my_pass_rate"), use_container_width=True)
df_download_button(by_subj, "student_subject_difficulty.csv")
st.text_area("Insight", value="Lower pass-rate subjects are relatively harder for the student.", key="stu3")

# ───────────────────────────────────────────────────────────────────────────────
# 4) Comparison with Class Average
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("4) Comparison with Class Average")
if USE_INGESTED:
    all_rows = list(col("grades_ingested").find({"Grade":{"$ne": None}}, {"SubjectCode":1,"Grade":1}).limit(500000))
    df_all_class = pd.DataFrame([{"subject_code": r.get("SubjectCode",""), "grade": r.get("Grade")}
                                 for r in all_rows if r.get("Grade") is not None])
else:
    all_rows = list(col("enrollments").find({"grade":{"$exists": True}}, {"subject.code":1,"grade":1}).limit(500000))
    df_all_class = pd.DataFrame([{"subject_code": r.get("subject",{}).get("code",""), "grade": r.get("grade")}
                                 for r in all_rows if r.get("grade") is not None])
my_avg = df_all.dropna(subset=["grade"]).groupby("subject_code")["grade"].mean().reset_index().rename(columns={"grade":"my_avg"})
class_avg = df_all_class.groupby("subject_code")["grade"].mean().reset_index().rename(columns={"grade":"class_avg"})
merged = pd.merge(my_avg, class_avg, on="subject_code", how="inner")
if not merged.empty:
    merged["delta"] = (merged["my_avg"] - merged["class_avg"]).round(2)
    st.dataframe(merged.sort_values("delta"), use_container_width=True)
    df_download_button(merged, "student_vs_class_avg.csv")
    st.text_area("Insight", value="Positive delta = above class average; negative = below.", key="stu4")
else:
    st.info("Not enough overlap to compare with class averages.")

# ───────────────────────────────────────────────────────────────────────────────
# 5) Passed vs Failed Summary
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("5) Passed vs Failed Summary")
counts = df_all["remark"].fillna("INC").value_counts().reindex(["PASSED","FAILED","INC"], fill_value=0)
c1, c2 = st.columns(2)
with c1: st.dataframe(counts.rename("count"))
with c2:
    fig5 = plt.figure(); plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%"); st.pyplot(fig5)
    fig_download_button(fig5, "student_pass_fail.png")
st.text_area("Insight", value=f"PASSED={int(counts.get('PASSED',0))}, FAILED={int(counts.get('FAILED',0))}, "
                              f"INC={int(counts.get('INC',0))}.", key="stu5")

# ───────────────────────────────────────────────────────────────────────────────
# 6) Curriculum and Subject Viewer
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("6) Curriculum and Subject Viewer")
cur = col("curriculum").find_one({}, {"subjects":1})
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
