from db import col

def ensure_indexes():
    # users
    col("users").create_index("email", unique=True)
    col("users").create_index([("role",1), ("active",1)])

    # grades_ingested (if you use it)
    g = col("grades_ingested")
    g.create_index("StudentID")
    g.create_index("SubjectCode")
    g.create_index("teacher.email")
    g.create_index([("term.school_year",1), ("term.semester",1)])

    # enrollments (fallback schema)
    e = col("enrollments")
    e.create_index("student.student_no")
    e.create_index("student.email")
    e.create_index("subject.code")
    e.create_index("teacher.email")
    e.create_index([("term.school_year",1), ("term.semester",1)])
