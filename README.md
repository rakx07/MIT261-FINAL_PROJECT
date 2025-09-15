# MIT261 Analytics (Streamlit) — Starter

## Quickstart
1) Create and activate a venv
2) `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and set:
   - `MONGODB_URI` (Atlas or local)
   - `DB_NAME` (e.g., mit261)
   - `APP_SECRET` (random string for session)
4) `streamlit run app.py`

## Login
- Username = **email** (student/teacher/admin)
- Password = set in `users` collection (hashed with bcrypt). Use Admin → Users to create/import from teachers/students.

## Collections expected
- students, teachers, subjects, semesters, programs, classes, enrollments, curriculum, colleges, universities
- users (for app logins): {email, role: 'admin'|'teacher'|'student', name, password_hash, active, ref_id?}
