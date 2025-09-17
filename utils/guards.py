import streamlit as st

def guard_role(*roles):
    u = st.session_state.get("user")
    if not u:
        st.error("Please log in from the home page.")
        st.stop()
    if roles and u.get("role") not in roles:
        st.warning(f"Access restricted to: {', '.join(roles)}")
        st.stop()
    return u
