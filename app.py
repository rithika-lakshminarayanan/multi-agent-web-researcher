"""Optional Streamlit demo for interactive research runs."""

from __future__ import annotations

import streamlit as st

from orchestrator import run_agent

st.set_page_config(page_title="Multi-Agent Web Researcher", layout="wide")
st.title("Multi-Agent Web Researcher")

query = st.text_area("Ask an open-domain research question", height=120)
mode = st.selectbox("Configuration", ["planner_only", "planner_browser", "full"], index=2)

if st.button("Run") and query.strip():
    with st.spinner("Running agents..."):
        answer, review = run_agent(query.strip(), mode=mode)

    st.subheader("Final Answer")
    st.write(answer)

    st.subheader("Critic Review")
    st.json(review)
