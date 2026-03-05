"""Optional Streamlit demo for interactive research runs."""

from __future__ import annotations

import streamlit as st

from orchestrator import run_agent

st.set_page_config(page_title="Multi-Agent Web Researcher", layout="wide")
st.title("Multi-Agent Web Researcher")

query = st.text_area("Ask an open-domain research question", height=120)

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Configuration", ["planner_only", "planner_browser", "full"], index=2)
with col2:
    enable_multihop = st.checkbox("Enable multi-hop reasoning", value=True)

use_memory = st.checkbox("Use cross-query vector memory", value=True)

col3, col4, col5 = st.columns(3)
with col3:
    memory_top_k = st.slider("Memory retrieval top-k", min_value=1, max_value=5, value=3)
with col4:
    memory_min_score = st.slider(
        "Memory similarity threshold",
        min_value=0.0,
        max_value=0.9,
        value=0.2,
        step=0.05,
    )
with col5:
    reflection_threshold = st.slider("Reflection threshold (1-10)", min_value=1, max_value=10, value=7)

if st.button("Run", type="primary") and query.strip():
    with st.spinner("Running agents..."):
        answer, review = run_agent(
            query.strip(),
            mode=mode,
            use_memory=use_memory,
            memory_top_k=memory_top_k,
            memory_min_score=memory_min_score,
            reflection_threshold=reflection_threshold,
            enable_multihop=enable_multihop,
        )

    st.subheader("Final Answer")
    st.write(answer)
    
    # Show multi-hop reasoning details if enabled
    if review.get("multihop_enabled"):
        st.subheader("Multi-Hop Reasoning Details")
        with st.expander("View reasoning chain"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Difficulty Level:**", review.get("difficulty_analysis", {}).get("difficulty_level", "N/A"))
                st.write("**Sub-Questions:**")
                for i, sub_q in enumerate(review.get("sub_questions", []), 1):
                    st.write(f"{i}. {sub_q}")
            with col2:
                st.write("**Reasoning Chain:**")
                st.write(review.get("difficulty_analysis", {}).get("reasoning_chain", "N/A"))
            
            chain_val = review.get("chain_validation", {})
            if chain_val:
                st.write("**Chain Validation:**")
                st.write(f"- Valid: {chain_val.get('is_valid', 'Unknown')}")
                st.write(f"- Confidence: {chain_val.get('confidence', 0):.2f}")
                if chain_val.get("gaps"):
                    st.write("- Gaps:", ", ".join(chain_val.get("gaps", [])))

    st.subheader("Critic Review")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{review.get('score', 'N/A')}/10")
    with col2:
        st.metric("Sources Used", review.get("num_sources", 0))
    with col3:
        st.metric("Reflections", review.get("num_reflections", 0))
    
    st.write("**Review:**", review.get("review", "N/A"))
    
    with st.expander("View full review metadata"):
        st.json(review)

