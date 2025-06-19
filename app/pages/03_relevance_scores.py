import polars as pl
import streamlit as st

from app.relevance import (
    load_query_df,
    calculate_relevance_df,
    DEFAULT_VIEW_WEIGHT,
    DEFAULT_HOLD_WEIGHT,
    DEFAULT_MAX_I,
    DEFAULT_ALPHA,
)


if "relevance_scores_i" not in st.session_state:
    st.session_state["relevance_scores_i"] = 0


st.write("### relevance scores")

query = st.text_input("query")

if not query:
    st.stop()

with st.expander("relevance parameters"):
    if st.button("reset to defaults"):
        st.session_state["relevance_scores_i"] += 1
    c1, c2, c3, c4 = st.columns(4)
    view_weight = c1.number_input("view weight", value=float(DEFAULT_VIEW_WEIGHT), min_value=0.0, step=0.5,
                                  key=f"view_weight_{st.session_state['relevance_scores_i']}")
    hold_weight = c2.number_input("hold weight", value=float(DEFAULT_HOLD_WEIGHT), min_value=0.0, step=0.5,
                                  key=f"hold_weight_{st.session_state['relevance_scores_i']}")
    alpha = c3.number_input("alpha", value=DEFAULT_ALPHA, min_value=0.0, step=0.1,
                            key=f"alpha_{st.session_state['relevance_scores_i']}")
    max_i = c4.number_input("max i", value=DEFAULT_MAX_I, min_value=1,
                            key=f"max_i_{st.session_state['relevance_scores_i']}")

query_df = load_query_df().filter(pl.col("query") == query)

relevance_df = calculate_relevance_df(
    query_df,
    view_weight=view_weight,
    hold_weight=hold_weight,
    alpha=alpha,
    max_i=max_i
)

st.write("### results")
st.write(
    relevance_df
    .select(
        "item_name",
        "median_index",
        "view_count",
        "hold_count",
        "view_plus_hold_count",
        "normalized_score",
        "rank",
    )
)
