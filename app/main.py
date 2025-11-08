import streamlit as st

st.set_page_config(
    page_title="CTL Search",
    layout="wide",
)

pages = {
    "Navigation": [
        st.Page("pages/01_search_results_comparison.py", title="Search Results Comparison"),
        st.Page("pages/02_search_metrics.py", title="Search Metrics"),
        st.Page("pages/03_relevance_scores.py", title="Relevance Scores"),
    ],
}

nav = st.navigation(pages, position="sidebar")
nav.run()
