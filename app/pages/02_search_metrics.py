from itertools import cycle

import streamlit as st

from app.elastic import get_es_client, get_default_elasticsearch_weights
from app.relevance import run_metrics, get_top_n_queries
from app.semantic import SemanticSearch, DEFAULT_EMBEDDING_MODEL, SUGGESTED_MODELS
from app.sql import get_default_pg_search_weights

st.title("Search Metrics")

if "i" not in st.session_state:
    st.session_state["i"] = 0

# take in comma separated list of query terms
default_terms = ", ".join(get_top_n_queries(20))
query_terms = st.text_input("query terms", value=default_terms)
query_terms = query_terms.split(",")
query_terms = [x.strip() for x in query_terms]
query_terms = [x for x in query_terms if x != ""]

if not query_terms:
    st.stop()

search_type = st.selectbox("search type", ["pg_search", "semantic search", "elasticsearch"])

semantic_search = None
pg_search_weights = None
es_weights = None

if search_type == "pg_search":
    with st.expander("pg_search weights"):
        if st.button("reset to defaults"):
            st.session_state.i += 1
        pg_search_weights = {}
        cols = cycle(st.columns(4))
        options = ["A", "B", "C", "D", None]
        for k, v in get_default_pg_search_weights().items():
            col = next(cols)
            w = col.selectbox(k, options, index=options.index(v), key=f"{k}_{st.session_state.i}")
            if w is not None:
                pg_search_weights[k] = w

elif search_type == "semantic search":
    semantic_search_model = st.selectbox(
        "semantic search model",
        options=SUGGESTED_MODELS,
        index=SUGGESTED_MODELS.index(DEFAULT_EMBEDDING_MODEL),
    )
    semantic_search = SemanticSearch(semantic_search_model)

elif search_type == "elasticsearch":
    with st.expander("elasticsearch weights"):
        if st.button("reset to defaults"):
            st.session_state.i += 1
        es_weights = {}
        cols = cycle(st.columns(4))
        for k, v in get_default_elasticsearch_weights().items():
            col = next(cols)
            w = col.number_input(k, value=float(v), min_value=0.0, step=1.0, key=f"{k}_elastic_{st.session_state.i}")
            if w is not None:
                es_weights[k] = w

else:
    raise NotImplementedError(f"{__name__}: {search_type=}")

level = st.selectbox("level", ["name", "number"])

metrics_df = run_metrics(
    query_list=query_terms,
    k_list=[1, 5, 10, 20, 50, 100],
    level=level,
    search_type=search_type,
    pg_search_weights=pg_search_weights,
    semantic_search=semantic_search,
    es_weights=es_weights,
)
metrics_df = metrics_df.sort("query")
st.write("query-level metrics")
st.write(metrics_df)
st.write("average metrics")
st.dataframe(metrics_df.mean())
