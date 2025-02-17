from itertools import cycle
import polars as pl
import streamlit as st

from app.elastic import search_with_elastic, get_default_elasticsearch_weights
from app.semantic import SemanticSearch, item_df_to_corpus, DEFAULT_EMBEDDING_MODEL, SUGGESTED_MODELS
from app.sql import pg_search, get_default_pg_search_weights


if "page_1_i" not in st.session_state:
    st.session_state["page_1_i"] = 0


def display_df(df: pl.DataFrame):
    for _, row in df.to_pandas().iterrows():
        st.write("---")
        st.write(f"Number: {row['number']}")
        st.write(f"name: {row['item_name']}")
        st.write(f"other names: {row['other_names']}")
        st.write(f"brand: {row['brand']}")
        st.write(trim_text(f"description: {row['plain_text_description']}"))
        st.write(f"size: {row['size']}")
        st.write(f"strength: {row['strength']}")
        st.write(f"rank = {row['search_rank']}")
        st.write(f"score = {row['search_score']}")
    st.write("---")


def trim_text(text: str, max_len: int = 100) -> str:
    if len(text) > max_len:
        return text[:max_len] + "..."
    else:
        return text


with st.sidebar:
    st.write("### pg_search")
    if st.button("reset to defaults", key=f"reset_pg"):
        st.session_state["page_1_i"] += 1
    cols = cycle(st.columns(2))
    options = ["A", "B", "C", "D", None]
    pg_search_weights = {}
    for k, v in get_default_pg_search_weights().items():
        col = next(cols)
        w = col.selectbox(k, options, index=options.index(v), key=f"{k}_{st.session_state['page_1_i']}")
        if w is not None:
            pg_search_weights[k] = w

    st.divider()

    st.write("### semantic search")
    semantic_search_model = st.selectbox(
        "semantic search model",
        options=SUGGESTED_MODELS,
        index=SUGGESTED_MODELS.index(DEFAULT_EMBEDDING_MODEL),
    )

    st.divider()

    st.write("### elasticsearch")
    if st.button("reset to defaults", key=f"reset_es"):
        st.session_state["page_1_i"] += 1
    cols = cycle(st.columns(2))
    es_weights = {}
    for k, v in get_default_elasticsearch_weights().items():
        col = next(cols)
        w = col.number_input(k, value=float(v), min_value=0.0, step=1.0,
                             key=f"{k}_elastic_{st.session_state['page_1_i']}")
        if w is not None:
            es_weights[k] = w

c1, c2 = st.columns(2)
query = c1.text_input("query")
query = query.strip()
n = c2.number_input("top n", value=10, min_value=1)
st.write("---")

c1, c2, c3 = st.columns(3)
if query != "":
    # pg_search
    with c1:
        st.write("### pg_search")
        results = pg_search(query, weights=pg_search_weights).head(n)
        display_df(results)

    # semantic search
    with c2:
        st.write("### semantic search")
        semantic_search = SemanticSearch(semantic_search_model)
        results = semantic_search.search(query, n)
        results_str = item_df_to_corpus(results)
        scores = results.select("search_score").to_series().to_list()
        display_df(results)

    # elasticsearch
    with c3:
        st.write("### elasticsearch")
        results = search_with_elastic(query, weights=es_weights, size=n)
        display_df(results)
