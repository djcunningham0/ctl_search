import polars as pl
import streamlit as st

from app.elastic import get_es_client, search_with_elastic
from app.semantic import SemanticSearch, item_df_to_corpus, DEFAULT_EMBEDDING_MODEL, SUGGESTED_MODELS
from app.sql import pg_search


def display_df(df: pl.DataFrame):
    for _, row in df.to_pandas().iterrows():
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
    semantic_search_model = st.selectbox(
        "semantic search model",
        options=SUGGESTED_MODELS,
        index=SUGGESTED_MODELS.index(DEFAULT_EMBEDDING_MODEL),
    )

semantic_search = SemanticSearch(semantic_search_model)

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
        results = pg_search(query).head(n)
        display_df(results)
        for _, row in results.head(n).to_pandas().iterrows():
            st.write(f"ID: {row['number']}")
            st.write(f"name: {row['item_name']}")
            st.write(f"other names: {row['other_names']}")
            st.write(f"brand: {row['brand']}")
            st.write(f"description: {row['plain_text_description']}")
            st.write(f"size: {row['size']}")
            st.write(f"strength: {row['strength']}")
            st.write(f"score = {row['search_rank']}")
            st.write("---")

    # semantic search
    with c2:
        st.write("### semantic search")
        results = semantic_search.search(query, n)
        results_str = item_df_to_corpus(results)
        scores = results.select("search_score").to_series().to_list()
        display_df(results)
        for result, score in zip(results_str, scores):
            st.write(result)
            st.write(f"score = {score}")
            st.write("---")

    # elasticsearch
    with c3:
        st.write("### elasticsearch")
        es_client = get_es_client()
        results = search_with_elastic(es_client, query, size=n)
        display_df(results)
