from typing import Optional

import polars as pl
import streamlit as st
from sentence_transformers import SentenceTransformer
from torch import Tensor

from app.sql import execute_query


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# https://sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models
SUGGESTED_MODELS = [
    "all-MiniLM-L6-v2",
    "multi-qa-mpnet-base-cos-v1",
    "multi-qa-mpnet-base-dot-v1",
    "all-mpnet-base-v2",  # surprisingly seems worse than all-MiniLM-L6-v2
    "multi-qa-distilbert-cos-v1",
    "multi-qa-MiniLM-L6-cos-v1",

    # MSMARCO models are designed to work better for asymmetric search, where queries are shorter than documents
    # https://sbert.net/docs/pretrained-models/msmarco-v5.html
    "msmarco-MiniLM-L6-cos-v5",  # cosine
    "msmarco-MiniLM-L12-cos-v5",  # cosine
    "msmarco-distilbert-cos-v5",  # cosine
    "msmarco-distilbert-base-tas-b",  # dot
    "msmarco-distilbert-dot-v5",  # dot
    "msmarco-bert-base-dot-v5",  # dot

    # I considered the ones below this line but they performed worse than the others
    # "all-distilroberta-v1",
    # "all-MiniLM-L12-v2",
]


class SemanticSearch:
    def __init__(self, model_str: str = DEFAULT_EMBEDDING_MODEL):
        self.model_str = model_str
        self.embedder = load_sentence_transformer_model(model_str)
        self.df = load_items_df()
        self.corpus = item_df_to_corpus(self.df)
        self.embedded_corpus = self.embed_corpus()

    def embed_corpus(self) -> Tensor:
        return _embed_corpus(self.embedder, self.corpus)

    def encode_query(self, query: str) -> Tensor:
        return self.embedder.encode_query(query, convert_to_tensor=True)  # type: ignore

    def search(self, query: str, n: Optional[int] = None) -> pl.DataFrame:
        query_embedding = self.encode_query(query)
        cos_scores: Tensor = self.embedder.similarity(query_embedding, self.embedded_corpus)[0]
        n = n or len(self.corpus)
        return (
            self.df
            .with_columns(pl.lit(query).alias("query"))
            .with_columns(pl.Series("search_score", cos_scores.tolist()))
            .with_columns(pl.col("search_score").rank(descending=True, method="ordinal").alias("search_rank"))
            .sort("search_rank")
            .head(n)
        )

def load_items_df() -> pl.DataFrame:
    query = """
        select *
        from items
        where status not in ('retired', 'missing', 'pending')
    """
    return execute_query(query).rename({"name": "item_name"})


def item_df_to_corpus(item_df: pl.DataFrame) -> list[str]:
    return (
        item_df
        .filter(~pl.col("status").is_in(["retired", "missing", "pending"]))
        .select(pl.format(
            "ID: {}"
            "\n\nname: {}"
            "\n\nother names: {}"
            "\n\nbrand: {}"
            "\n\ndescription: {}"
            "\n\nsize: {}"
            "\n\nstrength: {}",
            pl.col("number"),
            pl.col("item_name"),
            pl.col("other_names"),
            pl.col("brand"),
            pl.col("plain_text_description"),
            pl.col("size"),
            pl.col("strength"),
        ))
        .to_series()
        .to_list()
    )


@st.cache_resource
def load_sentence_transformer_model(model_str: str) -> SentenceTransformer:
    return SentenceTransformer(model_str)


@st.cache_data(hash_funcs={SentenceTransformer: lambda x: x.model_card_data.base_model})
def _embed_corpus(embedder: SentenceTransformer, corpus: list[str]) -> Tensor:
    return embedder.encode_document(corpus, convert_to_tensor=True)  # type: ignore
