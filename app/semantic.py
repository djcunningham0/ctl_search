import polars as pl
import streamlit as st
from sentence_transformers import SentenceTransformer
from torch import Tensor

from app.sql import execute_query


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUGGESTED_MODELS = [
    "all-MiniLM-L6-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-mpnet-base-v2",
    "multi-qa-distilbert-cos-v1",
    "multi-qa-MiniLM-L6-cos-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-MiniLM-L3-v2",
    "multi-qa-mpnet-base-cos-v1",
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
        return self.embedder.encode(query, convert_to_tensor=True)

    def search(self, query: str, n: int = None) -> pl.DataFrame:
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
    return execute_query("select * from items where status != 'retired'").rename({"name": "item_name"})


def item_df_to_corpus(item_df: pl.DataFrame) -> list[str]:
    return (
        item_df
        .filter(pl.col("status") != "retired")
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
    return embedder.encode(corpus, convert_to_tensor=True)
