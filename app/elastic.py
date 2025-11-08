import polars as pl
import streamlit as st
from elasticsearch import Elasticsearch

from app.sql import execute_query


@st.cache_resource
def get_es_client(url: str = "http://localhost:9200") -> Elasticsearch:
    client = Elasticsearch(url)
    build_items_index(client)
    return client


def create_items_index(client: Elasticsearch, delete_if_exists: bool = True):
    if client.indices.exists(index="items"):
        if delete_if_exists:
            client.indices.delete(index="items")
        else:
            return None

    mapping = {
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "text"},
            "number": {"type": "text"},
            "other_names": {"type": "text"},
            "brand": {"type": "text"},
            "plain_text_description": {"type": "text"},
            "size": {"type": "text"},
            "strength": {"type": "text"},
        }
    }
    client.indices.create(index="items", mappings=mapping)


def build_items_index(client: Elasticsearch):
    with st.spinner("building items index"):
        create_items_index(client, delete_if_exists=True)
        query = """
            SELECT id, name, number, other_names, brand, plain_text_description, size, strength, status
            FROM items
            WHERE status not in ('retired', 'missing', 'pending')
            ORDER BY number
        """
        df = execute_query(query).to_pandas()
        for _, row in df.iterrows():
            doc = {
                "id": row["id"],
                "name": row["name"],
                "number": row["number"],
                "other_names": row["other_names"],
                "brand": row["brand"],
                "plain_text_description": row["plain_text_description"],
                "size": row["size"],
                "strength": row["strength"],
            }
            client.index(index="items", document=doc)


def search_with_elastic(
    search_term: str,
    client: Elasticsearch = None,
    weights: dict[str, int] = None,
    **kwargs,
) -> pl.DataFrame:
    if client is None:
        client = get_es_client()

    if weights is None:
        weights = get_default_elasticsearch_weights()

    query = {
        "multi_match": {
            "query": search_term,
            "fields": [f"{k}^{v}" for k, v in weights.items()],  # e.g., "name^4"
        }
    }

    size = kwargs.pop("size", 1000)
    response = client.search(index="items", query=query, size=size, **kwargs)

    # put results into polars dataframe
    data = {"query": search_term, "id": [], "search_score": [], "search_rank": []}
    for i, result in enumerate(response.body["hits"]["hits"]):
        data["id"].append(result["_source"]["id"])
        data["search_score"].append(result["_score"])
        data["search_rank"].append(i + 1)

    results_df = pl.DataFrame(data)

    if results_df.shape[0] == 0:
        return pl.DataFrame()

    items_df = execute_query("select * from items")
    return (
        results_df.join(items_df, on=["id"], how="inner")
        .rename({"name": "item_name"})
        .sort("search_rank")
    )


def get_default_elasticsearch_weights() -> dict[str, int]:
    return {
        "name": 4,
        "number": 3,
        "other_names": 2,
        "brand": 2,
        "plain_text_description": 1,
        "size": 1,
        "strength": 1,
    }
