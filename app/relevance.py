from typing import Literal

import polars as pl
import streamlit as st
from urllib.parse import unquote

from app import metrics
from app.elastic import search_with_elastic
from app.semantic import SemanticSearch
from app.sql import execute_query, pg_search


DEFAULT_VIEW_WEIGHT = 1.0
DEFAULT_HOLD_WEIGHT = 2.0
DEFAULT_ALPHA = 0.4
DEFAULT_MAX_I = 100

# remove my visits because I was likely just testing the search behavior
MY_VISIT_IDS = [
    49827,
    67171,
    114696,
    116022,
    118109,
    118368,
    118995,
    124701,
    125090,
    127437,
]


def lowercase_and_strip(col: str):
    return pl.col(col).str.to_lowercase().str.strip_chars()


@st.cache_data
def load_query_df() -> pl.DataFrame:
    query_df = pl.read_csv("./data/relevant_search_results.csv")

    # clean up some queries
    query_df = (
        query_df
        .filter(~pl.col("visit_id").is_in(MY_VISIT_IDS))
        # remove URL encoding (e.g., "%20" -> " ")
        .with_columns(pl.col("query").map_elements(lambda x: unquote(x), return_dtype=pl.String))
        .with_columns(lowercase_and_strip("query").alias("query"))  # remove leading/trailing whitespace
    )
    return query_df


@st.cache_data
def get_top_n_queries(n: int) -> list[str]:
    return (
        load_query_df()
        .group_by("query")
        .agg(pl.len().alias("count"))
        .sort(["count", "query"], descending=[True, False])
        .head(n)
        .select("query")
        .to_series()
        .to_list()
    )


def calculate_relevance_df(
        query_df: pl.DataFrame,
        view_weight: float = DEFAULT_VIEW_WEIGHT,
        hold_weight: float = DEFAULT_HOLD_WEIGHT,
        alpha: float = DEFAULT_ALPHA,
        max_i: int = DEFAULT_MAX_I,
) -> pl.DataFrame:
    """
    Calculate relevance scores for items to each query. Position bias is taken into
    account with the `alpha` and `max_i` parameters.

    Relevance score calculation:
    For each interaction:
    - position_bias = 1 / (i**alpha) where i is the search index (capped at max_i)
    - views_score = view_weight / position_bias
    - holds_score = hold_weight / position_bias
    Aggregated scores:
    - raw_score = views_score + holds_score
    - relevance_score = raw_score / max_score_for_query
      - i.e., scaled such that most relevant item for query has relevance score 1

    Parameters
    ----------
    query_df
        Query data with views and holds (output from `load_query_df`)
    view_weight
        Weight to give to views
    hold_weight
        Weight to give to holds
    alpha
        Position bias adjustment factor -- P(view|position i) = 1 / (i**alpha)
    max_i
        Stop increasing positional bias beyond this index
    """
    return (
        query_df
        .with_columns(lowercase_and_strip("item_name").alias("item_name"))
        .with_columns(
            pl.when(pl.col("search_index") > max_i).then(max_i)
            .otherwise(pl.col("search_index"))
            .alias("capped_index")
        )
        .with_columns((1 / (pl.col("capped_index") ** alpha)).alias("bias_adjustment"))
        .with_columns(
            pl.when(pl.col("view_time").is_not_null())
            .then(view_weight / pl.col("bias_adjustment"))
            .alias("view_score")
        )
        .with_columns(
            pl.when(pl.col("hold_id").is_not_null())
            .then(hold_weight / pl.col("bias_adjustment"))
            .alias("hold_score")
        )
        .group_by(
            "query",
            "item_name",
        )
        .agg(
            pl.median("search_index").alias("median_index"),
            pl.count("view_time").alias("view_count"),
            pl.count("hold_id").alias("hold_count"),
            pl.sum("view_score").alias("view_score"),
            pl.sum("hold_score").alias("hold_score"),
        )
        .with_columns(
            (pl.col("view_count") + pl.col("hold_count")).alias("view_plus_hold_count"),
            (pl.col("view_score") + pl.col("hold_score")).alias("score"),
        )
        .with_columns((pl.col("score") / pl.max("score").over("query")).alias("normalized_score"))
        .with_columns(pl.sum("view_count").over("query").alias("query_view_count"))
        .with_columns(pl.col("score").rank(descending=True, method="ordinal").over("query").alias("rank"))
        .sort(["query_view_count", "query", "rank"], descending=[True, False, False])
    )


def collect_relevance_df(
        relevance_df: pl.DataFrame,
        level: Literal["name", "number"] = "name",
) -> pl.DataFrame:
    """
    Collect relevant items and relevance scores into lists for each query.

    Example output:
    | query     | relevant_list        | score_list |
    |-----------|----------------------|------------|
    | "query 1" | ["item 1", "item 2"] | [0.8, 0.5] |
    | "query 2" | ["item 3"]           | [0.6]      |

    Parameters
    ----------
    relevance_df
        Relevance data (output from `calculate_relevance_df`)
    level
        Level at which to collect relevant items (name or item number)
    """
    if level == "name":
        collect_col = "item_name"
    else:
        collect_col = "item_number"
        sql_query = "select name as item_name, number::text as item_number from items"
        item_df = (
            execute_query(sql_query)
            .with_columns(lowercase_and_strip("item_name").alias("item_name"))
            .with_columns(pl.concat_str("item_number", pl.lit("-"), "item_name").alias("item_number"))
        )
        relevance_df = relevance_df.join(item_df, on=["item_name"], how="inner")

    return (
        relevance_df
        .group_by("query")
        .agg(
            pl.col(collect_col).sort_by("rank").alias("relevant_list"),
            pl.col("normalized_score").sort_by("rank").alias("score_list"),
        )
    )


def get_retrieved_df(
        # TODO refactor so it just collects a dataframe, doesn't actually execute the query
        query_list: list[str],
        search_type: Literal["pg_search", "semantic search", "elasticsearch"] = "pg_search",
        level: Literal["name", "number"] = "name",
        pg_search_weights: dict[str, str] = None,
        semantic_search: SemanticSearch = None,
        es_weights: dict[str, float] = None,
) -> pl.DataFrame:
    """
    Get the list of items returned by a pg_search query.

    Example output:
    | query     | retrieved_list       |
    |-----------|----------------------|
    | "query 1" | ["item 1", "item 2"] |
    | "query 2" | ["item 3"]           |

    Parameters
    ----------
    query_list
        List of search terms
    level
        Level at which to collect relevant items (name or item number)
    pg_search_weights
        Weights for the pg_search queries
    """
    if search_type == "semantic search" and semantic_search is None:
        raise ValueError("Must provide `semantic_search` if `search_type` is 'semantic'")

    collect_col = "item_name" if level == "name" else "item_number"
    df_list = []

    for query in query_list:
        if search_type == "pg_search":
            _df = pg_search(query, weights=pg_search_weights)
        elif search_type == "semantic search":
            _df = semantic_search.search(query)
        elif search_type == "elasticsearch":
            _df = search_with_elastic(query, weights=es_weights)
        else:
            raise NotImplementedError(f"{search_type=}")
        _df = _df.with_columns(lowercase_and_strip("item_name").alias("item_name"))
        if level == "name":
            # only keep the first instance of each item name
            _df = _df.filter(pl.cum_count("item_name").over("query", "item_name") == 1)
        else:
            _df = _df.with_columns(pl.concat_str("number", pl.lit("-"), "item_name").alias("item_number"))
        _df = (
            _df
            .group_by("query")
            .agg(pl.col(collect_col).alias("retrieved_list"))
        )
        df_list.append(_df)

    return pl.concat(df_list)


def combine_retrieved_and_relevant_dfs(
        retrieved_df: pl.DataFrame,
        relevant_df: pl.DataFrame,
) -> pl.DataFrame:
    return retrieved_df.join(relevant_df, on=["query"], how="inner")


def calculate_ndcg(
        combined_df: pl.DataFrame,
        k_list: list[int],
        dcg_type: Literal["linear", "exponential"] = "linear",
        drop_lists: bool = True,
) -> pl.DataFrame:
    metrics_df = combined_df
    relevance_data = pl.struct("retrieved_list", "relevant_list", "score_list")

    for k in k_list:
        metrics_df = metrics_df.with_columns(
            relevance_data
            .map_elements(
                lambda x: metrics.ndcg_at_k(
                    retrieved=x["retrieved_list"],
                    relevant=dict(zip(x["relevant_list"], x["score_list"])),
                    k=k,
                    dcg_type=dcg_type,
                ),
                return_dtype=pl.Float32,
            )
            .alias(f"ndcg_at_{k}")
        )

    if drop_lists:
        metrics_df = metrics_df.drop(["retrieved_list", "relevant_list", "score_list"])

    return metrics_df


def run_metrics(
        query_list: list[str],
        k_list: list[int],
        level: Literal["name", "number"] = "name",
        search_type: Literal["pg_search", "semantic search", "elasticsearch"] = "pg_search",
        pg_search_weights: dict[str, str] = None,
        view_weight: float = DEFAULT_VIEW_WEIGHT,
        hold_weight: float = DEFAULT_HOLD_WEIGHT,
        alpha: float = DEFAULT_ALPHA,
        max_i: int = DEFAULT_MAX_I,
        dcg_type: Literal["linear", "exponential"] = "linear",
        drop_lists: bool = True,
        semantic_search: SemanticSearch = None,
        es_weights: dict[str, float] = None,
) -> pl.DataFrame:
    query_df = load_query_df()
    relevance_df = calculate_relevance_df(
        query_df=query_df,
        view_weight=view_weight,
        hold_weight=hold_weight,
        alpha=alpha,
        max_i=max_i,
    )
    collected_relevance = collect_relevance_df(relevance_df, level=level)
    retrieved_df = get_retrieved_df(
        query_list=query_list,
        search_type=search_type,
        level=level,
        pg_search_weights=pg_search_weights,
        semantic_search=semantic_search,
        es_weights=es_weights,
    )
    combined_df = combine_retrieved_and_relevant_dfs(
        retrieved_df=retrieved_df,
        relevant_df=collected_relevance,
    )
    return calculate_ndcg(
        combined_df=combined_df,
        k_list=k_list,
        dcg_type=dcg_type,
        drop_lists=drop_lists,
    )
