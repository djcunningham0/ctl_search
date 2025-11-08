from collections import defaultdict
from itertools import cycle
from typing import Literal

import streamlit as st
import plotly.express as px
import polars as pl

from app.elastic import get_default_elasticsearch_weights
from app.relevance import EvaluationConfig, run_metrics, get_top_n_queries
from app.semantic import SemanticSearch, DEFAULT_EMBEDDING_MODEL, SUGGESTED_MODELS
from app.sql import PgSearchConfig, get_default_pg_search_weights

st.write("## Search Metrics Comparison")

if "query_terms" not in st.session_state:
    st.session_state["query_terms"] = ", ".join(get_top_n_queries(20, exclude_camping=True))


def update_query_terms(n: int, exclude_camping: bool):
    st.session_state["query_terms"] = ", ".join(get_top_n_queries(n, exclude_camping=exclude_camping))


with st.expander("preset query lists"):
    exclude_camping = st.checkbox("exclude camping", value=True)
    cols = cycle(st.columns(4))
    for n in [5, 10, 20, 40, 60, 80, 100, 200]:
        next(cols).button(f"top {n} queries", on_click=update_query_terms, args=(n, exclude_camping))

# collect query terms
query_terms = st.text_input("query terms", value=st.session_state["query_terms"]).split(",")
query_terms = [x.strip() for x in query_terms if x.strip()]

level: Literal["name", "number"] = st.selectbox("level", ["name", "number"])

if not query_terms:
    st.stop()

if "methodologies" not in st.session_state:
    st.session_state["methodologies"] = []


def add_methodology():
    st.session_state["methodologies"].append({
        "type": "pg_search",
        "params": {
            "pg_search": PgSearchConfig(),
            "semantic_search": defaultdict(str),
            "es_weights": {},
        }
    })


def remove_methodology(index: int):
    st.session_state["methodologies"].pop(index)


@st.cache_data(hash_funcs={SemanticSearch: lambda x: x.model_str})
def _run_metrics(
        query_list: list[str],
        eval_config: EvaluationConfig,
        search_type: Literal["pg_search", "semantic search", "elasticsearch"],
        pg_search_config: PgSearchConfig,
        semantic_search: SemanticSearch,
        es_weights: dict[str, float],
):
    return run_metrics(
        query_list=query_list,
        eval_config=eval_config,
        search_type=search_type,
        pg_search_config=pg_search_config,
        semantic_search=semantic_search,
        es_weights=es_weights,
    )


def configure_methodology(index: int, methodology: dict):
    with st.container(border=True):
        st.write(f"### Methodology {index + 1}")
        c1, c2 = st.columns(2, vertical_alignment="bottom")
        search_type = c1.selectbox(
            "search type",
            options=["pg_search", "semantic search", "elasticsearch"],
            key=f"search_type_{index}",
            index=["pg_search", "semantic search", "elasticsearch"].index(methodology["type"])
        )
        methodology["type"] = search_type

        params = methodology["params"]

        if search_type == "pg_search":
            pg_params: PgSearchConfig= params["pg_search"]
            with st.expander("pg_search config"):
                # TODO: figure out reset button (not as easy as single methodology case
                #  -- don't want to reset *all* methodologies)
                cols = cycle(st.columns(4))
                options = ["A", "B", "C", "D", None]
                for k, v in get_default_pg_search_weights().items():
                    col = next(cols)
                    default_val = pg_params.pg_search_weights.get(k, v)
                    default_idx = options.index(default_val)
                    w = col.selectbox(k, options, index=default_idx, key=f"{k}_{index}")
                    pg_params.pg_search_weights[k] = w

                with st.container(horizontal=True, vertical_alignment="bottom"):
                    default_tsearch_weight = pg_params.tsearch_weight
                    tsearch_weight = st.number_input(
                        "tsearch weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(default_tsearch_weight),
                        step=0.1,
                        key=f"tsearch_weight_{index}",
                        width=200,
                    )
                    pg_params.tsearch_weight = tsearch_weight

                    default_normalization = pg_params.normalization
                    normalization = st.number_input(
                        "normalization",
                        min_value=0,
                        max_value=63,
                        value=int(default_normalization),
                        step=1,
                        key=f"normalization_{index}",
                        width=200,
                    )
                    pg_params.normalization = normalization

                    with st.container():
                        default_cover_density = pg_params.use_cover_density
                        use_cover_density = st.checkbox(
                            "use cover density",
                            value=default_cover_density,
                            key=f"use_cover_density_{index}",
                        )
                        pg_params.use_cover_density = use_cover_density

                        default_prefix = pg_params.prefix
                        use_prefix = st.checkbox(
                            "use prefix search",
                            value=default_prefix,
                            key=f"use_prefix_{index}",
                        )
                        pg_params.prefix = use_prefix

                with st.container(horizontal=True, vertical_alignment="bottom"):
                    default_trigram_threshold = pg_params.trigram_threshold
                    trigram_threshold = st.number_input(
                        "trigram threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(default_trigram_threshold),
                        step=0.1,
                        key=f"trigram_threshold_{index}",
                        width=200,
                    )
                    pg_params.trigram_threshold = trigram_threshold

                    default_trigram_sort_only = pg_params.trigram_sort_only
                    trigram_sort_only = st.checkbox(
                        "trigram sort only",
                        value=default_trigram_sort_only,
                        key=f"trigram_sort_only_{index}",
                    )
                    pg_params.trigram_sort_only = trigram_sort_only

        elif search_type == "semantic search":
            default_val = params["semantic_search"].get("model_name", DEFAULT_EMBEDDING_MODEL)
            default_idx = SUGGESTED_MODELS.index(default_val)
            semantic_search_model = st.selectbox(
                "semantic search model",
                options=SUGGESTED_MODELS,
                index=default_idx,
                key=f"semantic_model_{index}"
            )
            params["semantic_search"]["model_name"] = semantic_search_model
            params["semantic_search"]["model"] = SemanticSearch(semantic_search_model)

        elif search_type == "elasticsearch":
            with st.expander("elasticsearch weights"):
                cols = cycle(st.columns(4))
                params["es_weights"] = {}
                for k, v in get_default_elasticsearch_weights().items():
                    col = next(cols)
                    default_val = params["es_weights"].get(k, v)
                    w = col.number_input(k, value=float(default_val), min_value=0.0, step=1.0, key=f"{k}_elastic_{index}")
                    params["es_weights"][k] = w

        methodology["params"] = params
        if c2.button("Remove", key=f"remove_{index}"):
            remove_methodology(index)
            st.rerun()


# Configure each methodology
for i, methodology in enumerate(st.session_state["methodologies"]):
    configure_methodology(i, methodology)

st.button("Add Search Methodology", on_click=add_methodology)


# Run searches and compare results
results = []
for i, methodology in enumerate(st.session_state["methodologies"]):
    search_type = methodology["type"]
    params = methodology["params"]

    with st.spinner(f"executing methodology {i + 1}"):
        eval_config = EvaluationConfig(
            k_list=[1, 5, 10, 20, 50, 100],
            level=level,
        )
        metrics_df = (
            _run_metrics(
                query_list=query_terms,
                eval_config=eval_config,
                search_type=search_type,
                pg_search_config=params["pg_search"],
                semantic_search=params["semantic_search"]["model"],
                es_weights=params["es_weights"],
            )
            .with_columns(
                pl.lit(i + 1).alias("n"),
                pl.lit(search_type).alias("base_methodology"),
                pl.lit(f"{search_type} ({i + 1})").alias("name"),
            )
            .with_columns(
                pl.col("query").cast(pl.Enum(query_terms))
            )
        )
        results.append(metrics_df)

st.divider()

if results:
    final_df = pl.concat(results).sort("query", "n")

    st.write("### Query-level metrics")
    k = st.selectbox("k", [1, 5, 10, 20, 50, 100], index=2)
    fig = px.bar(final_df, x="query", y=f"ndcg_at_{k}", color="name", barmode="group")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig)

    st.write("### Overall metrics")
    agg_df = (
        final_df
        .drop("query")
        .group_by(["n", "name", "base_methodology"])
        .mean()
        .sort("n")
        .unpivot(
            index=["n", "name", "base_methodology"],
            variable_name="k",
            value_name="value",
        )
        .with_columns(pl.col("k").str.split("ndcg_at_").list.get(1).alias("k"))
    )
    fig = px.bar(agg_df, x="k", y="value", color="name", barmode="group")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig)
