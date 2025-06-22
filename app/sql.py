import re
from typing import Union

import polars as pl
from sqlalchemy import Engine, TextClause, create_engine, text


DEFAULT_ENGINE = create_engine("postgresql+psycopg2://Daniel@localhost:5432/circulate_development")


def execute_query(
        query: Union[str, TextClause],
        engine: Engine = DEFAULT_ENGINE,
        **kwargs,
) -> pl.DataFrame:
    return pl.read_database(query, engine, **kwargs)


def pg_search(
        search_term: str,
        weights: dict[str, str] = None,
        engine: Engine = DEFAULT_ENGINE,
) -> pl.DataFrame:
    if weights is None:
        weights = get_default_pg_search_weights()

    weights = {k: v for k, v in weights.items() if v is not None}

    ts_rank_str = " || ".join([f"setweight(to_tsvector('english', {k}::text), '{v}')" for k, v in weights.items()])
    ts_rank_str = f"ts_rank({ts_rank_str}, to_tsquery('english', :search_term))"
    where_str = " || ".join([f"to_tsvector('english', {k}::text)" for k in weights.keys()])
    where_str = f"({where_str} @@ to_tsquery('english', :search_term))"
    sanitized_search_term = re.sub(r"[^a-zA-Z0-9\s]", "", search_term)
    prefix_search_term = "&".join([f"{word}:*" for word in sanitized_search_term.split(" ")])

    query = text(f"""
        SELECT
            '{sanitized_search_term}' as query,
            id,
            name as item_name,
            number,
            other_names,
            brand,
            plain_text_description,
            size,
            strength,
            {ts_rank_str} AS search_score
        FROM items
        WHERE 
            {where_str}
            AND status != 'retired'
        ORDER BY
            search_score DESC,
            number
        ;
    """)
    query = query.bindparams(search_term=prefix_search_term)  # equivalent to {tsearch: {prefix: true}} in Rails app
    return (
        execute_query(query, engine)
        .with_columns(pl.col("search_score").rank(descending=True, method="ordinal").alias("search_rank"))
    )


def get_default_pg_search_weights() -> dict[str, str]:
    return {
        "name": "A",
        "number": "A",
        "other_names": "B",
        "brand": "C",
        "plain_text_description": "C",
        "size": "D",
        "strength": "D",
    }
