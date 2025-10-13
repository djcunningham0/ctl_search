import re
from dataclasses import dataclass
from typing import Optional, Union

import polars as pl
from sqlalchemy import Engine, TextClause, create_engine, text


DEFAULT_ENGINE = create_engine("postgresql+psycopg2://Daniel@localhost:5432/circulate_development")

# TODO: move this somewhere else?
with DEFAULT_ENGINE.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))


@dataclass
class PgSearchConfig:
    pg_search_weights: dict[str, str] = None  # e.g. {"name": "A", "number": "A", ...}
    tsearch_weight: float = 1.0

    def __post_init__(self):
        if self.pg_search_weights is None:
            self.pg_search_weights = get_default_pg_search_weights()


def execute_query(
        query: Union[str, TextClause],
        engine: Engine = DEFAULT_ENGINE,
        **kwargs,
) -> pl.DataFrame:
    return pl.read_database(query, engine, **kwargs)


def pg_search(
        search_term: str,
        weights: Optional[dict[str, str]] = None,
        engine: Engine = DEFAULT_ENGINE,
        tsearch_weight: float = 1.0,
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

    # Trigram similarity (pick fields to include)
    trigram_expr = "greatest(" + ", ".join(
        [f"similarity({k}::text, :search_term_raw)" for k in weights.keys()]
    ) + ")"

    trigram_weight = 1 - tsearch_weight

    query = text(f"""
        SELECT
            '{sanitized_search_term}' AS query,
            id,
            name AS item_name,
            number,
            other_names,
            brand,
            plain_text_description,
            size,
            strength,
            {ts_rank_str} AS tsearch_score,
            {trigram_expr} AS trigram_score,
            {ts_rank_str} * {tsearch_weight} + {trigram_expr} * {trigram_weight} AS search_score
        FROM items
        WHERE
            ({where_str} OR {trigram_expr} > 0.1)
            AND status != 'retired'
        ORDER BY
            search_score DESC,
            number;
    """)
    query = query.bindparams(search_term=prefix_search_term, search_term_raw=sanitized_search_term)
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
