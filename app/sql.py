import logging
import re
from dataclasses import dataclass
from typing import Optional, Union

import polars as pl
from sqlalchemy import Engine, TextClause, create_engine, text

logger = logging.getLogger(__name__)


DEFAULT_ENGINE = create_engine(
    "postgresql+psycopg2://Daniel@localhost:5432/circulate_development"
)

# TODO: move this somewhere else?
with DEFAULT_ENGINE.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))


@dataclass
class PgSearchConfig:
    pg_search_weights: dict[str, Optional[str]] = None  # e.g. {"name": "A", "number": "A", ...}  # fmt: skip
    tsearch_weight: float = 1.0
    use_cover_density: bool = False
    normalization: int = 0  # https://github.com/Casecommons/pg_search?tab=readme-ov-file#normalization  # fmt: skip
    prefix: bool = True
    trigram_threshold: float = 0.5
    trigram_sort_only: bool = True

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
    pg_search_config: PgSearchConfig = None,
    engine: Engine = DEFAULT_ENGINE,
) -> pl.DataFrame:
    if pg_search_config is None:
        pg_search_config = PgSearchConfig()

    weights = pg_search_config.pg_search_weights
    weights = {k: v for k, v in weights.items() if v is not None}

    # TODO: it's possible to customize the relative weights of A, B, C, D
    #  by passing an array of floats as the first argument to `ts_rank`,
    #  e.g., `ts_rank(array[0.1, 0.2, 0.4, 1.0]::real[], ...)`
    ts_rank_fn = "ts_rank_cd" if pg_search_config.use_cover_density else "ts_rank"
    ts_rank_str = " || ".join([
        f"setweight(to_tsvector('english', {k}::text), '{v}')"
        for k, v in weights.items()
    ])
    ts_rank_str = (
        f"{ts_rank_fn}({ts_rank_str}, to_tsquery('english', :search_term),"
        f" {pg_search_config.normalization})"
    )
    where_str = " || ".join(
        [f"to_tsvector('english', {k}::text)" for k in weights.keys()]
    )
    where_str = f"({where_str} @@ to_tsquery('english', :search_term))"
    sanitized_search_term = re.sub(r"[^a-zA-Z0-9\s]", "", search_term)

    if pg_search_config.prefix:
        prefix_search_term = "&".join(
            [f"{word}:*" for word in sanitized_search_term.split(" ")]
        )
    else:
        prefix_search_term = "&".join(
            [f"{word}" for word in sanitized_search_term.split(" ")]
        )

    trigram_expr = "similarity(name, :search_term_raw)"
    trigram_weight = 1 - pg_search_config.tsearch_weight

    if pg_search_config.trigram_sort_only:
        # do not include trigram score in `where` clause
        pass
    else:
        where_str = (
            f"({where_str} OR {trigram_expr} >"
            f" {float(pg_search_config.trigram_threshold)})"
        )

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
            {ts_rank_str} * {pg_search_config.tsearch_weight} + {trigram_expr} * {trigram_weight} AS search_score
        FROM items
        WHERE
            {where_str}
            AND status not in ('retired', 'missing', 'pending')
        ORDER BY
            search_score DESC,
            number;
    """)
    query = query.bindparams(
        search_term=prefix_search_term, search_term_raw=sanitized_search_term
    )

    logger.debug(
        "pg_search query: %s", query.compile(compile_kwargs={"literal_binds": True})
    )

    return execute_query(query, engine).with_columns(
        pl.col("search_score")
        .rank(descending=True, method="ordinal")
        .alias("search_rank")
    )


def get_default_pg_search_weights() -> dict[str, Optional[str]]:
    return {
        "name": "A",
        "number": "A",
        "other_names": "B",
        "brand": "C",
        "plain_text_description": "C",
        "size": "D",
        "strength": "D",
    }
