import argparse
import logging

import optuna

from app.sql import PgSearchConfig
from app.relevance import EvaluationConfig, get_top_n_queries, run_metrics

# set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Tune pg_search parameters")
parser.add_argument(
    "--log-level",
    choices=["debug", "info", "warning", "error", "critical"],
    default="warning",
    help="Set the logging level (default: warning)",
)
args, _ = parser.parse_known_args()
log_level = logging._nameToLevel.get(args.log_level.upper(), logging.WARNING)
logger.setLevel(log_level)

# parameters for evaluation
TOP_N_QUERIES = 50
K_LIST = [1, 3, 5, 10, 20, 50]
PRIMARY_K_VALS = [3, 10]  # NDCG will be optimized for these k values
DCG_TYPE = "linear"
LEVEL = "name"

eval_config = EvaluationConfig(
    k_list=K_LIST,
    dcg_type=DCG_TYPE,
    level=LEVEL,
)

logger.info("configuring Optuna study...")
STUDY_NAME = f"pg_search_tuning_top_{TOP_N_QUERIES}"
DB_NAME = "tuning_db.sqlite3"
logger.debug(f"{STUDY_NAME=}")

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=f"sqlite:///{DB_NAME}",
    load_if_exists=True,
    directions=["maximize", "maximize"],
)
metric_names = [f"ndcg@{k}" for k in PRIMARY_K_VALS]
logger.info(f"optimizing {metric_names}")
study.set_metric_names(metric_names)

logger.info(f"===== Run Optuna dashboard with `optuna-dashboard sqlite:///{DB_NAME}`")

logger.info("loading top queries...")
QUERY_LIST = get_top_n_queries(TOP_N_QUERIES, exclude_camping=True)
logger.debug(f"{QUERY_LIST=}")


def objective(trial: optuna.Trial) -> tuple[float, float]:
    # log some attributes (should be the same for all trials, but this will
    # keep a record in case the script is updated)
    trial.set_user_attr("top_n_queries", TOP_N_QUERIES)
    trial.set_user_attr("k_list", eval_config.k_list)
    trial.set_user_attr("dcg_type", eval_config.dcg_type)
    trial.set_user_attr("level", eval_config.level)
    trial.set_user_attr("view_weight", eval_config.view_weight)
    trial.set_user_attr("hold_weight", eval_config.hold_weight)
    trial.set_user_attr("alpha", eval_config.alpha)
    trial.set_user_attr("max_i", eval_config.max_i)

    # define parameter space
    options = ["A", "B", "C", "D"]
    other_names_weight = trial.suggest_categorical("other_names_weight", options)
    brand_weight = trial.suggest_categorical("brand_weight", options)
    plain_text_description_weight = trial.suggest_categorical(
        "plain_text_description_weight", options
    )
    size_weight = trial.suggest_categorical("size_weight", options)
    strength_weight = trial.suggest_categorical("strength_weight", options)
    weights = {
        "name": "A",
        "number": "A",
        "other_names": other_names_weight,
        "brand": brand_weight,
        "plain_text_description": plain_text_description_weight,
        "size": size_weight,
        "strength": strength_weight,
    }

    pg_search_config = PgSearchConfig(
        pg_search_weights=weights,
        tsearch_weight=trial.suggest_float("tsearch_weight", 0.0, 1.0, step=0.01),
        use_cover_density=trial.suggest_categorical("use_cover_density", [True, False]),
        normalization=0,  # TODO: tune normalization?
        prefix=trial.suggest_categorical("prefix", [True, False]),
        trigram_threshold=trial.suggest_float("trigram_threshold", 0.0, 1.0, step=0.05),
        trigram_sort_only=trial.suggest_categorical("trigram_sort_only", [True, False]),
    )

    metrics_df = run_metrics(
        query_list=QUERY_LIST,
        eval_config=eval_config,
        search_type="pg_search",
        pg_search_config=pg_search_config,
    )

    vals = {k: float(metrics_df[f"ndcg_at_{k}"].mean()) for k in K_LIST}
    for k, v in vals.items():
        trial.set_user_attr(f"ndcg_at_{k}", v)

    return tuple(vals[k] for k in PRIMARY_K_VALS)


logger.info("starting optimization...")

# study.enqueue_trial(
#     {
#         "name_weight": "A",
#         "number_weight": "A",
#         "other_names_weight": "B",
#         "brand_weight": "C",
#         "plain_text_description_weight": "C",
#         "size_weight": "D",
#         "strength_weight": "D",
#         "tsearch_weight": 1.0,
#         "use_cover_density": False,
#         "prefix": True,
#         "trigram_threshold": 0.5,
#         "trigram_sort_only": True,
#     }
# )
# study.enqueue_trial(
#     {
#         "name_weight": "A",
#         "number_weight": "A",
#         "other_names_weight": "B",
#         "brand_weight": "C",
#         "plain_text_description_weight": "C",
#         "size_weight": "D",
#         "strength_weight": "D",
#         "tsearch_weight": 0.8,
#         "use_cover_density": False,
#         "prefix": True,
#         "trigram_threshold": 0.5,
#         "trigram_sort_only": True,
#     }
# )

study.optimize(objective, timeout=2 * 60 * 60)
