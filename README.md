# ctl_search

Experimenting with different search methodologies for the Chicago Tool Library.

Corresponding blog posts:
* https://dannycunningham.com/posts/2025-11-16-chicago-tool-library-search-3/
* https://dannycunningham.com/posts/2025-12-01-chicago-tool-library-search-4/

## Contents

* `app/` contains code for a Streamlit app with several pages
  * `01_search_results_comparison.py`: compare search results for an input query using three different search methodologies (pg_search, semantic search, and elasticsearch) with adjustable parameters
  * `02_search_metrics.py`: evaluate search quality using NDCG@k on the top N queries for any number of search methodologies with adjustable parameters
  * `03_relevance_scores.py`: calculate the inferred relevance scores for a given query (see the `relevant_items.ipynb` notebook and/or the [corresponding blog post](https://dannycunningham.com/posts/2025-11-16-chicago-tool-library-search-3/) for methodology overview)
* `relevant_items.ipynb`: methodology for identifying relevant items for a given query and calculating NDCG@k
* `tune_pg_search.py`: script for running a parameter search using Optuna
