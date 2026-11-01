# Part 2 — Indexing and Evaluation

Tag: `IRWA-2026-part-2`. Cloning the repository is covered in the [main README](../../README.md).

| file | content |
|---|---|
| `IRWA_Part2.ipynb` | notebook: indexing (points 1–3) and evaluation (points 1–3a) |
| `irwa_indexing.py` | inverted index, conjunctive (AND) search and TF-IDF ranking, adapted from Practice Session 1 |
| `irwa_evaluation.py` | P@K, R@K, AP@K, F1@K, MAP, MRR and NDCG, adapted from Practice Session 2 |
| `test_queries_labels.csv` | our relevance labels (1/0) for the top 20 results of our five test queries |

The notebook reads `data/processed_corpus.pkl` (written by the Part 1 notebook) and `data/validation_labels.csv`.

## Reproduce the results

From the repository root, with the environment of the [main README](../../README.md):

1. Run the Part 1 notebook first if `data/processed_corpus.pkl` does not exist yet.
2. Run the notebook from its folder:
   ```bash
   cd project_progress/part_2
   jupyter notebook IRWA_Part2.ipynb
   ```
   Select *Run → Run All Cells* (or run it without the browser:
   `jupyter nbconvert --to notebook --execute --inplace IRWA_Part2.ipynb`). It takes less than a minute.
