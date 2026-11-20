# Part 3 — Ranking and Filtering

Tag: `IRWA-2026-part-3`. Cloning the repository and the environment are covered in the
[main README](../../README.md).

| file | content |
|---|---|
| `IRWA-2026-u231867-u231747-u233694-part-3.pdf` | report (LaTeX source: the `.tex` file with the same name) |
| `IRWA_Part3.ipynb` | notebook: the three rankings (point 1), word2vec (point 2) and the evaluation of the four rankings |
| `irwa_ranking.py` | `Ranker` class with the four ranking methods (TF-IDF + cosine, BM25, our score, word2vec + cosine) |
| `test_queries_labels_all.csv` | our relevance labels (1/0) for **every** product returned by the AND search for the five test queries |
| `figures/` | the plots of the report (written by the notebook) |
| `results/` | the result tables of the report as CSV (written by the notebook) |

The notebook also uses the Part 1 and Part 2 modules (`../part_1/irwa_preprocessing.py`,
`../part_2/irwa_indexing.py` and `../part_2/irwa_evaluation.py`), `data/processed_corpus.pkl` (written by the Part 1
notebook) and `data/validation_labels.csv`.

## Reproduce the results

From the repository root, with the environment of the [main README](../../README.md) (Part 3 adds `gensim==4.4.0`
to `requirements.txt`, so run `pip install -r requirements.txt` again if your environment comes from Part 2):

1. Run the Part 1 notebook first if `data/processed_corpus.pkl` does not exist yet.
2. Run the notebook from its folder:
   ```bash
   cd project_progress/part_3
   jupyter notebook IRWA_Part3.ipynb
   ```
   Select *Run → Run All Cells* (or run it without the browser:
   `jupyter nbconvert --to notebook --execute --inplace IRWA_Part3.ipynb`). It takes about one minute; training
   word2vec takes about 30 seconds of it. The word2vec model uses a fixed seed and one worker, so every run gives the
   same vectors and the same rankings.

## Selecting the ranking method

All the methods rank the same products, the ones that contain **all** the query terms (conjunctive AND search from
Part 2); only the order changes. The method is the `method` argument of `Ranker.search`:

| `method` | ranking | options |
|---|---|---|
| `"tfidf"` | TF-IDF vectors (raw tf × log(N/df)) + cosine similarity | — |
| `"bm25"` | Okapi BM25 | `Ranker(corpus, k1=1.2, b=0.75)` |
| `"ours"` | field-weighted BM25 (BM25F) × title factor × quality factor | `Ranker(corpus, field_weights={...}, title_weight=1.0)`; quality weights at the top of `irwa_ranking.py` |
| `"word2vec"` | mean word2vec vector of the query and of the product + cosine similarity | `ranker.train_word2vec(vector_size=..., window=...)` before searching |

```python
import sys
sys.path += ["../part_1", "../part_2"]
import pandas as pd
import irwa_ranking as rk

corpus = pd.read_pickle("../../data/processed_corpus.pkl")
ranker = rk.Ranker(corpus)                                   # builds the index (about 10 s)
ranker.search("women floral print shirt", method="bm25", top_k=10)      # [(pid, score), ...]
ranker.train_word2vec()                                      # only needed for method="word2vec"
rk.results_table(ranker, ranker.search("black running shoes for men", method="word2vec", top_k=20))
```

An unknown method name raises a `ValueError`, and `method="word2vec"` before `train_word2vec()` raises a
`RuntimeError`.

## Our score in short

`ours = BM25F × title factor × quality`:

* **BM25F**: BM25 where a match in the title counts more than a match in the description. Field weights: title 3,
  details 2, brand 1, sub-category 0.5, description 0.5. Each field has its own length normalisation.
* **title factor**: `1 + 1.0 × (share of the query terms that are in the title)`, so up to ×2.
* **quality**: `(1 + 0.1 × (rating − 1) / 4 + 0.05 × discount / 100) × (0.5 if out of stock)`; a missing rating takes
  the mean rating, a missing discount is 0 %. Between 1.000 and 1.141 for products in stock (0.510–0.569 out of stock:
  they stay in the results, lower).

## Test queries and labels

The test queries are numbered **Q1–Q5** (in Part 2 they were queries 3–7, after the two validation queries, which are
now **V1–V2**):

| id | query | AND matches | relevant |
|---|---|---|---|
| Q1 | men white formal shirt slim fit | 25 | 9 |
| Q2 | women floral print shirt | 35 | 27 |
| Q3 | women black track pants with zip pockets | 11 | 8 |
| Q4 | women pure cotton t-shirt pack of 3 | 140 | 26 |
| Q5 | black running shoes for men | 28 | 6 |

`test_queries_labels_all.csv` keeps the 91 labels of Part 2 unchanged (`labelled_in = part 2`) and adds the 148 AND
matches that were not in a Part 2 top-20 (`labelled_in = part 3`), so that every method is evaluated on complete
judgments and recall is exact.
