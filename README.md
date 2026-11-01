# IRWA Final Project 2026 — Fashion products search engine

Information Retrieval and Web Analytics, Universitat Pompeu Fabra.

| name | U-number | NIA |
|---|---|---|
| Èric Amargant | u231867 | 284566 |
| Josep Cardona | u231747 | 284888 |
| Jan Balayla | u233694 | 286572 |

## Deliverables

| part | topic | folder | tag | status |
|---|---|---|---|---|
| 1 | Text Processing and Exploratory Data Analysis | [`project_progress/part_1`](project_progress/part_1/) | `IRWA-2026-part-1` | delivered |
| 2 | Indexing and Evaluation | [`project_progress/part_2`](project_progress/part_2/) | `IRWA-2026-part-2` | delivered |
| 3 | Ranking and Filtering | [`project_progress/part_3`](project_progress/part_3/) | `IRWA-2026-part-3` | — |
| 4 | RAG, User Interface and Web Analytics | `web_app.py`, [`myapp/`](myapp/) | `IRWA-2026-part-4` | — |

Each part folder contains the report (PDF), the notebook and a short README.

## Repository structure

```
data/                     dataset (not in the repository) + validation_labels.csv
project_progress/part_N/  report, notebook and code of each part
myapp/search/             web app search code (course template; our engine is plugged in in Part 4)
web_app.py, templates/,   Flask web application (Part 4, from the course template)
static/
requirements.txt          Python dependencies
```

## Setup (once)

1. Clone the repository:
   ```bash
   git clone https://github.com/Janbacer/Final_Project_IRWA.git
   cd Final_Project_IRWA
   ```
2. Copy `fashion_products_dataset.json` (from Aula Global) into the `data/` folder.
   It is not in the repository because of its size (66 MB).
3. Create a virtual environment (Python 3.10 or newer) and install the requirements:
   ```bash
   python -m venv irwa_venv
   irwa_venv\Scripts\activate        # Windows
   source irwa_venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Part 1 — Text Processing and EDA

1. Check out the delivered version: `git checkout IRWA-2026-part-1`
2. Run the notebook from its folder:
   ```bash
   cd project_progress/part_1
   jupyter notebook IRWA_Part1.ipynb
   ```
   Select *Run → Run All Cells* (or run it without the browser:
   `jupyter nbconvert --to notebook --execute --inplace IRWA_Part1.ipynb`). It takes a few minutes and produces:
   * `figures/*.png`: the plots used in the report;
   * `data/processed_corpus.pkl`: the processed corpus (one row per `pid`), used in the next parts.

The pre-processing functions are in `project_progress/part_1/irwa_preprocessing.py`:

| function | what it does |
|---|---|
| `preprocess_text(text)` | lowercase → fashion words (t-shirt → tshirt...) → remove punctuation → split → remove stop words and numbers → Porter stemming |
| `preprocess_document(doc)` | one product → original output fields + processed text fields + numeric fields |
| `preprocess_corpus(docs)` | all products → DataFrame indexed by `pid` |
| `recover_brand(brand, url)` | full brand name from the product URL (the brand field is cut off) |
| `parse_price`, `parse_discount`, `parse_rating` | `"2,999"`, `"69% off"`, `"3.9"` → numbers |

```python
import irwa_preprocessing as pp
pp.preprocess_text("men slim jeans blue")   # ['men', 'slim', 'jean', 'blue']
```

## Part 2 — Indexing and Evaluation

1. Check out the delivered version: `git checkout IRWA-2026-part-2`
2. Run the Part 1 notebook first if `data/processed_corpus.pkl` does not exist yet (see above).
3. Run the notebook from its folder:
   ```bash
   cd project_progress/part_2
   jupyter notebook IRWA_Part2.ipynb
   ```
   Select *Run → Run All Cells* (or `jupyter nbconvert --to notebook --execute --inplace IRWA_Part2.ipynb`). It builds the
   index (about 10 seconds), ranks the 7 queries and computes the evaluation metrics.

| file (`project_progress/part_2/`) | content |
|---|---|
| `irwa_indexing.py` | inverted index with positions, conjunctive (AND) search and TF-IDF ranking (adapted from Practice Session 1) |
| `irwa_evaluation.py` | P@K, R@K, AP@K, F1@K, MAP, MRR and NDCG (adapted from Practice Session 2) |
| `test_queries_labels.csv` | our relevance judgments for the top 20 results of our five test queries |

```python
import irwa_preprocessing as pp   # project_progress/part_1
import irwa_indexing as ix        # project_progress/part_2
index, tf, df, idf = ix.create_index_tfidf(ix.build_docs_terms(corpus, "all_terms"))
ix.search_tf_idf(pp.preprocess_text("men slim jeans blue"), index, tf, idf)   # [(pid, score), ...]
```

More details in the [Part 2 README](project_progress/part_2/README.md).

## Web application (Part 4)

Not used yet. From the course template: put your settings in a `.env` file (never commit it), then run
`python web_app.py` and open http://127.0.0.1:8088/.

## Attribution

Project structure and web application from the course template
[trokhymovych/irwa-search-engine](https://github.com/trokhymovych/irwa-search-engine).
