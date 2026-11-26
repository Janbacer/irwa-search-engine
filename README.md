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
| 3 | Ranking and Filtering | [`project_progress/part_3`](project_progress/part_3/) | `IRWA-2026-part-3` | delivered |
| 4 | RAG, User Interface and Web Analytics | [`project_progress/part_4`](project_progress/part_4/), `web_app.py`, [`myapp/`](myapp/) | `IRWA-2026-part-4` | delivered |

Each part folder contains the report (PDF and LaTeX source), a short README and the code of that part: a notebook
for Parts 1–3, and the evaluation scripts for Part 4 (whose application code is in `web_app.py` and `myapp/`).

## Repository structure

```
data/                     dataset (not in the repository) + validation_labels.csv
project_progress/part_N/  report, README, notebook or scripts, results and figures of each part
web_app.py                Flask web application (Part 4): pages, tracking hooks
myapp/search/             search engine of the web app: pre-processing, index, ranking algorithms, search()
myapp/generation/         RAG: baseline (course template) and our improved version
myapp/analytics/          web analytics: data collection and storage, dashboard, traffic simulation
templates/, static/       HTML templates, CSS and the chart libraries (static/vendor)
.env.example              settings of the web app (copy it to .env)
requirements.txt          Python dependencies
```

## Setup (once)

1. Clone the repository:
   ```bash
   git clone https://github.com/Janbacer/irwa-search-engine.git
   cd irwa-search-engine
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

## Part 3 — Ranking and Filtering

1. Check out the delivered version: `git checkout IRWA-2026-part-3`
2. Install the requirements again if your environment comes from Part 2: Part 3 adds `gensim` (word2vec).
3. Run the Part 1 notebook first if `data/processed_corpus.pkl` does not exist yet (see above).
4. Run the notebook from its folder:
   ```bash
   cd project_progress/part_3
   jupyter notebook IRWA_Part3.ipynb
   ```
   Select *Run → Run All Cells* (or `jupyter nbconvert --to notebook --execute --inplace IRWA_Part3.ipynb`). It takes
   about one minute (half of it is training word2vec) and writes the figures to `figures/` and the result tables to
   `results/`.

All the rankings are in `project_progress/part_3/irwa_ranking.py`. The ranking method is selected with the `method`
argument of `Ranker.search`; every method ranks the same products (the conjunctive AND matches of Part 2):

| `method` | ranking |
|---|---|
| `"tfidf"` | TF-IDF vectors + cosine similarity |
| `"bm25"` | Okapi BM25 (`k1 = 1.2`, `b = 0.75`; change them with `Ranker(corpus, k1=..., b=...)`) |
| `"ours"` | our score: field-weighted BM25 (BM25F) × title factor × quality factor (rating, discount, stock) |
| `"word2vec"` | average word2vec vector of the query and of the product + cosine similarity (call `train_word2vec()` first) |

```python
import pandas as pd
import irwa_ranking as rk         # project_progress/part_3 (needs part_1 and part_2 in the path)
corpus = pd.read_pickle("data/processed_corpus.pkl")
ranker = rk.Ranker(corpus)
ranker.search("women floral print shirt", method="ours", top_k=10)   # [(pid, score), ...]
ranker.train_word2vec()            # about 30 s, same model every run (fixed seed)
ranker.search("women floral print shirt", method="word2vec", top_k=20)
```

The field weights, the title weight and the quality-factor weights are constants at the top of `irwa_ranking.py`. More details in the
[Part 3 README](project_progress/part_3/README.md).

## Part 4 — RAG, User Interface and Web Analytics

### Run the web application

1. Check out the delivered version: `git checkout IRWA-2026-part-4`, and install the requirements again
   (`pip install -r requirements.txt`): Part 4 adds `anthropic` and `tabulate`.
2. The dataset must be in `data/fashion_products_dataset.json` (step 2 of the setup). Part 4 does **not** need the
   notebooks of Parts 1–3: the web app builds its own index from the JSON file when it starts (a few seconds).
3. Create the settings file and add an LLM key for the AI summary (optional: without a key the app works and the
   summary box says that the RAG is not available):
   ```bash
   cp .env.example .env              # Windows: copy .env.example .env
   ```
   Then edit `.env`: put a free Groq key (https://console.groq.com/keys) in `GROQ_API_KEY`, or set
   `LLM_PROVIDER = "anthropic"` and `ANTHROPIC_API_KEY`. The default Groq model is `openai/gpt-oss-20b`: the
   template's `llama-3.1-8b-instant` is no longer offered by Groq.
4. Start the server from the repository root and open http://127.0.0.1:8088/
   ```bash
   python web_app.py
   ```

| page | what it shows |
|---|---|
| `/` | search box, ranking algorithm and example queries |
| `/search?q=...` | results: title, description, price, discount, rating, stock, brand, image, link to the original website; filters (category, price, rating, discount, in stock), sorting, pages, and the AI summary (RAG) |
| `/doc/<pid>` | details page with every field of the product |
| `/dashboard` | web analytics: key indicators and charts with explanations |
| `/stats` | the analytics tables (star schema), downloadable as CSV or JSON |

### Selecting the algorithms and options

* **Ranking algorithm**: drop-down on the search and results pages, or the URL parameter `method`:
  `ours` (default: BM25F × title factor × quality, Part 3), `bm25`, `tfidf`, `word2vec` (trained on its first use,
  about 30 s, or at start-up with `TRAIN_WORD2VEC = 1`).
* **Sorting**: `sort=relevance|price_asc|price_desc|rating|discount`. **Filters**: `sub_category`, `min_price`,
  `max_price`, `min_rating`, `min_discount`, `in_stock=1`. A price in the query ("shirt under 500") becomes a filter.
* **RAG variant**: `rag=improved` (default) or `rag=baseline` (the template's pipeline); the results page also has a
  link to show the baseline summary next to ours.
* From Python, the general search function is `SearchEngine.search(query, method, filters, sort, page, page_size)`
  in `myapp/search/search_engine.py`:
  ```python
  from myapp.search.search_engine import SearchEngine, SearchFilters
  engine = SearchEngine("data/fashion_products_dataset.json")
  response = engine.search("black running shoes for men", method="bm25", filters=SearchFilters(in_stock_only=True))
  [(r.rank, r.pid, r.title) for r in response.results[:5]]
  ```

### Web analytics data

Everything is kept in memory and is lost when the server stops. Optional persistence: set
`ANALYTICS_FILE = "data/analytics.json"` in `.env` and the tables are saved to that JSON file after each change and
loaded at start-up. To see the dashboard with data before real users arrive, generate simulated traffic (it goes
through the real application with the Flask test client; the dashboard says how many sessions are simulated):
```bash
python -m myapp.analytics.simulate --visitors 60 --days 14 --output data/analytics_simulated.json
ANALYTICS_FILE=data/analytics_simulated.json python web_app.py
# Windows (cmd): set ANALYTICS_FILE=data/analytics_simulated.json && python web_app.py
```

### Reproduce the results of the report

```bash
python project_progress/part_4/evaluate_search.py    # search engine: checks, metrics, speed (about 2 min)
python project_progress/part_4/evaluate_rag.py       # RAG: baseline vs improved (needs an API key in .env)
python project_progress/part_4/evaluate_rag.py --summary-only   # same summary from the saved answers, no key
```
More details in the [Part 4 README](project_progress/part_4/README.md).

## Attribution

Project structure and web application from the course template
[trokhymovych/irwa-search-engine](https://github.com/trokhymovych/irwa-search-engine). The dashboard charts use
Vega, Vega-Lite and Vega-Embed (BSD-3-Clause), copied in `static/vendor/`.
