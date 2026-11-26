# Part 4 — RAG, User Interface and Web Analytics

Tag: `IRWA-2026-part-4`. How to install and run the web application, and how to choose the algorithms and options,
is in the [main README](../../README.md#part-4--rag-user-interface-and-web-analytics).

| file | content |
|---|---|
| `IRWA-2026-u231867-u231747-u233694-part-4.pdf` | report (and its LaTeX source, `.tex`) |
| `evaluate_search.py` | evaluation of the search engine of the web app: reproduction of Part 3, metrics with the Part 3 labels, speed, fallback examples |
| `evaluate_rag.py` | comparison of the baseline RAG (course template) and our improved RAG on 16 queries |
| `results/` | the tables of the report, written by the two scripts |
| `figures/` | screenshots and diagrams of the report |

The code of the web application is outside this folder, where the course template puts it:

| file | content |
|---|---|
| `web_app.py` | Flask application: pages, tracking hooks (`before_request` / `after_request`), RAG endpoint |
| `myapp/search/preprocessing.py` | Part 1 text pre-processing, keeping numbers |
| `myapp/search/index.py` | index built at start-up: postings, statistics, precomputed BM25F frequencies, product fields |
| `myapp/search/algorithms.py` | TF-IDF + cosine, BM25, our score, word2vec (Part 3 formulas) |
| `myapp/search/search_engine.py` | the general `search()` function: query understanding, AND search, fallback, filters, sorting, pages |
| `myapp/generation/rag.py` | `BaselineRAG` (template), `ImprovedRAG` (ours), LLM client (Groq or Anthropic) |
| `myapp/analytics/analytics_data.py` | analytics tables (star schema), sessions, missions, clicks, dwell time |
| `myapp/analytics/context.py` | browser, OS, device, IP, country and city of a request |
| `myapp/analytics/dashboard.py` | indicators, charts and tables of the dashboard |
| `myapp/analytics/simulate.py` | traffic simulation through the real application (for the dashboard demo) |
| `templates/`, `static/` | pages, styles, and the chart libraries (`static/vendor/`) |

## Reproduce the results

From the repository root, with the environment of the main README and the dataset in `data/`:

```bash
python project_progress/part_4/evaluate_search.py   # about 2 minutes (it trains word2vec twice)
python project_progress/part_4/evaluate_rag.py      # needs an LLM key in .env (about 5 min with the free Groq tier)
python project_progress/part_4/evaluate_rag.py --summary-only   # no key: recomputes the summary of the saved answers
python project_progress/part_4/evaluate_rag.py --dry-run        # no key: checks the code with a fake LLM
python -m myapp.analytics.simulate --visitors 60 --days 14 --output data/analytics_simulated.json
```

* `evaluate_search.py` needs no API and gives the same numbers on every run.
* `evaluate_rag.py` calls the LLM (`openai/gpt-oss-20b` on Groq by default; the template's `llama-3.1-8b-instant` is
  no longer offered), whose answers can change a little between runs. The answers used in the report are saved in
  `results/rag_comparison.json` (model and date included), and `--summary-only` recomputes `results/rag_summary.csv`
  from them without any API call.
* The simulation uses a fixed seed (`--seed 7`), so it produces the same visitors and actions; only the dates move,
  because the simulated days end today.
