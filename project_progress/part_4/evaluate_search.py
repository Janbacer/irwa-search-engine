"""
IRWA Final Project - Part 4: evaluation of the search engine of the web application.

Run from the repository root:  python project_progress/part_4/evaluate_search.py

1. Check that the web-app engine reproduces Part 3: with the Part 1 pre-processing (numbers removed) its
   top 20 for Q1-Q5 must be the same as the Part 3 results (project_progress/part_3/results).
2. Effect of keeping the numbers: metrics of the four methods on Q1-Q5 (Part 3 labels, every AND match is
   judged) and V1-V2 (course labels), with the Part 2 evaluation functions, K = 5, 10, 20.
3. Speed: time per search of the Part 3 implementation and of the web-app engine, for the test queries and
   for broad queries with thousands of matches.
4. The fallback for queries without results (spelling suggestion, partial matches).

The tables are written to project_progress/part_4/results/.
"""
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "project_progress" / "part_1"), str(ROOT / "project_progress" / "part_2"),
                str(ROOT / "project_progress" / "part_3")]

import irwa_evaluation as ev                       # Part 2 metrics
from myapp.search.search_engine import SearchEngine

DATA = ROOT / "data" / "fashion_products_dataset.json"
PART3 = ROOT / "project_progress" / "part_3"
OUT = Path(__file__).resolve().parent / "results"
METHODS = ("tfidf", "bm25", "ours", "word2vec")
NAMES = {"tfidf": "TF-IDF + cosine", "bm25": "BM25", "ours": "Our score", "word2vec": "word2vec + cosine"}
BROAD_QUERIES = ["men", "women tshirt", "shirt", "black jeans"]

labels = pd.read_csv(PART3 / "test_queries_labels_all.csv")
val_labels = (pd.read_csv(ROOT / "data" / "validation_labels.csv")
              .rename(columns={"labels": "is_relevant"})[["query_id", "pid", "is_relevant"]])
TEST_QUERIES = labels.groupby("query_id")["query"].first().to_dict()                # Q1-Q5
VAL_QUERIES = {1: "women full sleeve sweatshirt cotton", 2: "men slim jeans blue"}   # V1-V2


def ranking(engine, query, method):
    """All the AND matches of the query, in the order of the method (no fallback, no filters)."""
    response = engine.search(query, method=method, page_size=10 ** 6, allow_fallback=False)
    return [item.pid for item in response.results]


def evaluation_input(engine, method, queries, judgments):
    """Judged products returned by the search, in rank order (Part 2 format; the score is minus the rank)."""
    rows = []
    for qid, query in queries.items():
        judged = judgments[judgments["query_id"] == qid].set_index("pid")["is_relevant"]
        for rank, pid in enumerate(ranking(engine, query, method), start=1):
            if pid in judged.index:
                rows.append({"query_id": qid, "pid": pid, "rank": rank, "predicted_relevance": -rank,
                             "is_relevant": int(judged[pid])})
    return pd.DataFrame(rows)


def metrics(engine, queries, judgments):
    """Mean metrics of every method (one row per method)."""
    rows = {}
    for method in METHODS:
        table = ev.evaluate(evaluation_input(engine, method, queries, judgments), ks=(5, 10, 20),
                            judgments=judgments)
        mean = table.loc["mean"]
        rows[NAMES[method]] = {f"{m}@{k}": mean[(m, k)] for k in (5, 10, 20) for m in ("P", "R", "AP", "NDCG")}
        rows[NAMES[method]]["MRR"] = mean[("RR", 10)]
    return pd.DataFrame(rows).T


def ndcg_per_query(engine):
    """NDCG@10 of each test query and method."""
    out = {}
    for method in METHODS:
        table = ev.evaluate(evaluation_input(engine, method, TEST_QUERIES, labels), ks=(10,), judgments=labels)
        out[NAMES[method]] = table[("NDCG", 10)].drop("mean")
    frame = pd.DataFrame(out)
    frame.index = [f"Q{q}" for q in frame.index]
    return frame


def timing(fn, repeats=20):
    """Median time of fn() in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    """Run the four evaluations and write their tables to results/."""
    OUT.mkdir(exist_ok=True)

    print("Loading the engines...")
    engine = SearchEngine(DATA, keep_numbers=True)
    engine.train_word2vec()
    engine_part1 = SearchEngine(DATA, keep_numbers=False)        # Part 1 pre-processing
    engine_part1.train_word2vec()

    # ---- 1. reproduction of Part 3
    part3 = pd.read_csv(PART3 / "results" / "top20_all_methods.csv")
    same = {}
    for method in METHODS:
        equal = 0
        for qid, query in TEST_QUERIES.items():
            expected = part3[(part3["method"] == method) & (part3["query_id"] == qid)].sort_values("rank")["pid"]
            equal += ranking(engine_part1, query, method)[:20] == list(expected)
        same[NAMES[method]] = f"{equal}/5"
    print("Top 20 identical to Part 3 (Part 1 pre-processing):", same)
    pd.Series(same, name="queries with the same top 20 as Part 3").to_csv(OUT / "reproduction_part3.csv")

    # ---- 2. metrics with and without numbers
    for name, eng in (("part1_preprocessing", engine_part1), ("web_app", engine)):
        test = metrics(eng, TEST_QUERIES, labels)
        val = metrics(eng, VAL_QUERIES, val_labels)
        test.to_csv(OUT / f"metrics_test_{name}.csv", float_format="%.3f")
        val.to_csv(OUT / f"metrics_validation_{name}.csv", float_format="%.3f")
        print(f"\nTest queries, {name}:\n", test[["P@10", "AP@10", "NDCG@10", "MRR"]].round(3))
        print(f"Validation queries, {name}:\n", val[["P@10", "AP@10", "NDCG@10", "MRR"]].round(3))
    per_query = pd.concat({"Part 1 pre-processing": ndcg_per_query(engine_part1),
                           "numbers kept (web app)": ndcg_per_query(engine)}, axis=1)
    per_query.to_csv(OUT / "ndcg10_per_query.csv", float_format="%.3f")
    print("\nNDCG@10 per query:\n", per_query.round(3))
    matches = {f"Q{qid}": (len(ranking(engine_part1, q, "ours")), len(ranking(engine, q, "ours")))
               for qid, q in TEST_QUERIES.items()}
    print("AND matches (Part 1 pre-processing, web app):", matches)

    # ---- 3. speed: Part 3 implementation against the web-app engine (same pre-processing)
    import irwa_preprocessing as pp1
    import irwa_ranking as rk
    import json
    with open(DATA, encoding="utf-8") as f:
        corpus = pp1.preprocess_corpus(json.load(f))
    ranker = rk.Ranker(corpus)
    rows = []
    for query in list(TEST_QUERIES.values()) + BROAD_QUERIES:
        n = len(ranker.match(query)[1])
        for method in ("tfidf", "bm25", "ours"):
            rows.append({"query": query, "matches": n, "method": NAMES[method],
                         "part3_ms": timing(lambda: ranker.search(query, method)),
                         "web_app_ms": timing(lambda: engine_part1.search(query, method=method,
                                                                           allow_fallback=False))})
    speed = pd.DataFrame(rows)
    speed["speed-up"] = speed["part3_ms"] / speed["web_app_ms"]
    speed.to_csv(OUT / "speed.csv", index=False, float_format="%.2f")
    print("\nSpeed (median ms per search):\n",
          speed.groupby("method")[["part3_ms", "web_app_ms", "speed-up"]].median().round(2))
    print(speed[speed["matches"] > 1000].round(1).to_string(index=False))

    # ---- 4. fallback examples
    rows = []
    for query in ["black runing shoes for men", "jens blue men", "women floral sweatshirt hoodie zipper xyzzy"]:
        r = engine.search(query)
        rows.append({"query": query, "match_type": r.match_type, "corrected_query": r.corrected_query,
                     "matched_terms": f"{r.matched_terms}/{len(set(r.terms))}" if r.match_type == "partial" else "",
                     "results": r.total, "first_result": r.results[0].title if r.results else ""})
    fallback = pd.DataFrame(rows)
    fallback.to_csv(OUT / "fallback_examples.csv", index=False)
    print("\nFallback:\n", fallback.to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
