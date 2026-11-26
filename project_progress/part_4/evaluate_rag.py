"""
IRWA Final Project - Part 4: comparison of the baseline RAG (course template) and our improved RAG.

Run from the repository root, with an API key in .env (see the README):

    python project_progress/part_4/evaluate_rag.py                 # Groq (default) or the LLM_PROVIDER of .env
    python project_progress/part_4/evaluate_rag.py --dry-run       # no API calls: checks the pipeline only
    python project_progress/part_4/evaluate_rag.py --summary-only  # no API calls: summary of the saved answers

For every query, the search engine retrieves the products (our score) and both generators write a summary with
the same LLM. Checks, done by code (no human judgement):
* status: a recommendation ("ok") or "no good products";
* hallucinated PIDs: PIDs in the answer that are not among the products given to the model;
* relevant best product: for Q1-Q5 (and the typo query, same need as Q5) every AND match has a relevance label
  (Part 3), so we know whether the recommended product is relevant;
* wrong "no good products": on the labelled queries (Q1-Q5, V1-V2, T1) relevant products were retrieved, so
  answering "no good products" is an error;
* facts: the prices, ratings and discounts written in the explanations are compared with the product data;
* correct abstention: for queries about products that the shop does not sell, the right answer is
  "no good products".
The answers are saved in project_progress/part_4/results/rag_comparison.json and .md (to read them).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from myapp.generation.rag import BaselineRAG, ImprovedRAG, LLMClient   # noqa: E402
from myapp.search.search_engine import SearchEngine                     # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
labels = pd.read_csv(ROOT / "project_progress" / "part_3" / "test_queries_labels_all.csv")
val_labels = pd.read_csv(ROOT / "data" / "validation_labels.csv")

# (id, query, kind, labels): kind "labelled" = relevance labels exist; "absent" = the shop has no such product;
# "open" = a real need without labels (answers read by us).
QUERIES = [(f"Q{qid}", q, "labelled", dict(zip(g["pid"], g["is_relevant"])))
           for qid, (q, g) in ((qid, (g["query"].iloc[0], g)) for qid, g in labels.groupby("query_id"))]
QUERIES += [
    ("V1", "women full sleeve sweatshirt cotton", "labelled",
     dict(zip(val_labels[val_labels.query_id == 1].pid, val_labels[val_labels.query_id == 1].labels))),
    ("V2", "men slim jeans blue", "labelled",
     dict(zip(val_labels[val_labels.query_id == 2].pid, val_labels[val_labels.query_id == 2].labels))),
    ("T1", "black runing shoes for men", "labelled", None),     # typo of Q5: Q5 labels
    ("A1", "men leather wallet", "absent", None),
    ("A2", "women high heels", "absent", None),
    ("A3", "kids school bag", "absent", None),
    ("A4", "wedding saree", "absent", None),
    ("A5", "laptop backpack", "absent", None),
    ("O1", "gift for my father", "open", None),
    ("O2", "comfortable shoes for walking", "open", None),
    ("O3", "cheap cotton shirt for men under 500", "open", None),
]
QUERIES = [(i, q, k, (QUERIES[4][3] if i == "T1" else lab)) for i, q, k, lab in QUERIES]   # T1 uses Q5 labels
FULLY_JUDGED = {"Q1", "Q2", "Q3", "Q4", "Q5", "T1"}   # every AND match has a label (V1-V2: only 20 judged products)


class DryRunClient(LLMClient):
    """Fake LLM for --dry-run: recommends the first product. Only to test the code, never for the report."""

    def available(self):
        """Always available (no key needed)."""
        return True

    def complete(self, user, system=None, temperature=None, json_mode=False, max_tokens=2000):
        """Recommend the first product of the prompt."""
        import re
        pid = re.search(r"PID:? ([A-Z0-9]{16})", user)
        pid = pid.group(1) if pid else "NONE000000000000"
        if json_mode:
            return json.dumps({"status": "ok", "best": {"pid": pid, "why": "dry run"}, "alternatives": [],
                               "summary": "dry run"})
        return f"- Best Product: {pid} (dry run)\n- Why: dry run"


def call_with_retry(generator, query, response, wait):
    """Generate, waiting and retrying when the API rate limit is reached (HTTP 429); other errors stop the run."""
    for attempt in range(4):
        result = generator.generate(query, response)
        error = result.get("error", "")
        if result["status"] != "unavailable":
            break
        if "429" not in error and "rate" not in error.lower():
            sys.exit(f"LLM error (not a rate limit), stopping: {error}")
        time.sleep(wait * (attempt + 2) * 4)
    time.sleep(wait)
    return result


def main():
    """Run both RAG variants on the queries (or only recompute the summary) and save the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="fake LLM, to test the pipeline without a key")
    parser.add_argument("--wait", type=float, default=2.5, help="seconds between API calls (rate limits)")
    parser.add_argument("--summary-only", action="store_true",
                        help="no API calls: recompute the summary from results/rag_comparison.json")
    args = parser.parse_args()

    if args.summary_only:
        with open(OUT / "rag_comparison.json", encoding="utf-8") as f:
            saved = json.load(f)
        engine = SearchEngine(ROOT / "data" / "fashion_products_dataset.json")
        write_summary(saved["meta"], saved["records"], engine)
        return

    client = DryRunClient() if args.dry_run else LLMClient()
    if not client.available():
        sys.exit(f"No API key for {client.provider}: set it in .env (see the README) or use --dry-run.")
    print(f"LLM: {client.provider} / {client.model}{' (DRY RUN)' if args.dry_run else ''}")
    engine = SearchEngine(ROOT / "data" / "fashion_products_dataset.json")
    generators = {"baseline": BaselineRAG(client), "improved": ImprovedRAG(client)}

    records = []
    for qid, query, kind, judged in QUERIES:
        response = engine.search(query, method="ours", top_n=20)
        for name, generator in generators.items():
            result = call_with_retry(generator, query, response, args.wait)
            best = result.get("best_pid")
            relevant = None
            if judged is not None and best in judged:
                relevant = int(judged[best])
            records.append({
                "id": qid, "query": query, "kind": kind, "variant": name, "match_type": response.match_type,
                "retrieved": response.total, "status": result["status"], "best_pid": best,
                "best_title": engine.corpus[best].title if best in engine.corpus else None,
                "best_relevant": relevant, "hallucinated_pids": result["cited_invalid"],
                "latency_ms": round(result["latency_ms"]), "prompt_chars": result["prompt_chars"],
                "answer": result.get("text") or json.dumps({k: result[k] for k in ("status", "best", "alternatives",
                                                                                 "summary")}, ensure_ascii=False),
            })
            print(f"{qid:3s} {name:8s} {result['status']:16s} best={best} relevant={relevant}")

    suffix = "_DRYRUN" if args.dry_run else ""
    OUT.mkdir(exist_ok=True)
    meta = {"provider": client.provider, "model": client.model, "dry_run": args.dry_run,
            "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(OUT / f"rag_comparison{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1, ensure_ascii=False)
    write_summary(meta, records, engine, suffix)


def fact_check(record, engine):
    """
    (checked, wrong): prices ("Rs 799"), ratings ("4.4/5") and discounts ("38%") written in the explanations of the
    improved answer, compared with the data of the product they describe.
    """
    import re
    if record["variant"] != "improved" or record["status"] != "ok":
        return 0, 0
    answer = json.loads(record["answer"])
    checked = wrong = 0
    for pick in [answer.get("best")] + list(answer.get("alternatives") or []):
        if not pick or pick.get("pid") not in engine.corpus:
            continue
        pid, why = pick["pid"], pick.get("why", "")
        truth = {"price": engine.index.price[pid], "rating": engine.index.rating[pid],
                 "discount": engine.index.discount[pid] or 0}
        claims = ([("price", float(x.replace(",", ""))) for x in re.findall(r"Rs\s*([\d,]+)", why)]
                  + [("rating", float(x)) for x in re.findall(r"(\d\.\d)\s*/\s*5", why)]
                  + [("discount", float(x)) for x in re.findall(r"(\d+)\s*%", why)])
        for kind, value in claims:
            checked += 1
            wrong += truth[kind] is None or abs(truth[kind] - value) > (0.05 if kind == "rating" else 1)
    return checked, wrong


def write_summary(meta, records, engine, suffix=""):
    """Summary table (CSV) and readable answers (Markdown) from the saved records."""
    frame = pd.DataFrame(records)
    rows = []
    for name, g in frame.groupby("variant", sort=False):
        judged = g[g["id"].isin(FULLY_JUDGED)]          # every retrieved product has a label
        labelled = g[g["kind"] == "labelled"]
        absent = g[g["kind"] == "absent"]
        facts = [fact_check(r, engine) for r in g.to_dict("records")]
        rows.append({
            "variant": name,
            "relevant recommendation (Q1-Q5, T1)": f"{int((judged['best_relevant'] == 1).sum())}/{len(judged)}",
            "'no good products' although relevant products were retrieved (labelled queries)":
                f"{int(labelled['status'].isin(['no_good_products', 'no_results']).sum())}/{len(labelled)}",
            "correct 'no good products' (products the shop does not sell)":
                f"{int(absent['status'].isin(['no_good_products', 'no_results']).sum())}/{len(absent)}",
            "answers with invented PIDs": f"{int((g['hallucinated_pids'] > 0).sum())}/{len(g)}",
            "prices, ratings and discounts in the explanations that match the data":
                (f"{sum(c for c, _ in facts) - sum(w for _, w in facts)}/{sum(c for c, _ in facts)}"
                 if name == "improved" else "no data in the prompt"),
            "LLM calls": int((g["prompt_chars"] > 0).sum()),
            "median latency (ms)": int(g.loc[g["prompt_chars"] > 0, "latency_ms"].median()),
            "median prompt (characters)": int(g.loc[g["prompt_chars"] > 0, "prompt_chars"].median()),
        })
    summary = pd.DataFrame(rows).set_index("variant").T
    summary.to_csv(OUT / f"rag_summary{suffix}.csv")
    print("\n", summary.to_string())

    with open(OUT / f"rag_comparison{suffix}.md", "w", encoding="utf-8") as f:
        f.write(f"# RAG comparison ({meta['provider']} / {meta['model']}, {meta['date']}"
                f"{', DRY RUN' if meta.get('dry_run') else ''})\n\n")
        f.write(summary.to_markdown() + "\n\n")
        for (qid, query), g in frame.groupby(["id", "query"], sort=False):
            f.write(f"## {qid}: {query}\n\n")
            for _, r in g.iterrows():
                f.write(f"**{r['variant']}** ({r['status']}, best = {r['best_pid']}, relevant = {r['best_relevant']}, "
                        f"invented PIDs = {r['hallucinated_pids']})\n\n```\n{r['answer']}\n```\n\n")
    print(f"\nSaved to {OUT}/rag_comparison{suffix}.json, .md and rag_summary{suffix}.csv")


if __name__ == "__main__":
    main()
