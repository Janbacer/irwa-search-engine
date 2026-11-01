"""
IRWA Final Project - Part 2: evaluation metrics.

Adapted from Practice Session 2 (P@K, AP@K, MAP, MRR, NDCG); R@K and F1@K are new.
As in the practice, one-query metrics take the binary labels ``y_true`` and the scores
``y_score`` of the returned products, and multi-query metrics take a DataFrame with the
columns ``query_id, pid, predicted_relevance, is_relevant``.

Main changes:
- ties keep the engine's order (stable sort), so we evaluate exactly the ranking it returns;
- R@K, F1@K and NDCG@K can also count the relevant products that the AND search missed
  (``n_relevant`` / ``y_ideal``, taken from all the judged products of the query);
- NDCG is not rounded inside the function, only when the results are reported.
"""
import numpy as np
import pandas as pd


def _sort_by_score(y_true, y_score):
    """Labels sorted by decreasing score; ties keep the input order."""
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    return np.take(np.asarray(y_true), order)


def precision_at_k(y_true, y_score, k=10):
    """P@K: fraction of the top K that is relevant."""
    y_true = _sort_by_score(y_true, y_score)
    return np.sum(y_true[:k]) / k


def recall_at_k(y_true, y_score, k=10, n_relevant=None):
    """R@K: fraction of the relevant products that is in the top K (n_relevant: all relevant products)."""
    if n_relevant is None:
        n_relevant = np.sum(y_true)
    if n_relevant == 0:
        return 0
    y_true = _sort_by_score(y_true, y_score)
    return np.sum(y_true[:k]) / n_relevant


def avg_precision_at_k(y_true, y_score, k=10):
    """AP@K (practice definition): mean of P@i at each relevant rank i <= K."""
    y_true = _sort_by_score(y_true, y_score)
    prec_at_i_list = []
    number_of_relevant = 0
    for i in range(min(k, len(y_true))):
        if y_true[i] == 1:
            number_of_relevant += 1
            prec_at_i_list.append(number_of_relevant / (i + 1))
    if number_of_relevant == 0:
        return 0
    return np.sum(prec_at_i_list) / number_of_relevant


def f1_at_k(y_true, y_score, k=10, n_relevant=None):
    """F1@K: harmonic mean of P@K and R@K."""
    precision = precision_at_k(y_true, y_score, k)
    recall = recall_at_k(y_true, y_score, k, n_relevant)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def rr_at_k(y_true, y_score, k=10):
    """RR@K: 1 / rank of the first relevant product in the top K (0 if none)."""
    y_true = _sort_by_score(y_true, y_score)[:k]
    if np.sum(y_true) == 0:
        return 0
    return 1 / (np.argmax(y_true) + 1)


def dcg_at_k(y_true, y_score, k=10):
    """DCG@K with the practice formula: sum of (2^rel - 1) / log2(rank + 1)."""
    y_true = _sort_by_score(y_true, y_score)[:k]
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_at_k(y_true, y_score, k=10, y_ideal=None):
    """NDCG@K: DCG@K / DCG@K of the ideal ranking (y_ideal: labels of all judged products)."""
    if y_ideal is None:
        y_ideal = y_true
    dcg_max = dcg_at_k(y_ideal, y_ideal, k)
    if not dcg_max:
        return 0
    return dcg_at_k(y_true, y_score, k) / dcg_max


def _query_data(search_res, judgments, q):
    """Labels and scores of query q, plus its number of relevant products and judged labels."""
    curr_data = search_res[search_res["query_id"] == q]
    labels = np.array(curr_data["is_relevant"])
    scores = np.array(curr_data["predicted_relevance"])
    if judgments is None:
        return labels, scores, None, None
    judged = np.array(judgments.loc[judgments["query_id"] == q, "is_relevant"])
    return labels, scores, judged.sum(), judged


def map_at_k(search_res, k=10):
    """MAP@K: mean AP@K over the queries (and the list of AP@K)."""
    avp = []
    for q in search_res["query_id"].unique():
        labels, scores, _, _ = _query_data(search_res, None, q)
        avp.append(avg_precision_at_k(labels, scores, k))
    return np.sum(avp) / len(avp), avp


def mrr_at_k(search_res, k=10):
    """MRR@K: mean RR@K over the queries (and the list of RR@K)."""
    RRs = []
    for q in search_res["query_id"].unique():
        labels, scores, _, _ = _query_data(search_res, None, q)
        RRs.append(rr_at_k(labels, scores, k))
    return np.sum(RRs) / len(RRs), RRs


def mean_ndcg_at_k(search_res, k=10, judgments=None):
    """Average NDCG@K over the queries (and the list of NDCG@K)."""
    ndcgs = []
    for q in search_res["query_id"].unique():
        labels, scores, _, judged = _query_data(search_res, judgments, q)
        ndcgs.append(ndcg_at_k(labels, scores, k, judged))
    return np.sum(ndcgs) / len(ndcgs), ndcgs


def evaluate(search_res, ks=(5, 10, 20), judgments=None):
    """
    Every metric for each query and K, plus a "mean" row (mean AP = MAP, mean RR = MRR).
    judgments: all judged products (query_id, pid, is_relevant), returned by the search or not.
    """
    rows = {}
    for q in search_res["query_id"].unique():
        labels, scores, n_relevant, judged = _query_data(search_res, judgments, q)
        row = {}
        for k in ks:
            row[("P", k)] = precision_at_k(labels, scores, k)
            row[("R", k)] = recall_at_k(labels, scores, k, n_relevant)
            row[("AP", k)] = avg_precision_at_k(labels, scores, k)
            row[("F1", k)] = f1_at_k(labels, scores, k, n_relevant)
            row[("RR", k)] = rr_at_k(labels, scores, k)
            row[("NDCG", k)] = ndcg_at_k(labels, scores, k, judged)
        rows[q] = row
    table = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["metric", "K"])
    table.index.name = "query_id"
    table.loc["mean"] = table.mean()
    return table
