"""
Ranking algorithms of the search engine (Part 3), rewritten to run on the precomputed ``SearchIndex``.

Every function receives the query terms and the candidate products (pids) and returns ``{pid: score}``.
The formulas are the ones of Part 3; only the implementation changes:

* ``tfidf``    : cosine similarity between TF-IDF vectors, weights tf x log(N / df).
* ``bm25``     : Okapi BM25 with idf = log(N / df), k1 = 1.2, b = 0.75.
* ``ours``     : BM25F (field weights) x title factor x quality factor (rating, discount, stock).
* ``word2vec`` : cosine between the average word2vec vectors of the query and of the product.
"""
import math
from collections import Counter

import numpy as np

METHODS = ("ours", "bm25", "tfidf", "word2vec")
METHOD_NAMES = {
    "ours": "Our score (BM25F x title x quality)",
    "bm25": "BM25",
    "tfidf": "TF-IDF + cosine",
    "word2vec": "word2vec + cosine",
}
TITLE_WEIGHT = 1.0   # title factor of our score: x2 when every query term is in the title


def score_tfidf(index, terms, pids):
    """cos(q, d) with weights tf x idf; |d| is the norm of the full TF-IDF vector of the product."""
    q_vec = {t: c * index.idf[t] for t, c in Counter(terms).items() if t in index.idf}
    q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
    scores = dict.fromkeys(pids, 0.0)
    if not q_norm:
        return scores
    for term, weight in q_vec.items():
        postings, idf = index.postings[term], index.idf[term]
        for pid in pids:
            tf = postings.get(pid)
            if tf:
                scores[pid] += weight * tf * idf
    return {pid: s / (q_norm * index.doc_norm[pid]) if index.doc_norm.get(pid) else 0.0
            for pid, s in scores.items()}


def score_bm25(index, terms, pids):
    """Sum over the distinct query terms of idf x tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))."""
    k1, b = index.k1, index.b
    norm = {pid: k1 * (1 - b + b * index.doc_len[pid] / index.avg_len) for pid in pids}
    scores = dict.fromkeys(pids, 0.0)
    for term in set(terms):
        if term not in index.postings:
            continue
        postings, idf = index.postings[term], index.idf[term]
        for pid in pids:
            tf = postings.get(pid)
            if tf:
                scores[pid] += idf * tf * (k1 + 1) / (tf + norm[pid])
    return scores


def score_bm25f(index, terms, pids):
    """BM25F: sum over the query terms of idf x tf~ (k1 + 1) / (tf~ + k1), with tf~ precomputed in the index."""
    k1 = index.k1
    scores = dict.fromkeys(pids, 0.0)
    for term in set(terms):
        if term not in index.bm25f_tf:
            continue
        postings, idf = index.bm25f_tf[term], index.idf[term]
        for pid in pids:
            tilde = postings.get(pid)
            if tilde:
                scores[pid] += idf * tilde * (k1 + 1) / (tilde + k1)
    return scores


def title_factor(index, terms, pid, weight=TITLE_WEIGHT):
    """1 + weight x (share of the distinct query terms that are in the title of the product)."""
    unique = set(terms)
    if not unique:
        return 1.0
    return 1 + weight * len(unique & index.title_terms[pid]) / len(unique)


def score_ours(index, terms, pids):
    """Our score = BM25F x title factor x quality factor."""
    return {pid: s * title_factor(index, terms, pid) * index.quality[pid]
            for pid, s in score_bm25f(index, terms, pids).items()}


class Word2VecModel:
    """
    word2vec trained on our corpus (Part 3 settings: skip-gram, 100 dimensions, window 5, min_count 2,
    10 epochs, one worker and a fixed seed). The product vectors are averaged and L2-normalised once,
    so the cosine with a query is a single dot product.
    """
    PARAMS = dict(vector_size=100, window=5, min_count=2, sg=1, epochs=10, seed=42, workers=1)

    def __init__(self, index):
        from gensim.models import Word2Vec
        sentences = [fields[f] for f in ("title", "details", "description")
                     for fields in index.field_terms.values() if fields[f]]   # one sentence per field
        self.model = Word2Vec(sentences=sentences, **self.PARAMS)
        self.vectors = {}
        for pid, fields in index.field_terms.items():
            self.vectors[pid] = self._unit(self.text_vector([t for f in fields.values() for t in f]))

    def text_vector(self, terms):
        """Average of the vectors of the terms (terms outside the vocabulary are skipped)."""
        vectors = [self.model.wv[t] for t in terms if t in self.model.wv]
        if not vectors:
            return np.zeros(self.model.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    @staticmethod
    def _unit(vector):
        """The vector divided by its length (unchanged if it is all zeros)."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def score(self, terms, pids):
        """Cosine between the average vector of the query and of each product."""
        q = self._unit(self.text_vector(terms))
        return {pid: float(np.dot(q, self.vectors[pid])) for pid in pids}


def score(method, index, terms, pids, word2vec=None):
    """Score ``pids`` for the query ``terms`` with ``method``; returns {pid: score}."""
    if method == "tfidf":
        return score_tfidf(index, terms, pids)
    if method == "bm25":
        return score_bm25(index, terms, pids)
    if method == "ours":
        return score_ours(index, terms, pids)
    if method == "word2vec":
        if word2vec is None:
            raise RuntimeError("the word2vec model is not trained")
        return word2vec.score(terms, pids)
    raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
