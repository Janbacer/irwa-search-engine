"""
IRWA Final Project - Part 3: ranking the conjunctive (AND) matches.

Every method ranks the same products (the ones that contain all the query terms, Part 2
``irwa_indexing.search``); only the order changes. Methods (``method`` argument of ``Ranker.search``):

- ``"tfidf"``    : TF-IDF vectors + cosine similarity
- ``"bm25"``     : Okapi BM25
- ``"ours"``     : BM25F (field weights) x title factor x quality factor (rating, discount, stock)
- ``"word2vec"`` : average word2vec vector of the query and of the product + cosine similarity

>>> ranker = Ranker(corpus)          # corpus = Part 1 processed_corpus.pkl
>>> ranker.search("men slim jeans blue", method="ours", top_k=20)   # [(pid, score), ...]
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

import irwa_indexing as ix          # Part 2: inverted index and AND search
import irwa_preprocessing as pp     # Part 1: text pre-processing

METHODS = ("tfidf", "bm25", "ours", "word2vec")

# BM25 (also used inside BM25F)
BM25_K1 = 1.2    # term-frequency saturation
BM25_B = 0.75    # length normalisation: 0 = none, 1 = full

# Our score. Field weights from the Part 1 analysis: the title and the product details hold the query
# terms of relevant products; the description holds as many for non-relevant ones; sub_category is broad.
FIELD_WEIGHTS = {"title_terms": 3.0, "details_terms": 2.0, "brand_terms": 1.0,
                 "sub_category_terms": 0.5, "description_terms": 0.5}
TITLE_WEIGHT = 1.0         # x2 when every query term is in the title
RATING_WEIGHT = 0.1        # at most +10% (rating 5)
DISCOUNT_WEIGHT = 0.05     # at most +5% (100% discount)
OUT_OF_STOCK_FACTOR = 0.5  # an out-of-stock product keeps half of its score

# word2vec (gensim): one worker + fixed seed = the same vectors on every run.
W2V_PARAMS = dict(vector_size=100, window=5, min_count=2, sg=1, epochs=10, seed=42, workers=1)
W2V_FIELDS = ("title_terms", "details_terms", "description_terms")  # one sentence per field


def _sort(scores):
    """[(pid, score)] sorted by decreasing score; ties broken by pid (as in Part 2)."""
    return sorted(scores, key=lambda pair: (-pair[1], pair[0]))


class Ranker:
    """AND search + the four ranking methods. The constructor builds the Part 2 index on all_terms and
    the statistics of the scores; word2vec is trained separately with ``train_word2vec``."""

    def __init__(self, corpus, k1=BM25_K1, b=BM25_B, field_weights=None, title_weight=TITLE_WEIGHT):
        self.corpus = corpus.set_index("pid", drop=False) if corpus.index.name != "pid" else corpus
        self.k1, self.b = k1, b
        self.field_weights = dict(FIELD_WEIGHTS if field_weights is None else field_weights)
        self.title_weight = title_weight

        docs_terms = ix.build_docs_terms(self.corpus, "all_terms")
        self.index = ix.create_index(docs_terms)
        self.N = len(docs_terms)
        self.df = {term: len(postings) for term, postings in self.index.items()}
        self.idf = {t: math.log(self.N / n) for t, n in self.df.items()}   # log(N / df), as in the labs

        self.counts = {pid: Counter(terms) for pid, terms in docs_terms.items()}
        self.doc_len = {pid: len(terms) for pid, terms in docs_terms.items()}
        self.avg_len = float(np.mean(list(self.doc_len.values())))
        self.doc_norm = {pid: math.sqrt(sum((tf * self.idf[t]) ** 2 for t, tf in cnt.items()))
                         for pid, cnt in self.counts.items()}             # norm of the full TF-IDF vector

        self.field_counts = {f: {pid: Counter(t) for pid, t in zip(self.corpus["pid"], self.corpus[f])}
                             for f in self.field_weights}
        self.field_len = {f: {pid: len(t) for pid, t in zip(self.corpus["pid"], self.corpus[f])}
                          for f in self.field_weights}
        self.field_avg_len = {f: max(float(np.mean(list(lens.values()))), 1e-9)
                              for f, lens in self.field_len.items()}
        self.title_terms = {pid: set(t) for pid, t in zip(self.corpus["pid"], self.corpus["title_terms"])}
        self.quality = quality_factor(self.corpus)

        self.w2v = None
        self.doc_vectors = None

    # ------------------------------------------------------------------ search
    def match(self, query):
        """Query terms (Part 1 cleaning) and pids of the products that contain ALL of them."""
        terms = pp.preprocess_text(query)
        return terms, ix.search(terms, self.index)

    def search(self, query, method="bm25", top_k=None):
        """AND search + ranking with method ("tfidf", "bm25", "ours" or "word2vec"): [(pid, score)]."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        terms, docs = self.match(query)
        scorer = {"tfidf": self.score_tfidf, "bm25": self.score_bm25,
                  "ours": self.score_ours, "word2vec": self.score_word2vec}[method]
        ranked = _sort(scorer(terms, docs))
        return ranked if top_k is None else ranked[:top_k]

    # ------------------------------------------------------------------ 1a. TF-IDF + cosine
    def score_tfidf(self, terms, docs):
        """cos(q, d) with weights tf x log(N/df), tf = raw count; |d| uses ALL the terms of the product."""
        q_vec = {t: c * self.idf.get(t, 0.0) for t, c in Counter(terms).items()}
        q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
        if q_norm == 0:
            return [(pid, 0.0) for pid in docs]
        return [(pid, sum(w * self.counts[pid][t] * self.idf[t] for t, w in q_vec.items())
                 / (q_norm * self.doc_norm[pid])) for pid in docs]

    # ------------------------------------------------------------------ 1b. BM25
    def score_bm25(self, terms, docs):
        """Sum over the distinct query terms of idf x tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))."""
        k1, b = self.k1, self.b
        scores = []
        for pid in docs:
            norm = k1 * (1 - b + b * self.doc_len[pid] / self.avg_len)
            scores.append((pid, sum(self.idf[t] * self.counts[pid][t] * (k1 + 1) / (self.counts[pid][t] + norm)
                                    for t in set(terms))))
        return scores

    # ------------------------------------------------------------------ 1c. our score
    def score_bm25f(self, terms, docs):
        """BM25F: tf~ = sum over fields of w_f x tf_f / (1 - b + b |d_f| / avg|d_f|), then
        sum over the query terms of idf x tf~ (k1 + 1) / (tf~ + k1)."""
        k1, b = self.k1, self.b
        scores = []
        for pid in docs:
            s = 0.0
            for t in set(terms):
                tf_tilde = sum(w * self.field_counts[f][pid][t]
                               / (1 - b + b * self.field_len[f][pid] / self.field_avg_len[f])
                               for f, w in self.field_weights.items() if self.field_counts[f][pid][t])
                s += self.idf[t] * tf_tilde * (k1 + 1) / (tf_tilde + k1)
            scores.append((pid, s))
        return scores

    def title_factor(self, terms, pid):
        """1 + title_weight x (share of the distinct query terms that are in the title)."""
        unique = set(terms)
        return 1 + self.title_weight * len(unique & self.title_terms[pid]) / len(unique) if unique else 1.0

    def score_ours(self, terms, docs):
        """Our score = BM25F x title factor x quality factor."""
        return [(pid, s * self.title_factor(terms, pid) * self.quality[pid])
                for pid, s in self.score_bm25f(terms, docs)]

    # ------------------------------------------------------------------ 2. word2vec + cosine
    def train_word2vec(self, **params):
        """Train word2vec on our corpus (one sentence per title, details and description, Part 1 terms)
        and compute the vector of every product: the average of the vectors of its all_terms."""
        from gensim.models import Word2Vec
        sentences = [terms for f in W2V_FIELDS for terms in self.corpus[f] if len(terms) > 0]
        self.w2v = Word2Vec(sentences=sentences, **dict(W2V_PARAMS, **params))
        self.doc_vectors = {pid: self.text_vector(terms)
                            for pid, terms in zip(self.corpus["pid"], self.corpus["all_terms"])}
        return self.w2v

    def text_vector(self, terms):
        """Average of the word2vec vectors of the terms (terms outside the vocabulary are skipped)."""
        vectors = [self.w2v.wv[t] for t in terms if t in self.w2v.wv]
        if not vectors:
            return np.zeros(self.w2v.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def score_word2vec(self, terms, docs):
        """Cosine similarity between the average vector of the query and of each product."""
        if self.w2v is None:
            raise RuntimeError("call train_word2vec() before using method='word2vec'")
        q = self.text_vector(terms)
        q_norm = np.linalg.norm(q)
        scores = []
        for pid in docs:
            d = self.doc_vectors[pid]
            denom = q_norm * np.linalg.norm(d)
            scores.append((pid, float(np.dot(q, d) / denom) if denom else 0.0))
        return scores


def quality_factor(corpus):
    """
    One value per pid: (1 + 0.1 (rating - 1) / 4 + 0.05 discount / 100) x (0.5 if out of stock).
    A missing rating gets the mean rating (not rated is not a bad rating); a missing discount is 0.
    """
    rating = corpus["average_rating_num"].fillna(corpus["average_rating_num"].mean())
    discount = corpus["discount_pct"].fillna(0)
    factor = 1 + RATING_WEIGHT * (rating - 1) / 4 + DISCOUNT_WEIGHT * discount / 100
    factor = factor.where(corpus["in_stock"].astype(bool), factor * OUT_OF_STOCK_FACTOR)
    return dict(zip(corpus["pid"], factor.astype(float)))


def results_table(ranker, ranked, columns=("title", "brand_full", "average_rating_num",
                                           "discount_pct", "in_stock")):
    """DataFrame with rank, pid, score and some product fields, for displaying a ranking."""
    rows = []
    for rank, (pid, score) in enumerate(ranked, start=1):
        row = {"rank": rank, "pid": pid, "score": round(score, 4)}
        row.update({c: ranker.corpus.at[pid, c] for c in columns})
        rows.append(row)
    return pd.DataFrame(rows).set_index("rank")
