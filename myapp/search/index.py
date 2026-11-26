"""
Search index of the web application, built once from the dataset when the application starts.

Everything that does not depend on the query is computed here, so that a search only has to read
postings and add numbers (see ``algorithms.py``):

* the inverted index of ``all_terms`` (term -> {pid: term frequency}) and the statistics of TF-IDF and BM25
  (df, idf, product length, TF-IDF norm of each product);
* for our score, the **BM25F pseudo-frequency** of every (term, product) pair. In Part 3 it was recomputed
  for every product at every query, but it does not depend on the query:
  ``tf~(t, d) = sum over fields f of w_f * tf_f(t, d) / (1 - b + b * |d_f| / avg|d_f|)``;
* the title terms and the quality factor of every product;
* the structured fields used by the filters (price, rating, discount, stock, sub-category, brand).

The fields and weights are the same as in Part 3 (``project_progress/part_3/irwa_ranking.py``).
"""
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np

from myapp.search import preprocessing as pp

# Fields of a product that are searched, and their weights in BM25F (Part 3).
FIELD_WEIGHTS = {"title": 3.0, "details": 2.0, "brand": 1.0, "sub_category": 0.5, "description": 0.5}
BM25_K1 = 1.2
BM25_B = 0.75

# Quality factor of our score (Part 3).
RATING_WEIGHT = 0.1        # at most +10 % (rating 5)
DISCOUNT_WEIGHT = 0.05     # at most +5 % (100 % discount)
OUT_OF_STOCK_FACTOR = 0.5  # an out-of-stock product keeps half of its score


def product_fields(doc):
    """The text of each searchable field of one raw product (a dict of the JSON dataset)."""
    details = pp.flatten_details(doc.get("product_details"))
    return {
        "title": doc.get("title") or "",
        "details": pp.details_to_text(details),
        "brand": pp.recover_brand(doc.get("brand"), doc.get("url")),
        "sub_category": doc.get("sub_category") or "",
        "description": doc.get("description") or "",
    }


class SearchIndex:
    """All the query-independent data of the search engine. Build it with ``SearchIndex.from_file(path)``."""

    def __init__(self, raw_docs, keep_numbers=True, k1=BM25_K1, b=BM25_B, field_weights=None):
        start = time.time()
        self.keep_numbers = keep_numbers
        self.k1, self.b = k1, b
        self.field_weights = dict(FIELD_WEIGHTS if field_weights is None else field_weights)

        self.pids = []
        self.field_terms = {}             # pid -> {field: [terms]}
        self.brand = {}                   # pid -> full brand name
        word_counts = Counter()           # surface word -> frequency (for the spelling suggestions)
        for doc in raw_docs:
            pid = doc["pid"]
            self.pids.append(pid)
            texts = product_fields(doc)
            self.brand[pid] = texts["brand"]
            fields = {}
            for field, text in texts.items():
                words = pp.tokenize(text, keep_numbers)
                word_counts.update(words)
                fields[field] = [pp.stem(w) for w in words]
            self.field_terms[pid] = fields
        self.N = len(self.pids)

        # ---- all_terms: inverted index with term frequencies, as in Parts 2 and 3
        self.postings = defaultdict(dict)                 # term -> {pid: tf}
        self.doc_len = {}
        for pid, fields in self.field_terms.items():
            counts = Counter(t for f in FIELD_WEIGHTS for t in fields[f])
            self.doc_len[pid] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term][pid] = tf
        self.postings = dict(self.postings)
        self.avg_len = float(np.mean(list(self.doc_len.values())))
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.idf = {t: math.log(self.N / n) for t, n in self.df.items()}          # log(N / df), as in Part 3
        norms = defaultdict(float)
        for term, plist in self.postings.items():
            idf = self.idf[term]
            for pid, tf in plist.items():
                norms[pid] += (tf * idf) ** 2
        self.doc_norm = {pid: math.sqrt(v) for pid, v in norms.items()}           # TF-IDF norm (cosine)

        # ---- BM25F pseudo-frequencies, one per (term, product)
        field_len = {f: {pid: len(self.field_terms[pid][f]) for pid in self.pids} for f in self.field_weights}
        field_avg = {f: max(float(np.mean(list(lens.values()))), 1e-9) for f, lens in field_len.items()}
        self.bm25f_tf = defaultdict(dict)                 # term -> {pid: tf~}
        for pid in self.pids:
            tilde = defaultdict(float)
            for f, w in self.field_weights.items():
                norm = 1 - b + b * field_len[f][pid] / field_avg[f]
                for term, tf in Counter(self.field_terms[pid][f]).items():
                    tilde[term] += w * tf / norm
            for term, value in tilde.items():
                self.bm25f_tf[term][pid] = value
        self.bm25f_tf = dict(self.bm25f_tf)
        self.title_terms = {pid: frozenset(self.field_terms[pid]["title"]) for pid in self.pids}

        # ---- structured fields (filters, sorting and the quality factor)
        self.price, self.rating, self.discount, self.in_stock = {}, {}, {}, {}
        self.sub_category = {}
        for doc in raw_docs:
            pid = doc["pid"]
            self.price[pid] = pp.parse_number(doc.get("selling_price"))
            self.rating[pid] = pp.parse_number(doc.get("average_rating"))
            self.discount[pid] = pp.parse_number(doc.get("discount"))
            self.in_stock[pid] = not doc.get("out_of_stock")
            self.sub_category[pid] = doc.get("sub_category") or "Other"
        self.quality = self._quality_factor()

        # ---- vocabulary of surface words for "did you mean" (only words seen at least twice)
        self.word_to_stem = {w: pp.stem(w) for w, c in word_counts.items() if c >= 2}
        self.word_freq = {w: c for w, c in word_counts.items() if c >= 2}
        self.build_seconds = time.time() - start

    def _quality_factor(self):
        """(1 + 0.1 (rating - 1) / 4 + 0.05 discount / 100) x (0.5 if out of stock); missing rating = mean."""
        known = [r for r in self.rating.values() if r is not None]
        mean_rating = float(np.mean(known)) if known else 3.0
        quality = {}
        for pid in self.pids:
            rating = self.rating[pid] if self.rating[pid] is not None else mean_rating
            discount = self.discount[pid] or 0.0
            factor = 1 + RATING_WEIGHT * (rating - 1) / 4 + DISCOUNT_WEIGHT * discount / 100
            quality[pid] = factor if self.in_stock[pid] else factor * OUT_OF_STOCK_FACTOR
        return quality

    @classmethod
    def from_file(cls, data_path, keep_numbers=True):
        """Build the index of the JSON dataset at ``data_path`` (a few seconds for the 28,080 products)."""
        with open(data_path, encoding="utf-8") as f:
            return cls(json.load(f), keep_numbers=keep_numbers)
