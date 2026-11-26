"""
The search engine of the web application: one general ``search`` function on top of the index
(``index.py``) and the ranking algorithms (``algorithms.py``).

``SearchEngine.search(query, method="ours", filters=None, sort="relevance", page=1, page_size=20)``

1. **Query understanding**: a price in the query ("under 500", "below rs 1000", "between 500 and 800") becomes a
   price filter, a rating ("over 4 stars") a rating filter and a discount ("50% off") a discount filter; words that
   describe the price and not the product ("cheap", "budget") are dropped; then the query is cleaned like the
   products (``preprocessing.preprocess_text``).
2. **Conjunctive search** (AND, as in Parts 2 and 3): the products that contain every query term.
3. **No results**: in a web application an empty page is a dead end, so the engine tries
   (a) a spelling suggestion for the words that are not in the vocabulary ("jens" -> "jeans") and, if there are
   still no results, (b) the products that contain the largest number of the query terms. The results page says
   which of the three cases happened.
4. **Filters** chosen by the user (sub-category, price, rating, discount, only in stock), applied to the matches.
5. **Ranking** with the chosen algorithm (default: our score), or sorting by price, rating or discount.
6. **Pagination**: ``page_size`` results per page.
"""
import difflib
import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from myapp.search import algorithms
from myapp.search import preprocessing as pp
from myapp.search.index import SearchIndex
from myapp.search.objects import Document, ResultItem

SORTS = {
    "relevance": "Relevance",
    "price_asc": "Price: low to high",
    "price_desc": "Price: high to low",
    "rating": "Rating",
    "discount": "Discount",
}


# A price: digits not followed by "%", "off" or "stars" ("70% off" is a discount and "4 stars" a rating, not a price).
_NOT_PRICE = r"(?![\d,]|\.\d|\s*(?:%|percent\b|per\s*cent\b|off\b|stars?\b|\*|★))"
_NUMBER = rf"(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?){_NOT_PRICE}\s*(?:rs\.?|rupees|inr|/-)?"
PRICE_PATTERNS = [   # (regex, kind): the price expressions understood in a query
    (re.compile(rf"\bbetween\s+{_NUMBER}\s+(?:and|-|to)\s+{_NUMBER}", re.I), "between"),
    (re.compile(rf"\b(?:under|below|less than|cheaper than|up to|upto|max(?:imum)?|within)\s+{_NUMBER}", re.I), "max"),
    (re.compile(rf"\b(?:over|above|more than|min(?:imum)?|from)\s+{_NUMBER}", re.I), "min"),
]
PRICE_WORDS = re.compile(r"\b(?:cheap|cheapest|budget|affordable|inexpensive|price|priced)\b", re.I)
# "4 stars", "over 4 stars", "4+ stars", "4.5 star rating" -> minimum rating
RATING_PATTERN = re.compile(
    r"\b(?:(?:over|above|more than|at least|min(?:imum)?)\s+)?([1-5](?:\.\d)?)\s*\+?\s*(?:stars?|\*|★)"
    r"(?:\s+(?:rating|rated|and above|or more|\+))?", re.I)
# "50% off", "at least 40% discount", "up to 70% off": a discount, not a price. "up to" / "under" give no minimum.
DISCOUNT_PATTERN = re.compile(
    r"(?:\b(up to|upto|max(?:imum)?|under|below|less than)\s+|\b(?:over|above|more than|at least|min(?:imum)?|flat)\s+)?"
    r"(\d{1,2})\s*(?:%|percent\b|per\s*cent\b)\s*(?:off\b|discount\b)?", re.I)


def understand_price(query):
    """
    Price constraints written in the query: (query without them, min_price, max_price).
    "cheap cotton shirt under 500" -> ("cotton shirt", None, 500.0).
    Discounts ("70% off") and ratings ("4 stars") are not prices: see ``understand_query``.
    """
    low, high = None, None
    for pattern, kind in PRICE_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        values = [float(v.replace(",", "")) for v in match.groups()]
        if kind == "between":
            low, high = min(values), max(values)
        elif kind == "max":
            high = values[0]
        else:
            low = values[0]
        query = query[:match.start()] + " " + query[match.end():]
    query = PRICE_WORDS.sub(" ", query)
    return " ".join(query.split()), low, high


def understand_query(query):
    """
    Everything the query says about the price, the rating and the discount, and the words left for the search:
    (text, min_price, max_price, min_rating, min_discount).
    "shirts over 4 stars at least 40% off under 800" -> ("shirts", None, 800.0, 4.0, 40.0).
    "up to 70% off" is removed from the text but sets no filter (it does not ask for a minimum discount).
    """
    min_rating = min_discount = None
    match = RATING_PATTERN.search(query)
    if match:
        min_rating = float(match.group(1))
        query = query[:match.start()] + " " + query[match.end():]
    match = DISCOUNT_PATTERN.search(query)
    if match:
        if not match.group(1):   # "50% off", "at least 50% off" (not "up to 50% off")
            min_discount = float(match.group(2))
        query = query[:match.start()] + " " + query[match.end():]
    text, low, high = understand_price(query)
    return text, low, high, min_rating, min_discount


@dataclass
class SearchFilters:
    """Filters chosen on the results page. ``None`` (or False) means "no filter"."""
    sub_category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    min_discount: Optional[float] = None
    in_stock_only: bool = False

    def active(self):
        """Dictionary with only the filters that are set (stored by the analytics)."""
        return {k: v for k, v in self.__dict__.items() if v not in (None, False, "")}

    def accepts(self, index, pid):
        """True if the product passes every filter (a product without price or rating fails that filter)."""
        if self.sub_category and index.sub_category[pid] != self.sub_category:
            return False
        if self.in_stock_only and not index.in_stock[pid]:
            return False
        price = index.price[pid]
        if self.min_price is not None and (price is None or price < self.min_price):
            return False
        if self.max_price is not None and (price is None or price > self.max_price):
            return False
        if self.min_rating is not None and (index.rating[pid] is None or index.rating[pid] < self.min_rating):
            return False
        if self.min_discount is not None and (index.discount[pid] or 0) < self.min_discount:
            return False
        return True


@dataclass
class SearchResponse:
    """Everything the results page (and the analytics and the RAG) need about one search."""
    query: str
    terms: List[str]
    method: str
    sort: str
    filters: SearchFilters
    query_price: tuple = (None, None)    # (min, max) price read from the query text
    query_rating: Optional[float] = None    # minimum rating read from the query text ("over 4 stars")
    query_discount: Optional[float] = None  # minimum discount read from the query text ("50% off")
    match_type: str = "all"              # "all", "corrected", "partial" or "none"
    corrected_query: Optional[str] = None
    matched_terms: int = 0               # for "partial": number of query terms every result contains
    total_matches: int = 0               # matches before the filters
    total: int = 0                       # results after the filters
    page: int = 1
    pages: int = 0
    page_size: int = 20
    results: List[ResultItem] = field(default_factory=list)   # the results of this page
    top_results: List[ResultItem] = field(default_factory=list)   # the first results (for the RAG)
    facets: Dict[str, int] = field(default_factory=dict)     # sub-category -> number of matches
    latency_ms: float = 0.0


class SearchEngine:
    """Loads the dataset, builds the index and answers the searches of the web application."""

    def __init__(self, data_path, keep_numbers=True, word2vec=False):
        start = time.time()
        with open(data_path, encoding="utf-8") as f:
            raw_docs = json.load(f)
        self.corpus: Dict[str, Document] = {d["pid"]: Document(**d) for d in raw_docs}
        self.index = SearchIndex(raw_docs, keep_numbers=keep_numbers)
        self.word2vec = None
        self._word2vec_lock = threading.Lock()   # two first word2vec searches at once train it only once
        if word2vec:
            self.train_word2vec()
        self.load_seconds = time.time() - start

    def train_word2vec(self):
        """Train the word2vec model (about 30 s); needed only for method="word2vec"."""
        with self._word2vec_lock:
            if self.word2vec is None:
                self.word2vec = algorithms.Word2VecModel(self.index)
        return self.word2vec

    # ------------------------------------------------------------------ matching
    def match_all(self, terms):
        """pids of the products that contain every term (intersection, shortest posting list first)."""
        unique = list(dict.fromkeys(terms))
        if not unique or any(t not in self.index.postings for t in unique):
            return []
        unique.sort(key=lambda t: self.index.df[t])
        docs = set(self.index.postings[unique[0]])
        for t in unique[1:]:
            docs.intersection_update(self.index.postings[t])
            if not docs:
                break
        return list(docs)

    def match_most(self, terms):
        """Products that contain the largest number of distinct query terms: (pids, number of terms)."""
        coverage = Counter()
        for t in set(terms):
            coverage.update(self.index.postings.get(t, {}).keys())
        if not coverage:
            return [], 0
        best = max(coverage.values())
        return [pid for pid, c in coverage.items() if c == best], best

    def suggest(self, query):
        """
        Spelling suggestion: every word of the query whose stem is not in the index is replaced by the most likely
        word of the products: difflib similarity >= 0.75, with a bonus for frequent words (``_suggestion_score``).
        Returns the corrected query, or None if nothing changes.
        """
        replacements = {}
        for w in pp.tokenize(query, self.index.keep_numbers):
            if pp.stem(w) in self.index.postings or w.isdigit() or w in replacements:
                continue
            candidates = difflib.get_close_matches(w, self.index.word_freq.keys(), n=8, cutoff=0.75)
            if candidates:
                replacements[w] = max(candidates, key=lambda c: self._suggestion_score(w, c))
        if not replacements:
            return None
        # replace only the misspelt words and keep the rest of the query as the user typed it
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, replacements)) + r")\b", re.IGNORECASE)
        return pattern.sub(lambda m: replacements[m.group(0).lower()], query)

    def _suggestion_score(self, word, candidate):
        """
        Similarity of the spelling (difflib ratio) plus a small bonus for frequent words, so that among similar
        words the common one wins: "shrit" -> "shirt" (10,338 occurrences), not "shrt" (2).
        """
        ratio = difflib.SequenceMatcher(None, word, candidate).ratio()
        return ratio + 0.05 * math.log10(self.index.word_freq[candidate])

    # ------------------------------------------------------------------ search
    def search(self, query, method="ours", filters=None, sort="relevance", page=1, page_size=20,
               top_n=10, allow_fallback=True):
        """
        General search function of the web application.

        :param query: text typed by the user.
        :param method: ranking algorithm: "ours" (default), "bm25", "tfidf" or "word2vec".
        :param filters: ``SearchFilters`` (or None for no filter).
        :param sort: "relevance" (the ranking of ``method``), "price_asc", "price_desc", "rating" or "discount".
        :param page: page number, from 1.
        :param page_size: results per page.
        :param top_n: number of first results kept in ``top_results`` (the context of the RAG).
        :param allow_fallback: try a spelling suggestion and partial matches when no product has all the terms.
        :return: ``SearchResponse``.
        """
        start = time.perf_counter()
        method = method if method in algorithms.METHODS else "ours"
        sort = sort if sort in SORTS else "relevance"
        filters = filters or SearchFilters()
        text, low, high, min_rating, min_discount = understand_query(query)
        # a filter chosen on the page wins over the one read from the query
        def keep(chosen, read):
            return chosen if chosen is not None else read
        filters = replace(filters, min_price=keep(filters.min_price, low), max_price=keep(filters.max_price, high),
                          min_rating=keep(filters.min_rating, min_rating),
                          min_discount=keep(filters.min_discount, min_discount))
        terms = pp.preprocess_text(text, self.index.keep_numbers)
        response = SearchResponse(query=query, terms=terms, method=method, sort=sort, filters=filters,
                                  page_size=page_size, query_price=(low, high), query_rating=min_rating,
                                  query_discount=min_discount)

        pids = self.match_all(terms)
        if not pids and terms and allow_fallback:
            corrected = self.suggest(text)
            if corrected:
                corrected_terms = pp.preprocess_text(corrected, self.index.keep_numbers)
                pids = self.match_all(corrected_terms)
                if pids:
                    response.match_type, response.corrected_query = "corrected", corrected
                    terms = response.terms = corrected_terms
            if not pids:
                pids, response.matched_terms = self.match_most(terms)
                response.match_type = "partial" if pids else "none"
        elif not pids:
            response.match_type = "none"
        response.total_matches = len(pids)
        # category counts with the other filters applied (so they add up to what the user can get)
        other_filters = replace(filters, sub_category=None)
        response.facets = dict(Counter(self.index.sub_category[p] for p in pids
                                       if other_filters.accepts(self.index, p)).most_common())

        pids = [p for p in pids if filters.accepts(self.index, p)]
        response.total = len(pids)
        if pids:
            if method == "word2vec":
                self.train_word2vec()
            scores = algorithms.score(method, self.index, terms, pids, self.word2vec)
            ranked = self._sort(pids, scores, sort)
            response.pages = math.ceil(len(ranked) / page_size)
            response.page = min(max(1, int(page)), response.pages)
            first = (response.page - 1) * page_size
            response.results = [self._item(pid, first + i + 1, scores[pid])
                                for i, pid in enumerate(ranked[first:first + page_size])]
            response.top_results = [self._item(pid, i + 1, scores[pid]) for i, pid in enumerate(ranked[:top_n])]
        response.latency_ms = (time.perf_counter() - start) * 1000
        return response

    def _sort(self, pids, scores, sort):
        """Order by score (ties by pid), or by a product field with the score as the second key."""
        ix = self.index
        keys = {
            "relevance": lambda p: (-scores[p], p),
            "price_asc": lambda p: (ix.price[p] is None, ix.price[p] or 0, -scores[p], p),
            "price_desc": lambda p: (ix.price[p] is None, -(ix.price[p] or 0), -scores[p], p),
            "rating": lambda p: (ix.rating[p] is None, -(ix.rating[p] or 0), -scores[p], p),
            "discount": lambda p: (-(ix.discount[p] or 0), -scores[p], p),
        }
        return sorted(pids, key=keys[sort])

    def _item(self, pid, rank, score):
        """Result record of one product for the results page."""
        doc, ix = self.corpus[pid], self.index
        return ResultItem(
            pid=pid, rank=rank, score=round(float(score), 4), title=" ".join(doc.title.split()),
            description=doc.description, brand=ix.brand[pid] or doc.brand, sub_category=doc.sub_category,
            selling_price=ix.price[pid], actual_price=pp.parse_number(doc.actual_price),
            discount=ix.discount[pid], average_rating=ix.rating[pid], in_stock=ix.in_stock[pid],
            url=doc.url, image=(doc.images or [None])[0],
            details=pp.flatten_details(doc.product_details),
        )
