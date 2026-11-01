"""
IRWA Final Project - Part 2: inverted index, conjunctive queries and TF-IDF ranking.

Adapted from Practice Session 1. Main changes:
- documents are products, identified by their ``pid``, and are already pre-processed
  (Part 1 ``all_terms``); queries are cleaned with ``irwa_preprocessing.preprocess_text``;
- ``search`` is conjunctive (AND): a product must contain every query term;
- ``rank_documents`` returns ``(pid, score)`` pairs, which the evaluation needs.

Main functions
--------------
- ``create_index(docs_terms)``                -> inverted index with positions
- ``compute_tfidf(index, N)``                 -> tf, df, idf
- ``search_tf_idf(terms, index, tf, idf)``    -> [(pid, score)] sorted by score
"""
import collections
import math
from array import array
from collections import defaultdict

import numpy as np
from numpy import linalg as la


def build_docs_terms(corpus, field="all_terms"):
    """{pid: list of terms} from one column of the Part 1 processed corpus."""
    return dict(zip(corpus["pid"], corpus[field]))


def create_index(docs_terms):
    """Inverted index {term: [[pid, array of positions], ...]}, as in the lab (Figure 2)."""
    index = defaultdict(list)
    for pid, terms in docs_terms.items():
        current_page_index = {}
        for position, term in enumerate(terms):
            if term in current_page_index:
                current_page_index[term][1].append(position)
            else:
                current_page_index[term] = [pid, array('I', [position])]
        for term, posting in current_page_index.items():
            index[term].append(posting)
    return index


def search(query_terms, index):
    """pids of the products that contain ALL the query terms (intersection of the posting lists)."""
    terms = list(dict.fromkeys(query_terms))  # unique terms
    if not terms or any(term not in index for term in terms):
        return []
    terms.sort(key=lambda term: len(index[term]))  # start with the shortest posting list
    docs = {posting[0] for posting in index[terms[0]]}
    for term in terms[1:]:
        docs &= {posting[0] for posting in index[term]}
    return list(docs)


def compute_tfidf(index, num_documents):
    """
    TF-IDF weights of the lab: tf = count / ||D|| (Euclidean norm of the product's term counts),
    idf = log(N / df). The count of a term in a product is the length of its list of positions.
    """
    squared_counts = defaultdict(int)
    for postings in index.values():
        for pid, positions in postings:
            squared_counts[pid] += len(positions) ** 2
    norms = {pid: math.sqrt(total) for pid, total in squared_counts.items()}

    tf = defaultdict(list)  # same order as index[term]
    df = defaultdict(int)
    idf = defaultdict(float)
    for term, postings in index.items():
        for pid, positions in postings:
            tf[term].append(np.round(len(positions) / norms[pid], 4))
        df[term] = len(postings)
        idf[term] = np.round(np.log(float(num_documents / df[term])), 4)
    return tf, df, idf


def create_index_tfidf(docs_terms):
    """create_index + compute_tfidf: returns index, tf, df, idf."""
    index = create_index(docs_terms)
    tf, df, idf = compute_tfidf(index, len(docs_terms))
    return index, tf, df, idf


def rank_documents(terms, docs, index, idf, tf):
    """
    Score the products in docs as in the lab: dot product between the query and product
    TF-IDF vectors (restricted to the query terms). Returns [(pid, score)] by decreasing
    score, ties broken by pid. A repeated query term gets one position in the vectors.
    """
    query_terms_count = collections.Counter(terms)
    unique_terms = list(query_terms_count)
    if not unique_terms:
        return []
    query_norm = la.norm(list(query_terms_count.values()))

    docs = set(docs)
    doc_vectors = {doc: [0.0] * len(unique_terms) for doc in docs}
    query_vector = [0.0] * len(unique_terms)

    for term_index, term in enumerate(unique_terms):
        if term not in index:
            continue
        query_vector[term_index] = (query_terms_count[term] / query_norm) * idf[term]
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][term_index] = tf[term][doc_index] * idf[term]

    doc_scores = [(doc, float(np.dot(vector, query_vector))) for doc, vector in doc_vectors.items()]
    doc_scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return doc_scores


def search_tf_idf(query_terms, index, tf, idf):
    """AND search + TF-IDF ranking: [(pid, score)] sorted by decreasing score."""
    docs = search(query_terms, index)
    return rank_documents(query_terms, docs, index, idf, tf)
