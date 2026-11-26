"""
Text pre-processing of the products and of the queries.

It is the Part 1 pipeline (``project_progress/part_1/irwa_preprocessing.py``) with one change for the web
application: **numbers are kept**. In Part 3 the query "women pure cotton t-shirt pack of 3" lost its "3",
so no ranking could tell a pack of 3 from a pack of 5. Users type sizes and pack sizes ("pack of 3",
"size 32"), so the search engine keeps the number tokens (even one digit). Other one-letter tokens are still removed.

A query and a product are cleaned with the same function (``preprocess_text``), so their terms match.
"""
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:   # first run on a new computer: download the NLTK stop-word list once
    import nltk
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

# "T-shirt" is written in several ways in the data (t-shirt, t shirt, tshirt, tee...). Writing these fashion
# words in a single way before removing the punctuation avoids "t-shirt" -> "t" + "shirt".
FASHION_WORDS = [
    (re.compile(r"\bt[\s\-]?shirts?\b|\btees?\b"), " tshirt "),
    (re.compile(r"\bsweat[\s\-]?shirts?\b"), " sweatshirt "),
    (re.compile(r"\bpoly[\s\-]?cotton\b"), " poly cotton "),
    (re.compile(r"\bslim[\s\-]?fit\b"), " slim fit "),
]
_PUNCTUATION = re.compile(r"[^\w\s]|_")

# product_details keys whose values are codes or boilerplate, not searchable text (as in Part 1).
DETAIL_KEYS_EXCLUDED = {"Style Code", "Fabric Care", "Sales Package", "Country of Origin"}


@lru_cache(maxsize=None)
def stem(token):
    """Porter stem of one token, cached: the corpus has few distinct words, so each is stemmed once."""
    return _STEMMER.stem(token)


def tokenize(text, keep_numbers=True):
    """
    lowercase -> fashion words -> punctuation to spaces -> split -> drop stop words and one-letter tokens
    (numbers are kept, unless ``keep_numbers`` is False as in Part 1). Returns the words before stemming.
    """
    if not text:
        return []
    text = str(text).lower()
    for pattern, replacement in FASHION_WORDS:
        text = pattern.sub(replacement, text)
    tokens = _PUNCTUATION.sub(" ", text).split()
    if keep_numbers:   # numbers are kept, even one digit ("pack of 3"); other one-letter tokens are removed
        return [t for t in tokens if t not in STOP_WORDS and (len(t) > 1 or t.isdigit())]
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1 and not t.isdigit()]   # Part 1


def preprocess_text(text, keep_numbers=True):
    """Full pipeline for a product field or a query: list of stemmed terms."""
    return [stem(t) for t in tokenize(text, keep_numbers)]


def flatten_details(details):
    """[{'Fabric': 'Cotton'}, {'Color': 'Blue'}] -> {'Fabric': 'Cotton', 'Color': 'Blue'}."""
    if isinstance(details, dict):
        return {str(k).strip(): str(v).strip() for k, v in details.items()}
    flat = {}
    for entry in details or []:
        for key, value in entry.items():
            flat[str(key).strip()] = str(value).strip()
    return flat


def details_to_text(flat):
    """Searchable text of product_details: the values (not the keys), without codes and Yes/No values."""
    return " ".join(value for key, value in flat.items()
                    if key not in DETAIL_KEYS_EXCLUDED and value.lower() not in ("yes", "no"))


def recover_brand(brand, url):
    """
    Full brand name from the product URL (Part 1). The brand field is cut ("REEB" for Reebok), but the URL
    starts with the full name: flipkart.com/reebok-.../p/... We take the shortest group of URL words that
    starts with the cut brand; if none matches, the original brand (lower case) is returned.
    """
    brand = (brand or "").strip().lower()
    target = re.sub(r"[^a-z0-9]", "", brand)
    match = re.search(r"flipkart\.com/([^/]+)/p/", url or "")
    if not target or not match:
        return brand
    words = match.group(1).split("-")
    joined = ""
    for i, word in enumerate(words):
        joined += word
        if len(joined) >= len(target):
            return " ".join(words[: i + 1]) if joined.startswith(target) else brand
    return brand


def parse_number(value):
    """'2,999' -> 2999.0, '69% off' -> 69.0, '3.9' -> 3.9, '' or None -> None."""
    match = re.search(r"\d+(?:\.\d+)?", str(value if value is not None else "").replace(",", ""))
    return float(match.group()) if match else None
