"""
Web analytics: collection and storage of how people use the website and the search engine.

Data model (star schema, see the report):

* dimension tables  ``visitors`` (a browser, identified by a long-lived cookie) and ``sessions``
  (a physical session or "sit-down": a new one starts after 30 minutes without requests);
  the products dimension is the corpus itself (``pid``), and the time dimension is derived from the timestamps;
* fact tables       ``requests`` (every HTTP request), ``queries`` (every search), ``clicks`` (every click on a
  result, with its query, rank and dwell time) and ``rag`` (every generated summary).

Everything is kept in memory (lists of dataclass rows). Optionally, if the environment variable
``ANALYTICS_FILE`` names a JSON file, the tables are loaded from it when the application starts and saved to it
after each change, so they survive a restart (see the README).
"""
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

SESSION_TIMEOUT = timedelta(minutes=30)   # physical session: a new one after 30 min without requests
MAX_DWELL = timedelta(minutes=30)         # a longer "dwell time" means the user left, not that they read
COMMON_TERM_SHARE = 0.2                   # terms in more than 20 % of the products do not link two queries


def new_id():
    """Short random identifier."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------------------------- tables
@dataclass
class Visitor:
    """Dimension: one browser (persistent cookie). Its first user agent and place."""
    visitor_id: str
    first_seen: str
    browser: str = ""
    os: str = ""
    device: str = ""
    ip: str = ""
    country: str = ""
    city: str = ""
    simulated: bool = False


@dataclass
class Session:
    """Dimension: one physical session (sit-down) of a visitor."""
    session_id: str
    visitor_id: str
    start: str
    end: str
    browser: str = ""
    os: str = ""
    device: str = ""
    ip: str = ""
    country: str = ""
    city: str = ""
    entry_path: str = ""
    n_requests: int = 0
    n_queries: int = 0
    n_clicks: int = 0
    simulated: bool = False


@dataclass
class Request:
    """Fact: one HTTP request."""
    request_id: str
    timestamp: str
    session_id: str
    visitor_id: str
    method: str
    path: str
    endpoint: str
    status: int = 0
    response_ms: float = 0.0
    referrer: str = ""


@dataclass
class Query:
    """Fact: one search (a new page or new filters of the same query are new searches)."""
    query_id: str
    timestamp: str
    session_id: str
    visitor_id: str
    query: str
    terms: List[str]
    n_terms: int
    order_in_session: int          # 1 = first search of the session
    mission_id: str                # logical session: queries with the same goal
    research_mission_id: str       # the same goal across several sessions of the visitor
    method: str
    sort: str
    filters: Dict[str, object]
    page: int
    match_type: str                # all, corrected, partial, none
    n_results: int
    latency_ms: float


@dataclass
class Click:
    """Fact: one click on a result (to its details page, or to the original website)."""
    click_id: str
    timestamp: str
    query_id: str
    session_id: str
    visitor_id: str
    pid: str
    rank: int
    page: int
    target: str = "details"        # "details" (our page) or "original" (the e-commerce website)
    dwell_ms: Optional[float] = None
    dwell_source: str = ""          # "beacon" (sent by the page) or "next_request" (estimated)


@dataclass
class RagEvent:
    """Fact: one generated summary of a results page."""
    rag_id: str
    timestamp: str
    query_id: str
    session_id: str
    variant: str                   # "improved" or "baseline"
    model: str
    outcome: str                   # ok, no_good_products, no_results, unavailable
    latency_ms: float
    n_products: int
    cited_valid: int = 0
    cited_invalid: int = 0


TABLES = {"visitors": Visitor, "sessions": Session, "requests": Request, "queries": Query,
          "clicks": Click, "rag": RagEvent}


# ---------------------------------------------------------------------------------------------- store
class AnalyticsData:
    """In-memory analytics store: the six tables plus the operations of the web application."""

    def __init__(self, path=None):
        self.path = path or os.getenv("ANALYTICS_FILE") or None
        self.lock = threading.RLock()
        self.visitors: Dict[str, Visitor] = {}
        self.sessions: Dict[str, Session] = {}
        self.requests: List[Request] = []
        self.queries: List[Query] = []
        self.clicks: List[Click] = []
        self.rag: List[RagEvent] = []
        if self.path and os.path.exists(self.path):
            self.load()

    # ------------------------------------------------------------------ persistence (optional)
    def to_dict(self):
        """All the tables as lists of dictionaries."""
        with self.lock:
            return {"visitors": [asdict(v) for v in self.visitors.values()],
                    "sessions": [asdict(s) for s in self.sessions.values()],
                    "requests": [asdict(r) for r in self.requests],
                    "queries": [asdict(q) for q in self.queries],
                    "clicks": [asdict(c) for c in self.clicks],
                    "rag": [asdict(r) for r in self.rag]}

    def save(self):
        """Write the tables to ``ANALYTICS_FILE`` (only if it is set). The file is replaced atomically."""
        if not self.path:
            return
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, self.path)

    def load(self):
        """Read the tables from ``ANALYTICS_FILE``."""
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        def rows(name):
            """The rows of one table as dataclass objects (unknown columns are ignored)."""
            names = {f.name for f in fields(TABLES[name])}
            return [TABLES[name](**{k: v for k, v in row.items() if k in names}) for row in data.get(name, [])]

        self.visitors = {v.visitor_id: v for v in rows("visitors")}
        self.sessions = {s.session_id: s for s in rows("sessions")}
        self.requests, self.queries = rows("requests"), rows("queries")
        self.clicks, self.rag = rows("clicks"), rows("rag")

    def clear(self):
        """Delete every row (used by the tests and the simulation)."""
        with self.lock:
            self.visitors, self.sessions = {}, {}
            self.requests, self.queries, self.clicks, self.rag = [], [], [], []

    def frame(self, name):
        """One table as a pandas DataFrame (with the right columns even when it is empty)."""
        with self.lock:
            data = self.to_dict()[name]
        columns = [f.name for f in fields(TABLES[name])]
        return pd.DataFrame(data, columns=columns)

    # ------------------------------------------------------------------ visitors and sessions
    def touch_session(self, visitor_id, session_id, now, context, path, simulated=False):
        """
        Return the (visitor_id, session_id) of this request, creating them if needed. A session expires after
        ``SESSION_TIMEOUT`` without requests; the visitor (cookie) is kept. ``context`` is the user context
        (browser, os, device, ip, country, city) of the request.
        """
        with self.lock:
            if visitor_id not in self.visitors:
                visitor_id = visitor_id or new_id()
                self.visitors[visitor_id] = Visitor(visitor_id=visitor_id, first_seen=now.isoformat(),
                                                    simulated=simulated, **context)
            session = self.sessions.get(session_id)
            if (session is None or session.visitor_id != visitor_id
                    or now - datetime.fromisoformat(session.end) > SESSION_TIMEOUT):
                session = Session(session_id=new_id(), visitor_id=visitor_id, start=now.isoformat(),
                                  end=now.isoformat(), entry_path=path, simulated=simulated, **context)
                self.sessions[session.session_id] = session
            session.end = now.isoformat()
            return visitor_id, session.session_id

    def context_of_session(self, session_id):
        """The stored user context of a session (to avoid a new place look-up at every request)."""
        s = self.sessions.get(session_id)
        return None if s is None else {k: getattr(s, k) for k in ("browser", "os", "device", "ip", "country", "city")}

    # ------------------------------------------------------------------ requests
    def save_request(self, now, session_id, visitor_id, method, path, endpoint, status, response_ms, referrer):
        """Store one HTTP request, and use it to estimate the dwell time of the previous click (if needed)."""
        with self.lock:
            self.requests.append(Request(request_id=new_id(), timestamp=now.isoformat(), session_id=session_id,
                                         visitor_id=visitor_id, method=method, path=path, endpoint=endpoint,
                                         status=status, response_ms=round(response_ms, 2), referrer=referrer))
            self.sessions[session_id].n_requests += 1

    def close_open_click(self, session_id, now):
        """
        Dwell time fallback: when the session makes a new request and its last click on a details page has no
        dwell time yet (no beacon arrived, e.g. JavaScript disabled), the time since the click is used.
        """
        with self.lock:
            for click in reversed(self.clicks):
                if click.session_id != session_id:
                    continue
                if click.target == "details" and click.dwell_ms is None:
                    elapsed = now - datetime.fromisoformat(click.timestamp)
                    if timedelta(0) <= elapsed <= MAX_DWELL:
                        click.dwell_ms = elapsed.total_seconds() * 1000
                        click.dwell_source = "next_request"
                break

    # ------------------------------------------------------------------ queries and missions
    def save_query(self, now, session_id, visitor_id, response, common_terms):
        """
        Store a search (``response`` is the ``SearchResponse``) and assign its mission.

        Mission (logical session): a query continues the current mission of the session if it shares at least one
        non-common term with it (``common_terms``: terms in more than 20 % of the products, like "men", "women");
        otherwise it starts a new mission. Research mission: a new mission that shares a non-common term with a
        mission of an earlier session of the same visitor continues that research mission.
        """
        with self.lock:
            session = self.sessions[session_id]
            session.n_queries += 1
            terms = list(response.terms)
            content = set(terms) - common_terms
            previous = [q for q in self.queries if q.session_id == session_id]
            mission_id, research_id = None, None
            if previous:
                last = previous[-1]
                mission_terms = set().union(*(set(q.terms) for q in previous if q.mission_id == last.mission_id))
                if (content & mission_terms) or (not content and set(terms) == set(last.terms)):
                    mission_id, research_id = last.mission_id, last.research_mission_id
            if mission_id is None:
                mission_id = new_id()
                earlier = [q for q in self.queries if q.visitor_id == visitor_id and q.session_id != session_id]
                for q in reversed(earlier):
                    if content & set(q.terms):
                        research_id = q.research_mission_id
                        break
                research_id = research_id or mission_id
            query = Query(query_id=new_id(), timestamp=now.isoformat(), session_id=session_id,
                          visitor_id=visitor_id, query=response.query, terms=terms, n_terms=len(terms),
                          order_in_session=len(previous) + 1, mission_id=mission_id,
                          research_mission_id=research_id, method=response.method, sort=response.sort,
                          filters=response.filters.active(), page=response.page, match_type=response.match_type,
                          n_results=response.total, latency_ms=round(response.latency_ms, 2))
            self.queries.append(query)
            return query.query_id

    def get_query(self, query_id):
        """The stored query with this id, or None."""
        with self.lock:
            return next((q for q in reversed(self.queries) if q.query_id == query_id), None)

    # ------------------------------------------------------------------ clicks and dwell time
    def save_click(self, now, session_id, visitor_id, query_id, pid, rank, page, target="details"):
        """Store a click on a result. Returns its id (sent back by the details page with the dwell time)."""
        with self.lock:
            click = Click(click_id=new_id(), timestamp=now.isoformat(), query_id=query_id or "",
                          session_id=session_id, visitor_id=visitor_id, pid=pid, rank=rank or 0, page=page or 1,
                          target=target)
            self.clicks.append(click)
            self.sessions[session_id].n_clicks += 1
            return click.click_id

    def save_dwell(self, click_id, dwell_ms, session_id):
        """Dwell time measured by the details page (time it was visible), sent when the user leaves it."""
        with self.lock:
            for click in reversed(self.clicks):
                if click.click_id == click_id and click.session_id == session_id:
                    click.dwell_ms = min(float(dwell_ms), MAX_DWELL.total_seconds() * 1000)
                    click.dwell_source = "beacon"
                    return True
            return False

    # ------------------------------------------------------------------ RAG
    def save_rag(self, now, query_id, session_id, variant, model, outcome, latency_ms, n_products,
                 cited_valid=0, cited_invalid=0):
        """Store one generated summary."""
        with self.lock:
            self.rag.append(RagEvent(rag_id=new_id(), timestamp=now.isoformat(), query_id=query_id or "",
                                     session_id=session_id, variant=variant, model=model, outcome=outcome,
                                     latency_ms=round(latency_ms, 1), n_products=n_products,
                                     cited_valid=cited_valid, cited_invalid=cited_invalid))
