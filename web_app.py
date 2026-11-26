"""
IRWA Final Project - Part 4: Flask web application (search engine, RAG summary and web analytics).

Run from the repository root:  python web_app.py   ->  http://127.0.0.1:8088/

Pages
-----
/                       search page
/search?q=...           results page (ranking method, sorting, filters and page are URL parameters)
/doc/<pid>              details page of a product (a click on a result goes through it)
/go/<pid>               redirect to the product on the original website (tracked as an outbound click)
/dashboard              web analytics dashboard
/stats                  the raw analytics tables (star schema) and CSV/JSON export

Internal endpoints: /api/summary (RAG summary, loaded by the results page), /track/dwell (dwell-time beacon).
"""
import os
import secrets
import time
from datetime import datetime
from urllib.parse import urlencode

from dotenv import load_dotenv
from flask import (Flask, Response, abort, g, jsonify, redirect, render_template, request, session,
                   url_for)

from myapp.analytics import dashboard as dash
from myapp.analytics.analytics_data import AnalyticsData
from myapp.analytics.context import parse_user_agent, user_context
from myapp.generation.rag import RAGGenerator
from myapp.search import algorithms
from myapp.search import preprocessing as pp
from myapp.search.search_engine import SORTS, SearchEngine, SearchFilters

load_dotenv()  # take environment variables from .env

ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(ROOT, os.getenv("DATA_FILE_PATH", "data/fashion_products_dataset.json"))
PAGE_SIZE = 20
VISITOR_COOKIE = "irwa_visitor"
EXAMPLE_QUERIES = ["men white formal shirt slim fit", "women floral print shirt", "black running shoes for men",
                   "women pure cotton t-shirt pack of 3", "men slim jeans blue"]

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)   # signs the session cookie
app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME", "IRWA_SEARCH_ENGINE")

print("Loading the dataset and building the index...")
search_engine = SearchEngine(DATA_FILE, word2vec=os.getenv("TRAIN_WORD2VEC", "0") == "1")
print(f"{len(search_engine.corpus):,} products indexed in {search_engine.load_seconds:.1f} s")
analytics_data = AnalyticsData()
rag_generator = RAGGenerator()
# terms in more than 20 % of the products ("men", "women"...): they do not link two queries into one mission
COMMON_TERMS = {t for t, n in search_engine.index.df.items() if n / search_engine.index.N > 0.2}
# not stored at all: static files, the dwell-time beacon, and the pages of the shop owner (dashboard, raw tables,
# exports), which are not traffic of the search engine
UNTRACKED = ("static", "track_dwell", "dashboard", "stats", "export_csv", "export_json")
# need the session (to check that the search is the user's own) but are not page views: the RAG summary is
# loaded by the results page itself and is stored in its own fact table (rag)
NOT_PAGE_VIEWS = ("rag_summary",)


# ---------------------------------------------------------------------------------------------- tracking
def current_time():
    """Now; the traffic simulation (test client only) can set the time with the X-Sim-Time header."""
    if app.testing and request.headers.get("X-Sim-Time"):
        return datetime.fromisoformat(request.headers["X-Sim-Time"])
    return datetime.now()


def request_context(session_id):
    """User context of the request: stored with the session, so the place is looked up once per session."""
    if app.testing and request.headers.get("X-Sim-Time"):   # simulated visitor (see analytics/simulate.py)
        context = parse_user_agent(request.headers.get("User-Agent"))
        context.update({"ip": request.headers.get("X-Sim-IP", ""),
                        "country": request.headers.get("X-Sim-Country", "Unknown"),
                        "city": request.headers.get("X-Sim-City", "")})
        return context
    return analytics_data.context_of_session(session_id) or user_context(request)


@app.before_request
def start_tracking():
    """Identify the visitor (cookie) and the physical session of every request, and close the previous click."""
    g.started = time.perf_counter()
    if request.endpoint in UNTRACKED or request.endpoint is None:
        return
    g.now = current_time()
    simulated = bool(app.testing and request.headers.get("X-Sim-Time"))
    visitor_id, session_id = analytics_data.touch_session(
        request.cookies.get(VISITOR_COOKIE), session.get("sid"), g.now,
        request_context(session.get("sid")), request.path, simulated=simulated)
    session["sid"] = session_id
    g.visitor_id, g.session_id = visitor_id, session_id
    if request.endpoint not in NOT_PAGE_VIEWS:   # the summary is loaded by the results page itself
        analytics_data.close_open_click(session_id, g.now)


@app.after_request
def end_tracking(response):
    """Store the request (status and server time) and keep the visitor cookie for one year."""
    if getattr(g, "session_id", None):
        if request.endpoint not in NOT_PAGE_VIEWS:
            analytics_data.save_request(g.now, g.session_id, g.visitor_id, request.method,
                                        request.full_path.rstrip("?"), request.endpoint or "", response.status_code,
                                        (time.perf_counter() - g.started) * 1000, request.referrer or "")
        response.set_cookie(VISITOR_COOKIE, g.visitor_id, max_age=365 * 24 * 3600, httponly=True, samesite="Lax")
        analytics_data.save()
    return response


# ---------------------------------------------------------------------------------------------- helpers
def _float(name):
    """A numeric URL parameter, or None."""
    try:
        value = request.args.get(name, "").strip()
        return float(value) if value else None
    except ValueError:
        return None


def filters_from_request():
    """SearchFilters from the URL parameters of the results page."""
    return SearchFilters(sub_category=request.args.get("sub_category") or None,
                         min_price=_float("min_price"), max_price=_float("max_price"),
                         min_rating=_float("min_rating"), min_discount=_float("min_discount"),
                         in_stock_only=request.args.get("in_stock") == "1")


def search_url(**changes):
    """URL of the results page with the current parameters, some of them changed (None removes one)."""
    params = {k: v for k, v in request.args.items() if v not in ("", None)}
    params.update({k: v for k, v in changes.items()})
    params = {k: v for k, v in params.items() if v not in ("", None)}
    return url_for("search") + "?" + urlencode(params)


def click_source():
    """Where the clicked link was: "rag" (a product recommended by the AI summary) or "results" (the list)."""
    return "rag" if request.args.get("src") == "rag" else "results"


app.jinja_env.globals.update(search_url=search_url, method_names=algorithms.METHOD_NAMES, sorts=SORTS)


@app.template_filter("money")
def money(value):
    """2999.0 -> 'Rs 2,999'."""
    return "" if value is None else f"Rs {value:,.0f}"


@app.template_filter("clean")
def clean(text):
    """Collapse white space (titles contain non-breaking spaces)."""
    return " ".join(str(text or "").split())


# ---------------------------------------------------------------------------------------------- pages
@app.route("/")
def index():
    """Search page."""
    return render_template("index.html", page_title="Search", examples=EXAMPLE_QUERIES,
                           methods=algorithms.METHODS, n_products=len(search_engine.corpus))


@app.route("/search", methods=["GET", "POST"])
def search():
    """Results page. A POST (the form of the template) is redirected to the GET URL, which can be shared."""
    if request.method == "POST":
        return redirect(url_for("search", q=request.form.get("search-query", ""),
                                method=request.form.get("method", "ours")))
    query = request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("index"))
    method = request.args.get("method", "ours")
    sort = request.args.get("sort", "relevance")
    page = request.args.get("page", 1, type=int)
    filters = filters_from_request()
    response = search_engine.search(query, method=method, filters=filters, sort=sort, page=page,
                                    page_size=PAGE_SIZE, top_n=20)
    query_id = analytics_data.save_query(g.now, g.session_id, g.visitor_id, response, COMMON_TERMS)
    session["last_search"] = request.full_path
    return render_template("results.html", page_title=query, response=response, query_id=query_id,
                           filters=response.filters, methods=algorithms.METHODS,
                           rag_variant=request.args.get("rag", "improved"))


@app.route("/doc/<pid>")
def doc_details(pid):
    """Details page of a product. Coming from a results page (qid, rank) it is stored as a click."""
    doc = search_engine.corpus.get(pid) or abort(404)
    click_id = None
    if request.args.get("qid"):
        click_id = analytics_data.save_click(g.now, g.session_id, g.visitor_id, request.args.get("qid"), pid,
                                             request.args.get("rank", 0, type=int),
                                             request.args.get("page", 1, type=int), source=click_source())
    ix = search_engine.index
    return render_template("doc_details.html", page_title=clean(doc.title), doc=doc, click_id=click_id,
                           details=pp.flatten_details(doc.product_details), brand_full=ix.brand.get(pid),
                           price=ix.price[pid], actual_price=pp.parse_number(doc.actual_price),
                           discount=ix.discount[pid], rating=ix.rating[pid], in_stock=ix.in_stock[pid],
                           crawled_at=(datetime.fromtimestamp(doc.crawled_at / 1000).strftime("%d/%m/%Y")
                                       if doc.crawled_at else None),
                           qid=request.args.get("qid"), rank=request.args.get("rank"),
                           back_url=session.get("last_search"))


@app.route("/doc_details")
def doc_details_legacy():
    """URL of the course template (/doc_details?pid=...), kept as an alias."""
    return redirect(url_for("doc_details", pid=request.args.get("pid", ""), **{
        k: v for k, v in request.args.items() if k != "pid"}))


@app.route("/go/<pid>")
def outbound(pid):
    """Redirect to the product on the original website, storing the click (target "original")."""
    doc = search_engine.corpus.get(pid) or abort(404)
    analytics_data.save_click(g.now, g.session_id, g.visitor_id, request.args.get("qid", ""), pid,
                              request.args.get("rank", 0, type=int), request.args.get("page", 1, type=int),
                              target="original", source=click_source())
    return redirect(doc.url or url_for("doc_details", pid=pid))


@app.route("/api/summary")
def rag_summary():
    """RAG summary of a results page (JSON). The search is repeated from the stored query (a few ms)."""
    stored = analytics_data.get_query(request.args.get("qid", ""))
    if stored is None or stored.session_id != g.session_id:
        return jsonify({"status": "error", "summary": "Unknown search."}), 404
    variant = request.args.get("variant", "improved")
    filters = SearchFilters(**{k: v for k, v in stored.filters.items() if k in SearchFilters.__dataclass_fields__})
    response = search_engine.search(stored.query, method=stored.method, filters=filters, sort=stored.sort,
                                    top_n=20)
    result = rag_generator.generate(stored.query, response, variant=variant)
    analytics_data.save_rag(g.now, stored.query_id, g.session_id, variant, result["model"], result["status"],
                            result["latency_ms"], result["n_products"], result["cited_valid"], result["cited_invalid"])
    result.pop("raw", None)
    return jsonify(result)


@app.route("/track/dwell", methods=["POST"])
def track_dwell():
    """Dwell-time beacon of the details page: {click_id, ms} (time the page was visible)."""
    data = request.get_json(silent=True, force=True) or {}
    try:
        ok = analytics_data.save_dwell(str(data.get("click_id", "")), float(data.get("ms", 0)), session.get("sid"))
    except (TypeError, ValueError):
        ok = False
    if ok:
        analytics_data.save()
    return ("", 204) if ok else ("", 400)


@app.route("/dashboard")
def dashboard():
    """Web analytics dashboard: key indicators and charts."""
    report = dash.build_report(analytics_data, search_engine)
    return render_template("dashboard.html", page_title="Dashboard", report=report)


@app.route("/stats")
def stats():
    """The analytics tables (star schema): number of rows and the last rows of each table."""
    tables = {name: analytics_data.frame(name) for name in ("visitors", "sessions", "requests", "queries",
                                                            "clicks", "rag")}
    return render_template("stats.html", page_title="Analytics data", tables=tables)


@app.route("/analytics/export/<name>.csv")
def export_csv(name):
    """Download one analytics table as CSV."""
    if name not in ("visitors", "sessions", "requests", "queries", "clicks", "rag"):
        abort(404)
    csv = analytics_data.frame(name).to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=irwa_{name}.csv"})


@app.route("/analytics/export.json")
def export_json():
    """Download every analytics table as one JSON file."""
    return jsonify(analytics_data.to_dict())


if __name__ == "__main__":
    # use_reloader=False: the reloader would load the dataset and build the index twice
    app.run(port=int(os.getenv("PORT", 8088)), host=os.getenv("HOST", "127.0.0.1"), threaded=True,
            debug=os.getenv("DEBUG", "False").lower() in ("1", "true"), use_reloader=False)
