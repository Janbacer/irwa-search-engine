"""
Analytics dashboard: key indicators, charts (Altair / Vega-Lite) and tables computed from the star schema.

``build_report(analytics, engine)`` returns a dictionary for ``templates/dashboard.html``:
``kpis`` (groups of indicators), ``charts`` (Vega-Lite specifications with an explanation each) and
``tables`` (top queries, queries without results, most clicked products). Every indicator is defined in the
report and next to the chart that shows it.
"""
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
METHOD_COLORS = {"tfidf": "#2a78d6", "bm25": "#eb6834", "ours": "#1baf7a", "word2vec": "#eda100"}
METHOD_LABELS = {"tfidf": "TF-IDF", "bm25": "BM25", "ours": "Our score", "word2vec": "word2vec"}
DWELL_BINS = [0, 10, 30, 60, 180, np.inf]
DWELL_LABELS = ["< 10 s", "10-30 s", "30-60 s", "1-3 min", "> 3 min"]
SHORT_CLICK_S = 10   # a click with less than 10 s on the product page is a "short click" (not satisfied)


def _chart(chart, height=220):
    """Common look of the charts, as a Vega-Lite JSON specification."""
    return (chart.properties(width="container", height=height)
            .configure_view(stroke=None)
            .configure_axis(labelColor="#555", titleColor="#555", gridColor="#eee", domainColor="#ccc")
            .configure_legend(labelColor="#555", titleColor="#555", orient="bottom")
            .to_json().replace("</", "<\\/"))   # safe inside a <script> element


def _bar(df, x, y, x_title, y_title, sort=None, horizontal=False, color=BLUE, tooltip=None):
    """A simple bar chart with a tooltip on every bar."""
    base = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=3)
    enc = {"tooltip": tooltip or [x, y]}
    if horizontal:
        return base.encode(y=alt.Y(f"{x}:N", title=x_title, sort=sort or "-x"),
                           x=alt.X(f"{y}:Q", title=y_title), **enc)
    return base.encode(x=alt.X(f"{x}:N", title=x_title, sort=sort, axis=alt.Axis(labelAngle=0)),
                       y=alt.Y(f"{y}:Q", title=y_title), **enc)


def _pct(part, total):
    """'12.5 %', or '–' when there is nothing to divide."""
    return f"{100 * part / total:.1f} %" if total else "–"


def _fmt_seconds(seconds):
    """Seconds as '45 s' or '2.5 min' ('–' if unknown)."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "–"
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def build_report(analytics, engine):
    """Indicators, charts and tables of the dashboard."""
    visitors = analytics.frame("visitors")
    sessions = analytics.frame("sessions")
    requests = analytics.frame("requests")
    queries = analytics.frame("queries")
    clicks = analytics.frame("clicks")
    rag = analytics.frame("rag")
    report = {"empty": queries.empty and requests.empty, "simulated_sessions": int(sessions["simulated"].sum())
              if not sessions.empty else 0, "n_sessions": len(sessions), "kpis": [], "charts": [], "tables": {},
              "generated": datetime.now().strftime("%d/%m/%Y %H:%M")}
    if report["empty"]:
        return report

    for frame, column in ((sessions, "start"), (sessions, "end"), (requests, "timestamp"), (queries, "timestamp"),
                          (clicks, "timestamp")):
        frame[column] = pd.to_datetime(frame[column], format="ISO8601")
    details = clicks[clicks["target"] == "details"].copy()
    details["dwell_s"] = details["dwell_ms"].astype(float) / 1000
    first_pages = queries[queries["page"] == 1]

    # ------------------------------------------------------------------ key indicators
    duration = (sessions["end"] - sessions["start"]).dt.total_seconds()
    multi = sessions["n_requests"] > 1
    clicked_queries = set(details["query_id"])
    first_click_rank = details.sort_values("timestamp").groupby("query_id")["rank"].first()
    rr = [1 / first_click_rank[q] if q in first_click_rank.index and first_click_rank[q] > 0 else 0
          for q in queries["query_id"]]
    missions = queries.groupby("mission_id")
    research = queries.groupby("research_mission_id")["session_id"].nunique()
    dwell = details["dwell_s"].dropna()
    generated = rag[rag["outcome"] != "unavailable"]
    report["kpis"] = [
        ("Traffic", [
            ("Visitors", f"{len(visitors):,}", "Different browsers (visitor cookie)."),
            ("Sessions", f"{len(sessions):,}", "Physical sessions: a new one after 30 min of inactivity."),
            ("Returning visitors", _pct((sessions.groupby("visitor_id").size() > 1).sum(), len(visitors)),
             "Visitors with more than one session."),
            ("Requests", f"{len(requests):,}", "Pages and actions requested (static files excluded)."),
            ("Requests / session", f"{sessions['n_requests'].mean():.1f}", "Average number of requests per session."),
            ("Session length", _fmt_seconds(duration[multi].median() if multi.any() else None),
             "Median time between the first and the last request (sessions with 2+ requests)."),
            ("Bounce rate", _pct((~multi).sum(), len(sessions)), "Sessions with a single request."),
        ]),
        ("Search", [
            ("Searches", f"{len(queries):,}", "Result pages served (a new page or filter is a new search)."),
            ("Distinct queries", f"{queries['query'].str.lower().nunique():,}", "Different query texts."),
            ("Terms / query", f"{first_pages['n_terms'].mean():.2f}", "Average number of terms after pre-processing."),
            ("Searches / session", f"{queries.groupby('session_id').size().mean():.1f}",
             "Average number of searches in a session that searched."),
            ("Missions", f"{queries['mission_id'].nunique():,}",
             f"Logical sessions (queries with the same goal); {missions.size().mean():.1f} searches each."),
            ("Research missions", f"{(research > 1).sum():,}", "Missions continued in a later session."),
            ("Zero results", _pct((queries["match_type"] == "none").sum(), len(queries)),
             "Searches with no product at all."),
            ("Rescued searches", _pct(queries["match_type"].isin(["corrected", "partial"]).sum(), len(queries)),
             "Searches answered by the spelling suggestion or the partial match."),
            ("Search latency", f"{queries['latency_ms'].median():.1f} ms",
             f"Median server time of a search (95th percentile {queries['latency_ms'].quantile(.95):.0f} ms)."),
        ]),
        ("Results", [
            ("Click-through rate", _pct(queries["query_id"].isin(clicked_queries).sum(), len(queries)),
             "Searches with at least one click on a result."),
            ("Mean clicked rank", f"{details['rank'].mean():.1f}" if len(details) else "–",
             "Average position of the clicked results."),
            ("MRR of clicks", f"{np.mean(rr):.2f}" if rr else "–",
             "Mean of 1 / rank of the first click (0 for searches without a click)."),
            ("Dwell time", _fmt_seconds(dwell.median() if len(dwell) else None),
             "Median time on a product page after a click."),
            ("Short clicks", _pct((dwell < SHORT_CLICK_S).sum(), len(dwell)),
             "Clicks with less than 10 s on the product page."),
            ("Outbound clicks", f"{(clicks['target'] == 'original').sum():,}",
             "Clicks to the product on the original website."),
            ("RAG summaries", f"{len(generated):,}",
             (f"{_pct((generated['outcome'] == 'no_good_products').sum(), len(generated))} said that no product "
              f"fits; median {generated['latency_ms'].median() / 1000:.1f} s." if len(generated) else
              "Summaries generated by the LLM.") + (f" {len(rag) - len(generated)} not available (no API key or "
                                                   "limit reached)." if len(rag) > len(generated) else "")),
        ]),
    ]

    # ------------------------------------------------------------------ charts
    charts = []
    per_day = queries.assign(day=queries["timestamp"].dt.strftime("%Y-%m-%d")).groupby("day").agg(
        searches=("query_id", "size"), sessions=("session_id", "nunique")).reset_index()
    long = per_day.melt("day", var_name="measure", value_name="count")
    charts.append(("activity", "Searches and sessions per day", _chart(
        alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
            x=alt.X("day:T", title="day", axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=7)),
            y=alt.Y("count:Q", title="number"),
            color=alt.Color("measure:N", title=None, scale=alt.Scale(range=[BLUE, "#eb6834"])),
            tooltip=[alt.Tooltip("day:T", format="%d/%m/%Y"), "measure", "count"])),
        "How much the search engine is used each day. A rise of searches without a rise of sessions means that each "
        "visitor searches more."))
    by_hour = (queries["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
               .rename_axis("hour").reset_index(name="searches"))
    charts.append(("hours", "Searches by hour of the day", _chart(
        _bar(by_hour, "hour", "searches", "hour of the day", "searches", sort=list(range(24)))),
        "When people search (server time). Useful to plan maintenance and to see whether the shop is used at work "
        "or in the evening."))
    lengths = first_pages["n_terms"].clip(upper=8).value_counts().sort_index().rename_axis("terms").reset_index(
        name="searches")
    lengths["terms"] = lengths["terms"].astype(int).astype(str).replace({"8": "8+"})
    charts.append(("length", "Query length (terms after pre-processing)", _chart(
        _bar(lengths, "terms", "searches", "number of terms", "searches", sort=None)),
        "Short queries are ambiguous (\"shirt\"); long ones are specific and more likely to have no product with all "
        "the terms. The share of long queries tells how much the fallback is needed."))
    if len(details):
        ranks = details["rank"].clip(upper=21).value_counts().reindex(range(1, 22), fill_value=0)
        ranks = ranks.rename_axis("rank").reset_index(name="clicks")
        ranks["rank"] = ranks["rank"].astype(str).replace({"21": "21+"})
        charts.append(("ranks", "Clicks by rank of the result", _chart(
            _bar(ranks, "rank", "clicks", "rank of the clicked result", "clicks",
                 sort=[str(i) for i in range(1, 21)] + ["21+"])),
            "Position bias: users click the first results much more. If many clicks go to low ranks, the ranking "
            "puts the products people want too low."))
        buckets = pd.cut(dwell, DWELL_BINS, labels=DWELL_LABELS, right=False).value_counts().reindex(DWELL_LABELS)
        buckets = buckets.fillna(0).rename_axis("dwell").reset_index(name="clicks")
        charts.append(("dwell", "Dwell time on the product page", _chart(
            _bar(buckets, "dwell", "clicks", "time on the product page", "clicks", sort=DWELL_LABELS)),
            "Time between the click on a result and leaving the product page. Short clicks (< 10 s) suggest that the "
            "result did not match what the user wanted; long ones that it was worth reading."))
        by_method = queries.assign(clicked=queries["query_id"].isin(clicked_queries)).groupby("method").agg(
            searches=("query_id", "size"), ctr=("clicked", "mean")).reset_index()
        mean_rank = details.merge(queries[["query_id", "method"]], on="query_id").groupby("method")["rank"].mean()
        by_method["mean_clicked_rank"] = by_method["method"].map(mean_rank).round(2)
        by_method["ctr"] = (100 * by_method["ctr"]).round(1)
        by_method["algorithm"] = by_method["method"].map(METHOD_LABELS)
        charts.append(("methods", "Click-through rate by ranking algorithm", _chart(
            alt.Chart(by_method).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("algorithm:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("ctr:Q", title="click-through rate (%)"),
                color=alt.Color("method:N", legend=None, scale=alt.Scale(
                    domain=list(METHOD_COLORS), range=list(METHOD_COLORS.values()))),
                tooltip=["algorithm", "searches", alt.Tooltip("ctr:Q", title="CTR %"), "mean_clicked_rank"])),
            "Online comparison of the ranking algorithms (users choose one on the results page). A better ranking "
            "gets more clicks and clicks on higher positions (mean clicked rank in the tooltip)."))
    funnel = pd.DataFrame({"step": ["sessions", "with a search", "with a click", "to the original site"],
                           "sessions": [len(sessions), queries["session_id"].nunique(),
                                        details["session_id"].nunique(),
                                        clicks.loc[clicks["target"] == "original", "session_id"].nunique()]})
    charts.append(("funnel", "Session funnel", _chart(
        _bar(funnel, "step", "sessions", None, "sessions", sort=list(funnel["step"]))),
        "How many sessions reach each step: search, click on a result, and go to buy on the original website."))
    for column, title in (("browser", "Browser"), ("os", "Operating system"), ("device", "Device")):
        counts = sessions[column].replace("", "Unknown").value_counts().head(8).rename_axis(column).reset_index(
            name="sessions")
        charts.append((column, f"{title} (sessions)", _chart(
            _bar(counts, column, "sessions", None, "sessions", horizontal=True), height=160),
            "From the User-Agent of the first request of each session."))
    countries = sessions["country"].replace("", "Unknown").value_counts().head(10).rename_axis(
        "country").reset_index(name="sessions")
    charts.append(("country", "Country (sessions)", _chart(
        _bar(countries, "country", "sessions", None, "sessions", horizontal=True), height=200),
        "From the IP address (ip-api.com). Local addresses, like 127.0.0.1 on a developer's computer, are "
        "\"Local network\"."))
    report["charts"] = [{"id": i, "title": t, "spec": s, "explanation": e} for i, t, s, e in charts]

    # ------------------------------------------------------------------ tables
    top = (first_pages.assign(q=first_pages["query"].str.lower().str.strip())
           .assign(clicked=lambda d: d["query_id"].isin(clicked_queries))
           .groupby("q").agg(searches=("query_id", "size"), results=("n_results", "median"),
                             ctr=("clicked", "mean")).sort_values("searches", ascending=False).head(10))
    report["tables"]["top_queries"] = [
        {"query": q, "searches": int(r.searches), "results": int(r.results), "ctr": f"{100 * r.ctr:.0f} %"}
        for q, r in top.iterrows()]
    failed = queries[queries["match_type"] != "all"].groupby(["query", "match_type"]).size().reset_index(
        name="searches").sort_values("searches", ascending=False).head(10)
    report["tables"]["failed_queries"] = failed.to_dict("records")
    if len(details):
        products = details.groupby("pid").agg(clicks=("click_id", "size"), mean_rank=("rank", "mean"),
                                              dwell=("dwell_s", "median")).sort_values("clicks", ascending=False)
        report["tables"]["top_products"] = [
            {"pid": pid, "title": " ".join(engine.corpus[pid].title.split()) if pid in engine.corpus else pid,
             "clicks": int(r.clicks), "mean_rank": f"{r.mean_rank:.1f}", "dwell": _fmt_seconds(r.dwell)}
            for pid, r in products.head(10).iterrows()]
    return report
