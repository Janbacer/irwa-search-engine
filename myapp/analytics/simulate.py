"""
Traffic simulation, to fill the analytics dashboard with realistic usage before real users arrive.

Run from the repository root:

    python -m myapp.analytics.simulate --visitors 60 --days 14 --output data/analytics_simulated.json
    ANALYTICS_FILE=data/analytics_simulated.json python web_app.py      # Windows: set ANALYTICS_FILE=... first

Every simulated action is a real HTTP request to the Flask application (through its test client), so the whole
tracking pipeline is exercised: cookies, sessions, queries, clicks, dwell-time beacons and outbound clicks.
Only the clock, the IP address and the place are simulated, with headers that the application accepts **only**
in test mode (``X-Sim-Time``, ``X-Sim-IP``, ``X-Sim-Country``, ``X-Sim-City``); every simulated visitor and
session is marked ``simulated=True`` and the dashboard says how many sessions are simulated.

Behaviour model (simple, documented in the report):
* each visitor has a device profile and a place, and 1 to 3 sessions on different days;
* a session has 1 or 2 missions; a mission is a goal (e.g. a white formal shirt) reached with a sequence of
  queries that become more specific, sometimes with a typo, a filter, a sort or another ranking algorithm;
* on a results page the visitor reads from the top (cascade model): the chance to look at a result falls with its
  rank, a result that matches the goal (its title has the goal words) is clicked much more often than one that
  does not, and the dwell time on the product page is long for a match and short otherwise;
* some satisfied visitors go on to the original website (outbound click);
* with ``--with-rag`` every first results page also loads its AI summary (``/api/summary``, as the browser does) and
  the visitor sometimes clicks the product it recommends (a click with ``source="rag"``). This calls the LLM once
  per search, so it needs the API key in ``.env`` and is slow (free-tier rate limits); it is off by default, and
  without a key the summaries are stored as "unavailable".
Clicks therefore depend on the real rankings of the engine: a ranking that shows matching products first gets more
and higher clicks.
"""
import argparse
import json
import os
import random
import re
from datetime import datetime, timedelta

PROFILES = [  # (weight, user agent)
    (30, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    (14, "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"),
    (18, "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"),
    (16, "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36"),
    (8, "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0"),
    (6, "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    (5, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"),
    (3, "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"),
]
PLACES = [  # (weight, country, city)
    (40, "Spain", "Barcelona"), (12, "Spain", "Madrid"), (8, "Spain", "Valencia"), (10, "India", "Bengaluru"),
    (6, "India", "Mumbai"), (6, "France", "Paris"), (5, "Italy", "Milan"), (5, "Germany", "Berlin"),
    (4, "United Kingdom", "London"), (4, "United States", "New York"),
]
HOUR_WEIGHTS = [1, 1, 0, 0, 0, 1, 2, 4, 6, 7, 8, 9, 10, 9, 8, 8, 9, 10, 12, 14, 15, 13, 9, 4]

# Missions: queries from general to specific, and the words a matching product has in its title.
MISSIONS = [
    (["shirt", "men formal shirt", "men white formal shirt slim fit"], ["men", "formal", "shirt"]),
    (["floral shirt", "women floral print shirt"], ["women", "floral", "shirt"]),
    (["track pants women", "women black track pants with zip pockets"], ["women", "black", "track"]),
    (["women t-shirt", "women pure cotton t-shirt pack of 3"], ["women", "pack of 3"]),
    (["running shoes", "black running shoes for men"], ["running", "shoes", "men"]),
    (["jeans", "men slim jeans blue"], ["men", "jeans", "blue"]),
    (["sweatshirt", "women full sleeve sweatshirt cotton"], ["women", "sweatshirt"]),
    (["men jacket", "men winter jacket"], ["men", "jacket"]),
    (["kurta", "women cotton kurta"], ["women", "kurta"]),
    (["socks", "men cotton socks pack of 3"], ["socks"]),
    (["raincoat", "women raincoat"], ["women", "raincoat"]),
    (["men shorts", "men sports shorts"], ["men", "shorts"]),
    (["tie", "men silk tie"], ["tie"]),
    (["leather wallet"], ["wallet"]),                  # not in the catalogue: partial matches
    (["laptop backpack"], ["backpack"]),               # not in the catalogue: no results
]
TYPOS = {"shirt": "shrit", "jeans": "jens", "running": "runing", "sweatshirt": "sweatshrit", "jacket": "jaket",
         "women": "womem", "track": "trak"}


def weighted(rng, items):
    """Random element of [(weight, *values)]."""
    choice = rng.choices(items, weights=[i[0] for i in items])[0]
    return choice[1] if len(choice) == 2 else choice[1:]


class Visitor:
    """One simulated browser: a test client (its own cookies), a device and a place."""

    def __init__(self, app, rng, number):
        self.client = app.test_client()
        self.rng = rng
        self.user_agent = weighted(rng, PROFILES)
        self.country, self.city = weighted(rng, PLACES)
        self.ip = f"198.51.{rng.randint(0, 255)}.{rng.randint(1, 254)}"   # documentation range (RFC 5737)
        self.number = number

    def headers(self, now):
        """Headers of a simulated request: user agent, and the clock, IP and place accepted in test mode."""
        return {"User-Agent": self.user_agent, "X-Sim-Time": now.isoformat(), "X-Sim-IP": self.ip,
                "X-Sim-Country": self.country, "X-Sim-City": self.city}

    def get(self, url, now):
        """GET request of this visitor at the simulated time ``now``."""
        return self.client.get(url, headers=self.headers(now))


def matches(title, words):
    """A result matches the goal if its title contains every goal word (as whole words)."""
    title = " ".join(title.lower().replace("-", " ").split())
    return all(re.search(rf"\b{re.escape(w)}\b", title) for w in words)


def read_summary(visitor, now, rng, qid, goal):
    """
    ``--with-rag``: load the AI summary of a results page; if it recommends a product that matches the goal, the
    visitor sometimes clicks it. Returns (time, satisfied).
    """
    data = visitor.get(f"/api/summary?qid={qid}", now).get_json(silent=True) or {}
    now += timedelta(seconds=rng.uniform(2, 6))   # reads the summary
    best = data.get("best")
    if data.get("status") == "ok" and best and matches(best.get("title", ""), goal) and rng.random() < 0.35:
        visitor.get(f"/doc/{best['pid']}?qid={qid}&rank={best.get('rank', 0)}&page=1&src=rag", now)
        now += timedelta(seconds=rng.lognormvariate(4.0, 0.6))
        return now, rng.random() < 0.6
    return now, False


def run_mission(visitor, now, rng, mission, with_rag=False):
    """Searches and clicks of one mission. Returns the time when it ends."""
    queries, goal = mission
    # sometimes the visitor starts directly with the specific query
    start = 0 if len(queries) == 1 or rng.random() < 0.6 else len(queries) - 1
    method = rng.choices(["ours", "bm25", "tfidf", "word2vec"], weights=[70, 12, 12, 6])[0]
    for query in queries[start:]:
        text = query
        if rng.random() < 0.12:   # typo
            for word, typo in TYPOS.items():
                if word in text.split():
                    text = text.replace(word, typo, 1)
                    break
        params = {"q": text, "method": method}
        if rng.random() < 0.15:
            params["sort"] = rng.choice(["price_asc", "rating", "discount"])
        if rng.random() < 0.15:
            params["in_stock"] = "1"
        if rng.random() < 0.1:
            params["min_rating"] = "4"
        url = "/search?" + "&".join(f"{k}={v.replace(' ', '+')}" for k, v in params.items())
        page = visitor.get(url, now).get_data(as_text=True)
        now += timedelta(seconds=rng.uniform(4, 15))
        qid = re.search(r'data-qid="([0-9a-f]+)"', page)
        results = re.findall(r'<a class="result-title" href="([^"]+)">([^<]+)</a>', page)
        if not qid or not results:
            now += timedelta(seconds=rng.uniform(3, 10))
            continue
        satisfied = False
        if with_rag:
            now, satisfied = read_summary(visitor, now, rng, qid.group(1), goal)
            if satisfied:
                break
        for rank, (href, title) in enumerate(results, start=1):   # cascade: read from the top
            if rng.random() > 0.88 ** (rank - 1):                   # stops reading
                break
            good = matches(title, goal)
            if rng.random() < (0.55 if good else 0.04):
                details = visitor.get(href.replace("&amp;", "&"), now).get_data(as_text=True)
                click_id = re.search(r'click_id: "([0-9a-f]+)"', details).group(1)
                dwell = rng.lognormvariate(4.0, 0.6) if good else rng.lognormvariate(1.8, 0.5)   # ~55 s vs ~6 s
                visitor.client.post("/track/dwell", data=json.dumps({"click_id": click_id, "ms": dwell * 1000}),
                                    content_type="application/json", headers=visitor.headers(now))
                now += timedelta(seconds=dwell)
                if good and rng.random() < 0.3:   # goes to buy on the original website
                    pid = href.split("/doc/")[1].split("?")[0]
                    visitor.get(f"/go/{pid}?qid={qid.group(1)}&rank={rank}", now)
                    now += timedelta(seconds=rng.uniform(5, 30))
                if good and rng.random() < 0.6:
                    satisfied = True
                    break
            now += timedelta(seconds=rng.uniform(1, 4))
        if satisfied and rng.random() < 0.7:
            break
        if rng.random() < 0.2 and len(results) >= 20:   # looks at the second page
            visitor.get(url + "&page=2", now)
            now += timedelta(seconds=rng.uniform(5, 20))
    return now


def simulate(app, analytics, visitors=60, days=14, seed=7, end=None, with_rag=False):
    """Simulate ``visitors`` visitors over the last ``days`` days (``with_rag``: also load the AI summaries)."""
    rng = random.Random(seed)
    end = end or datetime.now().replace(microsecond=0)
    for number in range(visitors):
        visitor = Visitor(app, rng, number)
        day_offsets = sorted(rng.sample(range(days), k=min(days, rng.choices([1, 2, 3], weights=[60, 28, 12])[0])))
        for offset in day_offsets:
            hour = rng.choices(range(24), weights=HOUR_WEIGHTS)[0]
            now = (end - timedelta(days=days - 1 - offset)).replace(hour=hour, minute=rng.randint(0, 59),
                                                                     second=rng.randint(0, 59))
            if now > end:
                now = end - timedelta(hours=rng.randint(1, 6))
            visitor.get("/", now)
            now += timedelta(seconds=rng.uniform(3, 12))
            for mission in rng.sample(MISSIONS, k=rng.choices([1, 2], weights=[65, 35])[0]):
                now = run_mission(visitor, now, rng, mission, with_rag) + timedelta(seconds=rng.uniform(5, 40))
            if rng.random() < 0.08:
                visitor.get("/dashboard", now)
    return analytics


def main():
    """Command line: simulate the visitors and save the analytics tables to a JSON file."""
    parser = argparse.ArgumentParser(description="Simulate visitors of the IRWA search engine.")
    parser.add_argument("--visitors", type=int, default=60)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="data/analytics_simulated.json")
    parser.add_argument("--with-rag", action="store_true",
                        help="also load the AI summary of every first results page (calls the LLM: slow, needs a key)")
    args = parser.parse_args()

    os.environ["ANALYTICS_FILE"] = ""          # the simulation writes its own file at the end
    import web_app
    app, analytics = web_app.app, web_app.analytics_data
    app.testing = True                          # enables the X-Sim-* headers
    analytics.path = None
    analytics.clear()
    simulate(app, analytics, args.visitors, args.days, args.seed, with_rag=args.with_rag)
    analytics.path = args.output
    analytics.save()
    print(f"{len(analytics.visitors)} visitors, {len(analytics.sessions)} sessions, {len(analytics.queries)} searches, "
          f"{len(analytics.clicks)} clicks, {len(analytics.rag)} summaries -> {args.output}")


if __name__ == "__main__":
    main()
