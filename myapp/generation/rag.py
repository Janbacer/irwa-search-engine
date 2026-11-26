"""
Retrieval-Augmented Generation: a summary of the results page written by an LLM from the retrieved products.

Two generators share the same interface, ``generate(query, response) -> dict``:

* ``BaselineRAG``: the pipeline given in the course template, unchanged (prompt, top 20 results, PID and title
  only). It is kept to compare against.
* ``ImprovedRAG``: our version, with three improvements (explained in the report):
    1. **Richer, smaller context**: the top 8 products with the metadata a shopper decides on (brand, price,
       discount, rating, stock, key product details and the start of the description) instead of 20 titles;
       near-duplicates (same title, brand and price: often one product in several colours) are listed once.
    2. **Grounded, structured prompt**: a system message with rules (use only the listed products, check every
       part of the request, prefer products in stock, cite PIDs, never invent prices or features), a low
       temperature and a JSON answer that the page can render and the code can check.
    3. **Handling "no good products"** before and after the LLM: no LLM call when the search found nothing; the
       model is told when the products only match part of the query; and every PID in the answer is checked
       against the retrieved products (invented PIDs are removed, and a recommendation with no valid PID becomes
       "no good products").

The LLM is reached through a third-party API: Groq (default, as in the course template; model openai/gpt-oss-20b,
because the template's llama-3.1-8b-instant is no longer offered) or Anthropic.
Configuration in ``.env``: ``LLM_PROVIDER`` (groq | anthropic), ``GROQ_API_KEY`` and ``GROQ_MODEL``, or
``ANTHROPIC_API_KEY`` and ``ANTHROPIC_MODEL``.
"""
import json
import os
import re
import time

NO_GOOD_PRODUCTS = "There are no good products that fit the request based on the retrieved results."
UNAVAILABLE = "RAG is not available. Check your credentials (.env file) or account limits."


# ---------------------------------------------------------------------------------------------- LLM client
class LLMClient:
    """Minimal wrapper of the chat API of Groq or Anthropic: ``complete(system, user) -> text``."""

    def __init__(self, provider=None, model=None):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "groq").lower()
        if self.provider == "anthropic":
            self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5"
        else:
            self.provider = "groq"
            # the template's llama-3.1-8b-instant is no longer offered by Groq (November 2026)
            self.model = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
        self.last_error = ""

    def available(self):
        """True if an API key is configured for the provider."""
        key = os.getenv("ANTHROPIC_API_KEY" if self.provider == "anthropic" else "GROQ_API_KEY", "")
        return bool(key) and not key.startswith("<")

    def complete(self, user, system=None, temperature=None, json_mode=False, max_tokens=2000):
        """
        Send one prompt and return the text of the answer (raises an exception on any API error).
        gpt-oss models are reasoning models: they get reasoning_effort="low" (their hidden reasoning also uses
        output tokens, hence max_tokens=2000). If the model refuses JSON mode, the request is sent without it.
        """
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            kwargs = {"system": system} if system else {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            answer = client.messages.create(model=self.model, max_tokens=max_tokens,
                                            messages=[{"role": "user", "content": user}], **kwargs)
            return "".join(block.text for block in answer.content if getattr(block, "type", "") == "text")
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kwargs = {"temperature": temperature} if temperature is not None else {}
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = "low"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            answer = client.chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                    **kwargs)
        except Exception as error:
            if not json_mode or "response_format" not in str(error):
                raise
            kwargs.pop("response_format")
            answer = client.chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                    **kwargs)
        return answer.choices[0].message.content or ""


def _result(variant, status, model, started, **extra):
    """The dictionary returned by both generators."""
    out = {"variant": variant, "status": status, "model": model, "latency_ms": (time.perf_counter() - started) * 1000,
           "text": "", "best": None, "alternatives": [], "summary": "", "cited_valid": 0, "cited_invalid": 0,
           "n_products": 0, "prompt_chars": 0}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------------------------- baseline
class BaselineRAG:
    """The RAG pipeline of the course template, unchanged (kept as the baseline)."""
    variant = "baseline"
    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Identify the single best product that matches the user's request.
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        3. If there is another product that could also work, mention it briefly as an alternative.
        4. If no product is a good fit, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """

    def __init__(self, client=None, top_n=20):
        self.client = client or LLMClient()
        self.top_n = top_n

    def generate(self, query, response):
        """Template behaviour: top 20 results (PID and title) in the prompt, free-text answer."""
        started = time.perf_counter()
        products = response.top_results[: self.top_n]
        if not self.client.available():
            return _result(self.variant, "unavailable", self.client.model, started, text=UNAVAILABLE)
        formatted = "\n".join(f"- PID: {p.pid}, Title: {p.title}" for p in products)
        prompt = self.PROMPT_TEMPLATE.format(retrieved_results=formatted, user_query=query)
        try:
            text = self.client.complete(prompt)
        except Exception as error:   # the template catches every API error the same way
            print(f"Error during RAG generation: {error}")
            return _result(self.variant, "unavailable", self.client.model, started, text=UNAVAILABLE,
                           error=str(error))
        valid, invalid = check_pids(text, {p.pid for p in products})
        status = "no_good_products" if NO_GOOD_PRODUCTS.lower()[:40] in text.lower() else "ok"
        return _result(self.variant, status, self.client.model, started, text=text, n_products=len(products),
                       prompt_chars=len(prompt), cited_valid=len(valid), cited_invalid=len(invalid),
                       best_pid=first_pid(text))


# ---------------------------------------------------------------------------------------------- improved
KEY_DETAILS = ("Color", "Fabric", "Fit", "Pattern", "Occasion", "Ideal For", "Type", "Sleeve", "Neck Type",
               "Closure", "Pack of", "Outer Material", "Sole Material", "Brand Color")


class ImprovedRAG:
    """Our RAG: richer context, grounded JSON prompt, and handling of "no good products" (see the module doc)."""
    variant = "improved"
    SYSTEM = (
        "You are a careful shopping assistant of a fashion e-commerce website. You only know the products listed in "
        "the message; you never invent products, PIDs, prices or features. You answer in JSON only."
    )
    PROMPT_TEMPLATE = """A user searched our shop for: "{query}"
{match_note}
These are the top {n} products returned by our search engine, in ranking order:

{products}

Task:
1. Find the explicit requirements of the request (product type, gender, colour, fit, material, size or pack size,
   occasion, budget...). A product is a good fit only if it meets all of them according to the data above.
2. Choose the single best product among the good fits. Prefer products in stock; use rating, price and discount
   only to choose between products that fit equally well.
3. Optionally choose up to 2 alternatives that also fit, and say how they differ (cheaper, better rated...).
4. If no product meets the requirements, set "status" to "no_good_products" and explain in "summary" what is
   missing (for example, the shop has no product of that type).

Answer with this JSON object only:
{{"status": "ok" or "no_good_products",
  "best": {{"pid": "...", "why": "one or two sentences with concrete facts from the data"}} or null,
  "alternatives": [{{"pid": "...", "why": "..."}}],
  "summary": "two sentences for the shopper: what the results contain and what to look at"}}"""

    def __init__(self, client=None, top_n=8, temperature=0.2):
        self.client = client or LLMClient()
        self.top_n = top_n
        self.temperature = temperature

    @staticmethod
    def describe(item):
        """One product as a compact block of facts for the prompt."""
        price = f"Rs {item.selling_price:,.0f}" if item.selling_price is not None else "unknown price"
        if item.actual_price and item.selling_price is not None and item.actual_price > item.selling_price:
            price += f" (was Rs {item.actual_price:,.0f}, {item.discount:.0f}% off)" if item.discount else ""
        rating = f"{item.average_rating:.1f}/5" if item.average_rating is not None else "no rating"
        details = "; ".join(f"{k}: {item.details[k]}" for k in KEY_DETAILS if item.details.get(k))
        description = " ".join((item.description or "").split())[:200]
        lines = [f"[{item.rank}] PID {item.pid} | {item.title}",
                 f"    brand: {item.brand or 'unknown'} | price: {price} | rating: {rating} | "
                 f"{'in stock' if item.in_stock else 'OUT OF STOCK'} | category: {item.sub_category}"]
        if details:
            lines.append(f"    details: {details}")
        if description:
            lines.append(f"    description: {description}")
        return "\n".join(lines)

    def build_prompt(self, query, response, products):
        """The user message of the prompt."""
        if response.match_type == "partial":
            note = (f"\nWARNING: no product contains all the words of the request; these products contain only "
                    f"{response.matched_terms} of its {len(set(response.terms))} key words, so they may not fit.\n")
        elif response.match_type == "corrected":
            note = f'\n(The search corrected the spelling of the request to "{response.corrected_query}".)\n'
        else:
            note = ""
        low, high = response.query_price
        if low is not None or high is not None:
            note += (f"\n(The search kept only products priced {'from Rs %.0f ' % low if low is not None else ''}"
                     f"{'up to Rs %.0f' % high if high is not None else ''}, as the request asks.)\n")
        return self.PROMPT_TEMPLATE.format(query=query, match_note=note, n=len(products),
                                           products="\n".join(self.describe(p) for p in products))

    def select(self, response):
        """The first ``top_n`` retrieved products, skipping near-duplicates (same title, brand and price)."""
        products, seen = [], set()
        for item in response.top_results:
            key = (item.title.lower(), (item.brand or "").lower(), item.selling_price)
            if key not in seen:
                seen.add(key)
                products.append(item)
            if len(products) == self.top_n:
                break
        return products

    def generate(self, query, response):
        """Summary of the results page as a dict (status ok / no_good_products / no_results / unavailable)."""
        started = time.perf_counter()
        products = self.select(response)
        if not products:   # improvement 3: nothing retrieved -> no LLM call, a clear answer
            return _result(self.variant, "no_results", self.client.model, started,
                           summary=f'No product of our shop matches "{query}". Try other words or fewer filters.')
        if not self.client.available():
            return _result(self.variant, "unavailable", self.client.model, started, text=UNAVAILABLE)
        prompt = self.build_prompt(query, response, products)
        try:
            text = self.client.complete(prompt, system=self.SYSTEM, temperature=self.temperature, json_mode=True)
        except Exception as error:
            print(f"Error during RAG generation: {error}")
            return _result(self.variant, "unavailable", self.client.model, started, text=UNAVAILABLE,
                           error=str(error))
        return self.validate(text, products, started, len(prompt))

    def validate(self, text, products, started, prompt_chars):
        """
        Parse the JSON answer and check it against the retrieved products (improvement 3): recommendations whose
        PID is not in the context are dropped; if the best product is dropped, the best valid alternative takes
        its place, and with none left the answer becomes "no good products".
        """
        by_pid = {p.pid: p for p in products}
        data = parse_json(text)
        if data is None:   # not JSON: show the text, but do not trust it as a recommendation
            valid, invalid = check_pids(text, set(by_pid))
            return _result(self.variant, "ok", self.client.model, started, text=text, n_products=len(products),
                           prompt_chars=prompt_chars, cited_valid=len(valid), cited_invalid=len(invalid))
        picks, invalid = [], 0
        for entry in [data.get("best")] + list(data.get("alternatives") or []):
            if not isinstance(entry, dict) or not entry.get("pid"):
                continue
            pid = str(entry["pid"]).strip().upper()
            if pid in by_pid and pid not in [p["pid"] for p in picks]:
                item = by_pid[pid]
                picks.append({"pid": pid, "why": str(entry.get("why", "")).strip(), "title": item.title,
                              "rank": item.rank, "price": item.selling_price, "rating": item.average_rating,
                              "in_stock": item.in_stock})
            else:
                invalid += 1
        status = str(data.get("status", "ok")).lower()
        if status != "no_good_products" and not picks:
            status = "no_good_products"
        if status == "no_good_products":
            picks = []
        return _result(self.variant, status, self.client.model, started, n_products=len(products),
                       prompt_chars=prompt_chars, best=picks[0] if picks else None, alternatives=picks[1:3],
                       summary=str(data.get("summary", "")).strip() or (NO_GOOD_PRODUCTS if not picks else ""),
                       cited_valid=len(picks), cited_invalid=invalid, best_pid=picks[0]["pid"] if picks else None,
                       raw=text)


# ---------------------------------------------------------------------------------------------- helpers
PID_PATTERN = re.compile(r"\b[A-Z]{3}[A-Z0-9]{13}\b")   # Flipkart PIDs: 16 upper-case letters and digits


def check_pids(text, allowed):
    """(valid, invalid): the PIDs written in ``text`` that are / are not among ``allowed``."""
    text = text or ""
    valid = [p for p in allowed if p in text]
    invalid = [p for p in dict.fromkeys(PID_PATTERN.findall(text)) if p not in allowed]
    return valid, invalid


def first_pid(text):
    """First PID written in a free-text answer (the baseline's "Best Product")."""
    found = PID_PATTERN.findall(text or "")
    return found[0] if found else None


def parse_json(text):
    """The first JSON object in ``text`` (models sometimes add text around it), or None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except ValueError:
                return None
    return None


class RAGGenerator:
    """Facade used by the web application: ``generate(query, response, variant="improved")``."""

    def __init__(self, client=None):
        self.client = client or LLMClient()
        self.generators = {"improved": ImprovedRAG(self.client), "baseline": BaselineRAG(self.client)}

    def generate(self, query, response, variant="improved"):
        """Summary of a results page with the chosen variant."""
        return self.generators.get(variant, self.generators["improved"]).generate(query, response)
