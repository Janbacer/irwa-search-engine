"""
User context of a request: browser, operating system, device type, IP address and (optional) country and city.

* Browser and OS come from the User-Agent header (``httpagentparser``, as in the course template).
* Device: "mobile", "tablet" or "computer", from the User-Agent.
* IP: the first address of X-Forwarded-For (when the app runs behind a proxy), else the remote address.
* Country and city: looked up once per IP with the free service ip-api.com (no key needed). The look-up is skipped
  for private and local addresses (e.g. 127.0.0.1 when the app runs on your computer), which are reported as
  "Local network". Set GEOIP_ENABLED=0 in .env to disable it (no data leaves the server then).
"""
import ipaddress
import json
import os
import urllib.request
from functools import lru_cache

import httpagentparser

GEO_URL = "http://ip-api.com/json/{ip}?fields=status,country,city"


def client_ip(request):
    """IP address of the client of a Flask request."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "")


def parse_user_agent(user_agent):
    """{'browser', 'os', 'device'} from a User-Agent string."""
    ua = user_agent or ""
    detected = httpagentparser.detect(ua)
    browser = detected.get("browser", {}).get("name") or "Unknown"
    os_name = (detected.get("os", {}).get("name") or detected.get("platform", {}).get("name") or "Unknown")
    low = ua.lower()
    if "ipad" in low or "tablet" in low or ("android" in low and "mobi" not in low):
        device = "tablet"
    elif "mobi" in low or "iphone" in low or "android" in low:
        device = "mobile"
    elif any(bot in low for bot in ("bot", "crawler", "spider", "curl", "python-requests")):
        device = "bot"
    else:
        device = "computer"
    return {"browser": browser, "os": os_name, "device": device}


@lru_cache(maxsize=4096)
def locate(ip):
    """(country, city) of an IP address; ("Local network", "") for private ones; ("Unknown", "") on failure."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return "Unknown", ""
    if address.is_private or address.is_loopback or address.is_link_local:
        return "Local network", ""
    if os.getenv("GEOIP_ENABLED", "1") == "0":
        return "Unknown", ""
    try:
        with urllib.request.urlopen(GEO_URL.format(ip=ip), timeout=1.5) as answer:
            data = json.load(answer)
        if data.get("status") == "success":
            return data.get("country") or "Unknown", data.get("city") or ""
    except (OSError, ValueError):
        pass
    return "Unknown", ""


def user_context(request):
    """Browser, OS, device, IP, country and city of a Flask request."""
    context = parse_user_agent(request.headers.get("User-Agent"))
    ip = client_ip(request)
    country, city = locate(ip)
    context.update({"ip": ip, "country": country, "city": city})
    return context
