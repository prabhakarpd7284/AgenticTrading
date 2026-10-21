"""Polite HTTP for the news collectors: browser UA, timeouts, per-host pacing.

Collectors call ``fetch`` (tests patch it). Never used to evade bot
protection — Moneycontrol, which blocks automated access, is not a source.
"""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import requests

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
DEFAULT_TIMEOUT = 15
# Minimum seconds between two requests to the same host.
HOST_PACING = {"news.google.com": 1.0, "www.nseindia.com": 0.5}
DEFAULT_PACING = 0.3
MAX_BYTES = 8 * 1024 * 1024   # refuse anything absurd

_last_hit: dict[str, float] = {}
_lock = threading.Lock()


class FetchError(Exception):
    pass


def _pace(host: str) -> None:
    gap = HOST_PACING.get(host, DEFAULT_PACING)
    with _lock:
        last = _last_hit.get(host, 0.0)
        wait = last + gap - time.monotonic()
        _last_hit[host] = max(time.monotonic(), last + gap)
    if wait > 0:
        time.sleep(wait)


def fetch(url: str, *, accept: str = "*/*", referer: str | None = None,
          timeout: float = DEFAULT_TIMEOUT) -> bytes:
    host = urlparse(url).netloc
    _pace(host)
    headers = {"User-Agent": BROWSER_UA, "Accept": accept,
               "Accept-Language": "en-IN,en;q=0.9"}
    if referer:
        headers["Referer"] = referer
    try:
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
    except requests.RequestException as e:
        raise FetchError(f"{host}: {type(e).__name__}: {e}") from e
    try:
        if r.status_code != 200:
            raise FetchError(f"{host}: HTTP {r.status_code}")
        body = r.raw.read(MAX_BYTES + 1, decode_content=True)
        if len(body) > MAX_BYTES:
            raise FetchError(f"{host}: response too large")
        return body
    finally:
        r.close()
