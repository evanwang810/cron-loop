"""Manifold API client. Reads need no key; orders are only possible when MANIFOLD_API_KEY is set."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.manifold.markets/v0"
KEY = os.environ.get("MANIFOLD_API_KEY", "").strip()
GAP = 0.14  # seconds between requests, about 430 a minute against Manifold's 500
_next = 0.0


def _wait() -> None:
    global _next
    now = time.monotonic()
    if now < _next:
        time.sleep(_next - now)
    _next = max(now, _next) + GAP


def get(path: str, **params: object) -> object:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{BASE}/{path}" + (f"?{query}" if query else "")
    req = urllib.request.Request(url, headers={"User-Agent": "loop"})
    for attempt in range(5):
        _wait()
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {url}")


def post(path: str, body: dict) -> dict:
    """One attempt only: retrying an order that may have gone through could bet twice."""
    if not KEY:
        raise RuntimeError("no API key")
    req = urllib.request.Request(f"{BASE}/{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"User-Agent": "loop", "Content-Type": "application/json", "Authorization": f"Key {KEY}"})
    _wait()
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)
