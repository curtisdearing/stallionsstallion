"""Tiny JSON-over-HTTP helper (standard library only)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, Optional


def get_json(url: str, params: Optional[Dict] = None, timeout: float = 15.0):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-value/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
