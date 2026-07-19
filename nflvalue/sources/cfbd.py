"""CollegeFootballData (CFBD) API client — stdlib only, budget-aware.

This is the thin data-access layer the CFB build depends on. It intentionally
mirrors the conventions of the existing NFL `sources/` clients (see
`_http.py`, `oddsapi.py`): standard-library HTTP only, no third-party client,
JSON in / dicts out. The one thing it adds over `_http.py` is a **call
counter**, because the CFBD free tier is capped at 1,000 calls/month and the
whole point of Phase 0 is to learn how many calls a real ingest burns.

Auth (verified 2026-07-17): base URL is `https://api.collegefootballdata.com`,
every request carries `Authorization: Bearer <key>`. Get a free key at
https://collegefootballdata.com — no payment info, 1,000 calls/month.

KEY HANDLING — never commit a key:
  1. env var `CFBD_API_KEY`  (preferred for scheduled/CI runs), else
  2. `cfbd_api_key` in the gitignored `config.local.json`, else
  3. `cfbd_api_key` in `config.json` (discouraged — that file IS tracked).

Endpoint notes learned while building this (these matter for the budget and
are exactly the kind of assumption Phase 0 exists to confirm):
  * `/games`  — bulk by (year[, conference][, seasonType]). 1 call/season.
  * `/lines`  — bulk by (year[, conference]). 1 call/season.
  * `/talent` — bulk by (year). ALL teams; filter to a conference client-side.
    1 call/season.
  * `/plays`  — NOT bulk. Requires year + week + seasonType, returns one
    week of plays. So a full regular season is ~15 calls, not 1. The
    portability plan's "most endpoints are bulk season pulls" holds for
    games/lines/talent but NOT for plays — surfaced here rather than
    discovered mid-ingest.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.75

BASE_URL = "https://api.collegefootballdata.com"


class CFBDError(RuntimeError):
    """Raised for auth failures, rate limits, or unexpected HTTP status."""


class CFBDClient:
    """Budget-aware CFBD REST client.

    Every successful or failed request increments ``self.calls`` so a caller
    (the smoke test) can report exactly how much of the 1,000/month budget a
    run consumed — the plan assumes "far fewer than 1,000" and this is how we
    check it instead of assuming.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: float = 20.0):
        self.api_key = api_key or resolve_api_key()
        if not self.api_key:
            raise CFBDError(
                "No CFBD API key. Set env CFBD_API_KEY, or add "
                '"cfbd_api_key" to config.local.json. Free key: '
                "https://collegefootballdata.com"
            )
        self.timeout = timeout
        self.calls = 0            # total HTTP requests attempted this session
        self.call_log: List[str] = []   # human-readable trail for the report

    def _get(self, path: str, params: Optional[Dict] = None,
             _attempt: int = 0) -> list:
        """One GET, with bounded retry on transient 5xx/network errors.

        Retry policy (Workstream A hardening):
          * 5xx and network errors  -> retry, exponential backoff, MAX_RETRIES.
          * 429 (rate limit)        -> NEVER retried. Retrying into a monthly
            cap just burns the cap faster and turns a clear failure into a
            confusing one. Raise immediately.
          * 401                     -> never retried; a bad key stays bad.
        Every attempt counts against `self.calls`, because every attempt is a
        real request against the 1,000/month budget — counting only successes
        would under-report the true burn.
        """
        url = BASE_URL + path
        if params:
            # drop None-valued params so callers can pass optionals freely
            params = {k: v for k, v in params.items() if v is not None}
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "stallionsstallion-cfb/0.1",
                "Accept": "application/json",
            },
        )
        self.calls += 1
        self.call_log.append(f"{path} {params or ''}".strip())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            body = ""
            try:
                body = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 401:
                raise CFBDError("401 Unauthorized — bad/missing CFBD key.") from exc
            if exc.code == 429:
                # Deliberately NOT retried. See docstring.
                raise CFBDError(
                    "429 Too Many Requests — monthly call budget likely "
                    "exhausted. Check https://collegefootballdata.com/api-tiers"
                ) from exc
            if 500 <= exc.code < 600 and _attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** _attempt))
                return self._get(path, params, _attempt=_attempt + 1)
            raise CFBDError(f"HTTP {exc.code} on {path}: {body}") from exc
        except urllib.error.URLError as exc:
            if _attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** _attempt))
                return self._get(path, params, _attempt=_attempt + 1)
            raise CFBDError(f"network error on {path}: {exc.reason}") from exc

    # --- endpoint wrappers (one method per Phase-0 endpoint) --------------

    def games(self, year: int, conference: Optional[str] = None,
              season_type: str = "regular") -> list:
        """Schedule + final scores. Bulk: 1 call."""
        return self._get("/games", {
            "year": year, "conference": conference, "seasonType": season_type,
        })

    def lines(self, year: int, conference: Optional[str] = None) -> list:
        """Betting lines (multi-book, historical). Bulk: 1 call."""
        return self._get("/lines", {"year": year, "conference": conference})

    def talent(self, year: int) -> list:
        """247 Team Talent Composite for ALL teams. Bulk: 1 call.

        No conference filter server-side — caller filters client-side.
        """
        return self._get("/talent", {"year": year})

    def plays(self, year: int, week: int, season_type: str = "regular",
              conference: Optional[str] = None) -> list:
        """Play-by-play for ONE week. NOT bulk — 1 call per week."""
        return self._get("/plays", {
            "year": year, "week": week, "seasonType": season_type,
            "conference": conference,
        })

    # --- Workstream A additions (probed live 2026-07-19 before coding) ------

    def teams(self, year: int) -> list:
        """All teams for a season incl. `conference` and `classification`.

        Bulk: 1 call, ~674 rows (133 of them FBS). This is the ONLY correct
        source for conference membership: it is per-season, so realignment is
        handled by re-pulling rather than by editing a hardcoded map. Also
        carries `classification` ('fbs'/'fcs'/...), which is how FCS opponents
        get tagged as a separate tier instead of being fed to the FBS fit at
        face value.
        """
        return self._get("/teams", {"year": year})

    def returning_production(self, year: int) -> list:
        """Returning production (percentPPA and usage splits). Bulk: 1 call.

        **DO NOT PASS A CONFERENCE FILTER.** Probed live 2026-07-19: this
        endpoint accepts `conference` but SILENTLY RETURNS ZERO ROWS for it —
        HTTP 200, empty list, no error — for every year tried. `year` alone
        returns 131 rows; `year` + `team` returns 1. An ingest that passed
        conference through uniformly (as it can for /games and /lines) would
        get an empty returning-production table and never be told. Pull by
        year and filter client-side, exactly like /talent.
        """
        return self._get("/player/returning", {"year": year})

    def portal(self, year: int) -> list:
        """Transfer portal entries for a season. Bulk: 1 call (~2.5k rows)."""
        return self._get("/player/portal", {"year": year})

    def roster(self, year: int, team: str) -> list:
        """Roster for ONE team. NOT bulk — 1 call per team, so ~133/season
        for all of FBS. Budget this deliberately; it is by far the most
        expensive pull in the project against the 1,000/month cap."""
        return self._get("/roster", {"year": year, "team": team})

    def venues(self) -> list:
        """Stadiums incl. dome/grass/elevation/lat-lon. Bulk: 1 call, no year."""
        return self._get("/venues", {})


def resolve_api_key() -> str:
    """env CFBD_API_KEY > config.local.json > config.json. Never raises."""
    env = os.environ.get("CFBD_API_KEY")
    if env:
        return env.strip()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("config.local.json", "config.json"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    key = json.load(fh).get("cfbd_api_key", "")
                if key:
                    return str(key).strip()
            except Exception:  # noqa: BLE001
                continue
    return ""
