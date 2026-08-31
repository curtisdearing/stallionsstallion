"""Canonical user-facing calendar labels for college-football schedules."""

from __future__ import annotations

from typing import Any

import pandas as pd

PUBLIC_TIMEZONE = "America/New_York"


def public_game_date(start_date: Any) -> str:
    """Return a scheduled kickoff's calendar date in the public ET timezone.

    CFBD kickoff timestamps are UTC. Public schedule labels must use the
    Eastern calendar so late-night Saturday games do not become Sunday rows.
    """
    timestamp = pd.to_datetime(start_date, errors="raise", utc=True)
    return timestamp.tz_convert(PUBLIC_TIMEZONE).strftime("%Y-%m-%d")


def public_slate(season: int, week: int, start_date: object) -> tuple[str, str]:
    """Return the public `(slate, date)` pair for a scheduled kickoff."""
    date = public_game_date(start_date)
    slate = "wk0" if int(week) == 1 and date < f"{int(season)}-09-01" else f"wk{int(week)}"
    return slate, date
