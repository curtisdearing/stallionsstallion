"""Synthetic frames for the context tests and for a panel smoke render.

Deliberately small and obviously fake: the contract and leakage tests must
never depend on the real cache being present, or on a particular season's
data being complete.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd

KICKOFF = "2026-08-29T23:00:00.000Z"
DEFAULT_OBSERVED = dt.datetime(2026, 8, 28, 3, 55, tzinfo=dt.timezone.utc)


def demo_game() -> Dict[str, Any]:
    return {"id": 999001, "season": 2026, "week": 3, "start_date": KICKOFF,
            "home_team": "Demo Home", "away_team": "Demo Away",
            "neutral_site": False}


def _stamp(df: pd.DataFrame, observed: dt.datetime) -> pd.DataFrame:
    df.attrs["observed_at"] = observed
    return df


def demo_frames(observed_at: Optional[dt.datetime] = None) -> Dict[str, pd.DataFrame]:
    obs = observed_at or DEFAULT_OBSERVED
    games = pd.DataFrame([
        {"id": 999001, "season": 2026, "week": 3, "start_date": KICKOFF,
         "home_team": "Demo Home", "away_team": "Demo Away",
         "neutral_site": False},
        {"id": 998001, "season": 2026, "week": 2,
         "start_date": "2026-08-22T23:00:00.000Z",
         "home_team": "Demo Home", "away_team": "Other", "neutral_site": False},
        {"id": 998002, "season": 2026, "week": 1,
         "start_date": "2026-08-15T23:00:00.000Z",
         "home_team": "Other", "away_team": "Demo Away", "neutral_site": False},
    ])
    teams = pd.DataFrame([
        {"season": 2026, "team": "Demo Home", "conference": "Demo Big"},
        {"season": 2026, "team": "Demo Away", "conference": "Demo Small"},
        {"season": 2025, "team": "Demo Home", "conference": "Demo Small"},
    ])
    talent = pd.DataFrame([
        {"season": 2026, "team": "Demo Home", "talent": 800.0,
         "unranked_recruiting": False},
        {"season": 2026, "team": "Demo Away", "talent": 0.0,
         "unranked_recruiting": True},
    ])
    returning = pd.DataFrame([
        {"season": 2026, "team": "Demo Home", "percent_ppa": 0.61,
         "percent_passing_ppa": 0.90, "percent_rushing_ppa": 0.5,
         "percent_receiving_ppa": 0.4},
        {"season": 2026, "team": "Demo Away", "percent_ppa": 0.22,
         "percent_passing_ppa": 0.10, "percent_rushing_ppa": 0.3,
         "percent_receiving_ppa": 0.2},
    ])
    lines = pd.DataFrame([
        {"game_id": 999001, "provider": "book_a", "spread_open": -6.5,
         "over_under_open": 51.5},
        {"game_id": 999001, "provider": "book_b", "spread_open": -6.0,
         "over_under_open": 52.0},
    ])
    ratings = pd.DataFrame([
        {"season": 2026, "week": 3, "team": "Demo Home", "connectivity": 1.0,
         "fbs_games": 80, "net_band": 7.3, "hfa": 2.4, "net": 8.0},
        {"season": 2026, "week": 3, "team": "Demo Away", "connectivity": 1.0,
         "fbs_games": 74, "net_band": 7.5, "hfa": 2.4, "net": -2.0},
    ])
    drives = pd.DataFrame([
        {"season": 2026, "week": 1, "game_id": 998002, "team": "Demo Away",
         "points_per_trip": 4.1, "start_field_position": 28.0,
         "special_teams_epa": -0.2, "turnover_luck": 0.4},
        {"season": 2026, "week": 2, "game_id": 998001, "team": "Demo Home",
         "points_per_trip": 5.2, "start_field_position": 32.0,
         "special_teams_epa": 0.3, "turnover_luck": -0.1},
        {"season": 2026, "week": 2, "game_id": 998001, "team": "Demo Away",
         "points_per_trip": 3.9, "start_field_position": 26.0,
         "special_teams_epa": -0.1, "turnover_luck": 0.2},
    ])
    return {k: _stamp(v, obs) for k, v in
            (("games", games), ("teams", teams), ("talent", talent),
             ("returning", returning), ("lines", lines), ("ratings", ratings),
             ("drives", drives))}
