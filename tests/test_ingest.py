"""Schema-contract and behaviour tests for the CFBD ingest (Workstream A).

Written fresh against the CFBD rewrite, 2026-07-19 — the previous contents
were a stub of the old nflverse-shaped test, which did not carry over.

Everything here runs OFFLINE against fake CFBD payloads. That is deliberate:
these tests must run in CI and on a laptop with no key, and they must not
burn the 1,000-call/month budget. Live shape confirmation is gate 0's job
(phase0_smoke_test.py), not this file's.
"""

import json
import os

import pandas as pd
import pytest

from nflvalue import ingest
from nflvalue.sources.cfbd import CFBDError


# --------------------------------------------------------------------------- #
# Fake CFBD payloads — camelCase, matching the real API shape confirmed live.
# --------------------------------------------------------------------------- #
TEAMS = [
    {"school": "Boise State", "conference": "Mountain West", "classification": "fbs"},
    {"school": "Air Force", "conference": "Mountain West", "classification": "fbs"},
    {"school": "Idaho State", "conference": "Big Sky", "classification": "fcs"},
]
GAMES = [
    {"id": 1, "season": 2023, "week": 1, "startDate": "2023-08-26T18:00:00.000Z",
     "neutralSite": False, "homeTeam": "Boise State", "awayTeam": "Idaho State",
     "homePoints": 42, "awayPoints": 7,
     "homeConference": "Mountain West", "awayConference": "Big Sky"},
    {"id": 2, "season": 2023, "week": 2, "startDate": "2023-09-02T18:00:00.000Z",
     "neutralSite": False, "homeTeam": "Air Force", "awayTeam": "Boise State",
     "homePoints": 21, "awayPoints": 24,
     "homeConference": "Mountain West", "awayConference": "Mountain West"},
]
LINES = [
    {"id": 1, "season": 2023, "week": 1, "homeTeam": "Boise State",
     "awayTeam": "Idaho State", "lines": [
         {"provider": "DraftKings", "spread": -31.5, "spreadOpen": -30.0,
          "overUnder": 55.5, "overUnderOpen": 54.0,
          "homeMoneyline": -20000, "awayMoneyline": 5000},
         {"provider": "Bovada", "spread": -31.0, "spreadOpen": -29.5,
          "overUnder": 55.0, "overUnderOpen": 54.5,
          "homeMoneyline": None, "awayMoneyline": None},
     ]},
]
TALENT = [
    {"year": 2023, "team": "Boise State", "talent": 579.88},
    {"year": 2023, "team": "Air Force", "talent": 0},      # sentinel
    {"year": 2023, "team": "Alabama", "talent": 1015.43},
]
RETURNING = [
    {"season": 2023, "team": "Boise State", "percentPPA": 0.61,
     "percentPassingPPA": 0.7, "percentRushingPPA": 0.5,
     "percentReceivingPPA": 0.55},
]
PLAYS_WK1 = [
    {"gameId": 1, "period": 1, "offense": "Boise State", "defense": "Idaho State",
     "offenseScore": 0, "defenseScore": 0, "down": 1, "distance": 10,
     "yardsToGoal": 75, "yardsGained": 6, "playType": "Rush", "ppa": 0.2},
    {"gameId": 1, "period": 4, "offense": "Boise State", "defense": "Idaho State",
     "offenseScore": 42, "defenseScore": 7, "down": 1, "distance": 10,
     "yardsToGoal": 60, "yardsGained": 3, "playType": "Rush", "ppa": 0.05},
]


class FakeClient:
    """Stands in for CFBDClient. Counts calls the same way the real one does."""

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def _bump(self, name):
        self.calls += 1
        if self.fail_on == name:
            raise CFBDError(f"simulated outage on {name}")

    def teams(self, year):                 self._bump("teams");     return TEAMS
    def games(self, year, conference=None): self._bump("games");    return GAMES
    def lines(self, year, conference=None): self._bump("lines");    return LINES
    def talent(self, year):                self._bump("talent");    return TALENT
    def returning_production(self, year):  self._bump("returning"); return RETURNING

    def plays(self, year, week, conference=None):
        self._bump("plays")
        return PLAYS_WK1 if week == 1 else []


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ingest, "TALLY_PATH", str(tmp_path / "_call_tally.json"))
    return tmp_path


# --------------------------------------------------------------------------- #
# Schema contracts
# --------------------------------------------------------------------------- #
def test_every_normalizer_satisfies_its_contract():
    fbs = {"Boise State", "Air Force"}
    ingest.assert_schema(ingest.normalize_teams(TEAMS, 2023), "teams")
    ingest.assert_schema(ingest.normalize_games(GAMES, 2023, fbs), "games")
    ingest.assert_schema(ingest.normalize_lines(LINES, 2023), "lines")
    ingest.assert_schema(ingest.normalize_talent(TALENT, 2023), "talent")
    ingest.assert_schema(ingest.normalize_returning(RETURNING, 2023), "returning")
    ingest.assert_schema(ingest.normalize_plays(PLAYS_WK1, 2023, 1), "plays")


def test_schema_violation_is_loud_not_silent():
    df = ingest.normalize_games(GAMES, 2023, {"Boise State", "Air Force"})
    with pytest.raises(ingest.SchemaError, match="missing required columns"):
        ingest.assert_schema(df.drop(columns=["home_points"]), "games")


# --------------------------------------------------------------------------- #
# The talent sentinel — the thing gate 0 caught
# --------------------------------------------------------------------------- #
def test_talent_zero_becomes_null_plus_flag_never_zero():
    t = ingest.normalize_talent(TALENT, 2023)
    af = t.loc[t["team"].astype(str) == "Air Force"].iloc[0]
    assert bool(af["unranked_recruiting"]) is True
    assert pd.isna(af["talent"]), "talent==0 must become NULL, never stay 0.0"
    bsu = t.loc[t["team"].astype(str) == "Boise State"].iloc[0]
    assert bool(bsu["unranked_recruiting"]) is False
    assert float(bsu["talent"]) == pytest.approx(579.88)
    # A measured team must never be silently swept into the sentinel bucket.
    assert t["unranked_recruiting"].sum() == 1


# --------------------------------------------------------------------------- #
# FCS as a separate tier
# --------------------------------------------------------------------------- #
def test_fcs_opponents_are_tagged_not_treated_as_weak_fbs():
    teams = ingest.normalize_teams(TEAMS, 2023)
    fbs = set(teams.loc[teams["is_fbs"], "team"].astype(str))
    assert "Idaho State" not in fbs
    g = ingest.normalize_games(GAMES, 2023, fbs)
    blowout = g.loc[g["id"] == 1].iloc[0]
    assert bool(blowout["away_fcs"]) is True
    conf_game = g.loc[g["id"] == 2].iloc[0]
    assert bool(conf_game["home_fcs"]) is False
    assert bool(conf_game["away_fcs"]) is False


# --------------------------------------------------------------------------- #
# Garbage time filtered at ingest, both tables kept
# --------------------------------------------------------------------------- #
def test_garbage_time_filtered_at_ingest_and_raw_retained(cache):
    res = ingest.refresh_season(2023, weeks=[1], client=FakeClient())
    assert not res.stale
    raw = ingest.load_plays([2023])
    clean = ingest.load_plays_clean([2023])
    assert len(raw) == 2
    assert len(clean) == 1, "the 4th-quarter 42-7 play is garbage time"
    assert bool(raw.loc[raw["period"] == 4, "garbage_time"].iloc[0]) is True
    assert bool(raw.loc[raw["period"] == 1, "garbage_time"].iloc[0]) is False


# --------------------------------------------------------------------------- #
# Per-season conference lookup
# --------------------------------------------------------------------------- #
def test_conference_is_looked_up_per_season_not_hardcoded(cache):
    ingest.refresh_season(2023, weeks=[1], client=FakeClient())
    assert ingest.conference_of(2023, "Boise State") == "Mountain West"
    assert ingest.conference_of(2023, "Idaho State") == "Big Sky"
    assert ingest.conference_of(2023, "Nonexistent Tech") == ""


# --------------------------------------------------------------------------- #
# Fail-loud staleness + atomicity + idempotency
# --------------------------------------------------------------------------- #
def test_outage_keeps_cache_and_reports_stale(cache):
    ok = ingest.refresh_season(2023, weeks=[1], client=FakeClient())
    assert not ok.stale
    before = ingest.load_games([2023])

    broken = ingest.refresh_season(2023, weeks=[1],
                                   client=FakeClient(fail_on="lines"))
    assert broken.stale is True
    assert "simulated outage" in broken.error
    after = ingest.load_games([2023])
    pd.testing.assert_frame_equal(before, after)


def test_refresh_is_idempotent(cache):
    a = ingest.refresh_season(2023, weeks=[1], client=FakeClient())
    first = {k: ingest.load(k, [2023]) for k in ("games", "talent", "plays")}
    b = ingest.refresh_season(2023, weeks=[1], client=FakeClient())
    second = {k: ingest.load(k, [2023]) for k in ("games", "talent", "plays")}
    assert a.rows == b.rows
    for k in first:
        pd.testing.assert_frame_equal(first[k], second[k])


def test_call_count_is_tallied_against_the_monthly_cap(cache):
    client = FakeClient()
    res = ingest.refresh_season(2023, weeks=[1, 2, 3], client=client)
    # teams + games + lines + talent + returning + 3 weeks of plays = 8
    assert res.calls == 8
    assert client.calls == 8
    assert ingest.calls_used_this_month() == 8
    ingest.refresh_season(2023, weeks=[1], client=FakeClient())
    assert ingest.calls_used_this_month() == 8 + 6


def test_missing_season_loads_empty_not_exception(cache):
    assert ingest.load_games([1999]).empty
