"""Weekly loop (Workstream I Phase 1) — the spec's DoD, tested offline.

Covers: single-writer lock (live blocks, stale overrides), budget guard
refusal, stage sentinels (resume, don't restart), manifest completeness,
offseason detection, and the corrected STATE_GLOBS actually matching this
repo's real state files.
"""

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WL = _load("weekly_loop")
SS = _load("state_store")


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(WL, "DATA", str(tmp_path))
    monkeypatch.setattr(WL, "MANIFESTS", str(tmp_path / "manifests"))
    monkeypatch.setattr(WL, "LOCK_PATH", str(tmp_path / ".weekly.lock"))
    return tmp_path


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #
def test_live_lock_blocks_second_run(sandbox):
    assert WL.acquire_lock("week-open") is True
    assert WL.acquire_lock("pregame") is False, \
        "a second concurrent run must exit clean, not race"
    WL.release_lock()
    assert WL.acquire_lock("pregame") is True
    WL.release_lock()


def test_stale_lock_is_overridden(sandbox):
    stale = (datetime.now(timezone.utc)
             - timedelta(hours=WL.LOCK_STALE_HOURS + 1)).isoformat()
    with open(WL.LOCK_PATH, "w") as fh:
        json.dump({"pid": 999999, "job": "week-open", "utc": stale}, fh)
    assert WL.acquire_lock("grade") is True, \
        "a crashed run must not jam the schedule forever"
    WL.release_lock()


def test_corrupt_lock_treated_as_stale(sandbox):
    open(WL.LOCK_PATH, "w").write("not json")
    assert WL.acquire_lock("grade") is True
    WL.release_lock()


# --------------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------------- #
def test_budget_guard_refuses_near_the_cap(monkeypatch):
    monkeypatch.setattr(WL.ingest, "calls_used_this_month", lambda: 940)
    with pytest.raises(WL.LoopError, match="refusing"):
        WL.stage_budget_guard(25)


def test_budget_guard_allows_normal_weeks(monkeypatch):
    monkeypatch.setattr(WL.ingest, "calls_used_this_month", lambda: 241)
    WL.stage_budget_guard(25)   # must not raise


# --------------------------------------------------------------------------- #
# Sentinels — resume, don't restart
# --------------------------------------------------------------------------- #
def test_sentinels_mark_and_detect(sandbox):
    assert not WL.stage_done(2026, 5, "week-open_refit")
    WL.mark_done(2026, 5, "week-open_refit")
    assert WL.stage_done(2026, 5, "week-open_refit")
    assert not WL.stage_done(2026, 6, "week-open_refit"), \
        "sentinels are per-week — next week must run fresh"


# --------------------------------------------------------------------------- #
# Offseason detection
# --------------------------------------------------------------------------- #
def test_offseason_detected_when_first_game_far_away(monkeypatch):
    future = (datetime.now(timezone.utc) + timedelta(days=40)).isoformat()
    games = pd.DataFrame([{
        "id": 1, "season": 2026, "week": 1, "start_date": future,
        "home_points": None, "away_points": None}])
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    season, week, in_season = WL.current_season_week()
    assert in_season is False
    assert week == 1


def test_in_season_when_next_game_this_week(monkeypatch):
    soon = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    games = pd.DataFrame([{
        "id": 1, "season": 2026, "week": 1, "start_date": soon,
        "home_points": None, "away_points": None}])
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    _, week, in_season = WL.current_season_week()
    assert in_season is True and week == 1


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def test_offseason_run_writes_manifest_and_exits_zero(sandbox, monkeypatch):
    far = (datetime.now(timezone.utc) + timedelta(days=60)).isoformat()
    games = pd.DataFrame([{
        "id": 1, "season": 2026, "week": 1, "start_date": far,
        "home_points": None, "away_points": None}])
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    code = WL.run("week-open")
    assert code == 0
    manifests = os.listdir(sandbox / "manifests")
    assert len(manifests) == 1
    m = json.load(open(sandbox / "manifests" / manifests[0]))
    for key in ("job", "season", "week", "config_hash", "git_sha", "seed",
                "calls_before", "calls_after", "data_as_of", "stages"):
        assert key in m, f"manifest missing {key} — numbers must trace"
    assert m["stages"]["offseason_check"]["ok"] is True


# --------------------------------------------------------------------------- #
# STATE_GLOBS points at files that exist in THIS repo
# --------------------------------------------------------------------------- #
def test_state_globs_capture_real_state():
    files = [str(p) for p in SS.state_files()]
    assert any("rating_params.json" in f for f in files), \
        "the ratings parameters are the core state and must be archived"
    assert any("factor_registry.json" in f for f in files)
    assert not any("ml_frame" in f or "weekly_props" in f for f in files), \
        "NFL/prop artifacts must be gone from the glob list"
