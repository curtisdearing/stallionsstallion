"""Weekly loop (Workstream I Phase 1) — the spec's DoD, tested offline.

Covers: single-writer lock (live blocks, stale overrides), budget guard
refusal, stage sentinels (resume, don't restart), manifest completeness,
offseason detection, the corrected STATE_GLOBS actually matching this
repo's real state files, and the loop's half of the public forecast
contract (`nflvalue/forecast_contract.py`, schema version 2) — one row per
scheduled game, numbers and refusals mutually exclusive, and every row
naming the model family that produced it.
"""

import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WL = _load("weekly_loop")
SS = _load("state_store")

# The contract and the producer policy as the LOOP sees them. Reached through
# `WL` on purpose: a test that imported its own copy could end up asserting
# against a different schema than the code under test writes, which is the
# exact class of drift schema version 2 exists to stop.
FC = WL.FC
PR = WL.PR
# Imported directly from the package, never via `WL.EV`: the bug under test
# is that `weekly_loop.py` calls `EV.benchmark_or_notice` without importing
# `nflvalue.evidence` as `EV` at all, so `WL.EV` must NOT be made to exist by
# this test file -- that would paper over the very NameError being caught.
from nflvalue import evidence as EV  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(WL, "DATA", str(tmp_path))
    monkeypatch.setattr(WL, "MANIFESTS", str(tmp_path / "manifests"))
    monkeypatch.setattr(WL, "LOCK_PATH", str(tmp_path / ".weekly.lock"))
    # `stage_simulate` loads the forecast-SD basis manifest out of its data
    # dir, and nothing may be priced without registered out-of-sample evidence
    # for the week it prices. That rule holds in a sandbox too, so the REAL
    # manifest is copied rather than a convenient one invented: a fixture SD
    # would let a test pass on evidence production does not actually have.
    shutil.copy(os.path.join(_ROOT, "data", "forecast_sd_basis.json"),
                str(tmp_path / "forecast_sd_basis.json"))
    return tmp_path


def _registry():
    """The registered out-of-sample evidence, as production reads it."""
    return FC.BasisRegistry.load(os.path.join(_ROOT, "data"))


def _provenance(season, week, model_version="test-fit"):
    """Who produced these rows — through the loop's own policy function, so
    the model family is decided where production decides it, not here."""
    return WL.provenance_for(season, week, model_version=model_version,
                             source_digest="test-digest")


def _rows(season, week, schedule, simulated=None, refusals=None, **kwargs):
    """Contract rows for an IN-SEASON week, through the loop's own writer.

    `simulated` keeps the simulator's own summary keys (`mc_margin` and
    friends): those are an internal handoff, not column names — the contract
    is what renames them, in one place, on the way out.
    """
    return WL.simulation_artifact(
        season, week, schedule, simulated or {}, refusals or {},
        _provenance(season, week), _registry(), **kwargs)


def _unpriced_rows(season, week, schedule):
    """Contract rows for a week in which nothing was priced.

    Weeks 0-3 belong to the ridge/preseason producer and weeks 4+ to the
    simulation, and `simulation_artifact` will not build a week the
    Monte-Carlo basis does not cover — so an early week's rows come from the
    one shared builder with the loop's own provenance, which is how
    `producers.build_early_week` reaches it too.
    """
    return FC.build_artifact(season, week, schedule, {}, {},
                             _provenance(season, week))


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


# --------------------------------------------------------------------------- #
# Prediction artifact contract — every scheduled game must be explicit
# --------------------------------------------------------------------------- #
def test_weekly_prediction_rows_preserve_full_schedule_and_refusals():
    """The writer's replacement, `simulation_artifact`, keeps the invariant:
    every scheduled game gets a row, and a game that was not priced says so
    in words instead of vanishing.

    An in-season week, because the Monte-Carlo summaries below are what that
    route produces — weeks 0-3 are priced by the ridge producer instead.
    """
    upcoming = pd.DataFrame([
        {"id": 101, "start_date": "2026-10-10T19:00:00Z",
         "home_team": "FBS Home", "away_team": "FBS Away",
         "neutral_site": False, "home_fcs": False, "away_fcs": False},
        {"id": 102, "start_date": "2026-10-10T23:00:00Z",
         "home_team": "FBS Home", "away_team": "FCS Away",
         "neutral_site": False, "home_fcs": False, "away_fcs": True},
        {"id": 103, "start_date": "2026-10-10T23:00:00Z",
         "home_team": "FCS Home", "away_team": "FCS Away",
         "neutral_site": True, "home_fcs": True, "away_fcs": True},
        {"id": 104, "start_date": "2026-10-10T23:00:00Z",
         "home_team": "New FBS", "away_team": "FBS Away",
         "neutral_site": False, "home_fcs": False, "away_fcs": False},
    ])
    simulated = {
        101: {"mc_margin": 4.5, "mc_margin_sd": 17.2, "mc_total": 48.0,
              "p_home_win": 0.61, "param_sd": 2.1},
    }
    refusals = {104: "NO RATING (DisconnectedTeamError): new-to-FBS or unseen team"}

    rows = _rows(2026, 6, upcoming, simulated, refusals)
    by_id = rows.set_index("game_id")

    assert len(rows) == 4, "a schedule row may never vanish because it is unpriced"
    # The contract is closed: exactly these columns, and none of the v1 names
    # whose meaning depended on which writer had run last.
    assert list(rows.columns) == list(FC.COLUMNS)
    assert not set(rows.columns) & set(FC.BANNED_COLUMNS)

    assert by_id.loc[101, "slate"] == "wk6"
    assert by_id.loc[101, "status"] == FC.Status.PRICED.value
    assert by_id.loc[101, "projected_home_margin"] == 4.5
    # v1 published this as `band_pm`, which meant the ~3-point rating band in
    # the other route. The two quantities are separate columns now and the
    # rows carry both, each saying which it is.
    assert by_id.loc[101, "forecast_margin_sd"] == 17.2
    assert by_id.loc[101, "rating_parameter_sd"] == 2.1
    assert by_id.loc[101, "projected_total"] == 48.0
    assert by_id.loc[101, "p_home_win"] == 0.61
    assert by_id.loc[101, "forecast_sd_basis"] == \
        _registry().basis_id("kalman_montecarlo", 6), \
        "a published number must name the evidence its spread comes from"
    assert pd.isna(by_id.loc[101, "refusal_reason"])

    assert by_id.loc[102, "status"] == FC.Status.REFUSED.value
    assert by_id.loc[102, "refusal_reason"] == \
        "FBS-vs-non-FBS: no validated cross-division model"
    assert by_id.loc[103, "status"] == FC.Status.REFUSED.value
    assert by_id.loc[103, "refusal_reason"] == \
        "Non-FBS matchup: no validated lower-division model"
    assert by_id.loc[104, "status"] == FC.Status.REFUSED.value
    assert by_id.loc[104, "refusal_reason"] == refusals[104]
    assert pd.isna(by_id.loc[102, "projected_home_margin"])


def test_weekly_prediction_rows_use_eastern_calendar_at_wk0_boundary():
    """Week 1 is an early-season week, so its rows come from the shared
    contract builder with the loop's own provenance — the same builder the
    simulation route ends in, which is why the calendar labelling can only
    be defined once."""
    upcoming = pd.DataFrame([{
        # This is Aug. 31 in the public Eastern-time calendar, despite being
        # Sept. 1 in UTC. Week 0 must not silently become Week 1.
        "id": 201, "start_date": "2026-09-01T00:30:00Z",
        "home_team": "FBS Home", "away_team": "FBS Away",
        "neutral_site": False, "home_fcs": False, "away_fcs": False,
    }])

    row = _unpriced_rows(2026, 1, upcoming).iloc[0]

    assert row["public_date"] == "2026-08-31"
    assert row["slate"] == "wk0"
    assert row["kickoff_utc"] == "2026-09-01T00:30:00Z", \
        "the instant itself stays UTC; only the public label is Eastern"


# --------------------------------------------------------------------------- #
# Game-window snapshots — sequential pregame runs, Tuesday through Saturday
# --------------------------------------------------------------------------- #
UTC = timezone.utc


def _sched(*starts, week=2, played=()):
    rows = []
    for i, s in enumerate(starts, start=101):
        rows.append({
            "id": i, "season": 2026, "week": week, "start_date": s,
            "home_team": f"Home{i}", "away_team": f"Away{i}",
            "neutral_site": False, "home_fcs": False, "away_fcs": False,
            "home_points": 21 if i in played else None,
            "away_points": 17 if i in played else None})
    return pd.DataFrame(rows)


def test_pregame_job_never_refreshes_so_a_rerun_costs_no_cfbd_calls():
    """Idempotency must be free: re-running pregame may not touch the API."""
    assert "refresh" not in WL.JOB_STAGES["pregame"], \
        "a pregame rerun that re-fetches would burn the 1,000/month budget"
    assert WL.projected_calls("pregame") == 0
    assert WL.projected_calls("week-open") > 0


def test_pregame_does_not_use_the_one_run_per_week_sentinel():
    """The bug this replaces: the first pregame run of a week marked the
    simulate stage done, so no later run that week could ever execute."""
    assert WL.sentinel_applies("week-open", "refit") is True
    assert WL.sentinel_applies("pregame", "simulate") is False


def test_pregame_decision_skips_when_nothing_changed(sandbox, monkeypatch):
    games = _sched("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

    first = WL.pregame_decision(2026, 2, now=now)
    assert first["run"] is True
    WL.SN.write_snapshot(str(sandbox), 2026, 2, "20260901T120000Z",
                         _unpriced_rows(2026, 2, games),
                         source_digest=first["source_digest"],
                         window=first["window"], taken_at=now)

    again = WL.pregame_decision(2026, 2, now=now)
    assert again["run"] is False and "unchanged" in again["reason"]


def test_two_pregame_runs_in_one_week_both_execute(sandbox, monkeypatch):
    games = _sched("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)

    tue = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    d1 = WL.pregame_decision(2026, 2, now=tue)
    WL.SN.write_snapshot(str(sandbox), 2026, 2, "20260901T120000Z",
                         _unpriced_rows(2026, 2, games),
                         source_digest=d1["source_digest"], window=d1["window"],
                         taken_at=tue)

    sat = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)   # noon slate has kicked off
    d2 = WL.pregame_decision(2026, 2, now=sat)
    assert d2["run"] is True and "window" in d2["reason"]
    assert d2["window"] == "2026-09-05T23:00:00Z"


def test_grade_sentinel_reopens_when_more_games_go_final(sandbox):
    early = WL.finals_fingerprint(_sched("a", "b", "c", played=(101,)), week=2)
    later = WL.finals_fingerprint(_sched("a", "b", "c", played=(101, 102)),
                                  week=2)
    assert early != later

    WL.mark_done(2026, 2, "grade_recap", fingerprint=early)
    assert WL.stage_done(2026, 2, "grade_recap", fingerprint=early) is True
    assert WL.stage_done(2026, 2, "grade_recap", fingerprint=later) is False, \
        "grade must resume when more finals land, not stay done for the week"


def test_simulatable_ids_exclude_started_games(sandbox):
    games = _sched("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    now = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)
    assert WL.simulatable_ids(games, now) == [102], \
        "a pregame forecast may never be computed after kickoff"


def test_kicked_off_games_are_frozen_not_resimulated(sandbox):
    games = _sched("2026-10-10T16:00:00Z", "2026-10-10T23:00:00Z", week=6)
    pre = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)
    rows, _ = WL.assemble_week_artifact(
        2026, 6, games,
        _rows(2026, 6, games,
              simulated={101: {"mc_margin": 3.5, "mc_margin_sd": 16.0,
                               "mc_total": 50.0, "p_home_win": 0.6,
                               "param_sd": 1.5},
                         102: {"mc_margin": -7.0, "mc_margin_sd": 16.0,
                               "mc_total": 44.0, "p_home_win": 0.32,
                               "param_sd": 1.5}}),
        pre, data_dir=str(sandbox))
    WL.SN.write_snapshot(str(sandbox), 2026, 6, "20261010T150000Z", rows,
                         source_digest="d1", window="w", taken_at=pre)

    post = datetime(2026, 10, 10, 18, 0, tzinfo=UTC)
    rows2, n_frozen = WL.assemble_week_artifact(
        2026, 6, games,
        _rows(2026, 6, games,
              simulated={102: {"mc_margin": -9.0, "mc_margin_sd": 16.0,
                               "mc_total": 44.0, "p_home_win": 0.28,
                               "param_sd": 1.5}},
              refusals={101: WL.KICKED_OFF_NOTE}),
        post, data_dir=str(sandbox))

    by_id = rows2.set_index("game_id")
    assert n_frozen == 1
    assert float(by_id.loc[101, "projected_home_margin"]) == 3.5, \
        "a started game keeps its final pregame number"
    # "Frozen" is a status now, not a boolean bolted onto the frame, and the
    # row names the snapshot it was republished from — so the claim that the
    # number predates kickoff is checkable rather than asserted.
    assert by_id.loc[101, "status"] == FC.Status.FROZEN.value
    assert by_id.loc[101, "frozen_from_stamp"] == "20261010T150000Z"
    assert float(by_id.loc[102, "projected_home_margin"]) == -9.0, \
        "a game that has not started may still move"
    assert by_id.loc[102, "status"] == FC.Status.PRICED.value
    assert pd.isna(by_id.loc[102, "frozen_from_stamp"])
    assert sorted(rows2["game_id"]) == [101, 102], \
        "freezing must not drop a scheduled game"


def test_already_final_games_stay_in_the_published_artifact(sandbox):
    """Regression for the schedule-truncation defect: the artifact denominator
    is the whole week's schedule, not just its unplayed games. A game with a
    final score must still occupy a row, or grade_week can never score the
    forecast that was published for it."""
    games = _sched("2026-10-08T23:00:00Z", "2026-10-10T16:00:00Z",
                   "2026-10-10T23:00:00Z", week=6, played=(101,))
    pre = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    rows, _ = WL.assemble_week_artifact(
        2026, 6, games,
        _rows(2026, 6, games,
              simulated={i: {"mc_margin": 2.0, "mc_margin_sd": 16.0,
                             "mc_total": 48.0, "p_home_win": 0.55,
                             "param_sd": 1.4}
                         for i in (101, 102, 103)}),
        pre, data_dir=str(sandbox))
    WL.SN.write_snapshot(str(sandbox), 2026, 6, "20261008T120000Z", rows,
                         source_digest="d1", window="w", taken_at=pre)

    # Saturday night: 101 is final, 102 has kicked off, 103 is still pregame.
    late = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)
    rows2, _ = WL.assemble_week_artifact(
        2026, 6, games,
        _rows(2026, 6, games,
              simulated={103: {"mc_margin": 5.0, "mc_margin_sd": 16.0,
                               "mc_total": 48.0, "p_home_win": 0.6,
                               "param_sd": 1.4}},
              refusals={101: WL.KICKED_OFF_NOTE, 102: WL.KICKED_OFF_NOTE}),
        late, data_dir=str(sandbox))

    assert sorted(rows2["game_id"]) == [101, 102, 103], \
        "a completed game may not vanish from the artifact it was published in"
    by_id = rows2.set_index("game_id")
    assert float(by_id.loc[101, "projected_home_margin"]) == 2.0, \
        "the completed game keeps the number it was actually published with"
    assert by_id.loc[101, "status"] == FC.Status.FROZEN.value
    assert by_id.loc[101, "frozen_from_stamp"] == "20261008T120000Z"


def test_pregame_run_records_its_decision_and_skips_cleanly(sandbox, monkeypatch):
    """Driver wiring: a pregame tick with nothing new must skip the simulate
    stage, say why in the manifest, and still exit 0."""
    soon = (datetime.now(timezone.utc) + timedelta(days=2))
    games = _sched(soon.isoformat(), (soon + timedelta(hours=6)).isoformat())
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    monkeypatch.setattr(WL.ingest, "calls_used_this_month", lambda: 25)
    # The health gate is dependency-aware now: (season, week, job). A healthy
    # report is the precondition for this test, which is about the DECISION
    # to skip, not about input health.
    monkeypatch.setattr(WL, "stage_freshness",
                        lambda season, week, job, **kw: {"healthy": True})
    monkeypatch.setattr(WL, "notify", lambda m: {"status": "skipped"})
    simulated = []
    monkeypatch.setattr(WL, "stage_simulate",
                        lambda *a, **k: simulated.append(1) or {"n": 1})

    assert WL.run("pregame", force_week=2) == 0
    assert simulated == [1], "the first pregame run of the week must simulate"

    d = WL.pregame_decision(2026, 2)
    WL.SN.write_snapshot(str(sandbox), 2026, 2, "20260901T120000Z",
                         _unpriced_rows(2026, 2, games),
                         source_digest=d["source_digest"], window=d["window"])
    assert WL.run("pregame", force_week=2) == 0
    assert simulated == [1], "an unchanged rerun must not re-simulate"

    manifests = sorted(os.listdir(sandbox / "manifests"))
    last = json.load(open(sandbox / "manifests" / manifests[-1]))
    assert last["stages"]["simulate"]["skipped"] is True
    assert "unchanged" in last["stages"]["simulate"]["reason"]
    for key in ("run", "reason", "window", "source_digest"):
        assert key in last["decision"], f"manifest must trace the {key} decision"


def test_stage_simulate_denominator_is_the_whole_week_schedule(sandbox, monkeypatch):
    """Exercises the PRODUCTION caller, not a hand-built frame.

    A hand-built frame cannot prove this property — the defect being guarded
    against lives in stage_simulate's own row selection, so the test has to go
    through stage_simulate. Every game here has kicked off, which takes the
    no-live-games path and skips the expensive model fits. An in-season week,
    because that cheap path belongs to the simulation route: weeks 0-3 now
    branch to the ridge producer before the live-game filter is ever reached.
    """
    games = _sched("2026-10-08T23:00:00Z", "2026-10-10T16:00:00Z",
                   "2026-10-10T23:00:00Z", week=6, played=(101, 102))
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    monkeypatch.setattr(WL.R, "load_params", lambda: {})
    monkeypatch.setattr(WL.R, "game_frame", lambda s: games)

    out = WL.stage_simulate(2026, 6, now=datetime(2026, 10, 11, 6, tzinfo=UTC))

    assert out["n_scheduled_games"] == 3
    assert out["n_schedule"] == 3, \
        "every scheduled game must occupy a row, played or not"
    assert out["n"] == 0
    # Read back through the contract, not with a bare `read_csv`: a consumer
    # that parses the public file its own way is how one artifact came to mean
    # different things to different readers.
    written = FC.read_artifact(str(sandbox / "predictions_2026_wk06.csv"))
    assert sorted(written["game_id"]) == [101, 102, 103]
    assert set(written["status"]) == {FC.Status.REFUSED.value}, \
        "nothing was forecastable, so every row is an explicit refusal"
    assert written["refusal_reason"].notna().all(), \
        "a refusal that does not say why is not a refusal"


class _ReachedEVBenchmark(Exception):
    """Marker: execution reached `EV.benchmark_or_notice`. Not a real error."""


def test_stage_simulate_week4plus_calls_the_ev_benchmark_lookup(sandbox,
                                                                monkeypatch):
    """Regression for a missing import.

    `stage_simulate`'s week>=4 path calls `EV.benchmark_or_notice(...)`
    (`nflvalue.evidence`) once at least one game is still genuinely pregame --
    the exact branch `test_stage_simulate_denominator_is_the_whole_week_schedule`
    skips on purpose by using an all-kicked-off schedule. Without
    `from nflvalue import evidence as EV` at module scope in
    `scripts/weekly_loop.py`, `EV` is an unbound name at exactly this call,
    invisible to every test that only exercises the all-kicked-off shortcut.

    Everything between the live-game filter and the EV call is monkeypatched
    to a cheap stand-in: this test is about reachability and the exact call
    made, not about producing a real forecast.
    """
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    games = _sched(future, week=6)
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    monkeypatch.setattr(WL.R, "load_params", lambda: {})
    # A one-row "history" frame gives the rate-pooling loop one iteration to
    # fill `pool` with -- an empty pool makes `pd.concat` raise before the
    # code under test is ever reached, which would be testing the fixture,
    # not the bug.
    monkeypatch.setattr(WL.R, "game_frame",
                        lambda s: pd.DataFrame([{"season": 2026, "week": 5}]))
    monkeypatch.setattr(WL.ingest, "load_drives", lambda s: pd.DataFrame(
        columns=["season", "week", "offense", "defense", "start_period"]))
    monkeypatch.setattr(
        WL.cfb_sim, "classify_drives",
        lambda drives: (pd.DataFrame(columns=["season", "week",
                                              "start_period"]), None))
    monkeypatch.setattr(WL.cfb_ot, "fit_ot_params", lambda drives: object())
    monkeypatch.setattr(WL.EG, "solve_production_table",
                        lambda *a, **k: object())
    monkeypatch.setattr(WL.cfb_sim, "label_week_drives",
                        lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(WL.cfb_sim.DriveRateModel, "fit", lambda df: "rates")
    monkeypatch.setattr(WL.cfb_sim.PaceModel, "fit", lambda drives: "pace")
    monkeypatch.setattr(WL.ingest, "load_teams",
                        lambda s: pd.DataFrame(columns=["team", "conference"]))

    calls = []

    def _spy(root, role):
        calls.append((root, role))
        raise _ReachedEVBenchmark()

    monkeypatch.setattr(EV, "benchmark_or_notice", _spy)

    now = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(_ReachedEVBenchmark):
        WL.stage_simulate(2026, 6, now=now)

    assert calls == [(WL.ROOT, EV.ROLE_WK4PLUS)], (
        "EV.benchmark_or_notice must be called exactly once, with the repo "
        "root and the week>=4 benchmark role")


# --------------------------------------------------------------------------- #
# Artifact identity — which engine priced the row, and under what status
# --------------------------------------------------------------------------- #
#: Every number a public row may carry, taken from the contract itself so a
#: column added there is covered here without anyone remembering to edit a
#: list. v1's `band_pm` is deliberately absent: it is deleted, not renamed.
NUMERIC_FORECAST_COLUMNS = FC.FORECAST_COLUMNS

BANNED_WORDS = ("confidence", "bet", "pick", "edge", "value", "recommend",
                "stake", "unit", "bankroll", "clv", "market", "spread")


def test_weekly_rows_carry_the_policy_engine_and_a_paper_only_status():
    """Weeks 0-3 are the ridge/preseason route's, and the row says so.

    v1 carried the paper-only promise in a prose `forecast_status` string, so
    it held only for as long as whoever wrote the sentence kept it clean. v2
    replaces the prose with a four-value enum and a refusal reason: there is
    no free-text status field left for a recommendation to hide in, and the
    only thing the row still has to declare for itself is which family of
    model produced it.
    """
    rows = _unpriced_rows(
        2026, 2, _sched("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z"))

    assert set(rows["model_family"]) == {"ridge_preseason_prior"}, \
        "an early-season row must name the model family that actually priced it"
    assert set(rows["model_family"]) == {PR.production_family(2)}, \
        "and it must be the family the loop's own routing policy chose"
    assert rows["model_version"].notna().all(), \
        "a family without a fitted version does not identify a producer"

    statuses = set(rows["status"])
    assert statuses <= {s.value for s in FC.Status}, \
        "status is a closed enum; a row may not invent a fifth thing to be"
    assert statuses == {FC.Status.REFUSED.value}, \
        "nothing was priced here, and every row says exactly that"
    for text in set(rows["status"]) | set(rows["refusal_reason"].dropna()):
        for word in BANNED_WORDS:
            assert word not in text.lower(), \
                f"what a row says about itself must stay non-actionable; " \
                f"found {word!r} in {text!r}"


def test_weekly_rows_name_the_configured_engine_after_week_three():
    rows = _rows(2026, 6, _sched("2026-10-10T16:00:00Z", week=6))
    assert set(rows["model_family"]) == \
        {PR.ENGINE_TO_FAMILY[WL.EG.margin_engine()]}, \
        "from week 4 the configured engine prices the week, under its own name"


def test_weekly_rows_fail_loud_on_duplicate_schedule_ids():
    sched = _sched("2026-10-10T16:00:00Z", week=6)
    dup = pd.concat([sched, sched], ignore_index=True)
    # The contract builder owns this check now — it refuses to emit an
    # artifact that is not its own schedule, before anything is written.
    with pytest.raises(FC.ContractError, match="duplicate"):
        _rows(2026, 6, dup)


def test_schedule_id_coverage_allows_a_frozen_game_dropped_from_the_schedule():
    """Regression: a game that kicked off, froze, and later vanished from the
    provider's schedule entirely (not just 'already played') must not
    permanently wedge every future publish. Its frozen row is a legitimate
    extra -- the historical record survives even though the schedule moved on.
    """
    schedule = _sched("2026-10-10T16:00:00Z", week=6)   # only 101 remains
    # There is no `frozen` column any more: a row is frozen when it names the
    # pre-kickoff snapshot it was republished from.
    rows = pd.DataFrame([
        {"game_id": 101, "frozen_from_stamp": None},
        {"game_id": 102, "frozen_from_stamp": "20261010T150000Z"},
    ])
    WL.assert_schedule_id_coverage(rows, schedule, where="test")   # must not raise


def test_schedule_id_coverage_still_rejects_a_non_frozen_extra_row():
    """The frozen-extra allowance must not become a blanket escape hatch: an
    extra row that is NOT frozen is still exactly the bug the check exists
    to catch."""
    schedule = _sched("2026-10-10T16:00:00Z", week=6)
    rows = pd.DataFrame([
        {"game_id": 101, "frozen_from_stamp": None},
        # Invented row: never frozen, never scheduled.
        {"game_id": 999, "frozen_from_stamp": None},
    ])
    with pytest.raises(WL.LoopError, match="not the schedule exactly"):
        WL.assert_schedule_id_coverage(rows, schedule, where="test")


def test_schedule_id_coverage_fails_loud_when_a_game_is_dropped():
    sched = _sched("2026-10-10T16:00:00Z", "2026-10-10T23:00:00Z", week=6)
    rows = _rows(2026, 6, sched)
    WL.assert_schedule_id_coverage(rows, sched, where="unit test")
    with pytest.raises(WL.LoopError):
        WL.assert_schedule_id_coverage(rows.iloc[:1], sched, where="unit test")
    extra = pd.concat([rows, rows.iloc[[0]].assign(game_id=999)],
                      ignore_index=True)
    with pytest.raises(WL.LoopError):
        WL.assert_schedule_id_coverage(extra, sched, where="unit test")


def test_refusal_rows_carry_no_numeric_forecast_fields():
    """Numbers and refusals are mutually exclusive — the contradiction the
    contract exists to make impossible. Both halves are checked: a refused row
    carries no number, AND it does not claim the out-of-sample basis that only
    a published number is entitled to."""
    rows = _rows(2026, 6,
                 _sched("2026-10-10T16:00:00Z", "2026-10-10T23:00:00Z", week=6),
                 refusals={101: "NO RATING: team lacks a validated FBS rating"})

    assert set(rows["status"]) == {FC.Status.REFUSED.value}
    assert not set(rows["status"]) & FC.FORECAST_BEARING
    assert rows["refusal_reason"].notna().all()
    for col in NUMERIC_FORECAST_COLUMNS:
        assert rows[col].isna().all(), \
            f"a refused game must publish no {col}, not a placeholder number"
    assert rows["forecast_sd_basis"].isna().all(), \
        "naming a validated basis on a row that publishes nothing would claim " \
        "evidence for a forecast that was never made"


def test_weekly_rows_expose_no_roster_or_context_field():
    """No personnel or context field may reach a forecast row — and under v2
    that is structural rather than a habit: the contract is a closed column
    set, so a field no reader knows how to interpret cannot ride along at
    all, whatever it is called."""
    rows = _rows(2026, 6, _sched("2026-10-10T16:00:00Z", week=6))
    forbidden = ("roster", "nil", "starter", "player", "depth", "injur",
                 "context", "personnel")
    for col in rows.columns:
        assert not any(f in col.lower() for f in forbidden), \
            f"no personnel/context input may reach a forecast row ({col})"
    assert list(rows.columns) == list(FC.COLUMNS), \
        "the published column set is exactly the contract's — no extras"
    assert not set(rows.columns) & set(FC.BANNED_COLUMNS), \
        "a v1 column name would make a reader guess which meaning it carries"


def test_published_week_csv_carries_engine_and_status(sandbox, monkeypatch):
    """Through the production caller, on the cheap no-live-games path — an
    in-season week, since that is the path's owner now."""
    games = _sched("2026-10-08T23:00:00Z", "2026-10-10T16:00:00Z", week=6,
                   played=(101, 102))
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    monkeypatch.setattr(WL.R, "load_params", lambda: {})
    monkeypatch.setattr(WL.R, "game_frame", lambda s: games)

    WL.stage_simulate(2026, 6, now=datetime(2026, 10, 11, 6, tzinfo=UTC))

    written = FC.read_artifact(str(sandbox / "predictions_2026_wk06.csv"))
    assert set(written["model_family"]) == {PR.production_family(6)}, \
        "the published file must name the family the routing policy chose"
    assert written["status"].notna().all()
    assert set(written["status"]) <= {s.value for s in FC.Status}
    assert set(written["schema_version"]) == {FC.SCHEMA_VERSION}, \
        "a v1 file is rejected, never migrated — the version is on every row"
    assert set(written["game_id"]) == set(games["id"]), \
        "the published artifact must cover the schedule exactly"
