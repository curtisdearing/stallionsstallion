"""What `--week` means to the grade job, pinned.

`weekly_loop.run(job, force_week=N)` — the `--week N` flag — sets the week the
loop is IN.  `week-open` and `pregame` price that week; `grade` recaps and
captures CLV for the week BEFORE it (`week - 1`), because on the Sunday the
grade job fires the loop's current week is already the one about to be
played.  These tests pin that semantic so a later "fix" cannot silently move
which week gets graded.

They also pin the one place the semantic bites.  `current_season_week()`
returns the lowest week with any future game, so a week whose last game is on
a Monday (CFBD week 1 of 2026 ends with SMU @ Florida State on Monday
2026-09-07) is still "current" on the Sunday before, so an unforced `grade`
run that Sunday targets `week - 1` — the week *before* the one that just
finished.

What that costs depends on the week.  In week 1 the target is 0, the stage
returns early and grades nothing.  In any later week the target is a week
that was already recapped the Sunday before, so the run re-grades it, reports
a plausible non-zero n, and the week that actually just finished is not
graded until the following Sunday.  Neither case announces itself, which is
why the recap and CLV stages now always write the week they targeted into the
manifest.

Nothing here changes the semantic.  Whether `--job grade --week N` should
mean "grade week N" is an owner decision recorded in the worklog
(2026-09-02T01:40Z); until it is made, `--week N+1` grades week N.
"""
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WL = _load("weekly_loop", os.path.join("scripts", "weekly_loop.py"))
UTC = timezone.utc


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(WL, "DATA", str(tmp_path))
    monkeypatch.setattr(WL, "MANIFESTS", str(tmp_path / "manifests"))
    monkeypatch.setattr(WL, "LOCK_PATH", str(tmp_path / ".weekly.lock"))
    return tmp_path


def _week1_with_a_monday_game(now):
    """CFBD 2026 week 1 in miniature: a Saturday game already final, and the
    Monday-night game still ahead of `now` (a Sunday)."""
    return pd.DataFrame([
        {"id": 101, "season": 2026, "week": 1,
         "start_date": (now - timedelta(days=1)).isoformat(),
         "home_points": 24, "away_points": 10},
        {"id": 102, "season": 2026, "week": 1,
         "start_date": (now + timedelta(days=1)).isoformat(),
         "home_points": None, "away_points": None},
        {"id": 201, "season": 2026, "week": 2,
         "start_date": (now + timedelta(days=6)).isoformat(),
         "home_points": None, "away_points": None},
    ])


# --------------------------------------------------------------------------- #
# The stage-level contract: grade targets week - 1, and says so
# --------------------------------------------------------------------------- #
def test_recap_targets_the_week_before_the_current_one(monkeypatch):
    from nflvalue import scoreboard
    seen = []

    def _spy(season, week):
        seen.append((season, week))
        return {"n_graded": 3, "model_mae": 1.0, "market_mae": 1.0,
                "ledger_new": 0}

    monkeypatch.setattr(scoreboard, "build_recap", _spy)
    out = WL.stage_recap(2026, 2)
    assert seen == [(2026, 1)], \
        "grade with the loop in week 2 must recap week 1, not week 2"
    assert out["target_week"] == 1 and out["week_recapped"] == 1
    assert out["n"] == 3


def test_a_recap_that_grades_nothing_still_names_its_target_week():
    """The silent case.  With the loop in week 1 the recap target is week 0;
    the manifest must say `target_week: 0` so a reader can see that the run
    graded the wrong week rather than that week 1 had no finals."""
    out = WL.stage_recap(2026, 1)
    assert out["n"] == 0
    assert out["target_week"] == 0, \
        "a zero-row recap must name the week it targeted"
    assert "week_recapped" not in out, \
        "week_recapped is reserved for a week that was actually graded"


def test_a_recap_refused_by_the_scoreboard_names_its_target_week(monkeypatch):
    from nflvalue import scoreboard

    def _refuse(season, week):
        raise scoreboard.ScoreboardError("no graded games")

    monkeypatch.setattr(scoreboard, "build_recap", _refuse)
    out = WL.stage_recap(2026, 2)
    assert out["n"] == 0 and out["target_week"] == 1
    assert "no graded games" in out["note"]


def test_clv_capture_targets_the_week_before_and_names_it(sandbox):
    """No published artifact for the target week -> n=0, but the manifest
    must still say which week the stage looked for."""
    out = WL.stage_clv_capture(2026, 2)
    assert out["n"] == 0
    assert out["target_week"] == 1
    out0 = WL.stage_clv_capture(2026, 1)
    assert out0["n"] == 0 and out0["target_week"] == 0


# --------------------------------------------------------------------------- #
# The trap: a Monday game keeps the week "current" through Sunday
# --------------------------------------------------------------------------- #
def test_sunday_before_a_monday_game_is_still_the_same_week(monkeypatch):
    now = datetime(2026, 9, 6, 13, 0, tzinfo=UTC)          # Sunday 09:00 ET
    monkeypatch.setattr(WL.ingest, "load_games",
                        lambda s: _week1_with_a_monday_game(now))
    season, week, in_season = WL.current_season_week(now=now)
    assert (season, week, in_season) == (2026, 1, True), \
        "the lowest week with a future game is still week 1 on that Sunday"
    # ...so an unforced grade run that Sunday targets week 0 and grades
    # nothing.  Pinned as a fact of the current semantic, not as a defect.
    assert WL.stage_recap(season, week)["target_week"] == 0


def test_after_the_monday_game_the_loop_moves_to_the_next_week(sandbox,
                                                               monkeypatch):
    from nflvalue import scoreboard
    now = datetime(2026, 9, 8, 14, 0, tzinfo=UTC)          # Tuesday
    games = _week1_with_a_monday_game(datetime(2026, 9, 6, 13, 0, tzinfo=UTC))
    games.loc[games["id"] == 102, ["home_points", "away_points"]] = [31, 28]
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: games)
    _, week, in_season = WL.current_season_week(now=now)
    assert (week, in_season) == (2, True)
    # `scoreboard` resolves its own DATA, so the `sandbox` fixture (which
    # patches WL.DATA) does not fence it.  Patched explicitly: a test that
    # reached the real recap would write data/recaps/ and the hypothesis
    # ledger as a side effect of running the suite, once 2026 is played.
    seen = []
    monkeypatch.setattr(scoreboard, "build_recap",
                        lambda season, wk: seen.append((season, wk)) or
                        {"n_graded": 2, "model_mae": 1.0, "market_mae": 1.0,
                         "ledger_new": 0})
    assert WL.stage_recap(2026, week)["target_week"] == 1, \
        "the first unforced grade after the Monday game recaps week 1"
    assert seen == [(2026, 1)]


# --------------------------------------------------------------------------- #
# The driver: --week N is the loop's week; grade recaps N - 1
# --------------------------------------------------------------------------- #
def test_forcing_week_2_on_the_grade_job_recaps_week_1(sandbox, monkeypatch):
    now = datetime(2026, 9, 6, 13, 0, tzinfo=UTC)
    games = _week1_with_a_monday_game(now)
    monkeypatch.setattr(WL.ingest, "load_games", lambda s, **kw: games)
    monkeypatch.setattr(WL.ingest, "calls_used_this_month", lambda: 10)
    monkeypatch.setattr(WL, "current_season_week",
                        lambda now=None: (2026, 1, True))
    monkeypatch.setattr(WL, "notify", lambda manifest: {"status": "skipped"})
    monkeypatch.setattr(WL, "data_as_of", lambda: {})
    monkeypatch.setattr(WL, "stage_refresh", lambda season: {"calls": 0})
    monkeypatch.setattr(WL, "stage_freshness",
                        lambda season, week, job, **kw: {"healthy": True})
    monkeypatch.setattr(WL, "stage_honesty", lambda season: {"ok": True})
    # The recap and CLV stages run for real — patching them out would leave
    # the manifest assertions below testing the stand-in's own arithmetic.
    # Only the scoreboard beneath the recap is patched, because it resolves
    # its own DATA and would otherwise grade against the real repository.
    from nflvalue import scoreboard
    recapped = []
    monkeypatch.setattr(scoreboard, "build_recap",
                        lambda season, wk: recapped.append((season, wk)) or
                        {"n_graded": 5, "model_mae": 1.0, "market_mae": 1.0,
                         "ledger_new": 0})

    assert WL.run("grade", force_week=2) == 0

    assert recapped == [(2026, 1)], \
        "--week sets the loop's week; the stage subtracts one itself"
    paths = sorted(os.listdir(sandbox / "manifests"))
    manifest = json.loads(
        (sandbox / "manifests" / [p for p in paths if p.endswith(".json")][-1])
        .read_text())
    assert manifest["week"] == 2
    assert manifest["stages"]["recap"]["target_week"] == 1
    assert manifest["stages"]["clv_capture"]["target_week"] == 1


# --------------------------------------------------------------------------- #
# --week must mean what it says, or refuse
# --------------------------------------------------------------------------- #
def test_forcing_a_week_below_one_is_refused_not_ignored(sandbox, monkeypatch):
    """`--week 0` used to be swallowed by a truthiness check (`if force_week:`)
    and the loop quietly ran the auto-detected week instead -- an operator who
    asked for one week got another, exit 0, no warning. A week the loop cannot
    run is a refusal, not a default."""
    monkeypatch.setattr(WL, "current_season_week",
                        lambda now=None: (2026, 7, True))
    for bad in (0, -1):
        with pytest.raises(WL.LoopError, match="week"):
            WL.run("grade", force_week=bad)


def test_the_cli_rejects_a_week_below_one(capsys):
    """argparse must refuse before any stage runs, with a message that says
    what is wrong rather than a bare usage dump."""
    for bad in ("0", "-2"):
        with pytest.raises(SystemExit):
            WL.main_args(["--job", "grade", "--week", bad])
        assert "week" in capsys.readouterr().err.lower()


def test_the_cli_still_accepts_a_real_week():
    args = WL.main_args(["--job", "grade", "--week", "2"])
    assert (args.job, args.week) == ("grade", 2)
    assert WL.main_args(["--job", "pregame"]).week is None
