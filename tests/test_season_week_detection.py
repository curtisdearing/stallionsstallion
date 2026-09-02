"""`current_season_week()` must always answer, or the loop dies before it starts.

This runs on the first line of `weekly_loop.run()`, outside the per-stage
try/except. If it raises there is no manifest, no notification and no exit
code the runner can classify -- just a traceback in the launchd log.

The shape that used to raise `int(NaN)` is NOT "the night game's final has not
landed yet". CFBD leaves a handful of games permanently unscored every season
-- lower-division fixtures and weather cancellations. In this repository's own
caches: 4 unscored in 2023 (weeks 9 and 11), 2 in 2024 (weeks 5 and 10), 2 in
2025 (weeks 9 and 12). So `played.all()` is never true for a real season, the
offseason branch that depended on it was unreachable, and once the last game
of the season had kicked off the old code selected an empty future frame and
raised -- every January.

Fixing only the raise is not enough, and this is the part worth keeping in
mind: answering "in season, week 5" in January is worse than crashing for the
`pregame` job, which is deliberately excluded from the per-week sentinels and
whose launchd job fires year round. It would re-simulate a week from October
on every fire. Scoredness cannot separate "the slate just finished" from "the
season ended months ago"; recency can, which is what `RESULTS_SETTLING` does.
"""
import importlib.util
import os
from datetime import datetime, timezone

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


def _use(monkeypatch, frame):
    monkeypatch.setattr(WL.ingest, "load_games", lambda s: frame)


def _season(weeks=15, *, unscored_weeks=(), season=2026, start="2026-08-29"):
    """A season-shaped frame: `weeks` weekly slates of two games each.

    Games in `unscored_weeks` never receive a score, which is what CFBD
    actually does with cancelled and lower-division fixtures -- the detail the
    old offseason branch assumed away.
    """
    first = pd.Timestamp(start, tz="UTC")
    rows, gid = [], 1
    for wk in range(1, weeks + 1):
        kickoff = first + pd.Timedelta(days=7 * (wk - 1))
        for slot in (16, 23):
            scored = wk not in unscored_weeks
            rows.append({
                "id": gid, "season": season, "week": wk,
                "start_date": (kickoff + pd.Timedelta(hours=slot)).isoformat(),
                "home_points": 21 if scored else None,
                "away_points": 17 if scored else None})
            gid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The regression, on frames that can actually occur
# --------------------------------------------------------------------------- #
def test_january_after_a_season_with_never_scored_games(monkeypatch):
    """The real crash. Season over, two October games CFBD never scored.

    The old code raised `ValueError: cannot convert float NaN to integer`
    here, because `played.all()` was False so the offseason branch was skipped
    and the future selection was empty.
    """
    _use(monkeypatch, _season(unscored_weeks=(5, 10)))
    season, week, in_season = WL.current_season_week(
        now=datetime(2027, 1, 20, tzinfo=UTC))
    assert (season, week, in_season) == (2026, 15, False), \
        "months after the last kickoff this is the offseason, whatever is unscored"


def test_the_offseason_exit_is_reachable_with_unscored_games(monkeypatch):
    """Guards the root cause rather than one date: no real season has every
    game scored, so an offseason answer must not depend on that."""
    for unscored in ((), (5,), (5, 10), (1, 15)):
        _use(monkeypatch, _season(unscored_weeks=unscored))
        _, _, in_season = WL.current_season_week(
            now=datetime(2027, 2, 1, tzinfo=UTC))
        assert in_season is False, f"unscored weeks {unscored} broke the offseason exit"


def test_results_still_landing_right_after_the_last_slate(monkeypatch):
    """The genuine 'still grading' case: the final slate kicked off hours ago
    and nothing is scored yet. Here `in_season` must stay True."""
    games = _season(weeks=3)
    games.loc[games["week"] == 3, ["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    last_kickoff = pd.Timestamp(games["start_date"].max())
    _, week, in_season = WL.current_season_week(
        now=(last_kickoff + pd.Timedelta(hours=6)).to_pydatetime())
    assert (week, in_season) == (3, True), \
        "six hours after kickoff the results are still landing"


def test_the_settling_window_closes(monkeypatch):
    """A week after the last kickoff, whatever is still unscored is not
    'landing' any more -- otherwise January looks like the season."""
    games = _season(weeks=3)
    games.loc[games["week"] == 3, ["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    last_kickoff = pd.Timestamp(games["start_date"].max())
    inside = (last_kickoff + WL.RESULTS_SETTLING - pd.Timedelta(hours=1))
    outside = (last_kickoff + WL.RESULTS_SETTLING + pd.Timedelta(hours=1))
    assert WL.current_season_week(now=inside.to_pydatetime())[2] is True
    assert WL.current_season_week(now=outside.to_pydatetime())[2] is False


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_an_unparseable_date_column_does_not_run_the_loop(monkeypatch):
    """With no kickoff times no game can be frozen at its own kickoff, so
    `freeze_kicked_off` would freeze nothing and a run would publish fresh
    numbers for games that may already have started. Refusing to run is the
    safe answer, not merely the non-crashing one."""
    games = _season(weeks=2)
    games["start_date"] = "not-a-date"
    _use(monkeypatch, games)
    season, week, in_season = WL.current_season_week(
        now=datetime(2026, 9, 2, tzinfo=UTC))
    assert in_season is False, "a schedule with no usable kickoff times must not run"
    assert isinstance(week, int) and week >= 1


def test_a_null_week_number_is_skipped_not_converted(monkeypatch):
    """Week 1 has been played and the next slate carries no week number, so
    the loop targets the next slate that names one. Skipping an unnameable
    week is the honest answer -- `int(NaN)` was the alternative -- and the
    input health gate checks week coverage separately."""
    games = _season(weeks=3)
    games.loc[games["week"] == 2, "week"] = float("nan")
    games.loc[games.index[2:], ["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    _, week, in_season = WL.current_season_week(
        now=datetime(2026, 9, 1, tzinfo=UTC))
    assert (week, in_season) == (3, True)


def test_future_games_with_no_week_at_all_do_not_become_week_one(monkeypatch):
    """Guessing week 1 would price the wrong week. If no upcoming game names
    its week, the loop has nothing it can honestly target."""
    games = _season(weeks=2)
    games["week"] = float("nan")
    games[["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    _, _, in_season = WL.current_season_week(
        now=datetime(2026, 8, 25, tzinfo=UTC))
    assert in_season is False


# --------------------------------------------------------------------------- #
# Shapes that already worked, pinned so the repair cannot move them
# --------------------------------------------------------------------------- #
def test_midseason_targets_the_lowest_future_week(monkeypatch):
    games = _season(weeks=6)
    games.loc[games["week"] >= 4, ["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    kick4 = pd.Timestamp(games.loc[games["week"] == 4, "start_date"].min())
    _, week, in_season = WL.current_season_week(
        now=(kick4 - pd.Timedelta(days=2)).to_pydatetime())
    assert (week, in_season) == (4, True)


def test_preseason_far_out_is_not_in_season(monkeypatch):
    games = _season(weeks=2)
    games[["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    _, week, in_season = WL.current_season_week(
        now=datetime(2026, 7, 1, tzinfo=UTC))
    assert (week, in_season) == (1, False), "more than 7 days out is offseason"


def test_in_season_once_the_first_game_is_close(monkeypatch):
    games = _season(weeks=2)
    games[["home_points", "away_points"]] = None
    _use(monkeypatch, games)
    _, week, in_season = WL.current_season_week(
        now=datetime(2026, 8, 26, tzinfo=UTC))
    assert (week, in_season) == (1, True)


def test_an_empty_schedule_is_the_offseason_not_a_crash(monkeypatch):
    _use(monkeypatch, pd.DataFrame())
    _, week, in_season = WL.current_season_week(
        now=datetime(2026, 5, 1, tzinfo=UTC))
    assert (week, in_season) == (1, False)


def test_a_frame_missing_the_columns_we_read_does_not_raise(monkeypatch):
    """`start_date` was column-guarded; `week` and `home_points` were not."""
    for frame in (pd.DataFrame([{"id": 1, "start_date": "2026-09-05T16:00:00Z"}]),
                  pd.DataFrame([{"id": 1, "start_date": "2026-09-05T16:00:00Z",
                                 "week": 1}])):
        _use(monkeypatch, frame)
        season, week, in_season = WL.current_season_week(
            now=datetime(2026, 9, 1, tzinfo=UTC))
        assert (season, week, in_season) == (2026, 1, False)


@pytest.mark.parametrize("month,expected", [(1, 2025), (2, 2026), (9, 2026)])
def test_the_season_year_rolls_in_february(monkeypatch, month, expected):
    """January belongs to the previous season (bowls); February starts the next."""
    _use(monkeypatch, pd.DataFrame())
    season, _, _ = WL.current_season_week(
        now=datetime(2026, month, 15, tzinfo=UTC))
    assert season == expected


# --------------------------------------------------------------------------- #
# Against the real caches -- the frames that actually caused this
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("year", [2023, 2024, 2025])
def test_a_real_completed_season_reads_as_the_offseason_in_january(monkeypatch,
                                                                   year):
    cache = os.path.join(_ROOT, "data", "cfbd", f"games_{year}.parquet")
    if not os.path.exists(cache):
        pytest.skip(f"no games_{year} cache in this checkout")
    games = pd.read_parquet(cache)
    assert not games["home_points"].notna().all(), (
        f"{year} is expected to contain never-scored games -- if CFBD has "
        "since filled them in, the premise of this test needs rechecking")
    _use(monkeypatch, games)
    season, week, in_season = WL.current_season_week(
        now=datetime(year + 1, 1, 20, tzinfo=UTC))
    assert in_season is False, \
        "a finished season with unscored games must not read as in season"
    assert season == year and week >= 1
