"""Unit coverage for nflvalue.cfb_calendar — the public slate/date labels.

This module has no companion test in the repo history; weekly_loop.py has
depended on it (nflvalue import at module scope) since it landed, so its
absence from version control is a release-blocking gap, not a style nit.
"""

from nflvalue import cfb_calendar


def test_public_game_date_converts_utc_to_eastern():
    assert cfb_calendar.public_game_date("2026-09-05T23:30:00Z") == "2026-09-05"


def test_public_game_date_moves_late_utc_kickoff_to_the_prior_eastern_day():
    # 2026-09-01T00:30:00Z is still 2026-08-31 evening in America/New_York.
    assert cfb_calendar.public_game_date("2026-09-01T00:30:00Z") == "2026-08-31"


def test_public_slate_week1_before_september_is_wk0():
    slate, date = cfb_calendar.public_slate(2026, 1, "2026-08-29T19:00:00Z")
    assert slate == "wk0"
    assert date == "2026-08-29"


def test_public_slate_week1_in_september_is_wk1():
    slate, date = cfb_calendar.public_slate(2026, 1, "2026-09-05T19:00:00Z")
    assert slate == "wk1"


def test_public_slate_later_weeks_use_the_week_number():
    slate, _ = cfb_calendar.public_slate(2026, 6, "2026-10-10T19:00:00Z")
    assert slate == "wk6"
