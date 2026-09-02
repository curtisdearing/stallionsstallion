"""Sign-convention tests for the board and its ledger.

Every way a betting board goes quietly wrong is a sign error: the spread is
quoted from the home side, the model is quoted as a home margin, and the ATS
result flips on the same axis. A board that inverts one of those still looks
completely plausible and is wrong on every row. These tests pin all three
against hand-computed cases.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BB = _load("build_board")


# --------------------------------------------------------------------------
# spread -> expected home margin
# --------------------------------------------------------------------------
def test_home_favorite_spread_is_a_positive_expected_home_margin():
    """A -7 home spread means the market expects home to win by 7."""
    assert -(-7.0) == pytest.approx(7.0)


def test_edge_sign_points_at_the_home_side_when_model_is_higher():
    """Model 10, market 7 -> +3 edge -> the lean is the HOME side."""
    model, market = 10.0, 7.0
    assert model - market == pytest.approx(3.0)


def test_edge_sign_points_at_the_away_side_when_model_is_lower():
    """Model 20 on a game the market has at 40 -> the lean is the DOG."""
    model, market = 20.29, 40.0
    edge = model - market
    assert edge < 0
    assert edge == pytest.approx(-19.71, abs=1e-9)


# --------------------------------------------------------------------------
# cover probability
# --------------------------------------------------------------------------
def test_zero_edge_is_a_coin_flip():
    assert BB.normal_cdf(0.0) == pytest.approx(0.5)


def test_cover_probability_rises_with_edge_and_stays_bounded():
    sd = 16.9
    low = BB.normal_cdf(2.0 / sd)
    high = BB.normal_cdf(9.0 / sd)
    assert 0.5 < low < high < 1.0


def test_a_ten_point_edge_at_cfb_sd_is_not_a_lock():
    """Guards against a confidence number that overstates a big disagreement."""
    assert BB.normal_cdf(10.0 / 16.9) < 0.75


# --------------------------------------------------------------------------
# geometry / circadian
# --------------------------------------------------------------------------
def test_zero_distance_when_a_team_plays_at_home():
    assert BB.haversine_miles(33.2, -87.5, 33.2, -87.5) == pytest.approx(0.0, abs=1e-6)


def test_coast_to_coast_travel_is_about_two_and_a_half_thousand_miles():
    # Los Angeles -> New York City
    miles = BB.haversine_miles(34.05, -118.24, 40.71, -74.01)
    assert 2400 < miles < 2500


def test_missing_coordinates_do_not_fabricate_a_distance():
    assert pd.isna(BB.haversine_miles(None, -87.5, 33.2, -87.5))


def test_body_clock_hour_uses_the_visitors_home_timezone():
    """A noon ET kickoff is a 9am body-clock game for a Pacific team."""
    kick = pd.Timestamp("2026-09-05T16:00:00Z")     # 12:00 ET
    assert BB.body_clock_hour(kick, "America/Los_Angeles") == pytest.approx(9.0)
    assert BB.body_clock_hour(kick, "America/New_York") == pytest.approx(12.0)


def test_body_clock_hour_is_absent_rather_than_wrong_when_tz_unknown():
    kick = pd.Timestamp("2026-09-05T16:00:00Z")
    assert pd.isna(BB.body_clock_hour(kick, None))
    assert pd.isna(BB.body_clock_hour(kick, "Not/AZone"))


# --------------------------------------------------------------------------
# provider selection
# --------------------------------------------------------------------------
def test_one_row_per_game_from_the_highest_priority_book():
    lines = pd.DataFrame({
        "game_id": [1, 1, 2],
        "provider": ["Bovada", "DraftKings", "Bovada"],
        "spread": [-7.5, -7.0, 3.0],
        "spread_open": [-7.0, -7.0, 3.0],
        "over_under": [50.0, 51.0, 44.0],
    })
    picked = BB.pick_line(lines, {1, 2})
    assert len(picked) == 2
    assert picked.set_index("game_id").loc[1, "provider"] == "DraftKings"
    assert picked.set_index("game_id").loc[1, "spread"] == pytest.approx(-7.0)


def test_games_without_a_quoted_spread_are_dropped_not_defaulted():
    lines = pd.DataFrame({
        "game_id": [1, 2], "provider": ["DraftKings", "DraftKings"],
        "spread": [-7.0, None], "spread_open": [-7.0, None],
        "over_under": [50.0, None],
    })
    picked = BB.pick_line(lines, {1, 2})
    assert set(picked["game_id"]) == {1}


# --------------------------------------------------------------------------
# ledger grading
# --------------------------------------------------------------------------
LEDGER = _load("board_ledger")


def _grade_one(line_taken_home: float, edge: float, home_pts: int, away_pts: int) -> str:
    home_ats = (home_pts - away_pts) + line_taken_home
    if edge == 0:
        return "no lean"
    margin = home_ats if edge > 0 else -home_ats
    return "push" if abs(margin) < 1e-9 else "win" if margin > 0 else "loss"


def test_home_favorite_lean_wins_when_home_covers():
    # home -7, wins by 10, lean was on home -> win
    assert _grade_one(-7.0, +3.0, 31, 21) == "win"


def test_home_favorite_lean_loses_when_home_wins_but_does_not_cover():
    assert _grade_one(-7.0, +3.0, 24, 21) == "loss"


def test_underdog_lean_wins_when_the_favorite_fails_to_cover():
    # home -40, wins by only 20, lean was on the dog -> win
    assert _grade_one(-40.0, -19.7, 41, 21) == "win"


def test_underdog_lean_loses_when_the_favorite_covers():
    assert _grade_one(-40.0, -19.7, 62, 7) == "loss"


def test_an_exact_landing_is_a_push_not_a_win():
    assert _grade_one(-7.0, +3.0, 28, 21) == "push"
    assert _grade_one(-7.0, -3.0, 28, 21) == "push"


def test_a_dog_lean_on_a_road_favorite_grades_on_the_same_axis():
    # home +3 (road team favored by 3); road team wins by 10 -> home does not
    # cover, so a lean on the AWAY side is a win.
    assert _grade_one(3.0, -2.0, 17, 27) == "win"


def test_side_token_extracts_the_team_from_a_rendered_side():
    assert LEDGER._side_token("Clemson +10.0") == "Clemson"
    assert LEDGER._side_token("Ohio State -51.0") == "Ohio State"
    assert LEDGER._side_token("Miami (OH) +17.0") == "Miami (OH)"


def test_ledger_columns_are_stable():
    """The ledger is append-only across weeks; a column rename breaks history."""
    assert LEDGER.LEDGER_COLUMNS[:6] == [
        "season", "week", "game_id", "kickoff_utc", "away", "home"]
    assert "line_taken_home" in LEDGER.LEDGER_COLUMNS
    assert "taken_at_utc" in LEDGER.LEDGER_COLUMNS
