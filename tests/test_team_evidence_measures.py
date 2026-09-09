"""Measure definitions, denominators and sample counts.

The failures these guard against are all the same shape: a number that looks like
a rate but is counted over the wrong denominator, or attributed to the wrong side
of the ball. Every test here names the denominator explicitly.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from nflvalue import ratings
from nflvalue.team_evidence import measures as M
from nflvalue.team_evidence.registry import MeasureRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def play(gid=1, week=1, off="A", dfn="B", play_type="Rush", down=1, distance=10,
         yards_to_goal=50, yards_gained=0, ppa=0.1, season=2026):
    return dict(game_id=gid, season=season, week=week, period=1, offense=off,
                defense=dfn, offense_score=0, defense_score=0, down=down,
                distance=distance, yards_to_goal=yards_to_goal,
                yards_gained=yards_gained, play_type=play_type, ppa=ppa,
                garbage_time=False)


def frame(rows):
    return pd.DataFrame(rows)


def value_of(out, team, measure_id):
    row = out[(out.team == team) & (out.measure_id == measure_id)]
    assert len(row) == 1, f"{measure_id} for {team}: {len(row)} rows"
    return row.iloc[0]


# ------------------------------------------------------------------ success
def test_success_matches_the_ratings_definition():
    """`_success` is a deliberate copy — `ratings.py` is hash-pinned. This asserts
    the copy cannot drift silently."""
    for down in (0, 1, 2, 3, 4, 5):
        for distance in (0, 1, 2, 5, 10, 17):
            for gained in (-3, 0, 1, 4, 7, 10, 25):
                assert M._success(down, distance, gained) == \
                    ratings._success(down, distance, gained), (down, distance, gained)
    for bad in (None, "x", float("nan")):
        assert M._success(bad, 10, 5) == ratings._success(bad, 10, 5)


# ------------------------------------------------------------- denominators
def test_a_success_rate_counts_only_gradeable_plays():
    rows = [play(yards_gained=6) for _ in range(4)]          # 1st & 10, gained 6 -> success
    rows += [play(yards_gained=1) for _ in range(6)]         # failure
    rows += [play(down=None, distance=None, yards_gained=9)]  # ungradeable
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    r = value_of(out, "A", "rush_success_rate_off")
    assert r.den == 10, "the ungradeable play must not enter the denominator"
    assert r.num == 4 and r.value == pytest.approx(0.4)


def test_the_sack_denominator_is_dropbacks_not_all_plays():
    rows = [play(play_type="Pass Reception", yards_gained=8) for _ in range(7)]
    rows += [play(play_type="Sack", yards_gained=-6) for _ in range(2)]
    rows += [play(play_type="Interception", yards_gained=0)]
    rows += [play(play_type="Rush", yards_gained=4) for _ in range(20)]
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    r = value_of(out, "A", "sack_rate_per_dropback_off")
    assert r.den == 10, "7 passes + 2 sacks + 1 interception"
    assert r.num == 2 and r.value == pytest.approx(0.2)


def test_the_sack_rate_registry_entry_says_it_is_not_a_pressure_rate():
    spec = MeasureRegistry.load(DATA).get("sack_rate_per_dropback_off")
    assert "not a pressure rate" in (spec.caveat or "")
    assert "scramble" in (spec.caveat or "").lower()


def test_fumble_recovery_has_a_real_denominator():
    """Retention rate is own recoveries over ALL own fumbles, which is what
    separates fumbling from fumble luck. A rate over 'fumbles lost' alone is 1."""
    rows = [play(play_type="Fumble Recovery (Own)") for _ in range(3)]
    rows += [play(play_type="Fumble Recovery (Opponent)") for _ in range(2)]
    rows += [play(play_type="Fumble Return Touchdown")]
    rows += [play(play_type="Rush", yards_gained=4) for _ in range(10)]
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    assert value_of(out, "A", "fumbles_total_off").num == 6
    assert value_of(out, "A", "fumbles_lost_off").num == 3
    ret = value_of(out, "A", "fumble_retention_rate_off")
    assert ret.den == 6 and ret.value == pytest.approx(0.5)
    # the forced side must see the same six events from the other direction
    assert value_of(out, "B", "fumbles_forced_def").num == 6
    assert value_of(out, "B", "fumble_recovery_share_def").value == pytest.approx(0.5)


def test_interception_play_types_are_matched_exactly_not_by_substring():
    """`Interception` and `Interception Return Touchdown` are different events; a
    substring match would count the second one twice."""
    rows = [play(play_type="Interception"),
            play(play_type="Pass Interception Return"),
            play(play_type="Interception Return Touchdown")]
    rows += [play(play_type="Pass Reception", yards_gained=5) for _ in range(7)]
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    assert value_of(out, "A", "interceptions_thrown").num == 3
    assert value_of(out, "A", "interception_rate_per_dropback").den == 10


def test_short_yardage_is_third_and_fourth_and_two_or_less():
    rows = [play(down=3, distance=2, yards_gained=3),
            play(down=4, distance=1, yards_gained=0),
            play(down=3, distance=3, yards_gained=5),      # not short yardage
            play(down=1, distance=1, yards_gained=5)]      # not third/fourth
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    r = value_of(out, "A", "short_yardage_conversion_rate_off")
    assert r.den == 2 and r.num == 1


def test_explosive_thresholds_are_the_declared_ones():
    rows = [play(play_type="Rush", yards_gained=M.EXPLOSIVE_RUSH_YARDS),
            play(play_type="Rush", yards_gained=M.EXPLOSIVE_RUSH_YARDS - 1),
            play(play_type="Pass Reception", yards_gained=M.EXPLOSIVE_PASS_YARDS),
            play(play_type="Pass Reception", yards_gained=M.EXPLOSIVE_PASS_YARDS - 1)]
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    assert value_of(out, "A", "explosive_play_rate_rush_off").num == 1
    assert value_of(out, "A", "explosive_play_rate_pass_off").num == 1


def test_the_explosive_measure_disclaims_the_broken_assignment_reading():
    """The id and label may not contain the overclaim (the registry tripwire
    enforces that); the caveat must actively disclaim it."""
    spec = MeasureRegistry.load(DATA).get("explosive_play_rate_pass_off")
    assert "threshold" in spec.definition.lower()
    caveat = (spec.caveat or "").lower()
    assert "says nothing about" in caveat
    assert "assignment" in caveat, "the caveat must name the reading it refuses"


# ------------------------------------------------------- created vs allowed
def test_created_and_allowed_are_two_views_of_the_same_plays():
    rows = [play(off="A", dfn="B", play_type="Rush", yards_gained=8) for _ in range(10)]
    rows += [play(off="B", dfn="A", play_type="Rush", yards_gained=1) for _ in range(10)]
    out = M.team_game_measures(frame(rows))
    assert value_of(out, "A", "rush_success_rate_off").value == \
        value_of(out, "B", "rush_success_rate_def").value
    assert value_of(out, "B", "rush_success_rate_off").value == \
        value_of(out, "A", "rush_success_rate_def").value


# --------------------------------------------------------- special teams
def test_kickoff_attribution_is_the_kicking_team():
    """Verified against real CFBD data: on a Kickoff row `offense` is the KICKING
    team (plays_2026 game 401867916, row 24 is a Robert Morris touchdown and row 25
    is a Kickoff with offense = Robert Morris)."""
    rows = [play(play_type="Kickoff", off="A", dfn="B") for _ in range(5)]
    rows += [play(play_type="Kickoff Return Touchdown", off="A", dfn="B")]
    rows += [play(off="B", dfn="A")]
    out = M.team_game_measures(frame(rows))
    # A kicked six times and allowed one return TD
    allowed = value_of(out, "A", "kick_return_td_allowed_per_kickoff")
    assert allowed.den == 6 and allowed.num == 1
    # B faced six kickoffs and scored one return TD
    scored = value_of(out, "B", "kick_return_td_per_kickoff_faced")
    assert scored.den == 6 and scored.num == 1


def test_field_goal_distance_is_yards_to_goal_plus_seventeen():
    rows = [play(play_type="Field Goal Good", yards_to_goal=23),   # 40 yards
            play(play_type="Field Goal Missed", yards_to_goal=38)]  # 55 yards
    rows += [play(off="B", dfn="A")]
    ann = M.annotate(frame(rows))
    got = sorted(ann.loc[ann["_fg_att"], "_fg_distance"].tolist())
    assert got == [40.0, 55.0]


def test_fg_distance_summary_is_robust_not_a_mean():
    """One 58-yard try moves a mean and does not move a median. That is the point."""
    rows = [play(play_type="Field Goal Good", yards_to_goal=y)
            for y in (13, 15, 18, 20)]                     # 30, 32, 35, 37
    rows += [play(play_type="Field Goal Missed", yards_to_goal=41)]  # 58
    rows += [play(off="B", dfn="A")]
    summary = M.robust_fg_summary(M.annotate(frame(rows)), "A")
    assert summary["n_attempts"] == 5
    assert summary["median"] == 35.0
    assert summary["buckets"]["50_plus"] == {"attempts": 1, "made": 0}
    assert summary["buckets"]["30_39"]["attempts"] == 4


def test_special_teams_yardage_is_blocked_not_computed():
    """`yards_gained` on ST rows is not the return distance. The layer refuses to
    compute it rather than shipping a number that looks right."""
    reg = MeasureRegistry.load(DATA)
    for mid in ("coverage_yards_allowed_per_return", "return_yards_per_return"):
        spec = reg.get(mid)
        assert spec.status == "blocked"
        assert "unverified" in spec.blocked_reason.lower()
    emitted = {M.base_measure_id(m) for m in
               M.team_game_measures(frame([play(), play(off="B", dfn="A")]))
               ["measure_id"].unique()}
    assert "coverage_yards_allowed_per_return" not in emitted
    assert "return_yards_per_return" not in emitted


def test_clock_derived_pace_is_blocked():
    reg = MeasureRegistry.load(DATA)
    for mid in ("seconds_per_play", "time_of_possession_share"):
        spec = reg.get(mid)
        assert spec.status == "blocked"
        assert "clock" in spec.blocked_reason.lower()


# --------------------------------------------------------------- structure
def test_every_emitted_measure_is_declared_in_the_registry():
    rows = []
    for pt in ("Rush", "Pass Reception", "Sack", "Interception",
               "Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
               "Field Goal Good", "Field Goal Missed", "Punt", "Kickoff",
               "Punt Return Touchdown", "Kickoff Return Touchdown"):
        rows.append(play(play_type=pt, yards_gained=5))
        rows.append(play(play_type=pt, off="B", dfn="A", yards_gained=5))
    out = M.team_game_measures(frame(rows))
    reg = MeasureRegistry.load(DATA)
    for mid in out["measure_id"].unique():
        reg.get(M.base_measure_id(mid))


def test_a_game_naming_other_than_two_teams_is_skipped_not_half_attributed():
    rows = [play(off="A", dfn="B"), play(off="C", dfn="B")]
    out = M.team_game_measures(frame(rows))
    assert out.empty


def test_the_frame_is_deterministic():
    rows = [play(play_type="Rush", yards_gained=i) for i in range(20)]
    rows += [play(off="B", dfn="A", play_type="Rush", yards_gained=i) for i in range(20)]
    a = M.team_game_measures(frame(rows))
    b = M.team_game_measures(frame(list(reversed(rows))))
    pd.testing.assert_frame_equal(a, b)
