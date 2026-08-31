"""Leakage and contract tests for the context-layer factor battery.

Written BEFORE the factor definitions, per the pre-registration in
`data/factor_prereg_2026-08-28.json` and the project's test-first rule.

Every factor here must be computable from information that existed before
kickoff. These tests poison the things a factor is forbidden to see — the
game's own result, its own drives, and any market price — and assert the
factor value does not move.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from nflvalue import evidence as EV
from nflvalue import factor_battery as FB
from scripts import factor_battery_context as FBC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _selected_benchmark(monkeypatch):
    """Hand the battery a benchmark, since it now refuses without one.

    `load_preds` selects through the evidence manifest and raises when no
    artifact is selected — which is the correct production behaviour and is
    tested in `tests/test_evidence_and_leakage.py`. These tests are about what
    the loaded frame may CONTAIN, so they supply a frame directly rather than
    turning into a second test of the selection gate. The candidate artifact
    is used when it exists so the columns are the real ones.
    """
    candidate = os.path.join(
        ROOT, "data", "backtest_c_corrected_wf_kalman_20260830.csv")
    source = candidate if os.path.exists(candidate) else os.path.join(
        ROOT, "data", "backtest_c.csv")
    frame = pd.read_csv(source)
    record = EV.ArtifactRecord(
        artifact_id="test_selected", role=EV.ROLE_WK4PLUS,
        status=EV.Status.SELECTED.value,
        path=os.path.relpath(source, ROOT), method="test",
        n_rows=int(len(frame)))
    monkeypatch.setattr(EV, "benchmark_or_notice",
                        lambda root, role=EV.ROLE_WK4PLUS: (frame, record, None))
    monkeypatch.setattr(FBC.EV, "benchmark_or_notice",
                        lambda root, role=EV.ROLE_WK4PLUS: (frame, record, None))
PREREG = os.path.join(ROOT, "data", "factor_prereg_2026-08-28.json")

FORBIDDEN = {"home_points", "away_points", "margin_actual", "result",
             "spread", "spread_open", "spread_close", "over_under",
             "over_under_open", "over_under_close", "home_moneyline",
             "away_moneyline", "clv_pts", "line_close", "provider"}


# ------------------------------------------------------------------ #
# the frozen registration governs the run
# ------------------------------------------------------------------ #
def test_prereg_exists_and_is_frozen_before_results():
    reg = json.load(open(PREREG))
    assert reg["registered_before_any_result_was_computed"] is True
    assert reg["multiplicity"]["q"] == 0.05
    assert reg["multiplicity"]["m"] == len(reg["hypotheses"])


def test_hypothesis_list_matches_the_registration_exactly():
    reg = json.load(open(PREREG))
    registered = [h["id"] for h in reg["hypotheses"]]
    assert FBC.HYPOTHESES == registered, (
        "the run must test exactly the pre-registered list — no additions "
        "after seeing a result, no quiet drops")
    assert FBC.HYPOTHESES == reg["multiplicity"]["corrected_across"]


def test_n_gates_are_the_registered_project_defaults():
    assert FBC.MIN_TREATED == 40
    assert FBC.MIN_CONTINUOUS_N == 800
    assert FBC.BH_Q == 0.05


def test_standing_exclusions_are_refused_before_any_evaluation():
    entries = FB.load_registry(os.path.join(ROOT, "data", "factor_registry.json"))
    for fid in ("raw_time_of_possession", "penalty_counts",
                "raw_turnover_margin", "raw_red_zone_td_pct",
                "raw_pressure_rate"):
        with pytest.raises(FB.ExcludedFactorError):
            FB.assert_not_excluded(entries, fid)


# ------------------------------------------------------------------ #
# rest / short week
# ------------------------------------------------------------------ #
@pytest.fixture
def sched():
    return pd.DataFrame([
        {"id": 1, "season": 2023, "week": 1, "home_team": "A", "away_team": "B",
         "start_date": "2023-09-02T18:00:00.000Z", "home_points": 30.0,
         "away_points": 20.0},
        {"id": 2, "season": 2023, "week": 2, "home_team": "C", "away_team": "A",
         "start_date": "2023-09-07T18:00:00.000Z", "home_points": 10.0,
         "away_points": 40.0},                       # A on 5 days
        {"id": 3, "season": 2023, "week": 2, "home_team": "B", "away_team": "D",
         "start_date": "2023-09-09T18:00:00.000Z", "home_points": 21.0,
         "away_points": 17.0},                       # B on 7 days
        {"id": 4, "season": 2023, "week": 3, "home_team": "A", "away_team": "B",
         "start_date": "2023-09-16T18:00:00.000Z", "home_points": None,
         "away_points": None},
    ])


def test_rest_days_read_only_strictly_earlier_kickoffs(sched):
    rest = FBC.rest_days(sched)
    # game 4: A last played 2023-09-07, B last played 2023-09-09
    row = rest[rest["id"] == 4].iloc[0]
    assert row["home_rest"] == pytest.approx(9.0)
    assert row["away_rest"] == pytest.approx(7.0)
    # game 1 is each team's first: no prior kickoff, so no value
    assert pd.isna(rest[rest["id"] == 1].iloc[0]["home_rest"])


def test_rest_days_ignore_the_games_own_result(sched):
    a = FBC.rest_days(sched)
    poisoned = sched.copy()
    poisoned["home_points"] = -999.0
    poisoned["away_points"] = 999.0
    b = FBC.rest_days(poisoned)
    pd.testing.assert_frame_equal(a, b)


def test_short_week_excludes_games_where_both_sides_are_short(sched):
    both = sched.copy()
    both.loc[both["id"] == 4, "start_date"] = "2023-09-12T18:00:00.000Z"
    ev = FBC.short_week_events(FBC.rest_days(both))
    assert 4 not in set(ev["id"]), (
        "a game with no rest differential is not a treated observation")


def test_short_week_marks_the_short_side_only(sched):
    short = sched.copy()
    short.loc[short["id"] == 4, "start_date"] = "2023-09-15T18:00:00.000Z"
    ev = FBC.short_week_events(FBC.rest_days(short))
    row = ev[ev["id"] == 4]
    assert len(row) == 1 and row.iloc[0]["side"] == "away"


# ------------------------------------------------------------------ #
# special teams — walk-forward only
# ------------------------------------------------------------------ #
@pytest.fixture
def drives():
    rows = []
    for wk, gid in ((1, 101), (2, 102), (3, 103)):
        for n, (off, deff, res, ytg) in enumerate([
                ("A", "B", "PUNT", 75), ("B", "A", "PUNT", 60),
                ("A", "B", "TD", 40), ("B", "A", "PUNT", 80)], start=1):
            rows.append({"game_id": gid, "season": 2023, "week": wk,
                         "drive_number": n, "offense": off, "defense": deff,
                         "drive_result": res, "start_yards_to_goal": ytg,
                         "start_period": 1, "garbage_time": False})
    return pd.DataFrame(rows)


def test_special_teams_never_reads_the_game_being_described(drives):
    games = pd.DataFrame([{"id": 103, "season": 2023, "week": 3,
                           "home_team": "A", "away_team": "B"}])
    a = FBC.special_teams_diffs(drives, games)
    poisoned = drives.copy()
    mask = poisoned["game_id"] == 103
    poisoned.loc[mask, "start_yards_to_goal"] = 1
    b = FBC.special_teams_diffs(poisoned, games)
    pd.testing.assert_frame_equal(a, b)


def test_special_teams_uses_only_strictly_earlier_weeks(drives):
    games = pd.DataFrame([{"id": 103, "season": 2023, "week": 3,
                           "home_team": "A", "away_team": "B"}])
    used = FBC.special_teams_rows_used(drives, games, game_id=103)
    assert used["weeks"] and all(w < 3 for w in used["weeks"])
    assert 103 not in used["game_ids"]


def test_special_teams_drops_garbage_time_and_overtime(drives):
    d = drives.copy()
    d.loc[d["week"] == 1, "garbage_time"] = True
    clean = FBC.usable_drives(d)
    assert set(clean["week"]) == {2, 3}
    d2 = drives.copy()
    d2.loc[d2["week"] == 2, "start_period"] = 5
    assert set(FBC.usable_drives(d2)["week"]) == {1, 3}


# ------------------------------------------------------------------ #
# returning production / coaching
# ------------------------------------------------------------------ #
def test_returning_diff_is_constant_within_a_season():
    ret = pd.DataFrame([{"season": 2023, "team": "A", "percent_ppa": 0.6},
                        {"season": 2023, "team": "B", "percent_ppa": 0.2}])
    games = pd.DataFrame([
        {"id": 1, "season": 2023, "week": 4, "home_team": "A", "away_team": "B"},
        {"id": 2, "season": 2023, "week": 11, "home_team": "A", "away_team": "B"}])
    d = FBC.returning_diffs(ret, games).set_index("id")["diff"]
    assert d.loc[1] == pytest.approx(d.loc[2]) == pytest.approx(0.4)


def test_coach_year2_requires_the_prior_season_to_exist():
    coaches = pd.DataFrame([
        {"season": 2023, "team": "A", "coach": "X", "first_season": False},
        {"season": 2022, "team": "A", "coach": "X", "first_season": True},
        {"season": 2023, "team": "B", "coach": "Y", "first_season": False},
    ])
    games = pd.DataFrame([{"id": 1, "season": 2023, "week": 5,
                           "home_team": "A", "away_team": "B"}])
    ev = FBC.coach_year2_events(coaches, games)
    assert list(ev["side"]) == ["home"], "only A qualifies; B has no 2022 row"


def test_coach_year2_rejects_a_different_coach_in_year_two():
    coaches = pd.DataFrame([
        {"season": 2023, "team": "A", "coach": "Z", "first_season": False},
        {"season": 2022, "team": "A", "coach": "X", "first_season": True},
    ])
    games = pd.DataFrame([{"id": 1, "season": 2023, "week": 5,
                           "home_team": "A", "away_team": "B"}])
    assert FBC.coach_year2_events(coaches, games).empty


# ------------------------------------------------------------------ #
# nothing the battery touches may carry a price or an outcome
# ------------------------------------------------------------------ #
def test_no_frame_handed_to_the_evaluator_carries_a_price_or_outcome():
    for frame in FBC.frames_handed_to_evaluator():
        bad = set(frame.columns) & FORBIDDEN
        assert not bad, f"a factor frame carries forbidden columns {sorted(bad)}"


def test_outcome_is_the_walk_forward_residual_and_nothing_else():
    preds = FBC.load_preds()
    assert set(preds.columns) == {"id", "season", "week", "home", "away",
                                  "pred_b", "margin"}
    assert not (set(preds.columns) & {"spread", "mc_margin", "p_home_win"})
