"""Leakage test — poison a future week, assert no earlier number moves.

Re-implemented for stallionsstallion (Workstream A, 2026-07-19). This was a
docstring-only stub: the pattern is the most valuable one inherited from
fablesfable, but the original was written against that project's player-prop
feature builders, which are deliberately out of scope here. Same technique,
new target — the CFBD ingest layer.

The idea: build every through-week-N view of a season, then go back and
corrupt weeks N+1..end with absurd values. Rebuild the through-week-N views.
If any of them changed, information from the future reached the past, and
every backtest number downstream is fiction.

SCOPE — updated 2026-07-19 (Workstream B). This file now covers TWO layers:

  1. the CFBD ingest layer (original, below), and
  2. the ridge ratings solve in `nflvalue/ratings.py` (added at the bottom).

The ingest section was NOT replaced when the ratings section was added, per the
instruction in the Workstream A handoff. Ingest coverage alone was necessary
and not sufficient; ingest + ratings is better and still not sufficient. When
Workstream C (simulation) and D (market) land, extend again — the poison
technique applies to any layer that consumes a time-ordered frame, and the
failure it catches is silent everywhere.

The ratings section poisons the future in three different ways on purpose,
because they fail differently: corrupting existing future rows catches a bad
filter, ADDING future rows catches an aggregate computed over the whole frame,
and corrupting the CURRENT week catches the classic `<=` -vs- `<` off-by-one
that makes a model look brilliant at predicting games it already saw.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nflvalue import ingest
from nflvalue import ratings as R


def _synthetic_season(season=2023, weeks=6, teams=("A", "B", "C", "D")):
    """A small, fully deterministic season we can poison without a network."""
    rows = []
    gid = 1
    for wk in range(1, weeks + 1):
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i + 1]
            rows.append({
                "game_id": gid, "season": season, "week": wk, "period": 1,
                "offense": home, "defense": away,
                "offense_score": 7 * wk, "defense_score": 3 * wk,
                "down": 1, "distance": 10, "yards_to_goal": 65,
                "yards_gained": 4, "play_type": "Rush", "ppa": 0.1 * wk,
            })
            gid += 1
    df = pd.DataFrame(rows)
    df["garbage_time"] = [
        ingest.is_garbage_time(p, o, d)
        for p, o, d in zip(df["period"], df["offense_score"], df["defense_score"])
    ]
    return df


def _through_week(df, week):
    """The only legitimate way to look at a season mid-flight."""
    return (df.loc[df["week"] <= week]
              .sort_values(["week", "game_id"], kind="mergesort")
              .reset_index(drop=True))


def _summary(df):
    """Stand-in for 'every number the model would compute from this view'."""
    if df.empty:
        return {}
    return {
        "n_plays": len(df),
        "mean_ppa": round(float(df["ppa"].mean()), 10),
        "garbage_share": round(float(df["garbage_time"].mean()), 10),
        "max_week": int(df["week"].max()),
    }


def test_future_weeks_cannot_move_earlier_numbers():
    clean = _synthetic_season()
    weeks = sorted(clean["week"].unique())
    before = {w: _summary(_through_week(clean, w)) for w in weeks}

    for cutoff in weeks[:-1]:
        poisoned = clean.copy()
        future = poisoned["week"] > cutoff
        # Absurd values: if any of this bleeds backwards it will be obvious.
        poisoned.loc[future, "ppa"] = 999.0
        poisoned.loc[future, "offense_score"] = 999
        poisoned.loc[future, "defense_score"] = 0

        after = _summary(_through_week(poisoned, cutoff))
        assert after == before[cutoff], (
            f"LEAKAGE at cutoff week {cutoff}: the through-week view changed "
            f"after only FUTURE weeks were poisoned.\n"
            f"  before={before[cutoff]}\n  after ={after}\n"
            "Something is reading past the cutoff."
        )


def test_garbage_time_uses_only_within_play_state():
    """Garbage time must be decided from the play's own score and period,
    never from the final score. Labelling early plays by how the game ended
    is leakage wearing a plausible disguise."""
    df = _synthetic_season()
    first = df.iloc[0]
    direct = ingest.is_garbage_time(first["period"], first["offense_score"],
                                    first["defense_score"])
    assert bool(first["garbage_time"]) == direct

    poisoned = df.copy()
    poisoned.loc[poisoned.index[1:], "offense_score"] = 999
    recomputed = ingest.is_garbage_time(
        poisoned.iloc[0]["period"], poisoned.iloc[0]["offense_score"],
        poisoned.iloc[0]["defense_score"])
    assert recomputed == direct


def test_garbage_time_thresholds_are_per_period():
    # 4th quarter has the tightest threshold (22), 1st the loosest (43).
    assert ingest.is_garbage_time(4, 30, 0) is True
    assert ingest.is_garbage_time(1, 30, 0) is False
    assert ingest.is_garbage_time(1, 50, 0) is True
    # Overtime (period 5+) has no threshold — never garbage by definition.
    assert ingest.is_garbage_time(5, 999, 0) is False


# =========================================================================== #
# WORKSTREAM B — the ridge ratings solve
#
# Same technique, new target. `ratings.solve(games, season, week, ...)` claims
# to see only games strictly before `week`. Everything below tries to prove it
# a liar.
# =========================================================================== #

_RP = R.RatingParams(lambda_team=2.0, lambda_conf=1.0, prior_strength=0.0)


def _synthetic_games(season=2023, weeks=8, teams=("A", "B", "C", "D")):
    """A deterministic multi-week league frame, no cache and no network."""
    strengths = {t: v for t, v in zip(teams, (10.0, 4.0, -3.0, -11.0))}
    rows, gid = [], 5000
    for wk in range(1, weeks + 1):
        for i, h in enumerate(teams):
            a = teams[(i + wk) % len(teams)]
            if a == h:
                continue
            hp = 24.0 + strengths[h] / 2 + 2.5
            ap = 24.0 + strengths[a] / 2
            rows.append({
                "id": gid, "season": season, "week": wk, "home_team": h,
                "away_team": a, "home_points": hp, "away_points": ap,
                "neutral_site": False, "home_fbs": True, "away_fbs": True,
                "home_conf": "TEST", "away_conf": "TEST", "margin": hp - ap,
            })
            gid += 1
    return pd.DataFrame(rows).sort_values(["season", "week", "id"]).reset_index(drop=True)


_FLAT = {t: 0.0 for t in "ABCD"}


def _fp(games, week):
    return R.solve(games, 2023, week, _RP, prior=_FLAT).fingerprint()


def test_ridge_solve_ignores_poisoned_future_weeks():
    clean = _synthetic_games()
    weeks = sorted(clean["week"].unique())
    before = {w: _fp(clean, w) for w in weeks[1:]}

    for cutoff in weeks[1:-1]:
        poisoned = clean.copy()
        future = poisoned["week"] >= cutoff       # >= : week `cutoff` is future too
        poisoned.loc[future, "home_points"] = 999.0
        poisoned.loc[future, "away_points"] = 0.0
        poisoned["margin"] = poisoned["home_points"] - poisoned["away_points"]
        assert _fp(poisoned, cutoff) == before[cutoff], (
            f"LEAKAGE: ratings for week {cutoff} changed after only weeks "
            f">= {cutoff} were poisoned. The solve is reading its own week or later.")


def test_ridge_solve_ignores_games_added_in_the_future():
    """Catches an aggregate (a league mean, a z-score, a team list) computed
    over the whole frame instead of over the pre-cutoff slice."""
    clean = _synthetic_games()
    cutoff = 5
    baseline = _fp(clean, cutoff)

    extra = pd.DataFrame([{
        "id": 90000 + i, "season": 2023, "week": 12, "home_team": "A",
        "away_team": "D", "home_points": 999.0, "away_points": 0.0,
        "neutral_site": False, "home_fbs": True, "away_fbs": True,
        "home_conf": "TEST", "away_conf": "TEST", "margin": 999.0,
    } for i in range(20)])
    assert _fp(pd.concat([clean, extra], ignore_index=True), cutoff) == baseline


def test_ridge_solve_excludes_the_week_it_is_predicting():
    """The `<` vs `<=` off-by-one, isolated.

    Corrupting week 5 must leave the week-5 rating untouched (it has not been
    played yet from that vantage point) and must move the week-6 rating.
    """
    clean = _synthetic_games()
    poisoned = clean.copy()
    wk5 = poisoned["week"] == 5
    poisoned.loc[wk5, "home_points"] = 999.0
    poisoned["margin"] = poisoned["home_points"] - poisoned["away_points"]

    assert _fp(poisoned, 5) == _fp(clean, 5), (
        "LEAKAGE: the week-5 rating moved when week 5's own scores changed — "
        "the solve is using `<=` where it must use `<`")
    assert _fp(poisoned, 6) != _fp(clean, 6), (
        "week 6 did NOT move when week 5 changed; the filter is excluding real "
        "history, which is the opposite failure and equally wrong")


def test_connectivity_ignores_the_future():
    clean = _synthetic_games()
    poisoned = pd.concat([clean, pd.DataFrame([{
        "id": 91000, "season": 2023, "week": 12, "home_team": "A",
        "away_team": "NEWCOMER", "home_points": 40.0, "away_points": 0.0,
        "neutral_site": False, "home_fbs": True, "away_fbs": True,
        "home_conf": "TEST", "away_conf": "TEST", "margin": 40.0,
    }])], ignore_index=True)
    a = R.solve(clean, 2023, 6, _RP, prior=_FLAT)
    b = R.solve(poisoned, 2023, 6, _RP, prior=_FLAT)
    assert a.connectivity.score == b.connectivity.score
    assert "NEWCOMER" not in b.net, \
        "a team whose only game is in the future appeared in an earlier rating"


def test_unit_ratings_ignore_poisoned_future_plays():
    games = _synthetic_games()
    rng = np.random.default_rng(11)
    rows, gid = [], 5000
    for wk in range(1, 9):
        for i, h in enumerate(("A", "B", "C", "D")):
            a = ("A", "B", "C", "D")[(i + wk) % 4]
            if a == h:
                continue
            for off, dfn in ((h, a), (a, h)):
                for j in range(30):
                    rows.append({
                        "game_id": gid, "season": 2023, "week": wk, "period": 2,
                        "offense": off, "defense": dfn, "offense_score": 7,
                        "defense_score": 7, "down": 1 + j % 3, "distance": 10,
                        "yards_to_goal": 55, "yards_gained": float(rng.integers(0, 13)),
                        "play_type": "Rush" if j % 2 else "Pass Reception",
                        "ppa": float(rng.normal(0.2, 1.0)), "garbage_time": False,
                    })
            gid += 1
    plays = pd.DataFrame(rows)
    units = R.play_units(plays)

    cutoff = 5
    before = R.solve_units(games, units, 2023, cutoff, _RP)

    poisoned_plays = plays.copy()
    fut = poisoned_plays["week"] >= cutoff
    poisoned_plays.loc[fut, "yards_gained"] = 99.0
    poisoned_plays.loc[fut, "ppa"] = 99.0
    after = R.solve_units(games, R.play_units(poisoned_plays), 2023, cutoff, _RP)

    for name in R.UNIT_NAMES:
        assert before[name] == after[name], (
            f"LEAKAGE in unit rating {name}: poisoning plays from week "
            f">= {cutoff} moved the week-{cutoff} rating")


def test_walk_forward_error_grades_each_week_against_the_prior_weeks_only():
    """The scorer must not quietly refit on the games it is grading."""
    clean = _synthetic_games()
    R.clear_cache()
    err_a, n_a = R.walk_forward_error(clean, 2023, _RP, first_week=4)

    poisoned = clean.copy()
    last = poisoned["week"] == poisoned["week"].max()
    poisoned.loc[last, "home_points"] = 999.0
    poisoned["margin"] = poisoned["home_points"] - poisoned["away_points"]
    R.clear_cache()
    err_b, n_b = R.walk_forward_error(poisoned, 2023, _RP, first_week=4)

    assert n_a == n_b
    assert err_b > err_a, (
        "poisoning the FINAL week's results did not worsen walk-forward error; "
        "the scorer is fitting on the games it grades")


def test_preseason_prior_cannot_move_when_results_change(monkeypatch):
    """The prior is preseason by definition: talent + returning production.

    If a game result can move it, the "preseason" prior is really an in-season
    rating and every early-week number built on it is contaminated.
    """
    talent = pd.DataFrame([
        {"season": 2023, "team": t, "talent": v, "unranked_recruiting": False}
        for t, v in [("A", 700.0), ("B", 600.0), ("C", 500.0), ("D", 400.0)]])
    returning = pd.DataFrame([
        {"season": 2023, "team": t, "percent_ppa": v}
        for t, v in [("A", 0.7), ("B", 0.6), ("C", 0.5), ("D", 0.4)]])
    monkeypatch.setattr(R.ingest, "load_talent", lambda s: talent)
    monkeypatch.setattr(R.ingest, "load_returning", lambda s: returning)

    conf = {t: "TEST" for t in "ABCD"}
    params = replace(_RP, prior_strength=8.0)
    p1 = R.preseason_prior(2023, list("ABCD"), conf, params)

    clean = _synthetic_games()
    poisoned = clean.copy()
    poisoned["home_points"] = 999.0
    poisoned["margin"] = poisoned["home_points"] - poisoned["away_points"]
    t = R.solve(poisoned, 2023, 6, params)
    for team in "ABCD":
        assert t.prior[team] == pytest.approx(p1[team]), (
            f"the preseason prior for {team} moved with the game results")
