"""Wiring tests for the shipped Kalman margin engine (bakeoff G winner).

Covers the four load-bearing claims of the wiring:
  1. Walk-forward: poisoning a FUTURE week cannot change the table.
  2. Adapter identity: predict_margin == filtered net diff + frozen HFA, and
     side_edges / decompose_margin reproduce it exactly.
  3. Fail-closed veto: unseen teams and FCS refuse to price (the recorded
     deviation from the bakeoff harness — conservative only).
  4. Dispatch: config selects the engine; unknown names fail loud; ridge
     stays fully wired as the drift baseline.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from nflvalue import engine as EG  # noqa: E402
from nflvalue import explain  # noqa: E402
from nflvalue import kalman_engine as K  # noqa: E402
from nflvalue import ratings as R  # noqa: E402
from nflvalue.cfb_sim import side_edges  # noqa: E402


def _games() -> pd.DataFrame:
    """Tiny deterministic FBS round-robin over two seasons + week 3 of s2."""
    rows = []
    gid = 0
    teams = ["AAA", "BBB", "CCC", "DDD"]
    margins = {("AAA", "BBB"): 14, ("CCC", "DDD"): -3, ("AAA", "CCC"): 10,
               ("BBB", "DDD"): 7, ("AAA", "DDD"): 21, ("BBB", "CCC"): -6}
    for season in (2023, 2024):
        wk = 1
        for (h, a), m in margins.items():
            gid += 1
            rows.append({
                "id": gid, "season": season, "week": wk,
                "home_team": h, "away_team": a,
                "home_conf": "X" if h in ("AAA", "BBB") else "Y",
                "away_conf": "X" if a in ("AAA", "BBB") else "Y",
                "home_points": 24 + m, "away_points": 24,
                "neutral_site": False, "home_fbs": True, "away_fbs": True,
                "margin": float(m),
            })
            if gid % 2 == 0:
                wk += 1
    assert set(teams) == set(r["home_team"] for r in rows) | set(
        r["away_team"] for r in rows)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1 — walk-forward / leakage
# --------------------------------------------------------------------------- #
def test_future_week_poison_cannot_change_the_table():
    games = _games()
    tbl = K.solve_kalman(games, 2024, 3)
    poisoned = games.copy()
    future = (poisoned["season"] == 2024) & (poisoned["week"] >= 3)
    assert future.any()  # the fixture has as-of-week games to poison
    poisoned.loc[future, "home_points"] = 99   # absurd future results
    poisoned.loc[future, "away_points"] = 0
    poisoned.loc[future, "margin"] = 99.0
    extra = games.iloc[[0]].copy()             # ...and a brand-new future game
    extra["id"], extra["season"], extra["week"] = 999, 2024, 7
    extra["home_points"], extra["away_points"], extra["margin"] = 99, 0, 99.0
    poisoned = pd.concat([poisoned, extra], ignore_index=True)
    tbl_p = K.solve_kalman(poisoned, 2024, 3)
    assert tbl.fingerprint() == tbl_p.fingerprint()


def test_earlier_weeks_do_change_the_table():
    games = _games()
    f1 = K.solve_kalman(games, 2024, 2).fingerprint()
    f2 = K.solve_kalman(games, 2024, 3).fingerprint()
    assert f1 != f2


# --------------------------------------------------------------------------- #
# 2 — adapter identity
# --------------------------------------------------------------------------- #
def test_predict_margin_is_net_diff_plus_frozen_hfa():
    games = _games()
    tbl = K.solve_kalman(games, 2024, 3)
    m = tbl.predict_margin("AAA", "CCC", neutral=False)
    expect = tbl.net["AAA"] - tbl.net["CCC"] + K.FROZEN.hfa
    assert m == pytest.approx(expect, abs=5e-6)  # ±net/2 rounding half-ulps
    assert tbl.predict_margin("AAA", "CCC", neutral=True) == pytest.approx(
        tbl.net["AAA"] - tbl.net["CCC"], abs=5e-6)


def test_state_matches_the_measured_bakeoff_filter():
    """The engine's state IS bakeoff.KalmanModel fed the same ordered games."""
    from nflvalue.bakeoff import KalmanModel
    games = _games()
    ref = KalmanModel(K.FROZEN)
    hist = games[(games["season"] < 2024)
                 | ((games["season"] == 2024) & (games["week"] < 3))]
    for g in hist.sort_values(["season", "week", "id"],
                              kind="mergesort").itertuples():
        ref.update(g)
    # Table ages state to the as-of week (bands only); means must be equal.
    tbl = K.solve_kalman(games, 2024, 3)
    for t, mu in ref.mu.items():
        assert tbl.net[t] == pytest.approx(mu, abs=1e-6)


def test_side_edges_reproduce_predict_margin():
    games = _games()
    tbl = K.solve_kalman(games, 2024, 3)
    e_h, e_a = side_edges(tbl, "AAA", "BBB", False)
    assert (e_h - e_a) == pytest.approx(
        tbl.predict_margin("AAA", "BBB", False), abs=1e-9)


def test_decomposition_is_exact_and_has_no_conference_row():
    games = _games()
    tbl = K.solve_kalman(games, 2024, 3)
    rows = explain.decompose_margin(tbl, "AAA", "BBB", neutral=False)
    keys = [r["key"] for r in rows]
    assert "conference" not in keys
    assert keys == ["team_rating", "hfa"]
    total = sum(r["points"] for r in rows)
    assert total == pytest.approx(
        tbl.predict_margin("AAA", "BBB", False), abs=1e-6)
    # with a sim margin the rows sum to the SIM margin instead
    rows_mc = explain.decompose_margin(tbl, "AAA", "BBB", neutral=False,
                                       mc_margin=3.25)
    assert sum(r["points"] for r in rows_mc) == pytest.approx(3.25, abs=1e-6)


def test_to_frame_schema_matches_ridge():
    games = _games()
    kf = K.solve_kalman(games, 2024, 3).to_frame()
    rf = R.solve(games, 2024, 3, R.RatingParams()).to_frame()
    assert list(kf.columns) == list(rf.columns)


def test_fingerprint_is_stable_and_content_sensitive():
    games = _games()
    a = K.solve_kalman(games, 2024, 3).fingerprint()
    b = K.solve_kalman(games.sample(frac=1.0, random_state=0), 2024,
                       3).fingerprint()
    assert a == b  # row order of the input frame must not matter
    changed = games.copy()
    changed.loc[changed.index[0], "home_points"] += 3
    changed.loc[changed.index[0], "margin"] += 3.0
    assert K.solve_kalman(changed, 2024, 3).fingerprint() != a


# --------------------------------------------------------------------------- #
# 3 — fail-closed veto
# --------------------------------------------------------------------------- #
def test_unseen_team_is_vetoed_not_guessed():
    games = _games()
    tbl = K.solve_kalman(games, 2024, 3)
    with pytest.raises(R.DisconnectedTeamError):
        tbl.predict_margin("AAA", "NEVER_PLAYED", False)


def test_fcs_pseudo_team_is_vetoed():
    games = _games()
    fcs = games.iloc[[0]].copy()
    fcs["id"], fcs["away_team"], fcs["away_fbs"] = 500, "Little Sisters", False
    tbl = K.solve_kalman(pd.concat([games, fcs], ignore_index=True), 2024, 3)
    assert tbl.veto[R.FCS_TEAM]
    with pytest.raises(R.DisconnectedTeamError):
        tbl.require_usable(R.FCS_TEAM)


def test_band_is_positive_and_widens_with_absence():
    games = _games()
    tbl = K.solve_kalman(games, 2024, 3)
    assert all(v > 0 for v in tbl.net_band.values())
    # a season later with no games in between, the same team's band is wider
    later = K.solve_kalman(games, 2025, 1)
    assert later.net_band["AAA"] > tbl.net_band["AAA"]


# --------------------------------------------------------------------------- #
# 4 — dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_returns_each_engine_and_fails_loud():
    games = _games()
    k = EG.solve_table(games, 2024, 3, engine="kalman")
    r = EG.solve_table(games, 2024, 3, R.RatingParams(), engine="ridge")
    assert isinstance(k, K.KalmanTable)
    assert isinstance(r, R.RatingsTable)
    with pytest.raises(ValueError):
        EG.solve_table(games, 2024, 3, engine="elo")


def test_config_selects_kalman_by_default():
    # config.json in this repo pins margin_engine=kalman (the registered
    # decision); the dispatcher must read it.
    assert EG.margin_engine() in ("kalman", "ridge")
    games = _games()
    tbl = EG.solve_table(games, 2024, 3)
    if EG.margin_engine() == "kalman":
        assert isinstance(tbl, K.KalmanTable)


def test_cached_solve_returns_identical_fingerprint():
    games = _games()
    K.clear_cache()
    a = K.solve_kalman_cached(games, 2024, 3)
    b = K.solve_kalman_cached(games, 2024, 3)
    assert a is b
    assert a.fingerprint() == K.solve_kalman(games, 2024, 3).fingerprint()


# --------------------------------------------------------------------------- #
# 5 — early-season production policy
# --------------------------------------------------------------------------- #
def test_production_engine_is_ridge_through_week_three():
    for week in (0, 1, 2, 3):
        assert EG.production_margin_engine(week) == "ridge", \
            "weeks 0-3 are priced by ridge + preseason prior, by policy"


def test_production_engine_defers_to_the_dispatcher_from_week_four():
    for week in (4, 5, 12):
        assert EG.production_margin_engine(week) == EG.margin_engine(), \
            "the policy may not override the configured engine after week 3"


def test_generic_solve_table_is_not_bound_by_the_early_season_policy():
    """Bakeoffs/backtests ask for an engine on purpose — that must still work."""
    games = _games()
    early = EG.solve_table(games, 2024, 2, engine="kalman")
    assert isinstance(early, K.KalmanTable), \
        "solve_table must honour an explicit engine even in weeks 0-3"
    assert isinstance(EG.solve_table(games, 2024, 2, R.RatingParams(),
                                     engine="ridge"), R.RatingsTable)


def test_early_season_table_carries_prior_year_team_strength():
    """The defect this locks out: ordinary ridge dispatch solves with
    prev_net=None, silently dropping prior-year team strength from the prior."""
    games = _games()
    params = R.RatingParams()
    R.clear_cache()

    tbl = EG.solve_early_season_ridge_table(games, 2025, 1, params)
    assert isinstance(tbl, R.RatingsTable)

    # What plain `R.solve_cached` (engine="ridge") would have produced: no
    # prev_net, so prior_w_prev_rating is zeroed out and the prior collapses.
    naive = R.solve(games, 2025, 1, params)
    assert all(v == 0.0 for v in naive.prior.values()), \
        "fixture check: without prev_net the prior has no signal at all"
    assert any(v != 0.0 for v in tbl.prior.values()), \
        "prior-year team strength must survive into the early-season prior"


def test_early_season_solve_gets_prev_net_and_never_a_prior_injection(monkeypatch):
    games = _games()
    R.clear_cache()
    calls = []
    real_solve = R.solve

    def spy(g, season, week, params, prev_net=None, prior=None):
        calls.append({"season": int(season), "week": int(week),
                      "prev_net": prev_net, "prior": prior})
        return real_solve(g, season, week, params, prev_net=prev_net,
                          prior=prior)

    monkeypatch.setattr(EG.R, "solve", spy)
    EG.solve_early_season_ridge_table(games, 2025, 1, R.RatingParams())

    prev_call = calls[0]      # previous season's final table, plain ridge
    assert (prev_call["season"], prev_call["week"]) == (2024, 4), \
        "prev_net comes from the previous season's final cached table"
    target = calls[-1]
    assert (target["season"], target["week"]) == (2025, 1)
    assert target["prior"] is None, \
        "production must use R.solve's own preseason_prior branch, not prior="
    assert target["prev_net"], "prev_net must reach the solve"
    assert set(target["prev_net"]) <= {"AAA", "BBB", "CCC", "DDD"}, \
        "prev_net is built from non-veto teams of the prior-year table"


def test_early_season_refuses_loudly_without_a_previous_season():
    games = _games()
    one_season = games[games["season"] == 2023]
    with pytest.raises(EG.EarlySeasonPriorError):
        EG.solve_early_season_ridge_table(one_season, 2023, 3,
                                          R.RatingParams())


def test_production_table_routes_by_week():
    games = _games()
    R.clear_cache()
    early = EG.solve_production_table(games, 2025, 1, R.RatingParams())
    assert isinstance(early, R.RatingsTable), \
        "weeks 0-3 are the early-season ridge table regardless of config"
    late = EG.solve_production_table(games, 2024, 4, R.RatingParams())
    assert type(late) is type(EG.solve_table(games, 2024, 4, R.RatingParams()))


def test_id_coverage_problem_names_the_exact_deviation():
    assert EG.id_coverage_problem([1, 2, 3], [3, 2, 1]) is None
    assert "duplicate" in EG.id_coverage_problem([1, 1, 2], [1, 2])
    assert "missing" in EG.id_coverage_problem([1], [1, 2])
    assert "extra" in EG.id_coverage_problem([1, 2, 9], [1, 2])


def test_forecast_status_is_non_actionable():
    banned = ("confidence", "bet", "pick", "edge", "value", "recommend",
              "stake", "unit", "bankroll", "clv", "market", "spread", "total")
    for week in (0, 1, 3, 4, 10):
        status = EG.forecast_status(week)
        assert "paper-only" in status.lower()
        for word in banned:
            assert word not in status.lower(), \
                f"forecast_status must stay non-actionable; found {word!r}"
    assert "early-season" in EG.forecast_status(1).lower()
