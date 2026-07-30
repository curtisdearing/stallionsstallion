"""Workstream D market layer — hand-worked sign conventions + walk-forward.

The classic silent bug in a market layer is a flipped spread sign that makes
CLV look brilliant. Every convention here is worked out by hand in the test
before the assertion.
"""

import numpy as np
import pandas as pd
import pytest

from nflvalue import gameline as GL


# --------------------------------------------------------------------------- #
# Blend weight math
# --------------------------------------------------------------------------- #
def test_blend_weight_prefers_the_better_input():
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 14, 4000)
    market = actual + rng.normal(0, 8, 4000)     # tighter
    model = actual + rng.normal(0, 16, 4000)     # looser
    w = GL.blend_weight(model, market, actual)
    assert 0.0 < w < 0.5, "the worse input must get the smaller weight"


def test_equal_inputs_get_equal_weight():
    rng = np.random.default_rng(2)
    actual = rng.normal(0, 14, 20000)
    a = actual + rng.normal(0, 10, 20000)
    b = actual + rng.normal(0, 10, 20000)
    assert GL.blend_weight(a, b, actual) == pytest.approx(0.5, abs=0.03)


# --------------------------------------------------------------------------- #
# Walk-forward discipline (the leakage concern, tested at this layer too)
# --------------------------------------------------------------------------- #
def _bt_frame(weeks=10, per_week=60, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for wk in range(1, weeks + 1):
        actual = rng.normal(0, 14, per_week)
        rows.append(pd.DataFrame({
            "season": 2023, "week": wk,
            "mc_margin": actual + rng.normal(0, 15, per_week),
            "market": actual + rng.normal(0, 12, per_week),
            "margin": actual,
        }))
    return pd.concat(rows, ignore_index=True)


def test_blend_weight_uses_only_earlier_weeks():
    df = _bt_frame()
    base = GL.walk_forward_blend(df, min_history=100)

    poisoned = df.copy()
    fut = poisoned["week"] > 5
    poisoned.loc[fut, "margin"] = 999.0
    after = GL.walk_forward_blend(poisoned, min_history=100)

    for wk in range(1, 6):
        a = base.loc[base["week"] == wk, "blend_w"]
        b = after.loc[after["week"] == wk, "blend_w"]
        assert a.equals(b), (
            f"LEAKAGE: week {wk}'s blend weight moved when weeks > 5 were "
            "poisoned — the weight fit is reading the future")
    later_a = base.loc[base["week"] == 8, "blend_w"].iloc[0]
    later_b = after.loc[after["week"] == 8, "blend_w"].iloc[0]
    assert later_a != later_b, (
        "poisoning weeks 6+ changed nothing at week 8 — the fit is not "
        "actually learning from history, the opposite failure")


def test_thin_history_is_skipped_not_guessed():
    df = _bt_frame(weeks=3, per_week=30)
    out = GL.walk_forward_blend(df, min_history=100)
    assert out.loc[out["week"] == 1, "blend"].isna().all()
    assert out.loc[out["week"] == 2, "blend"].isna().all(), \
        "only 30 pairs of history — below the floor, must be NaN"


# --------------------------------------------------------------------------- #
# Disagreement-OLS blend (D part 2 protocol)
# --------------------------------------------------------------------------- #
def test_disagreement_beta_recovers_known_signal():
    """Construct data where exactly 40% of the model's disagreement with the
    market is real signal; beta must find ~0.4."""
    rng = np.random.default_rng(21)
    n = 30000
    market = rng.normal(0, 10, n)
    disagreement = rng.normal(0, 5, n)
    actual = market + 0.4 * disagreement + rng.normal(0, 12, n)
    model = market + disagreement
    beta = GL.disagreement_beta(model, market, actual)
    assert beta == pytest.approx(0.4, abs=0.05)


def test_disagreement_beta_zero_when_model_is_pure_noise():
    rng = np.random.default_rng(22)
    n = 30000
    market = rng.normal(0, 10, n)
    actual = market + rng.normal(0, 12, n)
    model = market + rng.normal(0, 5, n)      # disagreement carries nothing
    assert abs(GL.disagreement_beta(model, market, actual)) < 0.05


def test_disagreement_blend_is_walk_forward():
    df = _bt_frame()
    base = GL.walk_forward_disagreement_blend(df, min_history=100)
    poisoned = df.copy()
    fut = poisoned["week"] > 5
    poisoned.loc[fut, "margin"] = 999.0
    after = GL.walk_forward_disagreement_blend(poisoned, min_history=100)
    for wk in range(1, 6):
        assert base.loc[base["week"] == wk, "blend_beta"].equals(
            after.loc[after["week"] == wk, "blend_beta"]), (
            f"LEAKAGE: week {wk}'s beta moved when weeks > 5 were poisoned")
    assert (base.loc[base["week"] == 8, "blend_beta"].iloc[0]
            != after.loc[after["week"] == 8, "blend_beta"].iloc[0])


def test_disagreement_blend_thin_history_skipped():
    df = _bt_frame(weeks=2, per_week=30)
    out = GL.walk_forward_disagreement_blend(df, min_history=100)
    assert out["blend"].isna().all()


# --------------------------------------------------------------------------- #
# CLV sign conventions — hand-worked
# --------------------------------------------------------------------------- #
def test_clv_home_bet_line_moves_toward_home():
    # Open home -7, close home -9. Took home at -7; the close says home is
    # 2 points better than the number we hold. CLV = +2.
    assert GL.clv_points_spread("home", -7.0, -9.0) == pytest.approx(2.0)


def test_clv_home_bet_line_moves_against_home():
    # Open -7, close -4.5: we hold -7 but the market now says -4.5. CLV -2.5.
    assert GL.clv_points_spread("home", -7.0, -4.5) == pytest.approx(-2.5)


def test_clv_away_bet_is_the_mirror():
    # Took away +7 at open; close +9 means away got 2 points worse to hold.
    assert GL.clv_points_spread("away", -7.0, -9.0) == pytest.approx(-2.0)
    assert GL.clv_points_spread("away", -7.0, -4.5) == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Slice tags
# --------------------------------------------------------------------------- #
def test_slice_tags():
    assert GL.slice_tag("SEC", "Big Ten", 8, False) == "P4vP4"
    assert GL.slice_tag("Mountain West", "SEC", 8, False) == "G5vP4"
    assert GL.slice_tag("MAC", "Sun Belt", 8, False) == "G5vG5"
    assert GL.slice_tag("SEC", "SEC", 1, False) == "week0-1"
    assert GL.slice_tag("SEC", "ACC", 8, True) == "neutral"


# --------------------------------------------------------------------------- #
# The CLV log end to end — hand-worked single game
# --------------------------------------------------------------------------- #
def test_clv_log_hand_worked_game():
    games = pd.DataFrame([{
        "id": 42, "season": 2023, "week": 8, "home_team": "H", "away_team": "A",
        "home_conf": "SEC", "away_conf": "SEC", "neutral_site": False,
    }])
    lines = pd.DataFrame([
        # Book 1: open -7 (home margin +7). Blend says +10.5 -> edge +3.5,
        # take HOME. Close -9 -> CLV +2.
        {"game_id": 42, "provider": "book1", "spread": -9.0, "spread_open": -7.0},
        # Book 2: open -10.5 (home margin +10.5). Blend +10.5 -> edge 0, skip.
        {"game_id": 42, "provider": "book2", "spread": -9.0, "spread_open": -10.5},
        # Book 3: no opener -> unusable for CLV, must be excluded.
        {"game_id": 42, "provider": "book3", "spread": -9.0, "spread_open": np.nan},
    ])
    log = GL.build_clv_log(games, lines, {42: 10.5}, min_edge_pts=2.0)
    assert len(log) == 1
    r = log.iloc[0]
    assert r["book"] == "book1"
    assert r["side"] == "home"
    assert r["edge_at_open"] == pytest.approx(3.5)
    assert r["clv_pts"] == pytest.approx(2.0)
    assert r["slice"] == "P4vP4"


def test_clv_signal_must_be_close_free():
    """Documents the 2026-07-30 retraction: a 'model' that embeds the closing
    line makes CLV mechanically positive — every candidate is by definition
    'the line moved toward the close', which the close-derived signal knew.
    If this invariant surprises you, read build_clv_log's docstring before
    passing anything blend-shaped into it."""
    games = pd.DataFrame([{
        "id": i, "season": 2023, "week": 8, "home_team": "H", "away_team": "A",
        "home_conf": "SEC", "away_conf": "SEC", "neutral_site": False,
    } for i in (1, 2)])
    lines = pd.DataFrame([
        {"game_id": 1, "provider": "b", "spread": -9.0, "spread_open": -6.0},
        {"game_id": 2, "provider": "b", "spread": -3.0, "spread_open": -6.5},
    ])
    close_derived = {1: 9.0, 2: 3.0}     # exactly the closing margins
    log = GL.build_clv_log(games, lines, close_derived, min_edge_pts=2.0)
    assert len(log) == 2
    assert (log["clv_pts"] > 0).all(), (
        "a close-derived signal must show uniformly positive CLV — that is "
        "the artifact; real signals must be close-free")


def test_clv_log_never_logs_without_a_blend():
    games = pd.DataFrame([{
        "id": 1, "season": 2023, "week": 8, "home_team": "H", "away_team": "A",
        "home_conf": "SEC", "away_conf": "SEC", "neutral_site": False,
    }])
    lines = pd.DataFrame([
        {"game_id": 1, "provider": "b", "spread": -9.0, "spread_open": -7.0}])
    assert GL.build_clv_log(games, lines, {}, 1.0).empty
    assert GL.build_clv_log(games, lines, {1: float("nan")}, 1.0).empty
