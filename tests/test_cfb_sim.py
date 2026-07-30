"""Drive-level CFB Monte Carlo — behaviour tests (Workstream C).

Offline and synthetic throughout. The walk-forward LEAKAGE coverage for this
layer lives in tests/test_leakage.py (extended, not replaced, per the
standing instruction); this file covers the model mechanics.
"""

import numpy as np
import pandas as pd
import pytest

from nflvalue import cfb_sim
from nflvalue.cfb_ot import OTParams
from nflvalue.cfb_sim import (CLASSES, DriveRateModel, PaceModel,
                              classify_drives, simulate_game)

OT = OTParams(p_td=0.42, p_fg=0.28, td_pts={7: 0.9, 6: 0.06, 8: 0.04},
              p_two=0.45, n_possessions=500, n_td_ot1=200, n_two=20)


def _drive_row(game_id, week, home_off, off_delta, def_delta, period=2):
    return {
        "game_id": game_id, "season": 2023, "week": week,
        "is_home_offense": home_off, "start_period": period,
        "start_offense_score": 10, "end_offense_score": 10 + off_delta,
        "start_defense_score": 10, "end_defense_score": 10 + def_delta,
    }


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_classifier_drops_and_counts_garbage_deltas():
    rows = [_drive_row(1, 1, True, 7, 0), _drive_row(1, 1, True, 3, 0),
            _drive_row(1, 1, False, 0, 2),
            _drive_row(1, 1, True, 14, 0),   # two scores in one row: noise
            _drive_row(1, 1, True, 0, 3)]    # a defensive FG does not exist
    d, dropped = classify_drives(pd.DataFrame(rows))
    assert len(d) == 3
    assert dropped == 2


# --------------------------------------------------------------------------- #
# Rate model
# --------------------------------------------------------------------------- #
def _labeled(n=6000, seed=3):
    """Synthetic drives whose TD rate genuinely rises with e."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 8, n)
    p_td = np.clip(0.28 + 0.012 * e, 0.02, 0.9)
    u = rng.random(n)
    off = np.where(u < p_td, 7, np.where(u < p_td + 0.10, 3, 0))
    return pd.DataFrame({"e": e, "off_pts": off, "def_pts": 0})


def test_rate_model_is_matchup_conditional():
    m = DriveRateModel.fit(_labeled(), n_bins=8, min_bin_n=200)
    assert m.p_td(10.0) > m.p_td(0.0) > m.p_td(-10.0), (
        "TD probability must rise with the offense edge — a flat model is "
        "exactly the league-wide-prior failure BUILD_SPEC 4 forbids")


def test_rate_model_interpolates_between_bins():
    m = DriveRateModel.fit(_labeled(), n_bins=6, min_bin_n=200)
    lo, mid, hi = m.p_td(-2.0), m.p_td(0.0), m.p_td(2.0)
    assert lo <= mid <= hi
    p = m.class_probs(1.234)
    assert p.shape == (len(CLASSES),)
    assert p.sum() == pytest.approx(1.0)


def test_rate_model_refuses_to_fit_unconditional():
    with pytest.raises(cfb_sim.SimError, match="unconditional"):
        DriveRateModel.fit(_labeled(300), n_bins=8, min_bin_n=400)


# --------------------------------------------------------------------------- #
# Pace model
# --------------------------------------------------------------------------- #
def _pace_drives(n_games=80, home_n=12, away_n=11):
    rows = []
    for gid in range(1, n_games + 1):
        rows += [_drive_row(gid, 1, True, 0, 0)] * home_n
        rows += [_drive_row(gid, 1, False, 0, 0)] * away_n
    return pd.DataFrame(rows)


def test_pace_pairs_are_sampled_jointly():
    p = PaceModel.fit(_pace_drives())
    rng = np.random.default_rng(5)
    nh, na = p.sample(rng, 500)
    assert (nh == 12).all() and (na == 11).all(), (
        "pairs must be sampled whole — independent marginals would break the "
        "home/away drive-count correlation")


# --------------------------------------------------------------------------- #
# Game simulation
# --------------------------------------------------------------------------- #
def _fit_models():
    rates = DriveRateModel.fit(_labeled(), n_bins=8, min_bin_n=200)
    pace = PaceModel.fit(_pace_drives())
    return rates, pace


def test_simulated_games_never_tie():
    rates, pace = _fit_models()
    rng = np.random.default_rng(9)
    sim = simulate_game(rng, 0.0, 0.0, rates, pace, OT, n=4000)
    assert (sim["margin"] != 0).all()
    assert sim["p_ot"] > 0, "even matchups must sometimes reach OT"


def test_stronger_side_wins_more_and_margin_tracks_edge():
    rates, pace = _fit_models()
    rng = np.random.default_rng(10)
    fav = simulate_game(rng, 7.0, -7.0, rates, pace, OT, n=6000)
    dog = simulate_game(rng, -7.0, 7.0, rates, pace, OT, n=6000)
    assert fav["p_home_win"] > 0.7 > 0.3 > dog["p_home_win"]
    assert fav["margin_mean"] > 5.0
    assert dog["margin_mean"] < -5.0


def test_full_distributions_are_returned_not_point_estimates():
    rates, pace = _fit_models()
    rng = np.random.default_rng(11)
    sim = simulate_game(rng, 3.0, -3.0, rates, pace, OT, n=2000)
    assert sim["margin"].shape == (2000,)
    assert sim["total"].shape == (2000,)
    assert sim["margin_sd"] > 5.0, "a near-deterministic margin is a bug"


def test_same_seed_same_distribution():
    rates, pace = _fit_models()
    a = simulate_game(np.random.default_rng(42), 2.0, -1.0, rates, pace, OT, n=1000)
    b = simulate_game(np.random.default_rng(42), 2.0, -1.0, rates, pace, OT, n=1000)
    assert np.array_equal(a["margin"], b["margin"])
    assert np.array_equal(a["total"], b["total"])


def test_blowout_edge_widens_expected_margin_monotonically():
    """The matchup-conditional requirement, end to end: bigger rating gaps
    must produce bigger simulated margins, not saturate at a league prior."""
    rates, pace = _fit_models()
    rng = np.random.default_rng(12)
    means = [simulate_game(rng, e, -e, rates, pace, OT, n=4000)["margin_mean"]
             for e in (0.0, 5.0, 10.0)]
    assert means[0] < means[1] < means[2]
