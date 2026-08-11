"""Rating-parameter uncertainty sampling in the Kalman->MC hookup (the
calibration deep-dive O fix: bands were ~8% narrow, PIT sd 1.084, because
the posterior variance the KalmanTable carries was never sampled).

THE GATE, asserted here: sampling must not change point predictions —
means exactly (antithetic pairing), medians statistically — only band
widths. And it must be scoped to the kalman engine so the registered ridge
gate-3 record stays reproducible bit-for-bit.
"""

import numpy as np
import pandas as pd
import pytest

from nflvalue import cfb_sim
from nflvalue.cfb_ot import OTParams
from nflvalue.cfb_sim import DriveRateModel, PaceModel, simulate_game

OT = OTParams(p_td=0.42, p_fg=0.28, td_pts={7: 0.9, 6: 0.06, 8: 0.04},
              p_two=0.45, n_possessions=500, n_td_ot1=200, n_two=20)


def _drive_row(game_id, week, home_off, off_delta, def_delta, period=2):
    return {
        "game_id": game_id, "season": 2023, "week": week,
        "is_home_offense": home_off, "start_period": period,
        "start_offense_score": 10, "end_offense_score": 10 + off_delta,
        "start_defense_score": 10, "end_defense_score": 10 + def_delta,
    }


def _labeled(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 8, n)
    p_td = np.clip(0.28 + 0.012 * e, 0.02, 0.9)
    u = rng.random(n)
    off = np.where(u < p_td, 7, np.where(u < p_td + 0.10, 3, 0))
    return pd.DataFrame({"e": e, "off_pts": off, "def_pts": 0})


def _pace_drives(n_games=80, home_n=12, away_n=11):
    rows = []
    for gid in range(1, n_games + 1):
        rows += [_drive_row(gid, 1, True, 0, 0)] * home_n
        rows += [_drive_row(gid, 1, False, 0, 0)] * away_n
    return pd.DataFrame(rows)


def _fit_models():
    rates = DriveRateModel.fit(_labeled(), n_bins=8, min_bin_n=200)
    pace = PaceModel.fit(_pace_drives())
    return rates, pace


def _pair(e_h=4.0, e_a=-3.0, n=4000, param_sd=3.0):
    """(base, sampled) sims with identical base random streams."""
    rates, pace = _fit_models()
    base = simulate_game(np.random.default_rng(11), e_h, e_a, rates, pace,
                         OT, n=n)
    samp = simulate_game(np.random.default_rng(11), e_h, e_a, rates, pace,
                         OT, n=n, param_sd=param_sd,
                         param_rng=np.random.default_rng([20260730, 42]))
    return base, samp


# --------------------------------------------------------------------------- #
# THE GATE: point predictions do not move; only band widths do
# --------------------------------------------------------------------------- #
def test_mean_margin_exactly_unchanged():
    base, samp = _pair()
    assert samp["margin_mean"] == pytest.approx(base["margin_mean"],
                                                abs=1e-9), (
        "antithetic pairing makes the sampled shift sum to exactly zero — "
        "the published point prediction must not move")


def test_median_margin_statistically_unchanged():
    base, samp = _pair(n=20000)
    med_b = float(np.median(base["margin"]))
    med_s = float(np.median(samp["margin"]))
    assert abs(med_s - med_b) < 0.5, (
        f"median moved {med_s - med_b:+.2f} pts — the shift is zero-mean "
        "symmetric noise; the median may jitter, never drift")


def test_band_widens_and_total_untouched():
    base, samp = _pair(param_sd=6.0)
    assert samp["margin_sd"] > base["margin_sd"] + 0.5, (
        "sampling 6 pts of rating uncertainty must visibly widen the band")
    # sd must grow in quadrature (independent additive noise), not explode
    expect = float(np.sqrt(base["margin_sd"] ** 2 + 6.0 ** 2))
    assert samp["margin_sd"] == pytest.approx(expect, rel=0.05)
    np.testing.assert_array_equal(samp["total"], base["total"])
    assert samp["total_mean"] == base["total_mean"]
    assert samp["total_sd"] == base["total_sd"]


def test_win_prob_pulled_toward_half_for_favorite():
    base, samp = _pair(e_h=7.0, e_a=-7.0, n=8000)
    assert base["p_home_win"] > 0.5
    assert 0.5 < samp["p_home_win"] < base["p_home_win"], (
        "wider margin bands must make the favorite less certain, and never "
        "flip the pick")


def test_param_sd_zero_is_bit_identical():
    rates, pace = _fit_models()
    a = simulate_game(np.random.default_rng(7), 2.0, -1.0, rates, pace, OT,
                      n=2000)
    b = simulate_game(np.random.default_rng(7), 2.0, -1.0, rates, pace, OT,
                      n=2000, param_sd=0.0)
    np.testing.assert_array_equal(a["margin"], b["margin"])
    np.testing.assert_array_equal(a["total"], b["total"])


def test_sampling_is_deterministic_given_param_rng():
    _, s1 = _pair()
    _, s2 = _pair()
    np.testing.assert_array_equal(s1["margin"], s2["margin"])


# --------------------------------------------------------------------------- #
# Engine scoping: kalman tables sample, ridge tables do not
# --------------------------------------------------------------------------- #
class _FakeKalman:
    engine = "kalman"
    net_sd = {"A": 3.0, "B": 4.0}


class _FakeRidge:  # no `engine` attr — like ratings.RatingsTable
    net_sd = {"A": 3.0, "B": 4.0}


def test_param_margin_sd_is_kalman_scoped():
    assert cfb_sim.param_margin_sd(_FakeKalman(), "A", "B") == pytest.approx(5.0)
    assert cfb_sim.param_margin_sd(_FakeRidge(), "A", "B") == 0.0, (
        "the ridge gate-3 record is registered UNSAMPLED — widening it "
        "silently would rewrite a recorded result")
    assert cfb_sim.param_margin_sd(_FakeKalman(), "A", "unknown") == \
        pytest.approx(3.0)
