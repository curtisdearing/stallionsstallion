"""CFB overtime module — unit tests against HAND-WORKED cases (Workstream C).

BUILD_SPEC section 4 DoD: "OT module unit-tested against hand-worked cases."
Each scripted-RNG test below is one overtime worked out by hand from the
2021+ NCAA rules before the assertion was written.
"""

import numpy as np
import pandas as pd
import pytest

from nflvalue import cfb_ot
from nflvalue.cfb_ot import OTParams, fit_ot_params, simulate_ot


# Round numbers so hand-working is exact: TD 40%, FG 30%, XP-style 7 always
# in OT1, 2-pt 50%.
P = OTParams(p_td=0.40, p_fg=0.30, td_pts={7: 1.0}, p_two=0.50,
             n_possessions=999, n_td_ot1=99, n_two=99)


class ScriptedRNG:
    """Returns pre-scripted uniforms so each branch is chosen by hand."""

    def __init__(self, values):
        self.values = list(values)

    def random(self, n=None):
        if n is None:
            return self.values.pop(0)
        out = np.array([self.values.pop(0) for _ in range(int(n))])
        return out


# --------------------------------------------------------------------------- #
# Hand-worked case 1 — OT1 decides it: home TD (+7), away FG (+3) -> home by 4.
# Draw order in _possession_points: outcome u for all n, then td-points draws.
# u=0.1 < 0.40 -> TD (td_pts {7:1.0} consumes one draw); u=0.5 in [0.40,0.70)
# -> FG. Game ends after round 1: margin +4, 10 OT points.
# --------------------------------------------------------------------------- #
def test_ot1_td_vs_fg_home_wins_by_4():
    rng = ScriptedRNG([0.1,   # home outcome: TD
                       0.5,   # home td_pts draw -> 7 (dist is {7: 1.0})
                       0.5,   # away outcome: FG
                       0.5])  # away td_pts draw (unused branch, still consumed)
    m, pts = simulate_ot(rng, 1, P)
    assert m[0] == 4.0
    assert pts[0] == 10.0


# --------------------------------------------------------------------------- #
# Hand-worked case 2 — OT2 mandatory two: a TD in round 2 is worth 6 or 8,
# NEVER 7. Round 1: both FG (3-3, tied). Round 2: home TD + successful two
# (8), away TD + failed two (6) -> home by 2, 3+3+8+6 = 20 OT points.
# --------------------------------------------------------------------------- #
def test_ot2_two_point_is_mandatory_no_sevens():
    rng = ScriptedRNG([
        0.5, 0.5,        # R1 home: FG (td_pts draw consumed even when no TD)
        0.5, 0.5,        # R1 away: FG -> still tied
        0.1, 0.2,        # R2 home: TD, two-pt u=0.2 < 0.5 -> success = 8
        0.1, 0.9,        # R2 away: TD, two-pt u=0.9 >= 0.5 -> fail = 6
    ])
    m, pts = simulate_ot(rng, 1, P)
    assert m[0] == 2.0
    assert pts[0] == 20.0


# --------------------------------------------------------------------------- #
# Hand-worked case 3 — the OT3+ duel. Rounds 1-2 both scoreless. Round 3:
# alternating single 2-pt tries — home converts (2), away fails (0) -> home
# by 2, total OT points 2.
# --------------------------------------------------------------------------- #
def test_ot3_duel_single_two_point_attempts():
    rng = ScriptedRNG([
        0.99, 0.5, 0.99, 0.5,   # R1: both empty
        0.99, 0.5, 0.99, 0.5,   # R2: both empty
        0.2,                    # R3 duel: home 2-pt success
        0.9,                    # R3 duel: away 2-pt fail
    ])
    m, pts = simulate_ot(rng, 1, P)
    assert m[0] == 2.0
    assert pts[0] == 2.0


# --------------------------------------------------------------------------- #
# Hand-worked case 4 — the duel continues through a tied round. R3 both
# convert (2-2, still tied), R4 home fails / away converts -> away by 2.
# OT points: R1/R2 empty + 2+2 + 0+2 = 6.
# --------------------------------------------------------------------------- #
def test_duel_continues_until_a_round_differs():
    rng = ScriptedRNG([
        0.99, 0.5, 0.99, 0.5,   # R1 empty
        0.99, 0.5, 0.99, 0.5,   # R2 empty
        0.2, 0.2,               # R3: both convert -> tied
        0.9, 0.2,               # R4: home fail, away convert
    ])
    m, pts = simulate_ot(rng, 1, P)
    assert m[0] == -2.0
    assert pts[0] == 6.0


# --------------------------------------------------------------------------- #
# Statistical invariants (real RNG)
# --------------------------------------------------------------------------- #
def test_no_college_game_ever_ties():
    rng = np.random.default_rng(7)
    m, _ = simulate_ot(rng, 20000, P)
    assert (m != 0).all()


def test_symmetric_ot_is_fair():
    rng = np.random.default_rng(11)
    m, _ = simulate_ot(rng, 40000, P)
    assert abs(float((m > 0).mean()) - 0.5) < 0.01


def test_tilt_moves_win_probability_the_right_way():
    rng = np.random.default_rng(13)
    m, _ = simulate_ot(rng, 40000, P, tilt_home=1.4, tilt_away=0.7)
    assert float((m > 0).mean()) > 0.55


# --------------------------------------------------------------------------- #
# Fitting: rules-era filter and fail-loud floors
# --------------------------------------------------------------------------- #
def _fake_ot_drives(season, period, ytg, deltas):
    return pd.DataFrame([{
        "season": season, "start_period": period, "start_yards_to_goal": ytg,
        "start_offense_score": 20, "end_offense_score": 20 + d,
    } for d in deltas])


def test_fit_uses_only_2021_plus_rules_era():
    old = _fake_ot_drives(2019, 5, 25, [7] * 200)      # pre-rules-change era
    new = pd.concat([
        _fake_ot_drives(2021, 5, 25, [7] * 40 + [3] * 30 + [0] * 50),
        _fake_ot_drives(2021, 6, 25, [8] * 5 + [6] * 5),
    ], ignore_index=True)
    p = fit_ot_params(pd.concat([old, new], ignore_index=True))
    assert p.n_possessions == 130, "2019-2020 possessions must be excluded"
    assert p.p_two == pytest.approx(0.5)
    assert p.n_two == 10


def test_fit_refuses_thin_samples():
    with pytest.raises(cfb_ot.OTFitError):
        fit_ot_params(_fake_ot_drives(2021, 5, 25, [7] * 30))
