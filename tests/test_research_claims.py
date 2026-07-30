"""Workstream M — research claims as falsifiable tests.

Everything in research/ is prose, and prose can't fail. Each test here
converts one quantitative claim into an assertion against THIS PROJECT'S OWN
measured data, citing its source document. The critical framing (BUILD_SPEC
§17): a failing test here means the literature doesn't hold on our data —
that is a finding to write back into the source doc, never a bound to loosen
until it passes.

Tests that need the local parquet cache skip cleanly on a data-less checkout
(CI has no cache — *.parquet is gitignored); they run wherever the real
cache exists, which is where the claims matter.

Claims deliberately NOT tested yet (recorded, not silent):
  * "penalties near-zero correlation with wins" — no penalty data cached
    (Tier-3 excluded input anyway; testing it needs a one-off pull).
  * "raw turnover margin ~2.6% forward-predictive" — needs an expected-
    turnover model to separate luck from rate; Workstream E follow-up.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HAS_CACHE = os.path.exists(os.path.join(DATA, "cfbd", "games_2023.parquet"))
needs_cache = pytest.mark.skipif(
    not HAS_CACHE, reason="needs the local CFBD parquet cache (gitignored)")


def _registry(factor_id):
    with open(os.path.join(DATA, "factor_registry.json")) as fh:
        for e in json.load(fh):
            if e["factor_id"] == factor_id:
                return e
    raise AssertionError(f"{factor_id} missing from registry")


# --------------------------------------------------------------------------- #
# HFA — research/win-factors-literature-scan.md: "~2.5-3.5 points"
# --------------------------------------------------------------------------- #
def test_fitted_league_hfa_is_in_the_literature_range():
    e = _registry("league_hfa")
    assert 2.5 <= float(e["effect_size"]) <= 3.5, (
        f"fitted league HFA {e['effect_size']} is outside the literature's "
        "2.5-3.5 — if this fails after a refit, write the measured value "
        "back into win-factors-literature-scan.md; do not widen this range")


@needs_cache
def test_raw_home_margin_is_consistent_with_a_positive_hfa():
    """The raw (strength-confounded) home margin should be positive and
    single-digit — a sanity check that the fitted number isn't an artifact
    of the solve. Source: win-factors-literature-scan.md."""
    import sys
    sys.path.insert(0, ROOT)
    from nflvalue import ingest
    g = ingest.load_games(range(2019, 2026))
    g = g.dropna(subset=["home_points", "away_points"])
    g = g[~g["home_fcs"] & ~g["away_fcs"] & ~g["neutral_site"].astype(bool)]
    raw = float((g["home_points"] - g["away_points"]).mean())
    assert 1.0 <= raw <= 6.0, f"raw home margin {raw:.2f} looks broken"


# --------------------------------------------------------------------------- #
# Explosiveness — proven-win-indicators.md: near-zero week-to-week stickiness
# (this project CONFIRMED the literature at n=133 teams, 2026-07-19)
# --------------------------------------------------------------------------- #
def test_explosiveness_stickiness_recorded_near_zero():
    e = _registry("explosiveness_stickiness")
    assert abs(float(e["effect_size"])) < 0.2, (
        "explosiveness stickiness left the near-zero band the literature "
        "and our own 2026-07-19 measurement agree on — re-measure before "
        "trusting either")


# --------------------------------------------------------------------------- #
# Success rate vs explosiveness — win-factors-literature-scan.md: success
# rate is the stickier signal week to week
# --------------------------------------------------------------------------- #
@needs_cache
def test_success_rate_is_stickier_than_explosiveness():
    import sys
    sys.path.insert(0, ROOT)
    from nflvalue import ingest, ratings as R
    plays = ingest.load_plays_clean([2023])
    ok = plays.dropna(subset=["down", "distance", "yards_gained"]).copy()
    succ = [R._success(d, dist, y) for d, dist, y in
            zip(ok["down"], ok["distance"], ok["yards_gained"])]
    ok["succ"] = pd.array(succ, dtype="boolean")
    ok = ok.dropna(subset=["succ"])
    ok["explosive"] = ok["yards_gained"] >= 15

    def lag1(col):
        tw = (ok.groupby(["offense", "week"], observed=True)[col]
              .mean().reset_index().sort_values(["offense", "week"]))
        tw["prev"] = tw.groupby("offense", observed=True)[col].shift(1)
        tw = tw.dropna(subset=["prev"])
        return float(np.corrcoef(tw[col].astype(float),
                                 tw["prev"].astype(float))[0, 1])

    s, e = lag1("succ"), lag1("explosive")
    assert s > e, (
        f"success-rate lag-1 r={s:.3f} is NOT stickier than explosiveness "
        f"r={e:.3f} on 2023 — the Five Factors ordering claim fails on our "
        "data; write this back into win-factors-literature-scan.md")


# --------------------------------------------------------------------------- #
# Garbage time — cfb-portability-plan.md: Connelly-style filter strips a
# large-but-not-crazy share of CFB plays (measured ~10% at ingest)
# --------------------------------------------------------------------------- #
@needs_cache
def test_garbage_time_share_is_in_the_expected_band():
    import sys
    sys.path.insert(0, ROOT)
    from nflvalue import ingest
    raw = len(ingest.load_plays([2023]))
    clean = len(ingest.load_plays_clean([2023]))
    share = 1 - clean / raw
    assert 0.06 <= share <= 0.15, (
        f"garbage-time share {share:.1%} left the 6-15% band — either the "
        "thresholds drifted or the data did; investigate before refitting")


# --------------------------------------------------------------------------- #
# Margin noise — core-modeling-philosophy.md / CFB literature: game margin
# residual SD ~16-17 points (wider than the NFL's ~13)
# --------------------------------------------------------------------------- #
def test_out_of_sample_margin_residual_sd_matches_cfb_literature():
    bt = pd.read_csv(os.path.join(DATA, "backtest_c.csv"))
    sd = float((bt["mc_margin"] - bt["margin"]).std())
    assert 15.0 <= sd <= 18.5, (
        f"residual SD {sd:.2f} outside the CFB literature's neighborhood — "
        "a much smaller number means leakage before it means genius")


# --------------------------------------------------------------------------- #
# Market efficiency — cfb-portability-plan.md: the close is a ~12-12.5 MAE
# forecaster and the realistic ceiling is thin
# --------------------------------------------------------------------------- #
def test_closing_line_mae_is_in_the_efficient_band():
    bt = pd.read_csv(os.path.join(DATA, "backtest_d.csv"))
    mae = float((bt["market"] - bt["margin"]).abs().mean())
    assert 11.5 <= mae <= 13.0, (
        f"market MAE {mae:.2f} left the efficient band — check the spread "
        "sign convention before anything else")


# --------------------------------------------------------------------------- #
# Overtime — situational-factors.md: a few percent of CFB games reach OT
# --------------------------------------------------------------------------- #
@needs_cache
def test_ot_frequency_is_a_few_percent():
    import sys
    sys.path.insert(0, ROOT)
    from nflvalue import ingest
    d = ingest.load_drives(range(2021, 2026))
    g = ingest.load_games(range(2021, 2026))
    ot_games = d.loc[d["start_period"] >= 5, "game_id"].nunique()
    fbs = g[~g["home_fcs"] | ~g["away_fcs"]]
    share = ot_games / len(fbs)
    assert 0.01 <= share <= 0.06, (
        f"OT share {share:.1%} is outside the few-percent band the OT "
        "module's tilt bound in backtest_c.py relies on")
