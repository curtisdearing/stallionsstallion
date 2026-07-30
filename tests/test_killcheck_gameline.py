"""Game-line kill-check bars — derivation and verdict tests (Workstream K prep)."""

import numpy as np
import pandas as pd
import pytest

from nflvalue import killcheck_gameline as KG


def test_min_effect_is_derived_from_measured_quantities_not_guessed():
    # phi(0)/16.74 ~= 0.0238 cover-prob per point; half the vig (0.0119)
    # divided by that ~= 0.50 points. Derived, and stable to recompute.
    assert KG.min_effect_pts() == pytest.approx(0.50, abs=0.02)


def test_n_gate_matches_the_power_formula_hand_worked():
    # sd=2.4, effect=0.5, one-sided alpha .05 (z=1.645), power .80 (z=0.842):
    # n = ((1.645+0.842)*2.4/0.5)^2 = (11.94)^2 ~= 143
    assert KG.n_gate(clv_sd=2.4, effect_pts=0.5) == pytest.approx(143, abs=2)


def test_n_gate_is_not_the_props_number_by_inheritance():
    # It can LAND near 150 for sd~2.45 — but it must move with the inputs,
    # which the inherited constant never would.
    assert KG.n_gate(clv_sd=1.2, effect_pts=0.5) < KG.n_gate(clv_sd=2.4,
                                                             effect_pts=0.5)
    assert KG.n_gate(clv_sd=2.4, effect_pts=1.0) < KG.n_gate(clv_sd=2.4,
                                                             effect_pts=0.5)


def _log(slice_tag, means, sd, n, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "game_id": range(n), "book": "b", "slice": slice_tag,
        "clv_pts": rng.normal(means, sd, n)})


def test_go_requires_both_n_and_significance():
    strong = _log("P4vP4", 0.8, 2.0, 400)
    v = KG.slice_verdict(strong, "P4vP4")
    assert v.verdict == "GO"
    assert v.n >= v.n_required


def test_underpowered_slice_is_insufficient_not_nogo():
    small = _log("neutral", 0.8, 2.4, 30)
    v = KG.slice_verdict(small, "neutral")
    assert v.verdict == "INSUFFICIENT_N", \
        "an unpowered test proves nothing either way — never NO-GO"


def test_null_slice_is_nogo():
    null = _log("G5vG5", 0.0, 2.4, 800, seed=7)
    v = KG.slice_verdict(null, "G5vG5")
    assert v.verdict in ("NO-GO", "INSUFFICIENT_N")
    if v.verdict == "NO-GO":
        assert v.p_value > 0.05 or v.mean_clv <= 0


def test_books_within_a_game_are_not_independent_observations():
    # 100 games x 4 books each: n must be 100 after dedup, not 400.
    rows = []
    rng = np.random.default_rng(3)
    for g in range(100):
        clv = rng.normal(0.5, 2.0)
        for b in range(4):
            rows.append({"game_id": g, "book": f"b{b}", "slice": "P4vP4",
                         "clv_pts": clv + rng.normal(0, 0.1)})
    v = KG.slice_verdict(pd.DataFrame(rows), "P4vP4")
    assert v.n == 100, "book duplicates must be collapsed per game"


def test_run_killcheck_never_pools_and_labels_research_mode():
    log = pd.concat([_log("P4vP4", 0.5, 2.0, 200),
                     _log("G5vG5", 0.0, 2.4, 200, seed=9)],
                    ignore_index=True)
    verdicts = KG.run_killcheck(log, research_mode=True)
    assert {v.slice_tag for v in verdicts} == {"P4vP4", "G5vG5"}
    assert all("RESEARCH-MODE" in v.note for v in verdicts), \
        "backtest-era runs must label themselves as not-gate-7"
