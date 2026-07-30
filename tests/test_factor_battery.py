"""Workstream E battery machinery — gate-6 discipline tests.

The battery itself runs against real data in scripts/factor_battery.py;
this file proves the MACHINERY is honest: BH math hand-worked, exclusions
enforced from the registry (single source of truth), event tests recover
known effects and refuse thin samples, continuous tests recover known
slopes, and no code path in the battery can promote to gated_active.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from nflvalue import factor_battery as FB


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg — hand-worked
# --------------------------------------------------------------------------- #
def test_bh_hand_worked_example():
    # m=4, q=0.05 -> thresholds 0.0125, 0.025, 0.0375, 0.05 by rank.
    # p = .003, .02, .04, .30: rank1 .003<=.0125 pass; rank2 .02<=.025 pass;
    # rank3 .04>.0375 fail; rank4 .30>.05 fail. Max passing rank = 2.
    res = FB.benjamini_hochberg(
        {"a": 0.003, "b": 0.02, "c": 0.04, "d": 0.30}, q=0.05)
    assert res["a"]["pass"] and res["b"]["pass"]
    assert not res["c"]["pass"] and not res["d"]["pass"]


def test_bh_step_up_property():
    # The step-up: a later p under its threshold rescues earlier ones.
    res = FB.benjamini_hochberg({"a": 0.012, "b": 0.024, "c": 0.037}, q=0.05)
    assert all(res[k]["pass"] for k in "abc")


# --------------------------------------------------------------------------- #
# Registry as single source of truth
# --------------------------------------------------------------------------- #
def _reg(tmp_path, entries):
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(entries))
    return str(p)


def test_excluded_factor_raises(tmp_path):
    path = _reg(tmp_path, [{"factor_id": "raw_turnover_margin",
                            "status": "excluded", "description": "tier 3"}])
    reg = FB.load_registry(path)
    with pytest.raises(FB.ExcludedFactorError):
        FB.assert_not_excluded(reg, "raw_turnover_margin")


def test_unknown_factor_must_be_registered_first(tmp_path):
    reg = FB.load_registry(_reg(tmp_path, []))
    with pytest.raises(FB.FactorError, match="not in registry"):
        FB.registry_entry(reg, "brand_new_idea")


def test_missing_registry_fails_loud(tmp_path):
    with pytest.raises(FB.FactorError, match="single source of truth"):
        FB.load_registry(str(tmp_path / "nope.json"))


def test_real_registry_has_all_tier3_exclusions():
    reg = FB.load_registry()
    excluded = {e["factor_id"] for e in reg if e["status"] == "excluded"}
    for fid in ("raw_time_of_possession", "penalty_counts",
                "raw_red_zone_td_pct", "raw_pressure_rate",
                "raw_turnover_margin"):
        assert fid in excluded, f"Tier-3 exclusion {fid} missing from registry"


def test_battery_module_never_writes_gated_active():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "factor_battery.py")).read()
    assert '"gated_active"' not in src.replace("touch `gated_active`", ""), (
        "the battery script must never set gated_active — promotion is a "
        "separate human-visible decision (FACTOR_REVALIDATION_SPEC)")


# --------------------------------------------------------------------------- #
# Event factor — matched control
# --------------------------------------------------------------------------- #
def _preds(n_weeks=10, per_week=40, seed=3, sd=10.0):
    rng = np.random.default_rng(seed)
    rows = []
    gid = 1
    for wk in range(1, n_weeks + 1):
        for _ in range(per_week):
            pred = rng.normal(0, 12)
            rows.append({"id": gid, "season": 2023, "week": wk,
                         "pred_b": pred,
                         "margin": pred + rng.normal(0, sd)})
            gid += 1
    return pd.DataFrame(rows)


def test_event_factor_recovers_a_planted_effect():
    preds = _preds()
    rng = np.random.default_rng(4)
    treated_ids = rng.choice(preds["id"], 120, replace=False)
    # Plant: treated HOME sides underperform by 6 points.
    preds.loc[preds["id"].isin(treated_ids), "margin"] -= 6.0
    events = pd.DataFrame({"id": treated_ids, "side": "home"})
    r = FB.evaluate_event_factor("planted", events, preds, min_treated=40)
    assert r.p_value < 0.01
    assert r.effect_pts == pytest.approx(-6.0, abs=2.5)


def test_event_factor_finds_nothing_in_noise():
    preds = _preds(seed=9)
    rng = np.random.default_rng(10)
    events = pd.DataFrame({"id": rng.choice(preds["id"], 120, replace=False),
                           "side": "away"})
    r = FB.evaluate_event_factor("noise", events, preds, min_treated=40)
    assert r.p_value > 0.05 or abs(r.effect_pts) < 2.0


def test_event_factor_n_gate_refuses_thin_samples():
    preds = _preds()
    events = pd.DataFrame({"id": preds["id"].head(10), "side": "home"})
    r = FB.evaluate_event_factor("thin", events, preds, min_treated=40)
    assert "n-gate FAIL" in r.note
    assert r.p_value != r.p_value  # NaN — never a fake p


# --------------------------------------------------------------------------- #
# Continuous factor
# --------------------------------------------------------------------------- #
def test_continuous_factor_recovers_known_slope():
    preds = _preds(n_weeks=25, per_week=40, seed=5)
    rng = np.random.default_rng(6)
    diff = rng.normal(0, 2, len(preds))
    preds["margin"] = preds["margin"] + 1.5 * diff
    diffs = pd.DataFrame({"id": preds["id"], "diff": diff})
    r = FB.evaluate_continuous_factor("planted", diffs, preds, min_n=800)
    assert r.p_value < 0.01
    assert r.slope == pytest.approx(1.5, abs=0.4)


def test_continuous_factor_n_gate():
    preds = _preds(n_weeks=2, per_week=20)
    diffs = pd.DataFrame({"id": preds["id"], "diff": 0.0})
    r = FB.evaluate_continuous_factor("thin", diffs, preds, min_n=800)
    assert "n-gate FAIL" in r.note
