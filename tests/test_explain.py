"""Workstream F explainer — gate-5 DoD tests.

DoD: contributions sum to the projection (asserted); plain language is bound
to the same values as the table; the why-not panel is never empty; the
paper-mode banner cannot be disabled without a verified kill-check GO; and
the display layer is architecturally read-only (import-direction test).
"""

import os
import re

import numpy as np
import pandas as pd
import pytest

from nflvalue import cfb_sim, explain, ratings as R
from nflvalue.cfb_ot import OTParams

_RP = R.RatingParams(lambda_team=2.0, lambda_conf=1.0, prior_strength=0.0)
_OT = OTParams(p_td=0.4, p_fg=0.3, td_pts={7: 1.0}, p_two=0.5,
               n_possessions=500, n_td_ot1=100, n_two=20)
TEAMS = tuple(f"T{i:02d}" for i in range(12))


def _games(weeks=8):
    strengths = {t: 12.0 - 2.2 * i for i, t in enumerate(TEAMS)}
    rows, gid = [], 7000
    for wk in range(1, weeks + 1):
        for i, h in enumerate(TEAMS):
            a = TEAMS[(i + wk) % len(TEAMS)]
            if a == h:
                continue
            hp = 28.0 + strengths[h] / 2 + 1.5
            ap = 28.0 + strengths[a] / 2
            rows.append({"id": gid, "season": 2023, "week": wk,
                         "home_team": h, "away_team": a, "home_points": hp,
                         "away_points": ap, "neutral_site": False,
                         "home_fbs": True, "away_fbs": True,
                         "home_conf": "TEST", "away_conf": "TEST",
                         "margin": hp - ap})
            gid += 1
    return pd.DataFrame(rows).sort_values(["season", "week", "id"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def tbl():
    R.clear_cache()
    return R.solve(_games(), 2023, 7, _RP, prior={t: 0.0 for t in TEAMS})


@pytest.fixture(scope="module")
def sim(tbl):
    rng = np.random.default_rng(6)
    labeled = pd.DataFrame({
        "e": rng.normal(0, 8, 4000),
        "off_pts": rng.choice([0, 3, 7], 4000, p=[0.6, 0.12, 0.28]),
        "def_pts": 0})
    rates = cfb_sim.DriveRateModel.fit(labeled, n_bins=4, min_bin_n=200)
    pace_rows = []
    for gid in range(60):
        for ho in (True, False):
            for _ in range(11):
                pace_rows.append({"game_id": gid, "is_home_offense": ho,
                                  "start_period": 2,
                                  "start_offense_score": 0,
                                  "end_offense_score": 0,
                                  "start_defense_score": 0,
                                  "end_defense_score": 0})
    pace = cfb_sim.PaceModel.fit(pd.DataFrame(pace_rows))
    e_h, e_a = cfb_sim.side_edges(tbl, "T00", "T11", False)
    return cfb_sim.simulate_game(rng, e_h, e_a, rates, pace, _OT, n=1500)


def _exp(tbl, sim, market=3.0):
    return explain.build_explanation(
        tbl=tbl,
        game={"id": 1, "season": 2023, "week": 7, "home_team": "T00",
              "away_team": "T11", "neutral_site": False},
        sim=sim, market=market, market_total=55.0, model_mae=13.2,
        slice_tag="P4vP4", clv_slices=None)


# --------------------------------------------------------------------------- #
# F1 — the sum is exact
# --------------------------------------------------------------------------- #
def test_contributions_sum_to_linear_prediction(tbl):
    rows = explain.decompose_margin(tbl, "T00", "T11", False)
    total = sum(r["points"] for r in rows)
    assert total == pytest.approx(tbl.predict_margin("T00", "T11", False),
                                  abs=1e-3)


def test_contributions_sum_to_simulated_margin(tbl, sim):
    e = _exp(tbl, sim)
    assert explain.contributions_sum(e) == pytest.approx(e["mc_margin"],
                                                         abs=1e-9)


def test_neutral_site_zeroes_hfa_row(tbl):
    rows = explain.decompose_margin(tbl, "T00", "T11", True)
    hfa = next(r for r in rows if r["key"] == "hfa")
    assert hfa["points"] == 0.0


def test_excluded_rows_carry_zero_points_and_reasons():
    for r in explain.excluded_rows():
        assert r["points"] == 0.0
        assert r["tier"] in ("excluded", "ungated")
        assert len(r["why"]) > 20


# --------------------------------------------------------------------------- #
# F3 — prose bound to the same values (property test)
# --------------------------------------------------------------------------- #
def test_prose_numbers_match_table_numbers(tbl, sim):
    e = _exp(tbl, sim)
    by_key = {r["key"]: r for r in e["contributions"]}
    for key, text in e["prose"].items():
        pts = by_key[key]["points"]
        nums = [abs(float(x)) for x in re.findall(r"-?\d+\.\d", text)]
        assert any(abs(n - abs(round(pts, 1))) < 0.051 for n in nums), (
            f"prose for {key!r} ({text!r}) does not contain its own value "
            f"{pts:+.1f} — the words have drifted from the numbers")


# --------------------------------------------------------------------------- #
# F5 — why-not never empty
# --------------------------------------------------------------------------- #
def test_why_not_never_empty_even_in_the_best_case(tbl, sim):
    e = _exp(tbl, sim, market=3.0)
    assert len(e["why_not"]) >= 1
    assert any("kill-check" in r.lower() for r in e["why_not"])


def test_why_not_flags_missing_market(tbl, sim):
    e = _exp(tbl, sim, market=float("nan"))
    assert any("no market line" in r.lower() for r in e["why_not"])


# --------------------------------------------------------------------------- #
# F7 — paper banner cannot be talked off
# --------------------------------------------------------------------------- #
def test_paper_banner_on_by_default():
    assert "PAPER MODE" in explain.paper_mode_banner()
    assert "1-800-GAMBLER" in explain.paper_mode_banner()


def test_paper_banner_refuses_to_disable_without_verified_go():
    with pytest.raises(explain.PaperModeError):
        explain.paper_mode_banner(disable=True)
    with pytest.raises(explain.PaperModeError):
        explain.paper_mode_banner(disable=True, killcheck_go={"go": True})
    with pytest.raises(explain.PaperModeError):
        explain.paper_mode_banner(
            disable=True,
            killcheck_go={"go": True, "slice": "P4vP4", "n": 80,
                          "clv_mean": -0.2})


def test_paper_banner_accepts_a_complete_go_object_only():
    txt = explain.paper_mode_banner(
        disable=True,
        killcheck_go={"go": True, "slice": "P4vP4", "n": 80, "clv_mean": 0.4})
    assert "P4vP4" in txt and "1-800-GAMBLER" in txt


# --------------------------------------------------------------------------- #
# F8 — architecturally read-only: nothing upstream imports the explainer
# --------------------------------------------------------------------------- #
def test_no_upstream_module_imports_explain():
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "nflvalue")
    pattern = re.compile(
        r"^\s*(?:from\s+[.\w]*\bexplain\b|import\s+[\w.]*\bexplain\b"
        r"|from\s+[.\w]+\s+import\s+[^#\n]*\bexplain\b)", re.M)
    for mod in ("ratings.py", "cfb_sim.py", "cfb_ot.py", "gameline.py",
                "ingest.py", "montecarlo.py"):
        src = open(os.path.join(root, mod)).read()
        assert not pattern.search(src), (
            f"{mod} imports the explainer — the F8 read-only guarantee is "
            "broken; explanations must never be able to move a projection")


def test_the_import_pattern_itself_catches_violations():
    """The F8 test must have teeth: verify the regex fires on the shapes a
    violation would actually take."""
    pattern = re.compile(
        r"^\s*(?:from\s+[.\w]*\bexplain\b|import\s+[\w.]*\bexplain\b"
        r"|from\s+[.\w]+\s+import\s+[^#\n]*\bexplain\b)", re.M)
    for bad in ("from . import explain",
                "from .explain import build_explanation",
                "import nflvalue.explain",
                "from nflvalue import ingest, explain"):
        assert pattern.search(bad), f"pattern misses {bad!r}"
    assert not pattern.search("import numpy as np")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_week_html_carries_banner_whynot_and_scoreboard(tbl, sim):
    e = _exp(tbl, sim)
    clv = pd.DataFrame({"count": [10], "mean": [0.1]}, index=["P4vP4"])
    html_text = explain.render_week_html(
        [e], clv_slices=clv, title="test week", generated_note="test")
    assert "PAPER MODE" in html_text
    assert "Why not" in html_text
    assert "Slice CLV scoreboard" in html_text
    assert "NOT RUN" in html_text
    assert "1-800-GAMBLER" in html_text
