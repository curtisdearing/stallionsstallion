"""The vault picks page: read-only over the artifact and the cards, every
FBS game and nothing else, intervals that are the contract's own
distribution, and byte-stable output."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import forecast_contract as FC  # noqa: E402


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RP = _load("render_picks", "scripts/render_picks.py")

FBS = {"Home U", "Away U", "Neutral A", "Neutral B", "Big State"}


def _artifact():
    rows = [
        {"game_id": 1, "home": "Home U", "away": "Away U", "neutral_site": False,
         "kickoff_utc": "2026-09-05T23:30:00Z", "status": "priced",
         "projected_home_margin": 7.0, "forecast_margin_sd": 16.0,
         "p_home_win": 0.6688, "model_family": "ridge_preseason_prior",
         "forecast_sd_basis": "early_ridge_wk1_3_2021_2025"},
        {"game_id": 2, "home": "Neutral A", "away": "Neutral B",
         "neutral_site": True, "kickoff_utc": "2026-09-05T20:00:00Z",
         "status": "priced", "projected_home_margin": -4.0,
         "forecast_margin_sd": 16.0, "p_home_win": 0.4013,
         "model_family": "ridge_preseason_prior",
         "forecast_sd_basis": "early_ridge_wk1_3_2021_2025"},
        {"game_id": 3, "home": "Big State", "away": "Tiny College",
         "neutral_site": False, "kickoff_utc": "2026-09-05T16:00:00Z",
         "status": "priced", "projected_home_margin": 30.0,
         "forecast_margin_sd": 17.9, "p_home_win": 0.9533,
         "model_family": "ridge_cross_division",
         "forecast_sd_basis": "cross_division_ridge_wk1_15_2021_2025"},
        {"game_id": 4, "home": "Tiny College", "away": "Other Tiny",
         "neutral_site": False, "kickoff_utc": "2026-09-05T17:00:00Z",
         "status": "refused", "refusal_reason": FC.DEFAULT_REFUSALS["non_fbs"],
         "model_family": "ridge_preseason_prior"},
        {"game_id": 5, "home": "Home U", "away": "Tiny College",
         "neutral_site": False, "kickoff_utc": "2026-09-04T00:00:00Z",
         "status": "refused",
         "refusal_reason": "REFUSED: no schema-v2 pregame snapshot existed before kickoff",
         "model_family": "ridge_preseason_prior"},
    ]
    frame = pd.DataFrame(rows)
    for col in ("generated_at_utc", "snapshot_id", "public_date"):
        frame[col] = {"generated_at_utc": "2026-09-02T00:00:00Z",
                      "snapshot_id": "S1", "public_date": "2026-09-05"}[col]
    frame["season"], frame["week"] = 2026, 1
    return frame


def _cards():
    return {
        1: {"why": [{"label": "Rating differential", "points": 4.0},
                    {"label": "Home-field effect", "points": 3.0}],
            "caution": {"text": "early"}},
        2: {"why": [{"label": "Rating differential", "points": -4.0}],
            "caution": {"text": "early"}},
        3: {"why": [{"label": "Rating differential (FCS side: pooled FCS pseudo-team)",
                     "points": 27.0},
                    {"label": "Home-field effect", "points": 3.0}],
            "caution": {"text": "pooled"}},
        4: {"caution": {"text": "non-fbs", "detail": "x"}},
        5: {"caution": {"text": "no record", "detail": "y"}},
    }


BASES = {
    "early_ridge_wk1_3_2021_2025": {"forecast_margin_sd": 16.8982, "n": 724},
    "cross_division_ridge_wk1_15_2021_2025": {
        "forecast_margin_sd": 17.8957, "n": 599, "mean_residual": -4.6397},
}


def test_every_fbs_game_and_only_fbs_games():
    picks = RP.build(_artifact(), _cards(), BASES, FBS)
    assert [p["game_id"] for p in picks] == [5, 3, 2, 1]   # kickoff order
    assert 4 not in {p["game_id"] for p in picks}          # FCS-vs-FCS is out


def test_pick_and_intervals_are_the_contracts_own_distribution():
    by_id = {p["game_id"]: p for p in RP.build(_artifact(), _cards(), BASES, FBS)}
    home = by_id[1]
    assert home["pick"] == "Home U" and home["p_pick"] == 0.669
    assert home["margin_pick"] == 7.0
    assert home["ci80_lo"] == round(7.0 - RP.Z80 * 16.0, 1)
    assert home["ci95_hi"] == round(7.0 + RP.Z95 * 16.0, 1)
    away = by_id[2]
    assert away["pick"] == "Neutral B"
    assert away["p_pick"] == round(1 - 0.4013, 3)
    assert away["margin_pick"] == 4.0
    assert away["ci80_lo"] == round(4.0 - RP.Z80 * 16.0, 1)


def test_cross_division_rationale_names_the_registered_bias():
    by_id = {p["game_id"]: p for p in RP.build(_artifact(), _cards(), BASES, FBS)}
    text = by_id[3]["rationale"]
    assert "pooled FCS pseudo-team" in text
    assert "4.6 pts" in text and "n=599" in text and "not corrected" in text


def test_refused_fbs_game_is_listed_with_its_reason_not_dropped():
    by_id = {p["game_id"]: p for p in RP.build(_artifact(), _cards(), BASES, FBS)}
    assert by_id[5]["pick"] is None
    assert by_id[5]["rationale"].startswith("Not forecast: REFUSED")


def test_a_card_missing_for_a_game_is_refused_loudly():
    cards = _cards()
    del cards[1]
    with pytest.raises(SystemExit, match="no insight card"):
        RP.build(_artifact(), cards, BASES, FBS)


def test_render_is_byte_stable_and_reads_no_clock(tmp_path):
    art = _artifact()
    picks = RP.build(art, _cards(), BASES, FBS)
    a = RP.render_markdown(picks, 2026, 1, art, BASES)
    b = RP.render_markdown(RP.build(art, _cards(), BASES, FBS), 2026, 1, art, BASES)
    assert a == b
    assert "Coverage:** 4 FBS games" in a
    assert "3 with a projected winner, 1 not forecast" in a
    md, csv_path = RP.write_outputs(picks, 2026, 1, str(tmp_path), a)
    assert Path(md).read_text() == a
    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == list(RP.CSV_COLUMNS)
    assert len(frame) == 4


def test_probability_format_never_prints_a_certainty():
    assert RP._fmt_p(0.9986) == "99.9%"
    assert RP._fmt_p(0.669) == "67%"
    assert RP._fmt_p(None) == ""


def _ledger():
    return {1: {"ats_side": "Away U +10.0", "ats_p": 0.72, "ats_p_adj": 0.66,
                "ats_line_home": -10.0, "ats_book": "Test", "ats_taken_at": "t"},
            3: {"ats_side": "Big State -20.0", "ats_p": 0.60, "ats_p_adj": 0.55,
                "ats_line_home": -20.0, "ats_book": "Test", "ats_taken_at": "t"}}


def test_compact_shows_the_spread_lean_only_when_it_is_the_more_confident_call():
    picks = RP.build(_artifact(), _cards(), BASES, FBS, _ledger())
    by_id = {p["game_id"]: RP.best_call(p) for p in picks}
    # game 1: straight-up 0.669 < spread 0.72 -> the spread lean, no flag
    assert by_id[1] == {"kind": "spread", "pick": "Away U +10.0", "p": 0.72,
                        "flag": False, "p_adj": 0.66}
    # game 3: straight-up 0.953 > spread 0.60 -> the winner, even with a line
    assert by_id[3]["kind"] == "straight_up" and by_id[3]["p"] == 0.953
    # game 2: no line at all -> the winner
    assert by_id[2]["kind"] == "straight_up"
    # refused game -> no call
    assert by_id[5] is None


def test_compact_flags_large_spread_leans_and_never_shows_a_total():
    ledger = _ledger()
    ledger[3]["ats_p"] = 0.99   # now the more confident call, on a 20-pt spread
    picks = RP.build(_artifact(), _cards(), BASES, FBS, ledger)
    text = RP.render_compact(picks, 2026, 1, _artifact())
    assert "| Big State -20.0 | 99%\\*" in text
    assert "spread leans sit on a market spread over 14" in text
    assert "Over/under: none" in text
    assert "over" not in text.lower().replace("over/under", "").replace("over 14", "")
    assert text == RP.render_compact(picks, 2026, 1, _artifact())   # byte-stable


def test_odds_arithmetic():
    assert RP.payout(-110) == pytest.approx(100 / 110)
    assert RP.payout(+250) == 2.5
    assert RP.implied(-110) == pytest.approx(0.5238, abs=1e-4)
    assert RP.implied(+250) == pytest.approx(1 / 3.5)
    # fair coin at even money is zero EV; 60% at -110 is positive
    assert RP.expected_value(0.5, +100) == 0
    assert RP.expected_value(0.6, -110) > 0


def test_value_page_drops_the_priced_in_favourite_and_cuts_the_long_shot():
    """Game 3: Big State 95% to win. At -10000 that is NO value; the dog at
    +2500 with a 4.7% model probability is positive-EV (4.7% * 25 > 95.3%)
    and is CUT anyway -- under 50% is not a pick. Neither side appears."""
    picks = RP.build(_artifact(), _cards(), BASES, FBS, {})
    markets = {3: {"provider": "Test", "spread": -30.0,
                   "home_ml": -10000.0, "away_ml": 2500.0},
               1: {"provider": "Test", "spread": -4.0,
                   "home_ml": -300.0, "away_ml": 240.0}}
    text = RP.render_value(picks, _artifact(), markets, "2026-09-02T01:03Z",
                           2026, 1)
    # The dog is positive-EV but a 4.7% chance: CUT by the >= 50% rule.
    assert "Tiny College ML +2500" not in text
    assert "Big State ML" not in text          # priced in: negative EV
    # game 1: home 66.9% at -300 (needs 75%) loses money; the away dog at
    # +240 is profitable but only 33% -- cut. The spread (model +7 against
    # a -4 line: 3 points of edge, ~57% to cover) is the only pick that
    # clears both bars; it is the model's own margin against the cached
    # line, so it is marked unfrozen.
    assert "Away U ML +240" not in text
    assert "Home U ML -300" not in text
    assert "(u) Home U -4.0" in text
    assert "Over/under: none" in text
    assert "Chance it hits" in text and "Needs to hit" in text


def test_value_page_prefers_the_frozen_ledger_spread_over_the_cache():
    picks = RP.build(_artifact(), _cards(), BASES, FBS, _ledger())
    markets = {1: {"provider": "Test", "spread": -3.0,
                   "home_ml": None, "away_ml": None}}
    plays = RP.value_plays(picks[-1], _artifact().set_index("game_id").loc[1]
                           .to_dict() | {"game_id": 1}, markets[1])
    spread = [x for x in plays if x["kind"] == "spread"][0]
    assert spread["frozen"] is True and spread["pick"] == "Away U +10.0"
    assert spread["p"] == 0.72 and spread["p_adj"] == 0.66


def test_scale_adjusted_ev_is_reported_beside_the_registered_one_never_instead():
    """Slope 0.8 / intercept 0: a registered +7 home margin re-scales to
    +8.75. The registered EV column is untouched; the adjusted one is
    computed from the re-scaled margin; the both-views section lists only
    plays positive under both."""
    picks = RP.build(_artifact(), _cards(), BASES, FBS, {})
    scale = {"slope": 0.8, "intercept": 0.0}
    row = _artifact().set_index("game_id").loc[1].to_dict() | {"game_id": 1}
    market = {"provider": "T", "spread": -7.0, "home_ml": -300.0, "away_ml": 240.0}
    plain = {x["pick"]: x for x in RP.value_plays(picks[-1], row, market)}
    adj = {x["pick"]: x for x in RP.value_plays(picks[-1], row, market, scale)}
    for pick in plain:
        assert adj[pick]["ev"] == plain[pick]["ev"]          # registered unchanged
    assert plain["Home U ML -300"]["ev_adj"] is None
    p_adj = RP._phi(8.75 / 16.0)
    assert adj["Home U ML -300"]["p_adj"] == pytest.approx(p_adj)
    assert adj["Home U ML -300"]["ev_adj"] == pytest.approx(
        RP.expected_value(p_adj, -300))
    text = RP.render_value(picks, _artifact(), {1: market}, "t", 2026, 1, scale)
    assert "## Picks that survive both views" in text
    assert "Same, scale-adjusted" in text


def test_under_fifty_percent_is_cut_whatever_the_payout():
    picks = RP.build(_artifact(), _cards(), BASES, FBS, {})
    markets = {2: {"provider": "T", "spread": None,
                   "home_ml": +300.0, "away_ml": -120.0}}
    # game 2: away 59.9% at -120 (needs 54.5%) -> a pick; home 40% at +300
    # would be the higher-EV play (0.4*3 - 0.6 = +0.6) but is under 50%.
    text = RP.render_value(picks, _artifact(), markets, "t", 2026, 1)
    assert "Neutral B ML -120" in text
    assert "Neutral A ML +300" not in text
    assert RP.MIN_PLAY_PROB == 0.5


def test_units_come_from_the_smaller_of_the_two_views_and_cap_at_three():
    assert RP.units_for({"ev": 0.90, "ev_adj": 0.45}) == 3.0
    assert RP.units_for({"ev": 0.90, "ev_adj": 0.30}) == 2.0
    assert RP.units_for({"ev": 0.12, "ev_adj": 0.60}) == 1.0
    assert RP.units_for({"ev": 0.05, "ev_adj": 0.04}) == 0.5
    assert RP.units_for({"ev": 0.50, "ev_adj": -0.01}) == 0.0   # pass
    assert RP.units_for({"ev": 0.50, "ev_adj": None}) == 3.0    # one view only
    assert max(u for _, u in RP.UNIT_LADDER) == 3.0


def test_value_page_groups_by_day_with_units():
    picks = RP.build(_artifact(), _cards(), BASES, FBS, {})
    markets = {1: {"provider": "T", "spread": -4.0, "home_ml": None, "away_ml": None}}
    text = RP.render_value(picks, _artifact(), markets, "t", 2026, 1,
                           {"slope": 0.8, "intercept": 0.0})
    assert "## Picks by day, sized in units" in text
    assert "### 2026-09-05 — " in text
    assert "| Units |" in text


# --------------------------------------------------------------------------- #
# The upcoming page: one call per game, sorted by date then confidence
# --------------------------------------------------------------------------- #

def _by_id(picks):
    return {p["game_id"]: p for p in picks}


def test_call_for_game_prefers_the_more_confident_of_spread_and_moneyline():
    picks = _by_id(RP.build(_artifact(), _cards(), BASES, FBS, _ledger()))
    row1 = _artifact().set_index("game_id").loc[1].to_dict() | {"game_id": 1}
    market1 = {"provider": "T", "spread": -4.0, "home_ml": -300.0, "away_ml": 240.0}
    # game 1: frozen spread lean 0.72 beats the ML win probability 0.669
    call1 = RP.call_for_game(picks[1], row1, market1)
    assert call1 == {"kind": "spread", "pick": "Away U +10.0", "p": 0.72,
                     "price": RP.SPREAD_PRICE, "flag": False, "frozen": True,
                     "has_market": True}

    row3 = _artifact().set_index("game_id").loc[3].to_dict() | {"game_id": 3}
    market3 = {"provider": "T", "spread": -20.0, "home_ml": -500.0,
              "away_ml": 400.0}
    # game 3: ML win probability 0.9533 beats the frozen spread lean 0.60,
    # and the underlying line is over the large-spread threshold either way
    call3 = RP.call_for_game(picks[3], row3, market3)
    assert call3["kind"] == "moneyline" and call3["pick"] == "Big State ML"
    assert call3["p"] == pytest.approx(0.9533)
    assert call3["flag"] is True


def test_call_for_game_falls_back_to_the_models_own_probability_with_no_market():
    picks = _by_id(RP.build(_artifact(), _cards(), BASES, FBS, {}))
    row2 = _artifact().set_index("game_id").loc[2].to_dict() | {"game_id": 2}
    call2 = RP.call_for_game(picks[2], row2, None)
    assert call2 == {"kind": "moneyline", "pick": "Neutral B ML",
                     "p": pytest.approx(0.599), "price": None, "flag": False,
                     "frozen": None, "has_market": False}


def test_call_for_game_is_none_for_a_refused_game():
    picks = _by_id(RP.build(_artifact(), _cards(), BASES, FBS, {}))
    row5 = _artifact().set_index("game_id").loc[5].to_dict() | {"game_id": 5}
    assert RP.call_for_game(picks[5], row5, None) is None


def test_render_upcoming_sorts_by_date_then_confidence_and_never_shows_a_total():
    picks = RP.build(_artifact(), _cards(), BASES, FBS, _ledger())
    markets = {1: {"provider": "T", "spread": -4.0, "home_ml": -300.0, "away_ml": 240.0},
              3: {"provider": "T", "spread": -20.0, "home_ml": -500.0, "away_ml": 400.0}}
    text = RP.render_upcoming(picks, _artifact(), markets, "t", 2026, 1,
                              days=1, today="2026-09-05")
    assert "Over/under: none" in text
    # highest confidence first within the one day in the fixture: game 3
    # (95%), then game 1's spread lean (72%), then game 2 (60%)
    i3, i1, i2 = text.index("Big State ML"), text.index("Away U +10.0"), \
        text.index("Neutral B ML")
    assert i3 < i1 < i2
    assert "\\* Big State ML" in text          # large-spread flag carries over
    assert "(u) Away U +10.0" not in text       # frozen spread: not tagged (u)
    assert "(no market) Neutral B ML" in text   # no line or price at all


def test_render_upcoming_window_excludes_games_outside_the_days_range():
    picks = RP.build(_artifact(), _cards(), BASES, FBS, {})
    text = RP.render_upcoming(picks, _artifact(), {}, "t", 2026, 1,
                              days=3, today="2026-09-01")
    assert "0 calls" in text
    assert "Home U" not in text and "Big State" not in text
