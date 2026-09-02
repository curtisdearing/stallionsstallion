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
