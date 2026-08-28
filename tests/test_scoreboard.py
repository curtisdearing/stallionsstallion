"""Weekly scoreboard — WEEKLY_SCOREBOARD_SPEC DoD tests.

The single most important test here is the architectural guarantee (§0/§4):
no code path from the recap or the hypothesis ledger back into the ratings,
simulation, or blend. The rest covers: grade-what-was-published, the noise
framing on a good week (tested, not eyeballed), ledger propose/revisit
mechanics, and pending entries never expiring.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from nflvalue import scoreboard as SB


# --------------------------------------------------------------------------- #
# The architectural guarantee
# --------------------------------------------------------------------------- #
_IMPORT_PAT = re.compile(
    r"^\s*(?:from\s+[.\w]*\bscoreboard\b|import\s+[\w.]*\bscoreboard\b"
    r"|from\s+[.\w]+\s+import\s+[^#\n]*\bscoreboard\b)", re.M)


def test_no_model_module_imports_the_scoreboard():
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "nflvalue")
    for mod in ("ratings.py", "cfb_sim.py", "cfb_ot.py", "gameline.py",
                "ingest.py", "montecarlo.py", "bakeoff.py"):
        src = open(os.path.join(root, mod)).read()
        assert not _IMPORT_PAT.search(src), (
            f"{mod} imports the scoreboard — the recap/ledger must have no "
            "path back into a rating, simulation, or blend")


def test_the_scoreboard_import_pattern_has_teeth():
    for bad in ("from . import scoreboard",
                "from .scoreboard import build_recap",
                "from nflvalue import ingest, scoreboard"):
        assert _IMPORT_PAT.search(bad), f"pattern misses {bad!r}"
    assert not _IMPORT_PAT.search("import numpy as np")


# --------------------------------------------------------------------------- #
# Fixtures: a sandboxed data dir with a PUBLISHED projections file
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    d = tmp_path / "data"
    (d / "cfbd").mkdir(parents=True)
    games = pd.DataFrame([
        {"id": i, "season": 2024, "week": 10, "home_team": f"H{i}",
         "away_team": f"A{i}", "home_points": 30 + i, "away_points": 20,
         "neutral_site": False, "home_fcs": False, "away_fcs": False}
        for i in range(1, 9)])
    lines = pd.DataFrame([
        {"game_id": i, "season": 2024, "provider": "book",
         "spread": -(8.0 + i), "over_under": 55.0}
        for i in range(1, 9)])
    from nflvalue import ingest
    monkeypatch.setattr(ingest, "load_games", lambda s: games)
    monkeypatch.setattr(ingest, "load_lines", lambda s: lines)
    monkeypatch.setattr(ingest, "load_coaches",
                        lambda s: (_ for _ in ()).throw(RuntimeError("none")))
    monkeypatch.setattr(ingest, "load_qb_starts",
                        lambda s: (_ for _ in ()).throw(RuntimeError("none")))
    monkeypatch.setattr(SB, "DATA", str(d))
    return d


def _publish(d, preds):
    pd.DataFrame(preds).to_csv(d / "predictions_2024_wk10.csv", index=False)


def _pub_rows(pred_fn):
    return [{"season": 2024, "week": 10, "id": i, "home": f"H{i}",
             "away": f"A{i}", "mc_margin": pred_fn(i),
             "mc_margin_sd": 16.0, "mc_total": 52.0, "p_home_win": 0.6}
            for i in range(1, 9)]


# --------------------------------------------------------------------------- #
# Grade what was published — never regenerate
# --------------------------------------------------------------------------- #
def test_recap_refuses_without_a_published_file(sandbox):
    with pytest.raises(SB.ScoreboardError, match="PUBLISHED"):
        SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))


def test_recap_grades_the_published_numbers(sandbox):
    _publish(sandbox, _pub_rows(lambda i: 8.0))       # published +8 everywhere
    r = SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))
    assert r["n_graded"] == 8
    # actual margins are 10+i; published 8 -> per-game error i+2
    assert r["model_mae"] == pytest.approx(np.mean([i + 2 for i in range(1, 9)]),
                                           abs=0.01)
    assert os.path.exists(sandbox / "recaps" / "2024_wk10.md")
    assert os.path.exists(sandbox / "recaps" / "2024_wk10.json")


def test_recap_excludes_vetoed_and_unpriced_rows(sandbox):
    rows = _pub_rows(lambda i: 8.0)[:3]
    rows[0]["veto"] = False
    rows[1].update({"veto": True, "note": "unvalidated matchup",
                    "mc_margin": None, "mc_margin_sd": None,
                    "mc_total": None, "p_home_win": None})
    rows[2].update({"veto": False, "note": "missing model output",
                    "mc_margin": None, "mc_margin_sd": None,
                    "mc_total": None, "p_home_win": None})
    _publish(sandbox, rows)

    recap = SB.build_recap(2024, 10, data_dir=str(sandbox),
                           ledger_path=str(sandbox / "ledger.json"))

    assert recap["n_graded"] == 1
    assert recap["model_mae"] == pytest.approx(3.0, abs=0.01)
    assert [m["id"] for m in recap["misses"]] == [1]


def test_recap_writes_only_recaps_and_ledger(sandbox):
    _publish(sandbox, _pub_rows(lambda i: 8.0))
    before = {}
    for root, _, files in os.walk(sandbox):
        if "recaps" in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            before[p] = os.path.getmtime(p)
    SB.build_recap(2024, 10, data_dir=str(sandbox),
                   ledger_path=str(sandbox / "ledger.json"))
    for path, mtime in before.items():
        if path.endswith("ledger.json"):
            continue
        assert os.path.getmtime(path) == mtime, (
            f"{path} was modified by the recap — the scoreboard's write "
            "surface is data/recaps/ and the ledger, nothing else")


# --------------------------------------------------------------------------- #
# The noise framing on a good week — tested, not eyeballed
# --------------------------------------------------------------------------- #
def test_good_week_renders_noise_framing(sandbox):
    # Published = exact actual margins: model MAE 0, crushes the market.
    _publish(sandbox, _pub_rows(lambda i: 10.0 + i))
    r = SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))
    assert r["model_beat_market"] is True
    assert "NOISE" in r["noise_note"]
    md = open(sandbox / "recaps" / "2024_wk10.md").read()
    assert "NOISE" in md and "PAPER-ONLY" in md


def test_normal_week_states_the_baseline(sandbox):
    _publish(sandbox, _pub_rows(lambda i: -20.0))     # model far off
    r = SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))
    assert r["model_beat_market"] is False
    assert "baseline" in r["noise_note"]


# --------------------------------------------------------------------------- #
# Hypothesis ledger mechanics
# --------------------------------------------------------------------------- #
def _misses(flags):
    return pd.DataFrame({
        "id": range(1, len(flags) + 1),
        "first_season_coach": flags,
        "qb_change": [False] * len(flags)})


def test_hypothesis_proposed_when_pattern_marks_two_misses():
    entries = SB.propose_hypotheses(_misses([True, True, False, False, False]),
                                    2026, 3)
    assert len(entries) == 1
    assert entries[0]["pattern"] == "first_season_coach"
    assert entries[0]["status"] == "pending"


def test_no_hypothesis_from_a_single_hit():
    assert SB.propose_hypotheses(_misses([True, False, False, False, False]),
                                 2026, 3) == []


def test_revisits_resolve_after_min_observations():
    e = {"id": "x", "pattern": "first_season_coach", "status": "pending",
         "revisits": []}
    ledger = [e]
    for wk in (4, 5, 6):     # three supporting weeks
        SB.revisit_ledger(ledger, _misses([True, True, True, False, False]),
                          2026, wk)
    assert e["status"] == "supported"
    assert len(e["revisits"]) == 3


def test_contradicted_after_absent_weeks():
    e = {"id": "x", "pattern": "first_season_coach", "status": "pending",
         "revisits": []}
    for wk in (4, 5, 6):     # pattern never recurs
        SB.revisit_ledger([e], _misses([False] * 5), 2026, wk)
    assert e["status"] == "contradicted"


def test_pending_entries_never_expire():
    e = {"id": "x", "pattern": "first_season_coach", "status": "pending",
         "revisits": []}
    # one ambiguous week (exactly 1 hit): no observation recorded, stays open
    SB.revisit_ledger([e], _misses([True, False, False, False, False]),
                      2026, 4)
    assert e["status"] == "pending"
    assert e["revisits"] == []
