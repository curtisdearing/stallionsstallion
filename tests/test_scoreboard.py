"""Weekly scoreboard — WEEKLY_SCOREBOARD_SPEC DoD tests.

The single most important test here is the architectural guarantee (§0/§4):
no code path from the recap or the hypothesis ledger back into the ratings,
simulation, or blend. The rest covers: grade-what-was-published, the noise
framing on a good week (tested, not eyeballed), ledger propose/revisit
mechanics, and pending entries never expiring.

Everything the recap reads is a `forecast_contract` v2 artifact, so the
fixtures here publish through the contract rather than hand-rolling a CSV.
That is not ceremony: the artifacts this module used to grade were two
schemas sharing one filename, and the tests that hand-rolled them were the
last place the ambiguity was still allowed to exist.
"""

import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nflvalue import forecast_contract as FC
from nflvalue import scoreboard as SB
from nflvalue import snapshots as SN


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
#: The week each producer route is registered for, the out-of-sample basis it
#: publishes against (data/forecast_sd_basis.json), and the spread of the
#: predictive distribution it produces. Both weeks are stocked below so each
#: route can be graded in a week its own registered evidence actually covers —
#: naming a weeks-0-3 basis on a week-10 row would be the extrapolation the
#: contract exists to refuse, even in a fixture. The simulated route has no
#: pooled SD in the manifest (it is per-game), so 16.0 is simply the width of
#: the sample these fixtures draw.
_ROUTES = {
    "ridge_preseason_prior": (2, "early_ridge_wk1_3_2021_2025", 16.8982),
    "kalman_montecarlo": (10, "sim_kalman_wk4_plus", 16.0),
}
_WEEKS = tuple(sorted(week for week, _, _ in _ROUTES.values()))
_KICKOFF = {2: "2024-09-07T18:00:00Z", 10: "2024-11-09T18:00:00Z"}


def _gid(week, i):
    """Game ids are unique across the sandbox's two weeks."""
    return week * 100 + i


#: Home wins by 10+i in both weeks, so a recap of one week is directly
#: comparable with a recap of the other.
_GAMES = pd.DataFrame([
    {"id": _gid(w, i), "season": 2024, "week": w, "home_team": f"H{i}",
     "away_team": f"A{i}", "start_date": _KICKOFF[w],
     "home_points": 30 + i, "away_points": 20,
     "neutral_site": False, "home_fcs": False, "away_fcs": False}
    for w in _WEEKS for i in range(1, 9)])

_LINES = pd.DataFrame([
    {"game_id": _gid(w, i), "season": 2024, "provider": "book",
     "spread": -(8.0 + i), "over_under": 55.0}
    for w in _WEEKS for i in range(1, 9)])

#: One fixed standardised sample (mean exactly 0) that the simulated route
#: shifts to each game's margin. Fixing it keeps the published SD and win
#: probability reproducible run to run, and centring it keeps the published
#: margin exactly the number the test asked for.
_SIM_DRAWS = np.random.default_rng(20241105).normal(
    0.0, _ROUTES["kalman_montecarlo"][2], 8000)
_SIM_DRAWS = _SIM_DRAWS - _SIM_DRAWS.mean()


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    d = tmp_path / "data"
    (d / "cfbd").mkdir(parents=True)
    from nflvalue import ingest
    monkeypatch.setattr(ingest, "load_games", lambda s: _GAMES)
    monkeypatch.setattr(ingest, "load_lines", lambda s: _LINES)
    monkeypatch.setattr(ingest, "load_coaches",
                        lambda s: (_ for _ in ()).throw(RuntimeError("none")))
    monkeypatch.setattr(ingest, "load_qb_starts",
                        lambda s: (_ for _ in ()).throw(RuntimeError("none")))
    monkeypatch.setattr(SB, "DATA", str(d))
    return d


def _provenance(family):
    return FC.Provenance(model_family=family,
                         model_version=f"{family}@test",
                         generated_at_utc="2024-11-05T12:00:00Z")


def _predictive(family, margin):
    """One predictive distribution, built the way that route builds it.

    The ridge route has a point estimate plus a registered residual SD and
    reads p(home win) off that normal; the simulated route reads mean, SD and
    p(home win) off one sample. Two genuinely different objects — which is the
    point: what reaches the artifact is one `projected_home_margin` either way.
    """
    _, basis, sd = _ROUTES[family]
    if family == "ridge_preseason_prior":
        return FC.PredictiveMargin.from_normal(margin, sd, basis)
    draws = _SIM_DRAWS + margin
    return FC.PredictiveMargin.from_samples(
        draws, basis, totals=np.full(len(draws), 52.0))


def _artifact(pred_fn, *, family="kalman_montecarlo", priced=None,
              refusals=None):
    """A valid v2 artifact over the sandbox schedule for this route's week.

    `pred_fn(i)` is the projected home margin for game i. Games left out of
    `priced` come back `refused`, because a public artifact is its own
    schedule: a row with no number is still published, it just says so.
    """
    week, _, _ = _ROUTES[family]
    schedule = _GAMES[_GAMES["week"] == week][
        ["id", "start_date", "home_team", "away_team", "neutral_site"]]
    wanted = set(range(1, 9) if priced is None else priced)
    forecasts = {_gid(week, i): FC.Forecast(
                     predictive=_predictive(family, pred_fn(i)))
                 for i in wanted}
    reasons = {_gid(week, i): reason
               for i, reason in (refusals or {}).items()}
    return FC.build_artifact(2024, week, schedule, forecasts, reasons,
                             _provenance(family))


def _publish(d, frame):
    """Publish through the contract writer, so nothing invalid reaches disk."""
    week = int(frame["week"].iloc[0])
    return FC.write_artifact(
        frame, str(d / f"predictions_2024_wk{week:02d}.csv"))


def _publish_raw(d, rows, week=10):
    """Write a file `FC.write_artifact` would have refused.

    Deliberately bypasses the contract writer: the test below is about an
    artifact that reached disk without passing through it, which is exactly
    how a leftover v1 file — or a v2 file something has written a v1 column
    back into — turns up in front of the recap.
    """
    pd.DataFrame(rows).to_csv(
        d / f"predictions_2024_wk{week:02d}.csv", index=False)


# --------------------------------------------------------------------------- #
# Grade what was published — never regenerate
# --------------------------------------------------------------------------- #
def test_recap_refuses_without_a_published_file(sandbox):
    with pytest.raises(SB.ScoreboardError, match="PUBLISHED"):
        SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))


def test_recap_grades_the_published_numbers(sandbox):
    _publish(sandbox, _artifact(lambda i: 8.0))       # published +8 everywhere
    r = SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))
    assert r["n_graded"] == 8
    # actual margins are 10+i; published 8 -> per-game error i+2
    assert r["model_mae"] == pytest.approx(np.mean([i + 2 for i in range(1, 9)]),
                                           abs=0.01)
    assert os.path.exists(sandbox / "recaps" / "2024_wk10.md")
    assert os.path.exists(sandbox / "recaps" / "2024_wk10.json")


def test_recap_excludes_refused_and_degraded_rows(sandbox):
    """Only `priced` and `frozen` rows may carry a number, so only they can
    reach the grading denominator. A `refused` row (the model declined this
    game) and a `degraded` row (the run could not stand behind a number) stay
    in the published feed — the artifact is its own schedule — and can never
    be graded. Under v1 this was a `veto` flag beside a free-text `note`,
    which made "declined" and "published a number that then went missing"
    look the same to a grader."""
    frame = _artifact(lambda i: 8.0, priced=(1,),
                      refusals={i: "unvalidated matchup" for i in range(2, 9)})
    degraded = frame["game_id"].isin([_gid(10, 7), _gid(10, 8)])
    frame.loc[degraded, "status"] = FC.Status.DEGRADED.value
    frame.loc[degraded, "refusal_reason"] = "simulation did not converge"
    _publish(sandbox, frame)

    published = FC.read_artifact(str(sandbox / "predictions_2024_wk10.csv"))
    assert len(published) == 8                 # every scheduled game published
    assert set(published["status"]) == {"priced", "refused", "degraded"}

    recap = SB.build_recap(2024, 10, data_dir=str(sandbox),
                           ledger_path=str(sandbox / "ledger.json"))

    # The denominator is the contract's own predicate, not "whatever parsed".
    bearing = {int(i) for i in FC.forecast_bearing(published)["game_id"]}
    assert bearing == {_gid(10, 1)}
    assert recap["n_graded"] == len(bearing) == 1
    assert {m["id"] for m in recap["misses"]} == bearing
    assert recap["model_mae"] == pytest.approx(3.0, abs=0.01)


def test_recap_writes_only_recaps_and_ledger(sandbox):
    _publish(sandbox, _artifact(lambda i: 8.0))
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
# One contract, two producer routes — the grader cannot tell them apart
# --------------------------------------------------------------------------- #
def test_both_producer_routes_grade_identically_through_one_field(sandbox):
    """The early-season ridge writer and the in-season simulation writer build
    their predictive distributions completely differently, and under v1 they
    published that difference as a difference in SCHEMA — `pred_margin` here,
    `mc_margin` there — which the grader coped with by preferring whichever
    column it found. Coping is what kept two kinds of number readable as one.

    Both routes now come through `forecast_contract.build_artifact`, so the
    only field grading reads is `projected_home_margin` and it means the same
    thing on both. The two artifacts below disagree about family, spread, win
    probability and whether a total exists at all, and grade identically."""
    assert SB.MODEL_MARGIN_FIELD == "projected_home_margin"
    assert not hasattr(SB, "MODEL_MARGIN_FIELDS"), (
        "a preference tuple is what let two kinds of number share one grader; "
        "there is exactly one published margin field now")

    _publish(sandbox, _artifact(lambda i: 8.0, family="ridge_preseason_prior"))
    _publish(sandbox, _artifact(lambda i: 8.0, family="kalman_montecarlo"))

    early_frame = FC.read_artifact(str(sandbox / "predictions_2024_wk02.csv"))
    late_frame = FC.read_artifact(str(sandbox / "predictions_2024_wk10.csv"))
    assert set(early_frame["model_family"]) == {"ridge_preseason_prior"}
    assert set(late_frame["model_family"]) == {"kalman_montecarlo"}
    # The ridge route runs no simulation, so it projects no total; and the two
    # routes' spreads (hence their win probabilities) are not the same number.
    assert early_frame["projected_total"].isna().all()
    assert late_frame["projected_total"].notna().all()
    assert not np.allclose(early_frame["forecast_margin_sd"],
                           late_frame["forecast_margin_sd"])
    assert not np.allclose(early_frame["p_home_win"], late_frame["p_home_win"])

    early = SB.build_recap(2024, 2, data_dir=str(sandbox),
                           ledger_path=str(sandbox / "ledger_early.json"))
    late = SB.build_recap(2024, 10, data_dir=str(sandbox),
                          ledger_path=str(sandbox / "ledger_late.json"))

    expected_mae = np.mean([i + 2 for i in range(1, 9)])
    assert early["n_graded"] == late["n_graded"] == 8
    assert early["model_mae"] == pytest.approx(expected_mae, abs=0.01)
    assert late["model_mae"] == early["model_mae"]
    assert late["model_mae_h2h"] == early["model_mae_h2h"]
    assert late["model_beat_market"] == early["model_beat_market"]
    assert ([m["published"] for m in late["misses"]]
            == [m["published"] for m in early["misses"]] == [8.0] * 5)
    assert ([m["error"] for m in late["misses"]]
            == [m["error"] for m in early["misses"]])


def test_recap_refuses_a_v1_or_mixed_schema_artifact_before_grading(
        sandbox, monkeypatch):
    """A v1 artifact — or a v2 one something has written a v1 column back
    into — is refused loudly, by name, before a single row is graded.

    This is where the old "no model margin field" test's coverage lands now.
    The failure it guarded against was a bare KeyError once something started
    reading a resolved margin column; the failure that matters now is worse
    and quieter — a file in which `mc_margin` and `pred_margin` are different
    kinds of number and `band_pm` is two different quantities, graded anyway
    under whichever reading happened to parse. So the refusal has to be typed,
    has to name the offending columns AND what replaced them, and has to
    happen before the results feed is touched."""
    from nflvalue import ingest
    monkeypatch.setattr(ingest, "load_games", lambda s: pytest.fail(
        "finals were loaded before the artifact was validated — a bad "
        "artifact must be refused before anything is graded"))

    v1_rows = [{"season": 2024, "week": 10, "id": _gid(10, i), "home": f"H{i}",
                "away": f"A{i}", "date": "2024-11-09", "neutral": False,
                "mc_margin": 8.0, "mc_margin_sd": 16.0, "mc_total": 52.0,
                "band_pm": 3.1, "p_home_win": 0.6, "veto": False,
                "note": "", "engine": "montecarlo"}
               for i in range(1, 9)]
    widened = _artifact(lambda i: 8.0)
    widened["pred_margin"] = widened["projected_home_margin"]

    for label, rows, banned in (
            ("a v1 artifact", v1_rows, ("mc_margin", "band_pm", "veto")),
            ("a v2 artifact widened back to v1", widened, ("pred_margin",))):
        _publish_raw(sandbox, rows)

        with pytest.raises(SB.ScoreboardError) as caught:
            SB.build_recap(2024, 10, data_dir=str(sandbox),
                           ledger_path=str(sandbox / "ledger.json"))

        message = str(caught.value)
        assert f"forecast contract v{FC.SCHEMA_VERSION}" in message, label
        assert "schema-version-1 columns present" in message, label
        for column in banned:
            assert column in message, f"{label}: does not name {column}"
        assert "projected_home_margin" in message, (
            f"{label}: names the banned column but not what replaced it")
        assert not os.path.exists(sandbox / "recaps"), (
            f"{label}: a recap was written for an artifact that was refused")


def test_recap_reads_the_snapshot_published_artifact_not_a_recomputed_forecast(
        sandbox):
    """The flat file grade_week reads is the verbatim copy of an immutable,
    digest-verified snapshot (per nflvalue.snapshots) -- never a live
    re-simulation. Build a real snapshot, publish it exactly as
    scripts/weekly_loop.py's _publish_week does (a straight copy of the
    newest snapshot's predictions.csv), and confirm grading reproduces
    THOSE numbers."""
    rows = _artifact(lambda i: 8.0)
    snap = SN.write_snapshot(
        str(sandbox), 2024, 10, "20240101T120000Z", rows,
        source_digest="d1", window="w",
        taken_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))
    assert SN.verify_snapshot(snap["path"]) is True

    published_path = sandbox / "predictions_2024_wk10.csv"
    pd.read_csv(os.path.join(snap["path"], "predictions.csv")).to_csv(
        published_path, index=False)

    r = SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))

    assert r["n_graded"] == 8
    assert r["model_mae"] == pytest.approx(np.mean([i + 2 for i in range(1, 9)]),
                                           abs=0.01)


# --------------------------------------------------------------------------- #
# The noise framing on a good week — tested, not eyeballed
# --------------------------------------------------------------------------- #
def test_good_week_renders_noise_framing(sandbox):
    # Published = exact actual margins: model MAE 0, crushes the market.
    _publish(sandbox, _artifact(lambda i: 10.0 + i))
    r = SB.build_recap(2024, 10, data_dir=str(sandbox),
                       ledger_path=str(sandbox / "ledger.json"))
    assert r["model_beat_market"] is True
    assert "NOISE" in r["noise_note"]
    md = open(sandbox / "recaps" / "2024_wk10.md").read()
    assert "NOISE" in md and "PAPER-ONLY" in md


def test_normal_week_states_the_baseline(sandbox):
    _publish(sandbox, _artifact(lambda i: -20.0))     # model far off
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
