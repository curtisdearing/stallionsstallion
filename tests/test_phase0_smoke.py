"""Offline tests for the Phase 0 CFBD smoke test.

These verify the *validation logic* without a key or network, so the smoke
test can't silently rot before someone runs it live. They also pin the two
field-shape facts most likely to break: camelCase vs snake_case tolerance,
and legitimately-null PPA on some plays.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import phase0_smoke_test as p0  # noqa: E402


def _fixtures():
    path = os.path.join(ROOT, "tests", "fixtures", "cfbd_smoke_fixtures.json")
    with open(path) as fh:
        return json.load(fh)


def test_selftest_runs_and_passes():
    report = p0.run_selftest()
    assert report["verdict"] in ("PASS", "PARTIAL")
    assert report["calls_used"] == 0                 # offline: zero budget spent


def test_pick_tolerates_both_case_styles():
    assert p0.pick({"homeTeam": "X"}, "homeTeam", "home_team") == "X"
    assert p0.pick({"home_team": "Y"}, "homeTeam", "home_team") == "Y"
    assert p0.pick({}, "a", "b", default="z") == "z"


def test_plays_validator_flags_missing_ppa():
    findings = p0.validate_plays([
        {"down": 1, "distance": 10, "yardsToGoal": 75, "playType": "Rush"},
        {"down": 2, "distance": 5, "yardsToGoal": 70, "playType": "Pass"},
        {"down": 3, "distance": 5, "yardsToGoal": 70, "playType": "Punt"},
    ])
    ppa = [f for f in findings if f[0] == "PPA present"][0]
    assert ppa[1] is False        # 0% PPA -> hard fail, not a warn


def test_lines_validator_detects_single_book():
    findings = p0.validate_lines(_fixtures()["lines"])
    multi = [f for f in findings if f[0] == "multi-book coverage"][0]
    assert "1/2" in multi[2]      # fixture: 1 of 2 games has >=2 books


def test_empty_endpoints_fail_loud():
    assert p0.verdict(p0.validate_plays([])) == "FAIL"
    assert p0.verdict(p0.validate_lines([])) == "FAIL"


def test_resolve_api_key_no_key_returns_empty(monkeypatch):
    from nflvalue.sources import cfbd
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    # should never raise even with no key configured
    assert isinstance(cfbd.resolve_api_key(), str)
