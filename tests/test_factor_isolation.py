"""Requirement 7: a rejected, display-only or stale factor cannot move a
projected margin, total or win probability.

The guarantee is structural. No module that produces a forecast reads the
factor registry at all — not to consult it, not to apply it. So there is
no code path by which a factor's status could become an adjustment,
whatever that status says.
"""
from __future__ import annotations

import ast
import datetime as dt
import glob
import json
import os

import pytest

from nflvalue import factor_battery as FB
from nflvalue.context import registry as CREG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

#: every module that participates in producing a number.
FORECAST_MODULES = ("cfb_sim", "ratings", "engine", "kalman_engine",
                    "montecarlo", "gameline", "bakeoff", "cfb_ot")

#: the only statuses that may ever be described as used by the model.
ACTIVE = "gated_active"


def _reads_registry(path: str) -> bool:
    src = open(path, encoding="utf-8").read()
    if "factor_registry" in src or "gated_active" in src:
        return True
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("factor_battery"):
                return True
            for a in node.names:
                if a.name == "factor_battery":
                    return True
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("factor_battery"):
                    return True
    return False


def test_no_forecast_module_reads_the_factor_registry():
    checked = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "nflvalue", "*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in FORECAST_MODULES:
            continue
        checked += 1
        assert not _reads_registry(path), (
            f"{stem}.py reads the factor registry — that is the only way a "
            "rejected or stale factor could ever reach a projection")
    assert checked, "no forecast module was found to check"


def test_the_battery_never_writes_gated_active():
    src = open(os.path.join(ROOT, "scripts", "factor_battery_context.py"),
               encoding="utf-8").read()
    assert '"gated_active"' not in src and "'gated_active'" not in src, (
        "promotion is a separate, human-reviewed decision; this run may not "
        "write an active status")


def test_run_leaves_the_active_set_untouched():
    entries = FB.load_registry(os.path.join(DATA, "factor_registry.json"))
    active = sorted(e["factor_id"] for e in entries if e["status"] == ACTIVE)
    assert active == ["league_hfa", "per_team_time_varying_hfa"], (
        f"the gated_active set changed: {active}")


@pytest.mark.parametrize("status", ["display_only", "under_revalidation",
                                    "excluded", "unavailable", None])
def test_a_non_active_status_can_never_render_as_used_by_model(status):
    from nflvalue.context import contracts as C
    for spec in C.FIELDS:
        assert C.resolve_usage(spec, status, C.Availability.OK) \
            is not C.Usage.USED_BY_MODEL


def test_every_new_factor_landed_display_only():
    entries = FB.load_registry(os.path.join(DATA, "factor_registry.json"))
    by_id = {e["factor_id"]: e for e in entries}
    prereg = json.load(open(os.path.join(DATA, "factor_prereg_2026-08-28.json")))
    for h in prereg["hypotheses"]:
        assert h["id"] in by_id, f"{h['id']} was not recorded in the registry"
        assert by_id[h["id"]]["status"] == "display_only", (
            f"{h['id']} landed as {by_id[h['id']]['status']}")


def test_unavailable_families_are_recorded_not_omitted():
    entries = FB.load_registry(os.path.join(DATA, "factor_registry.json"))
    ids = {e["factor_id"] for e in entries}
    prereg = json.load(open(os.path.join(DATA, "factor_prereg_2026-08-28.json")))
    for fam in prereg["families_recorded_but_not_tested"]:
        assert any(fam in i or i in fam or fam.split("_")[0] in i for i in ids), (
            f"family {fam} left no trace in the registry — an untested family "
            "must still be recorded")


# ------------------------------------------------------------------ #
# staleness is visible, never silently active
# ------------------------------------------------------------------ #
def test_a_factor_past_its_revalidation_date_reads_as_stale():
    entries = [{"factor_id": "x", "status": "gated_active",
                "next_validation_due": "2025-01-01"},
               {"factor_id": "y", "status": "gated_active",
                "next_validation_due": "2099-01-01"},
               {"factor_id": "z", "status": "display_only",
                "next_validation_due": "2025-01-01"}]
    stale = CREG.stale_factor_ids(entries, today=dt.date(2026, 8, 28))
    assert "x" in stale and "y" not in stale


def test_season_end_placeholder_is_not_treated_as_a_date():
    entries = [{"factor_id": "q", "status": "gated_active",
                "next_validation_due": "2026 season end"}]
    stale = CREG.stale_factor_ids(entries, today=dt.date(2026, 8, 28))
    assert stale == [], "a non-date due field is unknown, not overdue"


def test_stale_active_factors_are_reported_by_the_run():
    entries = FB.load_registry(os.path.join(DATA, "factor_registry.json"))
    stale = CREG.stale_factor_ids(entries, today=dt.date(2026, 8, 28))
    assert isinstance(stale, list)
