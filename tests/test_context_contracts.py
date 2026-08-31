"""Source-contract tests for the pregame-context layer (written BEFORE the
implementation, per the project's test-first rule and BUILD_SPEC section 0).

What these lock down:
  * every context value carries a full provenance envelope;
  * `data/factor_registry.json` is the ONLY thing that decides whether a
    factor is excluded / display-only / under revalidation / active;
  * the five standing exclusions stay excluded;
  * an unapproved source cannot emit a value, whatever code exists for it;
  * missing / stale / conflicting / low-coverage data renders VISIBLY as
    unavailable and is never imputed as a number.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from nflvalue.context import contracts as C
from nflvalue.context import registry as REG
from nflvalue.context import sources as SRC
from nflvalue.context import assemble as ASM

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KICK = dt.datetime(2026, 8, 29, 23, 0, tzinfo=dt.timezone.utc)
BEFORE = dt.datetime(2026, 8, 28, 3, 55, tzinfo=dt.timezone.utc)

# --------------------------------------------------------------------- #
# fixtures — synthetic frames, so these tests never need the real cache
# --------------------------------------------------------------------- #
from nflvalue.context import fixtures as FX  # noqa: E402


@pytest.fixture
def demo_game():
    return FX.demo_game()


@pytest.fixture
def demo_frames():
    return FX.demo_frames()


@pytest.fixture
def demo_context(demo_game, demo_frames):
    return ASM.assemble_game_context(
        game=demo_game, frames=demo_frames, data_dir=DATA,
        now=dt.datetime(2026, 8, 28, 12, 0, tzinfo=dt.timezone.utc))


PROVENANCE_FIELDS = ("source", "source_url_or_id", "observed_at",
                     "effective_for_game", "confidence", "coverage")


# --------------------------------------------------------------------- #
# the manifest itself
# --------------------------------------------------------------------- #
def test_manifest_parses_and_declares_every_required_key():
    man = SRC.SourceManifest.load(DATA)
    assert man.sources, "manifest must declare sources"
    for name, s in man.sources.items():
        for key in ("provider", "external", "approval", "serves",
                    "max_age_hours", "min_coverage"):
            assert key in s, f"{name} is missing contract key {key!r}"
        appr = s["approval"]
        assert appr["status"] in SRC.APPROVAL_STATES, f"{name}: bad approval state"
        assert appr.get("decision_record"), f"{name}: approval needs a decision record"


def test_approved_sources_name_an_origin_and_gated_ones_do_not():
    man = SRC.SourceManifest.load(DATA)
    for name, s in man.sources.items():
        if man.is_approved(name):
            assert s.get("loader") or s.get("source_url_or_id"), (
                f"{name} is approved but names no loader and no source id")
        else:
            assert not s.get("loader"), (
                f"{name} is not approved yet declares a live loader — an "
                "unapproved source must not look runnable")


def test_new_external_provider_needs_a_documented_approval_gate():
    man = SRC.SourceManifest.load(DATA)
    for name, s in man.sources.items():
        if s["external"] and not man.is_approved(name):
            assert s.get("requires"), (
                f"{name} is an unapproved external provider and must list what "
                "an approval would require")
            assert len(s["approval"]["decision_record"]) > 80, (
                f"{name}: the gate must be documented, not a one-liner")


def test_an_unapproved_source_can_never_emit_a_value():
    man = SRC.SourceManifest.load(DATA)
    for name in man.sources:
        if man.is_approved(name):
            continue
        v = C.ContextValue.unavailable(
            spec=C.FIELDS_BY_SOURCE[name][0],
            availability=C.Availability.UNAVAILABLE_NOT_APPROVED,
            note="test", registry_status=None)
        assert v.value is None
        assert v.usage is C.Usage.MISSING


# --------------------------------------------------------------------- #
# provenance envelope
# --------------------------------------------------------------------- #
def test_available_values_carry_the_full_provenance_envelope(demo_context):
    for v in demo_context.values:
        if v.availability is C.Availability.OK:
            for f in PROVENANCE_FIELDS:
                assert getattr(v, f) is not None, (
                    f"{v.key}: {f} is required when available")


def test_provenance_is_mandatory_at_construction():
    spec = C.FIELDS[0]
    with pytest.raises(C.ContextContractError):
        C.ContextValue.available(spec=spec, value=1.0, source="x",
                                 source_url_or_id=None, observed_at=BEFORE,
                                 effective_for_game=KICK, confidence="high",
                                 coverage=1.0, registry_status=None)


def test_unavailable_values_carry_no_number_at_all():
    spec = C.FIELDS[0]
    v = C.ContextValue.unavailable(spec=spec,
                                   availability=C.Availability.UNAVAILABLE_NO_DATA,
                                   note="nothing cached", registry_status=None)
    assert v.value is None and v.display.lower().startswith("unavailable")


# --------------------------------------------------------------------- #
# the factor registry is the single source of truth
# --------------------------------------------------------------------- #
def test_registry_loads_and_covers_the_preserved_exclusions():
    reg = REG.FactorRegistry.load(DATA)
    for fid in REG.PRESERVED_EXCLUSIONS:
        assert reg.status(fid) == "excluded", f"{fid} must stay excluded"


def test_preserved_exclusions_never_emit_a_value(demo_context):
    by_factor = {v.registry_factor_id: v for v in demo_context.values
                 if v.registry_factor_id}
    for fid in REG.PRESERVED_EXCLUSIONS:
        if fid in by_factor:
            v = by_factor[fid]
            assert v.usage is C.Usage.EXCLUDED
            assert v.value is None
    rendered = {r["factor_id"] for r in demo_context.excluded_rows}
    assert set(REG.PRESERVED_EXCLUSIONS) <= rendered, (
        "every standing exclusion must be rendered as an explicit excluded row")


def test_usage_bucket_is_decided_only_by_the_registry(monkeypatch):
    reg = REG.FactorRegistry.load(DATA)
    spec = next(s for s in C.FIELDS if s.registry_factor_id == "finishing_drives_diff")
    assert reg.status(spec.registry_factor_id) == "display_only"
    usage = C.resolve_usage
    assert usage(spec, "display_only", C.Availability.OK) is C.Usage.DISPLAY_ONLY
    assert usage(spec, "excluded", C.Availability.OK) is C.Usage.EXCLUDED
    assert usage(spec, "under_revalidation",
                 C.Availability.OK) is C.Usage.DISPLAY_ONLY


def test_a_display_only_factor_can_never_be_marked_used_by_model():
    spec = next(s for s in C.FIELDS if s.registry_factor_id == "finishing_drives_diff")
    for status in ("display_only", "under_revalidation", "excluded", None):
        assert C.resolve_usage(spec, status,
                               C.Availability.OK) is not C.Usage.USED_BY_MODEL


def test_used_by_model_needs_a_gated_active_registry_entry():
    for spec in C.FIELDS:
        for status in ("display_only", "under_revalidation", "excluded", None):
            assert C.resolve_usage(
                spec, status, C.Availability.OK) is not C.Usage.USED_BY_MODEL
    active = [s for s in C.FIELDS if s.model_consumed]
    assert active, "at least one field should describe something the model does use"
    for spec in active:
        assert C.resolve_usage(
            spec, "gated_active", C.Availability.OK) is C.Usage.USED_BY_MODEL


def test_unregistered_field_is_display_only_never_used():
    spec = next(s for s in C.FIELDS if s.registry_factor_id is None
                and not s.model_consumed)
    assert C.resolve_usage(spec, None, C.Availability.OK) is C.Usage.DISPLAY_ONLY


# --------------------------------------------------------------------- #
# degraded data renders, it does not disappear and it is not imputed
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("availability", [
    C.Availability.UNAVAILABLE_NO_SOURCE,
    C.Availability.UNAVAILABLE_NOT_APPROVED,
    C.Availability.UNAVAILABLE_NO_DATA,
    C.Availability.STALE,
    C.Availability.CONFLICTING,
    C.Availability.LOW_COVERAGE,
    C.Availability.REFUSED_POST_KICKOFF,
])
def test_degraded_states_render_as_missing_with_a_reason(availability):
    spec = C.FIELDS[0]
    v = C.ContextValue.unavailable(spec=spec, availability=availability,
                                   note="because", registry_status=None)
    assert v.value is None
    assert v.usage is C.Usage.MISSING
    assert v.note, "a degraded value must say why"
    assert availability.value in v.display or "unavailable" in v.display.lower()


def test_stale_observation_is_flagged_not_used():
    man = SRC.SourceManifest.load(DATA)
    old = KICK - dt.timedelta(hours=999)
    assert SRC.freshness(man, "cfbd_games", observed_at=old, now=KICK) \
        is C.Availability.STALE
    assert SRC.freshness(man, "cfbd_games", observed_at=KICK - dt.timedelta(hours=1),
                         now=KICK) is C.Availability.OK


def test_low_coverage_is_flagged_not_used():
    man = SRC.SourceManifest.load(DATA)
    assert SRC.coverage_state(man, "cfbd_talent", 0.10) is C.Availability.LOW_COVERAGE
    assert SRC.coverage_state(man, "cfbd_talent", 0.99) is C.Availability.OK


def test_conflicting_sources_render_as_conflict():
    st = C.reconcile([("a", 3.0), ("b", 3.0)], tolerance=0.1)
    assert st[0] is C.Availability.OK and st[1] == 3.0
    st = C.reconcile([("a", 3.0), ("b", 9.0)], tolerance=0.1)
    assert st[0] is C.Availability.CONFLICTING and st[1] is None


# --------------------------------------------------------------------- #
# nothing is silently dropped
# --------------------------------------------------------------------- #
def test_every_declared_field_appears_in_the_context(demo_context):
    assert {v.key for v in demo_context.values} == {s.key for s in C.FIELDS}


def test_every_declared_family_is_represented(demo_context):
    assert {v.family for v in demo_context.values} == set(C.FAMILIES)


def test_context_is_serializable_with_its_provenance(demo_context):
    blob = json.dumps(demo_context.to_dict(), default=str)
    assert '"observed_at"' in blob and '"source_url_or_id"' in blob
