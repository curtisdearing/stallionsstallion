"""The observation contract, the measure registry, and the retrospective block.

Fixtures are defined locally. `tests/conftest.py` is a docstring-only stub
inherited from fablesfable and offers nothing usable here.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from nflvalue.context.contracts import Availability
from nflvalue.team_evidence import contracts as C
from nflvalue.team_evidence import news as NEWS
from nflvalue.team_evidence.registry import (FORBIDDEN_ID_TOKENS, MeasureRegistry,
                                             RegistryError)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NOW = dt.datetime(2026, 9, 9, 12, 0, tzinfo=dt.timezone.utc)
KICK = dt.datetime(2026, 9, 6, 23, 0, tzinfo=dt.timezone.utc)


def _present(**over):
    kw = dict(
        measure_id="rush_success_rate_off", team="Alabama", season=2026,
        as_of_week=3, label="Rush success rate (created)", unit="rate",
        value=0.44, n=120, n_games=2,
        availability_class=C.AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR,
        source_id="model_drive_features", source_url_or_id="internal:x",
        source_version="sha256:abcdef0123456789", retrieved_at=NOW,
        effective_through=KICK)
    kw.update(over)
    return C.Observation.present(**kw)


# ------------------------------------------------------------------ vocabulary
def test_every_context_availability_value_is_representable():
    """One vocabulary across the two layers, or a consumer comparing strings gets
    a silently wrong answer."""
    for av in Availability:
        assert C.EvidenceAvailability(av.value).value == av.value
        assert C.EvidenceAvailability.from_context(av).value == av.value


def test_the_two_extra_states_are_not_in_the_context_vocabulary():
    extra = {C.EvidenceAvailability.INSUFFICIENT_SAMPLE,
             C.EvidenceAvailability.BLOCKED_UNVERIFIED_AVAILABILITY}
    assert not {e.value for e in extra} & {a.value for a in Availability}


# ------------------------------------------------------------------ the ctors
def test_a_present_observation_demands_the_whole_provenance_envelope():
    for field in ("source_id", "source_url_or_id", "source_version",
                  "retrieved_at", "effective_through", "value"):
        with pytest.raises(C.EvidenceContractError):
            _present(**{field: None})


def test_a_naive_timestamp_is_refused():
    with pytest.raises(C.EvidenceContractError):
        _present(retrieved_at=dt.datetime(2026, 9, 9, 12, 0))


def test_a_missing_observation_cannot_carry_a_number():
    obs = C.Observation.missing(
        measure_id="drives_per_game", team="Alabama", season=2026, as_of_week=3,
        label="Drives per game", unit="drives",
        availability=C.EvidenceAvailability.UNAVAILABLE_NOT_APPROVED,
        missingness=C.Missingness.SOURCE_NOT_APPROVED, reason="pending approval")
    assert obs.value is None
    assert not obs.is_present
    assert "value" not in {k for k, v in obs.to_dict().items() if v not in (None,)} \
        or obs.to_dict()["value"] is None


def test_a_missing_observation_may_not_claim_to_be_ok():
    with pytest.raises(C.EvidenceContractError):
        C.Observation.missing(
            measure_id="x", team="A", season=2026, as_of_week=1, label="x",
            unit=None, availability=C.EvidenceAvailability.OK,
            missingness=C.Missingness.NO_OBSERVATIONS, reason="r")


def test_a_missing_observation_needs_a_reason():
    with pytest.raises(C.EvidenceContractError):
        C.Observation.missing(
            measure_id="x", team="A", season=2026, as_of_week=1, label="x",
            unit=None, availability=C.EvidenceAvailability.UNAVAILABLE_NO_DATA,
            missingness=C.Missingness.NO_OBSERVATIONS, reason="")


def test_opponent_adjustment_is_both_or_neither():
    with pytest.raises(C.EvidenceContractError):
        _present(opponent_adjusted=True)
    with pytest.raises(C.EvidenceContractError):
        _present(opponent_adjusted=False, opponent_adjustment_method="ridge")


def test_negative_denominators_are_refused():
    with pytest.raises(C.EvidenceContractError):
        _present(n=-1)


def test_there_is_no_third_constructor():
    """Anything that could default-fill a value would have to be a third path."""
    ctors = [n for n in dir(C.Observation)
             if not n.startswith("_") and isinstance(
                 getattr(C.Observation, n, None), type(C.Observation.present))]
    assert set(ctors) == {"present", "missing"}


# ------------------------------------------------------- retrospective block
def test_a_reconstructed_observation_cannot_back_a_predictive_claim():
    obs = _present()
    assert obs.availability_class is C.AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR
    for kind in C.PREDICTIVE_CLAIM_KINDS:
        with pytest.raises(C.RetrospectiveClaimBlocked):
            C.assert_retrospective_claim_allowed(obs, kind)


def test_a_reconstructed_observation_may_still_describe():
    obs = _present()
    for kind in C.DESCRIPTIVE_CLAIM_KINDS:
        C.assert_retrospective_claim_allowed(obs, kind)


def test_a_proven_pregame_observation_may_back_a_predictive_claim():
    obs = _present(availability_class=C.AvailabilityClass.PROVEN_PREGAME)
    C.assert_retrospective_claim_allowed(obs, "predictive")


def test_an_unknown_claim_kind_raises_rather_than_defaulting_to_allowed():
    with pytest.raises(C.EvidenceContractError):
        C.assert_retrospective_claim_allowed(_present(), "vibes")


def test_the_block_has_no_override_flag():
    import inspect
    sig = inspect.signature(C.assert_retrospective_claim_allowed)
    assert list(sig.parameters) == ["obs", "claim_kind"]


# ----------------------------------------------------------------- registry
def test_the_shipped_registry_loads():
    reg = MeasureRegistry.load(DATA)
    assert len(reg.measures) >= 50
    assert reg.emittable() and reg.blocked()


def test_every_blocked_measure_names_a_reason():
    reg = MeasureRegistry.load(DATA)
    for spec in reg.blocked():
        assert spec.blocked_reason and len(spec.blocked_reason) > 30, spec.measure_id


def test_registry_names_do_not_overclaim():
    """A sack rate is not a pressure rate; capacity is not crowd noise; a long
    gain is not a broken assignment; a record is not a line ranking."""
    reg = MeasureRegistry.load(DATA)
    for spec in reg.measures.values():
        blob = f"{spec.measure_id} {spec.label}".lower()
        for token in FORBIDDEN_ID_TOKENS:
            assert token not in blob, f"{spec.measure_id} uses {token!r}"


def test_the_registry_refuses_an_overclaiming_measure(tmp_path):
    doc = {
        "schema": "stallionsstallion.team_evidence_registry.v1",
        "measures": [{
            "measure_id": "pressure_rate_def", "label": "Pressure rate",
            "family": "line_proxy", "side": "allowed", "definition": "sacks/dropback",
            "denominator": "dropbacks", "source_id": "model_drive_features",
            "status": "display_only", "availability_class": "reconstructed_week_granular",
        }]}
    (tmp_path / "team_evidence_registry.json").write_text(json.dumps(doc))
    with pytest.raises(RegistryError, match="pressure"):
        MeasureRegistry.load(str(tmp_path))


def test_the_registry_refuses_a_blocked_measure_with_no_reason(tmp_path):
    doc = {
        "schema": "stallionsstallion.team_evidence_registry.v1",
        "measures": [{
            "measure_id": "seconds_per_play", "label": "Seconds per play",
            "family": "pace", "side": "team", "definition": "BLOCKED",
            "denominator": "n/a", "source_id": "model_drive_features",
            "status": "blocked", "availability_class": "unverifiable",
        }]}
    (tmp_path / "team_evidence_registry.json").write_text(json.dumps(doc))
    with pytest.raises(RegistryError, match="names no reason"):
        MeasureRegistry.load(str(tmp_path))


def test_every_emittable_measure_names_a_declared_source():
    """`data/context_sources.json` is the only place a source may be declared. A
    measure that can emit must name one that is there."""
    from nflvalue.context.sources import SourceManifest
    reg = MeasureRegistry.load(DATA)
    man = SourceManifest.load(DATA)
    for spec in reg.emittable():
        man.get(spec.source_id)          # raises if undeclared


def test_an_undeclared_source_can_only_back_blocked_measures():
    """The four drive measures point at `cfbd_drives`, which is deliberately NOT
    declared yet — declaring it is Curtis's decision and it belongs in the context
    layer's own manifest with a provider. Until then those measures must be
    blocked, and the block must say how to lift it."""
    from nflvalue.context.sources import SourceError, SourceManifest
    reg = MeasureRegistry.load(DATA)
    man = SourceManifest.load(DATA)
    undeclared = set()
    for sid in reg.source_ids():
        try:
            man.get(sid)
        except SourceError:
            undeclared.add(sid)
    assert "cfbd_drives" in undeclared, "the drives gap is a documented finding"
    for spec in reg.measures.values():
        if spec.source_id in undeclared:
            assert spec.is_blocked, f"{spec.measure_id} reads an undeclared source"
            assert "UNBLOCK" in (spec.blocked_reason or ""), \
                f"{spec.measure_id} is blocked without saying how to lift it"


def test_the_proxy_measures_say_proxy():
    reg = MeasureRegistry.load(DATA)
    for spec in reg.measures.values():
        if "stuff" in spec.measure_id:
            assert "proxy" in spec.measure_id


def test_only_the_existing_adjustment_is_named_as_an_adjustment():
    """No second rating engine. Every adjusted measure must cite solve_units."""
    reg = MeasureRegistry.load(DATA)
    adjusted = [s for s in reg.measures.values() if s.opponent_adjusted]
    assert adjusted
    for spec in adjusted:
        assert "solve_units" in (spec.opponent_adjustment_method or "")


def test_attention_signals_are_refused_as_measures():
    reg = MeasureRegistry.load(DATA)
    for spec in reg.measures.values():
        NEWS.assert_not_an_attention_signal(spec.measure_id)
    for bad in ("article_count", "ap_rank", "media_attention", "mention volume"):
        with pytest.raises(NEWS.NewsContractError):
            NEWS.assert_not_an_attention_signal(bad)
