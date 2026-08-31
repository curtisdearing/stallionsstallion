"""Snapshot tests for the insight card: one forecast, four sections, no prose.

Every card is asserted against an exact expected shape rather than a set of
"contains" checks, because the failure this surface is replacing was not a
missing field — it was a surface that kept accreting fields until the forecast
was buried, and each addition looked individually reasonable. A snapshot test
makes an addition visible as a diff and forces someone to justify it.

The seven cases the layer has to get right: priced, refused, frozen, degraded,
no market reference, an available market reference, and a context field whose
source is missing.
"""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import insight as IN  # noqa: E402
from nflvalue.context import contracts as CTX  # noqa: E402

UTC = dt.timezone.utc

PROVENANCE = {
    "model_family": "kalman_montecarlo",
    "model_version": "fp-kalman-0001",
    "generated_at_utc": "2026-10-14T12:00:00Z",
    "snapshot_id": "20261014T120000Z",
    "source_digest": "digest-abc123",
    "config_hash": "cfg-9",
}
DATA_AS_OF = "2026-10-13T09:00:00Z"


def _row(**overrides):
    """A valid contract row. Overrides win; nothing else is invented."""
    row = {column: None for column in FC.COLUMNS}
    row.update({
        "schema_version": FC.SCHEMA_VERSION,
        "season": 2026, "week": 8, "game_id": 401900101,
        "home": "Ohio State", "away": "Michigan",
        "kickoff_utc": "2026-10-17T16:00:00Z", "neutral_site": False,
        "slate": "wk8", "public_date": "2026-10-17",
        "status": FC.Status.PRICED.value,
        # 10.0 against a 16.0 spread: comfortably outside the
        # narrow-projection rule, so the BASELINE card carries the fallback
        # caution and the specific ones get their own cases. A 6.5-point
        # projection here would (correctly) trip
        # `uncertainty_exceeds_projection`, which is a real result and not a
        # fixture to design around.
        "projected_home_margin": 10.0, "forecast_margin_sd": 16.0,
        "p_home_win": 0.734, "projected_total": 51.0,
        "rating_parameter_sd": 2.1,
        "forecast_sd_basis": "sim_kalman_wk4_plus",
        **PROVENANCE,
    })
    row.update(overrides)
    return row


def _components(rating=6.5, home_field=2.0, simulation=1.5):
    out = [IN.WhyComponent(IN.RATING_KEY, "Rating differential", rating,
                           PROVENANCE["model_version"])]
    if home_field is not None:
        out.append(IN.WhyComponent(IN.HOME_FIELD_KEY, "Home-field effect",
                                   home_field, PROVENANCE["model_version"]))
    if simulation is not None:
        out.append(IN.WhyComponent(IN.SIMULATION_KEY, "Simulation adjustment",
                                   simulation, "drive-level simulation"))
    return out


def _card(**kw):
    return IN.build_card(_row(**kw.pop("row", {})),
                         data_as_of_utc=DATA_AS_OF, **kw)


# --------------------------------------------------------------------------- #
# 1. priced
# --------------------------------------------------------------------------- #
def test_snapshot_priced():
    card = _card(components=_components())

    assert card["status"] == "priced"
    assert card["call"] == {
        "headline": "Ohio State projected by 10.0",
        "projected_ahead": "Ohio State",
        "projected_margin": 10.0,
        "projected_home_margin": 10.0,
        "home_win_probability": 0.734,
        "margin_uncertainty_sd": 16.0,
        "uncertainty_statement":
            "Predictive spread of the projected margin: 16.0 points (1 SD).",
        "projected_total": 51.0,
    }
    assert card["why"] == [
        {"key": "rating_differential", "label": "Rating differential",
         "points": 6.5, "source": "fp-kalman-0001"},
        {"key": "home_field", "label": "Home-field effect",
         "points": 2.0, "source": "fp-kalman-0001"},
        {"key": "simulation", "label": "Simulation adjustment",
         "points": 1.5, "source": "drive-level simulation"},
    ]
    assert card["provenance"] == {
        "model_family": "kalman_montecarlo",
        "model_version": "fp-kalman-0001",
        "generated_at_utc": "2026-10-14T12:00:00Z",
        "snapshot_id": "20261014T120000Z",
        "source_digest": "digest-abc123",
        "data_as_of_utc": DATA_AS_OF,
        "forecast_sd_basis": "sim_kalman_wk4_plus",
        "schema_version": 2,
    }
    assert card["caution"] == {"code": "paper_only",
                               "text": IN.CAUTIONS["paper_only"]}
    assert "context" not in card, "no context source, so no context section"
    assert set(card) == {"schema_version", "game_id", "season", "week",
                         "matchup", "status", "call", "why", "provenance",
                         "market_reference", "caution", "disclaimer"}


def test_a_priced_card_is_small_enough_for_an_api_card():
    blob = IN.canonical_json(_card(components=_components()))
    assert len(blob) < 2048, f"{len(blob)} bytes is not a card, it is a page"


def test_the_away_side_is_named_when_it_is_projected_ahead():
    card = _card(row={"projected_home_margin": -4.25, "p_home_win": 0.395},
                 components=_components(rating=-5.75, home_field=2.0,
                                        simulation=-0.5))
    assert card["call"]["headline"] == "Michigan projected by 4.2"
    assert card["call"]["projected_ahead"] == "Michigan"


# --------------------------------------------------------------------------- #
# 2. refused
# --------------------------------------------------------------------------- #
def test_snapshot_refused():
    card = _card(row={
        "status": FC.Status.REFUSED.value,
        "refusal_reason": FC.DEFAULT_REFUSALS["cross_division"],
        "projected_home_margin": None, "forecast_margin_sd": None,
        "p_home_win": None, "projected_total": None,
        "rating_parameter_sd": None, "forecast_sd_basis": None})

    assert "call" not in card, "a refusal publishes no number"
    assert "why" not in card, "there is no reasoning behind a number not made"
    assert card["caution"] == {
        "code": "refused_cross_division",
        "text": IN.CAUTIONS["refused_cross_division"],
        "detail": FC.DEFAULT_REFUSALS["cross_division"]}
    assert card["provenance"]["model_family"] == "kalman_montecarlo"


@pytest.mark.parametrize("reason,code", [
    (FC.DEFAULT_REFUSALS["non_fbs"], "refused_non_fbs"),
    (FC.DEFAULT_REFUSALS["cross_division"], "refused_cross_division"),
    (FC.DEFAULT_REFUSALS["unrated"], "refused_unrated"),
    ("REFUSED: no schema-v2 pregame snapshot existed before kickoff",
     "refused_no_pregame_record"),
    ("something the rules do not name", "refused_other"),
])
def test_every_refusal_maps_to_a_specific_code(reason, code):
    card = _card(row={
        "status": FC.Status.REFUSED.value, "refusal_reason": reason,
        "projected_home_margin": None, "forecast_margin_sd": None,
        "p_home_win": None, "projected_total": None,
        "rating_parameter_sd": None, "forecast_sd_basis": None})
    assert card["caution"]["code"] == code


# --------------------------------------------------------------------------- #
# 3. frozen
# --------------------------------------------------------------------------- #
def test_snapshot_frozen():
    card = _card(row={"status": FC.Status.FROZEN.value,
                      "frozen_from_stamp": "20261016T180000Z"})

    assert card["call"]["headline"] == "Ohio State projected by 10.0", \
        "a frozen row keeps the number it was published with"
    assert "why" not in card, (
        "the table has moved since the freeze; a decomposition built now "
        "would explain the number with reasoning that did not produce it")
    assert card["provenance"]["frozen_from_stamp"] == "20261016T180000Z"
    assert card["caution"] == {"code": "frozen_pre_kickoff",
                               "text": IN.CAUTIONS["frozen_pre_kickoff"]}


# --------------------------------------------------------------------------- #
# 4. degraded
# --------------------------------------------------------------------------- #
def test_snapshot_degraded():
    card = _card(row={
        "status": FC.Status.DEGRADED.value,
        "refusal_reason": "games feed stale",
        "projected_home_margin": None, "forecast_margin_sd": None,
        "p_home_win": None, "projected_total": None,
        "rating_parameter_sd": None, "forecast_sd_basis": None})

    assert "call" not in card and "why" not in card
    assert card["caution"] == {"code": "degraded_inputs",
                               "text": IN.CAUTIONS["degraded_inputs"],
                               "detail": "games feed stale"}


# --------------------------------------------------------------------------- #
# 4b. uncertainty unvalidated
# --------------------------------------------------------------------------- #
def test_snapshot_uncertainty_unvalidated():
    card = _card(row={
        "status": FC.Status.UNCERTAINTY_UNVALIDATED.value,
        "refusal_reason": "no registered out-of-sample basis for week 9",
        "projected_home_margin": None, "forecast_margin_sd": None,
        "p_home_win": None, "projected_total": None,
        "rating_parameter_sd": None, "forecast_sd_basis": None})

    assert "call" not in card and "why" not in card
    assert card["caution"] == {
        "code": "uncertainty_unvalidated",
        "text": IN.CAUTIONS["uncertainty_unvalidated"],
        "detail": "no registered out-of-sample basis for week 9"}


# --------------------------------------------------------------------------- #
# 5. no market reference
# --------------------------------------------------------------------------- #
def test_snapshot_no_market_reference():
    card = _card(components=_components())
    assert card["market_reference"] == {
        "state": "disabled",
        "reason": "Market comparison is intentionally switched off for this "
                  "surface."}
    blob = IN.canonical_json(card).lower()
    for phrase in ("no market line", "no line cached", "not cached",
                   "unavailable"):
        assert phrase not in blob, (
            f"the card claims {phrase!r}; absence of a comparison is not "
            "evidence that the data is missing")


def test_a_disabled_market_reference_must_say_why():
    with pytest.raises(IN.InsightError, match="say why"):
        IN.MarketReference.disabled("")


# --------------------------------------------------------------------------- #
# 6. available market reference
# --------------------------------------------------------------------------- #
def test_snapshot_available_market_reference():
    market = IN.MarketReference.available(
        home_spread=-6.5, observed_at_utc="2026-10-16T15:00:00Z",
        source="cached lines median (5 books)", n_books=5)
    card = _card(components=_components(), market=market)

    assert card["market_reference"] == {
        "state": "available",
        "home_spread": -6.5,
        "observed_at_utc": "2026-10-16T15:00:00Z",
        "source": "cached lines median (5 books)",
        "n_books": 5,
        "note": ("Reference only, shown with its observation time. This "
                 "surface makes no comparison against it."),
    }


@pytest.mark.parametrize("field", ["observed_at_utc", "source"])
def test_a_market_reference_without_provenance_is_refused(field):
    kwargs = {"home_spread": -6.5, "observed_at_utc": "2026-10-16T15:00:00Z",
              "source": "cached lines median"}
    kwargs[field] = ""
    with pytest.raises(IN.InsightError, match="rumour"):
        IN.MarketReference.available(**kwargs)


# --------------------------------------------------------------------------- #
# 7. missing context source
# --------------------------------------------------------------------------- #
SPEC = CTX.FieldSpec(key="venue_altitude_ft", family="venue_context",
                     label="Venue altitude", source="cfbd_venues", unit="ft")


class _Context:
    def __init__(self, values):
        self.values = values


def test_snapshot_missing_context_source_hides_the_section():
    """An empty framework reads as data that failed to load. Say nothing."""
    unavailable = CTX.ContextValue.unavailable(
        spec=SPEC, availability=CTX.Availability.UNAVAILABLE_NO_SOURCE,
        note="no approved provider", registry_status="display_only")
    card = _card(components=_components(), context=_Context([unavailable]))
    assert "context" not in card
    assert "altitude" not in IN.canonical_json(card).lower()


def test_a_stale_context_field_is_hidden_even_though_it_has_provenance():
    """Isolates the availability check from the provenance check.

    A stale field carries its source, its reference and its observation time —
    everything the provenance filter looks for — and is still not something to
    show. It would render as a row reading "unavailable (stale)", which is the
    empty framework the rule forbids: it tells the reader a thing is there and
    broken, when the honest answer is to say nothing about it.
    """
    stale = CTX.ContextValue.unavailable(
        spec=SPEC, availability=CTX.Availability.STALE,
        note="last refreshed before the season", registry_status="display_only",
        source="cfbd_venues", source_url_or_id="cfbd:venues/2025",
        observed_at=dt.datetime(2025, 8, 1, tzinfo=UTC), coverage=1.0)
    assert stale.source and stale.source_url_or_id and stale.observed_at, \
        "the fixture must have provenance, or it proves nothing"
    card = _card(components=_components(), context=_Context([stale]))
    assert "context" not in card
    assert "stale" not in IN.canonical_json(card).lower()


def test_a_context_field_with_full_provenance_renders_display_only():
    observed = dt.datetime(2026, 10, 14, 8, 0, tzinfo=UTC)
    available = CTX.ContextValue.available(
        spec=SPEC, value=5280, source="cfbd_venues",
        source_url_or_id="cfbd:venues/2026", observed_at=observed,
        effective_for_game=dt.datetime(2026, 10, 17, 16, 0, tzinfo=UTC),
        confidence="high", coverage=1.0, registry_status="display_only")
    card = _card(components=_components(), context=_Context([available]))
    assert card["context"] == [{
        "key": "venue_altitude_ft",
        "label": "Venue altitude",
        "display": available.display,
        "source": "cfbd_venues",
        "source_ref": "cfbd:venues/2026",
        "observed_at_utc": observed.isoformat(),
        "display_only": True,
    }]


def test_context_never_appears_in_the_call_or_the_why():
    """Display-only means the forecast cannot have moved because of it."""
    observed = dt.datetime(2026, 10, 14, 8, 0, tzinfo=UTC)
    available = CTX.ContextValue.available(
        spec=SPEC, value=5280, source="cfbd_venues",
        source_url_or_id="cfbd:venues/2026", observed_at=observed,
        effective_for_game=dt.datetime(2026, 10, 17, 16, 0, tzinfo=UTC),
        confidence="high", coverage=1.0, registry_status="display_only")
    plain = _card(components=_components())
    with_context = _card(components=_components(),
                         context=_Context([available]))
    assert plain["call"] == with_context["call"]
    assert plain["why"] == with_context["why"]


# --------------------------------------------------------------------------- #
# Byte stability, arithmetic, and the vocabulary guard
# --------------------------------------------------------------------------- #
def test_identical_inputs_give_identical_bytes():
    first = IN.canonical_json(_card(components=_components()))
    second = IN.canonical_json(_card(components=_components()))
    assert first == second
    assert json.loads(first) == json.loads(second)


def test_the_card_reads_no_clock():
    """Every timestamp on a card comes from the row it describes."""
    card = _card(components=_components())
    stamps = {v for k, v in card["provenance"].items()
              if isinstance(v, str) and v.endswith("Z")}
    assert stamps <= {PROVENANCE["generated_at_utc"],
                      PROVENANCE["snapshot_id"], DATA_AS_OF}


def test_components_must_sum_to_the_published_margin():
    with pytest.raises(IN.InsightError, match="arithmetic does not close"):
        _card(components=_components(rating=99.0))


def test_a_priced_card_must_say_why():
    with pytest.raises(IN.InsightError, match="must say why"):
        _card(components=[])


def test_a_neutral_site_shows_no_home_field_term():
    with pytest.raises(IN.InsightError, match="no home-field term"):
        _card(row={"neutral_site": True}, components=_components())
    card = _card(row={"neutral_site": True, "projected_home_margin": 8.0},
                 components=_components(rating=6.5, home_field=None,
                                        simulation=1.5))
    assert [c["key"] for c in card["why"]] == ["rating_differential",
                                              "simulation"]


def test_a_non_simulating_family_cannot_show_a_simulation_adjustment():
    with pytest.raises(IN.InsightError, match="runs no simulation"):
        _card(row={"model_family": "ridge_preseason_prior",
                   "projected_total": None},
              components=_components())


def test_the_vocabulary_guard_refuses_actionable_language():
    with pytest.raises(IN.InsightError, match="actionable vocabulary"):
        IN.assert_non_actionable({"caution": "the model likes this pick"})
    with pytest.raises(IN.InsightError, match="actionable vocabulary"):
        IN.assert_non_actionable({"note": "a 3-point edge against the number"})


def test_the_guard_does_not_censor_a_team_called_lock_haven():
    """A guard that blocks a forecast over the opponent's name gets disabled."""
    card = IN.build_card(
        _row(away="Lock Haven", status=FC.Status.REFUSED.value,
             refusal_reason=FC.DEFAULT_REFUSALS["non_fbs"],
             projected_home_margin=None, forecast_margin_sd=None,
             p_home_win=None, projected_total=None,
             rating_parameter_sd=None, forecast_sd_basis=None),
        data_as_of_utc=DATA_AS_OF)
    assert card["matchup"]["away"] == "Lock Haven"


def test_every_shipped_caution_string_is_non_actionable():
    IN.assert_non_actionable(IN.CAUTIONS)
    IN.assert_non_actionable(IN.DISCLAIMER)
    IN.assert_non_actionable(IN._LABELS)


@pytest.mark.parametrize("week,code", [(1, "early_season_prior"),
                                       (8, "paper_only")])
def test_early_weeks_carry_the_prior_dominance_caution(week, code):
    card = _card(row={"week": week}, components=_components())
    assert card["caution"]["code"] == code


def test_a_wide_or_narrow_projection_is_called_out():
    narrow = _card(row={"projected_home_margin": 1.0, "p_home_win": 0.525},
                   components=_components(rating=0.5, home_field=0.3,
                                          simulation=0.2))
    assert narrow["caution"]["code"] == "uncertainty_exceeds_projection"
    wide = _card(row={"projected_home_margin": 12.0, "p_home_win": 0.75,
                      "forecast_margin_sd": 19.0},
                 components=_components(rating=9.0, home_field=2.0,
                                        simulation=1.0))
    assert wide["caution"]["code"] == "wide_uncertainty"


def test_a_week_of_cards_counts_its_statuses():
    rows = [_row(), _row(game_id=2, status=FC.Status.REFUSED.value,
                         refusal_reason=FC.DEFAULT_REFUSALS["non_fbs"],
                         projected_home_margin=None, forecast_margin_sd=None,
                         p_home_win=None, projected_total=None,
                         rating_parameter_sd=None, forecast_sd_basis=None)]
    payload = IN.build_week(rows, components_by_game={401900101: _components()},
                            data_as_of_utc=DATA_AS_OF)
    assert payload["n_cards"] == 2
    assert payload["by_status"] == {"priced": 1, "refused": 1}
    assert IN.canonical_json(payload) == IN.canonical_json(
        IN.build_week(rows, components_by_game={401900101: _components()},
                      data_as_of_utc=DATA_AS_OF))
