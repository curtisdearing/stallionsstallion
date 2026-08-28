"""Data contracts for the pregame-context layer.

Every context value the panel can show is declared here as a `FieldSpec`,
bound to exactly one source in `data/context_sources.json` and, where a
measured factor exists, to one factor id in `data/factor_registry.json`.

Two rules are enforced by construction rather than by convention:

1. **A value exists or it does not.** `ContextValue.available` demands the
   whole provenance envelope — source, source_url_or_id, observed_at,
   effective_for_game, confidence, coverage — and refuses to build without
   it. `ContextValue.unavailable` is the only other constructor and it
   cannot carry a number. There is no path that produces a defaulted,
   mean-filled or zero-filled context value.
2. **The factor registry decides usage.** `resolve_usage` reads a status
   string that came from `data/factor_registry.json` and nothing else. Only
   `gated_active` can ever yield `used_by_model`, and only for a field that
   also declares `model_consumed=True` — a field cannot promote itself.

This module imports nothing from the modelling stack; a test asserts that.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ContextContractError(RuntimeError):
    """A context value was built without the provenance its contract demands."""


class LeakageError(RuntimeError):
    """Something tried to read a column this layer is forbidden to see."""


class Availability(str, Enum):
    OK = "ok"
    UNAVAILABLE_NO_SOURCE = "unavailable_no_source"
    UNAVAILABLE_NOT_APPROVED = "unavailable_not_approved"
    UNAVAILABLE_NO_DATA = "unavailable_no_data"
    STALE = "stale"
    CONFLICTING = "conflicting"
    LOW_COVERAGE = "low_coverage"
    REFUSED_POST_KICKOFF = "refused_post_kickoff"


class Usage(str, Enum):
    USED_BY_MODEL = "used_by_model"
    DISPLAY_ONLY = "display_only"
    EXCLUDED = "excluded"
    MISSING = "missing"


#: Registry statuses that this layer understands. Anything else is treated
#: as display-only, never as active — an unknown status must not promote.
REGISTRY_STATUSES = ("gated_active", "display_only", "under_revalidation",
                     "excluded")

#: Column names this layer may never read. The closing line is denied by
#: name because the project has a retraction on record from reading it.
FORBIDDEN_COLUMNS = frozenset({
    "spread_close", "close", "closing_spread", "over_under_close",
    "total_close", "formatted_spread_close", "closing_line", "clv_pts",
    # `lines_*.parquet` carries the LIVE price under these plain names; it
    # becomes the closing line as kickoff approaches. Only the explicit
    # `*_open` columns are readable here.
    "spread", "over_under", "home_moneyline", "away_moneyline",
    "home_points", "away_points", "points", "margin", "total_actual",
    "result", "final", "winner", "score", "home_score", "away_score",
})

FAMILIES = (
    "qb_availability",
    "unit_availability",
    "roster_and_talent",
    "coaching_continuity",
    "venue_context",
    "weather_forecast",
    "rest_travel_bodyclock",
    "schedule_connectivity",
    "five_factors",
    "market_open_diagnostic",
)

FAMILY_LABELS = {
    "qb_availability": "Quarterback availability",
    "unit_availability": "Offensive line and key units",
    "roster_and_talent": "Roster turnover, portal, returning production, talent",
    "coaching_continuity": "Coaching and staff continuity",
    "venue_context": "Venue, neutral site, altitude, home field",
    "weather_forecast": "Forecast weather",
    "rest_travel_bodyclock": "Rest, short week, travel, body clock",
    "schedule_connectivity": "Connectivity, realignment, new-to-FBS",
    "five_factors": "Field position, finishing drives, special teams, turnover luck",
    "market_open_diagnostic": "Market open (diagnostic reference only)",
}


def assert_column_allowed(column: str) -> None:
    """Raise if `column` is one this layer may never read."""
    if column in FORBIDDEN_COLUMNS:
        raise LeakageError(
            f"column {column!r} is denied to the context layer: it is either a "
            "closing price or a post-kickoff outcome, and this project has a "
            "recorded retraction from letting one of those in")


def kickoff_guard(*, observed_at: Optional[dt.datetime],
                  effective_for_game: Optional[dt.datetime],
                  kickoff: dt.datetime) -> Tuple[Availability, Optional[str]]:
    """Refuse anything observed at or after kickoff.

    A pregame packet may only carry what was knowable before the game
    started. This is checked on the observation time AND on the time the
    value claims to be effective for.
    """
    if observed_at is not None and observed_at >= kickoff:
        return (Availability.REFUSED_POST_KICKOFF,
                f"refused: observed_at {observed_at.isoformat()} is at or "
                f"after kickoff {kickoff.isoformat()}")
    # `effective_for_game` MAY equal kickoff — that is what "effective for
    # this game" means. Anything later describes a different game.
    if effective_for_game is not None and effective_for_game > kickoff:
        return (Availability.REFUSED_POST_KICKOFF,
                f"refused: effective_for_game {effective_for_game.isoformat()} "
                f"is after kickoff {kickoff.isoformat()}")
    return Availability.OK, None


def reconcile(observations: Sequence[Tuple[str, Optional[float]]],
              tolerance: float) -> Tuple[Availability, Optional[float]]:
    """Agree on a number across sources, or refuse to pick one.

    Returns `(OK, value)` when every observation sits inside `tolerance`,
    `(CONFLICTING, None)` when they do not. Disagreement never resolves to a
    mean — a conflict is information, and averaging it away hides it.
    """
    vals = [v for _, v in observations if v is not None]
    if not vals:
        return Availability.UNAVAILABLE_NO_DATA, None
    if max(vals) - min(vals) > tolerance:
        return Availability.CONFLICTING, None
    return Availability.OK, float(vals[0])


@dataclass(frozen=True)
class FieldSpec:
    key: str
    family: str
    label: str
    source: str
    unit: Optional[str] = None
    registry_factor_id: Optional[str] = None
    #: True only for a field the shipped engine actually consumes. Necessary
    #: but not sufficient for `used_by_model` — the registry still decides.
    model_consumed: bool = False
    note: Optional[str] = None


def resolve_usage(spec: FieldSpec, registry_status: Optional[str],
                  availability: Availability) -> Usage:
    """Which display bucket a field belongs in. The registry is the gate."""
    if registry_status == "excluded":
        return Usage.EXCLUDED
    if availability is not Availability.OK:
        return Usage.MISSING
    if registry_status == "gated_active" and spec.model_consumed:
        return Usage.USED_BY_MODEL
    return Usage.DISPLAY_ONLY


@dataclass(frozen=True)
class ContextValue:
    key: str
    family: str
    label: str
    source_name: str
    value: Optional[Any]
    display: str
    unit: Optional[str]
    availability: Availability
    usage: Usage
    registry_status: Optional[str]
    registry_factor_id: Optional[str]
    source: Optional[str]
    source_url_or_id: Optional[str]
    observed_at: Optional[dt.datetime]
    effective_for_game: Optional[dt.datetime]
    confidence: Optional[str]
    coverage: Optional[float]
    note: Optional[str] = None

    @classmethod
    def available(cls, *, spec: FieldSpec, value: Any, source: str,
                  source_url_or_id: Optional[str],
                  observed_at: Optional[dt.datetime],
                  effective_for_game: Optional[dt.datetime],
                  confidence: Optional[str], coverage: Optional[float],
                  registry_status: Optional[str],
                  display: Optional[str] = None,
                  note: Optional[str] = None) -> "ContextValue":
        envelope = {"source": source, "source_url_or_id": source_url_or_id,
                    "observed_at": observed_at,
                    "effective_for_game": effective_for_game,
                    "confidence": confidence, "coverage": coverage}
        missing = [k for k, v in envelope.items() if v is None]
        if missing:
            raise ContextContractError(
                f"{spec.key}: cannot publish a value without {missing} — an "
                "unprovenanced number is not a context value")
        if value is None:
            raise ContextContractError(
                f"{spec.key}: available values must carry a value")
        return cls(key=spec.key, family=spec.family, label=spec.label,
                   source_name=spec.source, value=value,
                   display=display if display is not None else _fmt(value, spec.unit),
                   unit=spec.unit, availability=Availability.OK,
                   usage=resolve_usage(spec, registry_status, Availability.OK),
                   registry_status=registry_status,
                   registry_factor_id=spec.registry_factor_id,
                   note=note, **envelope)

    @classmethod
    def unavailable(cls, *, spec: FieldSpec, availability: Availability,
                    note: str, registry_status: Optional[str],
                    source: Optional[str] = None,
                    source_url_or_id: Optional[str] = None,
                    observed_at: Optional[dt.datetime] = None,
                    effective_for_game: Optional[dt.datetime] = None,
                    coverage: Optional[float] = None) -> "ContextValue":
        if availability is Availability.OK:
            raise ContextContractError(
                f"{spec.key}: `unavailable` cannot carry an OK state")
        return cls(key=spec.key, family=spec.family, label=spec.label,
                   source_name=spec.source, value=None,
                   display=f"unavailable ({availability.value})",
                   unit=spec.unit, availability=availability,
                   usage=resolve_usage(spec, registry_status, availability),
                   registry_status=registry_status,
                   registry_factor_id=spec.registry_factor_id,
                   source=source, source_url_or_id=source_url_or_id,
                   observed_at=observed_at,
                   effective_for_game=effective_for_game,
                   confidence=None, coverage=coverage, note=note)

    def to_dict(self) -> Dict[str, Any]:
        d = {f: getattr(self, f) for f in (
            "key", "family", "label", "source_name", "value", "display",
            "unit", "registry_status", "registry_factor_id", "source",
            "source_url_or_id", "confidence", "coverage", "note")}
        d["availability"] = self.availability.value
        d["usage"] = self.usage.value
        d["observed_at"] = (self.observed_at.isoformat()
                            if self.observed_at else None)
        d["effective_for_game"] = (self.effective_for_game.isoformat()
                                   if self.effective_for_game else None)
        return d


def _fmt(value: Any, unit: Optional[str]) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        s = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        s = str(value)
    return f"{s} {unit}" if unit else s


#: Every market field says this. The blend that would let a market price
#: legitimately touch a model number FAILED out-of-sample twice and is
#: disabled; the opening price is here as a DIAGNOSTIC reference only, and
#: the closing price is denied to this layer by name.
_MKT = ("diagnostic reference only — the market blend failed out-of-sample "
        "and is disabled; opening prices only, the close is denied to this "
        "layer by contract")


def _pair(key, family, label, source, unit=None, factor=None,
          model_consumed=False, note=None):
    """Home/away pair of the same field."""
    return [FieldSpec(key=f"{key}_{side}", family=family,
                      label=f"{label} ({side})", source=source, unit=unit,
                      registry_factor_id=factor, model_consumed=model_consumed,
                      note=note)
            for side in ("home", "away")]


#: The whole declared surface. A family with no approved source still
#: appears here — that is the point: the panel shows the question was asked
#: and answers "unavailable", rather than leaving a silent hole.
FIELDS: Tuple = tuple(
    # --- 1. quarterback availability ---------------------------------- #
    _pair("qb_status", "qb_availability", "Confirmed/probable starting QB",
          "qb_availability_feed", factor="backup_qb_pregame",
          note="no approved availability feed; the cached qb_starts proxy is "
               "retrospective and its pregame variant was REJECTED (p=0.075)")
    + _pair("qb_starter_change", "qb_availability",
            "Starter differs from the established primary",
            "qb_availability_feed", factor="backup_qb_pregame")
    # --- 2. offensive line and key units ------------------------------- #
    + _pair("ol_availability", "unit_availability",
            "Offensive-line availability", "unit_availability_feed")
    + _pair("key_unit_availability", "unit_availability",
            "Other key-unit availability", "unit_availability_feed")
    # --- 3. roster turnover, portal, returning production, talent ------ #
    + _pair("returning_ppa", "roster_and_talent",
            "Returning production (share of prior-season PPA)", "cfbd_returning")
    + _pair("returning_passing_ppa", "roster_and_talent",
            "Returning passing production", "cfbd_returning")
    + _pair("talent_composite", "roster_and_talent",
            "247 talent composite", "cfbd_talent")
    + _pair("talent_unranked", "roster_and_talent",
            "Talent is a missing-data sentinel, not a low score", "cfbd_talent")
    + _pair("portal_net", "roster_and_talent",
            "Net transfer-portal movement", "cfbd_portal")
    # --- 4. coaching and staff continuity ------------------------------ #
    + _pair("coach_first_year", "coaching_continuity",
            "First-season head coach", "cfbd_coaches",
            factor="coaching_change_y1")
    + _pair("coach_name", "coaching_continuity", "Head coach", "cfbd_coaches")
    # --- 5. venue, neutral site, altitude, home field ------------------ #
    + [FieldSpec("neutral_site", "venue_context", "Neutral site", "cfbd_games"),
       FieldSpec("venue_name", "venue_context", "Venue", "cfbd_venues"),
       FieldSpec("venue_altitude_ft", "venue_context", "Venue altitude",
                 "cfbd_venues", unit="ft"),
       FieldSpec("hfa_applied_home", "venue_context",
                 "Home-field advantage applied by the model", "model_ratings",
                 unit="pts", registry_factor_id="per_team_time_varying_hfa",
                 model_consumed=True,
                 note="read back OUT of the finished ratings table; shown so "
                      "the panel can state what the model actually used")]
    # --- 6. forecast weather ------------------------------------------- #
    + [FieldSpec("wx_wind_mph", "weather_forecast", "Forecast wind",
                 "weather_forecast", unit="mph"),
       FieldSpec("wx_precip_prob", "weather_forecast",
                 "Forecast precipitation probability", "weather_forecast"),
       FieldSpec("wx_temp_f", "weather_forecast", "Forecast temperature",
                 "weather_forecast", unit="F"),
       FieldSpec("wx_forecast_issued_at", "weather_forecast",
                 "Forecast issue time", "weather_forecast",
                 note="distinct from observed_at: a forecast has both an issue "
                      "time and a fetch time, and a pregame packet needs both")]
    # --- 7. rest, short week, travel, body clock ----------------------- #
    + _pair("rest_days", "rest_travel_bodyclock", "Days since previous game",
            "cfbd_games", unit="days")
    + _pair("short_week", "rest_travel_bodyclock", "Short week (< 6 days)",
            "cfbd_games")
    + [FieldSpec("kickoff_et_hour", "rest_travel_bodyclock",
                 "Kickoff, Eastern", "cfbd_games"),
       FieldSpec("kickoff_local_hour", "rest_travel_bodyclock",
                 "Kickoff hour, venue-local", "cfbd_venues", unit="h"),
       FieldSpec("travel_distance_mi_away", "rest_travel_bodyclock",
                 "Away-team travel distance", "cfbd_venues", unit="mi"),
       FieldSpec("tz_delta_away", "rest_travel_bodyclock",
                 "Time zones crossed by the away team", "cfbd_venues"),
       FieldSpec("body_clock_hour_away", "rest_travel_bodyclock",
                 "Kickoff on the away team's body clock", "cfbd_venues",
                 unit="h",
                 note="PROPOSED ONLY — research/circadian-timezone-factor-plan.md "
                      "requires a slice census and a priced-in test before any "
                      "effect is estimated; this is descriptive, not an effect")]
    # --- 8. connectivity, realignment, new-to-FBS ---------------------- #
    + _pair("connectivity", "schedule_connectivity",
            "Schedule-graph connectivity", "model_ratings")
    + _pair("rated_games", "schedule_connectivity",
            "Games behind the rating", "model_ratings")
    + _pair("rating_band", "schedule_connectivity", "Rating band", "model_ratings",
            unit="pts")
    + _pair("conference", "schedule_connectivity", "Conference (this season)",
            "cfbd_teams")
    + _pair("conference_changed", "schedule_connectivity",
            "Changed conference since last season", "cfbd_teams")
    + _pair("new_to_fbs", "schedule_connectivity", "New to FBS this season",
            "cfbd_teams")
    # --- 9. five factors ------------------------------------------------ #
    + [FieldSpec("finishing_drives_diff", "five_factors",
                 "Finishing drives (points per trip inside the 40), walk-forward",
                 "model_drive_features", unit="pts/trip",
                 registry_factor_id="finishing_drives_diff"),
       FieldSpec("field_position_diff", "five_factors",
                 "Average starting field position differential, walk-forward",
                 "model_drive_features", unit="yd",
                 registry_factor_id="field_position_diff"),
       FieldSpec("special_teams_diff", "five_factors",
                 "Special-teams differential, walk-forward",
                 "model_drive_features",
                 note="never gate-tested by this project; unregistered, so "
                      "display-only by default and never promotable here")]
    + _pair("turnover_luck", "five_factors",
            "Turnover luck (recovery rate vs expectation), walk-forward",
            "model_drive_features",
            note="turnover LUCK, deliberately not raw turnover margin — "
                 "raw_turnover_margin is a standing exclusion")
    # --- 10. market open, diagnostic only ------------------------------- #
    + [FieldSpec("open_spread", "market_open_diagnostic",
                 "Opening spread (diagnostic)", "cfbd_lines_open", unit="pts",
                 note=_MKT),
       FieldSpec("open_total", "market_open_diagnostic",
                 "Opening total (diagnostic)", "cfbd_lines_open", unit="pts",
                 note=_MKT),
       FieldSpec("open_book_count", "market_open_diagnostic",
                 "Books with an opening price", "cfbd_lines_open", note=_MKT),
       FieldSpec("open_spread_dispersion", "market_open_diagnostic",
                 "Spread disagreement across books", "cfbd_lines_open",
                 unit="pts", note=_MKT)]
)

FIELDS_BY_KEY: Dict[str, FieldSpec] = {s.key: s for s in FIELDS}

FIELDS_BY_SOURCE: Dict[str, List[FieldSpec]] = {}
for _s in FIELDS:
    FIELDS_BY_SOURCE.setdefault(_s.source, []).append(_s)

FIELDS_BY_FAMILY: Dict[str, List[FieldSpec]] = {}
for _s in FIELDS:
    FIELDS_BY_FAMILY.setdefault(_s.family, []).append(_s)
