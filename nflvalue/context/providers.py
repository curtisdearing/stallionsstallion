"""Providers: one per declared source, each able only to describe.

A provider never computes anything a projection could use. It reads a
frame the model has already finished with, or it reports that it cannot.
The base class does the work that must never be skipped:

  * refuses to construct if it declares a forbidden column;
  * asks the approval gate first and emits refusals for its whole field
    set when the source is not approved;
  * applies the kickoff guard, then freshness, then coverage;
  * emits exactly one value for every field bound to its source, so a
    field can never silently vanish from the panel.
"""
from __future__ import annotations

import math
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Sequence,
                    Tuple)

from . import sources as SRC
from .contracts import (FIELDS_BY_SOURCE, FORBIDDEN_COLUMNS, Availability,
                        ContextValue, FieldSpec, assert_column_allowed,
                        kickoff_guard)

if TYPE_CHECKING:                      # pragma: no cover - typing only
    from .assemble import Env

#: source name -> the key its frame arrives under
FRAME_KEYS = {
    "cfbd_games": "games", "cfbd_teams": "teams", "cfbd_talent": "talent",
    "cfbd_returning": "returning", "cfbd_lines_open": "lines",
    "cfbd_venues": "venues", "cfbd_coaches": "coaches",
    "cfbd_portal": "portal", "model_ratings": "ratings",
    "model_drive_features": "drives",
}


class Provider:
    name: str = "provider"
    source: str = ""
    reads: Tuple[str, ...] = ()
    excludes_current_game: bool = True

    def __init__(self) -> None:
        for col in self.reads:
            assert_column_allowed(col)

    # -- fields this provider owns ------------------------------------- #
    @property
    def fields(self) -> Sequence[FieldSpec]:
        return FIELDS_BY_SOURCE.get(self.source, ())

    # -- the one public entry point ------------------------------------ #
    def emit(self, env: "Env") -> List[ContextValue]:
        gate, why = env.manifest.gate(self.source)
        if gate is not Availability.OK:
            return [self._refuse(env, spec, gate, why) for spec in self.fields]

        observed = env.observed_at(self.source)
        if observed is None:
            return [self._refuse(env, spec, Availability.UNAVAILABLE_NO_DATA,
                                 "no observation time recorded for this source")
                    for spec in self.fields]

        guard, note = kickoff_guard(observed_at=observed,
                                    effective_for_game=env.kickoff,
                                    kickoff=env.kickoff)
        if guard is not Availability.OK:
            return [self._refuse(env, spec, guard, note, observed_at=observed)
                    for spec in self.fields]

        fresh = SRC.freshness(env.manifest, self.source, observed_at=observed,
                              now=env.now)
        if fresh is not Availability.OK:
            age = (env.now - observed).total_seconds() / 3600.0
            return [self._refuse(env, spec, fresh,
                                 f"observed {age:.1f}h ago, older than the "
                                 f"contract's max_age_hours",
                                 observed_at=observed) for spec in self.fields]

        frame = env.frames.get(FRAME_KEYS.get(self.source, ""))
        if frame is None or len(frame) == 0:
            return [self._refuse(env, spec, Availability.UNAVAILABLE_NO_DATA,
                                 f"no cached {FRAME_KEYS.get(self.source)} frame",
                                 observed_at=observed) for spec in self.fields]

        try:
            raw = self.read(env, frame)
        except KeyError as exc:
            return [self._refuse(env, spec, Availability.UNAVAILABLE_NO_DATA,
                                 f"expected column {exc} absent from the cache",
                                 observed_at=observed) for spec in self.fields]

        coverage = self.coverage(env, frame)
        cov_state = SRC.coverage_state(env.manifest, self.source, coverage)
        out = []
        for spec in self.fields:
            val = raw.get(spec.key)
            if cov_state is not Availability.OK:
                out.append(self._refuse(env, spec, cov_state,
                                        f"source covers {coverage:.0%} of what "
                                        "this game needs",
                                        observed_at=observed, coverage=coverage))
                continue
            if val is None or (isinstance(val, float) and math.isnan(val)):
                out.append(self._refuse(env, spec, Availability.UNAVAILABLE_NO_DATA,
                                        "no row for this team/game in the cache",
                                        observed_at=observed, coverage=coverage))
                continue
            out.append(ContextValue.available(
                spec=spec, value=val,
                source=env.manifest.provider(self.source) or self.source,
                source_url_or_id=env.manifest.url(self.source),
                observed_at=observed, effective_for_game=env.kickoff,
                confidence=SRC.confidence_for(env.manifest, self.source,
                                              Availability.OK),
                coverage=coverage,
                registry_status=env.registry.status(spec.registry_factor_id),
                note=spec.note))
        return out

    # -- subclass hooks ------------------------------------------------- #
    def read(self, env: "Env", frame) -> Dict[str, Any]:
        raise NotImplementedError

    def coverage(self, env: "Env", frame) -> float:
        return 1.0

    # -- helper ---------------------------------------------------------- #
    def _refuse(self, env: "Env", spec: FieldSpec, availability: Availability,
                note: Optional[str], observed_at=None,
                coverage=None) -> ContextValue:
        return ContextValue.unavailable(
            spec=spec, availability=availability,
            note=note or "unavailable",
            registry_status=env.registry.status(spec.registry_factor_id),
            source=env.manifest.provider(self.source),
            source_url_or_id=env.manifest.url(self.source),
            observed_at=observed_at, effective_for_game=env.kickoff,
            coverage=coverage)


# --------------------------------------------------------------------- #
# families with no approved source — the gate does all the work
# --------------------------------------------------------------------- #
class QBAvailabilityFeed(Provider):
    name = "qb_availability_feed"
    source = "qb_availability_feed"


class UnitAvailabilityFeed(Provider):
    name = "unit_availability_feed"
    source = "unit_availability_feed"


class WeatherForecast(Provider):
    name = "weather_forecast"
    source = "weather_forecast"


class PortalMovement(Provider):
    name = "portal"
    source = "cfbd_portal"


class VenueContext(Provider):
    name = "venues"
    source = "cfbd_venues"


# --------------------------------------------------------------------- #
# approved, cached sources
# --------------------------------------------------------------------- #
def _get(row, col):
    return None if row is None else row.get(col)


def _num_of(row, col):
    return _num(_get(row, col))


def _team_rows(frame, env) -> Tuple[Optional[Any], Optional[Any]]:
    idx = frame.set_index("team") if "team" in frame.columns else frame
    h = idx.loc[env.home] if env.home in idx.index else None
    a = idx.loc[env.away] if env.away in idx.index else None
    return h, a


def _team_coverage(frame, env) -> float:
    if "team" not in frame.columns:
        return 0.0
    have = set(frame["team"])
    return sum(t in have for t in (env.home, env.away)) / 2.0


class ReturningProduction(Provider):
    name = "returning_production"
    source = "cfbd_returning"
    reads = ("team", "percent_ppa", "percent_passing_ppa")

    def coverage(self, env, frame):
        return _team_coverage(frame, env)

    def read(self, env, frame):
        h, a = _team_rows(frame, env)
        return {"returning_ppa_home": _num_of(h, "percent_ppa"),
                "returning_ppa_away": _num_of(a, "percent_ppa"),
                "returning_passing_ppa_home": _num_of(h, "percent_passing_ppa"),
                "returning_passing_ppa_away": _num_of(a, "percent_passing_ppa")}


class TalentComposite(Provider):
    name = "talent"
    source = "cfbd_talent"
    reads = ("team", "talent", "unranked_recruiting")

    def coverage(self, env, frame):
        return _team_coverage(frame, env)

    def read(self, env, frame):
        h, a = _team_rows(frame, env)
        out = {}
        for side, row in (("home", h), ("away", a)):
            unranked = None if row is None else bool(row.get("unranked_recruiting"))
            out[f"talent_unranked_{side}"] = unranked
            # A sentinel is not a low score: refuse the number, keep the flag.
            out[f"talent_composite_{side}"] = (
                None if row is None or unranked else _num(row.get("talent")))
        return out


class CoachingContinuity(Provider):
    name = "coaches"
    source = "cfbd_coaches"
    reads = ("team", "first_year", "coach", "season")

    def coverage(self, env, frame):
        f = frame[frame["season"] == env.season] if "season" in frame.columns else frame
        return _team_coverage(f, env)

    def read(self, env, frame):
        f = frame[frame["season"] == env.season] if "season" in frame.columns else frame
        h, a = _team_rows(f, env)
        return {"coach_first_year_home": _bool(_get(h, "first_year")),
                "coach_first_year_away": _bool(_get(a, "first_year")),
                "coach_name_home": _get(h, "coach"),
                "coach_name_away": _get(a, "coach")}


class ScheduleStructure(Provider):
    name = "schedule_structure"
    source = "cfbd_games"
    reads = ("id", "season", "week", "start_date", "home_team", "away_team",
             "neutral_site")

    def read(self, env, frame):
        rest = _rest_days(frame, env)
        return {"neutral_site": bool(env.game.get("neutral_site", False)),
                "kickoff_et_hour": env.kickoff_et.strftime("%-I:%M %p"),
                "rest_days_home": rest["home"], "rest_days_away": rest["away"],
                "short_week_home": (None if rest["home"] is None
                                    else rest["home"] < 6),
                "short_week_away": (None if rest["away"] is None
                                    else rest["away"] < 6)}


class RatingsReadback(Provider):
    name = "ratings_readback"
    source = "model_ratings"
    reads = ("team", "connectivity", "fbs_games", "net_band", "hfa")

    def coverage(self, env, frame):
        return _team_coverage(frame, env)

    def read(self, env, frame):
        h, a = _team_rows(frame, env)
        neutral = bool(env.game.get("neutral_site", False))
        hfa = 0.0 if neutral else _num_of(h, "hfa")
        return {"connectivity_home": _num_of(h, "connectivity"),
                "connectivity_away": _num_of(a, "connectivity"),
                "rated_games_home": _num_of(h, "fbs_games"),
                "rated_games_away": _num_of(a, "fbs_games"),
                "rating_band_home": _num_of(h, "net_band"),
                "rating_band_away": _num_of(a, "net_band"),
                "hfa_applied_home": hfa}


class Realignment(Provider):
    """Conference this season, whether it changed, and new-to-FBS status.

    New-to-FBS reads the membership flag where the cache carries one
    (`is_fbs`), and falls back to "absent from last season's table" only
    when it does not — a reclassifying team is a refusal reason for the
    projection, so getting it from a guess would be the wrong trade.
    """
    name = "realignment"
    source = "cfbd_teams"
    reads = ("team", "conference", "season", "is_fbs", "classification")

    def _fbs(self, frame):
        if "is_fbs" in frame.columns:
            return frame[frame["is_fbs"].fillna(False).astype(bool)]
        if "classification" in frame.columns:
            return frame[frame["classification"].astype(str).str.lower() == "fbs"]
        return frame

    def coverage(self, env, frame):
        f = frame[frame["season"] == env.season] if "season" in frame.columns else frame
        return _team_coverage(f, env)

    def read(self, env, frame):
        has_season = "season" in frame.columns
        cur = frame[frame["season"] == env.season] if has_season else frame
        prev = (frame[frame["season"] == env.season - 1] if has_season
                else cur.iloc[0:0])
        out = {}
        prev_teams = set(self._fbs(prev)["team"]) if len(prev) else set()
        prev_conf = (dict(zip(prev["team"], prev["conference"]))
                     if len(prev) else {})
        cur_conf = dict(zip(cur["team"], cur["conference"]))
        for side, team in (("home", env.home), ("away", env.away)):
            out[f"conference_{side}"] = cur_conf.get(team)
            if not prev_teams:
                out[f"conference_changed_{side}"] = None
                out[f"new_to_fbs_{side}"] = None
            else:
                out[f"new_to_fbs_{side}"] = team not in prev_teams
                out[f"conference_changed_{side}"] = (
                    None if team not in prev_conf
                    else prev_conf[team] != cur_conf.get(team))
        return out


class FiveFactorsWalkForward(Provider):
    """Drive-derived Five-Factors context, walk-forward by construction.

    Only completed games from EARLIER weeks of the same season are read.
    The game being described is excluded, and a test asserts it.
    """
    name = "five_factors"
    source = "model_drive_features"
    reads = ("season", "week", "game_id", "team", "points_per_trip",
             "start_field_position", "special_teams_epa", "turnover_luck")

    def rows_considered(self, frames, game) -> Dict[str, Any]:
        d = frames.get("drives")
        if d is None or len(d) == 0:
            return {"game_ids": [], "weeks": []}
        past = d[(d["season"] == game["season"]) & (d["week"] < game["week"])]
        return {"game_ids": sorted(set(past.get("game_id", []))),
                "weeks": sorted(set(past["week"]))}

    def coverage(self, env, frame):
        past = frame[(frame["season"] == env.season)
                     & (frame["week"] < env.week)]
        if len(past) == 0:
            return 0.0
        return _team_coverage(past, env)

    def read(self, env, frame):
        past = frame[(frame["season"] == env.season)
                     & (frame["week"] < env.week)]
        if len(past) == 0:
            return {}
        agg = past.groupby("team").mean(numeric_only=True)

        def g(team, col):
            if team in agg.index and col in agg.columns:
                return _num(agg.loc[team, col])
            return None

        return {
            "finishing_drives_diff": _diff(g(env.home, "points_per_trip"),
                                           g(env.away, "points_per_trip")),
            "field_position_diff": _diff(g(env.home, "start_field_position"),
                                         g(env.away, "start_field_position")),
            "special_teams_diff": _diff(g(env.home, "special_teams_epa"),
                                        g(env.away, "special_teams_epa")),
            "turnover_luck_home": g(env.home, "turnover_luck"),
            "turnover_luck_away": g(env.away, "turnover_luck"),
        }


class MarketOpenDiagnostic(Provider):
    """Opening prices only, and only ever as a diagnostic reference.

    `line_kind` is pinned to "open" and the closing-line column names are
    denied to this layer by `contracts.FORBIDDEN_COLUMNS`, so the base
    class refuses to construct a provider that reads one.
    """
    name = "market_open"
    source = "cfbd_lines_open"
    line_kind = "open"
    reads = ("game_id", "provider", "spread_open", "over_under_open")

    def opening_only(self, frame):
        """Drop every price column that is not explicitly an opening price.

        Structural, not incidental: the frame is narrowed before `read` can
        touch it, so a future edit cannot reach a live or closing price by
        accident.
        """
        keep = [c for c in frame.columns if c not in FORBIDDEN_COLUMNS]
        return frame[keep]

    def read(self, env, frame):
        frame = self.opening_only(frame)
        rows = (frame[frame["game_id"] == env.game["id"]]
                if "game_id" in frame.columns else frame.iloc[0:0])
        if len(rows) == 0:
            return {}
        spreads = [x for x in rows.get("spread_open", []) if x == x]
        totals = [x for x in rows.get("over_under_open", []) if x == x]
        disp = (max(spreads) - min(spreads)) if len(spreads) > 1 else 0.0
        return {"open_spread": _num(sum(spreads) / len(spreads)) if spreads else None,
                "open_total": _num(sum(totals) / len(totals)) if totals else None,
                "open_book_count": int(len(rows)),
                "open_spread_dispersion": _num(disp) if spreads else None}


def all_providers() -> List[Provider]:
    return [QBAvailabilityFeed(), UnitAvailabilityFeed(), ReturningProduction(),
            TalentComposite(), PortalMovement(), CoachingContinuity(),
            ScheduleStructure(), VenueContext(), WeatherForecast(),
            RatingsReadback(), Realignment(), FiveFactorsWalkForward(),
            MarketOpenDiagnostic()]


# --------------------------------------------------------------------- #
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else round(v, 4)


def _bool(x):
    return None if x is None or x != x else bool(x)


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 4)


def _rest_days(frame, env) -> Dict[str, Optional[float]]:
    out = {"home": None, "away": None}
    if "start_date" not in frame.columns:
        return out
    import pandas as pd
    ts = pd.to_datetime(frame["start_date"], utc=True, format="ISO8601",
                        errors="coerce")
    for side, team in (("home", env.home), ("away", env.away)):
        mask = ((frame["home_team"] == team) | (frame["away_team"] == team))
        prior = ts[mask & (ts < env.kickoff)]
        if len(prior):
            out[side] = round((env.kickoff - prior.max()).total_seconds() / 86400.0, 1)
    return out
