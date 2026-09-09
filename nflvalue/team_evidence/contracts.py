"""The observation type, its vocabulary, and the retrospective-claim block.

Design note: this mirrors `nflvalue/context/contracts.py` deliberately. That module
already proved the shape — two constructors, one of which refuses to build without a
full provenance envelope and the other of which cannot carry a number — and a second
vocabulary for the same idea would be a maintenance trap. `EvidenceAvailability`
reproduces every `Availability` value by the same string so the two are comparable,
and adds only the two states the context layer has no need for.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..context.contracts import Availability


class EvidenceContractError(RuntimeError):
    """A value tried to exist without the provenance that makes it a value."""


class RetrospectiveClaimBlocked(RuntimeError):
    """A predictive claim was attempted from an observation whose pregame
    availability was never established. See 00_contract.md §3."""


class EvidenceAvailability(str, Enum):
    """Superset of `context.contracts.Availability`.

    Every Availability member appears here with the SAME string value, so
    `EvidenceAvailability(av.value)` always succeeds and a consumer comparing
    strings across the two layers gets the right answer. Asserted by
    `tests/test_team_evidence_contracts.py`.
    """

    OK = "ok"
    UNAVAILABLE_NO_SOURCE = "unavailable_no_source"
    UNAVAILABLE_NOT_APPROVED = "unavailable_not_approved"
    UNAVAILABLE_NO_DATA = "unavailable_no_data"
    STALE = "stale"
    CONFLICTING = "conflicting"
    LOW_COVERAGE = "low_coverage"
    REFUSED_POST_KICKOFF = "refused_post_kickoff"

    #: the measure exists but n is below its registered gate — a rate over three
    #: plays is not a rate, and rendering it as one is the mistake this prevents
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    #: the field cannot be computed honestly from what is cached, or its pregame
    #: availability cannot be established at all
    BLOCKED_UNVERIFIED_AVAILABILITY = "blocked_unverified_availability"

    @classmethod
    def from_context(cls, availability: Availability) -> "EvidenceAvailability":
        return cls(availability.value)


class Missingness(str, Enum):
    """WHY a number is absent. `availability` says what state the value is in;
    this says what happened. They are separate because 'no data' from an approved
    source and 'no data' because the source was never approved are different
    facts about the world, and collapsing them is how 'unknown' turns into
    'zero'."""

    PRESENT = "present"
    SOURCE_NOT_DECLARED = "source_not_declared"
    SOURCE_NOT_APPROVED = "source_not_approved"
    NO_OBSERVATIONS = "no_observations"
    NO_PRIOR_GAMES = "no_prior_games"
    BELOW_SAMPLE_GATE = "below_sample_gate"
    FIELD_NOT_IN_CACHE = "field_not_in_cache"
    UNVERIFIED_FIELD_SEMANTICS = "unverified_field_semantics"
    AVAILABILITY_UNVERIFIABLE = "availability_unverifiable"


class AvailabilityClass(str, Enum):
    """How well we can defend the claim 'this was knowable before kickoff'."""

    #: a git-tracked artifact records that this exact value existed pre-kickoff
    PROVEN_PREGAME = "proven_pregame"
    #: derived from a cache with no observation timestamp; availability is the
    #: assumption "everything from week <= W-1 was known by week W"
    RECONSTRUCTED_WEEK_GRANULAR = "reconstructed_week_granular"
    #: availability cannot be established at all
    UNVERIFIABLE = "unverifiable"


#: claim kinds that assert a measure would have forecast something
PREDICTIVE_CLAIM_KINDS = ("predictive", "backtest", "calibration", "promotion")
#: claim kinds that only describe what happened
DESCRIPTIVE_CLAIM_KINDS = ("descriptive", "profile", "note", "exploratory")


def _require(name: str, value: Any) -> Any:
    if value is None:
        raise EvidenceContractError(
            f"{name} is required — an unprovenanced number is not an observation")
    if isinstance(value, str) and not value.strip():
        raise EvidenceContractError(f"{name} must be non-empty")
    return value


def _require_aware(name: str, value: Optional[dt.datetime]) -> dt.datetime:
    _require(name, value)
    assert value is not None
    if value.tzinfo is None:
        raise EvidenceContractError(
            f"{name} must be timezone-aware — a naive timestamp is a leak waiting "
            "to happen")
    return value


@dataclass(frozen=True)
class Observation:
    """One number (or one refusal) about one team-season, as of one week.

    Construct through `.present` or `.missing`. There is no third path, so a
    defaulted, mean-filled or zero-filled value cannot exist in this layer.
    """

    # identity
    measure_id: str
    team: str
    season: int
    as_of_week: int
    label: str
    unit: Optional[str]

    # the number
    value: Optional[float]
    n: int
    n_games: int

    # state
    availability: EvidenceAvailability
    missingness: Missingness
    availability_class: AvailabilityClass
    reason: Optional[str]

    # provenance
    source_id: Optional[str]
    source_url_or_id: Optional[str]
    source_version: Optional[str]
    retrieved_at: Optional[dt.datetime]
    effective_through: Optional[dt.datetime]

    # adjustment
    opponent_adjusted: bool
    opponent_adjustment_method: Optional[str]

    # audit
    game_ids: Tuple[int, ...] = ()
    caveat: Optional[str] = None

    # ---------------------------------------------------------------- ctors --
    @classmethod
    def present(cls, *, measure_id: str, team: str, season: int, as_of_week: int,
                label: str, unit: Optional[str], value: float, n: int,
                n_games: int, availability_class: AvailabilityClass,
                source_id: str, source_url_or_id: str, source_version: str,
                retrieved_at: dt.datetime, effective_through: dt.datetime,
                opponent_adjusted: bool = False,
                opponent_adjustment_method: Optional[str] = None,
                game_ids: Tuple[int, ...] = (),
                caveat: Optional[str] = None) -> "Observation":
        _require("value", value)
        _require("source_id", source_id)
        _require("source_url_or_id", source_url_or_id)
        _require("source_version", source_version)
        _require("availability_class", availability_class)
        _require_aware("retrieved_at", retrieved_at)
        _require_aware("effective_through", effective_through)
        if int(n) < 0 or int(n_games) < 0:
            raise EvidenceContractError("denominators cannot be negative")
        if opponent_adjusted and not opponent_adjustment_method:
            raise EvidenceContractError(
                "opponent_adjusted=True without naming the method is half a "
                "provenance — say which adjustment produced it")
        if not opponent_adjusted and opponent_adjustment_method:
            raise EvidenceContractError(
                "opponent_adjustment_method named on an unadjusted observation")
        return cls(
            measure_id=measure_id, team=team, season=int(season),
            as_of_week=int(as_of_week), label=label, unit=unit,
            value=float(value), n=int(n), n_games=int(n_games),
            availability=EvidenceAvailability.OK,
            missingness=Missingness.PRESENT,
            availability_class=AvailabilityClass(availability_class),
            reason=None,
            source_id=source_id, source_url_or_id=source_url_or_id,
            source_version=source_version, retrieved_at=retrieved_at,
            effective_through=effective_through,
            opponent_adjusted=bool(opponent_adjusted),
            opponent_adjustment_method=opponent_adjustment_method,
            game_ids=tuple(int(g) for g in game_ids), caveat=caveat)

    @classmethod
    def missing(cls, *, measure_id: str, team: str, season: int, as_of_week: int,
                label: str, unit: Optional[str],
                availability: EvidenceAvailability, missingness: Missingness,
                reason: str,
                availability_class: AvailabilityClass = AvailabilityClass.UNVERIFIABLE,
                n: int = 0, n_games: int = 0,
                source_id: Optional[str] = None,
                source_url_or_id: Optional[str] = None,
                source_version: Optional[str] = None,
                retrieved_at: Optional[dt.datetime] = None,
                caveat: Optional[str] = None) -> "Observation":
        if availability is EvidenceAvailability.OK:
            raise EvidenceContractError(
                "a missing observation cannot be OK — that is the whole point")
        if missingness is Missingness.PRESENT:
            raise EvidenceContractError(
                "a missing observation cannot claim missingness=PRESENT")
        _require("reason", reason)
        return cls(
            measure_id=measure_id, team=team, season=int(season),
            as_of_week=int(as_of_week), label=label, unit=unit,
            value=None, n=int(n), n_games=int(n_games),
            availability=availability, missingness=missingness,
            availability_class=AvailabilityClass(availability_class),
            reason=reason,
            source_id=source_id, source_url_or_id=source_url_or_id,
            source_version=source_version, retrieved_at=retrieved_at,
            effective_through=None,
            opponent_adjusted=False, opponent_adjustment_method=None,
            game_ids=(), caveat=caveat)

    # ----------------------------------------------------------- behaviour --
    @property
    def is_present(self) -> bool:
        return self.availability is EvidenceAvailability.OK

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        for k in ("availability", "missingness", "availability_class"):
            out[k] = out[k].value if hasattr(out[k], "value") else str(out[k])
        for k in ("retrieved_at", "effective_through"):
            v = getattr(self, k)
            out[k] = v.isoformat() if v is not None else None
        out["game_ids"] = list(self.game_ids)
        return out


def assert_retrospective_claim_allowed(obs: Observation, claim_kind: str) -> None:
    """Refuse a predictive claim built on availability we cannot defend.

    `01_architecture_audit.md` §2.2: nothing in `data/cfbd/*.parquet` records when
    a row became knowable. A profile built from those caches is honest as a
    DESCRIPTION and dishonest as evidence that the measure "would have predicted"
    anything, because the reconstruction cannot separate a Thursday game from the
    Saturday game in the same week.

    There is no override flag. If you need this to pass, fix availability at the
    source (stamp `observed_at` at write time) rather than at the assertion.
    """
    kind = str(claim_kind).strip().lower()
    if kind in DESCRIPTIVE_CLAIM_KINDS:
        return
    if kind not in PREDICTIVE_CLAIM_KINDS:
        raise EvidenceContractError(
            f"unknown claim kind {claim_kind!r}; predictive kinds are "
            f"{PREDICTIVE_CLAIM_KINDS}, descriptive kinds are "
            f"{DESCRIPTIVE_CLAIM_KINDS}")
    if obs.availability_class is not AvailabilityClass.PROVEN_PREGAME:
        raise RetrospectiveClaimBlocked(
            f"{obs.measure_id!r} for {obs.team} {obs.season} wk{obs.as_of_week} is "
            f"{obs.availability_class.value}, so a {kind} claim is blocked: no "
            "artifact records that this value existed before the relevant kickoff. "
            "See docs/team_evidence/00_contract.md section 3.")


@dataclass(frozen=True)
class TeamSeasonProfile:
    """Everything this layer knows about one team-season as of one week."""

    team: str
    season: int
    as_of_week: int
    generated_at_utc: dt.datetime
    observations: Tuple[Observation, ...]
    games_used: Tuple[int, ...]
    opponents_faced: Tuple[str, ...]
    neutral_site_games: Tuple[int, ...]
    personnel_facts: Tuple[Dict[str, Any], ...] = ()
    notes: Tuple[str, ...] = ()

    def by_measure(self) -> Dict[str, Observation]:
        return {o.measure_id: o for o in self.observations}

    def present(self) -> List[Observation]:
        return [o for o in self.observations if o.is_present]

    def blocked(self) -> List[Observation]:
        return [o for o in self.observations if not o.is_present]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "team_id": f"{self.season}:{self.team}",
            "season": self.season,
            "as_of_week": self.as_of_week,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "games_used": list(self.games_used),
            "opponents_faced": list(self.opponents_faced),
            "neutral_site_games": list(self.neutral_site_games),
            "observations": [o.to_dict() for o in self.observations],
            "personnel_facts": [dict(f) for f in self.personnel_facts],
            "notes": list(self.notes),
        }
