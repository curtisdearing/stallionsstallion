"""Bounded structured personnel facts, with citations, dedup and hard refusals.

CURRENT STATE, STATED UP FRONT
------------------------------
`qb_availability_feed`, `unit_availability_feed` and `cfbd_portal` are all
`not_requested` in `data/context_sources.json`. So this channel emits
`UNAVAILABLE_NOT_APPROVED` for every team today. That is the correct state, not a
defect: the manifest is the only place a source may be declared, and there is no
code-only path to a live one. The machinery below is what runs the day Curtis
approves a feed, and it is tested now so approval is a data change rather than a
code change.

WHAT A FACT IS
--------------
A bounded record: subject identity, a claim from a closed vocabulary, a status from
a closed vocabulary, a confidence, a publication time, a retrieval time, a source,
a URL, and one quoted citation. Free prose beyond the quote is not a fact.

THE FOUR REFUSALS
-----------------
1. Absence is UNKNOWN. "No report found" never renders as healthy, cleared, or
   full-strength.
2. Syndication does not multiply. Ten wire copies of one report are one fact with
   `evidence_weight == 1`.
3. Attention is not quality. Article count, mention volume, media attention, poll
   rank and name familiarity cannot become measures.
4. Prose does not price. A fact whose text assigns a point value is rejected — a
   number that has not been through the model is not a model output.

And one more, structural: a fact with no URL, or whose source names a model rather
than a manifest source, raises. Unrestricted recall of historical games is not a
source.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..context.contracts import Availability
from ..context.sources import SourceManifest
from .contracts import (AvailabilityClass, EvidenceAvailability, Missingness,
                        Observation)

# --------------------------------------------------------------------------- #
# Closed vocabularies
# --------------------------------------------------------------------------- #
SUBJECT_TYPES = ("player", "unit", "team", "coach")

CLAIMS = ("availability", "depth_chart", "transfer", "suspension",
          "coaching_change", "injury_designation")

#: Deliberately NOT the NFL module's OK/RISK/OUT. `data/context_sources.json`'s
#: approval record for `qb_availability_feed` demands this vocabulary, and the
#: mismatch with `nflvalue/sources/availability.py` is recorded in the audit.
STATUSES = ("confirmed_available", "probable", "questionable", "doubtful",
            "out", "unknown")

CONFIDENCES = ("confirmed", "reported", "speculative")

#: Reason string used whenever nothing was found. Never "healthy".
NO_REPORT_REASON = "no_report_found_is_unknown_not_healthy"

#: Signals that must never become measures. Media volume is a measure of media.
FORBIDDEN_DERIVED_SIGNALS = frozenset({
    "article_count", "story_count", "mention_count", "mention_volume",
    "media_attention", "buzz", "coverage_volume", "ap_rank", "ap_poll",
    "coaches_poll", "poll_rank", "brand", "name_recognition", "team_familiarity",
})

#: A model is not a source.
_MODEL_SOURCE_TOKENS = ("llm", "gpt", "claude", "model_recall", "memory",
                        "training_data", "assistant")

#: Patterns that PRICE something, as distinct from patterns that merely mention
#: points. "He scored 14 points in the opener" is a fact about a game; "worth
#: about 3 points on the spread" is an unregistered model output in prose.
_NUM = r"[+-]?\d+(?:\.\d+)?"
_PTS = r"(?:point|points|pt|pts)"
_POINT_ADJUSTMENT = re.compile(
    # an explicitly signed point value: "+4 points", "-2.5 pts"
    rf"[+-]\d+(?:\.\d+)?\s*{_PTS}\b"
    # "worth about 3 points", "worth 6 pts"
    rf"|\bworth\s+(?:about\s+|roughly\s+|around\s+|~\s*)?{_NUM}\s*{_PTS}?"
    # "a 4 point swing", "3 points on the spread", "2 points of value"
    rf"|\b{_NUM}\s*{_PTS}\s+(?:swing|of value|on the (?:spread|line|total)"
    rf"|to the (?:spread|line|total))"
    # "adjust the line 3", "move the total 3.5 points down"
    rf"|\b(?:add|subtract|adjust|move|shift)\s+(?:the\s+)?(?:line|spread|total)\s+"
    rf"(?:by\s+)?{_NUM}"
    # "the line by 3 points"
    rf"|\b(?:line|spread|total)\s+by\s+{_NUM}\s*{_PTS}?",
    re.IGNORECASE)


class NewsContractError(RuntimeError):
    pass


class UnsourcedFactError(NewsContractError):
    """A fact with no citable origin, or whose origin is a language model."""


class ProsePricingError(NewsContractError):
    """Prose tried to assign a numerical point adjustment."""


def assert_no_point_adjustment(text: str, *, where: str = "fact") -> None:
    """Refuse prose that prices a fact.

    A point value that has not been through the model, a pre-registration and a
    calibration is an opinion wearing a number's clothes. If a personnel change is
    worth points, that is a modelling result, and it belongs in the model.
    """
    if not text:
        return
    hit = _POINT_ADJUSTMENT.search(str(text))
    if hit:
        raise ProsePricingError(
            f"{where} assigns a numerical point adjustment ({hit.group(0)!r}). "
            "Personnel prose describes what changed; it does not price it. See "
            "docs/team_evidence/00_contract.md section 6.")


def assert_not_an_attention_signal(name: str) -> None:
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if key in FORBIDDEN_DERIVED_SIGNALS:
        raise NewsContractError(
            f"{name!r} is a measure of media, not of football. Article counts, "
            "poll rank and name familiarity are refused as evidence signals.")


def _normalise_quote(text: str) -> str:
    """Fingerprint a citation so wire copies of one report collapse.

    Unicode-normalised, punctuation stripped, whitespace collapsed, lower-cased.
    Syndicated copies differ in the dateline, the headline and the trailing
    boilerplate, not in the sentence that carries the claim.
    """
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = t.encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalise_quote(text).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PersonnelFact:
    """One bounded, cited claim about one subject."""

    fact_id: str
    subject_type: str
    subject_id: str
    subject_name: str
    team: str
    season: int
    week: int
    claim: str
    status: str
    status_confidence: str
    published_at: dt.datetime
    retrieved_at: dt.datetime
    source_id: str
    source_url: str
    citation_quote: str
    uncertainty: str
    duplicate_of: Tuple[str, ...] = ()
    evidence_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.subject_type not in SUBJECT_TYPES:
            raise NewsContractError(f"subject_type {self.subject_type!r}")
        if self.claim not in CLAIMS:
            raise NewsContractError(f"claim {self.claim!r} is not in {CLAIMS}")
        if self.status not in STATUSES:
            raise NewsContractError(f"status {self.status!r} is not in {STATUSES}")
        if self.status_confidence not in CONFIDENCES:
            raise NewsContractError(
                f"status_confidence {self.status_confidence!r} is not in {CONFIDENCES}")
        for name in ("subject_id", "subject_name", "team", "citation_quote",
                     "uncertainty"):
            if not str(getattr(self, name) or "").strip():
                raise NewsContractError(f"{name} is required on a personnel fact")
        for name in ("published_at", "retrieved_at"):
            stamp = getattr(self, name)
            if stamp is None or stamp.tzinfo is None:
                raise NewsContractError(f"{name} must be timezone-aware")
        if not str(self.source_url or "").strip():
            raise UnsourcedFactError(
                f"{self.fact_id}: a fact with no URL cannot be checked. Unrestricted "
                "recall of historical games is not a source.")
        sid = str(self.source_id or "").lower()
        if not sid:
            raise UnsourcedFactError(f"{self.fact_id}: no source_id")
        if any(tok in sid for tok in _MODEL_SOURCE_TOKENS):
            raise UnsourcedFactError(
                f"{self.fact_id}: source_id {self.source_id!r} names a model. A "
                "model is not a source; features may not come from unrestricted "
                "recall of historical games.")
        assert_no_point_adjustment(self.citation_quote, where=f"{self.fact_id} quote")
        assert_no_point_adjustment(self.uncertainty, where=f"{self.fact_id} uncertainty")
        if float(self.evidence_weight) != 1.0:
            raise NewsContractError(
                "evidence_weight is 1 per canonical fact, always. Duplicate "
                "syndicated copies attach as duplicate_of and add nothing.")

    @property
    def dedup_key(self) -> Tuple[str, str, str, str, str]:
        """`(subject, claim, status, publication DATE, quote fingerprint)`.

        The date rather than the timestamp: syndicated copies of one report carry
        different filing minutes for the same underlying announcement.
        """
        return (self.subject_id, self.claim, self.status,
                self.published_at.astimezone(dt.timezone.utc).date().isoformat(),
                quote_fingerprint(self.citation_quote))

    def core_fields(self) -> Dict[str, Any]:
        """Every constructor argument except the dedup bookkeeping."""
        return {
            "fact_id": self.fact_id, "subject_type": self.subject_type,
            "subject_id": self.subject_id, "subject_name": self.subject_name,
            "team": self.team, "season": self.season, "week": self.week,
            "claim": self.claim, "status": self.status,
            "status_confidence": self.status_confidence,
            "published_at": self.published_at, "retrieved_at": self.retrieved_at,
            "source_id": self.source_id, "source_url": self.source_url,
            "citation_quote": self.citation_quote,
            "uncertainty": self.uncertainty,
        }

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.core_fields())
        out["published_at"] = self.published_at.isoformat()
        out["retrieved_at"] = self.retrieved_at.isoformat()
        out["duplicate_of"] = list(self.duplicate_of)
        out["evidence_weight"] = self.evidence_weight
        return out


def deduplicate(facts: Sequence[PersonnelFact]) -> List[PersonnelFact]:
    """Collapse syndicated copies. The canonical fact keeps `evidence_weight == 1`.

    Canonical = the EARLIEST publication time in the group (the original report,
    not the wire pickup), ties broken by `fact_id` so the result is deterministic.
    """
    groups: Dict[Tuple[str, str, str, str, str], List[PersonnelFact]] = {}
    for f in facts:
        groups.setdefault(f.dedup_key, []).append(f)
    out: List[PersonnelFact] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda f: (f.published_at, f.fact_id))
        head, rest = group[0], group[1:]
        out.append(PersonnelFact(
            **{**head.core_fields(),
               "duplicate_of": tuple(f.fact_id for f in rest),
               "evidence_weight": 1.0}))
    return sorted(out, key=lambda f: (f.published_at, f.fact_id))


def facts_before_kickoff(facts: Sequence[PersonnelFact],
                         kickoff: dt.datetime) -> List[PersonnelFact]:
    """Only what was published strictly before kickoff.

    Same rule as `context.contracts.kickoff_guard`: a report filed at or after
    kickoff was not knowable pregame, and a report filed later still cannot alter
    an earlier feature.
    """
    return [f for f in facts
            if f.published_at < kickoff and f.retrieved_at < kickoff]


def collect(team: str, season: int, week: int, *, data_dir: str,
            manifest: Optional[SourceManifest] = None,
            facts: Optional[Sequence[PersonnelFact]] = None,
            kickoff: Optional[dt.datetime] = None,
            source_id: str = "qb_availability_feed",
            ) -> Tuple[Observation, List[PersonnelFact]]:
    """Gate the feed, then return an observation plus the surviving facts.

    Returns an `UNAVAILABLE_NOT_APPROVED` observation while the feed is
    unapproved, and an `UNAVAILABLE_NO_DATA` observation carrying
    `no_report_found_is_unknown_not_healthy` when the feed is approved and silent.
    Neither is ever rendered as "available".
    """
    man = manifest if manifest is not None else SourceManifest.load(data_dir)
    label = "Quarterback availability"
    availability, reason = man.gate(source_id)
    if availability is not Availability.OK:
        return Observation.missing(
            measure_id="personnel_status_qb", team=team, season=season,
            as_of_week=week, label=label, unit="status",
            availability=EvidenceAvailability.from_context(availability),
            missingness=Missingness.SOURCE_NOT_APPROVED,
            reason=(reason or "source not approved"),
            availability_class=AvailabilityClass.UNVERIFIABLE,
            source_id=source_id, source_url_or_id=man.url(source_id)), []

    supplied = list(facts or [])
    if kickoff is not None:
        supplied = facts_before_kickoff(supplied, kickoff)
    surviving = [f for f in deduplicate(supplied)
                 if f.team == team and f.season == season and f.week == week]
    if not surviving:
        return Observation.missing(
            measure_id="personnel_status_qb", team=team, season=season,
            as_of_week=week, label=label, unit="status",
            availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
            missingness=Missingness.NO_OBSERVATIONS,
            reason=NO_REPORT_REASON,
            availability_class=AvailabilityClass.UNVERIFIABLE,
            source_id=source_id,
            source_url_or_id=man.url(source_id)), []

    # A present personnel observation is a STATUS, not a number, so it is returned
    # alongside the facts rather than as a value: the facts carry the citations,
    # and collapsing them to one float would discard exactly what makes them
    # checkable.
    return Observation.missing(
        measure_id="personnel_status_qb", team=team, season=season,
        as_of_week=week, label=label, unit="status",
        availability=EvidenceAvailability.LOW_COVERAGE,
        missingness=Missingness.NO_OBSERVATIONS,
        reason=(f"{len(surviving)} cited fact(s) available; rendered as facts with "
                "citations rather than collapsed into a single status value"),
        availability_class=AvailabilityClass.UNVERIFIABLE,
        source_id=source_id, source_url_or_id=man.url(source_id)), surviving
