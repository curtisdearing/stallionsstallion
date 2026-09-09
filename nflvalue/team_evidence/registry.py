"""Measure registry — `data/team_evidence_registry.json` is the source of truth.

A measure absent from that file cannot be emitted. A measure present but `blocked`
emits an explicit unavailability state and never a number. This mirrors
`nflvalue/context/registry.py`: the JSON is auditable by a human, the module is a
loader with a tripwire.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "stallionsstallion.team_evidence_registry.v1"

STATUSES = ("display_only", "gated_pending_source", "blocked")
SIDES = ("created", "allowed", "team")

#: Words that would turn a measurement into a claim it cannot support. These are
#: the specific overclaims the brief called out, kept as a tripwire so a future
#: measure cannot quietly reintroduce one.
FORBIDDEN_ID_TOKENS = (
    "pressure",       # a sack rate is not a pressure measurement
    "crowd",          # capacity is not crowd noise
    "noise",
    "broken",         # a large gain is not a broken assignment
    "blown",
    "bust",
    "assignment",
    "ol_rank",        # a win-loss record is not an offensive-line ranking
    "oline_rank",
)

#: A measure whose definition or caveat admits it is a proxy must say so in its
#: id, so a reader who never opens the registry still knows.
_PROXY_HINT = re.compile(r"\bprox(?:y|ies)\b", re.I)


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class MeasureSpec:
    measure_id: str
    label: str
    family: str
    side: str
    unit: Optional[str]
    definition: str
    denominator: str
    source_id: str
    opponent_adjusted: bool
    opponent_adjustment_method: Optional[str]
    min_n: int
    status: str
    availability_class: str
    blocked_reason: Optional[str]
    caveat: Optional[str]

    @property
    def is_blocked(self) -> bool:
        return self.status in ("blocked", "gated_pending_source")


@dataclass(frozen=True)
class MeasureRegistry:
    measures: Dict[str, MeasureSpec]
    path: str

    # ------------------------------------------------------------------ load --
    @classmethod
    def load(cls, data_dir: str) -> "MeasureRegistry":
        path = os.path.join(data_dir, "team_evidence_registry.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("schema") != SCHEMA:
            raise RegistryError(
                f"{path}: schema is {raw.get('schema')!r}, expected {SCHEMA!r}")
        out: Dict[str, MeasureSpec] = {}
        for entry in raw["measures"]:
            spec = cls._spec(entry, path)
            if spec.measure_id in out:
                raise RegistryError(f"{path}: duplicate measure {spec.measure_id!r}")
            out[spec.measure_id] = spec
        if not out:
            raise RegistryError(f"{path}: declares no measures")
        return cls(measures=out, path=path)

    @staticmethod
    def _spec(entry: Dict[str, Any], path: str) -> MeasureSpec:
        required = ("measure_id", "label", "family", "side", "definition",
                    "denominator", "source_id", "status", "availability_class")
        for key in required:
            if not entry.get(key):
                raise RegistryError(
                    f"{path}: measure {entry.get('measure_id')!r} is missing {key!r}")
        mid = str(entry["measure_id"])
        if entry["status"] not in STATUSES:
            raise RegistryError(f"{path}: {mid} has status {entry['status']!r}")
        if entry["side"] not in SIDES:
            raise RegistryError(f"{path}: {mid} has side {entry['side']!r}")
        blob = f"{mid} {entry['label']}".lower()
        for token in FORBIDDEN_ID_TOKENS:
            if token in blob:
                raise RegistryError(
                    f"{path}: {mid} uses the word {token!r} in its id or label. "
                    "That word claims more than the measurement supports — see "
                    "docs/team_evidence/00_contract.md section 2.")
        if entry["status"] in ("blocked", "gated_pending_source") and not entry.get("blocked_reason"):
            raise RegistryError(
                f"{path}: {mid} is {entry['status']} but names no reason. A blocked "
                "field without a reason is indistinguishable from a forgotten one.")
        prose = f"{entry.get('definition', '')} {entry.get('caveat') or ''}"
        if _PROXY_HINT.search(prose) and "proxy" not in mid.lower():
            raise RegistryError(
                f"{path}: {mid} describes itself as a proxy but does not say so in "
                "its id. A reader who never opens this file should still know.")
        adj = bool(entry.get("opponent_adjusted"))
        method = entry.get("opponent_adjustment_method")
        if adj and not method:
            raise RegistryError(f"{path}: {mid} is opponent-adjusted but names no method")
        if not adj and method:
            raise RegistryError(f"{path}: {mid} names an adjustment method but is not adjusted")
        return MeasureSpec(
            measure_id=mid, label=str(entry["label"]), family=str(entry["family"]),
            side=str(entry["side"]), unit=entry.get("unit"),
            definition=str(entry["definition"]), denominator=str(entry["denominator"]),
            source_id=str(entry["source_id"]), opponent_adjusted=adj,
            opponent_adjustment_method=method, min_n=int(entry.get("min_n") or 0),
            status=str(entry["status"]),
            availability_class=str(entry["availability_class"]),
            blocked_reason=entry.get("blocked_reason"), caveat=entry.get("caveat"))

    # ------------------------------------------------------------- accessors --
    def get(self, measure_id: str) -> MeasureSpec:
        try:
            return self.measures[measure_id]
        except KeyError:
            raise RegistryError(
                f"{measure_id!r} is not declared in {self.path} — an undeclared "
                "measure cannot be emitted") from None

    def ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.measures))

    def by_family(self, family: str) -> List[MeasureSpec]:
        return [s for s in self.measures.values() if s.family == family]

    def by_source(self, source_id: str) -> List[MeasureSpec]:
        return [s for s in self.measures.values() if s.source_id == source_id]

    def source_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({s.source_id for s in self.measures.values()}))

    def emittable(self) -> List[MeasureSpec]:
        return [s for s in self.measures.values() if s.status == "display_only"]

    def blocked(self) -> List[MeasureSpec]:
        return [s for s in self.measures.values() if s.is_blocked]
