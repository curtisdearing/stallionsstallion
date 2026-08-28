"""Source manifest, approval gate, freshness and coverage policy.

`data/context_sources.json` is the only place a source may be declared. A
source whose approval status is anything but `approved` cannot emit a
value: `SourceManifest.gate` returns the refusal state and the providers
return it verbatim. Adding a new external provider or a new dependency
therefore means editing that file with a named approver, a date and a
decision record — there is no code-only path to a live source.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .contracts import Availability

APPROVAL_STATES = ("approved", "pending", "not_requested", "denied")


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceManifest:
    sources: Dict[str, Dict[str, Any]]
    path: str

    @classmethod
    def load(cls, data_dir: str) -> "SourceManifest":
        path = os.path.join(data_dir, "context_sources.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        srcs = raw["sources"]
        for name, s in srcs.items():
            st = s.get("approval", {}).get("status")
            if st not in APPROVAL_STATES:
                raise SourceError(f"{name}: approval status {st!r} is not one "
                                  f"of {APPROVAL_STATES}")
        return cls(sources=srcs, path=path)

    def get(self, name: str) -> Dict[str, Any]:
        try:
            return self.sources[name]
        except KeyError:
            raise SourceError(
                f"{name!r} is not declared in {self.path} — an undeclared "
                "source cannot be read") from None

    def is_approved(self, name: str) -> bool:
        return self.get(name)["approval"]["status"] == "approved"

    def gate(self, name: str) -> Tuple[Availability, Optional[str]]:
        """`(OK, None)` if the source may be read, else the refusal + reason."""
        s = self.get(name)
        appr = s["approval"]
        if appr["status"] == "approved":
            return Availability.OK, None
        if s.get("provider") is None:
            return (Availability.UNAVAILABLE_NO_SOURCE,
                    f"no approved source: {appr['decision_record']}")
        return (Availability.UNAVAILABLE_NOT_APPROVED,
                f"approval {appr['status']}: {appr['decision_record']}")

    def url(self, name: str) -> Optional[str]:
        return self.get(name).get("source_url_or_id")

    def provider(self, name: str) -> Optional[str]:
        return self.get(name).get("provider")


def freshness(man: SourceManifest, name: str, *,
              observed_at: Optional[dt.datetime],
              now: dt.datetime) -> Availability:
    """STALE when the observation is older than the source's contract allows."""
    if observed_at is None:
        return Availability.UNAVAILABLE_NO_DATA
    max_age = man.get(name).get("max_age_hours")
    if max_age is None:
        return Availability.OK
    age_h = (now - observed_at).total_seconds() / 3600.0
    return Availability.STALE if age_h > float(max_age) else Availability.OK


def coverage_state(man: SourceManifest, name: str,
                   coverage: Optional[float]) -> Availability:
    """LOW_COVERAGE when the source covers less of the league than promised."""
    if coverage is None:
        return Availability.UNAVAILABLE_NO_DATA
    floor = man.get(name).get("min_coverage")
    if floor is None:
        return Availability.OK
    return (Availability.LOW_COVERAGE if float(coverage) < float(floor)
            else Availability.OK)


def confidence_for(man: SourceManifest, name: str,
                   availability: Availability) -> Optional[str]:
    """Coarse confidence label. Never 'high' for a degraded source."""
    if availability is not Availability.OK:
        return None
    s = man.get(name)
    if s.get("external"):
        return "medium"
    return "high" if float(s.get("min_coverage", 0)) >= 0.9 else "medium"
