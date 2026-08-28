"""The factor registry as the single source of truth for factor status.

`data/factor_registry.json` already decides whether a factor is excluded,
display-only, under revalidation or gated-active. This module reads it and
nothing else. The context layer never carries its own opinion about a
factor's status, and there is no code path that promotes one — promotion
happens by editing the registry after a factor clears n-gate + matched
control + BH q<0.05, which is Workstream E's job, not this layer's.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#: Exclusions this project has already argued and settled. They are listed
#: here so a test can assert they are still excluded in the registry — the
#: constant is a tripwire, not a second source of truth.
PRESERVED_EXCLUSIONS = (
    "raw_time_of_possession",
    "penalty_counts",
    "raw_turnover_margin",
    "raw_red_zone_td_pct",
    "raw_pressure_rate",
)


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class FactorRegistry:
    entries: Dict[str, Dict[str, Any]]
    path: str

    @classmethod
    def load(cls, data_dir: str) -> "FactorRegistry":
        path = os.path.join(data_dir, "factor_registry.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        entries = {e["factor_id"]: e for e in raw}
        missing = [f for f in PRESERVED_EXCLUSIONS if f not in entries]
        if missing:
            raise RegistryError(
                f"standing exclusions {missing} are absent from {path} — "
                "refusing to run a context layer that cannot prove they are "
                "still excluded")
        return cls(entries=entries, path=path)

    def status(self, factor_id: Optional[str]) -> Optional[str]:
        if factor_id is None:
            return None
        entry = self.entries.get(factor_id)
        return entry.get("status") if entry else None

    def entry(self, factor_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return self.entries.get(factor_id) if factor_id else None

    def excluded_rows(self) -> List[Dict[str, Any]]:
        """Every excluded factor, rendered so its absence is explained."""
        return [{"factor_id": fid,
                 "description": e.get("description", ""),
                 "why": e.get("source_measurement", ""),
                 "source_doc": e.get("source_doc"),
                 "last_validated": e.get("last_validated")}
                for fid, e in sorted(self.entries.items())
                if e.get("status") == "excluded"]

    def gated_active(self) -> List[str]:
        return sorted(f for f, e in self.entries.items()
                      if e.get("status") == "gated_active")


def stale_factor_ids(entries, today) -> List[str]:
    """Factor ids whose revalidation date has passed while still active.

    `next_validation_due` is often a human phrase such as "2026 season end"
    rather than a date. An unparseable value is UNKNOWN, not overdue — a
    factor is never called stale on a guess, and never silently treated as
    fresh either: `unknown_due_dates` lists them for the caller to report.
    """
    import datetime as _dt
    out = []
    for e in entries:
        if e.get("status") != "gated_active":
            continue
        due = e.get("next_validation_due")
        if not isinstance(due, str):
            continue
        try:
            when = _dt.date.fromisoformat(due.strip())
        except ValueError:
            continue                      # not a date: unknown, not overdue
        if when < today:
            out.append(e["factor_id"])
    return sorted(out)


def unknown_due_dates(entries) -> List[str]:
    """Active factors whose revalidation date cannot be parsed."""
    import datetime as _dt
    out = []
    for e in entries:
        if e.get("status") != "gated_active":
            continue
        due = e.get("next_validation_due")
        if not isinstance(due, str):
            out.append(e["factor_id"])
            continue
        try:
            _dt.date.fromisoformat(due.strip())
        except ValueError:
            out.append(e["factor_id"])
    return sorted(out)
