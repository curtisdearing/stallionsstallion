"""Assemble the pregame context for one game — strictly after the fact.

The assembler is handed a game, frames the model has already finished
with, and (optionally) the finished projection. It returns a `GameContext`:
one value per declared field, each carrying its provenance and its display
bucket. It cannot alter the projection — the projection is copied into a
read-only mapping on the way in, and no provider ever sees it.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

from . import providers as P
from . import registry as REG
from . import sources as SRC
from .contracts import FIELDS, ContextValue, Usage

ET = ZoneInfo("America/New_York")


class AssemblyError(RuntimeError):
    pass


@dataclass
class Env:
    game: Dict[str, Any]
    frames: Dict[str, Any]
    manifest: SRC.SourceManifest
    registry: REG.FactorRegistry
    now: dt.datetime
    kickoff: dt.datetime
    _observed: Dict[str, dt.datetime] = field(default_factory=dict)

    @property
    def home(self): return self.game["home_team"]

    @property
    def away(self): return self.game["away_team"]

    @property
    def season(self): return int(self.game["season"])

    @property
    def week(self): return int(self.game["week"])

    @property
    def kickoff_et(self): return self.kickoff.astimezone(ET)

    def observed_at(self, source: str) -> Optional[dt.datetime]:
        if source in self._observed:
            return self._observed[source]
        frame = self.frames.get(P.FRAME_KEYS.get(source, ""))
        attrs = getattr(frame, "attrs", None) if frame is not None else None
        ts = attrs.get("observed_at") if attrs else None
        return ts


@dataclass(frozen=True)
class GameContext:
    game_id: int
    season: int
    week: int
    home: str
    away: str
    kickoff: dt.datetime
    generated_at: dt.datetime
    values: List[ContextValue]
    excluded_rows: List[Dict[str, Any]]
    source_states: List[Dict[str, Any]]
    projection: Mapping[str, Any]

    @property
    def kickoff_et(self) -> dt.datetime:
        return self.kickoff.astimezone(ET)

    def by_usage(self, usage: Usage) -> List[ContextValue]:
        return [v for v in self.values if v.usage is usage]

    def counts(self) -> Dict[str, int]:
        return {u.value: len(self.by_usage(u)) for u in Usage}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id, "season": self.season, "week": self.week,
            "home": self.home, "away": self.away,
            "kickoff_utc": self.kickoff.isoformat(),
            "kickoff_et": self.kickoff_et.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "counts": self.counts(),
            "values": [v.to_dict() for v in self.values],
            "excluded": self.excluded_rows,
            "sources": self.source_states,
            "projection": dict(self.projection),
            "display_only_guarantee":
                "No value in this object entered the projection. The context "
                "package cannot import the modelling modules and the "
                "modelling modules cannot import it; both directions are "
                "asserted by tests/test_context_leakage.py.",
        }


def _kickoff_of(game: Dict[str, Any]) -> dt.datetime:
    import pandas as pd
    ts = pd.to_datetime(game["start_date"], utc=True, format="ISO8601")
    return ts.to_pydatetime()


def assemble_game_context(*, game: Dict[str, Any], frames: Dict[str, Any],
                          data_dir: str, now: dt.datetime,
                          projection: Optional[Mapping[str, Any]] = None,
                          observed_at: Optional[Dict[str, dt.datetime]] = None
                          ) -> GameContext:
    manifest = SRC.SourceManifest.load(data_dir)
    registry = REG.FactorRegistry.load(data_dir)
    kickoff = _kickoff_of(game)
    env = Env(game=game, frames=frames, manifest=manifest, registry=registry,
              now=now, kickoff=kickoff, _observed=dict(observed_at or {}))

    values: List[ContextValue] = []
    source_states: List[Dict[str, Any]] = []
    for prov in P.all_providers():
        values.extend(prov.emit(env))
        gate, why = manifest.gate(prov.source)
        s = manifest.get(prov.source)
        source_states.append({
            "source": prov.source,
            "provider": s.get("provider"),
            "external": bool(s.get("external")),
            "approval": s["approval"]["status"],
            "approval_record": s["approval"]["decision_record"],
            "requires": s.get("requires"),
            "gate": gate.value,
            "gate_note": why,
            "source_url_or_id": s.get("source_url_or_id"),
            "observed_at": (env.observed_at(prov.source).isoformat()
                            if env.observed_at(prov.source) else None),
            "max_age_hours": s.get("max_age_hours"),
            "min_coverage": s.get("min_coverage"),
        })

    emitted = {v.key for v in values}
    declared = {s.key for s in FIELDS}
    if emitted != declared:
        raise AssemblyError(
            "the context layer dropped or invented fields: missing="
            f"{sorted(declared - emitted)} extra={sorted(emitted - declared)}")

    order = {s.key: i for i, s in enumerate(FIELDS)}
    values.sort(key=lambda v: order[v.key])

    return GameContext(
        game_id=int(game["id"]), season=int(game["season"]),
        week=int(game["week"]), home=game["home_team"], away=game["away_team"],
        kickoff=kickoff, generated_at=now, values=values,
        excluded_rows=registry.excluded_rows(), source_states=source_states,
        projection=MappingProxyType(dict(projection or {})))
