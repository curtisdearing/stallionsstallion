"""Source identity, approval gating, artifact versioning and retrieval time.

Two jobs:

1. Gate. Every measure names a `source_id` that must be declared in
   `data/context_sources.json`; `nflvalue.context.sources.SourceManifest` decides
   whether it may be read. There is no code-only path to a live source.

2. Version and time. Every observation must carry the version of the artifact it
   came from and when we obtained that artifact.

THE HONEST PART
---------------
`retrieved_at` for a parquet cache is the file's modification time, because
nothing in `ingest.py` records an observation time (`01_architecture_audit.md`
§2.2). That is when *we wrote the file*, not when the fact became knowable. So
`ArtifactVersion.availability_class` is `RECONSTRUCTED_WEEK_GRANULAR`, never
`PROVEN_PREGAME`, and it says so rather than letting an mtime pass for a
provenance timestamp.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..context.contracts import Availability
from ..context.sources import SourceManifest
from .contracts import AvailabilityClass, EvidenceAvailability, Missingness

#: how much of a parquet file to hash. The caches run to 2.6 MB; a full hash is
#: cheap and unambiguous, so we take the whole file rather than a size/mtime
#: fingerprint (which is what snapshots.py does, deliberately, for its own
#: latency reasons).
_CHUNK = 1 << 20


class SourceGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactVersion:
    """Identity of one on-disk input."""

    path: str
    sha256: str
    size_bytes: int
    mtime_utc: dt.datetime
    availability_class: AvailabilityClass

    @property
    def version(self) -> str:
        return f"sha256:{self.sha256[:16]}"

    @property
    def retrieved_at(self) -> dt.datetime:
        """When WE obtained the artifact. Not when the fact became knowable."""
        return self.mtime_utc


def artifact_version(path: str) -> ArtifactVersion:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            size += len(block)
            h.update(block)
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc)
    return ArtifactVersion(
        path=path, sha256=h.hexdigest(), size_bytes=size, mtime_utc=mtime,
        # An mtime is not an observation time. Anything derived from these caches
        # is a reconstruction, and this is where that is recorded once.
        availability_class=AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR)


@dataclass(frozen=True)
class SourceState:
    """The gate's answer for one source, plus everything needed to cite it."""

    source_id: str
    availability: EvidenceAvailability
    missingness: Missingness
    reason: Optional[str]
    source_url_or_id: Optional[str]
    version: Optional[str]
    retrieved_at: Optional[dt.datetime]
    availability_class: AvailabilityClass

    @property
    def ok(self) -> bool:
        return self.availability is EvidenceAvailability.OK


class EvidenceSources:
    """Approval gate plus artifact versioning for the evidence layer."""

    def __init__(self, data_dir: str, manifest: Optional[SourceManifest] = None):
        self.data_dir = data_dir
        self.manifest = manifest if manifest is not None else SourceManifest.load(data_dir)
        self._versions: Dict[str, ArtifactVersion] = {}

    # ------------------------------------------------------------------ gate --
    def state(self, source_id: str, *, seasons: Tuple[int, ...] = ()) -> SourceState:
        """Gate `source_id` and, if it passes, fingerprint the artifacts behind it."""
        try:
            declared = self.manifest.get(source_id)
        except Exception as exc:  # SourceError — an undeclared source
            return SourceState(
                source_id=source_id,
                availability=EvidenceAvailability.UNAVAILABLE_NO_SOURCE,
                missingness=Missingness.SOURCE_NOT_DECLARED,
                reason=str(exc), source_url_or_id=None, version=None,
                retrieved_at=None,
                availability_class=AvailabilityClass.UNVERIFIABLE)

        availability, reason = self.manifest.gate(source_id)
        if availability is not Availability.OK:
            return SourceState(
                source_id=source_id,
                availability=EvidenceAvailability.from_context(availability),
                missingness=Missingness.SOURCE_NOT_APPROVED,
                reason=reason, source_url_or_id=self.manifest.url(source_id),
                version=None, retrieved_at=None,
                availability_class=AvailabilityClass.UNVERIFIABLE)

        version, retrieved_at, aclass = self._fingerprint(declared, seasons)
        if version is None:
            return SourceState(
                source_id=source_id,
                availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                missingness=Missingness.NO_OBSERVATIONS,
                reason=("approved, but no cached artifact exists on disk for "
                        f"seasons {list(seasons)}"),
                source_url_or_id=self.manifest.url(source_id), version=None,
                retrieved_at=None, availability_class=AvailabilityClass.UNVERIFIABLE)

        return SourceState(
            source_id=source_id, availability=EvidenceAvailability.OK,
            missingness=Missingness.PRESENT, reason=None,
            source_url_or_id=self.manifest.url(source_id) or declared.get("cache"),
            version=version, retrieved_at=retrieved_at, availability_class=aclass)

    def derived_state(self, source_id: str, inputs) -> SourceState:
        """Gate a source whose value is COMPUTED at call time, not read from a file.

        The opponent-adjusted unit ratings are the case this exists for: they are
        produced by `ratings.solve_units` from the plays cache, and they never read
        `data/ratings_{season}.parquet`. Fingerprinting them against that file would
        be a false dependency — a missing 2025 ratings parquet would block a value
        that was computed from the 2025 plays cache and is perfectly well defined.

        So the IDENTITY (and the approval gate) is the named producer, and the
        VERSION is the combined version of the inputs it was actually computed
        from. Retrieval time is the newest of those inputs.
        """
        try:
            self.manifest.get(source_id)
        except Exception as exc:                              # SourceError
            return SourceState(
                source_id=source_id,
                availability=EvidenceAvailability.UNAVAILABLE_NO_SOURCE,
                missingness=Missingness.SOURCE_NOT_DECLARED, reason=str(exc),
                source_url_or_id=None, version=None, retrieved_at=None,
                availability_class=AvailabilityClass.UNVERIFIABLE)
        availability, reason = self.manifest.gate(source_id)
        if availability is not Availability.OK:
            return SourceState(
                source_id=source_id,
                availability=EvidenceAvailability.from_context(availability),
                missingness=Missingness.SOURCE_NOT_APPROVED, reason=reason,
                source_url_or_id=self.manifest.url(source_id), version=None,
                retrieved_at=None,
                availability_class=AvailabilityClass.UNVERIFIABLE)

        usable = [st for st in inputs if st is not None and st.ok and st.version]
        if not usable:
            return SourceState(
                source_id=source_id,
                availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                missingness=Missingness.NO_OBSERVATIONS,
                reason=("approved, but none of the inputs it is computed from are "
                        "available"),
                source_url_or_id=self.manifest.url(source_id), version=None,
                retrieved_at=None,
                availability_class=AvailabilityClass.UNVERIFIABLE)
        combined = hashlib.sha256(
            "|".join(f"{st.source_id}:{st.version}"
                     for st in sorted(usable, key=lambda s: s.source_id)).encode()
        ).hexdigest()
        stamps = [st.retrieved_at for st in usable if st.retrieved_at]
        return SourceState(
            source_id=source_id, availability=EvidenceAvailability.OK,
            missingness=Missingness.PRESENT, reason=None,
            source_url_or_id=self.manifest.url(source_id),
            version=f"derived:sha256:{combined[:16]}",
            retrieved_at=max(stamps) if stamps else None,
            availability_class=AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR)

    # ----------------------------------------------------------- versioning --
    def _fingerprint(self, declared: Dict, seasons: Tuple[int, ...]):
        """Combined version over every cache file the source resolves to.

        A source that spans several seasons gets ONE version string built from the
        per-file hashes in season order, so a change to any input changes the
        version. `retrieved_at` is the NEWEST mtime, which is the earliest moment
        at which all of the inputs were simultaneously on disk in this form.
        """
        template = declared.get("cache")
        if not template:
            # A model-derived source (e.g. model_ratings) has no cache file. Its
            # version is the version of what it was computed from, which the
            # caller supplies; here we only certify that it is approved.
            now = dt.datetime.now(dt.timezone.utc)
            return ("derived:no-cache-file", now,
                    AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR)

        repo_root = os.path.dirname(os.path.abspath(self.data_dir))
        parts, newest = [], None
        aclass = AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR
        for season in (sorted(seasons) or [None]):
            rel = template.format(season=season) if season is not None else template
            path = os.path.normpath(os.path.join(repo_root, rel))
            if not os.path.exists(path):
                continue
            av = self._versions.get(path)
            if av is None:
                av = artifact_version(path)
                self._versions[path] = av
            parts.append(f"{os.path.basename(path)}:{av.sha256[:16]}")
            newest = av.mtime_utc if newest is None else max(newest, av.mtime_utc)
            aclass = av.availability_class
        if not parts:
            return None, None, AvailabilityClass.UNVERIFIABLE
        combined = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return f"sha256:{combined[:16]}", newest, aclass
