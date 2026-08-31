"""The evidence manifest: the only way to choose a benchmark artifact.

A benchmark file on disk is not evidence about the current model. It is
evidence about whatever code produced it, over whatever window that code was
given, under whatever method that code implemented at the time. Those three
things drift, and when they do the file stops describing the model that is
running now — silently, because the filename does not change.

That is what happened here. `data/backtest_c.csv` was produced on 2026-07-30
by a walk-forward that fitted `PaceModel` on the full pooled multi-season
frame and OT parameters on the whole 2021-2025 window: both in-window
relative to the seasons being graded, both leaks against the walk-forward
guarantee the model claims. The file is still on disk, still named the same,
and the site, the explainer, the calibration report and the factor battery
all still read it — so a corrected model was being described in public by
numbers the corrected method cannot reproduce.

This module makes that impossible to do by accident:

* **Archival is a status, not a filename.** A record marked `archival` can
  never be returned by :meth:`Manifest.select`. It is preserved, labelled, and
  quotable only as history.
* **A candidate is not a claim.** A corrected run lands as `candidate`. It
  becomes `selected` only when a human writes that into the manifest against a
  stated selection rule — never because the numbers look better.
* **Selection is verified, not asserted.** Every read re-hashes the artifact,
  the code that produced it and the inputs it consumed, and compares the CSV
  header against the recorded schema. Any mismatch raises: an artifact whose
  producing code has changed underneath it is no longer certifiably the output
  of the method it claims.
* **No selected artifact means no number.** :func:`claim_or_notice` returns a
  truthful "awaiting revalidation" state instead of the last good figure,
  because the last good figure is exactly the misleading answer.

Nothing here evaluates anything or changes engine policy. It decides which
file a consumer is allowed to read, and refuses when the honest answer is
"we do not currently have one".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

#: Lives in the data dir; `scripts/build_evidence_manifest.py` writes it.
MANIFEST_FILE = "evidence_manifest.json"

#: Per-run records written by a backtest at the moment it produces an
#: artifact. The manifest is assembled from these, so the provenance of a
#: candidate is recorded by the run itself rather than typed in afterwards.
RUN_RECORD_DIR = os.path.join("evidence", "runs")


class EvidenceError(RuntimeError):
    """The manifest could not be read, or an artifact failed verification."""


class BenchmarkUnavailable(EvidenceError):
    """No artifact is currently selected for this role.

    Not an error in the pipeline: it is the correct state while a correction
    is being revalidated, and the correct thing for a consumer to do with it
    is show :data:`AWAITING_REVALIDATION` rather than a stale number.
    """


class Status(str, Enum):
    ARCHIVAL = "archival"    #: pre-correction; preserved, never a current claim
    CANDIDATE = "candidate"  #: produced by the current method, not yet accepted
    SELECTED = "selected"    #: the one artifact a consumer may quote

    def __str__(self) -> str:
        return self.value


#: The claims a benchmark can back. Kept separate because they are separate
#: models on separate windows: weeks 0-3 are priced by ridge + preseason prior
#: and evaluated on early-week residuals; weeks 4+ are priced by the Kalman
#: simulation and evaluated on the walk-forward gate-3 window. Pooling them
#: would average two different things into one meaningless figure.
ROLE_WK4PLUS = "benchmark_wk4plus"
ROLE_EARLY = "benchmark_early"
ROLES = (ROLE_WK4PLUS, ROLE_EARLY)

AWAITING_REVALIDATION = (
    "current correction awaiting revalidation — the benchmark for this claim "
    "is being re-derived under the corrected strict walk-forward method. "
    "Pre-correction figures are archival: they describe an earlier method, "
    "not the model running now, and are not shown here as claims about it.")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header(path: str) -> List[str]:
    with open(path) as handle:
        line = handle.readline().strip()
    return line.split(",") if line else []


@dataclass(frozen=True)
class ArtifactRecord:
    """One benchmark artifact and everything needed to trust or distrust it."""

    artifact_id: str
    role: str
    status: str
    path: str                      # relative to the repo root
    method: str
    description: str = ""
    engine_policy: Dict[str, Any] = field(default_factory=dict)
    seasons: List[int] = field(default_factory=list)
    weeks: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    n_rows: Optional[int] = None
    n_sims_per_game: Optional[int] = None
    exclusions: Dict[str, Any] = field(default_factory=dict)
    code_hashes: Dict[str, str] = field(default_factory=dict)
    input_hashes: Dict[str, str] = field(default_factory=dict)
    schema: List[str] = field(default_factory=list)
    content_sha256: str = ""
    generated_at_utc: str = ""
    metrics: Optional[Dict[str, Any]] = None
    selection_rule: str = ""
    not_selected_reason: str = ""
    #: Written only by an explicit human selection: why the rule is satisfied.
    #: Its presence is the difference between "this artifact was chosen" and
    #: "this artifact happened to be the newest one on disk".
    selection_acknowledgement: str = ""
    supersedes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArtifactRecord":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise EvidenceError(
                f"artifact record {payload.get('artifact_id')!r} carries "
                f"unrecognised fields {sorted(unknown)}; the manifest schema is "
                "closed so a field no reader understands cannot ride along")
        missing = [f for f in ("artifact_id", "role", "status", "path", "method")
                   if not payload.get(f)]
        if missing:
            raise EvidenceError(
                f"artifact record is missing {missing}: "
                f"{payload.get('artifact_id') or payload}")
        if payload["status"] not in {s.value for s in Status}:
            raise EvidenceError(
                f"{payload['artifact_id']}: status {payload['status']!r} is not "
                f"one of {sorted(s.value for s in Status)}")
        if payload["role"] not in ROLES:
            raise EvidenceError(
                f"{payload['artifact_id']}: role {payload['role']!r} is not one "
                f"of {list(ROLES)}")
        return cls(**payload)

    @property
    def selectable(self) -> bool:
        return self.status == Status.SELECTED.value


class Manifest:
    """The manifest, and the only selection API consumers may use."""

    def __init__(self, payload: Dict[str, Any], root: str) -> None:
        self.root = root
        self.schema_version = int(payload.get("schema_version", 0))
        if self.schema_version != SCHEMA_VERSION:
            raise EvidenceError(
                f"evidence manifest schema_version {self.schema_version} is not "
                f"{SCHEMA_VERSION}; refusing to read a manifest this code does "
                "not understand")
        self.generated_at_utc = payload.get("generated_at_utc", "")
        self.note = payload.get("note", "")
        self.records: Dict[str, ArtifactRecord] = {}
        for item in payload.get("artifacts", []):
            record = ArtifactRecord.from_dict(item)
            if record.artifact_id in self.records:
                raise EvidenceError(
                    f"duplicate artifact_id {record.artifact_id!r}")
            self.records[record.artifact_id] = record
        for role in ROLES:
            chosen = [r for r in self.records.values()
                      if r.role == role and r.selectable]
            if len(chosen) > 1:
                raise EvidenceError(
                    f"role {role} has {len(chosen)} selected artifacts "
                    f"({[r.artifact_id for r in chosen]}); exactly one may be "
                    "selected, or none")

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def path_for(cls, root: str) -> str:
        return os.path.join(root, "data", MANIFEST_FILE)

    @classmethod
    def load(cls, root: str) -> "Manifest":
        path = cls.path_for(root)
        if not os.path.exists(path):
            raise EvidenceError(
                f"no evidence manifest at {path}. No benchmark artifact may be "
                "read without one — a file on disk is not evidence about the "
                "current model until the manifest says which method produced "
                "it and whether it is selected")
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise EvidenceError(
                f"{path} could not be read ({exc}). A damaged manifest fails "
                "closed; it never degrades to 'select whatever is on disk'"
            ) from None
        return cls(payload, root)

    # -- selection -------------------------------------------------------- #
    def for_role(self, role: str) -> List[ArtifactRecord]:
        if role not in ROLES:
            raise EvidenceError(f"unknown role {role!r}")
        return [r for r in self.records.values() if r.role == role]

    def select(self, role: str, *, verify: bool = True) -> ArtifactRecord:
        """The one artifact a consumer may quote for this role.

        Raises :class:`BenchmarkUnavailable` when nothing is selected. That is
        a state to display, not an error to swallow.
        """
        chosen = [r for r in self.for_role(role) if r.selectable]
        if not chosen:
            archival = [r.artifact_id for r in self.for_role(role)
                        if r.status == Status.ARCHIVAL.value]
            candidates = [r.artifact_id for r in self.for_role(role)
                          if r.status == Status.CANDIDATE.value]
            raise BenchmarkUnavailable(
                f"no artifact is selected for {role}. "
                f"archival: {archival or 'none'}; awaiting acceptance: "
                f"{candidates or 'none'}. {AWAITING_REVALIDATION}")
        record = chosen[0]
        # Status is a field in a JSON file, so "archival" alone is only as
        # strong as whoever last edited it. An artifact with no recorded code
        # provenance can never be selected whatever its status says: there is
        # nothing to verify it against, and that is precisely the property the
        # pre-correction artifacts have. The builder refuses to select them;
        # this refuses to honour a hand-edit that did.
        if not record.code_hashes:
            raise EvidenceError(
                f"{record.artifact_id} is marked selected but records no code "
                "provenance, so nothing can establish which method produced "
                "it. An artifact that cannot be verified cannot back a claim "
                "about the current model — re-run the producer so a run "
                "record exists, or leave the role unselected")
        if verify:
            self.verify(record)
        return record

    # -- verification ----------------------------------------------------- #
    def verify(self, record: ArtifactRecord) -> ArtifactRecord:
        """Re-derive every hash the record claims. Any mismatch raises.

        Content, code and inputs are checked separately because they fail for
        different reasons and the operator needs to know which: a content
        mismatch means the artifact was edited, a code mismatch means the
        producing method changed underneath it, an input mismatch means the
        data it was computed from is not the data on disk any more.
        """
        path = os.path.join(self.root, record.path)
        if not os.path.exists(path):
            raise EvidenceError(
                f"{record.artifact_id}: {record.path} is not on disk")

        if record.content_sha256:
            actual = sha256_file(path)
            if actual != record.content_sha256:
                raise EvidenceError(
                    f"{record.artifact_id}: content sha256 mismatch for "
                    f"{record.path} (manifest {record.content_sha256[:12]}…, "
                    f"file {actual[:12]}…). The artifact has been edited since "
                    "it was recorded")

        if record.schema:
            header = csv_header(path)
            if header != list(record.schema):
                raise EvidenceError(
                    f"{record.artifact_id}: schema mismatch for {record.path}. "
                    f"manifest expects {list(record.schema)}, file has "
                    f"{header}")

        for label, declared in (("code", record.code_hashes),
                                ("input", record.input_hashes)):
            for rel, expected in sorted(declared.items()):
                target = os.path.join(self.root, rel)
                if not os.path.exists(target):
                    raise EvidenceError(
                        f"{record.artifact_id}: declared {label} {rel} is "
                        "missing, so the artifact's provenance cannot be "
                        "checked")
                actual = sha256_file(target)
                if actual != expected:
                    raise EvidenceError(
                        f"{record.artifact_id}: {label} hash mismatch for "
                        f"{rel} (manifest {expected[:12]}…, file "
                        f"{actual[:12]}…). This artifact is no longer "
                        f"certifiably the output of the {label} it names")
        return record

    def verify_all(self) -> Dict[str, str]:
        """Verify every record. Returns {artifact_id: 'ok' | reason}."""
        out = {}
        for artifact_id, record in sorted(self.records.items()):
            try:
                self.verify(record)
                out[artifact_id] = "ok"
            except EvidenceError as exc:
                out[artifact_id] = str(exc)
        return out


# --------------------------------------------------------------------------- #
# The consumer API
# --------------------------------------------------------------------------- #
def claim_or_notice(root: str, role: str) -> Tuple[Optional[ArtifactRecord],
                                                   Optional[str]]:
    """`(record, None)` when a verified artifact is selected, else `(None, notice)`.

    The whole point of returning a notice rather than raising is that a
    consumer must render something. What it must NOT render is the previous
    number, and this signature makes the honest branch the easy one.
    """
    try:
        return Manifest.load(root).select(role), None
    except BenchmarkUnavailable as exc:
        return None, str(exc)
    except EvidenceError as exc:
        return None, (f"benchmark unavailable: {exc}. " + AWAITING_REVALIDATION)


def benchmark_frame(root: str, role: str):
    """The selected artifact as a DataFrame, or raise. Verified first."""
    import pandas as pd
    record = Manifest.load(root).select(role)
    return pd.read_csv(os.path.join(root, record.path)), record


def run_record_dir(root: str) -> str:
    return os.path.join(root, "data", RUN_RECORD_DIR)


def write_run_record(root: str, payload: Dict[str, Any]) -> str:
    """Persist one run's provenance beside the artifact it produced."""
    from . import atomic
    directory = run_record_dir(root)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{payload['artifact_id']}.json")
    atomic.write_json(path, payload)
    return path


def load_run_records(root: str) -> List[Dict[str, Any]]:
    directory = run_record_dir(root)
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as handle:
            out.append(json.load(handle))
    return out


def benchmark_or_notice(root: str, role: str = ROLE_WK4PLUS):
    """`(frame, record, None)` when a verified artifact is selected, else
    `(None, None, notice)`.

    The one call every consumer makes. Returning the notice rather than
    raising is deliberate: a consumer must render SOMETHING, and the shape of
    this return makes the honest branch the path of least resistance. What
    none of them may do is fall back to the previous number.
    """
    record, notice = claim_or_notice(root, role)
    if record is None:
        return None, None, notice
    import pandas as pd
    return pd.read_csv(os.path.join(root, record.path)), record, None
