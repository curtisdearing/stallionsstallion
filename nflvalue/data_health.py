"""What the pipeline must actually have before it may publish a number.

The freshness gate this replaces checked one feed -- the season's games file --
and hardcoded `schema_ok=True`, so it could only ever answer "is the schedule
recent?". Everything else the forecast depends on was unchecked: the drive
history the Monte Carlo is built from, the team table the refusal rule reads,
the ratings inputs, and the benchmark records the published explainer quotes
back at the reader as evidence. A run could therefore pass its own freshness
gate with no drives on disk at all.

Worse, passing was not load-bearing. A failed gate marked the run `degraded`
and then carried on into simulation and publication, so the failure mode the
gate exists to prevent -- confident-looking numbers over data nobody checked --
was reachable through the gate rather than around it.

This module answers a narrower, more useful question: *for this job, priced by
this model family, in this week, which capabilities does the run actually have
the inputs for?* A capability whose inputs are missing, malformed, empty,
stale, future-dated, or short of the target week is BLOCKED, and a job that
needs a blocked capability does not run. Optional capabilities are different in
kind: they never block, they only suppress the claims that cite them.

Design notes worth keeping:

* Timestamp, staleness and leakage semantics are NOT reimplemented here. Each
  checked file becomes a :class:`nflvalue.freshness.Feed` and the existing,
  tested :func:`nflvalue.freshness.gate` makes that call, so there is one
  definition of "stale" and one definition of "future-dated data is unusable".
* Historical caches carry no staleness threshold on purpose. `games_2021` is
  an immutable record of a finished season; calling it stale after eight days
  would block every run forever. Only the current season ages.
* Parquet is inspected through its footer (schema + row count) rather than
  loaded. A health check that costs as much as the forecast will get skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import freshness

#: Seasons of history the ratings and drive models are fitted over.
HISTORY_START = 2019

#: Age at which a current-season cache stops being trustworthy. Eight days
#: matches the weekly cadence with a full week of slack: a Tuesday refresh that
#: failed silently is caught before the following Tuesday's publication.
CURRENT_SEASON_STALENESS_HOURS = 24.0 * 8


@dataclass(frozen=True)
class InputSpec:
    """One file the pipeline reads, and what makes it usable."""

    name: str
    template: str                       # relative to the data dir; {season} ok
    kind: str                           # "parquet" | "csv" | "json"
    scope: str                          # "current" | "history" | "fixed"
    min_records: int
    required_columns: Tuple[str, ...] = ()
    key_columns: Tuple[str, ...] = ()   # must not be entirely null
    coverage: str = "none"              # "none" | "target_week"
    staleness_hours: Optional[float] = None
    why: str = ""


INPUTS: Dict[str, InputSpec] = {
    "games": InputSpec(
        name="games", template="cfbd/games_{season}.parquet", kind="parquet",
        scope="current", min_records=1,
        required_columns=("id", "season", "week", "start_date", "home_team",
                          "away_team"),
        key_columns=("id", "start_date"), coverage="target_week",
        staleness_hours=CURRENT_SEASON_STALENESS_HOURS,
        why="the schedule, the results, and the denominator every artifact "
            "must cover exactly"),
    "games_history": InputSpec(
        name="games_history", template="cfbd/games_{season}.parquet",
        kind="parquet", scope="history", min_records=1,
        required_columns=("id", "season", "week", "home_team", "away_team"),
        why="prior seasons the ratings are fitted over, and the previous "
            "season the preseason prior is built from"),
    "teams": InputSpec(
        name="teams", template="cfbd/teams_{season}.parquet", kind="parquet",
        scope="current", min_records=1,
        required_columns=("team", "is_fbs"), key_columns=("team",),
        staleness_hours=24.0 * 120,
        why="the FBS classification the refusal rule reads; without it the "
            "model cannot tell which games are inside its validated division"),
    "drives": InputSpec(
        name="drives", template="cfbd/drives_{season}.parquet", kind="parquet",
        scope="current", min_records=1,
        required_columns=("season", "week", "offense", "defense"),
        staleness_hours=CURRENT_SEASON_STALENESS_HOURS,
        why="the drive-level Monte Carlo is fitted on these; stale drives "
            "mean a simulation trained on a stale picture of the season"),
    "drives_history": InputSpec(
        name="drives_history", template="cfbd/drives_{season}.parquet",
        kind="parquet", scope="history", min_records=1,
        required_columns=("season", "week", "offense", "defense"),
        why="the drive-rate and pace models need prior seasons; without them "
            "there is no simulation to run"),
    "lines": InputSpec(
        name="lines", template="cfbd/lines_{season}.parquet", kind="parquet",
        scope="current", min_records=1,
        required_columns=("game_id",),
        staleness_hours=CURRENT_SEASON_STALENESS_HOURS,
        why="market comparison and CLV capture only; no forecast depends on "
            "it, so its absence suppresses a claim rather than blocking a run"),
    "evidence_manifest": InputSpec(
        name="evidence_manifest", template="evidence_manifest.json",
        kind="json", scope="fixed", min_records=1,
        why="the benchmark selection gate: no consumer may quote a model-MAE, "
            "Brier or calibration figure without it, and its absence is a "
            "hard block because the fallback would be reading whichever "
            "backtest file happens to be on disk"),
    "clv_log_backtest": InputSpec(
        name="clv_log_backtest", template="clv_log_backtest.csv", kind="csv",
        scope="fixed", min_records=100, required_columns=("slice", "clv_pts"),
        why="the registered CLV slices the explainer quotes as evidence"),
    "forecast_sd_basis": InputSpec(
        name="forecast_sd_basis", template="forecast_sd_basis.json",
        kind="json", scope="fixed", min_records=1,
        why="the registered out-of-sample basis without which no row may "
            "carry a forecast SD"),
    "backtest_early": InputSpec(
        name="backtest_early", template="backtest_early.csv", kind="csv",
        scope="fixed", min_records=300,
        required_columns=("season", "week", "residual"),
        why="the per-game evidence behind the early-week basis; without it "
            "that SD is a number nobody can audit"),
}

#: capability -> the inputs it cannot function without.
CAPABILITIES: Dict[str, Tuple[str, ...]] = {
    "schedule": ("games", "teams"),
    "ratings": ("games", "games_history"),
    "early_season_prior": ("games", "games_history", "teams"),
    "drive_simulation": ("games", "games_history", "teams", "drives",
                         "drives_history"),
    "benchmark_claims": ("evidence_manifest", "clv_log_backtest"),
    "forecast_sd_basis": ("forecast_sd_basis", "backtest_early"),
    "market_insight": ("lines",),
}

#: Capabilities that never block a run. Their absence suppresses the insights
#: that cite them -- which is the whole distinction the lines feed needs: no
#: forecast depends on a market line, but a displayed market comparison does.
OPTIONAL_CAPABILITIES = ("market_insight",)

#: Which model family needs which pricing capability.
FAMILY_CAPABILITY = {
    "kalman_montecarlo": "drive_simulation",
    "ridge_preseason_prior": "early_season_prior",
}


class HealthError(RuntimeError):
    """The health of the inputs could not be established."""


@dataclass
class InputResult:
    name: str
    ok: bool
    paths: List[str] = field(default_factory=list)
    n_records: Optional[int] = None
    age_hours: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "paths": self.paths, "n_records": self.n_records,
                "age_hours": (None if self.age_hours is None
                              else round(self.age_hours, 2)),
                "reason": self.reason}


@dataclass
class HealthReport:
    season: int
    week: int
    job: str
    model_family: str
    as_of: str
    inputs: Dict[str, InputResult]
    required: Tuple[str, ...]
    blocked: Tuple[str, ...]
    optional_unavailable: Tuple[str, ...]
    leakage_suspected: bool
    reasons: Tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.blocked and not self.leakage_suspected

    def capability_ok(self, capability: str) -> bool:
        return capability not in self.blocked

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "season": self.season, "week": self.week, "job": self.job,
            "model_family": self.model_family, "as_of": self.as_of,
            "required_capabilities": list(self.required),
            "blocked_capabilities": list(self.blocked),
            "optional_unavailable": list(self.optional_unavailable),
            "leakage_suspected": self.leakage_suspected,
            "reasons": list(self.reasons),
            "inputs": {name: result.to_dict()
                       for name, result in sorted(self.inputs.items())},
        }


# --------------------------------------------------------------------------- #
# File inspection
# --------------------------------------------------------------------------- #
def _mtime_utc(path: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)


def _parquet_facts(path: str) -> Tuple[List[str], int]:
    """(columns, rows) from the footer -- no data is read."""
    try:
        import pyarrow.parquet as pq
        handle = pq.ParquetFile(path)
        return list(handle.schema_arrow.names), int(handle.metadata.num_rows)
    except ImportError:
        import pandas as pd
        frame = pd.read_parquet(path)
        return list(frame.columns), int(len(frame))


def _csv_facts(path: str) -> Tuple[List[str], int]:
    import pandas as pd
    frame = pd.read_csv(path)
    return list(frame.columns), int(len(frame))


def _inspect(spec: InputSpec, path: str, season: int, week: int) -> Tuple[bool, str, Optional[int]]:
    """(ok, reason, n_records) for one concrete file."""
    if not os.path.exists(path):
        return False, f"missing: {path}", None
    if os.path.getsize(path) == 0:
        return False, f"empty file: {path}", 0
    try:
        if spec.kind == "parquet":
            columns, rows = _parquet_facts(path)
        elif spec.kind == "csv":
            columns, rows = _csv_facts(path)
        else:
            with open(path) as fh:
                payload = json.load(fh)
            columns, rows = list(payload) if isinstance(payload, dict) else [], 1
    except Exception as exc:  # noqa: BLE001 - any read failure is unusable
        return False, f"unreadable ({type(exc).__name__}: {exc}): {path}", None

    missing = [c for c in spec.required_columns if c not in columns]
    if missing:
        return False, f"schema: {path} is missing {missing}", rows
    if rows < spec.min_records:
        return False, (f"record sanity: {path} holds {rows} row(s), fewer than "
                       f"the {spec.min_records} this input needs"), rows

    if spec.coverage == "target_week" or spec.key_columns:
        import pandas as pd
        frame = (pd.read_parquet(path) if spec.kind == "parquet"
                 else pd.read_csv(path))
        for column in spec.key_columns:
            if column in frame.columns and frame[column].isna().all():
                return False, (f"record sanity: every {column} in {path} is "
                               "null"), rows
        if spec.coverage == "target_week":
            if "week" not in frame.columns:
                return False, f"coverage: {path} has no week column", rows
            covered = pd.to_numeric(frame["week"], errors="coerce")
            if not bool((covered == int(week)).any()):
                return False, (f"coverage: {path} holds no rows for "
                               f"{season} week {week}; the run would publish a "
                               "week its own source does not contain"), rows
    return True, "", rows


def _paths_for(spec: InputSpec, data_dir: str, season: int) -> List[str]:
    if spec.scope == "history":
        return [os.path.join(data_dir, spec.template.format(season=s))
                for s in range(HISTORY_START, int(season))]
    if spec.scope == "current":
        return [os.path.join(data_dir, spec.template.format(season=season))]
    return [os.path.join(data_dir, spec.template)]


def check_input(spec: InputSpec, data_dir: str, season: int, week: int,
                as_of: datetime) -> Tuple[InputResult, List[freshness.Feed]]:
    """Inspect every file this input covers; also emit its freshness feeds."""
    paths = _paths_for(spec, data_dir, season)
    result = InputResult(name=spec.name, ok=True, paths=paths)
    feeds: List[freshness.Feed] = []
    problems: List[str] = []
    total = 0

    if not paths:
        result.ok = False
        result.reason = f"{spec.name}: no files declared for season {season}"
        return result, feeds

    for path in paths:
        ok, reason, rows = _inspect(spec, path, season, week)
        total += rows or 0
        if not ok:
            problems.append(reason)
        feeds.append(freshness.Feed(
            name=spec.name,
            timestamp=(_mtime_utc(path) if os.path.exists(path) else None),
            n_records=(rows or 0), schema_ok=ok,
            load_bearing=False,          # set by the caller, per capability
            detail=reason or spec.why))

    result.n_records = total
    ages = [f.age_hours(as_of) for f in feeds
            if f.age_hours(as_of) is not None]
    result.age_hours = max(ages) if ages else None
    if problems:
        result.ok = False
        result.reason = "; ".join(problems[:3])
    return result, feeds


# --------------------------------------------------------------------------- #
# Requirements
# --------------------------------------------------------------------------- #
def required_capabilities(job: str, model_family: str) -> Tuple[str, ...]:
    """What this job, priced by this family, cannot proceed without.

    `grade` needs no pricing capability: it reads the artifact that was
    published pregame and never recomputes a forecast. It does need the
    schedule, because a stale games file means finals it cannot see, and a
    recap graded against a partial week is a wrong number rather than a
    missing one.
    """
    if job == "grade":
        return ("schedule", "ratings")
    pricing = FAMILY_CAPABILITY.get(model_family)
    if pricing is None:
        raise HealthError(
            f"no pricing capability declared for model family "
            f"{model_family!r}; refusing to guess which inputs a forecast "
            "needs")
    return ("schedule", "ratings", pricing, "forecast_sd_basis",
            "benchmark_claims")


def evaluate(data_dir: str, season: int, week: int, job: str,
             model_family: str, *, as_of: Optional[datetime] = None,
             staleness_hours: Optional[Dict[str, float]] = None
             ) -> HealthReport:
    """The health of every input this job depends on."""
    as_of = as_of or datetime.now(timezone.utc)
    required = required_capabilities(job, model_family)
    needed = {name for capability in required
              for name in CAPABILITIES[capability]}

    results: Dict[str, InputResult] = {}
    feeds: List[freshness.Feed] = []
    interesting = set(needed)
    for capability in OPTIONAL_CAPABILITIES:
        interesting |= set(CAPABILITIES[capability])

    for name in sorted(interesting):
        spec = INPUTS[name]
        result, spec_feeds = check_input(spec, data_dir, season, week, as_of)
        results[name] = result
        for feed in spec_feeds:
            feeds.append(freshness.Feed(
                name=feed.name, timestamp=feed.timestamp,
                n_records=feed.n_records, schema_ok=feed.schema_ok,
                load_bearing=(name in needed), detail=feed.detail))

    thresholds = {name: spec.staleness_hours
                  for name, spec in INPUTS.items()
                  if spec.staleness_hours is not None}
    # An input with no threshold is an immutable historical record; give it a
    # limit nothing can exceed rather than the module-level fallback, which
    # would declare every finished season stale.
    for name, spec in INPUTS.items():
        if spec.staleness_hours is None:
            thresholds[name] = float("inf")
    if staleness_hours:
        thresholds.update(staleness_hours)

    verdict = freshness.gate(feeds, as_of=as_of, staleness_hours=thresholds)

    unusable = set(verdict["missing_feeds"]) | set(verdict["stale_feeds"]) \
        | set(verdict["future_dated_feeds"])
    for name, result in results.items():
        if name in unusable and result.ok:
            result.ok = False
            result.reason = result.reason or (
                f"{name}: failed the freshness gate")

    blocked, optional_out = [], []
    for capability, inputs in CAPABILITIES.items():
        if any(not results[name].ok for name in inputs if name in results):
            if capability in OPTIONAL_CAPABILITIES:
                optional_out.append(capability)
            elif capability in required:
                blocked.append(capability)

    reasons = list(verdict["reasons"])
    for capability in blocked:
        broken = [results[n].reason for n in CAPABILITIES[capability]
                  if n in results and not results[n].ok]
        reasons.append(f"capability {capability} BLOCKED: " + "; ".join(broken))

    return HealthReport(
        season=int(season), week=int(week), job=job,
        model_family=model_family, as_of=verdict["as_of"], inputs=results,
        required=tuple(required), blocked=tuple(sorted(blocked)),
        optional_unavailable=tuple(sorted(optional_out)),
        leakage_suspected=bool(verdict["leakage_suspected"]),
        reasons=tuple(reasons))


# --------------------------------------------------------------------------- #
# The public, deliberately non-numeric status
# --------------------------------------------------------------------------- #
#: Written to the data dir; the hub feed forwards it verbatim.
PUBLIC_STATUS_FILE = "pipeline_health.json"

STATUS_OK = "ok"
STATUS_LIMITED = "limited"
STATUS_UNAVAILABLE = "data_unavailable"

HEADLINES = {
    STATUS_OK: "Forecasts are current.",
    STATUS_LIMITED: ("Forecasts are current; some context is unavailable and "
                     "the insights that cite it are suppressed."),
    STATUS_UNAVAILABLE: ("Forecast data unavailable — the inputs did not pass "
                         "their health check, so no new numbers were "
                         "published."),
}


def status_of(report: HealthReport) -> str:
    if not report.healthy:
        return STATUS_UNAVAILABLE
    return STATUS_LIMITED if report.optional_unavailable else STATUS_OK


def public_status(report: HealthReport, *, published: bool) -> dict:
    """A status a reader can act on that contains no forecast numbers at all.

    Deliberately non-numeric. The point of the unhealthy branch is that there
    is nothing trustworthy to say numerically; a status carrying a margin, a
    probability, or a count of "current" forecasts would be doing exactly what
    the halt exists to prevent -- looking fresh.
    """
    status = status_of(report)
    return {
        "schema_version": 1,
        "status": status,
        "headline": HEADLINES[status],
        "as_of": report.as_of,
        "season": report.season,
        "week": report.week,
        "job": report.job,
        "numeric_forecast_published": bool(published),
        "blocked_capabilities": list(report.blocked),
        "suppressed_insights": list(report.optional_unavailable),
        "leakage_suspected": report.leakage_suspected,
        "reasons": list(report.reasons),
        "paper_only": True,
    }


def write_status(data_dir: str, report: HealthReport, *,
                 published: bool) -> dict:
    """Write the public status atomically; return the payload written."""
    from . import atomic
    payload = public_status(report, published=published)
    atomic.write_json(os.path.join(data_dir, PUBLIC_STATUS_FILE), payload)
    return payload


def read_status(data_dir: str) -> Optional[dict]:
    """The last written status, or None when the pipeline has never run.

    A status file that EXISTS but cannot be parsed raises: an unreadable
    health record is not a healthy one, and treating it as absent is how a
    stale "ok" quietly survives a corrupted write.
    """
    path = os.path.join(data_dir, PUBLIC_STATUS_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise HealthError(
            f"{path} exists but could not be read ({exc}). A corrupt health "
            "record fails closed rather than reading as 'no problem'") from None


def mark_published(data_dir: str, *, published: bool) -> Optional[dict]:
    """Flip the already-written status to record whether numbers went out.

    The status is written when the health check runs, before anything is
    published, so that a crash between the two leaves the cautious answer on
    disk rather than a stale optimistic one. This updates it afterwards, in
    place and atomically, once the run knows what actually happened.
    """
    from . import atomic
    payload = read_status(data_dir)
    if payload is None:
        return None
    payload["numeric_forecast_published"] = bool(published)
    if not published and payload.get("status") == STATUS_OK:
        payload["headline"] = ("Inputs are healthy; no new forecast was "
                               "published on this run.")
    atomic.write_json(os.path.join(data_dir, PUBLIC_STATUS_FILE), payload)
    return payload
