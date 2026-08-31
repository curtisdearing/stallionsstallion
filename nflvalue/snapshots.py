"""Game-window snapshot store for the CFB weekly loop (Workstream I).

The weekly loop must be able to run `pregame` several times between Tuesday
and Saturday.  The original stage sentinel
(`data/manifests/_done_{season}_wk{NN}_pregame_simulate`) made that impossible:
the first pregame run of a week marked the stage done and every later run in
that week skipped it, so a Saturday-morning run could not react to a Friday
injury report or to the noon slate kicking off.

This module replaces "run once per week" with "run when something that could
change the answer has changed", and makes every run leave an immutable record.

Three ideas, kept deliberately separate:

1.  **Source fingerprint** (`collect_source_digest`) — a digest over the inputs
    that are *allowed* to move a forecast: the target week's schedule (ids,
    kickoff times, teams, neutral flag, division tier), the cached source
    artifacts on disk (games / lines / drives / teams / rosters / availability /
    weather), and the code+config version (config hash, git sha, seed).  When
    the digest moves, the forecast may legitimately move.

2.  **Kickoff window** (`kickoff_window`) — the next kickoff that has not
    happened yet, floored to the hour.  As each slate starts, the earliest
    *pending* kickoff advances, so a CFB Saturday produces a natural
    noon -> afternoon -> night sequence of windows without hardcoding clock
    bands for "noon", "afternoon" and "night".  `WINDOW_RESOLUTION_HOURS` is a
    scheduling resolution, not a fitted model parameter.

3.  **Freeze at kickoff** (`freeze_kicked_off`) — once a game starts, its final
    pregame row is copied out of the newest snapshot *taken strictly before
    that kickoff* and stored in `frozen.json`.  Every later run re-publishes
    that row verbatim.  This is the structural no-post-kickoff-leakage
    guarantee from BUILD_SPEC section 0.4: a published number for a started
    game can only have come from a snapshot that predates the start.  A game
    with no pre-kickoff snapshot is frozen as an explicit *refusal* — never
    back-filled from a later run, which would be leakage wearing a forecast's
    clothes.

Layout, under `data/snapshots/{season}_wk{NN}/`:

    {stamp}/predictions.csv      immutable; rewriting a stamp raises
    {stamp}/snapshot.json        stamp, digest, window, degraded, sha256, n
    {stamp}/explainer.html       optional, same immutability
    latest.json                  POINTER to the newest stamp (never a copy)
    frozen.json                  append-only game_id -> final pregame row

Nothing here ranks, prices, or recommends anything.  It is storage and
bookkeeping for a PAPER-ONLY research pipeline.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from . import atomic
from . import cfb_calendar
from . import forecast_contract as FC
from . import producers as PR

import pandas as pd

UTC = timezone.utc

WINDOW_NONE = "none"

# Resolution at which a pending kickoff is bucketed into a "window".  This is a
# scheduling knob (how often the loop is allowed to notice that a slate has
# started), NOT a fitted constant -- BUILD_SPEC section 8's "never guess a
# constant" rule is about model parameters.  One hour makes a CFB Saturday
# resolve into its natural noon / afternoon / night slates.
WINDOW_RESOLUTION_HOURS = 1

# Cached artifacts whose contents are allowed to change a pregame forecast.
# A file only contributes if it actually exists, so an unconfigured optional
# source (weather, availability) simply does not appear in the fingerprint.
SOURCE_PREFIXES = ("games", "lines", "drives", "teams", "rosters",
                   "availability", "weather", "talent", "returning")


class SnapshotIntegrityError(RuntimeError):
    """Stored snapshot state exists but cannot be trusted.

    Kept distinct from :class:`SnapshotError` because the two call for
    opposite reactions. A `SnapshotError` means "this operation was refused"
    -- a rerun of an immutable stamp, say. A `SnapshotIntegrityError` means
    the STORE is damaged, and the only safe response is to stop: every
    integrity failure here used to be swallowed into an empty dict or a
    `None`, which reads identically to "this week has no snapshots yet" and
    therefore lets the next run publish over a record it cannot see.
    """


class SnapshotError(RuntimeError):
    """Raised on any attempt to rewrite history."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip() or str(value).lower() == "nan"


def _jsonable(value):
    """NaN/NaT/numpy scalars -> plain JSON types, so a frozen row round-trips."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else float(value)
    if hasattr(value, "item"):                      # numpy scalar
        try:
            value = value.item()
        except Exception:                           # noqa: BLE001
            return str(value)
        return _jsonable(value)
    if isinstance(value, (bool, int, str)):
        return value
    if value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return None if pd.isna(value) else str(value)
    except (TypeError, ValueError):
        return str(value)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_utc(series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def week_dir(data_dir: str, season: int, week: int) -> str:
    return os.path.join(data_dir, "snapshots", f"{season}_wk{week:02d}")


def snapshot_dir(data_dir: str, season: int, week: int, stamp: str) -> str:
    return os.path.join(week_dir(data_dir, season, week), stamp)


def latest_path(data_dir: str, season: int, week: int) -> str:
    return os.path.join(week_dir(data_dir, season, week), "latest.json")


def frozen_path(data_dir: str, season: int, week: int) -> str:
    return os.path.join(week_dir(data_dir, season, week), "frozen.json")


# --------------------------------------------------------------------------- #
# 1. Source fingerprint
# --------------------------------------------------------------------------- #
def source_digest(parts: dict) -> str:
    """Stable, order-independent digest of a nested plain-data structure."""
    blob = json.dumps(parts, sort_keys=True, default=str,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _schedule_part(games: pd.DataFrame, season: int, week: int) -> list:
    if games is None or games.empty:
        return []
    wk = games[games["week"] == week] if "week" in games.columns else games
    if "season" in wk.columns:
        wk = wk[wk["season"] == season]
    cols = ("id", "start_date", "home_team", "away_team", "neutral_site",
            "home_fcs", "away_fcs")
    rows = []
    for rec in wk.to_dict("records"):
        rows.append([_jsonable(rec.get(c)) for c in cols])
    return sorted(rows, key=lambda r: (str(r[0]), str(r[1])))


def _sources_part(data_dir: str) -> dict:
    """Fingerprint the cached source artifacts.

    Uses (size, mtime_ns) rather than a content hash: the parquet caches are
    tens of megabytes and this runs on every pregame tick.  A touched-but-
    unchanged file therefore triggers a re-run -- the safe direction, and free,
    because the pregame job has no refresh stage and so spends no CFBD calls.
    """
    out = {}
    for sub in ("cfbd", "sources"):
        root = os.path.join(data_dir, sub)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.startswith(SOURCE_PREFIXES):
                continue
            p = os.path.join(root, name)
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
            out[name] = [st.st_size, st.st_mtime_ns]
    return out


def collect_source_digest(data_dir: str, season: int, week: int,
                          games: pd.DataFrame,
                          version: Optional[dict] = None) -> tuple:
    """(digest, parts) over everything allowed to move this week's forecast."""
    parts = {
        "schedule": _schedule_part(games, season, week),
        "sources": _sources_part(data_dir),
        "version": dict(version or {}),
    }
    return source_digest(parts), parts


# --------------------------------------------------------------------------- #
# 2. Kickoff windows
# --------------------------------------------------------------------------- #
def _pending(games: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if games is None or games.empty or "start_date" not in games.columns:
        return games.iloc[0:0] if games is not None else pd.DataFrame()
    starts = _as_utc(games["start_date"])
    return games[starts > pd.Timestamp(now)]


def pending_game_ids(games: pd.DataFrame, now: datetime) -> List[int]:
    """Game ids that have NOT kicked off yet, ascending."""
    pend = _pending(games, now)
    if pend.empty:
        return []
    return sorted(int(i) for i in pend["id"])


def kickoff_window(games: pd.DataFrame, now: datetime) -> str:
    """The next kickoff that has not happened, floored to the resolution.

    Returns WINDOW_NONE once every game in the frame has started.  As each
    slate kicks off the value advances, which is what lets a Saturday run
    three times without a hardcoded noon/afternoon/night table.
    """
    pend = _pending(games, now)
    if pend.empty:
        return WINDOW_NONE
    nxt = _as_utc(pend["start_date"]).min()
    if pd.isna(nxt):
        return WINDOW_NONE
    nxt = nxt.to_pydatetime().astimezone(UTC)
    step = timedelta(hours=WINDOW_RESOLUTION_HOURS)
    floored = nxt - timedelta(
        seconds=(nxt - nxt.replace(hour=0, minute=0, second=0, microsecond=0))
        .total_seconds() % step.total_seconds())
    return floored.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# 3. Snapshot store
# --------------------------------------------------------------------------- #
def write_snapshot(data_dir: str, season: int, week: int, stamp: str,
                   rows: pd.DataFrame, *, source_digest: str, window: str,
                   degraded: bool = False,
                   degraded_reason: Optional[str] = None,
                   taken_at: Optional[datetime] = None,
                   explainer_html: Optional[str] = None) -> dict:
    """Write one immutable snapshot and advance the `latest` pointer.

    Rewriting an existing stamp raises: history is append-only.  A degraded run
    still publishes -- visibly marked on every row -- because a silent gap is
    worse than a labelled one.
    """
    out_dir = snapshot_dir(data_dir, season, week, stamp)
    if os.path.exists(out_dir):
        raise SnapshotError(
            f"snapshot {stamp} already exists and is immutable — a rerun must "
            f"write a new stamp, never rewrite {out_dir}")

    # VALIDATE FIRST, then create anything. A contract mismatch must fail
    # before any public output exists -- not after an empty snapshot
    # directory has already been published into the store, where the next
    # run would read it as a stamp that produced nothing.
    #
    # The rows arrive already in the public contract, degradation included:
    # `forecast_contract.build_artifact(degraded=True)` gives every non-frozen
    # row `status="degraded"` and a reason, and withholds the numbers.
    #
    # This function used to bolt `frozen`, `degraded` and a mutated `note`
    # onto the frame on the way past. That is exactly how v1 came to mean two
    # things at one `schema_version`: a writer downstream of the schema
    # quietly widening it. The contract is closed now, so the snapshot stores
    # what it was handed and records run-level state in `snapshot.json`, where
    # run-level state belongs.
    body = FC.validate_frame(rows.copy(), where=f"write_snapshot {stamp}")
    statuses = set(body["status"]) if len(body) else set()
    if degraded and statuses and statuses <= {FC.Status.FROZEN.value}:
        pass                       # every row frozen: nothing left to degrade
    elif degraded and not (statuses & FC.RUN_DEGRADED_STATUSES):
        raise SnapshotError(
            f"snapshot {stamp} is being written degraded, but no row carries "
            f"a run-degraded status ({sorted(FC.RUN_DEGRADED_STATUSES)}). "
            "Degradation must be built into the rows by "
            "build_artifact(degraded=True), not stamped on afterwards, or the "
            "published numbers and the manifest disagree")

    os.makedirs(out_dir, exist_ok=False)
    csv_path = os.path.join(out_dir, "predictions.csv")
    body.to_csv(csv_path, index=False)
    if explainer_html is not None:
        with open(os.path.join(out_dir, "explainer.html"), "w") as fh:
            fh.write(explainer_html)

    taken = (taken_at or datetime.now(UTC)).astimezone(UTC)
    meta = {
        "stamp": stamp, "season": season, "week": week,
        "source_digest": source_digest, "window": window,
        "degraded": bool(degraded), "degraded_reason": degraded_reason,
        "taken_at": taken.isoformat(),
        "n_rows": int(len(body)),
        "schema_version": FC.SCHEMA_VERSION,
        "n_priced": int(body["status"].isin(FC.FORECAST_BEARING).sum())
        if len(body) else 0,
        "by_status": {s.value: int((body["status"] == s.value).sum())
                      for s in FC.Status} if len(body) else {},
        "sha256": _sha256_file(csv_path),
        "paper_only": True,
    }
    with open(os.path.join(out_dir, "snapshot.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # The pointer moves LAST and atomically. Until os.replace lands, a reader
    # sees the previous stamp; after it, the new one. There is no instant at
    # which `latest` names a snapshot that is not fully on disk.
    atomic.write_json(latest_path(data_dir, season, week),
                      {**meta, "path": out_dir})

    return {**meta, "path": out_dir}


def list_snapshots(data_dir: str, season: int, week: int) -> List[dict]:
    """Every snapshot for the week, oldest stamp first (replay order).

    Damaged state RAISES `SnapshotIntegrityError` rather than being skipped.
    Three kinds of manifest are untrustworthy: unreadable JSON, a `stamp` that
    does not match the directory it lives in (forged or duplicated identity),
    and content whose sha256 no longer matches what its own manifest recorded
    (tampered).

    Skipping them was the wrong instinct even though it kept a bad snapshot
    out of the freeze. A week whose store is damaged then reads as a week with
    FEWER snapshots -- and in the limit, as a week with none, which
    `decide` interprets as "nothing published yet, go ahead and publish". A
    damaged store must stop the run and be looked at, not quietly shrink.

    A directory with no `snapshot.json` at all is a different thing: an
    interrupted write that never became a snapshot. Those are ignored, because
    the atomic pointer means they were never pointed at.
    """
    out = []
    seen_stamps = set()
    week_root = week_dir(data_dir, season, week)
    for meta_path in sorted(glob.glob(os.path.join(week_root, "*",
                                                   "snapshot.json"))):
        path = os.path.dirname(meta_path)
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            raise SnapshotIntegrityError(
                f"{meta_path} could not be read ({exc}). A snapshot manifest "
                "that cannot be parsed is damaged state, not an absent "
                "snapshot") from None
        if meta.get("stamp") != os.path.basename(path):
            raise SnapshotIntegrityError(
                f"{meta_path} claims stamp {meta.get('stamp')!r} but lives in "
                f"{os.path.basename(path)!r}. A snapshot whose identity is "
                "forged cannot be replayed, and cannot be ignored either")
        if meta["stamp"] in seen_stamps:
            raise SnapshotIntegrityError(
                f"stamp {meta['stamp']} is claimed by more than one directory "
                f"under {week_root}")
        if not verify_snapshot(path):
            raise SnapshotIntegrityError(
                f"{path}/predictions.csv no longer matches the sha256 its own "
                "manifest recorded. The immutable record has been altered")
        seen_stamps.add(meta["stamp"])
        meta["path"] = path
        out.append(meta)
    return sorted(out, key=lambda m: str(m.get("stamp", "")))


def read_latest(data_dir: str, season: int, week: int) -> Optional[dict]:
    """The `latest` pointer, or None when the week genuinely has no snapshot.

    An absent pointer is a real state: nothing has been published for this
    week yet. A pointer that EXISTS but cannot be read, or that names a stamp
    which is not on disk, is damaged state and raises. Returning `None` for
    those meant `decide` read "no prior snapshot for this week" and published
    over a record it had simply failed to open.
    """
    path = latest_path(data_dir, season, week)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            latest = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SnapshotIntegrityError(
            f"{path} exists but could not be read ({exc}). Refusing to treat "
            "an unreadable pointer as 'no snapshot yet'") from None
    if not isinstance(latest, dict) or not latest.get("stamp"):
        raise SnapshotIntegrityError(
            f"{path} carries no stamp; the pointer is damaged")
    target = snapshot_dir(data_dir, season, week, str(latest["stamp"]))
    if not os.path.isdir(target):
        raise SnapshotIntegrityError(
            f"{path} points at stamp {latest['stamp']}, which is not on disk "
            f"({target}). The record it names is gone")
    return latest


def verify_snapshot(path: str) -> bool:
    """True when predictions.csv still matches the digest its manifest recorded."""
    try:
        with open(os.path.join(path, "snapshot.json")) as fh:
            meta = json.load(fh)
        return _sha256_file(os.path.join(path, "predictions.csv")) == meta["sha256"]
    except (OSError, ValueError, KeyError):
        return False


# --------------------------------------------------------------------------- #
# 4. The run decision
# --------------------------------------------------------------------------- #
def decide(data_dir: str, season: int, week: int, *, source_digest: str,
           window: str, degraded: bool = False, force: bool = False) -> dict:
    """Should this pregame run do real work?

    Pure: reads only the snapshot store, touches no source and spends no API
    budget.  Skips ONLY when the source fingerprint and the pending kickoff
    window are both unchanged and the last publication was not degraded.
    """
    latest = read_latest(data_dir, season, week)
    if force:
        return {"run": True, "reason": "forced by caller", "latest": latest}
    if latest is None:
        return {"run": True, "reason": "no prior snapshot for this week",
                "latest": None}
    if latest.get("source_digest") != source_digest:
        return {"run": True,
                "reason": (f"source digest changed "
                           f"{latest.get('source_digest')} -> {source_digest}"),
                "latest": latest}
    if latest.get("window") != window:
        return {"run": True,
                "reason": (f"kickoff window advanced "
                           f"{latest.get('window')} -> {window}"),
                "latest": latest}
    if latest.get("degraded") and not degraded:
        return {"run": True,
                "reason": "previous snapshot was degraded; sources are healthy now",
                "latest": latest}
    return {"run": False,
            "reason": (f"source fingerprint and kickoff window unchanged since "
                       f"{latest.get('stamp')}"),
            "latest": latest}


# --------------------------------------------------------------------------- #
# 5. Freeze at kickoff
# --------------------------------------------------------------------------- #
def load_frozen(data_dir: str, season: int, week: int) -> Dict[int, dict]:
    """Frozen rows for the week. A corrupt file RAISES; it never reads empty.

    This is the most consequential of the fail-closed changes. `frozen.json`
    is the record of which games have already kicked off and what was
    published for them before they did. Returning `{}` for an unreadable file
    told the caller "nothing has kicked off", so the next run would recompute
    a started game and publish a post-kickoff number as a pregame forecast --
    the exact leakage the freeze exists to make structurally impossible.
    """
    path = frozen_path(data_dir, season, week)
    if not os.path.exists(path):
        return {}                       # genuinely nothing frozen yet
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SnapshotIntegrityError(
            f"{path} exists but could not be read ({exc}). Refusing to treat "
            "an unreadable freeze record as 'no game has kicked off' — that "
            "would let a started game be re-forecast") from None
    if not isinstance(raw, dict):
        raise SnapshotIntegrityError(
            f"{path} is not a game_id -> row mapping; the freeze record is "
            "damaged")
    try:
        return {int(k): v for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise SnapshotIntegrityError(
            f"{path} contains a non-integer game id ({exc})") from None


def _save_frozen(data_dir: str, season: int, week: int,
                 frozen: Dict[int, dict]) -> None:
    os.makedirs(week_dir(data_dir, season, week), exist_ok=True)
    atomic.write_json(frozen_path(data_dir, season, week),
                      {str(k): v for k, v in sorted(frozen.items())})


#: Why a started game can end up with no forecast at all. Names the schema
#: explicitly because the commonest cause, at a version bump, is that the only
#: pre-kickoff snapshots for the week were written under the previous contract
#: and cannot be read back unambiguously.
NO_PREGAME_SNAPSHOT = (
    "REFUSED: no schema-v{version} pregame snapshot existed before kickoff — "
    "not back-filled, because a number produced after kickoff is not a "
    "forecast")


def _refusal_row(game, season: int, week: int, *,
                 model_family: str, model_version: str,
                 generated_at_utc: str) -> dict:
    """A full contract row for a started game that was never priced.

    `model_family` names the model that *would* have priced it, so the row
    still says which route it belonged to; `model_version` records that no
    fitted model stands behind it, which is the honest answer.
    """
    kickoff = game.get("start_date")
    slate, public_date = cfb_calendar.public_slate(season, week, kickoff)
    row = {column: None for column in FC.COLUMNS}
    row.update({
        "schema_version": FC.SCHEMA_VERSION,
        "season": int(season), "week": int(week),
        "game_id": int(game["id"]),
        "home": _jsonable(game.get("home_team")),
        "away": _jsonable(game.get("away_team")),
        "kickoff_utc": pd.to_datetime(kickoff, utc=True, errors="raise")
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "neutral_site": bool(game.get("neutral_site")),
        "slate": slate, "public_date": public_date,
        "status": FC.Status.REFUSED.value,
        "refusal_reason": NO_PREGAME_SNAPSHOT.format(
            version=FC.SCHEMA_VERSION),
        "model_family": model_family,
        "model_version": model_version,
        "generated_at_utc": generated_at_utc,
    })
    return row


def freeze_kicked_off(data_dir: str, season: int, week: int,
                      games: pd.DataFrame, now: datetime,
                      *, model_family: Optional[str] = None,
                      model_version: str = "none: no pregame snapshot",
                      generated_at_utc: Optional[str] = None) -> dict:
    """Freeze the final pregame row of every game that has now started.

    The row is taken from the newest snapshot whose `taken_at` is strictly
    before that game's kickoff.  Append-only: an already-frozen game is never
    re-frozen, so a later run cannot revise it even by accident.
    """
    frozen = load_frozen(data_dir, season, week)
    if games is None or games.empty or "start_date" not in games.columns:
        return {"n_frozen": len(frozen), "newly_frozen": []}

    snaps = list_snapshots(data_dir, season, week)
    cache: Dict[str, pd.DataFrame] = {}

    def _rows_for(stamp_path: str) -> pd.DataFrame:
        """Rows from one snapshot, but only if they satisfy the contract.

        A snapshot written under an earlier schema is history, not a
        mixed-schema artifact someone is trying to publish now -- so it is
        skipped rather than raised on. The cost is real and is stated on the
        row: a game whose only pre-kickoff snapshots predate the bump freezes
        as an explicit refusal naming the schema, never as a number rebuilt
        after kickoff. Bump between weeks, not inside one.
        """
        if stamp_path not in cache:
            try:
                body = FC.coerce_frame(
                    pd.read_csv(os.path.join(stamp_path, "predictions.csv")))
                FC.validate_frame(body, where=stamp_path)
            except (OSError, ValueError, FC.ContractError):
                body = pd.DataFrame()
            cache[stamp_path] = body
        return cache[stamp_path]

    starts = _as_utc(games["start_date"])
    newly = []
    for pos, game in enumerate(games.to_dict("records")):
        kickoff = starts.iloc[pos]
        if pd.isna(kickoff) or kickoff > pd.Timestamp(now):
            continue                                   # not started yet
        gid = int(game["id"])
        if gid in frozen:
            continue                                   # append-only

        source = None
        for meta in snaps:                             # oldest -> newest
            taken = pd.to_datetime(meta.get("taken_at"), errors="coerce",
                                   utc=True)
            if pd.isna(taken) or taken >= kickoff:
                continue                               # never post-kickoff
            body = _rows_for(meta["path"])
            if body.empty or "game_id" not in body.columns:
                continue
            hit = body[body["game_id"].astype("Int64") == gid]
            if len(hit):
                source = (meta["stamp"],
                          {k: _jsonable(v) for k, v in
                           hit.iloc[-1].to_dict().items()})

        if source is None:
            frozen[gid] = {"row": _refusal_row(
                               game, season, week,
                               model_family=(model_family
                                             or PR.production_family(week)),
                               model_version=model_version,
                               generated_at_utc=(
                                   generated_at_utc
                                   or now.astimezone(UTC).isoformat())),
                           "from_stamp": None,
                           "kickoff": _jsonable(kickoff.isoformat()),
                           "frozen_at": now.astimezone(UTC).isoformat()}
        else:
            stamp, row = source
            frozen[gid] = {"row": row, "from_stamp": stamp,
                           "kickoff": _jsonable(kickoff.isoformat()),
                           "frozen_at": now.astimezone(UTC).isoformat()}
        newly.append(gid)

    if newly:
        _save_frozen(data_dir, season, week, frozen)
    return {"n_frozen": len(frozen), "newly_frozen": sorted(newly)}


def apply_frozen(rows: pd.DataFrame, frozen: Dict[int, dict]) -> tuple:
    """Splice frozen rows back in verbatim, as `status="frozen"`.

    "Frozen" used to be an extra boolean column welded onto the frame here.
    It is a status now, which is what it always was: a row republished
    verbatim from a snapshot taken strictly before kickoff. A frozen row that
    was never priced keeps `status="refused"` and its reason -- the freeze is
    recorded either way in `frozen_from_stamp`, so no information is lost and
    no row has to claim two states at once.

    A frozen game missing from `rows` is re-appended rather than dropped: the
    full-schedule invariant survives freezing.
    """
    body = rows.copy()
    if "game_id" not in body.columns or not frozen:
        return body, 0

    def _spliced(record: dict, entry: dict) -> dict:
        merged = {**record, **entry["row"]}
        merged["frozen_from_stamp"] = entry.get("from_stamp")
        if merged.get("status") == FC.Status.PRICED.value:
            merged["status"] = FC.Status.FROZEN.value
        return merged

    records = body.to_dict("records")
    seen, out = set(), []
    for record in records:
        gid = int(record["game_id"])
        seen.add(gid)
        entry = frozen.get(gid)
        out.append(_spliced(record, entry) if entry else record)
    for gid, entry in sorted(frozen.items()):
        if gid not in seen:
            out.append(_spliced({column: None for column in FC.COLUMNS}, entry))

    result = pd.DataFrame(out).reindex(columns=list(FC.COLUMNS))
    FC.validate_frame(result, where="apply_frozen")
    return result, len(seen & set(frozen))
