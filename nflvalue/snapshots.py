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
    os.makedirs(out_dir, exist_ok=False)

    body = rows.copy()
    body["degraded"] = bool(degraded)
    if degraded:
        banner = ("DEGRADED: "
                  + (degraded_reason or "source freshness gate did not pass"))
        if "note" not in body.columns:
            body["note"] = None
        body["note"] = [banner if _blank(n) else f"{banner} | {n}"
                        for n in body["note"]]

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
        "n_priced": int((~body["veto"].astype(bool)).sum())
        if "veto" in body.columns else None,
        "sha256": _sha256_file(csv_path),
        "paper_only": True,
    }
    with open(os.path.join(out_dir, "snapshot.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    with open(latest_path(data_dir, season, week), "w") as fh:
        json.dump({**meta, "path": out_dir}, fh, indent=2, sort_keys=True)

    return {**meta, "path": out_dir}


def list_snapshots(data_dir: str, season: int, week: int) -> List[dict]:
    """Every snapshot for the week, oldest stamp first (replay order)."""
    out = []
    for meta_path in glob.glob(os.path.join(
            week_dir(data_dir, season, week), "*", "snapshot.json")):
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        meta["path"] = os.path.dirname(meta_path)
        out.append(meta)
    return sorted(out, key=lambda m: str(m.get("stamp", "")))


def read_latest(data_dir: str, season: int, week: int) -> Optional[dict]:
    path = latest_path(data_dir, season, week)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


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
    path = frozen_path(data_dir, season, week)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {int(k): v for k, v in raw.items()}


def _save_frozen(data_dir: str, season: int, week: int,
                 frozen: Dict[int, dict]) -> None:
    os.makedirs(week_dir(data_dir, season, week), exist_ok=True)
    with open(frozen_path(data_dir, season, week), "w") as fh:
        json.dump({str(k): v for k, v in sorted(frozen.items())}, fh,
                  indent=2, sort_keys=True)


def _refusal_row(game, season: int, week: int) -> dict:
    return {
        "season": season, "week": week, "id": int(game["id"]),
        "home": _jsonable(game.get("home_team")),
        "away": _jsonable(game.get("away_team")),
        "mc_margin": None, "mc_margin_sd": None, "mc_total": None,
        "p_home_win": None, "param_sd": None,
        "pred_margin": None, "band_pm": None,
        "veto": True,
        "note": ("REFUSED: no pregame snapshot existed before kickoff — "
                 "not back-filled, because a number produced after kickoff "
                 "is not a forecast"),
    }


def freeze_kicked_off(data_dir: str, season: int, week: int,
                      games: pd.DataFrame, now: datetime) -> dict:
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
        if stamp_path not in cache:
            try:
                cache[stamp_path] = pd.read_csv(
                    os.path.join(stamp_path, "predictions.csv"))
            except (OSError, ValueError):
                cache[stamp_path] = pd.DataFrame()
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
            if body.empty or "id" not in body.columns:
                continue
            hit = body[body["id"].astype("Int64") == gid]
            if len(hit):
                source = (meta["stamp"],
                          {k: _jsonable(v) for k, v in
                           hit.iloc[-1].to_dict().items()})

        if source is None:
            frozen[gid] = {"row": _refusal_row(game, season, week),
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
    """Splice frozen rows back in verbatim; mark every row `frozen` true/false.

    A frozen game that is missing from `rows` is re-appended rather than
    dropped -- the full-schedule invariant survives freezing.
    """
    body = rows.copy()
    if "id" not in body.columns:
        body["frozen"] = False
        return body, 0
    if not frozen:
        body["frozen"] = False
        return body, 0

    records = body.to_dict("records")
    seen, out = set(), []
    for rec in records:
        gid = int(rec["id"])
        seen.add(gid)
        if gid in frozen:
            out.append({**rec, **frozen[gid]["row"], "frozen": True})
        else:
            out.append({**rec, "frozen": False})
    for gid, entry in sorted(frozen.items()):
        if gid not in seen:
            out.append({**entry["row"], "frozen": True})

    return pd.DataFrame(out, columns=list(body.columns) + ["frozen"]
                        if "frozen" not in body.columns
                        else list(body.columns)), len(seen & set(frozen))
