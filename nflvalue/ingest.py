"""CFBD data ingest — season-partitioned parquet cache for college football.

Workstream A. Replaces the nflverse/NFL-shaped ingest that shipped with the
fablesfable duplication (that version never imported here; it assumed
nflverse column names, NFL season conventions, and NFL schedule density).

    data/cfbd/games_{year}.parquet       schedule + final scores
    data/cfbd/plays_{year}.parquet       play-by-play, RAW (every play)
    data/cfbd/plays_clean_{year}.parquet play-by-play, garbage time REMOVED
    data/cfbd/lines_{year}.parquet       one row per (game, book) w/ open+close
    data/cfbd/talent_{year}.parquet      247 composite, sentinel-aware
    data/cfbd/teams_{year}.parquet       per-season conference + classification
    data/cfbd/returning_{year}.parquet   returning production
    data/cfbd/_call_tally.json           running monthly CFBD call count

Design notes that are load-bearing — read before changing anything:

* **Per-season conference lookup, never hardcoded.** Conference membership
  comes from `/teams?year=`, re-pulled per season. CFB realignment moves teams
  constantly; a hardcoded map is wrong within one offseason.

* **FCS is a separate tier, not a weak FBS team.** `/teams` carries
  `classification`; games against non-FBS opponents are tagged
  `opponent_fcs=True` so the ratings fit can down-weight or exclude them
  rather than swallowing a 70-3 result as evidence about FBS strength.

* **Garbage time is filtered AT INGEST, before any rate stat exists.** Both
  the raw and filtered play tables are written, so the threshold stays
  tunable and nothing is destroyed. Filtering after aggregation is the
  classic way to end up with a "clean" number that quietly isn't.

* **talent == 0 is a MISSING-DATA SENTINEL, not a measurement.** CFBD serves
  0 for Army, Navy and Air Force — service academies whose recruits are
  largely unranked in the 247 composite. The lowest genuine value league-wide
  is 4.33. Ingested as 0.0 these become the least-talented teams in FBS when
  they are simply unmeasured, and Air Force is a Mountain West team, so this
  sits inside the first modelling slice. Here they become NULL plus
  `unranked_recruiting=True`. Curtis's call, 2026-07-18; gate 0 caught it.

* **Staleness fails loud.** A refresh that cannot reach CFBD KEEPS the cached
  parquet and returns `stale=True`. It never silently serves old data as
  fresh, and never half-writes: every write is temp-file + atomic rename.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .sources.cfbd import CFBDClient, CFBDError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cfbd")
TALLY_PATH = os.path.join(DATA_DIR, "_call_tally.json")
CALL_BUDGET_MONTHLY = 1000

#: talent == this means "not measured", never "measured as zero".
TALENT_MISSING_SENTINEL = 0.0

#: Garbage-time thresholds: a play is garbage time if the score margin exceeds
#: the threshold for its quarter. This is the widely used college-football
#: definition (Connelly / Football Study Hall). Parameterised on purpose —
#: BUILD_SPEC requires the threshold stay tunable, and CFB blowouts are far
#: more common than NFL ones, so this filter matters much more here than it
#: did in the NFL parent project.
GARBAGE_TIME_MARGIN_BY_PERIOD = {1: 43, 2: 37, 3: 27, 4: 22}


class SchemaError(RuntimeError):
    """A cached/pulled frame did not match its declared contract."""


# --------------------------------------------------------------------------- #
# Schema contracts — asserted on every load, so upstream API drift breaks a
# test instead of quietly corrupting a rating.
# --------------------------------------------------------------------------- #
SCHEMAS: Dict[str, Dict[str, str]] = {
    "games": {
        "id": "integer", "season": "integer", "week": "integer",
        "home_team": "string", "away_team": "string",
        "home_points": "numeric", "away_points": "numeric",
        "neutral_site": "boolean", "home_fcs": "boolean", "away_fcs": "boolean",
    },
    "plays": {
        "game_id": "integer", "season": "integer", "week": "integer",
        "period": "integer", "offense": "string", "defense": "string",
        "down": "numeric", "distance": "numeric", "yards_to_goal": "numeric",
        "play_type": "string", "ppa": "numeric",
        "offense_score": "numeric", "defense_score": "numeric",
        "garbage_time": "boolean",
    },
    "lines": {
        "game_id": "integer", "season": "integer", "provider": "string",
        "spread": "numeric", "over_under": "numeric",
    },
    "talent": {
        "season": "integer", "team": "string", "talent": "numeric",
        "unranked_recruiting": "boolean",
    },
    "teams": {
        "season": "integer", "team": "string", "conference": "string",
        "classification": "string", "is_fbs": "boolean",
    },
    "returning": {
        "season": "integer", "team": "string", "percent_ppa": "numeric",
    },
}


def _kind_ok(series: pd.Series, kind: str) -> bool:
    dt = series.dtype
    if kind == "integer":
        return pd.api.types.is_integer_dtype(dt)
    if kind == "numeric":
        return pd.api.types.is_numeric_dtype(dt)
    if kind == "boolean":
        return pd.api.types.is_bool_dtype(dt)
    if kind == "string":
        return (pd.api.types.is_string_dtype(dt)
                or isinstance(dt, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(dt))
    raise ValueError(f"unknown kind {kind!r}")


def assert_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Raise SchemaError unless `df` satisfies the contract for `name`."""
    contract = SCHEMAS[name]
    missing = [c for c in contract if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{name}: missing required columns {missing}. "
            f"Present: {sorted(df.columns)[:20]}. "
            "This usually means the CFBD response shape changed — fix the "
            "mapping, do not delete the contract."
        )
    bad = [f"{c}(want {k}, got {df[c].dtype})"
           for c, k in contract.items() if not _kind_ok(df[c], k)]
    if bad:
        raise SchemaError(f"{name}: wrong dtypes: {bad}")
    return df


# --------------------------------------------------------------------------- #
# Atomic IO
# --------------------------------------------------------------------------- #
def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write via temp file + rename so an interrupted refresh can never leave
    a half-written parquet that a later load would happily read."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _record_calls(n: int) -> Dict:
    """Add `n` to the running monthly tally against the 1,000/month free cap."""
    os.makedirs(DATA_DIR, exist_ok=True)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    tally = {}
    if os.path.exists(TALLY_PATH):
        try:
            with open(TALLY_PATH) as fh:
                tally = json.load(fh)
        except Exception:  # noqa: BLE001
            tally = {}
    tally[month] = int(tally.get(month, 0)) + int(n)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".json.tmp")
    os.close(fd)
    with open(tmp, "w") as fh:
        json.dump(tally, fh, indent=2, sort_keys=True)
    os.replace(tmp, TALLY_PATH)
    return tally


def calls_used_this_month() -> int:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    if not os.path.exists(TALLY_PATH):
        return 0
    try:
        with open(TALLY_PATH) as fh:
            return int(json.load(fh).get(month, 0))
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------- #
# Normalisers: CFBD camelCase JSON -> our snake_case contract
# --------------------------------------------------------------------------- #
def _get(row: Dict, *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


def normalize_teams(rows: List[Dict], season: int) -> pd.DataFrame:
    df = pd.DataFrame([{
        "season": season,
        "team": _get(r, "school", "team"),
        "conference": _get(r, "conference", default="") or "",
        "classification": (_get(r, "classification", default="") or "").lower(),
    } for r in rows])
    if df.empty:
        return df
    df["is_fbs"] = df["classification"].eq("fbs")
    df["season"] = df["season"].astype("int64")
    for c in ("team", "conference", "classification"):
        df[c] = df[c].astype("category")
    return assert_schema(df, "teams")


def normalize_games(rows: List[Dict], season: int, fbs: set) -> pd.DataFrame:
    df = pd.DataFrame([{
        "id": _get(r, "id"),
        "season": _get(r, "season", default=season),
        "week": _get(r, "week"),
        "start_date": _get(r, "startDate", "start_date"),
        "neutral_site": bool(_get(r, "neutralSite", "neutral_site", default=False)),
        "home_team": _get(r, "homeTeam", "home_team"),
        "away_team": _get(r, "awayTeam", "away_team"),
        "home_points": _get(r, "homePoints", "home_points"),
        "away_points": _get(r, "awayPoints", "away_points"),
        "home_conference": _get(r, "homeConference", "home_conference", default=""),
        "away_conference": _get(r, "awayConference", "away_conference", default=""),
    } for r in rows])
    if df.empty:
        return df
    # FCS tagging: anything not in this season's FBS set is a different tier.
    df["home_fcs"] = ~df["home_team"].isin(fbs)
    df["away_fcs"] = ~df["away_team"].isin(fbs)
    df["id"] = df["id"].astype("int64")
    df["season"] = df["season"].astype("int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype("int64")
    for c in ("home_points", "away_points"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("home_team", "away_team", "home_conference", "away_conference"):
        df[c] = df[c].astype("category")
    return assert_schema(df, "games")


def is_garbage_time(period, offense_score, defense_score) -> bool:
    """True if this play should be excluded from rate stats."""
    try:
        p = int(period)
        margin = abs(float(offense_score) - float(defense_score))
    except (TypeError, ValueError):
        return False
    threshold = GARBAGE_TIME_MARGIN_BY_PERIOD.get(p)
    if threshold is None:          # OT and anything unexpected: never garbage
        return False
    return margin > threshold


def normalize_plays(rows: List[Dict], season: int, week: int) -> pd.DataFrame:
    df = pd.DataFrame([{
        "game_id": _get(r, "gameId", "game_id"),
        "season": season,
        "week": week,
        "period": _get(r, "period"),
        "offense": _get(r, "offense"),
        "defense": _get(r, "defense"),
        "offense_score": _get(r, "offenseScore", "offense_score"),
        "defense_score": _get(r, "defenseScore", "defense_score"),
        "down": _get(r, "down"),
        "distance": _get(r, "distance"),
        "yards_to_goal": _get(r, "yardsToGoal", "yards_to_goal"),
        "yards_gained": _get(r, "yardsGained", "yards_gained"),
        "play_type": _get(r, "playType", "play_type", default=""),
        "ppa": _get(r, "ppa"),
    } for r in rows])
    if df.empty:
        return df
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").fillna(0).astype("int64")
    for c in ("season", "week"):
        df[c] = df[c].astype("int64")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype("int64")
    for c in ("offense_score", "defense_score", "down", "distance",
              "yards_to_goal", "yards_gained", "ppa"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["garbage_time"] = [
        is_garbage_time(p, o, d)
        for p, o, d in zip(df["period"], df["offense_score"], df["defense_score"])
    ]
    for c in ("offense", "defense", "play_type"):
        df[c] = df[c].astype("category")
    return assert_schema(df, "plays")


def normalize_lines(rows: List[Dict], season: int) -> pd.DataFrame:
    """Flatten CFBD's nested per-game `lines: [...]` into one row per book.

    Opening AND closing numbers are both kept from day one — the market
    layer (Workstream D) needs the open/close pair to measure CLV at all,
    and backfilling opens later is not possible.
    """
    out = []
    for g in rows:
        for b in (g.get("lines") or []):
            out.append({
                "game_id": _get(g, "id"),
                "season": _get(g, "season", default=season),
                "week": _get(g, "week"),
                "home_team": _get(g, "homeTeam", "home_team"),
                "away_team": _get(g, "awayTeam", "away_team"),
                "provider": _get(b, "provider", default=""),
                "spread": _get(b, "spread"),
                "spread_open": _get(b, "spreadOpen", "spread_open"),
                "over_under": _get(b, "overUnder", "over_under"),
                "over_under_open": _get(b, "overUnderOpen", "over_under_open"),
                "home_moneyline": _get(b, "homeMoneyline", "home_moneyline"),
                "away_moneyline": _get(b, "awayMoneyline", "away_moneyline"),
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["game_id"] = df["game_id"].astype("int64")
    df["season"] = df["season"].astype("int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype("int64")
    for c in ("spread", "spread_open", "over_under", "over_under_open",
              "home_moneyline", "away_moneyline"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("provider", "home_team", "away_team"):
        df[c] = df[c].astype("category")
    return assert_schema(df, "lines")


def normalize_talent(rows: List[Dict], season: int) -> pd.DataFrame:
    """Sentinel-aware. talent==0 becomes NULL + unranked_recruiting=True."""
    df = pd.DataFrame([{
        "season": season,
        "team": _get(r, "team", "school"),
        "talent": pd.to_numeric(_get(r, "talent"), errors="coerce"),
    } for r in rows])
    if df.empty:
        return df
    df["unranked_recruiting"] = df["talent"].eq(TALENT_MISSING_SENTINEL)
    df.loc[df["unranked_recruiting"], "talent"] = pd.NA
    df["talent"] = pd.to_numeric(df["talent"], errors="coerce")
    df["season"] = df["season"].astype("int64")
    df["team"] = df["team"].astype("category")
    return assert_schema(df, "talent")


def normalize_returning(rows: List[Dict], season: int) -> pd.DataFrame:
    df = pd.DataFrame([{
        "season": season,
        "team": _get(r, "team"),
        "percent_ppa": pd.to_numeric(_get(r, "percentPPA", "percent_ppa"),
                                     errors="coerce"),
        "percent_passing_ppa": pd.to_numeric(
            _get(r, "percentPassingPPA"), errors="coerce"),
        "percent_rushing_ppa": pd.to_numeric(
            _get(r, "percentRushingPPA"), errors="coerce"),
        "percent_receiving_ppa": pd.to_numeric(
            _get(r, "percentReceivingPPA"), errors="coerce"),
    } for r in rows])
    if df.empty:
        return df
    df["season"] = df["season"].astype("int64")
    df["team"] = df["team"].astype("category")
    return assert_schema(df, "returning")


# --------------------------------------------------------------------------- #
# Refresh
# --------------------------------------------------------------------------- #
@dataclass
class RefreshResult:
    season: int
    stale: bool = False
    error: str = ""
    calls: int = 0
    written: List[str] = field(default_factory=list)
    rows: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        head = f"[ingest] season={self.season} calls={self.calls}"
        if self.stale:
            return f"{head} STALE ({self.error}) — kept existing cache"
        return f"{head} rows={self.rows}"


def _season_path(kind: str, season: int) -> str:
    return os.path.join(DATA_DIR, f"{kind}_{season}.parquet")


def refresh_season(season: int, conference: Optional[str] = None,
                   weeks: Iterable[int] = range(1, 16),
                   client: Optional[CFBDClient] = None) -> RefreshResult:
    """Pull one season into the parquet cache. Idempotent.

    On any CFBD failure the existing cache is left untouched and the result is
    marked `stale=True` — the caller (freshness gate) decides what to do. This
    function never serves stale data as though it were fresh, and never
    partially overwrites a season.
    """
    res = RefreshResult(season=season)
    try:
        c = client or CFBDClient()
    except CFBDError as exc:
        res.stale, res.error = True, str(exc)
        return res

    try:
        teams_raw = c.teams(season)
        teams = normalize_teams(teams_raw, season)
        fbs = set(teams.loc[teams["is_fbs"], "team"].astype(str))

        games = normalize_games(c.games(season, conference=conference), season, fbs)
        lines = normalize_lines(c.lines(season, conference=conference), season)
        talent = normalize_talent(c.talent(season), season)
        # NOTE: no conference filter — the endpoint silently returns zero rows
        # for one. See CFBDClient.returning_production.
        returning = normalize_returning(c.returning_production(season), season)

        play_frames = []
        for wk in weeks:
            wk_rows = c.plays(season, wk, conference=conference)
            if not wk_rows:
                continue
            play_frames.append(normalize_plays(wk_rows, season, wk))
        plays = (pd.concat(play_frames, ignore_index=True)
                 if play_frames else pd.DataFrame())
    except CFBDError as exc:
        res.stale, res.error, res.calls = True, str(exc), c.calls
        _record_calls(c.calls)
        return res

    writes = {
        "teams": teams, "games": games, "lines": lines,
        "talent": talent, "returning": returning, "plays": plays,
    }
    if not plays.empty:
        writes["plays_clean"] = plays.loc[~plays["garbage_time"]].reset_index(drop=True)

    for kind, df in writes.items():
        if df is None or df.empty:
            continue
        path = _season_path(kind, season)
        _atomic_write_parquet(df, path)
        res.written.append(os.path.basename(path))
        res.rows[kind] = len(df)

    res.calls = c.calls
    _record_calls(c.calls)
    return res


# --------------------------------------------------------------------------- #
# Loaders — compose seasons, prune columns, assert the contract every time
# --------------------------------------------------------------------------- #
def load(kind: str, seasons: Iterable[int],
         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and concatenate one kind across seasons.

    Deterministic: sorted by season then by the frame's natural key, so repeat
    calls return identical frames (part of Workstream A's DoD).
    """
    frames = []
    for s in sorted(seasons):
        path = _season_path(kind, s)
        if not os.path.exists(path):
            continue
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    contract_name = "plays" if kind == "plays_clean" else kind
    if columns is None and contract_name in SCHEMAS:
        assert_schema(df, contract_name)
    sort_keys = [k for k in ("season", "week", "game_id", "id", "team")
                 if k in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    return df


def load_games(seasons, **kw):     return load("games", seasons, **kw)
def load_plays(seasons, **kw):     return load("plays", seasons, **kw)
def load_plays_clean(seasons, **kw): return load("plays_clean", seasons, **kw)
def load_lines(seasons, **kw):     return load("lines", seasons, **kw)
def load_talent(seasons, **kw):    return load("talent", seasons, **kw)
def load_teams(seasons, **kw):     return load("teams", seasons, **kw)
def load_returning(seasons, **kw): return load("returning", seasons, **kw)


def conference_of(season: int, team: str) -> str:
    """Per-season conference lookup. Never hardcode this."""
    t = load_teams([season])
    if t.empty:
        return ""
    hit = t.loc[t["team"].astype(str) == team, "conference"]
    return str(hit.iloc[0]) if len(hit) else ""
