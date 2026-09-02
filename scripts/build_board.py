#!/usr/bin/env python3
"""Build the weekly BOARD: every priced FBS game, its market spread, the model's
number, the disagreement between them, and the pregame context around it.

WHAT THIS IS
------------
The published forecast artifact (`data/predictions_{season}_wk{NN}.csv`) is a
model number and nothing else -- by design it never touches a betting line.
This script is the missing last mile: it joins the artifact to the market,
computes the disagreement, ranks it, and attaches the display-only context
factors so a human can see WHY a game is on the board.

WHAT THIS IS NOT
----------------
Validated. As of this file's creation the factor registry has exactly two
`gated_active` factors (league HFA and per-team HFA); weather, travel/timezone,
surface, portal and QB availability are `display_only` or untested, and the
live-2026 CLV slice is n=0. Nothing here has been shown to beat a closing
number. Every row is a LEAN to be graded, not a recommendation. The ledger
(`scripts/board_ledger.py`) exists so that the record decides, not the author.

SIGN CONVENTIONS (the thing that silently breaks boards)
-------------------------------------------------------
* CFBD `spread` is quoted from the HOME side: negative = home favored.
* `market_home_margin = -spread`, so both model and market are expressed as
  "expected home points minus away points". They are directly comparable.
* `edge = model_home_margin - market_home_margin`. Positive edge = the model
  likes the HOME side against the number; negative = the AWAY side.

Usage:
    python3 scripts/build_board.py --season 2026 --week 1
    python3 scripts/build_board.py --season 2026 --week 1 --min-edge 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CFBD = os.path.join(DATA, "cfbd")
PUBLIC_TZ = ZoneInfo("America/New_York")

#: Providers in preference order. The first one that quotes a game wins, so the
#: board reports ONE book's number rather than a median that no one can bet.
PROVIDER_PRIORITY = ["DraftKings", "Bovada", "ESPN Bet", "consensus"]

#: A game is only shown as a lean when the disagreement clears this. Below it
#: the model and the market are saying the same thing and the "edge" is noise.
DEFAULT_MIN_EDGE = 1.5


# --------------------------------------------------------------------------
# geometry / circadian helpers
# --------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in statute miles. NaN if any coordinate is absent."""
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    r = 3958.7613
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def utc_offset_hours(tz_name, when) -> float:
    """UTC offset in hours for `tz_name` at instant `when`. NaN if unknown."""
    if not tz_name or pd.isna(when):
        return float("nan")
    try:
        return when.astimezone(ZoneInfo(str(tz_name))).utcoffset().total_seconds() / 3600.0
    except Exception:
        return float("nan")


def body_clock_hour(kickoff_utc, home_tz) -> float:
    """Kickoff hour on the travelling team's HOME clock.

    The circadian literature's mechanism is not distance, it is the mismatch
    between local kickoff time and the visitor's internal clock. A 10pm ET
    kickoff is a 7pm body-clock game for a Pacific team (fine) and a 1pm ET
    kickoff is a 10am body-clock game for that same team (the documented
    penalty). Distance alone cannot see that; this can.
    """
    if not home_tz or pd.isna(kickoff_utc):
        return float("nan")
    try:
        local = kickoff_utc.astimezone(ZoneInfo(str(home_tz)))
        return local.hour + local.minute / 60.0
    except Exception:
        return float("nan")


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_artifact(season: int, week: int) -> pd.DataFrame:
    path = os.path.join(DATA, f"predictions_{season}_wk{week:02d}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"no forecast artifact at {path}. Run "
            f"`python3 scripts/weekly_loop.py --job pregame --week {week}` first.")
    df = pd.read_csv(path)
    if "status" not in df.columns:
        raise SystemExit(
            f"{path} predates schema v2 (no `status` column). Regenerate it "
            "with the weekly loop before building a board.")
    priced = df[df["status"] == "priced"].copy()
    if priced.empty:
        raise SystemExit(
            f"{path} has {len(df)} rows and none of them are priced. The board "
            "refuses to invent numbers the forecast declined to produce.")
    return priced


def pick_line(lines: pd.DataFrame, game_ids: set) -> pd.DataFrame:
    """One quoted line per game, from the highest-priority provider that has one."""
    sub = lines[lines["game_id"].isin(game_ids)].dropna(subset=["spread"]).copy()
    if sub.empty:
        return sub.assign(provider=[], spread=[], spread_open=[], over_under=[])
    rank = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}
    sub["_rank"] = sub["provider"].map(lambda p: rank.get(p, len(rank)))
    sub = sub.sort_values(["game_id", "_rank"])
    return sub.groupby("game_id", as_index=False).first().drop(columns=["_rank"])


def rest_days(schedule: pd.DataFrame, season: int) -> dict:
    """(team, game_id) -> days since that team's previous kickoff this season."""
    sched = schedule.copy()
    sched["kick"] = pd.to_datetime(sched["start_date"], utc=True, errors="coerce")
    long = pd.concat([
        sched[["id", "kick", "home_team"]].rename(columns={"home_team": "team"}),
        sched[["id", "kick", "away_team"]].rename(columns={"away_team": "team"}),
    ]).dropna(subset=["kick"]).sort_values(["team", "kick"])
    long["prev"] = long.groupby("team")["kick"].shift(1)
    long["rest"] = (long["kick"] - long["prev"]).dt.total_seconds() / 86400.0
    return {(r.team, int(r.id)): r.rest for r in long.itertuples()}


# --------------------------------------------------------------------------
# the board
# --------------------------------------------------------------------------
def build(season: int, week: int, min_edge: float) -> pd.DataFrame:
    art = load_artifact(season, week)
    lines = pd.read_parquet(os.path.join(CFBD, f"lines_{season}.parquet"))
    sched = pd.read_parquet(os.path.join(CFBD, f"games_{season}.parquet"))
    venues = pd.read_parquet(os.path.join(CFBD, "venues.parquet"))
    locs = pd.read_parquet(os.path.join(CFBD, f"team_locations_{season}.parquet"))
    gv = pd.read_parquet(os.path.join(CFBD, f"game_venues_{season}.parquet"))

    quoted = pick_line(lines, set(art["game_id"].astype(int)))
    quoted = quoted.set_index("game_id")
    venues_by_id = venues.set_index("venue_id")
    loc_by_team = locs.set_index("team")
    venue_of_game = gv.set_index("game_id")["venue_id"].to_dict()
    rest = rest_days(sched, season)

    now = datetime.now(timezone.utc)
    out = []
    for r in art.itertuples():
        gid = int(r.game_id)
        if gid not in quoted.index:
            continue                       # unpriced by the market: not a board row
        q = quoted.loc[gid]
        spread = float(q["spread"])
        market = -spread                   # market's expected home margin
        model = float(r.projected_home_margin)
        sd = float(r.forecast_margin_sd)
        edge = model - market

        kick = pd.to_datetime(r.kickoff_utc, utc=True, errors="coerce")
        if pd.notna(kick) and kick <= now:
            continue                       # already kicked: not a forward board

        # cover probability for the side the edge points at, from the model's
        # own forecast SD -- not a tuned confidence number.
        p_home_cover = normal_cdf(edge / sd) if sd > 0 else float("nan")
        side_is_home = edge > 0
        p_cover = p_home_cover if side_is_home else 1.0 - p_home_cover

        vid = venue_of_game.get(gid)
        v = venues_by_id.loc[vid] if vid in venues_by_id.index else None
        hl = loc_by_team.loc[r.home] if r.home in loc_by_team.index else None
        al = loc_by_team.loc[r.away] if r.away in loc_by_team.index else None

        v_lat = v["lat"] if v is not None else (hl["home_lat"] if hl is not None else np.nan)
        v_lon = v["lon"] if v is not None else (hl["home_lon"] if hl is not None else np.nan)
        v_tz = v["timezone"] if v is not None else (hl["home_timezone"] if hl is not None else None)

        away_travel = haversine_miles(
            al["home_lat"] if al is not None else np.nan,
            al["home_lon"] if al is not None else np.nan, v_lat, v_lon)
        home_travel = haversine_miles(
            hl["home_lat"] if hl is not None else np.nan,
            hl["home_lon"] if hl is not None else np.nan, v_lat, v_lon)

        away_tz = al["home_timezone"] if al is not None else None
        tz_shift = (utc_offset_hours(v_tz, kick) - utc_offset_hours(away_tz, kick))
        away_clock = body_clock_hour(kick, away_tz)

        surface = ("dome/indoor" if (v is not None and bool(v["dome"]))
                   else "grass" if (v is not None and v["grass"] is True)
                   else "turf" if (v is not None and v["grass"] is False)
                   else "unknown")

        out.append({
            "kickoff_et": kick.astimezone(PUBLIC_TZ).strftime("%a %b %d %-I:%M%p")
                          if pd.notna(kick) else "",
            "kickoff_utc": r.kickoff_utc,
            "game_id": gid,
            "away": r.away,
            "home": r.home,
            "neutral": bool(r.neutral_site),
            "book": q["provider"],
            "market_spread_home": spread,
            "market_home_margin": round(market, 2),
            "model_home_margin": round(model, 2),
            "edge_pts": round(edge, 2),
            "side": (f"{r.home} {spread:+.1f}" if side_is_home
                     else f"{r.away} {-spread:+.1f}"),
            "p_cover": round(p_cover, 4),
            "forecast_sd": round(sd, 2),
            "p_home_win": r.p_home_win,
            "total_open": q.get("over_under_open"),
            "total_now": q.get("over_under"),
            "line_move": (round(spread - float(q["spread_open"]), 2)
                          if pd.notna(q.get("spread_open")) else np.nan),
            # ---- display-only context: NOT in the model's number ----
            "ctx_surface": surface,
            "ctx_capacity": (int(v["capacity"]) if v is not None
                             and pd.notna(v["capacity"]) else np.nan),
            "ctx_elevation_ft": (round(float(v["elevation"]) * 3.28084, 0)
                                 if v is not None and pd.notna(v["elevation"]) else np.nan),
            "ctx_away_travel_mi": (round(away_travel) if pd.notna(away_travel) else np.nan),
            "ctx_home_travel_mi": (round(home_travel) if pd.notna(home_travel) else np.nan),
            "ctx_tz_shift_hrs": (round(tz_shift, 1) if pd.notna(tz_shift) else np.nan),
            "ctx_away_body_clock": (f"{int(away_clock):02d}:{int((away_clock%1)*60):02d}"
                                    if pd.notna(away_clock) else ""),
            "ctx_rest_home": rest.get((r.home, gid), np.nan),
            "ctx_rest_away": rest.get((r.away, gid), np.nan),
            "ctx_venue": (v["venue"] if v is not None else ""),
        })

    board = pd.DataFrame(out)
    if board.empty:
        return board

    # ---- SCALE DIAGNOSTIC (added at first board, 2026-09-02) ----------------
    # Regressing the model's margin on the market's exposes whether the model
    # is on the market's SCALE or a compressed one. A ridge fit shrinks toward
    # the mean, so its slope runs below 1 -- and a slope below 1 makes every
    # large-spread game look like an underdog edge. That is ONE bias reported
    # 20 times, not 20 independent edges, and it is the single most likely way
    # this board is wrong.
    #
    # The rescaled edge is published as a labelled DIAGNOSTIC beside the raw
    # edge, never in place of it: the raw edge is what the registered forecast
    # path actually says, and silently swapping in a rescaled number would be
    # exactly the unregistered tunable this project refuses. The ledger grades
    # both, and the record decides which is right.
    slope, intercept, n_fit = np.nan, np.nan, len(board)
    if n_fit >= 10 and board["market_home_margin"].std() > 0:
        slope, intercept = np.polyfit(board["market_home_margin"],
                                      board["model_home_margin"], 1)
        rescaled = (board["model_home_margin"] - intercept) / slope
        board["edge_pts_scale_adj"] = (rescaled - board["market_home_margin"]).round(2)
        board["p_cover_scale_adj"] = [
            round(normal_cdf(abs(e) / s), 4) if s > 0 else np.nan
            for e, s in zip(board["edge_pts_scale_adj"], board["forecast_sd"])]
        board["side_scale_adj"] = [
            (f"{h} {sp:+.1f}" if e > 0 else f"{a} {-sp:+.1f}")
            for h, a, sp, e in zip(board["home"], board["away"],
                                   board["market_spread_home"],
                                   board["edge_pts_scale_adj"])]
    else:
        board["edge_pts_scale_adj"] = np.nan
        board["p_cover_scale_adj"] = np.nan
        board["side_scale_adj"] = ""
    board.attrs["scale_slope"] = float(slope) if pd.notna(slope) else None
    board.attrs["scale_intercept"] = float(intercept) if pd.notna(intercept) else None
    board.attrs["dog_leans"] = int((board["edge_pts"] < 0).sum())
    board.attrs["fav_leans"] = int((board["edge_pts"] > 0).sum())

    board["abs_edge"] = board["edge_pts"].abs()
    board["tier"] = pd.cut(
        board["abs_edge"], bins=[-0.01, min_edge, 3.0, 6.0, 999],
        labels=["no disagreement", "lean", "play", "strong disagreement"])
    board["ctx_rest_diff"] = board["ctx_rest_home"] - board["ctx_rest_away"]
    for c in ("ctx_rest_home", "ctx_rest_away", "ctx_rest_diff"):
        board[c] = board[c].round(1)
    return board.sort_values("abs_edge", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    args = ap.parse_args()

    board = build(args.season, args.week, args.min_edge)
    if board.empty:
        print("no forward-looking board rows: every priced game has already kicked "
              "or has no quoted line")
        return 1

    out_csv = os.path.join(DATA, f"board_{args.season}_wk{args.week:02d}.csv")
    board.to_csv(out_csv, index=False)

    meta = {
        "season": args.season, "week": args.week,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games": int(len(board)),
        "min_edge": args.min_edge,
        "tiers": {str(k): int(v) for k, v in board["tier"].value_counts().items()},
        "status": "UNVALIDATED — leans for grading, not recommendations",
        "scale_diagnostic": {
            "model_vs_market_slope": board.attrs.get("scale_slope"),
            "model_vs_market_intercept": board.attrs.get("scale_intercept"),
            "underdog_leans": board.attrs.get("dog_leans"),
            "favorite_leans": board.attrs.get("fav_leans"),
            "note": ("slope below 1.0 means the model is compressed relative to "
                     "the market and will systematically lean underdog on large "
                     "spreads; the *_scale_adj columns are the diagnostic "
                     "alternative, graded in parallel, never substituted"),
        },
    }
    with open(os.path.join(DATA, f"board_{args.season}_wk{args.week:02d}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[board] {out_csv}  ({len(board)} games)")
    print(json.dumps(meta["tiers"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
