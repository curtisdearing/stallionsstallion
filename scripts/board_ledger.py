#!/usr/bin/env python3
"""The results ledger: freeze every board lean before kickoff, grade it after.

This is the half of the board that makes it worth anything. A board with no
ledger is an opinion; a board with a ledger is a record that can prove the
model wrong. The point of this file is that the record decides.

TWO COMMANDS
------------
    --record   Append this week's board to data/board_ledger.csv. FIRST WRITE
               WINS: once a game is in the ledger its line, its lean and its
               timestamp are frozen and no later run may alter them.
    --grade    Join finals, score every frozen lean against the number that was
               actually taken, and print the running record.

WHY FIRST-WRITE-WINS IS NOT OPTIONAL
------------------------------------
If a re-run were allowed to refresh the line on a game already in the ledger,
the ledger would drift toward the closing number and every result would look
better than it was. That single leak manufactured a fake +3 points of CLV in
this project's history. The line stored here is the line that was available
when the lean was made, and nothing is permitted to update it.

GRADING
-------
`result_raw`     the lean from `edge_pts` (the registered forecast path)
`result_adj`     the lean from `edge_pts_scale_adj` (the shrinkage diagnostic)

Both are graded from the same frozen line, on the same games, so the season
answers the open question -- is the model's compression against the market a
real edge or a scale artefact -- from outcomes rather than argument. A push is
recorded as a push and excluded from win rate, never rounded into a win.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CFBD = os.path.join(DATA, "cfbd")
LEDGER = os.path.join(DATA, "board_ledger.csv")

LEDGER_COLUMNS = [
    "season", "week", "game_id", "kickoff_utc", "away", "home",
    "book", "line_taken_home", "taken_at_utc",
    "model_home_margin", "forecast_sd",
    "edge_pts", "side_raw", "p_cover_raw",
    "edge_pts_scale_adj", "side_adj", "p_cover_adj",
    "ctx_surface", "ctx_away_travel_mi", "ctx_tz_shift_hrs",
    "ctx_rest_diff", "ctx_capacity", "ctx_elevation_ft",
]


def _side_token(side: str) -> str:
    """'Clemson +10.0' -> 'Clemson'. The team the lean is on."""
    return str(side).rsplit(" ", 1)[0].strip() if side else ""


def record(season: int, week: int) -> int:
    board_path = os.path.join(DATA, f"board_{season}_wk{week:02d}.csv")
    if not os.path.exists(board_path):
        raise SystemExit(f"no board at {board_path} — run scripts/build_board.py first")
    board = pd.read_csv(board_path)

    now = datetime.now(timezone.utc)
    kicks = pd.to_datetime(board["kickoff_utc"], utc=True, errors="coerce")
    future = board[kicks > now].copy()
    late = len(board) - len(future)
    if future.empty:
        print(f"[ledger] every game on the week-{week} board has kicked; "
              f"nothing recorded (a lean logged after kickoff is not a lean)")
        return 1

    rows = pd.DataFrame({
        "season": season, "week": week,
        "game_id": future["game_id"].astype(int),
        "kickoff_utc": future["kickoff_utc"],
        "away": future["away"], "home": future["home"],
        "book": future["book"],
        "line_taken_home": future["market_spread_home"],
        "taken_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_home_margin": future["model_home_margin"],
        "forecast_sd": future["forecast_sd"],
        "edge_pts": future["edge_pts"],
        "side_raw": future["side"],
        "p_cover_raw": future["p_cover"],
        "edge_pts_scale_adj": future.get("edge_pts_scale_adj"),
        "side_adj": future.get("side_scale_adj"),
        "p_cover_adj": future.get("p_cover_scale_adj"),
        "ctx_surface": future.get("ctx_surface"),
        "ctx_away_travel_mi": future.get("ctx_away_travel_mi"),
        "ctx_tz_shift_hrs": future.get("ctx_tz_shift_hrs"),
        "ctx_rest_diff": future.get("ctx_rest_diff"),
        "ctx_capacity": future.get("ctx_capacity"),
        "ctx_elevation_ft": future.get("ctx_elevation_ft"),
    })[LEDGER_COLUMNS]

    if os.path.exists(LEDGER):
        existing = pd.read_csv(LEDGER)
        have = set(zip(existing["season"], existing["week"], existing["game_id"]))
        fresh = rows[~rows.apply(
            lambda r: (r["season"], r["week"], r["game_id"]) in have, axis=1)]
        frozen = len(rows) - len(fresh)
        merged = pd.concat([existing, fresh], ignore_index=True)
    else:
        fresh, frozen, merged = rows, 0, rows

    merged.to_csv(LEDGER, index=False)
    print(f"[ledger] {LEDGER}")
    print(f"  recorded {len(fresh)} new lean(s); {frozen} already frozen "
          f"(left untouched); {late} already-kicked row(s) skipped")
    print(f"  ledger now holds {len(merged)} lean(s)")
    return 0


def grade(season: int | None = None) -> int:
    if not os.path.exists(LEDGER):
        raise SystemExit("no ledger yet — record a board first")
    led = pd.read_csv(LEDGER)
    if season:
        led = led[led["season"] == season]
    if led.empty:
        print("[ledger] nothing to grade")
        return 0

    finals = []
    for yr in sorted(led["season"].unique()):
        p = os.path.join(CFBD, f"games_{yr}.parquet")
        if os.path.exists(p):
            finals.append(pd.read_parquet(p)[
                ["id", "home_points", "away_points"]].rename(columns={"id": "game_id"}))
    if not finals:
        raise SystemExit("no games cache to grade against")
    g = pd.concat(finals).dropna(subset=["home_points", "away_points"])

    df = led.merge(g, on="game_id", how="left")
    played = df.dropna(subset=["home_points"]).copy()
    if played.empty:
        print(f"[ledger] {len(df)} lean(s) recorded, 0 played yet — nothing to grade")
        return 0

    played["actual_home_margin"] = played["home_points"] - played["away_points"]
    # ATS from the frozen line: home covers when actual margin beats the number.
    played["home_ats"] = played["actual_home_margin"] + played["line_taken_home"]

    def score(side_col: str, edge_col: str) -> pd.Series:
        res = []
        for r in played.itertuples():
            edge = getattr(r, edge_col)
            if pd.isna(edge) or edge == 0:
                res.append("no lean"); continue
            margin = r.home_ats if edge > 0 else -r.home_ats
            res.append("push" if abs(margin) < 1e-9
                       else "win" if margin > 0 else "loss")
        return pd.Series(res, index=played.index)

    played["result_raw"] = score("side_raw", "edge_pts")
    played["result_adj"] = score("side_adj", "edge_pts_scale_adj")

    out = os.path.join(DATA, "board_ledger_graded.csv")
    played.to_csv(out, index=False)

    print(f"[ledger] {len(df)} recorded, {len(played)} played, "
          f"{len(df) - len(played)} pending")
    for label, col in (("registered edge", "result_raw"),
                       ("scale-adjusted diagnostic", "result_adj")):
        c = played[col].value_counts()
        w, l, p = int(c.get("win", 0)), int(c.get("loss", 0)), int(c.get("push", 0))
        decided = w + l
        rate = f"{w / decided:.1%}" if decided else "n/a"
        need = "" if decided >= 100 else f"  [n={decided}, far below any n-gate]"
        print(f"  {label:26s} {w}-{l}-{p}  ({rate}){need}")
    print(f"  wrote {out}")
    print("\n  A week is not evidence. Nothing here authorises a stake; the "
          "gate-7 kill-check still owns that decision.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args()

    if args.record:
        if not (args.season and args.week):
            raise SystemExit("--record needs --season and --week")
        return record(args.season, args.week)
    if args.grade:
        return grade(args.season)
    raise SystemExit("pick one: --record or --grade")


if __name__ == "__main__":
    raise SystemExit(main())
