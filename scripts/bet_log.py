#!/usr/bin/env python3
"""The real-bet ledger: log a bet Curtis actually placed, grade it after.

This is a different instrument from `board_ledger.py`. That file freezes and
grades a spread lean on every FBS-vs-FBS game each week, whether or not
anyone acted on it -- it exists to answer the open compression question from
outcomes. This file exists to answer a narrower, personal question: of the
bets Curtis actually places (real side, real price, real book, real stake),
how many win, and what's the ROI? It covers spread AND moneyline, FBS-vs-FBS
AND FBS-vs-FCS -- anything with a real bet on it -- and nothing that wasn't
actually bet never appears here at all. An empty log is not evidence of
anything; it just means nothing has been logged yet.

TWO COMMANDS
------------
    --add      Record one bet: the game, the side, the price you got, the
               stake. Looks the game up in that week's forecast artifact so
               the away/home/kickoff can't be mistyped, and refuses to log a
               bet on a game that has already kicked off or already has a
               final score cached (a bet logged after the outcome is known
               proves nothing).
    --grade    Join finals, score every logged bet, and print the running
               win-loss-push record and ROI, overall and by bet kind.

FIRST WRITE WINS, same as the results ledger: an identical bet (same game,
kind, side, book) already in the log is left untouched, not overwritten.

This tool does not place bets, does not talk to a sportsbook, and does not
know what "should" be bet. It only logs what Curtis tells it he bet, and
grades it once the game is final. Nothing here is investment advice, and a
handful of graded bets is not a track record -- it's the start of one.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import pandas as pd  # noqa: E402

from nflvalue import forecast_contract as FC  # noqa: E402
import render_picks as RP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CFBD = os.path.join(DATA, "cfbd")
LOG = os.path.join(DATA, "bet_log.csv")
GRADED = os.path.join(DATA, "bet_log_graded.csv")

BET_LOG_COLUMNS = [
    "bet_id", "season", "week", "game_id", "kickoff_utc", "away", "home",
    "kind", "side", "price", "stake_units", "book", "placed_at_utc", "note",
]


def _next_bet_id(log_path: str) -> int:
    if not os.path.exists(log_path):
        return 1
    existing = pd.read_csv(log_path)
    if existing.empty or "bet_id" not in existing.columns:
        return 1
    return int(existing["bet_id"].max()) + 1


def _parse_spread_side(side: str) -> tuple[str, float]:
    team, _, num = side.strip().rpartition(" ")
    if not team:
        raise SystemExit(
            f"can't parse a spread side from {side!r} -- want e.g. "
            f"'Clemson +10.0' (team, then the signed number)")
    try:
        return team, float(num)
    except ValueError:
        raise SystemExit(
            f"can't parse a spread number from {side!r} -- want e.g. "
            f"'Clemson +10.0'")


def add(season: int, week: int, game_id: int, kind: str, side: str,
        price: float, stake: float, book: str, note: str,
        artifact_path: Optional[str] = None, log_path: Optional[str] = None,
        games_path: Optional[str] = None,
        now: Optional[pd.Timestamp] = None) -> int:
    if kind not in ("spread", "moneyline"):
        raise SystemExit("--kind must be 'spread' or 'moneyline'")

    log_path = log_path or LOG
    art_path = artifact_path or os.path.join(
        DATA, f"predictions_{season}_wk{week:02d}.csv")
    if not os.path.exists(art_path):
        raise SystemExit(f"no forecast artifact at {art_path}")
    rows = FC.read_artifact(art_path)
    game = rows[rows["game_id"] == int(game_id)]
    if game.empty:
        raise SystemExit(
            f"game_id {game_id} is not on the season {season} week {week} "
            f"artifact -- check the id against the picks page")
    g = game.iloc[0]
    home, away, kickoff = str(g["home"]), str(g["away"]), str(g["kickoff_utc"])

    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    kickoff_ts = pd.Timestamp(kickoff, tz="UTC")
    if now >= kickoff_ts:
        raise SystemExit(
            f"{away} @ {home} kicked off at {kickoff} -- a bet logged after "
            f"kickoff is not a real pre-game bet")

    side = side.strip()
    if kind == "moneyline":
        team = side
        if team not in (home, away):
            raise SystemExit(f"{team!r} is not {away!r} or {home!r}")
    else:
        team, _num = _parse_spread_side(side)
        if team not in (home, away):
            raise SystemExit(f"{team!r} is not {away!r} or {home!r}")

    games_path = games_path or os.path.join(CFBD, f"games_{season}.parquet")
    if os.path.exists(games_path):
        finals = pd.read_parquet(games_path)
        id_col = "id" if "id" in finals.columns else "game_id"
        match = finals[finals[id_col] == int(game_id)]
        if not match.empty and pd.notna(match.iloc[0].get("home_points")):
            raise SystemExit(
                f"game_id {game_id} already has a final score cached -- a "
                f"bet can't be logged after the outcome is known")

    new_row = {
        "bet_id": _next_bet_id(log_path),
        "season": season, "week": week, "game_id": int(game_id),
        "kickoff_utc": kickoff, "away": away, "home": home,
        "kind": kind, "side": side, "price": float(price),
        "stake_units": float(stake), "book": book or "",
        "placed_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note or "",
    }

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        dup = existing[
            (existing["season"] == season) & (existing["week"] == week)
            & (existing["game_id"] == int(game_id))
            & (existing["kind"] == kind) & (existing["side"] == side)
            & (existing["book"] == (book or ""))]
        if not dup.empty:
            print(f"[bet_log] already logged this bet (bet_id "
                  f"{int(dup.iloc[0]['bet_id'])}) -- first write wins, "
                  f"nothing added")
            return 0
        merged = pd.concat([existing, pd.DataFrame([new_row])],
                          ignore_index=True)
    else:
        merged = pd.DataFrame([new_row])

    merged[BET_LOG_COLUMNS].to_csv(log_path, index=False)
    print(f"[bet_log] {log_path}")
    print(f"  logged bet_id {new_row['bet_id']}: {stake:g}u on {side} "
          f"({kind}) at {price:+.0f}, {away} @ {home}"
          + (f", {book}" if book else ""))
    return 0


def _grade_row(r) -> str:
    if r.kind == "moneyline":
        team = r.side.strip()
        if team == r.home:
            margin = r.home_points - r.away_points
        elif team == r.away:
            margin = r.away_points - r.home_points
        else:
            return "unresolved"
        return "push" if margin == 0 else ("win" if margin > 0 else "loss")
    if r.kind == "spread":
        team, _, num = r.side.strip().rpartition(" ")
        try:
            line = float(num)
        except ValueError:
            return "unresolved"
        if team == r.home:
            actual_margin = r.home_points - r.away_points
        elif team == r.away:
            actual_margin = r.away_points - r.home_points
        else:
            return "unresolved"
        covered = actual_margin + line
        return "push" if abs(covered) < 1e-9 else ("win" if covered > 0 else "loss")
    return "unresolved"


def _report(label: str, sub: pd.DataFrame) -> None:
    c = sub["result"].value_counts()
    w, l, psh = int(c.get("win", 0)), int(c.get("loss", 0)), int(c.get("push", 0))
    decided = sub[sub["result"].isin(["win", "loss"])]
    risked = decided["stake_units"].sum()
    net = sub["profit_units"].sum()
    roi = f"{net / risked:.1%}" if risked else "n/a"
    print(f"  {label:10s} {w}-{l}-{psh}  risked {risked:.1f}u  "
          f"net {net:+.2f}u  ROI {roi}")


def grade(season: Optional[int] = None, log_path: Optional[str] = None,
          out_path: Optional[str] = None, cfbd_dir: Optional[str] = None) -> int:
    log_path = log_path or LOG
    out_path = out_path or GRADED
    cfbd_dir = cfbd_dir or CFBD
    if not os.path.exists(log_path):
        raise SystemExit("no bet log yet -- record a bet with --add first")
    log = pd.read_csv(log_path)
    if season:
        log = log[log["season"] == season]
    if log.empty:
        print("[bet_log] nothing to grade")
        return 0

    finals = []
    for yr in sorted(log["season"].unique()):
        p = os.path.join(cfbd_dir, f"games_{yr}.parquet")
        if os.path.exists(p):
            f = pd.read_parquet(p)
            id_col = "id" if "id" in f.columns else "game_id"
            finals.append(f[[id_col, "home_points", "away_points"]]
                          .rename(columns={id_col: "game_id"}))
    if not finals:
        raise SystemExit("no games cache to grade against")
    g = pd.concat(finals).dropna(subset=["home_points", "away_points"])

    df = log.merge(g, on="game_id", how="left")
    played = df.dropna(subset=["home_points"]).copy()
    pending = len(df) - len(played)
    if played.empty:
        print(f"[bet_log] {len(df)} bet(s) recorded, 0 played yet -- "
              f"nothing to grade")
        return 0

    played["result"] = played.apply(_grade_row, axis=1)
    played["profit_units"] = played.apply(
        lambda r: (r.stake_units * RP.payout(r.price) if r.result == "win"
                   else -r.stake_units if r.result == "loss" else 0.0),
        axis=1)

    played.to_csv(out_path, index=False)

    print(f"[bet_log] {len(df)} bet(s) recorded, {len(played)} played, "
          f"{pending} pending")
    _report("overall", played)
    for kind, sub in played.groupby("kind"):
        _report(kind, sub)
    print(f"  wrote {out_path}")
    print("\n  A handful of graded bets is not a track record. Nothing here "
          "says bet more -- it says keep logging and keep grading.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--game-id", type=int)
    ap.add_argument("--kind", choices=["spread", "moneyline"])
    ap.add_argument("--side", type=str)
    ap.add_argument("--price", type=float)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--book", type=str, default="")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    if args.add:
        needed = {"season": args.season, "week": args.week,
                  "game_id": args.game_id, "kind": args.kind,
                  "side": args.side, "price": args.price}
        missing = [n for n, v in needed.items() if v is None]
        if missing:
            flags = ", ".join("--" + m.replace("_", "-") for m in missing)
            raise SystemExit(f"--add needs: {flags}")
        return add(args.season, args.week, args.game_id, args.kind,
                   args.side, args.price, args.stake, args.book, args.note)
    if args.grade:
        return grade(args.season)
    raise SystemExit("pick one: --add or --grade")


if __name__ == "__main__":
    raise SystemExit(main())
