#!/usr/bin/env python3
"""Phase 0 — CFBD smoke test (runnable implementation of PHASE0_SMOKE_TEST.md).

This is the executable form of the task brief in `PHASE0_SMOKE_TEST.md`. It
pulls one season of play-by-play, betting lines, and talent composite for a
single conference and checks that `/plays`, `/lines`, and `/talent` return the
fields the whole modeling plan (`research/cfb-portability-plan.md`) assumes —
before any engineering time goes into rewriting `ingest.py`.

Default slice (decided 2026-07-17, see PHASE0_DECISIONS.md): Mountain West,
2023 — a membership-stable G5 conference the year BEFORE the 2026 realignment,
with modern PPA coverage. Override with --conference / --year.

USAGE
  # 1. Get a free key at https://collegefootballdata.com (1,000 calls/month)
  # 2. Provide it WITHOUT committing it:
  export CFBD_API_KEY=xxxxx        # or add "cfbd_api_key" to config.local.json
  # 3. Run:
  python phase0_smoke_test.py                       # live, default slice
  python phase0_smoke_test.py --conference SBC --year 2023
  python phase0_smoke_test.py --selftest            # OFFLINE — no key, no network

EXIT CODE: 0 = PASS, 1 = FAIL/PARTIAL (see the printed report + data/phase0_report.md).

Design note: all the *judgement* lives in pure validate_*() functions that take
raw API rows and return findings. The network layer only fetches. That split is
what lets --selftest exercise every check against recorded fixtures with no key
and no network — the same reason fablesfable keeps `sources/demo.py` around.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

# Conference-name aliases -> the abbreviation CFBD's `conference` param wants.
CONFERENCE_ALIASES = {
    "mountain west": "MWC", "mwc": "MWC", "mw": "MWC",
    "sun belt": "SBC", "sbc": "SBC",
    "mac": "MAC", "mid-american": "MAC",
    "aac": "American", "american": "American",
    "cusa": "CUSA", "conference usa": "CUSA",
}

# A finding is (name, ok, detail). ok is True/False/None(=warn/unknown).
Finding = Tuple[str, Optional[bool], str]


# --------------------------------------------------------------------------
# field access: CFBD has shipped both snake_case and camelCase over its life,
# so never hard-code one spelling — try known aliases and record which hit.
# --------------------------------------------------------------------------
def pick(row: Dict, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def frac_nonnull(rows: List[Dict], *keys) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for r in rows if pick(r, *keys) is not None)
    return hits / len(rows)


# --------------------------------------------------------------------------
# pure validators — the heart of the smoke test, fully offline-testable
# --------------------------------------------------------------------------
def validate_games(games: List[Dict]) -> List[Finding]:
    f: List[Finding] = []
    f.append(("games returned", bool(games), f"{len(games)} games"))
    if not games:
        return f
    weeks = sorted({pick(g, "week") for g in games if pick(g, "week") is not None})
    f.append(("weeks populated", len(weeks) > 0, f"weeks {weeks[:1]}..{weeks[-1:]} ({len(weeks)} distinct)"))
    scored = frac_nonnull(games, "homePoints", "home_points")
    f.append(("final scores present", scored > 0.5, f"{scored:.0%} of games have home points"))
    return f


def validate_plays(plays: List[Dict]) -> List[Finding]:
    f: List[Finding] = []
    f.append(("plays returned", bool(plays), f"{len(plays)} plays"))
    if not plays:
        return f
    ppa = frac_nonnull(plays, "ppa", "PPA")
    # PPA is legitimately null on some plays (kneels, penalties). "Meaningful
    # fraction" per the brief — treat >=40% non-null as healthy, warn below.
    f.append(("PPA present", ppa >= 0.40 or None if ppa > 0 else False,
              f"{ppa:.0%} of plays have non-null PPA (CFB EPA equivalent)"))
    dd = min(frac_nonnull(plays, "down"), frac_nonnull(plays, "distance"),
             frac_nonnull(plays, "yardsToGoal", "yards_to_goal"))
    f.append(("down/distance/yardline populated", dd > 0.8, f"{dd:.0%} rows have down+distance+yardline"))
    ptypes = {pick(p, "playType", "play_type") for p in plays}
    ptypes.discard(None)
    f.append(("play type groupable", len(ptypes) >= 3,
              f"{len(ptypes)} distinct play types (need this for std/passing-downs splits)"))
    return f


def validate_lines(lines: List[Dict]) -> List[Finding]:
    f: List[Finding] = []
    f.append(("line rows returned", bool(lines), f"{len(lines)} game-line rows"))
    if not lines:
        return f
    # each game carries a nested `lines` list, one entry per sportsbook
    book_counts, has_spread, has_total, has_ml = [], 0, 0, 0
    for g in lines:
        books = pick(g, "lines", default=[]) or []
        book_counts.append(len(books))
        for b in books:
            if pick(b, "spread") is not None:
                has_spread += 1
            if pick(b, "overUnder", "over_under") is not None:
                has_total += 1
            if pick(b, "homeMoneyline", "home_moneyline") is not None:
                has_ml += 1
    multi = sum(1 for c in book_counts if c >= 2)
    f.append(("multi-book coverage", multi > 0,
              f"{multi}/{len(lines)} games have >=2 books (needed for de-vig/consensus)"))
    f.append(("spread present", has_spread > 0, f"{has_spread} book-rows carry a spread"))
    f.append(("total present", has_total > 0, f"{has_total} book-rows carry an over/under"))
    f.append(("moneyline present", has_ml > 0 or None,
              f"{has_ml} book-rows carry a moneyline (may be sparse for G5 — verify)"))
    return f


# CFBD serves talent==0 as a MISSING-DATA SENTINEL, not as a measurement.
# The 2026-07-18 live run found exactly three such teams — Army, Navy and
# Air Force — the service academies, whose recruits are largely unranked in
# the 247 composite. The next-lowest nonzero value league-wide is 4.33
# (Villanova), so 0 is categorically not "very low talent"; it is "no data".
# Air Force is a Mountain West team, so this lands inside the target slice.
#
# Curtis's call (2026-07-18): ingest 0 as NULL plus an `unranked_recruiting`
# flag, and let Workstream B fall back to a conference-mean prior for those
# teams. Never let 0.0 reach a rating as though it were a measurement.
#
# NOTE ON THIS EDIT: the scale check below was changed AFTER it failed, which
# normally would be gate-tuning. It is defensible only because the net effect
# is STRICTER, not looser: the scale assertion now runs over measured values
# (so it still catches a genuinely broken scale), and a NEW assertion pins the
# sentinel set to exactly the three known academies — so if CFBD ever starts
# emitting 0 for a fourth team, or stops emitting it for these, the gate FAILS
# and someone has to look. Do not "simplify" this back to `lo >= 0`.
TALENT_MISSING_SENTINEL = 0.0
EXPECTED_UNRANKED_TEAMS = {"Army", "Navy", "Air Force"}


def validate_talent(talent: List[Dict], team_names: set) -> List[Finding]:
    f: List[Finding] = []
    f.append(("talent rows returned", bool(talent), f"{len(talent)} teams league-wide"))
    if not talent:
        return f

    parsed = []
    for t in talent:
        raw = str(pick(t, "talent", default=""))
        if raw.replace(".", "", 1).replace("-", "", 1).isdigit():
            parsed.append((pick(t, "team", "school"), float(raw)))

    sentinel = {name for name, v in parsed if v == TALENT_MISSING_SENTINEL}
    measured = [v for _, v in parsed if v != TALENT_MISSING_SENTINEL]

    if measured:
        lo, hi = min(measured), max(measured)
        f.append(("talent scale sane (measured values only)", hi > lo > 0,
                  f"range {lo:.2f}..{hi:.2f} over {len(measured)} measured teams; "
                  f"{len(sentinel)} sentinel-0 teams excluded as missing"))

    # Stricter than the check it replaces: the sentinel set must be EXACTLY the
    # known service academies. A new name here means an unexamined data change.
    f.append(("talent sentinel set is exactly the known academies",
              sentinel == EXPECTED_UNRANKED_TEAMS,
              f"talent==0 for {sorted(sentinel) or 'nobody'} "
              f"(expected {sorted(EXPECTED_UNRANKED_TEAMS)}) — "
              f"these must ingest as NULL + unranked_recruiting=True, never as 0.0"))

    matched = {pick(t, "team", "school") for t in talent} & team_names
    f.append(("conference teams covered", len(matched) > 0,
              f"{len(matched)}/{len(team_names)} of the conference's teams have a talent row"))
    return f


def spot_check_game(games: List[Dict]) -> Dict:
    """Return one game's identity fields for a MANUAL box-score comparison."""
    if not games:
        return {}
    g = sorted(games, key=lambda x: str(pick(x, "startDate", "start_date", default="")))[0]
    return {
        "date": pick(g, "startDate", "start_date"),
        "home": pick(g, "homeTeam", "home_team"),
        "away": pick(g, "awayTeam", "away_team"),
        "home_points": pick(g, "homePoints", "home_points"),
        "away_points": pick(g, "awayPoints", "away_points"),
        "note": "Compare this against ESPN / Sports-Reference by hand (brief step 4).",
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def verdict(findings: List[Finding]) -> str:
    if any(ok is False for _, ok, _ in findings):
        return "FAIL"
    if any(ok is None for _, ok, _ in findings):
        return "PARTIAL"
    return "PASS"


def run_live(conference: str, year: int) -> Dict:
    from nflvalue.sources.cfbd import CFBDClient  # noqa: WPS433

    conf = CONFERENCE_ALIASES.get(conference.lower(), conference)
    client = CFBDClient()
    findings: List[Finding] = []

    games = client.games(year, conference=conf)
    findings += [("--- /games ---", None, "")] + validate_games(games)
    team_names = {pick(g, "homeTeam", "home_team") for g in games} | \
                 {pick(g, "awayTeam", "away_team") for g in games}
    team_names.discard(None)
    weeks = sorted({pick(g, "week") for g in games if pick(g, "week") is not None}) or [1]

    all_plays: List[Dict] = []
    for wk in weeks:                       # /plays is per-week — this is the budget cost
        all_plays += client.plays(year, wk, conference=conf)
    findings += [("--- /plays ---", None, f"{len(weeks)} week-calls")] + validate_plays(all_plays)

    lines = client.lines(year, conference=conf)
    findings += [("--- /lines ---", None, "")] + validate_lines(lines)

    talent = client.talent(year)
    findings += [("--- /talent ---", None, "")] + validate_talent(talent, team_names)

    return {
        "mode": "live", "conference": conf, "year": year,
        "calls_used": client.calls, "call_budget": 1000,
        "call_trail": client.call_log,
        "findings": findings,
        "spot_check": spot_check_game(games),
        "verdict": verdict([f for f in findings if f[1] is not None]),
    }


def run_selftest() -> Dict:
    """Offline: run every validator against recorded fixtures — no key/network."""
    fx_path = os.path.join(ROOT, "tests", "fixtures", "cfbd_smoke_fixtures.json")
    with open(fx_path) as fh:
        fx = json.load(fh)
    findings: List[Finding] = []
    findings += [("--- /games ---", None, "")] + validate_games(fx["games"])
    findings += [("--- /plays ---", None, "")] + validate_plays(fx["plays"])
    findings += [("--- /lines ---", None, "")] + validate_lines(fx["lines"])
    team_names = {g["homeTeam"] for g in fx["games"]} | {g["awayTeam"] for g in fx["games"]}
    findings += [("--- /talent ---", None, "")] + validate_talent(fx["talent"], team_names)
    return {
        "mode": "selftest", "conference": "MWC(fixture)", "year": 2023,
        "calls_used": 0, "call_budget": 1000, "call_trail": [],
        "findings": findings, "spot_check": spot_check_game(fx["games"]),
        "verdict": verdict([f for f in findings if f[1] is not None]),
    }


def render(report: Dict) -> str:
    icon = {True: "PASS", False: "FAIL", None: "warn"}
    lines = [
        f"# Phase 0 smoke test — {report['verdict']}",
        f"mode={report['mode']}  conference={report['conference']}  year={report['year']}",
        f"CFBD calls used: {report['calls_used']} / {report['call_budget']} monthly budget",
        "",
    ]
    for name, ok, detail in report["findings"]:
        if name.startswith("---"):
            lines.append(f"\n{name}")
        else:
            lines.append(f"  [{icon[ok]:4}] {name}: {detail}")
    sc = report.get("spot_check") or {}
    if sc:
        lines += ["", "Manual spot-check game (compare to ESPN/Sports-Reference):",
                  f"  {sc.get('date')}  {sc.get('away')} {sc.get('away_points')} "
                  f"@ {sc.get('home')} {sc.get('home_points')}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="CFBD Phase 0 smoke test")
    ap.add_argument("--conference", default="MWC", help="conference name or abbr (default MWC)")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--selftest", action="store_true", help="offline fixture run, no key/network")
    args = ap.parse_args()

    try:
        report = run_selftest() if args.selftest else run_live(args.conference, args.year)
    except Exception as exc:  # noqa: BLE001
        print(f"[phase0] ERROR: {exc}")
        return 1

    text = render(report)
    print(text)
    os.makedirs(DATA_DIR, exist_ok=True)
    report["generated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(DATA_DIR, "phase0_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(os.path.join(DATA_DIR, "phase0_report.md"), "w") as fh:
        fh.write(text + "\n")
    print(f"\n[phase0] wrote data/phase0_report.{{json,md}}  verdict={report['verdict']}")
    return 0 if report["verdict"] in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
