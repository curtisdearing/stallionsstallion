#!/usr/bin/env python3
"""Backfill the CFBD /drives cache for a range of seasons.

Workstream C prerequisite (2026-07-30). The plays cache has NO drive
identifiers, so the drive-level Monte Carlo in BUILD_SPEC section 4 cannot be
built from it. /drives is bulk: **1 call per season** (probed live before this
was written — 2023 week 1 alone returned 2,913 fully-populated rows), so a
2019-2025 backfill costs ~7 calls against the 1,000/month cap.

Requires the games cache for each season (week is joined from it — /drives
serves no week column). Refuses to start if the projected cost exceeds the
remaining budget unless --force.

Usage:
    python3 scripts/backfill_drives.py --start 2019 --end 2025
    python3 scripts/backfill_drives.py --start 2023 --end 2023 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import ingest  # noqa: E402

CALLS_PER_SEASON = 1


def sanity_report(season: int) -> list:
    """Post-pull honesty checks. Returns a list of FAILURE strings (empty=ok).

    These are the same style of assertion the Phase 0 smoke test makes: loud,
    specific, and never auto-loosened to make a pull look good.
    """
    problems = []
    d = ingest.load_drives([season])
    g = ingest.load_games([season])
    if d.empty:
        return [f"{season}: drives table empty after refresh"]
    games_with_drives = d["game_id"].nunique()
    # Coverage is measured against FBS-INVOLVING games, not all of /games.
    # Learned on the first live run (2026-07-30): /games returns every
    # college game (~2,700-2,900 non-FBS matchups per season) while /drives
    # only covers games with play-by-play; measured coverage on the modeling
    # slice was 99.9-100% for 2019-2025 while all-games coverage was 35-54%.
    # The denominator was wrong, not the data.
    fbs_any = g[~g["home_fcs"] | ~g["away_fcs"]]
    covered = fbs_any["id"].isin(set(d["game_id"].unique()))
    cover = float(covered.mean()) if len(fbs_any) else 0.0
    if cover < 0.98:
        problems.append(
            f"{season}: only {int(covered.sum())}/{len(fbs_any)} FBS-involving "
            f"games ({cover:.1%}) have drives — expected >=98%")
    per_game = len(d) / max(games_with_drives, 1)
    if not 18 <= per_game <= 36:
        problems.append(
            f"{season}: {per_game:.1f} drives/game is outside the sane "
            "18-36 band — wrong join or truncated pull")
    for col in ("start_yards_to_goal", "drive_result", "start_offense_score"):
        share = float(d[col].notna().mean()) if col != "drive_result" else \
            float((d[col].astype(str) != "").mean())
        if share < 0.95:
            problems.append(f"{season}: {col} only {share:.0%} populated")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.end < args.start:
        ap.error("--end must be >= --start")

    seasons = list(range(args.start, args.end + 1))
    projected = len(seasons) * CALLS_PER_SEASON
    used = ingest.calls_used_this_month()
    remaining = ingest.CALL_BUDGET_MONTHLY - used
    print(f"seasons        : {seasons}")
    print(f"projected calls: ~{projected}")
    print(f"budget         : {used} used / {ingest.CALL_BUDGET_MONTHLY} "
          f"({remaining} remaining)")
    if args.dry_run:
        print("\n[dry-run] nothing pulled.")
        return 0
    if projected > remaining and not args.force:
        print(f"\nREFUSING TO RUN: ~{projected} calls exceeds the {remaining} "
              "remaining this month.", file=sys.stderr)
        return 2

    failures = []
    for season in seasons:
        res = ingest.refresh_drives(season, checkpoint=True)
        print(res)
        if res.stale:
            failures.append((season, res.error))
            continue
        failures.extend((season, p) for p in sanity_report(season))

    print(f"\nmonthly tally now: {ingest.calls_used_this_month()} / "
          f"{ingest.CALL_BUDGET_MONTHLY}")
    if failures:
        print("\nPROBLEMS (cache for failing seasons left/kept as-is):",
              file=sys.stderr)
        for season, err in failures:
            print(f"  {season}: {err}", file=sys.stderr)
        return 1
    print("all seasons backfilled and sane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
