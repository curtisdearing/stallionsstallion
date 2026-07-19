#!/usr/bin/env python3
"""Bootstrap the CFBD parquet cache for a range of seasons.

Workstream A. Replaces the nflverse-shaped bootstrap that shipped with the
fablesfable duplication (that version pulled nflverse play-by-play and never
imported here).

BUDGET — read this before running. The CFBD free tier is 1,000 calls/month
and this script is the single largest consumer in the project:

    per season = 1 teams + 1 games + 1 lines + 1 talent + 1 returning
                 + ~15 weekly /plays calls          =  ~20 calls

So a 5-season backfill is ~100 calls, or 10% of the monthly cap. The script
refuses to start if the projected cost would exceed the remaining budget
unless you pass --force, and it prints the running tally when it finishes.
`--dry-run` prints the plan and the projected cost without spending anything.

Usage:
    python3 scripts/bootstrap_history.py --start 2019 --end 2023 --conference MWC
    python3 scripts/bootstrap_history.py --start 2023 --end 2023 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import ingest  # noqa: E402

CALLS_PER_SEASON = 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--conference", default=None,
                    help="CFBD conference abbreviation, e.g. MWC. Omit for all FBS "
                         "(much larger /plays payloads, same call count).")
    ap.add_argument("--weeks", type=int, default=15,
                    help="regular-season weeks to pull per season (default 15)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the projected cost exceeds the "
                         "remaining monthly budget")
    args = ap.parse_args(argv)

    if args.end < args.start:
        ap.error("--end must be >= --start")

    seasons = list(range(args.start, args.end + 1))
    projected = len(seasons) * (5 + args.weeks)
    used = ingest.calls_used_this_month()
    remaining = ingest.CALL_BUDGET_MONTHLY - used

    print(f"seasons        : {seasons}")
    print(f"conference     : {args.conference or 'ALL FBS'}")
    print(f"projected calls: ~{projected}")
    print(f"budget         : {used} used / {ingest.CALL_BUDGET_MONTHLY} this "
          f"month ({remaining} remaining)")

    if args.dry_run:
        print("\n[dry-run] nothing pulled.")
        return 0

    if projected > remaining and not args.force:
        print(f"\nREFUSING TO RUN: projected ~{projected} calls exceeds the "
              f"{remaining} remaining this month. Narrow the range, or pass "
              "--force if you know the tally is wrong.", file=sys.stderr)
        return 2

    failures = []
    for season in seasons:
        res = ingest.refresh_season(season, conference=args.conference,
                                    weeks=range(1, args.weeks + 1))
        print(res)
        if res.stale:
            # Fail loud, keep going: one bad season must not silently truncate
            # the backfill, and the cache for that season is left untouched.
            failures.append((season, res.error))

    print(f"\nmonthly tally now: {ingest.calls_used_this_month()} / "
          f"{ingest.CALL_BUDGET_MONTHLY}")

    if failures:
        print("\nSEASONS THAT DID NOT REFRESH (cache left as-is):", file=sys.stderr)
        for season, err in failures:
            print(f"  {season}: {err}", file=sys.stderr)
        return 1

    print("all seasons refreshed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
