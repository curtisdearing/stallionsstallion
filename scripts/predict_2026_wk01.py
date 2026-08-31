#!/usr/bin/env python3
"""DEMO/FIXTURE ONLY — the early-season slate, rendered for inspection.

This script used to be the SECOND writer of the live public artifact. It and
`scripts/weekly_loop.py` both owned `data/predictions_{season}_wk{NN}.csv`,
emitted different schemas into it, and whichever ran last won. That is the
defect the versioned forecast contract exists to remove, so this file no
longer writes the live artifact at all:

  * every number it shows comes from `nflvalue.producers.build_early_week`,
    the same shared producer the weekly loop calls for weeks 0-3, through the
    same `forecast_contract.build_artifact`. There is no second assembly site
    left, so there is no second schema to drift into;
  * `--out` is confined to a demo directory. Pointing it at the live artifact,
    a snapshot, or the published site is refused, not warned about. The weekly
    loop is the only supported path to a published forecast.

Weeks 0-3 are the least predictable of the season and the forecast SD says so:
it is the registered out-of-sample early-week residual SD from
`data/forecast_sd_basis.json` (pre-registered 2026-08-30T17:20Z,
`scripts/backtest_early.py`), not the rating band and not the week-4-15 window
the previous version of this script borrowed. PAPER-ONLY: nothing here is a
bet, an edge, a stake, a recommendation or an actionable pick.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import engine as EG, forecast_contract as FC  # noqa: E402
from nflvalue import ingest, producers as PR, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEMO_DIR = os.path.join(DATA, "demo")
HISTORY = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEASON = 2026
WEEK = 1                       # CFBD labels week-0 games as week 1

#: The live artifact's filename shape. Writing one of these from here is what
#: created the two-writer problem in the first place.
LIVE_ARTIFACT = re.compile(r"predictions_\d{4}_wk\d+\.csv$")

#: Trees this script may never write into, whatever name is passed.
PROTECTED = ("snapshots", "site")


class NotAPublisherError(RuntimeError):
    """This script tried to write somewhere only the weekly loop may write."""


def check_destination(path: str) -> str:
    """Refuse any destination that is, or feeds, the public artifact."""
    resolved = os.path.abspath(path)
    relative = os.path.relpath(resolved, DATA)
    inside_data = not relative.startswith(os.pardir)

    if LIVE_ARTIFACT.search(os.path.basename(resolved)):
        raise NotAPublisherError(
            f"{resolved} has the live artifact's name. This script is a demo "
            "renderer; `scripts/weekly_loop.py --job pregame` is the only "
            "supported path to a published forecast")
    if inside_data:
        head = relative.split(os.sep)[0]
        if head in PROTECTED:
            raise NotAPublisherError(
                f"{resolved} is inside data/{head}, which only the weekly "
                "loop may write")
        if os.sep not in relative:
            raise NotAPublisherError(
                f"{resolved} is in the data/ root, where the published "
                "artifacts live. Use data/demo/ instead")
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--week", type=int, default=WEEK)
    ap.add_argument("--out", default=None,
                    help="demo destination (default: data/demo/...). Cannot "
                         "be the live artifact, a snapshot, or the site.")
    ap.add_argument("--no-write", action="store_true",
                    help="render to stdout only")
    args = ap.parse_args()

    if not EG.is_early_season(args.week):
        raise SystemExit(
            f"week {args.week} is priced by the in-season simulation route, "
            "not the early-season ridge route this script renders. Run "
            "`scripts/weekly_loop.py --job pregame` instead.")

    out = args.out or os.path.join(
        DEMO_DIR, f"early_{args.season}_wk{args.week:02d}_demo.csv")
    destination = check_destination(out)

    params = R.load_params()
    games = R.game_frame(HISTORY + ([args.season]
                                    if args.season not in HISTORY else []))
    schedule = ingest.load_games([args.season])
    schedule = schedule[schedule["week"] == args.week].sort_values("start_date")
    if schedule.empty:
        raise SystemExit(f"no {args.season} week {args.week} games cached")

    registry = FC.BasisRegistry.load(DATA)
    basis = registry.require("ridge_preseason_prior", args.week)
    rows = PR.build_early_week(
        args.season, args.week, games=games, schedule=schedule,
        generated_at_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        data_dir=DATA, params=params, registry=registry)

    if not args.no_write:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        FC.write_artifact(rows, destination)

    summary = FC.summarise(rows)
    print(f"[demo] {args.season} week {args.week}: {summary['n_rows']} rows, "
          f"{summary['by_status']} (schema v{FC.SCHEMA_VERSION})")
    print(f"[demo] forecast SD basis {basis['basis_id']}: "
          f"sd={basis['forecast_margin_sd']} over n={basis['n']} "
          f"({basis['window']})")
    shown = ["game_id", "slate", "public_date", "home", "away", "status",
             "projected_home_margin", "forecast_margin_sd", "p_home_win",
             "rating_parameter_sd"]
    priced = FC.forecast_bearing(rows)
    print(priced[shown].to_string(index=False) if len(priced)
          else "[demo] nothing priced")

    talent = ingest.load("talent", [args.season])
    returning = ingest.load("returning", [args.season])
    components = {"talent": 0 if talent.empty else len(talent),
                  "returning": 0 if returning.empty else len(returning)}
    absent = [k for k, n in components.items() if not n]
    print("\nPrior components present: "
          + ", ".join(f"{k} (n={n})" for k, n in components.items() if n)
          + (f"; ABSENT (z=0 fallback): {', '.join(absent)}" if absent else ""))
    if not args.no_write:
        print(f"[demo] wrote {destination} — NOT a published artifact")
    print("PAPER-ONLY research output. Not a bet, edge, stake or "
          "recommendation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
