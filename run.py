#!/usr/bin/env python3
"""Build the current NFL value-bet slate and (re)generate the dashboard.

    python3 run.py            # auto: live if an API key is set, else demo
    python3 run.py --demo     # force realistic demo data (no key needed)
    python3 run.py --live     # force live data (requires odds_api_key)

Open dashboard.html in a browser; it auto-refreshes, so leaving this on a
schedule keeps the page current.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nflvalue import config, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="force demo data")
    ap.add_argument("--live", action="store_true", help="force live data (needs key)")
    args = ap.parse_args()

    cfg = config.load_config()
    mode = "demo" if args.demo else ("live" if args.live else None)
    data = pipeline.run(cfg, mode=mode)

    s = data["summary"]
    print(f"\n  Mode: {data['mode'].upper()}   Games: {s['n_games']}   "
          f"Value bets: {s['n_value_bets']}   Value props: {s['n_value_props']}")
    picks = (data["value_bets"] + data["value_props"])[:10]
    if picks:
        print("\n  Top edges:")
        for b in sorted(picks, key=lambda x: -x["ev"])[:10]:
            print(f"    {b['ev']*100:+5.1f}%  {b['outcome'][:40]:<40} "
                  f"{b['price_american']:>6}  {b['best_book']}")
    else:
        print("\n  No +EV plays on this slate (the market is efficient — normal).")
    print(f"\n  Dashboard: {config.DASHBOARD_PATH}\n")


if __name__ == "__main__":
    main()
