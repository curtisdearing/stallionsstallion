#!/usr/bin/env python3
"""Grade finished games and let the model learn from what it got wrong.

    python3 update_results.py                 # live: pull final scores, grade, learn
    python3 update_results.py --simulate-weeks 10
                                              # demo: play out 10 past weeks so the
                                              # model learns BEFORE today, then build
                                              # today's slate with the smarter weights

Run this after games finish (e.g. on a daily schedule). It updates the factor
weights and the dashboard's Model & Learning tab.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nflvalue import config, pipeline  # noqa: E402
from nflvalue.sources import live  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate-weeks", type=int, default=0,
                    help="DEMO: bootstrap learning over N simulated past weeks")
    args = ap.parse_args()
    cfg = config.load_config()

    if args.simulate_weeks > 0:
        print(f"  Simulating {args.simulate_weeks} weeks to bootstrap learning...")
        res = pipeline.simulate_learning(cfg, weeks=args.simulate_weeks)
        m = res["metrics"]
        print(f"  Graded {res['graded']} predictions across {res['weeks']} weeks.")
        print(f"  Bet hit rate {m['bets_win_rate']*100:.1f}%  |  Bet ROI {m['bets_roi']*100:+.1f}%  |  "
              f"Brier {m['brier']:.3f}  |  Bankroll {m['bankroll']:.1f}u")
        # rebuild today's slate using the freshly learned weights
        pipeline.run(cfg, mode="demo")
        print(f"\n  Dashboard updated: {config.DASHBOARD_PATH}\n")
        return

    if not cfg.get("odds_api_key"):
        print("  No API key set, and no --simulate-weeks given.")
        print("  In demo mode, run:  python3 update_results.py --simulate-weeks 10\n")
        return

    print("  Pulling final scores from The Odds API...")
    results = live.fetch_live_results(cfg)
    out = pipeline.grade_and_learn(cfg, results)
    m = out["metrics"]
    print(f"  Graded {out['graded']} new results.  "
          f"Bet hit rate {m['bets_win_rate']*100:.1f}%  |  Bet ROI {m['bets_roi']*100:+.1f}%  |  "
          f"Brier {m['brier']:.3f}")
    print(f"\n  Dashboard updated: {config.DASHBOARD_PATH}\n")


if __name__ == "__main__":
    main()
