#!/usr/bin/env python3
"""Out-of-sample forecast-margin SD for the EARLY season (CFBD weeks 1-3).

Pre-registered 2026-08-30T17:20Z in `_worklog.md`, BEFORE any residual was
computed.  Read that entry before changing anything here: the population,
the estimator, and every decision rule were fixed in advance precisely so
that the number this produces cannot be chosen after seeing it.

Why this exists
---------------
The public forecast contract requires `forecast_margin_sd` to be the
validated out-of-sample predictive margin SD *for the week being priced*.
Every registered evidence window in this repository covers weeks 4-15/16
only -- backtest_b, backtest_c, backtest_c_kalman, backtest_c_2024,
calibration-O and the close-free CLV log all start at week 4.  Weeks 0-3
therefore had n=0, and the shipped early-season writer was forming
`p_home_win` from backtest_b's week-4-15 SD: an out-of-window extrapolation
presented as a forecast.

Method (fixed in advance)
-------------------------
* Seasons 2021-2025.  2019 has no prior season in the cache, so the preseason
  prior cannot be built; 2020 was excluded in advance for COVID schedule
  distortion and n=49.
* CFBD weeks 1, 2, 3; games with a final result; FBS-vs-FBS only, which is the
  production refusal rule.  Unratable games are counted as refusals, never
  priced.
* The production early-season branch and nothing else:
  `engine.solve_production_table` -> ridge + `ratings.preseason_prior`.
* Walk-forward by construction: the frame handed to each solve is truncated to
  (season < s) or (season == s and week < w), so no same-week or later
  information can reach the fit.  The prior-year net is derived from that same
  truncated frame.

Outputs `data/backtest_early.csv` (per-game rows) and
`data/forecast_sd_basis.json` (the registered basis manifest the forecast
contract validates against).  Cached data only; makes no CFBD call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import engine as EG, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_CSV = os.path.join(DATA, "backtest_early.csv")
OUT_BASIS = os.path.join(DATA, "forecast_sd_basis.json")

PREREG = "2026-08-30T17:20Z"

#: Fixed in advance. 2019: no prior season cached. 2020: COVID, n=49.
SEASONS = (2021, 2022, 2023, 2024, 2025)
EARLY_WEEKS = (1, 2, 3)
HISTORY_START = 2019

#: Pre-registered power floor. Below this the estimate is declared
#: underpowered and weeks 0-3 publish as `degraded` with no numbers.
MIN_N = 300

#: The registered week-4-15 anchor, quoted for comparison only. Not
#: recomputed here -- backtest_b.csv is a registered record.
ANCHOR_PATH = os.path.join(DATA, "backtest_b.csv")


class EarlyBacktestError(RuntimeError):
    """The pre-registered recipe could not be executed as written."""


def _truncated(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Everything strictly before (season, week). The walk-forward cut."""
    return games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))]


def collect_rows(games: pd.DataFrame, params) -> tuple:
    """Per-game walk-forward residual rows, plus a refusal tally."""
    rows, refusals = [], {"unratable": 0, "unsolvable_weeks": []}
    for season in SEASONS:
        for week in EARLY_WEEKS:
            frame = _truncated(games, season, week)
            try:
                tbl = EG.solve_production_table(frame, season, week, params)
            except (R.RatingsError, np.linalg.LinAlgError) as exc:
                refusals["unsolvable_weeks"].append(
                    f"{season} wk{week}: {type(exc).__name__}")
                continue
            target = games[(games["season"] == season)
                           & (games["week"] == week)
                           & games["margin"].notna()]
            for g in target.itertuples():
                if not (bool(g.home_fbs) and bool(g.away_fbs)):
                    continue
                try:
                    pred = tbl.predict_margin(str(g.home_team),
                                              str(g.away_team),
                                              bool(g.neutral_site))
                except R.RatingsError:
                    refusals["unratable"] += 1
                    continue
                rows.append({
                    "season": int(season), "week": int(week), "id": int(g.id),
                    "home": str(g.home_team), "away": str(g.away_team),
                    "neutral": bool(g.neutral_site),
                    "engine": EG.production_margin_engine(week),
                    "pred_margin": float(pred),
                    "margin": float(g.margin),
                    "residual": float(pred) - float(g.margin),
                })
    return pd.DataFrame(rows), refusals


def _describe(resid: pd.Series) -> dict:
    return {"n": int(len(resid)),
            "sd": (None if len(resid) < 2
                   else round(float(resid.std(ddof=1)), 4)),
            "mean": (None if resid.empty else round(float(resid.mean()), 4)),
            "mae": (None if resid.empty else round(float(resid.abs().mean()), 4))}


def build_basis(df: pd.DataFrame, refusals: dict) -> dict:
    """The registered basis manifest. Reports; never selects."""
    pooled = _describe(df["residual"])
    per_week = {str(w): _describe(df.loc[df["week"] == w, "residual"])
                for w in sorted(df["week"].unique())}
    per_season = {str(s): _describe(df.loc[df["season"] == s, "residual"])
                  for s in sorted(df["season"].unique())}

    anchor = None
    if os.path.exists(ANCHOR_PATH):
        bt = pd.read_csv(ANCHOR_PATH)
        anchor = {
            "source": "data/backtest_b.csv",
            "window": "seasons 2021-2023, weeks 4-15, walk-forward ridge",
            "n": int(len(bt)),
            "sd": round(float((bt["pred"] - bt["margin"]).std(ddof=1)), 4),
        }

    findings = []
    if pooled["n"] < MIN_N:
        findings.append(
            f"UNDERPOWERED: n={pooled['n']} is below the pre-registered floor "
            f"of {MIN_N}. Weeks 0-3 must publish as degraded with no forecast "
            "numbers rather than use this SD.")
    for week, stats in per_week.items():
        if stats["sd"] and pooled["sd"] and \
                abs(stats["sd"] - pooled["sd"]) / pooled["sd"] > 0.25:
            findings.append(
                f"week {week} SD {stats['sd']} deviates more than 25% from the "
                f"pooled {pooled['sd']}. RECORDED per the pre-registration; "
                "the pooled figure still ships this pass -- switching to "
                "per-week SDs after seeing this would be estimator selection "
                "on the result.")
    if anchor and pooled["sd"] and pooled["sd"] < anchor["sd"]:
        findings.append(
            f"UNEXPECTED: pooled early SD {pooled['sd']} is BELOW the "
            f"registered week-4-15 SD {anchor['sd']}. The prior expectation is "
            "that early weeks are less predictable. Pre-registration rule (b): "
            "flag for review as a suspected bug, do not publish quietly.")

    usable = pooled["n"] >= MIN_N
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
                                    .isoformat(timespec="seconds"),
        "prereg": PREREG,
        "bases": [
            {
                "basis_id": "early_ridge_wk1_3_2021_2025",
                "model_family": "ridge_preseason_prior",
                "covers_weeks": [0, 1, 2, 3],
                "sd_source": "pooled_out_of_sample_residual",
                "forecast_margin_sd": pooled["sd"] if usable else None,
                "usable": bool(usable),
                "n": pooled["n"],
                "mean_residual": pooled["mean"],
                "mae": pooled["mae"],
                "window": ("seasons 2021-2025, CFBD weeks 1-3, FBS-vs-FBS, "
                           "walk-forward, production early-season ridge + "
                           "preseason_prior"),
                "source": "data/backtest_early.csv",
                "per_week": per_week,
                "per_season": per_season,
                "refusals": refusals,
            },
            {
                "basis_id": "sim_kalman_wk4_plus",
                "model_family": "kalman_montecarlo",
                "covers_weeks": list(range(4, 21)),
                "sd_source": "per_game_predictive_simulation",
                "forecast_margin_sd": None,
                "usable": True,
                "note": ("The simulation states its own predictive margin SD "
                         "per game; there is no pooled constant to publish. "
                         "The window below is the registered evidence that "
                         "the simulation is calibrated, not the SD itself."),
                "window": ("backtest_c_kalman: seasons 2021-2023, weeks 4-15, "
                           "n=1766; confirmed on the 2024 holdout, n=612"),
                "source": "data/backtest_c_kalman.csv",
            },
        ],
        "anchor_week4_15": anchor,
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-basis", default=OUT_BASIS)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing")
    args = ap.parse_args()

    params = R.load_params()
    games = R.game_frame(list(range(HISTORY_START, max(SEASONS) + 1)))
    df, refusals = collect_rows(games, params)
    if df.empty:
        raise EarlyBacktestError(
            "no early-season games survived the pre-registered inclusion "
            "rules — refusing to write an empty basis")

    basis = build_basis(df, refusals)
    early = basis["bases"][0]

    if not args.dry_run:
        df.to_csv(args.out_csv, index=False)
        with open(args.out_basis, "w") as fh:
            json.dump(basis, fh, indent=2)
            fh.write("\n")

    print(f"early-season walk-forward residuals: n={early['n']}")
    print(f"  pooled SD          {early['forecast_margin_sd']}")
    print(f"  mean residual      {early['mean_residual']}  (bias)")
    print(f"  MAE                {early['mae']}")
    print(f"  refusals           {refusals}")
    print("  per week:")
    for week, stats in early["per_week"].items():
        print(f"    wk{week}: n={stats['n']:<5} sd={stats['sd']:<8} "
              f"mean={stats['mean']:<9} mae={stats['mae']}")
    print("  per season:")
    for season, stats in early["per_season"].items():
        print(f"    {season}: n={stats['n']:<5} sd={stats['sd']:<8} "
              f"mean={stats['mean']:<9} mae={stats['mae']}")
    if basis["anchor_week4_15"]:
        a = basis["anchor_week4_15"]
        print(f"  registered wk4-15 anchor: sd={a['sd']} (n={a['n']})")
    for finding in basis["findings"]:
        print(f"  FINDING: {finding}")
    if not args.dry_run:
        print(f"wrote {args.out_csv} and {args.out_basis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
