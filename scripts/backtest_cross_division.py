#!/usr/bin/env python3
"""Out-of-sample forecast-margin SD for FBS-vs-FCS games (the cross-division basis).

Pre-registered 2026-09-02T23:25Z in `_worklog.md`, BEFORE any residual was
computed. Read that entry before changing anything here: the population, the
estimator and every decision rule were fixed in advance so that the number
this produces cannot be chosen after seeing it.

Why this exists
---------------
The contract refused every FBS-vs-FCS game as "no validated cross-division
model". The ridge already fits a pooled `__FCS__` pseudo-team (down-weighted,
walk-forward), so a projection *exists*; what did not exist was any
out-of-sample evidence of how wrong it is. This script produces that evidence
and registers it. Whether the result is usable is decided by the rules below,
not by whoever reads the output.

Method (fixed in advance)
-------------------------
* Seasons 2021-2025 (2019: no prior season cached; 2020: COVID, excluded as
  in `backtest_early.py`).
* Every CFBD week present with a final margin; exactly one FBS side by the
  game frame's `home_fbs` / `away_fbs` flags.
* Table: `cross_division.solve_table` -- the early-season ridge for
  weeks 0-3, the plain in-season ridge for weeks 4+. Never the Kalman.
* Projection: `cross_division.predict_margin`, the FCS side
  standing on the pooled pseudo-team's own fitted coefficients.
* Walk-forward by construction: the frame handed to each solve is truncated
  to (season < s) or (season == s and week < w).
* Estimator: pooled residual SD (ddof=1) over all rows; usable iff n >= 300.
  ONE pooled SD covers every week; nothing per-week is selected afterwards.
* Decision rules: (a) a week-group SD (1-3 vs 4+) >25% from pooled is
  RECORDED, pooled still ships; (b) |mean residual| > 3.0 is RECORDED as a
  bias finding and NOT corrected; (c) a pooled SD BELOW the FBS-vs-FBS early
  basis is flagged as a suspected bug and the basis is marked unusable.

Outputs `data/backtest_cross_division.csv` and MERGES one basis entry into
`data/forecast_sd_basis.json` (the existing bases are never touched).
Cached data only; makes no CFBD call.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cross_division as XD, engine as EG, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_CSV = os.path.join(DATA, "backtest_cross_division.csv")
OUT_BASIS = os.path.join(DATA, "forecast_sd_basis.json")

PREREG = "2026-09-02T23:25Z"
BASIS_ID = "cross_division_ridge_wk1_15_2021_2025"
FAMILY = "ridge_cross_division"
COVERS_WEEKS = list(range(0, 21))

SEASONS = (2021, 2022, 2023, 2024, 2025)
HISTORY_START = 2019
MIN_N = 300
GROUP_TOL = 0.25
BIAS_TOL = 3.0

#: The FBS-vs-FBS early basis, for rule (c). Read from the manifest, never
#: recomputed.
FBS_EARLY_BASIS_ID = "early_ridge_wk1_3_2021_2025"


class CrossDivisionBacktestError(RuntimeError):
    """The pre-registered recipe could not be executed as written."""


def _truncated(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def collect_rows(games: pd.DataFrame, params) -> tuple:
    """Per-game walk-forward residual rows, plus a refusal tally."""
    rows, refusals = [], {"unratable": 0, "unsolvable_weeks": []}
    for season in SEASONS:
        weeks = sorted(int(w) for w in
                       games.loc[games["season"] == season, "week"].unique())
        for week in weeks:
            target = games[(games["season"] == season)
                           & (games["week"] == week)
                           & games["margin"].notna()]
            target = target[target["home_fbs"].astype(bool)
                            ^ target["away_fbs"].astype(bool)]
            if target.empty:
                continue
            frame = _truncated(games, season, week)
            try:
                tbl = XD.solve_table(frame, season, week, params)
            except (R.RatingsError, np.linalg.LinAlgError) as exc:
                refusals["unsolvable_weeks"].append(
                    f"{season} wk{week}: {type(exc).__name__}")
                continue
            for g in target.itertuples():
                fbs_is_home = bool(g.home_fbs)
                fbs_team = str(g.home_team) if fbs_is_home else str(g.away_team)
                try:
                    pred = XD.predict_margin(
                        tbl, fbs_team, fbs_is_home=fbs_is_home,
                        neutral=bool(g.neutral_site))
                except R.RatingsError:
                    refusals["unratable"] += 1
                    continue
                rows.append({
                    "season": int(season), "week": int(week), "id": int(g.id),
                    "home": str(g.home_team), "away": str(g.away_team),
                    "fbs_side": "home" if fbs_is_home else "away",
                    "neutral": bool(g.neutral_site),
                    "table": ("early_ridge" if EG.is_early_season(week)
                              else "ridge"),
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


def build_entry(df: pd.DataFrame, refusals: dict, fbs_early_sd) -> dict:
    """The registered basis entry. Reports; never selects."""
    pooled = _describe(df["residual"])
    early = df["week"] <= 3
    groups = {"weeks_1_3": _describe(df.loc[early, "residual"]),
              "weeks_4_plus": _describe(df.loc[~early, "residual"])}
    per_season = {str(s): _describe(df.loc[df["season"] == s, "residual"])
                  for s in sorted(df["season"].unique())}

    findings = []
    usable = pooled["n"] >= MIN_N
    if not usable:
        findings.append(
            f"UNDERPOWERED: n={pooled['n']} is below the pre-registered floor "
            f"of {MIN_N}. FBS-vs-FCS games stay refused.")
    for name, stats in groups.items():
        if stats["sd"] and pooled["sd"] and \
                abs(stats["sd"] - pooled["sd"]) / pooled["sd"] > GROUP_TOL:
            findings.append(
                f"{name} SD {stats['sd']} deviates more than "
                f"{int(GROUP_TOL * 100)}% from the pooled {pooled['sd']}. "
                "RECORDED per the pre-registration; the pooled figure still "
                "ships -- switching after seeing this would be estimator "
                "selection on the result.")
    if pooled["mean"] is not None and abs(pooled["mean"]) > BIAS_TOL:
        findings.append(
            f"BIAS: mean residual {pooled['mean']} exceeds {BIAS_TOL} pts "
            "(positive = the projection favours the home side by more than "
            "the result did). RECORDED; NOT corrected -- a correction would "
            "be an unregistered tunable. Surfaced to readers as the card "
            "caution.")
    if fbs_early_sd is not None and pooled["sd"] and pooled["sd"] < fbs_early_sd:
        usable = False
        findings.append(
            f"SUSPECTED BUG: pooled cross-division SD {pooled['sd']} is BELOW "
            f"the FBS-vs-FBS early basis {fbs_early_sd}. The pooled pseudo-team "
            "hides FCS heterogeneity, so it should be LESS predictable. Marked "
            "unusable per pre-registration rule (c); review before publishing.")

    descriptives = None
    if pooled["sd"]:
        sd = pooled["sd"]
        sign = np.where(df["fbs_side"] == "home", 1.0, -1.0)
        p_fbs = [_phi(float(m) * s / sd) for m, s in zip(df["pred_margin"], sign)]
        fbs_won = (df["margin"] * sign > 0)
        descriptives = {
            "fbs_win_rate_observed": round(float(fbs_won.mean()), 4),
            "fbs_win_rate_implied_mean": round(float(np.mean(p_fbs)), 4),
            "brier_fbs_win": round(float(np.mean(
                (np.array(p_fbs) - fbs_won.to_numpy(dtype=float)) ** 2)), 4),
            "note": "reported, not gating; p implied from Phi(pred/sd)",
        }

    return {
        "basis_id": BASIS_ID,
        "model_family": FAMILY,
        "covers_weeks": COVERS_WEEKS,
        "sd_source": "pooled_out_of_sample_residual",
        "forecast_margin_sd": pooled["sd"] if usable else None,
        "usable": bool(usable),
        "prereg": PREREG,
        "n": pooled["n"],
        "mean_residual": pooled["mean"],
        "mae": pooled["mae"],
        "window": ("seasons 2021-2025, every CFBD week with a final margin, "
                   "exactly one FBS side, walk-forward; early-season ridge "
                   "(wk0-3) / in-season ridge (wk4+) against the pooled "
                   "__FCS__ pseudo-team"),
        "source": "data/backtest_cross_division.csv",
        "fbs_vs_fbs_early_sd_for_comparison": fbs_early_sd,
        "per_group": groups,
        "per_season": per_season,
        "descriptives": descriptives,
        "refusals": refusals,
        "findings": findings,
        "note": ("Pooled FCS opposition: every FCS opponent is projected as the "
                 "one fitted pseudo-team, so the SD carries FCS heterogeneity "
                 "on top of forecast error. Nothing is corrected for the "
                 "recorded bias."),
    }


def merge_into_manifest(path: str, entry: dict) -> dict:
    """Replace-or-insert this basis by id; every other entry is untouched."""
    with open(path) as fh:
        manifest = json.load(fh)
    bases = [b for b in manifest.get("bases", [])
             if b.get("basis_id") != entry["basis_id"]]
    bases.append(entry)
    manifest["bases"] = bases
    manifest.setdefault("merged", [])
    manifest["merged"] = [m for m in manifest["merged"]
                          if m.get("basis_id") != entry["basis_id"]]
    manifest["merged"].append({
        "basis_id": entry["basis_id"],
        "merged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by": "scripts/backtest_cross_division.py",
    })
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-basis", default=OUT_BASIS)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing")
    args = ap.parse_args()

    with open(args.out_basis) as fh:
        manifest = json.load(fh)
    fbs_early = next((b for b in manifest.get("bases", [])
                      if b.get("basis_id") == FBS_EARLY_BASIS_ID), None)
    fbs_early_sd = (None if fbs_early is None
                    else fbs_early.get("forecast_margin_sd"))

    params = R.load_params()
    games = R.game_frame(list(range(HISTORY_START, max(SEASONS) + 1)))
    df, refusals = collect_rows(games, params)
    if df.empty:
        raise CrossDivisionBacktestError(
            "no FBS-vs-FCS games survived the pre-registered inclusion rules "
            "-- refusing to write an empty basis")

    entry = build_entry(df, refusals, fbs_early_sd)

    if not args.dry_run:
        df.to_csv(args.out_csv, index=False)
        merged = merge_into_manifest(args.out_basis, entry)
        with open(args.out_basis, "w") as fh:
            json.dump(merged, fh, indent=2)
            fh.write("\n")

    print(f"cross-division walk-forward residuals: n={entry['n']}")
    print(f"  pooled SD          {entry['forecast_margin_sd']}"
          f"{'' if entry['usable'] else '  (UNUSABLE)'}")
    print(f"  mean residual      {entry['mean_residual']}  (bias)")
    print(f"  MAE                {entry['mae']}")
    print(f"  FBS-vs-FBS early SD for comparison: {fbs_early_sd}")
    print(f"  refusals           {refusals}")
    print("  per group:")
    for name, stats in entry["per_group"].items():
        print(f"    {name}: n={stats['n']:<5} sd={stats['sd']:<8} "
              f"mean={stats['mean']:<9} mae={stats['mae']}")
    print("  per season:")
    for season, stats in entry["per_season"].items():
        print(f"    {season}: n={stats['n']:<5} sd={stats['sd']:<8} "
              f"mean={stats['mean']:<9} mae={stats['mae']}")
    if entry["descriptives"]:
        print(f"  descriptives       {entry['descriptives']}")
    for finding in entry["findings"]:
        print(f"  FINDING: {finding}")
    if not args.dry_run:
        print(f"wrote {args.out_csv} and merged {BASIS_ID} into {args.out_basis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
