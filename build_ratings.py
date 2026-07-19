#!/usr/bin/env python3
"""Build CFB team ratings — driver for the Workstream B ridge engine.

REWRITTEN 2026-07-19 for college football. This file previously held
fablesfable's NFL rating builder: a sequential, single-game Elo-style update
(`off[h] += K * error` after each game, K=0.08) walking the schedule in order.
That is a fine estimator when every team plays 17 games and the league graph is
dense. It is the wrong one here — in the 2023 Mountain West slice, 39 of the 56
teams that appear play exactly ONE game in the data, and a sequential updater
cannot propagate strength through a chain it traverses once. BUILD_SPEC section
3 replaces it with a joint ridge solve over the whole game graph, refit weekly.

Nothing was deleted: the NFL original is intact in `nfl-sim/fablesfable/`, and
its `league_priors()` drive-rate/HFA estimation belongs to Workstream C
(simulation), not to the ratings layer. The engine itself lives in
`nflvalue/ratings.py`; this file is the CLI around it.

    python3 build_ratings.py --fit              # walk-forward parameter search
    python3 build_ratings.py --report           # solve every week, write outputs

Outputs (in ./data/):
    rating_params.json               fitted parameters — written only by --fit
    ratings_{season}.parquet         one row per (week, team); each row is the
                                     rating as known BEFORE that week's games
    ratings_units_{season}.parquet   the 8 unit ratings, same convention
    ratings_search_{season}.csv      the full parameter-search trace

NOTHING HERE RECOMMENDS A BET. This is a rating table. Paper-only until the
gate-7 kill-check returns a slice-specific GO.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from nflvalue import ingest, ratings as R

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

#: KEPT ON PURPOSE, and it is not CFB code. `nflvalue/sources/availability.py`
#: imports this map to invert ESPN's NFL display names back to nflverse
#: abbreviations. That module is an NFL-side utility inherited from the parent
#: project; deleting the map here would silently empty its lookup table rather
#: than fail, and `tests/test_availability.py` covers it. Left in place until
#: something in this project actually retires the NFL availability path.
ABBR = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders",
    "OAK": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "SD": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "STL": "Los Angeles Rams",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders", "WSH": "Washington Commanders",
}


def weekly_frames(games, season, params, units=None):
    """Solve every week. Week `w`'s row uses only games from weeks < `w`."""
    rows, urows = [], []
    weeks = sorted(int(w) for w in games.loc[games["season"] == season, "week"].unique())
    for w in weeks:
        try:
            tbl = R.solve_cached(games, season, w, params)
        except R.RatingsError:
            continue                          # week 1: nothing to fit on yet
        rows.append(tbl.to_frame())
        if units is None:
            continue
        try:
            u = R.solve_units(games, units, season, w, params)
        except R.RatingsError:
            continue
        teams = sorted({t for d in u.values() for t in d})
        urows.append(pd.DataFrame(
            [{"season": season, "week": w, "team": t,
              **{k: u[k].get(t) for k in R.UNIT_NAMES}} for t in teams]))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.concat(urows, ignore_index=True) if urows else pd.DataFrame())


def fit(games, season, units=None) -> R.RatingParams:
    print(f"[fit] walk-forward parameter search, season {season}")
    trace = R.walk_forward_search(games, season, R.DEFAULT_GRID)
    best = trace.attrs["best"]
    os.makedirs(DATA, exist_ok=True)
    trace.to_csv(os.path.join(DATA, f"ratings_search_{season}.csv"), index=False)

    # Boundary hits are reported, never quietly accepted. A parameter that
    # lands on the edge of its grid has not been fit — the data is asking for
    # something the grid does not contain, and in two cases here the grid edge
    # IS the guardrail (fcs_weight, prior_strength).
    for name, values in R.DEFAULT_GRID.items():
        chosen = getattr(best, name)
        if chosen in (min(values), max(values)):
            edge = "LOW" if chosen == min(values) else "HIGH"
            print(f"[fit] BOUNDARY: {name}={chosen} is the {edge} edge of its "
                  "grid — treat as unfit, not as an optimum")

    mae, n = R.walk_forward_error(games, season, best)
    print(f"[fit] best walk-forward MAE {mae:.3f} pts on n={n} gradable games")
    if n < 200:
        print(f"[fit] WARNING: n={n} cannot separate these settings. The spread "
              "across the whole grid is smaller than the sampling error. Treat "
              "the selected values as PROVISIONAL, not fit.")

    if units is not None and len(units):
        v, utrace = R.fit_explosiveness_shrinkage(games, units, season, best)
        utrace.to_csv(os.path.join(DATA, f"ratings_unit_search_{season}.csv"),
                      index=False)
        print(f"[fit] explosiveness_shrinkage={v} (fit against next-week "
              "explosiveness, not against margin — the margin objective is "
              "blind to it)")
        if v == max(utrace["value"]):
            print("[fit] BOUNDARY: the unit objective wants the MOST shrinkage "
                  "offered, i.e. it wants the explosiveness ratings pinned to "
                  "zero. Read that as 'no usable explosiveness signal on this "
                  "slice', not as a tuned value.")
        best = R.replace(best, explosiveness_shrinkage=v)
    else:
        print("[fit] explosiveness_shrinkage NOT fit (no play data); left at "
              f"{best.explosiveness_shrinkage}")

    print(f"[fit] wrote {R.save_params(best)}")
    return best


def report(games, season, params, units=None) -> None:
    print(f"\n=== ratings report: season {season} ===")
    weeks = sorted(int(w) for w in games.loc[games["season"] == season, "week"].unique())
    print(f"{'week':>5} {'games':>6} {'ratable':>9} {'mean_conn':>10} "
          f"{'mean_band':>10} {'hfa':>6} {'cond':>9}")
    for w in weeks:
        try:
            t = R.solve_cached(games, season, w, params)
        except R.RatingsError:
            continue
        f = t.to_frame()
        live = f[~f["veto"]]
        band = live["net_band"].mean() if len(live) else float("nan")
        print(f"{w:>5} {t.n_games_fit:>6} {len(live):>4}/{len(f):<4} "
              f"{f['connectivity'].mean():>10.3f} {band:>10.2f} "
              f"{t.hfa_league:>6.2f} {t.condition:>9.1f}")

    final = R.solve_cached(games, season, max(weeks), params)
    f = final.to_frame()
    print(f"\nfinal-week ratable teams ({int((~f['veto']).sum())} of {len(f)}):")
    print(f[~f["veto"]][["team", "off", "def", "net", "net_band",
                         "connectivity", "fbs_games"]].to_string(index=False))
    print(f"\nleague HFA, FIT not inherited from the NFL parent: "
          f"{final.hfa_league:.2f} pts")
    print(f"residual sd {final.residual_sd:.2f} | condition {final.condition:.1f} | "
          f"prior-misfit sd {final.prior_misfit_sd:.2f}")
    print("conference terms (net pts, games behind each):")
    for c in sorted(final.conf_off):
        print(f"   {c:<20} {final.conf_off[c] - final.conf_def[c]:+7.2f}  "
              f"n={final.conf_games.get(c, 0)}")
    if units is not None and len(units):
        print("\nunit stickiness (lag-1 autocorrelation of the raw per-game "
              "value, NOT opponent-adjusted, small n — a diagnostic):")
        print(R.unit_stickiness(games, units, season).to_string(index=False))
        print("   near-zero or negative means the unit does not persist week to "
              "week and should not be driving a projection yet.")

    print("\nfingerprint:", final.fingerprint())
    print("\nThis is a rating table, not advice. Paper-only until gate 7.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CFB ridge ratings builder")
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--fit", action="store_true",
                    help="run the walk-forward parameter search and save params")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    games = R.game_frame([args.season])
    if games.empty:
        print(f"no cached games for {args.season}; run the ingest first", file=sys.stderr)
        return 2

    try:
        units = R.play_units(ingest.load_plays_clean([args.season]))
    except Exception as e:                    # noqa: BLE001 — reported, not hidden
        print(f"[warn] unit ratings skipped: {e}")
        units = None

    if args.fit:
        params = fit(games, args.season, units)
    else:
        try:
            params = R.load_params()
        except R.RatingsError as e:
            print(str(e), file=sys.stderr)
            return 2

    ratings, unit_df = weekly_frames(games, args.season, params, units)
    os.makedirs(DATA, exist_ok=True)
    rp = os.path.join(DATA, f"ratings_{args.season}.parquet")
    ratings.to_parquet(rp, index=False)
    print(f"wrote {rp}  ({len(ratings)} rows, "
          f"{ratings['week'].nunique() if len(ratings) else 0} weekly refits)")
    if len(unit_df):
        up = os.path.join(DATA, f"ratings_units_{args.season}.parquet")
        unit_df.to_parquet(up, index=False)
        print(f"wrote {up}  ({len(unit_df)} rows)")

    if args.report:
        report(games, args.season, params, units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
