#!/usr/bin/env python3
"""Workstream G — run the pre-registered model bakeoff.

Protocol registered in _worklog.md 2026-07-30 BEFORE any candidate ran; see
nflvalue/bakeoff.py's docstring for the restatement. Summary of what decides:
pooled 2021-2024 margin MSE on the intersection of games all candidates
price; within 1% of the incumbent ridge = tie, ridge stays; a non-incumbent
winner must also beat ridge on the single 2025 holdout pass.

Writes data/bakeoff_g_results.csv (per-candidate per-game predictions) and
data/bakeoff_g_report.txt. Losers keep their numbers.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LassoCV  # noqa: E402

from nflvalue import bakeoff as BK, ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT = 2025
FIRST_WEEK = 4
SEED = 20260730
TIE_BAND = 0.01     # within 1% of incumbent MSE = tie


def ridge_preds(games, seasons) -> pd.DataFrame:
    params = R.load_params()
    rows = []
    for s in seasons:
        for w in sorted(games.loc[games["season"] == s, "week"].unique()):
            if w < FIRST_WEEK:
                continue
            try:
                tbl = R.solve_cached(games, int(s), int(w), params)
            except (R.RatingsError, np.linalg.LinAlgError):
                continue
            wk = games[(games["season"] == s) & (games["week"] == w)]
            for g in wk.itertuples():
                if not (g.home_fbs and g.away_fbs):
                    continue
                try:
                    p = tbl.predict_margin(g.home_team, g.away_team,
                                           bool(g.neutral_site))
                except R.DisconnectedTeamError:
                    continue
                rows.append({"id": int(g.id), "season": int(s),
                             "week": int(w), "pred": p,
                             "margin": float(g.margin)})
    return pd.DataFrame(rows)


def main() -> int:
    all_seasons = list(range(2019, HOLDOUT + 1))
    games = R.game_frame(all_seasons)
    games_sel = games[games["season"] <= max(EVAL_SEASONS)]

    print("[bakeoff] burn-in grid fits (2019-2020 only, then frozen)")
    elo_grid = [{"k": k, "hfa_elo": h, "pts_per_elo": s,
                 "season_regress": r, "mov": False}
                for k in (12, 20, 32) for h in (40, 55, 70)
                for s in (0.03, 0.04, 0.055) for r in (0.2, 0.35)]
    elo_cfg, elo_mae = BK.fit_online_params(
        games_sel, lambda c: BK.EloModel(BK.EloParams(**c)), elo_grid)
    print(f"  elo frozen: {elo_cfg} (burn-in MAE {elo_mae:.2f})")
    gelo_grid = [dict(c, mov=True) for c in elo_grid]
    gelo_cfg, gelo_mae = BK.fit_online_params(
        games_sel, lambda c: BK.EloModel(BK.EloParams(**c)), gelo_grid)
    print(f"  g-elo frozen: {gelo_cfg} (burn-in MAE {gelo_mae:.2f})")
    kal_grid = [{"obs_sd": o, "week_sd": w, "season_sd": s,
                 "prior_sd": 10.0, "hfa": h}
                for o in (14.0, 16.0, 18.0) for w in (0.4, 0.8, 1.4)
                for s in (4.0, 6.0, 9.0) for h in (2.4, 2.8, 3.2)]
    kal_cfg, kal_mae = BK.fit_online_params(
        games_sel, lambda c: BK.KalmanModel(BK.KalmanParams(**c)), kal_grid)
    print(f"  kalman frozen: {kal_cfg} (burn-in MAE {kal_mae:.2f})")

    print("[bakeoff] building Tier-1 feature frame")
    feats = BK.feature_frame([s for s in all_seasons if s <= max(EVAL_SEASONS)])

    candidates = {}
    print("[bakeoff] ridge (incumbent)")
    candidates["ridge"] = ridge_preds(games_sel, EVAL_SEASONS)
    print("[bakeoff] elo")
    candidates["elo"] = BK.run_online_model(
        BK.EloModel(BK.EloParams(**elo_cfg)), games_sel, EVAL_SEASONS)
    print("[bakeoff] g-elo")
    candidates["g_elo"] = BK.run_online_model(
        BK.EloModel(BK.EloParams(**gelo_cfg)), games_sel, EVAL_SEASONS)
    print("[bakeoff] kalman state-space (diagonal appx)")
    candidates["kalman"] = BK.run_online_model(
        BK.KalmanModel(BK.KalmanParams(**kal_cfg)), games_sel, EVAL_SEASONS)
    print("[bakeoff] gbdt")
    candidates["gbdt"] = BK.run_sklearn_model(
        lambda: HistGradientBoostingRegressor(random_state=SEED),
        feats, EVAL_SEASONS)
    print("[bakeoff] lasso")
    candidates["lasso"] = BK.run_sklearn_model(
        lambda: LassoCV(cv=5, random_state=SEED), feats, EVAL_SEASONS)
    print("[bakeoff] random forest")
    candidates["rf"] = BK.run_sklearn_model(
        lambda: RandomForestRegressor(n_estimators=200, random_state=SEED,
                                      n_jobs=-1),
        feats, EVAL_SEASONS)

    common = None
    for name, df in candidates.items():
        ids = set(df["id"])
        common = ids if common is None else (common & ids)
    lines_out = [f"=== WORKSTREAM G BAKEOFF — pre-registered folds "
                 f"{EVAL_SEASONS} wk>={FIRST_WEEK}; intersection n={len(common)} ==="]

    merged = []
    results = {}
    market = None
    lines = ingest.load_lines(EVAL_SEASONS)
    med = lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median()
    for name, df in candidates.items():
        d = df[df["id"].isin(common)].copy().sort_values("id")
        mse = float(((d["pred"] - d["margin"]) ** 2).mean())
        mae = float((d["pred"] - d["margin"]).abs().mean())
        corr = float(np.corrcoef(d["pred"], d["margin"])[0, 1])
        results[name] = {"mse": mse, "mae": mae, "corr": corr}
        d["candidate"] = name
        merged.append(d)
        if market is None:
            mk = d.copy()
            mk["mpred"] = [-med.get(g, np.nan) for g in mk["id"]]
            mk = mk.dropna(subset=["mpred"])
            market = {"mse": float(((mk["mpred"] - mk["margin"]) ** 2).mean()),
                      "mae": float((mk["mpred"] - mk["margin"]).abs().mean()),
                      "n": len(mk)}
    pd.concat(merged, ignore_index=True).to_csv(
        os.path.join(ROOT, "data", "bakeoff_g_results.csv"), index=False)

    for name in sorted(results, key=lambda n: results[n]["mse"]):
        r = results[name]
        lines_out.append(f"{name:8s} MSE {r['mse']:7.2f}  MAE {r['mae']:5.2f}  "
                         f"corr {r['corr']:.3f}")
    lines_out.append(f"market   MSE {market['mse']:7.2f}  MAE "
                     f"{market['mae']:5.2f}  (n={market['n']}, closing median "
                     "— context, not a candidate)")

    inc = results["ridge"]["mse"]
    best = min(results, key=lambda n: results[n]["mse"])
    if best != "ridge" and results[best]["mse"] >= inc * (1 - TIE_BAND):
        lines_out.append(
            f"DECISION: TIE — best non-incumbent ({best}, "
            f"{results[best]['mse']:.2f}) is within {TIE_BAND:.0%} of ridge "
            f"({inc:.2f}); the incumbent stays (registered tie rule).")
        best = "ridge"
    elif best == "ridge":
        lines_out.append(f"DECISION: incumbent ridge wins outright ({inc:.2f}).")
    else:
        lines_out.append(
            f"DECISION (selection window): {best} "
            f"({results[best]['mse']:.2f} vs ridge {inc:.2f}) — must ALSO "
            f"beat ridge on the single {HOLDOUT} holdout pass to ship.")

    report = "\n".join(lines_out)
    print(report)
    with open(os.path.join(ROOT, "data", "bakeoff_g_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print("\nHoldout pass NOT run automatically — it is a single, deliberate "
          "action once the selection result is recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
