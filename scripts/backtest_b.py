#!/usr/bin/env python3
"""Walk-forward backtest of the re-fit B ratings vs the market, 2021-2023.

Honesty rules (BUILD_SPEC section 0): every number here is out-of-sample by
construction — week N is predicted from games strictly before week N — and the
market baseline is reported next to the model because a CFB model that beats
the closing line on the first try has a bug, not an edge.

Market baseline = median closing spread across books (home perspective:
CFBD's `spread` is negative when home is favoured, so implied home margin is
-spread). Only games where BOTH a model prediction and a market line exist are
compared head-to-head, so the two MAEs are computed on the same set.

Resumable: per-season results append to data/backtest_b.csv; re-invoke until
exit 0 (exit 3 = more seasons to do). NOTHING HERE RECOMMENDS A BET.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "backtest_b.csv")

HISTORY = [2019, 2020, 2021, 2022, 2023]
EVAL_SEASONS = [2021, 2022, 2023]
FIRST_WEEK = 4


def season_rows(games: pd.DataFrame, season: int, params: R.RatingParams) -> pd.DataFrame:
    lines = ingest.load(  # median close per game across books
        "lines", [season])
    med = (lines.dropna(subset=["spread"])
           .groupby("game_id")["spread"].median())

    recs = []
    weeks = sorted(int(w) for w in games.loc[games["season"] == season, "week"].unique())
    for w in weeks:
        if w < FIRST_WEEK:
            continue
        try:
            tbl = R.solve_cached(games, season, w, params)
        except R.RatingsError:
            continue
        wk = games[(games["season"] == season) & (games["week"] == w)]
        for r in wk.itertuples():
            if not (r.home_fbs and r.away_fbs):
                continue
            try:
                pred = tbl.predict_margin(r.home_team, r.away_team, bool(r.neutral_site))
            except R.DisconnectedTeamError:
                continue
            spread = med.get(r.id)
            recs.append({"season": season, "week": w, "id": r.id,
                         "pred": pred, "margin": float(r.margin),
                         "market": (-float(spread)) if spread is not None and not
                         np.isnan(spread) else np.nan})
    return pd.DataFrame(recs)


def main() -> int:
    params = R.load_params()
    games = R.game_frame(HISTORY)
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["season"])
    for season in EVAL_SEASONS:
        if len(done) and (done["season"] == season).any():
            continue
        df = season_rows(games, season, params)
        df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"[backtest] season {season}: {len(df)} predictions written")
        return 3  # one season per invocation keeps us under the 45s kill

    df = pd.read_csv(OUT)
    print(f"\n=== B walk-forward backtest, weeks >={FIRST_WEEK}, params from "
          f"refit_b (pre-registered eval: {EVAL_SEASONS}) ===")
    for season in EVAL_SEASONS:
        d = df[df["season"] == season]
        both = d.dropna(subset=["market"])
        mae_m = (d["pred"] - d["margin"]).abs().mean()
        corr = float(np.corrcoef(d["pred"], d["margin"])[0, 1])
        mae_model_h2h = (both["pred"] - both["margin"]).abs().mean()
        mae_market = (both["market"] - both["margin"]).abs().mean()
        corr_mkt = float(np.corrcoef(both["market"], both["margin"])[0, 1])
        print(f"season {season}: n={len(d)} | model MAE {mae_m:.2f} corr {corr:.3f}"
              f" || head-to-head n={len(both)}: model {mae_model_h2h:.2f}"
              f" vs market {mae_market:.2f} (market corr {corr_mkt:.3f})")
    both = df.dropna(subset=["market"])
    print(f"\npooled head-to-head n={len(both)}: model MAE "
          f"{(both['pred']-both['margin']).abs().mean():.2f} vs market "
          f"{(both['market']-both['margin']).abs().mean():.2f}")
    print("Model is NOT expected to beat the closing line; if it does, suspect "
          "a bug before an edge. Paper-only until gate 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
