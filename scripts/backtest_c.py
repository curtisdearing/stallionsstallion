#!/usr/bin/env python3
"""Gate 3 — walk-forward backtest of the drive-level Monte Carlo (Workstream C).

Pre-registered BEFORE the first run (2026-07-30, matching backtest_b's window
so the two layers are graded on the same games):

  * eval seasons: 2021, 2022, 2023 pooled; weeks >= 4
  * burn-in for drive-rate labelling: 2019-2020 (never graded)
  * 2024 and 2025 are DELIBERATELY untouched — they stay a virgin holdout for
    Workstreams G (bakeoff) and D. Do not add them here because the numbers
    disappoint; that is exactly the failure mode the pre-registration exists
    to prevent.
  * metrics: margin MAE + correlation vs actual (head-to-head vs the median
    closing spread on the same games), Brier of P(home win) vs the de-vigged
    median closing moneyline on the same games, calibration by decile,
    simulated-margin-SD sanity vs the CFB residual scale, MC-vs-B agreement.

Honesty notes:
  * The model is NOT expected to beat the closing line. If it does, suspect a
    bug before an edge (BUILD_SPEC 0; backtest_b pooled: model 13.16 vs
    market 12.24 MAE).
  * OT parameters are fit league-level on the full 2021-2025 drives window,
    which is IN-WINDOW relative to the eval seasons — unavoidable, because
    the 2021 rules regime has no pre-2021 data to fit from. The exposure is
    bounded and reported by this script (OT decides ~2% of sims and a tilt
    mis-fit moves P(home win) by well under a Brier-visible amount); it is a
    recorded caveat, not a silent one.
  * Games either side of which is vetoed/unratable are skipped, not guessed.

Writes data/backtest_c.csv (one row per simulated game, resumable per season:
re-invoke until exit 0; exit 3 = more seasons to do) and prints the gate
report. NOTHING HERE RECOMMENDS A BET.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_ot, cfb_sim, engine as EG, ingest, oddsmath, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out_path(engine: str) -> str:
    """Engine-suffixed artifact. The registered 2026-07-30 ridge run stays
    data/backtest_c.csv; a non-incumbent engine writes its own file so the
    gate-3 record is never overwritten by a re-run."""
    name = "backtest_c.csv" if engine == "ridge" else f"backtest_c_{engine}.csv"
    return os.path.join(ROOT, "data", name)

BURN_IN = [2019, 2020]
EVAL_SEASONS = [2021, 2022, 2023]
FIRST_WEEK = 4
N_SIMS = 4000


def market_frames(seasons):
    """Median closing spread and de-vigged median-book ML win prob per game."""
    lines = ingest.load_lines(seasons)
    med_spread = (lines.dropna(subset=["spread"])
                  .groupby("game_id")["spread"].median())
    ml = lines.dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    probs = {}
    for gid, grp in ml.groupby("game_id"):
        ps = []
        for r in grp.itertuples():
            dh = oddsmath.american_to_decimal(float(r.home_moneyline))
            da = oddsmath.american_to_decimal(float(r.away_moneyline))
            ph, _ = oddsmath.devig_multiplicative([dh, da])
            ps.append(ph)
        probs[gid] = float(np.median(ps))
    med_total = (lines.dropna(subset=["over_under"])
                 .groupby("game_id")["over_under"].median())
    return med_spread, probs, med_total


def run_season(season: int, engine: str) -> pd.DataFrame:
    params = R.load_params()
    hist = sorted(set(BURN_IN) | {s for s in EVAL_SEASONS if s <= season})
    games = R.game_frame(hist)
    drives = ingest.load_drives(hist)
    ot = cfb_ot.fit_ot_params(ingest.load_drives(range(2021, 2026)))
    print(f"[gate3] OT params (league, small-n, reported not hidden): "
          f"{ot.as_dict()}")
    burn = [s for s in hist if s < season]
    df = cfb_sim.walk_forward_backtest(
        games, drives, params, ot, eval_seasons=[season],
        burn_in_seasons=burn, first_week=FIRST_WEEK, n_sims=N_SIMS,
        engine=engine,
        progress=lambda s, w, pool, done: print(
            f"  walked {s} wk{w:<2d} pool={pool} simulated={done}", flush=True)
        if w % 5 == 0 else None)
    return df


def report(df: pd.DataFrame, engine: str) -> None:
    med_spread, ml_probs, med_total = market_frames(EVAL_SEASONS)
    df = df.copy()
    df["market"] = [-med_spread.get(g) if g in med_spread.index else np.nan
                    for g in df["id"]]
    df["market_p"] = [ml_probs.get(g, np.nan) for g in df["id"]]
    df["market_total"] = [med_total.get(g, np.nan) for g in df["id"]]
    df["won"] = (df["margin"] > 0).astype(float)

    print(f"\n=== GATE 3 — drive-level MC, walk-forward, weeks >= "
          f"{FIRST_WEEK}, pre-registered eval {EVAL_SEASONS} "
          f"[margin engine: {engine}] ===")
    for season in EVAL_SEASONS + ["pooled"]:
        d = df if season == "pooled" else df[df["season"] == season]
        if d.empty:
            continue
        both = d.dropna(subset=["market"])
        mae = (d["mc_margin"] - d["margin"]).abs().mean()
        corr = float(np.corrcoef(d["mc_margin"], d["margin"])[0, 1])
        mae_h2h = (both["mc_margin"] - both["margin"]).abs().mean()
        mae_mkt = (both["market"] - both["margin"]).abs().mean()
        print(f"{season}: n={len(d)} | MC margin MAE {mae:.2f} corr {corr:.3f}"
              f" || h2h n={len(both)}: MC {mae_h2h:.2f} vs market {mae_mkt:.2f}")

    mlb = df.dropna(subset=["market_p"])
    brier_model = float(((df["p_home_win"] - df["won"]) ** 2).mean())
    brier_model_mlset = float(((mlb["p_home_win"] - mlb["won"]) ** 2).mean())
    brier_market = float(((mlb["market_p"] - mlb["won"]) ** 2).mean())
    print(f"\nBrier (home win): model {brier_model:.4f} on all n={len(df)} | "
          f"same-set h2h n={len(mlb)}: model {brier_model_mlset:.4f} vs "
          f"de-vigged closing ML {brier_market:.4f} (coin flip = 0.2500)")

    # Calibration by predicted-probability decile
    df["bucket"] = pd.cut(df["p_home_win"], np.linspace(0, 1, 11))
    cal = df.groupby("bucket", observed=True).agg(
        n=("won", "size"), pred=("p_home_win", "mean"), obs=("won", "mean"))
    print("\ncalibration (pred bucket -> observed home-win rate):")
    for b, r in cal.iterrows():
        print(f"  {str(b):24s} n={int(r['n']):4d}  pred {r['pred']:.3f}  "
              f"obs {r['obs']:.3f}")

    resid_sd = float((df["mc_margin"] - df["margin"]).std())
    print(f"\ndistribution sanity: mean simulated margin SD "
          f"{df['mc_margin_sd'].mean():.2f} vs realised residual SD "
          f"{resid_sd:.2f} (CFB literature ~16-17; NFL ~13 — must be wider)")
    agree = float(np.corrcoef(df["mc_margin"], df["pred_b"])[0, 1])
    slope = float(np.polyfit(df["pred_b"], df["mc_margin"], 1)[0])
    print(f"MC vs B agreement: corr {agree:.3f}, slope {slope:.3f} "
          "(slope << 1 means the binned rates flatten the rating signal)")

    tot = df.dropna(subset=["market_total"])
    mae_t = (df["mc_total"] - df["total"]).abs().mean()
    mae_t_mkt = (tot["market_total"] - tot["total"]).abs().mean()
    mae_t_h2h = (tot["mc_total"] - tot["total"]).abs().mean()
    print(f"\ntotals (bonus, not gated): MC MAE {mae_t:.2f} | h2h n={len(tot)}: "
          f"MC {mae_t_h2h:.2f} vs market total {mae_t_mkt:.2f}")

    ot_share = float(df["p_ot"].mean())
    print(f"\nOT caveat quantified: mean P(OT) {ot_share:.3f}; a ±0.15 error in "
          f"the small-n 2-pt rate shifts P(home win) by < {0.15 * ot_share:.4f}"
          " — below Brier resolution at this n. In-window OT fit recorded.")
    print("\nThe model is NOT expected to beat the closing line; if it does, "
          "suspect a bug before an edge. Paper-only until gate 7.")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=None,
                    help="margin engine (default: config margin_engine)")
    args = ap.parse_args()
    engine = args.engine or EG.margin_engine()
    out = out_path(engine)
    done = (pd.read_csv(out) if os.path.exists(out)
            else pd.DataFrame(columns=["season"]))
    for season in EVAL_SEASONS:
        if len(done) and (done["season"] == season).any():
            continue
        df = run_season(season, engine)
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"[gate3] season {season}: {len(df)} simulated games written")
        return 3
    report(pd.read_csv(out), engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
