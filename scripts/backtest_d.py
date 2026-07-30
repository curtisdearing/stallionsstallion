#!/usr/bin/env python3
"""Phase 2 gate (Workstream D) — does the blend beat BOTH inputs out-of-sample?

Pre-registered BEFORE the first run (2026-07-30):

  * inputs: the walk-forward MC margins from data/backtest_c.csv (gate 3) and
    the median closing spread; eval = the same 2021-2023 wk>=4 window, so
    every layer is graded on the same games. 2024-2025 stay untouched.
  * blend rule: nfelo-style inverse-MSE weighting, weight fit walk-forward on
    all accumulated OOS pairs strictly before the predicted week, floor 150
    pairs (below it: skipped, recorded). No other blend rule will be tried
    on this window — if inverse-MSE fails the gate, that is the finding.
    ("Repeatedly adjusting until a number clears a threshold is how a model
    gets fooled." — AGENT_BUILD_PROMPT failure mode #1.)
  * gate metric: mean squared error on margins. PASS iff
    MSE(blend) < MSE(model) AND MSE(blend) < MSE(market) on the pooled
    graded set (blend-defined rows).
  * CLV diagnostics (not the gate): paper log of open-taken candidates vs
    same-book close at the pre-registered primary edge threshold of 2.0
    points, plus a threshold sweep for context. Slice cuts per the spec
    (P4vP4 / G5vP4 / G5vG5 / week0-1 / neutral). This is Workstream K's raw
    material, NOT a bet recommendation. PAPER-ONLY until gate 7.

Writes data/backtest_d.csv (graded per-game rows) + data/clv_log_backtest.csv.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import gameline as GL  # noqa: E402
from nflvalue import ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_C = os.path.join(ROOT, "data", "backtest_c.csv")
OUT = os.path.join(ROOT, "data", "backtest_d.csv")
CLV_OUT = os.path.join(ROOT, "data", "clv_log_backtest.csv")

EVAL_SEASONS = [2021, 2022, 2023]
PRIMARY_EDGE_PTS = 2.0


def main() -> int:
    if not os.path.exists(BT_C):
        print("run scripts/backtest_c.py to completion first", file=sys.stderr)
        return 2
    df = pd.read_csv(BT_C)
    lines = ingest.load_lines(EVAL_SEASONS)
    med_close = (lines.dropna(subset=["spread"])
                 .groupby("game_id")["spread"].median())
    df["market"] = [-med_close.get(g) if g in med_close.index else np.nan
                    for g in df["id"]]

    df = GL.walk_forward_blend(df)      # adds blend, blend_w (walk-forward)
    graded = df.dropna(subset=["blend", "market"]).copy()
    skipped = int(df["blend"].isna().sum())

    print(f"=== PHASE 2 GATE — inverse-MSE blend, walk-forward, "
          f"pre-registered {EVAL_SEASONS} wk>=4 ===")
    print(f"graded n={len(graded)} | skipped (thin history, recorded) "
          f"n={skipped} | blend weight on model: "
          f"first {graded['blend_w'].iloc[0]:.3f} -> "
          f"last {graded['blend_w'].iloc[-1]:.3f}")

    def mse(col):
        return float(((graded[col] - graded["margin"]) ** 2).mean())

    def mae(col):
        return float((graded[col] - graded["margin"]).abs().mean())

    rows = []
    for season in EVAL_SEASONS + ["pooled"]:
        d = graded if season == "pooled" else graded[graded["season"] == season]
        if d.empty:
            continue
        r = {"season": season, "n": len(d)}
        for name, col in (("model", "mc_margin"), ("market", "market"),
                          ("blend", "blend")):
            r[f"mse_{name}"] = float(((d[col] - d["margin"]) ** 2).mean())
            r[f"mae_{name}"] = float((d[col] - d["margin"]).abs().mean())
        rows.append(r)
        print(f"{season}: n={r['n']:4} | MSE model {r['mse_model']:7.2f}  "
              f"market {r['mse_market']:7.2f}  blend {r['mse_blend']:7.2f} | "
              f"MAE {r['mae_model']:5.2f} / {r['mae_market']:5.2f} / "
              f"{r['mae_blend']:5.2f}")

    pooled = rows[-1]
    beats_model = pooled["mse_blend"] < pooled["mse_model"]
    beats_market = pooled["mse_blend"] < pooled["mse_market"]
    verdict = "PASS" if (beats_model and beats_market) else "FAIL"
    print(f"\nGATE VERDICT: {verdict} — blend "
          f"{'<' if beats_model else '>='} model and "
          f"{'<' if beats_market else '>='} market on pooled OOS MSE.")
    if verdict == "FAIL":
        print("A recorded FAIL is a successful run. Do NOT swap blend rules "
              "on this window to chase a PASS.")

    graded.to_csv(OUT, index=False)

    # ------------------------------------------------------------------ #
    # CLV diagnostics (paper log; Workstream K's raw material, not a gate)
    # ------------------------------------------------------------------ #
    games = R.game_frame(EVAL_SEASONS)
    # CLOSE-FREE signal only (retraction 2026-07-30): the raw MC margin. The
    # blend embeds the closing median, which turns CLV-vs-open into a
    # look-ahead artifact (+2.9 pts of fake CLV measured on this window).
    signal_by_game = dict(zip(graded["id"].astype(int), graded["mc_margin"]))
    log = GL.build_clv_log(games, lines, signal_by_game, PRIMARY_EDGE_PTS)
    log.to_csv(CLV_OUT, index=False)
    print(f"\n--- paper CLV vs same-book open, primary threshold "
          f"{PRIMARY_EDGE_PTS} pts (pre-registered) ---")
    if log.empty:
        print("no candidates at the primary threshold")
    else:
        print(f"candidates n={len(log)} | mean CLV {log['clv_pts'].mean():+.3f} "
              f"pts | median {log['clv_pts'].median():+.2f} | "
              f"share positive {float((log['clv_pts'] > 0).mean()):.3f}")
        by_slice = log.groupby("slice")["clv_pts"].agg(["count", "mean"])
        for s, r in by_slice.iterrows():
            print(f"  slice {s:8s} n={int(r['count']):4d}  "
                  f"mean CLV {r['mean']:+.3f}")
    print("\nthreshold sweep (context only, not pre-registered):")
    for th in (1.0, 2.0, 3.0, 4.0, 5.0):
        lg = GL.build_clv_log(games, lines, signal_by_game, th)
        if lg.empty:
            print(f"  edge>={th:.0f}: n=0")
            continue
        print(f"  edge>={th:.0f}: n={len(lg):5d}  mean CLV "
              f"{lg['clv_pts'].mean():+.3f}  pos {float((lg['clv_pts']>0).mean()):.3f}")

    print("\nPAPER-ONLY. Nothing above recommends a bet; the kill-check "
          "(gate 7) has not run, let alone passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
