#!/usr/bin/env python3
"""Workstream D part 2 — the pre-registered 2024 holdout evaluation.

THE PROTOCOL WAS REGISTERED IN _worklog.md (2026-07-30) BEFORE ANY 2024
NUMBER WAS LOOKED AT, after the Phase-2 gate FAILED on 2021-2023 for the
inverse-MSE blend (blend beat the raw model, lost to the market; measured
cause: model/market error correlation 0.902, which inverse-MSE ignores).

Registered protocol, verbatim intent:
  * eval window: 2024 wk>=4 only. 2025 STAYS VIRGIN (reserved for G/L).
  * model input: walk-forward MC margins for 2024, produced by the same
    gate-3 machinery (burn-in labelling 2019-2023, params from
    rating_params.json which was fit on data through 2023 — past-fit, so
    no 2024 information reaches it).
  * two blend rules graded independently, both walk-forward with history
    accumulated from 2021 onward (pooled 2021-2023 OOS pairs + expanding
    2024), same 150-pair floor:
      (a) inverse-MSE — replication: does the 2021-2023 FAIL replicate?
      (b) disagreement-OLS: pred = market + beta * (model - market).
  * SHIP RULE: disagreement-OLS becomes the project's blend iff it beats
    BOTH the raw model and the raw market on pooled 2024 MSE. Otherwise
    nothing ships and the FAIL is recorded.
  * CLV replication at the 2.0-pt primary threshold reported alongside.

No third rule gets tried if these disappoint. PAPER-ONLY throughout.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_ot, cfb_sim, gameline as GL  # noqa: E402
from nflvalue import ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_C = os.path.join(ROOT, "data", "backtest_c.csv")          # 2021-2023 (gate 3)
BT_C_2024 = os.path.join(ROOT, "data", "backtest_c_2024.csv")
OUT = os.path.join(ROOT, "data", "backtest_d2_2024.csv")
CLV_OUT = os.path.join(ROOT, "data", "clv_log_2024.csv")

HOLDOUT = 2024
HISTORY = [2019, 2020, 2021, 2022, 2023]
PRIMARY_EDGE_PTS = 2.0


def simulate_2024() -> pd.DataFrame:
    if os.path.exists(BT_C_2024):
        return pd.read_csv(BT_C_2024)
    params = R.load_params()
    seasons = HISTORY + [HOLDOUT]
    games = R.game_frame(seasons)
    drives = ingest.load_drives(seasons)
    ot = cfb_ot.fit_ot_params(ingest.load_drives(range(2021, 2026)))
    df = cfb_sim.walk_forward_backtest(
        games, drives, params, ot, eval_seasons=[HOLDOUT],
        burn_in_seasons=HISTORY, first_week=4, n_sims=4000)
    df.to_csv(BT_C_2024, index=False)
    return df


def attach_market(df: pd.DataFrame, seasons) -> pd.DataFrame:
    lines = ingest.load_lines(seasons)
    med = (lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median())
    df = df.copy()
    df["market"] = [-med.get(g) if g in med.index else np.nan for g in df["id"]]
    return df


def grade(d: pd.DataFrame, label: str) -> dict:
    r = {"rule": label, "n": len(d)}
    for name, col in (("model", "mc_margin"), ("market", "market"),
                      ("blend", "blend")):
        r[f"mse_{name}"] = float(((d[col] - d["margin"]) ** 2).mean())
        r[f"mae_{name}"] = float((d[col] - d["margin"]).abs().mean())
    return r


def main() -> int:
    hist = attach_market(pd.read_csv(BT_C), [2021, 2022, 2023])
    ev = attach_market(simulate_2024(), [HOLDOUT])
    print(f"[d2] 2024 walk-forward sims: n={len(ev)}")
    both = pd.concat([hist, ev], ignore_index=True)

    results = []
    for label, fn in (("inverse-MSE", GL.walk_forward_blend),
                      ("disagreement-OLS", GL.walk_forward_disagreement_blend)):
        blended = fn(both)
        d24 = blended[(blended["season"] == HOLDOUT)].dropna(
            subset=["blend", "market"])
        res = grade(d24, label)
        results.append((label, res, blended))
        extra = ""
        if "blend_beta" in d24.columns and d24["blend_beta"].notna().any():
            extra = (f" | beta first {d24['blend_beta'].iloc[0]:.3f} -> "
                     f"last {d24['blend_beta'].iloc[-1]:.3f}")
        if "blend_w" in d24.columns and d24["blend_w"].notna().any():
            extra = (f" | w first {d24['blend_w'].iloc[0]:.3f} -> "
                     f"last {d24['blend_w'].iloc[-1]:.3f}")
        print(f"\n{label}: n={res['n']} | MSE model {res['mse_model']:.2f}  "
              f"market {res['mse_market']:.2f}  blend {res['mse_blend']:.2f} | "
              f"MAE {res['mae_model']:.2f} / {res['mae_market']:.2f} / "
              f"{res['mae_blend']:.2f}{extra}")
        passed = (res["mse_blend"] < res["mse_model"]
                  and res["mse_blend"] < res["mse_market"])
        print(f"  -> {label}: {'PASS' if passed else 'FAIL'} on pooled 2024 MSE")

    # Ship decision, exactly as registered
    ship_label, ship_res, ship_blended = results[1]
    ships = (ship_res["mse_blend"] < ship_res["mse_model"]
             and ship_res["mse_blend"] < ship_res["mse_market"])
    print(f"\nSHIP DECISION (registered rule): disagreement-OLS "
          f"{'SHIPS' if ships else 'DOES NOT SHIP — recorded FAIL'}")

    d24 = ship_blended[ship_blended["season"] == HOLDOUT].dropna(
        subset=["blend", "market"])
    d24.to_csv(OUT, index=False)

    # CLV replication on 2024 — CLOSE-FREE signal only (retraction
    # 2026-07-30): the raw MC margin, never the blend, which embeds the
    # closing median and manufactures ~+3.5 pts of fake CLV on this window.
    games = R.game_frame([HOLDOUT])
    lines = ingest.load_lines([HOLDOUT])
    signal_by_game = dict(zip(d24["id"].astype(int), d24["mc_margin"]))
    log = GL.build_clv_log(games, lines, signal_by_game, PRIMARY_EDGE_PTS)
    log.to_csv(CLV_OUT, index=False)
    if log.empty:
        print("\nCLV replication: no candidates at 2.0 pts")
    else:
        pg = log.groupby("game_id")["clv_pts"].mean()
        t = pg.mean() / (pg.std() / np.sqrt(len(pg))) if len(pg) > 2 else float("nan")
        print(f"\nCLV replication on 2024 (threshold 2.0): n={len(log)} "
              f"candidates, mean {log['clv_pts'].mean():+.3f} pts | per-game "
              f"dedup n={len(pg)}, mean {pg.mean():+.3f}, t={t:.2f}")
        for s, r in log.groupby("slice")["clv_pts"].agg(["count", "mean"]).iterrows():
            print(f"  slice {s:8s} n={int(r['count']):4d}  mean {r['mean']:+.3f}")

    print("\n2025 remains untouched. PAPER-ONLY until gate 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
