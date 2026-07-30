#!/usr/bin/env python3
"""Workstream O — calibration deep-dive on the gate-3 simulation outputs.

Why (BUILD_SPEC §19): a well-calibrated model that loses to the closing line
is a fundamentally different object from a miscalibrated one, and raw Brier
alone can't tell them apart.

Computes, on the pooled 2021-2024 walk-forward rows:
  * Brier decomposition (Murphy): reliability / resolution / uncertainty
  * reliability table by predicted-probability bucket
  * band coverage: do the stated 50% and 80% margin bands contain the
    outcome 50% / 80% of the time? (Normal approximation from the stored
    mc_margin_sd — the per-sim quantiles were not persisted; this is a
    recorded approximation, and the PIT sd below cross-checks it.)
  * PIT-style z = (margin - mc_margin) / mc_margin_sd: sd(z) > 1 means the
    simulated distributions are too NARROW by that factor.
  * per-slice calibration (same slice constant as F6/gameline).

Post-hoc calibration correction (isotonic/Platt) is NOT applied: it would be
fit and evaluated on the same window, which is circular (the spec allows it
only if the correction itself survives out-of-sample — future, registered).

Writes data/calibration_report.txt. Diagnostic only; PAPER-ONLY stands.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from nflvalue import gameline as GL, ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ybar = float(y.mean())
    rel = res = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        w = m.mean()
        pb, yb = float(p[m].mean()), float(y[m].mean())
        rel += w * (pb - yb) ** 2
        res += w * (yb - ybar) ** 2
    unc = ybar * (1 - ybar)
    return rel, res, unc


def main() -> int:
    df = pd.concat([pd.read_csv(os.path.join(ROOT, "data", "backtest_c.csv")),
                    pd.read_csv(os.path.join(ROOT, "data",
                                             "backtest_c_2024.csv"))],
                   ignore_index=True)
    df["won"] = (df["margin"] > 0).astype(float)
    p = df["p_home_win"].to_numpy()
    y = df["won"].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    rel, res, unc = brier_decomposition(p, y)

    lines = [f"=== CALIBRATION DEEP-DIVE (Workstream O) — pooled 2021-2024 "
             f"walk-forward, n={len(df)} ===",
             f"Brier {brier:.4f} = reliability {rel:.4f} - resolution "
             f"{res:.4f} + uncertainty {unc:.4f}",
             f"  (decomposition identity check: {rel - res + unc:.4f}; "
             "small reliability + large resolution is the healthy shape)"]

    lines.append("\nreliability by decile (pred -> observed, n):")
    df["bucket"] = pd.cut(df["p_home_win"], np.linspace(0, 1, 11))
    for b, r in df.groupby("bucket", observed=True).agg(
            n=("won", "size"), pred=("p_home_win", "mean"),
            obs=("won", "mean")).iterrows():
        lines.append(f"  {str(b):14s} n={int(r['n']):4d} "
                     f"pred {r['pred']:.3f} obs {r['obs']:.3f} "
                     f"gap {r['obs'] - r['pred']:+.3f}")

    # Band coverage under the normal approximation
    zz = (df["margin"] - df["mc_margin"]) / df["mc_margin_sd"]
    for want, zc in ((0.50, stats.norm.ppf(0.75)), (0.80, stats.norm.ppf(0.90))):
        got = float((zz.abs() <= zc).mean())
        lines.append(f"\n{want:.0%} margin band coverage: {got:.1%} "
                     f"({'NARROW' if got < want else 'ok/wide'}; normal "
                     "approx from stored sd — see PIT below)")
    lines.append(f"PIT z: mean {float(zz.mean()):+.3f}, sd {float(zz.std()):.3f} "
                 "(sd > 1 = simulated distributions too narrow by that "
                 "factor; the known cause is rating-parameter uncertainty "
                 "not being simulated — net_band exists but is not sampled)")

    # Per-slice calibration
    games = R.game_frame([2021, 2022, 2023, 2024])
    meta = {int(g.id): GL.slice_tag(str(g.home_conf), str(g.away_conf),
                                    int(g.week), bool(g.neutral_site))
            for g in games.itertuples()}
    df["slice"] = [meta.get(i, "?") for i in df["id"]]
    lines.append("\nper-slice Brier + coverage(80):")
    for s, d in df.groupby("slice"):
        bb = float(np.mean((d["p_home_win"] - d["won"]) ** 2))
        zz_s = (d["margin"] - d["mc_margin"]) / d["mc_margin_sd"]
        cov = float((zz_s.abs() <= stats.norm.ppf(0.90)).mean())
        lines.append(f"  {s:8s} n={len(d):4d} Brier {bb:.4f} "
                     f"cov80 {cov:.1%}")

    lines.append("\nNo post-hoc calibration applied: fitting a correction on "
                 "the evaluation window is circular. If wanted, register an "
                 "isotonic fit on 2021-2023 graded on 2024+ first. "
                 "PAPER-ONLY; diagnostics, not an edge.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(ROOT, "data", "calibration_report.txt"), "w") as fh:
        fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
