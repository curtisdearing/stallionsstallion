#!/usr/bin/env python3
"""Evaluate one benchmark artifact. Writes a metrics record; selects nothing.

    python3 scripts/evaluate_benchmark.py --artifact backtest_c_corrected_wf_kalman_20260830
    python3 scripts/evaluate_benchmark.py --early          # the weeks 0-3 ridge route

Deliberately separate from `scripts/backtest_c.py`, which produces artifacts,
and from `scripts/build_evidence_manifest.py`, which records them. Computing a
metric must never be able to promote an artifact: the numbers this prints are
an input to a human's selection decision, not the decision.

Weeks 0-3 and weeks 4+ are evaluated SEPARATELY and never pooled. They are
different models (ridge + preseason prior vs the Kalman drive simulation) over
different windows, and a pooled figure would describe neither.

Metrics, all on the artifact's own rows:
  * margin MAE, RMSE and bias
  * Brier score for P(home win), against a coin flip at 0.2500
  * calibration / reliability bins (predicted probability -> observed rate)
  * coverage of the stated 50% and 80% central margin intervals, plus the PIT
    scale sd(z): z = (actual - projected) / stated sd. sd(z) > 1 means the
    stated intervals are too NARROW by that factor.
  * the refusal/exclusion counts the producing run recorded

The interval coverage uses a normal approximation from the stated SD, because
per-simulation quantiles are not persisted. That is the same recorded
approximation `scripts/calibration_o.py` uses, and sd(z) cross-checks it.
Cached data only; makes no API call.
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

from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

#: Central-interval levels whose coverage is reported.
LEVELS = (0.50, 0.80)


def _phi_inv(p: float) -> float:
    """Inverse standard normal, via the bisection of erf. stdlib only."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def coverage(actual: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> dict:
    out = {}
    for level in LEVELS:
        z = _phi_inv(0.5 + level / 2)
        inside = np.abs(actual - mean) <= z * sd
        out[f"p{int(level * 100)}"] = {
            "nominal": level,
            "observed": round(float(np.mean(inside)), 4),
            "n": int(len(actual)),
        }
    z_scores = (actual - mean) / sd
    out["pit_sd"] = round(float(np.std(z_scores, ddof=1)), 4)
    out["pit_mean"] = round(float(np.mean(z_scores)), 4)
    out["pit_note"] = ("sd(z) > 1 means the stated intervals are too narrow by "
                       "that factor; 1.0 is correctly scaled")
    return out


def reliability(p: np.ndarray, won: np.ndarray, n_bins: int = 10) -> list:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append({
            "bin": f"({edges[b]:.1f}, {edges[b + 1]:.1f}]",
            "n": int(mask.sum()),
            "pred": round(float(p[mask].mean()), 4),
            "observed": round(float(won[mask].mean()), 4),
        })
    return rows


def evaluate_frame(frame: pd.DataFrame, *, mean_col: str, sd_col: str,
                   prob_col: str, label: str) -> dict:
    actual = frame["margin"].to_numpy(dtype=float)
    mean = frame[mean_col].to_numpy(dtype=float)
    sd = frame[sd_col].to_numpy(dtype=float)
    prob = frame[prob_col].to_numpy(dtype=float)
    won = (actual > 0).astype(float)
    error = mean - actual
    return {
        "label": label,
        "n": int(len(frame)),
        "seasons": sorted(int(s) for s in frame["season"].unique()),
        "weeks": sorted(int(w) for w in frame["week"].unique()),
        "margin": {
            "mae": round(float(np.abs(error).mean()), 4),
            "rmse": round(float(np.sqrt((error ** 2).mean())), 4),
            "bias": round(float(error.mean()), 4),
            "corr": round(float(np.corrcoef(mean, actual)[0, 1]), 4),
        },
        "brier": {
            "score": round(float(((prob - won) ** 2).mean()), 4),
            "coin_flip": 0.25,
            "base_rate": round(float(won.mean()), 4),
        },
        "reliability_bins": reliability(prob, won),
        "interval_coverage": coverage(actual, mean, sd),
    }


def evaluate_wk4plus(artifact_id: str) -> dict:
    records = {r["artifact_id"]: r for r in EV.load_run_records(ROOT)}
    if artifact_id not in records:
        raise SystemExit(
            f"no run record for {artifact_id!r}; produce it with "
            "scripts/backtest_c.py --variant <name> first. Refusing to "
            "evaluate an artifact whose provenance was never recorded.")
    record = records[artifact_id]
    path = os.path.join(ROOT, record["path"])
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(
            f"{record['path']} holds no rows — every evaluated week was "
            "excluded. Reporting no metric rather than a metric over nothing.")
    out = evaluate_frame(frame, mean_col="mc_margin", sd_col="mc_margin_sd",
                         prob_col="p_home_win",
                         label=f"weeks >= 4 simulation ({artifact_id})")
    out["artifact_id"] = artifact_id
    out["path"] = record["path"]
    out["method"] = record["method"]
    out["engine_policy"] = record["engine_policy"]
    out["exclusions"] = record.get("exclusions", {})
    return out


def evaluate_early() -> dict:
    """Weeks 0-3 ridge, from its own registered artifact and basis SD.

    The early route publishes a point estimate plus a pooled out-of-sample SD,
    so its probability is Phi(margin / sd) by construction. Evaluating it here
    with the SAME sd the contract publishes is the only honest comparison —
    scoring it against some other spread would grade a model nobody runs.
    """
    with open(os.path.join(DATA, "forecast_sd_basis.json")) as handle:
        basis = json.load(handle)
    early = next(b for b in basis["bases"]
                 if b["model_family"] == "ridge_preseason_prior")
    if not early.get("usable"):
        raise SystemExit("the early basis is marked unusable; nothing to evaluate")
    sd = float(early["forecast_margin_sd"])

    frame = pd.read_csv(os.path.join(DATA, "backtest_early.csv"))
    frame = frame.assign(
        stated_sd=sd,
        p_home_win=[0.5 * (1 + math.erf(m / (sd * math.sqrt(2))))
                    for m in frame["pred_margin"]])
    out = evaluate_frame(frame, mean_col="pred_margin", sd_col="stated_sd",
                         prob_col="p_home_win",
                         label="weeks 1-3 ridge + preseason prior")
    out["artifact_id"] = "backtest_early_20260830"
    out["path"] = "data/backtest_early.csv"
    out["method"] = "walk_forward_ridge_preseason_prior_v1_20260830"
    out["stated_sd"] = sd
    out["basis_id"] = early["basis_id"]
    out["exclusions"] = {"by_cause": early.get("refusals", {}),
                         "note": ("unratable games were counted and excluded "
                                  "by scripts/backtest_early.py")}
    return out


def render(payload: dict) -> None:
    print(f"\n=== {payload['label']} ===")
    print(f"artifact: {payload['path']}  (n={payload['n']})")
    print(f"seasons {payload['seasons']}  weeks "
          f"{payload['weeks'][0]}-{payload['weeks'][-1]}")
    m = payload["margin"]
    print(f"margin: MAE {m['mae']}  RMSE {m['rmse']}  bias {m['bias']}  "
          f"corr {m['corr']}")
    b = payload["brier"]
    print(f"Brier:  {b['score']}  (coin flip {b['coin_flip']}, "
          f"home-win base rate {b['base_rate']})")
    print("reliability (predicted -> observed):")
    for row in payload["reliability_bins"]:
        print(f"  {row['bin']:14s} n={row['n']:4d}  pred {row['pred']:.3f}  "
              f"obs {row['observed']:.3f}")
    cov = payload["interval_coverage"]
    for level in LEVELS:
        key = f"p{int(level * 100)}"
        print(f"coverage {key}: nominal {cov[key]['nominal']:.2f}  observed "
              f"{cov[key]['observed']:.4f}")
    print(f"PIT: sd(z) {cov['pit_sd']}  mean(z) {cov['pit_mean']}  "
          f"({cov['pit_note']})")
    exclusions = (payload.get("exclusions") or {}).get("by_cause") or {}
    print(f"exclusions by cause: {exclusions or 'none recorded'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", help="run-record artifact_id (weeks 4+)")
    ap.add_argument("--early", action="store_true",
                    help="evaluate the weeks 0-3 ridge route instead")
    ap.add_argument("--out-dir", default=os.path.join(DATA, "evidence"))
    args = ap.parse_args()
    if bool(args.artifact) == bool(args.early):
        raise SystemExit("give exactly one of --artifact or --early")

    payload = evaluate_early() if args.early else evaluate_wk4plus(args.artifact)
    payload["evaluated_at_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    payload["paper_only"] = True
    payload["note"] = ("Diagnostic evidence. Computing this changes no engine "
                       "policy and selects no artifact.")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"eval_{payload['artifact_id']}.json")
    from nflvalue import atomic
    atomic.write_json(out, payload)
    render(payload)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
