"""Workstream E — the situational factor battery.

Every factor is a hypothesis with a test attached; nothing touches a
probability until it clears **n-gate + matched control + BH q<0.05** (the
project's non-negotiable #6 — there is a retraction on record from skipping
exactly this). Rejections are recorded and kept: the rejection list is as
valuable as the acceptance list.

The registry (`data/factor_registry.json`) is the single source of truth for
factor status, including the Tier-3 EXCLUSION list — code asserts against it
rather than against any second hand-maintained list
(FACTOR_REVALIDATION_SPEC §4). Promotion to `gated_active` is never done by
code in this module: a factor that PASSES here is recorded as `display_only`
with a passed-gate note, and wiring it into a probability is a separate,
human-visible decision.

Test shapes:

* **Event factors** (backup QB, first-year coach): treated games vs a
  MATCHED CONTROL — each treated game matched (without replacement) to the
  nearest untreated game by (same season, week within ±1, closest predicted
  margin). The outcome is the SIGNED walk-forward residual
  (margin - pred_b) oriented so negative = the treated side underperformed
  its rating. Welch's t on treated vs matched control.

* **Continuous factors** (finishing drives, field position): the factor is a
  walk-forward team differential (computed strictly from weeks before the
  game); OLS of the residual on the differential; t-test on the slope.

Both shapes score against the RESIDUAL of the walk-forward rating
prediction, so "the factor predicts strength" is already used up — only
information the rating missed can register. That is the matched-control
idea carried into the regression shape.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "data", "factor_registry.json")


class FactorError(RuntimeError):
    pass


class ExcludedFactorError(FactorError):
    """Raised when code tries to use a Tier-3 excluded variable."""


# --------------------------------------------------------------------------- #
# Registry — single source of truth
# --------------------------------------------------------------------------- #
def load_registry(path: Optional[str] = None) -> List[Dict]:
    p = path or REGISTRY_PATH
    if not os.path.exists(p):
        raise FactorError(f"{p} missing — the registry must exist; it is the "
                          "single source of truth for factor status")
    with open(p) as fh:
        return json.load(fh)


def save_registry(entries: List[Dict], path: Optional[str] = None) -> None:
    p = path or REGISTRY_PATH
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, p)


def registry_entry(entries: List[Dict], factor_id: str) -> Dict:
    for e in entries:
        if e.get("factor_id") == factor_id:
            return e
    raise FactorError(f"factor {factor_id!r} not in registry — add it (as "
                      "untested) before testing it")


def assert_not_excluded(entries: List[Dict], factor_id: str) -> None:
    e = registry_entry(entries, factor_id)
    if e.get("status") == "excluded":
        raise ExcludedFactorError(
            f"{factor_id} is Tier-3 EXCLUDED ({e.get('description', '')}); "
            "re-arguing an exclusion is a registry decision, not a code path")


def gated_active_factors(entries: List[Dict]) -> List[str]:
    return [e["factor_id"] for e in entries if e.get("status") == "gated_active"]


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: Dict[str, float], q: float = 0.05
                       ) -> Dict[str, Dict]:
    """Returns {name: {'p':, 'q_crit':, 'pass': bool}} under BH FDR control."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    max_pass_rank = 0
    for rank, (name, p) in enumerate(items, start=1):
        if p <= q * rank / m:
            max_pass_rank = rank
    for rank, (name, p) in enumerate(items, start=1):
        out[name] = {"p": float(p), "q_crit": q * rank / m,
                     "pass": rank <= max_pass_rank}
    return out


# --------------------------------------------------------------------------- #
# Event factors — matched control
# --------------------------------------------------------------------------- #
@dataclass
class EventResult:
    factor_id: str
    n_treated: int
    n_control: int
    effect_pts: float          # mean signed residual, treated minus control
    treated_mean: float
    control_mean: float
    p_value: float
    note: str = ""


def evaluate_event_factor(factor_id: str, events: pd.DataFrame,
                          preds: pd.DataFrame, min_treated: int = 40
                          ) -> EventResult:
    """`events`: columns (id, side) — side in {'home','away'}, one row per
    treated team-game. `preds`: walk-forward rows with id, season, week,
    pred_b, margin. Outcome is the signed residual for the treated side."""
    df = preds.copy()
    df["resid"] = df["margin"] - df["pred_b"]
    ev = events.merge(df, on="id", how="inner")
    if len(ev) < min_treated:
        return EventResult(factor_id, len(ev), 0, float("nan"), float("nan"),
                           float("nan"), float("nan"),
                           note=f"n-gate FAIL: {len(ev)} treated < {min_treated}")
    sign = np.where(ev["side"] == "home", 1.0, -1.0)
    ev["y"] = ev["resid"] * sign            # negative = treated side underperformed
    ev["pred_side"] = ev["pred_b"] * sign   # treated side's expected margin

    treated_ids = set(ev["id"])
    ctl = df[~df["id"].isin(treated_ids)].copy()
    # control outcome oriented the same way as its matched treated row's side
    matches = []
    used = set()
    for t in ev.itertuples():
        pool = ctl[(ctl["season"] == t.season)
                   & ((ctl["week"] - t.week).abs() <= 1)
                   & (~ctl.index.isin(used))]
        if pool.empty:
            continue
        s = 1.0 if t.side == "home" else -1.0
        j = (pool["pred_b"] * s - t.pred_side).abs().idxmin()
        used.add(j)
        matches.append({"y_t": t.y, "y_c": float(
            (ctl.loc[j, "margin"] - ctl.loc[j, "pred_b"]) * s)})
    m = pd.DataFrame(matches)
    if len(m) < min_treated:
        return EventResult(factor_id, len(ev), len(m), float("nan"),
                           float("nan"), float("nan"), float("nan"),
                           note=f"n-gate FAIL after matching: {len(m)}")
    t_stat, p = stats.ttest_ind(m["y_t"], m["y_c"], equal_var=False)
    return EventResult(
        factor_id, n_treated=len(m), n_control=len(m),
        effect_pts=float(m["y_t"].mean() - m["y_c"].mean()),
        treated_mean=float(m["y_t"].mean()),
        control_mean=float(m["y_c"].mean()),
        p_value=float(p))


# --------------------------------------------------------------------------- #
# Continuous factors — residual regression
# --------------------------------------------------------------------------- #
@dataclass
class ContinuousResult:
    factor_id: str
    n: int
    slope: float               # residual points per unit of factor diff
    r: float
    p_value: float
    note: str = ""


def evaluate_continuous_factor(factor_id: str, diffs: pd.DataFrame,
                               preds: pd.DataFrame, min_n: int = 800
                               ) -> ContinuousResult:
    """`diffs`: columns (id, diff) — the WALK-FORWARD factor differential
    (home minus away, computed from strictly-earlier weeks)."""
    df = preds.merge(diffs.dropna(subset=["diff"]), on="id", how="inner")
    if len(df) < min_n:
        return ContinuousResult(factor_id, len(df), float("nan"),
                                float("nan"), float("nan"),
                                note=f"n-gate FAIL: {len(df)} < {min_n}")
    resid = (df["margin"] - df["pred_b"]).to_numpy()
    x = df["diff"].to_numpy(dtype=float)
    res = stats.linregress(x, resid)
    return ContinuousResult(factor_id, n=len(df), slope=float(res.slope),
                            r=float(res.rvalue), p_value=float(res.pvalue))
