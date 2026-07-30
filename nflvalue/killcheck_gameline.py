"""Workstream K machinery — GAME-LINE kill-check bars, derived, not inherited.

`killcheck.py` (kept as reference) is fablesfable's PLAYER-PROP tool: a leans
table keyed on player_id, DEFAULT_MIN_SAMPLE=150 and POSITIVE_RATE_BAR=0.52
chosen for prop volume. _SETUP.md and FACTOR_REVALIDATION_SPEC §4 both flag
that Workstream K must NOT silently inherit those numbers. This module
derives game-line bars from this project's OWN measured quantities:

  * **Minimum detectable effect** — not guessed. One point of spread CLV is
    worth ~phi(0)/sigma of cover probability, with sigma the MEASURED
    out-of-sample margin residual SD (16.74, gate 3). The bar is the CLV
    that recovers HALF the spread vig (breakeven 52.38% -> 2.38% edge):
        min_effect_pts = (vig_edge / 2) / (phi(0) / sigma)  ~= 0.50 pts.
  * **n-gate** — the sample needed for a one-sided alpha=0.05 test to detect
    that effect with 80% power at the MEASURED per-game CLV dispersion
    (sd ~2.3-2.6 in the 2021-2024 close-free logs):
        n_min = ((z_alpha + z_power) * sd / min_effect)^2   ~= 130-160.
  * **Verdict** — per slice, never pooled (BUILD_SPEC §15: pooling hides
    where the edge isn't): GO only if n >= n_min AND the one-sided t-test
    on mean CLV > 0 clears alpha. Anything under the n-gate is
    INSUFFICIENT_N, not NO-GO — an unpowered test proves nothing either way.

**Process discipline this module cannot enforce but its callers must:** the
real gate-7 evaluation is ONE pre-registered run on live-season data, not
re-run until it returns GO. Running these functions on backtest-era logs is
research calibration of the machinery, and every printed verdict from such a
run must say so. Nothing here authorizes staking; a GO verdict from this
code is an input to Curtis's decision at gate 7, not the decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

#: Measured out-of-sample margin residual SD (gate 3, pooled 2021-2023).
MEASURED_MARGIN_SD = 16.74
#: Standard -110 two-way spread pricing: breakeven 52.38%.
VIG_EDGE = 0.0238
ALPHA = 0.05
POWER = 0.80


class KillCheckError(RuntimeError):
    pass


def prob_per_point(margin_sd: float = MEASURED_MARGIN_SD) -> float:
    """Cover-probability value of one point of CLV: phi(0)/sigma."""
    return float(stats.norm.pdf(0.0) / margin_sd)


def min_effect_pts(margin_sd: float = MEASURED_MARGIN_SD,
                   vig_edge: float = VIG_EDGE) -> float:
    """The smallest CLV worth proving: half the vig, in points."""
    return float((vig_edge / 2.0) / prob_per_point(margin_sd))


def n_gate(clv_sd: float, effect_pts: Optional[float] = None,
           alpha: float = ALPHA, power: float = POWER) -> int:
    """Sample size for a one-sided test to detect `effect_pts` at the
    measured CLV dispersion. Standard normal-approximation power formula."""
    effect = effect_pts if effect_pts is not None else min_effect_pts()
    if effect <= 0 or clv_sd <= 0:
        raise KillCheckError("degenerate inputs to the n-gate derivation")
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return int(math.ceil((z * clv_sd / effect) ** 2))


@dataclass
class SliceVerdict:
    slice_tag: str
    n: int
    n_required: int
    mean_clv: float
    sd_clv: float
    ci_low: float
    ci_high: float
    p_value: float
    verdict: str          # GO | NO-GO | INSUFFICIENT_N
    note: str = ""

    def line(self) -> str:
        return (f"{self.slice_tag:8s} n={self.n:4d}/{self.n_required} "
                f"mean {self.mean_clv:+.3f} pts "
                f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
                f"p={self.p_value:.4f} -> {self.verdict}"
                + (f" ({self.note})" if self.note else ""))


def slice_verdict(log: pd.DataFrame, slice_tag: str,
                  alpha: float = ALPHA,
                  dedup_per_game: bool = True) -> SliceVerdict:
    """One slice's verdict from a close-free CLV log.

    dedup_per_game=True averages books within a game first — multiple books
    on one game are not independent observations, and counting them as such
    would flatter the n-gate.
    """
    d = log[log["slice"] == slice_tag]
    if dedup_per_game and len(d):
        d = d.groupby("game_id", as_index=False).agg(
            clv_pts=("clv_pts", "mean"))
    n = len(d)
    if n < 3:
        return SliceVerdict(slice_tag, n, n_gate(2.4), float("nan"),
                            float("nan"), float("nan"), float("nan"),
                            float("nan"), "INSUFFICIENT_N",
                            note="too few observations to even estimate sd")
    mean = float(d["clv_pts"].mean())
    sd = float(d["clv_pts"].std())
    req = n_gate(sd)
    se = sd / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    t_stat = mean / se
    p = float(1 - stats.t.cdf(t_stat, n - 1))       # one-sided, H1: mean > 0
    if n < req:
        verdict = "INSUFFICIENT_N"
        note = "unpowered — proves nothing either way"
    elif p < alpha and mean > 0:
        verdict = "GO"
        note = "input to the gate-7 decision, not the decision"
    else:
        verdict = "NO-GO"
        note = "recorded plainly; the expected outcome for most slices"
    return SliceVerdict(slice_tag, n, req, mean, sd,
                        mean - t_crit * se, mean + t_crit * se, p,
                        verdict, note)


def run_killcheck(log: pd.DataFrame, research_mode: bool = True
                  ) -> List[SliceVerdict]:
    """Per-slice verdicts, never pooled. `research_mode` marks a run on
    backtest-era data (machinery calibration); the single real gate-7 run
    sets it False and must be pre-registered in the worklog first."""
    if log.empty:
        raise KillCheckError("empty CLV log — nothing to check")
    out = [slice_verdict(log, s) for s in sorted(log["slice"].unique())]
    header = ("RESEARCH-MODE (backtest-era data; NOT gate 7)"
              if research_mode else
              "GATE-7 RUN — must have been pre-registered in _worklog.md")
    for v in out:
        v.note = (v.note + "; " + header).strip("; ")
    return out
