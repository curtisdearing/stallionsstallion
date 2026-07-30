"""CFB overtime module — 2021+ rules, rates FIT from cached drives, never guessed.

Workstream C (2026-07-30). The NFL parent's sudden-death logic cannot be
ported: college overtime is a different game. Rules implemented here (NCAA,
in force since 2021):

  * OT1: each team gets one possession from the opponent's 25-yard line.
    After a touchdown the team may kick the XP or go for two (in practice
    mostly XP — modelled from the OT1 TD points actually observed).
  * OT2: one possession each, but after a touchdown a TWO-POINT attempt is
    MANDATORY — a 7 is impossible from OT2 onward.
  * OT3 and beyond: no more possessions. Teams alternate single two-point
    conversion attempts from the 3-yard line, one each per round, until one
    round produces a difference.

Every rate is fit from the drives cache via `fit_ot_params`, restricted to
seasons >= 2021 because 2019-2020 played under the OLD rules (2-pt duels only
from the 5th OT) and pooling regimes would contaminate the fit. Sample sizes
are small — OT is rare — so they are carried on the params object and must be
REPORTED next to any number this module influences, not hidden.

Matchup conditioning: OT possessions are too rare to fit outcome rates per
rating-differential bin. Instead the caller may pass per-team TD-rate tilts
(measured on the regulation rate model, e.g. p_td(e)/p_td(0)); the base rates
here stay league-level and fitted. Symmetric by default. There is no
home-field term in OT: possessions start at a fixed spot and the sample is
far too small to measure a crowd effect, so none is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

#: OT possessions start here under the 2021+ rules; anything else in the
#: drives table (punts after penalties, data noise) is not a standard OT
#: possession and is excluded from the fit.
OT_START_YTG = 25
RULES_ERA_FIRST_SEASON = 2021
#: Rounds 3+ are 2-pt duels; a tie can in principle persist indefinitely.
#: 40 rounds of identical outcomes has probability < 1e-11 at any plausible
#: conversion rate; beyond it the sim resolves by fair coin (2-pt margin) so
#: the returned margins are never zero. Purely a numerical guard, not a rule.
MAX_ROUNDS = 40


class OTFitError(RuntimeError):
    """The drives cache does not contain enough 2021+ OT data to fit."""


@dataclass(frozen=True)
class OTParams:
    p_td: float                 # P(possession from the 25 ends in a TD)
    p_fg: float                 # P(... ends in a made FG)
    td_pts: Dict[int, float]    # OT1 TD points distribution over {6, 7, 8}
    p_two: float                # 2-pt conversion success (mandatory tries + duels)
    n_possessions: int = 0      # fit sample sizes — report these, always
    n_td_ot1: int = 0
    n_two: int = 0

    def as_dict(self) -> Dict:
        return {
            "p_td": round(self.p_td, 4), "p_fg": round(self.p_fg, 4),
            "td_pts": {int(k): round(v, 4) for k, v in self.td_pts.items()},
            "p_two": round(self.p_two, 4),
            "n_possessions": self.n_possessions,
            "n_td_ot1": self.n_td_ot1, "n_two": self.n_two,
        }


def fit_ot_params(drives: pd.DataFrame,
                  min_possessions: int = 100) -> OTParams:
    """Fit OT rates from cached drives (2021+ rules era only). Fail loud."""
    d = drives.copy()
    d["off_delta"] = d["end_offense_score"] - d["start_offense_score"]
    ot = d[(d["start_period"] >= 5)
           & (d["season"] >= RULES_ERA_FIRST_SEASON)
           & (d["start_yards_to_goal"] == OT_START_YTG)]
    if len(ot) < min_possessions:
        raise OTFitError(
            f"only {len(ot)} standard OT possessions in the 2021+ era — "
            f"below the {min_possessions} floor; refusing to fit on noise")

    delta = ot["off_delta"]
    is_td = delta.isin([6, 7, 8])
    is_fg = delta.eq(3)
    p_td = float(is_td.mean())
    p_fg = float(is_fg.mean())

    # OT1 TD points: what teams actually did with the untied first possession.
    ot1_td = ot[(ot["start_period"] == 5) & is_td]
    counts = ot1_td["off_delta"].value_counts()
    n_td_ot1 = int(counts.sum())
    if n_td_ot1 == 0:
        raise OTFitError("no OT1 touchdowns in the fit window")
    td_pts = {int(k): float(v) / n_td_ot1 for k, v in counts.items()}

    # 2-pt success: OT2+ touchdowns are FORCED to attempt two, so the 8-vs-6
    # split among them is a clean conversion-rate estimate. n is small (OT2+
    # is rare); it is recorded and must be reported, not laundered.
    ot2_td = ot[(ot["start_period"] >= 6) & delta.isin([6, 8])]
    n_two = int(len(ot2_td))
    if n_two < 5:
        raise OTFitError(
            f"only {n_two} mandatory 2-pt attempts in the 2021+ era — too few "
            "to estimate a conversion rate at all; extend the cache first")
    p_two = float(ot2_td["off_delta"].eq(8).mean())

    return OTParams(p_td=p_td, p_fg=p_fg, td_pts=td_pts, p_two=p_two,
                    n_possessions=int(len(ot)), n_td_ot1=n_td_ot1, n_two=n_two)


def _possession_points(rng, n: int, params: OTParams, tilt: float,
                       forced_two: bool) -> np.ndarray:
    """Points from one OT possession for n simulated games."""
    p_td = float(np.clip(params.p_td * tilt, 0.02, 0.90))
    p_fg = min(params.p_fg, 1.0 - p_td)
    u = rng.random(n)
    td = u < p_td
    fg = (u >= p_td) & (u < p_td + p_fg)
    pts = np.zeros(n)
    pts[fg] = 3.0
    if forced_two:
        pts[td] = 6.0 + 2.0 * (rng.random(n)[td] < params.p_two)
    else:
        keys = sorted(params.td_pts)
        probs = np.array([params.td_pts[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        idx = np.searchsorted(np.cumsum(probs), rng.random(n), side="right")
        pts[td] = np.array(keys, dtype=float)[np.clip(idx, 0, len(keys) - 1)][td]
    return pts


def simulate_ot(rng, n: int, params: OTParams,
                tilt_home: float = 1.0, tilt_away: float = 1.0
                ) -> "tuple[np.ndarray, np.ndarray]":
    """Simulate n college overtimes.

    Returns (margins, points): home-minus-away OT margins (never 0 — college
    games cannot tie) and TOTAL points scored in overtime by both teams
    (duel conversions count 2 each, as they do in real box scores).

    Rounds 1-2 are possession pairs (round 2 with the mandatory 2-pt after a
    TD); rounds 3+ are alternating 2-pt duels. Ends the moment a round
    produces a difference — exactly the rule.
    """
    margins = np.zeros(n)
    points = np.zeros(n)
    active = np.ones(n, dtype=bool)
    for rnd in (1, 2):
        if not active.any():
            break
        forced = rnd >= 2
        hp = _possession_points(rng, n, params, tilt_home, forced)
        ap = _possession_points(rng, n, params, tilt_away, forced)
        margins = np.where(active, margins + hp - ap, margins)
        points = np.where(active, points + hp + ap, points)
        active = active & (margins == 0)
    rnd = 3
    while active.any() and rnd <= MAX_ROUNDS:
        h2 = 2.0 * (rng.random(n) < params.p_two)
        a2 = 2.0 * (rng.random(n) < params.p_two)
        margins = np.where(active, margins + h2 - a2, margins)
        points = np.where(active, points + h2 + a2, points)
        active = active & (margins == 0)
        rnd += 1
    if active.any():
        coin = np.where(rng.random(n) < 0.5, 2.0, -2.0)
        margins = np.where(active, coin, margins)
        points = np.where(active, points + 2.0, points)
    return margins, points
