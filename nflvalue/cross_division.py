"""The cross-division (FBS-vs-FCS) projection, on the ridge table it already has.

Why a separate module
---------------------
`ratings.py` and `engine.py` are code provenance for the selected early-season
evidence record (`data/evidence_manifest.json` hashes them); a function added
to either would un-certify that record without changing one published
number. So the cross-division projection lives here, reads the fitted table
and writes nothing back into it. Nothing in the two hashed modules changes.

What the projection is
----------------------
The ridge pools every non-FBS opponent into one pseudo-team, `__FCS__`, with
its own fitted offence/defence and conference term (down-weighted rows,
`fcs_weight`). The pseudo-team is vetoed as a *rating* on purpose -- a real
FBS rating must never be formed through FCS results -- but it can stand on
the far side of a matchup it was fit to represent. This module is the ONE
place that veto is stepped around, for the pseudo-team only:

    margin = hfa(home side) + (off[home] + def[away]) - (off[away] + def[home])

with the FCS side = `__FCS__`. How wrong that is out of sample is a
registered question, answered by `scripts/backtest_cross_division.py` and
recorded in `data/forecast_sd_basis.json` (basis
`cross_division_ridge_wk1_15_2021_2025`). Nothing here chooses an SD.

Which table
-----------
Weeks 0-3: the same early-season ridge (prior + current talent) that prices
FBS-vs-FBS games. Weeks 4+: the plain in-season ridge (`R.solve_cached`, the
backtest_b/c path) -- explicitly the ridge, never the Kalman, because the
pooled coefficient is a ridge coefficient and the basis was registered
against exactly this pair of tables. Changing either branch invalidates it.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import engine as EG
from . import ratings as R

#: The pooled pseudo-team the FCS side stands on. Re-exported so callers do
#: not reach into `ratings` for a label that is this module's concern.
FCS_TEAM = R.FCS_TEAM


def solve_table(games: pd.DataFrame, season: int, week: int,
                params: Optional[R.RatingParams] = None) -> R.RatingsTable:
    """The ridge table an FBS-vs-FCS game is projected from, in ANY week."""
    params = params or R.load_params()
    if EG.is_early_season(week):
        return EG.solve_early_season_ridge_table(games, season, week, params)
    return R.solve_cached(games, season, week, params)


def predict_margin(tbl: R.RatingsTable, fbs_team: str, *, fbs_is_home: bool,
                   neutral: bool = False) -> float:
    """Home-minus-away expected margin, the FCS side = the pooled pseudo-team.

    Raises `DisconnectedTeamError` on an unratable FBS side and
    `RatingsError` on a table with no fitted FCS coefficient (no FBS-vs-FCS
    game in the fit window), so a caller cannot get a number for a matchup
    the table never saw the shape of.
    """
    tbl.require_usable(fbs_team)
    if FCS_TEAM not in tbl.off or FCS_TEAM not in tbl.def_:
        raise R.RatingsError(
            f"no pooled FCS coefficient in the table at season={tbl.season} "
            f"week={tbl.week}: no FBS-vs-FCS game was in the fit window, so "
            "there is nothing to project against")
    home, away = ((fbs_team, FCS_TEAM) if fbs_is_home
                  else (FCS_TEAM, fbs_team))
    return round(tbl.hfa(home, neutral)
                 + (tbl.off.get(home, 0.0) + tbl.def_.get(away, 0.0))
                 - (tbl.off.get(away, 0.0) + tbl.def_.get(home, 0.0)),
                 R.ROUND)


def rating_parameter_sd(tbl: R.RatingsTable, fbs_team: str) -> float:
    """The summed rating band of the two sides -- the technical quantity the
    contract labels `rating_parameter_sd`, never the forecast's spread."""
    return float(tbl.net_band[fbs_team]) + float(tbl.net_band[FCS_TEAM])
