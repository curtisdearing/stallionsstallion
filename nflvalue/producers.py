"""The production forecast routes. Both end in one artifact builder.

Weeks 0-3 and weeks 4+ are priced by genuinely different models -- the
early-season route is ridge fitted around the preseason prior (the only path
carrying a prior-year component), the in-season route is the Kalman filter
driving the drive-level Monte Carlo. That difference is real and registered.

What is NOT allowed to differ is what they publish. Both routes assemble a
:class:`~nflvalue.forecast_contract.PredictiveMargin`, hand it to
:func:`~nflvalue.forecast_contract.build_artifact`, and get back rows in the
one public contract. Nothing here writes a column; the contract module is the
only thing that does. That is what stops the two schemas this module was
written to collapse from re-diverging: there is no second place to drift to.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Set, Tuple

import pandas as pd

from . import engine as EG
from . import forecast_contract as FC
from . import ingest
from . import ratings as R

#: Engine name (as `engine.production_margin_engine` returns it) -> the public
#: model family that names it in the artifact. The engine name is an internal
#: dispatch key; the family is what a reader sees, and it says what kind of
#: object produced the number.
ENGINE_TO_FAMILY: Dict[str, str] = {
    "ridge": "ridge_preseason_prior",
    "kalman": "kalman_montecarlo",
}


class ProducerError(RuntimeError):
    """A production route could not honestly produce what it was asked for."""


class UncertaintyUnvalidatedError(ProducerError):
    """A margin model exists for this week, but no validated out-of-sample SD
    backs a forecast for it -- the specific, recoverable cause that routes a
    week to `Status.UNCERTAINTY_UNVALIDATED` rather than aborting the run."""


def production_family(week: int) -> str:
    """The public model family that prices a published week."""
    engine = EG.production_margin_engine(week)
    try:
        return ENGINE_TO_FAMILY[engine]
    except KeyError:
        raise ProducerError(
            f"engine {engine!r} has no public model family. A forecast whose "
            "producer cannot be named must not be published") from None


def fbs_teams(season: int) -> Set[str]:
    """Teams the season's own `/teams` pull marks FBS.

    Used to classify which games are refusable, never to substitute a
    pseudo-team for a real opponent. This is the stricter of the two
    classifications in the repo -- the schedule's `home_fcs`/`away_fcs` flags
    say "known FCS", which is not the same as "known FBS" for a programme new
    to the division -- and it is the one the published early-season artifact
    has always used.
    """
    teams = ingest.load("teams", [int(season)])
    if teams.empty or "is_fbs" not in teams.columns:
        raise ProducerError(
            f"no FBS classification cached for {season}; refusing to guess "
            "which games are inside the model's validated division")
    return {str(row.team) for row in teams.itertuples() if bool(row.is_fbs)}


def division_refusal(home: str, away: str, fbs: Set[str]) -> Optional[str]:
    """Why this matchup is outside the validated model, or None."""
    home_fbs, away_fbs = home in fbs, away in fbs
    if not home_fbs and not away_fbs:
        return FC.DEFAULT_REFUSALS["non_fbs"]
    if not (home_fbs and away_fbs):
        # The pooled FCS pseudo-team is excluded from connectivity ON PURPOSE.
        # Pricing this game by standing it in for the real opponent would
        # publish a number for a matchup the model has never been validated
        # on -- the same rule the backtest scores by.
        return FC.DEFAULT_REFUSALS["cross_division"]
    return None


def early_ridge_forecasts(tbl, schedule: pd.DataFrame, registry: FC.BasisRegistry,
                          week: int, *, fbs: Set[str]
                          ) -> Tuple[Dict[int, FC.Forecast], Dict[int, str]]:
    """Price an early-season week from the ridge table. Returns (forecasts, refusals).

    The forecast SD is NOT the rating band. It is the registered out-of-sample
    residual SD for the weeks actually being priced, and `p_home_win` is read
    off that same distribution -- which is the whole reason
    :meth:`PredictiveMargin.from_normal` is the only way in. The summed rating
    band still travels, as `rating_parameter_sd`, clearly labelled as the
    separate technical quantity it is.
    """
    family = "ridge_preseason_prior"
    try:
        sd = registry.margin_sd(family, week)
    except FC.ContractError as exc:
        raise UncertaintyUnvalidatedError(str(exc)) from exc
    if sd is None:
        raise UncertaintyUnvalidatedError(
            f"the registered basis for {family} week {week} carries no pooled "
            "forecast SD, but the ridge route has no per-game distribution to "
            "fall back on")
    basis_id = registry.basis_id(family, week)
    bands = tbl.to_frame().set_index("team")

    forecasts: Dict[int, FC.Forecast] = {}
    refusals: Dict[int, str] = {}
    for game in schedule.itertuples():
        game_id = int(game.id)
        home, away = str(game.home_team), str(game.away_team)
        reason = division_refusal(home, away, fbs)
        if reason:
            refusals[game_id] = reason
            continue
        try:
            margin = tbl.predict_margin(home, away, bool(game.neutral_site))
        except R.RatingsError as exc:
            refusals[game_id] = (f"NO RATING ({type(exc).__name__}): "
                                 "new-to-FBS or unseen team")
            continue
        # predict_margin has already refused both sides' vetoes, so a value
        # here means both teams are ratable and both bands exist.
        rating_sd = (float(bands.loc[home, "net_band"])
                     + float(bands.loc[away, "net_band"]))
        forecasts[game_id] = FC.Forecast(
            predictive=FC.PredictiveMargin.from_normal(margin, sd, basis_id),
            rating_parameter_sd=rating_sd)
    return forecasts, refusals


def build_early_week(season: int, week: int, *, games: pd.DataFrame,
                     schedule: pd.DataFrame, generated_at_utc: str,
                     data_dir: str, params=None,
                     registry: Optional[FC.BasisRegistry] = None,
                     snapshot_id: Optional[str] = None,
                     source_digest: Optional[str] = None,
                     config_hash: Optional[str] = None,
                     degraded: bool = False,
                     degraded_reason: Optional[str] = None,
                     table_out: Optional[dict] = None) -> pd.DataFrame:
    """The whole early-season production path, ending in contract rows.

    One assembly site, called by the weekly loop for weeks 0-3 and by the
    demo script. The demo script cannot reach a different code path, so it
    cannot produce a different schema.
    """
    if not EG.is_early_season(week):
        raise ProducerError(
            f"week {week} is not an early-season week; weeks 4+ are priced by "
            f"{ENGINE_TO_FAMILY['kalman']}, not the ridge route")
    registry = registry or FC.BasisRegistry.load(data_dir)
    params = params or R.load_params()
    table = EG.solve_production_table(games, season, week, params)
    if table_out is not None:
        # The insight layer needs the same table this priced the week from.
        # Handing it back beats re-solving: a second solve could differ from
        # the one that produced the numbers, and then the card would explain
        # the forecast with reasoning that did not make it.
        table_out["table"] = table
    provenance = FC.Provenance(
        model_family="ridge_preseason_prior",
        model_version=table.fingerprint(),
        generated_at_utc=generated_at_utc,
        snapshot_id=snapshot_id,
        source_digest=source_digest,
        config_hash=config_hash)
    if degraded:
        return FC.build_artifact(season, week, schedule, {}, {}, provenance,
                                 degraded=True, degraded_reason=degraded_reason)
    try:
        forecasts, refusals = early_ridge_forecasts(
            table, schedule, registry, week, fbs=fbs_teams(season))
    except UncertaintyUnvalidatedError as exc:
        # A margin model exists; its uncertainty does not. Publishing the
        # margin anyway, with no SD to size it, would be precision this run
        # cannot support -- so the whole week is honestly labelled instead of
        # crashing the run or emitting a number the run cannot stand behind.
        return FC.build_artifact(
            season, week, schedule, {}, {}, provenance,
            degraded=True,
            degraded_reason=f"UNCERTAINTY UNVALIDATED: {exc}",
            degraded_status=FC.Status.UNCERTAINTY_UNVALIDATED)
    return FC.build_artifact(season, week, schedule, forecasts, refusals,
                             provenance)
