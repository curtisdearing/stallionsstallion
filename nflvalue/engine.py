"""Margin-engine dispatch — ONE place decides which model prices games.

`config.json`'s `"margin_engine"` selects the engine for every in-season
consumer (weekly loop, gate-3 sim harness, explainer). The registered
default is `"kalman"` (Workstream G decision, 2026-07-30 — see
`kalman_engine`'s module docstring for the record). `"ridge"` remains fully
wired on purpose: it is the H-spec drift-comparison baseline, and flipping
the config key back is the entire rollback story.

Consumers call `solve_table(games, season, week, params)` exactly where they
used to call `ratings.solve_cached(...)`. The ridge `RatingParams` argument
is accepted (and ignored) under the kalman engine so call sites don't fork.

PUBLIC-FORECAST POLICY (added 2026-08-28). `solve_table` answers "which
engine did you ask for"; `solve_production_table` answers "which engine is
allowed to price a *published* week". They are deliberately separate: the
bakeoff and the backtests choose an engine on purpose and must keep getting
exactly what they asked for. See `solve_early_season_ridge_table` for the
one place the two diverge.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Iterable, Optional, Union

import pandas as pd

from . import ratings as R
from . import kalman_engine as K

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The registered default (bakeoff G, 2026-07-30). config.json may override.
DEFAULT_ENGINE = "kalman"

#: Weeks the public forecast treats as early season. CFBD labels week-0 games
#: as week 1, so the label space that matters here is 0..3 inclusive.
EARLY_SEASON_WEEKS = (0, 1, 2, 3)

#: The only engine allowed to price a published early-season week: ridge,
#: because it is the one that carries a preseason prior. The kalman filter
#: has no prior-year component at all — before it has seen games it is
#: pricing from nothing, which is exactly when the prior matters most.
EARLY_SEASON_ENGINE = "ridge"

#: Row-level status strings. Non-actionable by construction: they describe
#: what the number is, never what to do about it.
EARLY_SEASON_STATUS = ("early-season paper-only research forecast: the "
                       "preseason prior still dominates; not actionable")
REGULAR_STATUS = "paper-only research forecast: not actionable"

Table = Union[R.RatingsTable, K.KalmanTable]


class EarlySeasonPriorError(R.RatingsError):
    """The early-season prior could not be built from real prior-year play.

    A `RatingsError` on purpose: callers that already skip unratable weeks
    (the drive-labelling loops) skip this the same way, and callers that
    publish a week let it fail the run. What must never happen is silently
    falling back to `prev_net=None`, which drops prior-year team strength
    out of the prior without saying so.
    """


def margin_engine() -> str:
    """Engine name from config.json (`margin_engine`), else the default."""
    for name in ("config.local.json", "config.json"):
        path = os.path.join(_ROOT, name)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    cfg = json.load(fh)
            except Exception:  # noqa: BLE001 — a broken config must not
                continue       # silently flip the engine; fall through.
            if "margin_engine" in cfg:
                return str(cfg["margin_engine"])
    return DEFAULT_ENGINE


def solve_table(games: pd.DataFrame, season: int, week: int,
                params: Optional[R.RatingParams] = None,
                engine: Optional[str] = None) -> Table:
    """Weekly table from the configured engine. Fails loud on unknown names."""
    eng = engine or margin_engine()
    if eng == "kalman":
        return K.solve_kalman_cached(games, season, week)
    if eng == "ridge":
        return R.solve_cached(games, season, week, params or R.load_params())
    raise ValueError(
        f"unknown margin_engine {eng!r} — expected 'kalman' or 'ridge'. "
        "Refusing to guess which model prices games.")


# --------------------------------------------------------------------------- #
# Public-forecast policy
# --------------------------------------------------------------------------- #
def is_early_season(week: int) -> bool:
    return int(week) in EARLY_SEASON_WEEKS


def production_margin_engine(week: int) -> str:
    """The engine a PUBLISHED week is priced by. Config wins from week 4 on."""
    return EARLY_SEASON_ENGINE if is_early_season(week) else margin_engine()


def forecast_status(week: int) -> str:
    """Row-level, non-actionable label for what a published number is."""
    return EARLY_SEASON_STATUS if is_early_season(week) else REGULAR_STATUS


def previous_season_net(games: pd.DataFrame, season: int,
                        params: Optional[R.RatingParams] = None
                        ) -> Dict[str, float]:
    """`prev_net` for `season`'s preseason prior: last season's final ratings.

    Fails loud rather than returning an empty map: an empty `prev_net` is
    indistinguishable, inside `preseason_prior`, from "there is no prior-year
    component" — and that silence is the bug this whole path exists to stop.
    """
    prev_season = int(season) - 1
    prev = games[games["season"] == prev_season]
    if prev.empty:
        raise EarlySeasonPriorError(
            f"no {prev_season} games in the frame, so the {season} preseason "
            "prior would carry no prior-year team strength — refusing to "
            "publish an early-season forecast without it")
    last_week = prev["week"].max()
    if pd.isna(last_week):
        raise EarlySeasonPriorError(
            f"{prev_season} games carry no week labels — cannot locate that "
            "season's final table")
    # max(week) + 1 is "after everything that season played": solve() filters
    # strictly BEFORE the as-of week, so this is the season's final table.
    final = R.solve_cached(games, prev_season, int(last_week) + 1,
                           params or R.load_params())
    net = {str(row.team): float(row.net)
           for row in final.to_frame().itertuples()
           if not bool(row.veto) and str(row.team) != R.FCS_TEAM}
    if not net:
        raise EarlySeasonPriorError(
            f"every team in the final {prev_season} table is vetoed — there "
            f"is no prior-year strength to carry into {season}")
    return net


def solve_early_season_ridge_table(games: pd.DataFrame, season: int, week: int,
                                   params: Optional[R.RatingParams] = None
                                   ) -> R.RatingsTable:
    """Weeks 0-3: ridge fitted around the full preseason prior.

    `prev_net` is handed to `R.solve`, which then builds the prior through its
    own production branch (`preseason_prior`) — current-season talent and
    returning production enter if and only if their caches exist, and the
    fitted HFA handling is untouched. `prior=` is NOT used: that hook exists
    so tests can solve without a parquet cache, and routing production through
    it would mean the published prior is assembled somewhere other than the
    one function that documents how a prior is assembled.
    """
    params = params or R.load_params()
    prev_net = previous_season_net(games, season, params)
    return R.solve(games, season, week, params, prev_net=prev_net)


def solve_production_table(games: pd.DataFrame, season: int, week: int,
                           params: Optional[R.RatingParams] = None) -> Table:
    """The table a PUBLISHED week is priced from. Weeks 4+ are unchanged."""
    if is_early_season(week):
        return solve_early_season_ridge_table(games, season, week, params)
    return solve_table(games, season, week, params)


def id_coverage_problem(row_ids: Iterable, schedule_ids: Iterable
                        ) -> Optional[str]:
    """Describe how a writer's rows deviate from its schedule, else None.

    The artifact contract is exact set equality with no duplicates: a public
    row set that is not its schedule is either hiding a game or inventing one,
    and both are worse than a failed run.
    """
    rows = [int(i) for i in row_ids]
    schedule = {int(i) for i in schedule_ids}
    counts = Counter(rows)
    duplicated = sorted(i for i, n in counts.items() if n > 1)
    if duplicated:
        return f"duplicate row ids {duplicated[:10]}"
    seen = set(counts)
    missing = sorted(schedule - seen)
    extra = sorted(seen - schedule)
    if missing or extra:
        return (f"row ids are not the schedule exactly: "
                f"missing {missing[:10]}, extra {extra[:10]}")
    return None
