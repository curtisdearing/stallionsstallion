"""The shipped margin engine — Workstream G's bakeoff winner, wired in.

WHAT SHIPPED AND WHY (registered record, do not re-litigate here):
  * Workstream G ran fully pre-registered on 2026-07-30. The
    diagonal-covariance Kalman state-space candidate (`bakeoff.KalmanModel`)
    beat the incumbent ridge decisively on the selection window
    (254.71 vs 281.64 pooled MSE, 2021-2024 wk>=4, n=2,377) and CONFIRMED on
    the single virgin-2025 holdout pass (234.87 vs 291.13, n=617). Per the
    registered decision rule it ships as the margin engine.
  * Hyperparameters below are the FROZEN burn-in grid fit (2019-2020 only,
    recorded in _worklog.md 2026-07-30). Changing them mid-season without a
    new registered run is a protocol violation by construction.
  * The state model is the DIAGONAL approximation, exactly as measured: the
    full coupled-covariance filter remains FUTURE, UNREGISTERED work — do not
    "upgrade" the filter here, that would ship an unmeasured candidate.

RECORDED DEVIATIONS from the measured bakeoff candidate (both fail-closed;
neither changes any number the engine does emit):
  1. Veto semantics. The bakeoff harness priced every FBS-vs-FBS eval game,
     including teams the filter had never observed (they got the raw prior,
     mu=0). The shipped engine REFUSES those games instead
     (`DisconnectedTeamError`), matching the product-wide "veto, don't
     guess" rule ridge already enforces. Refusing to price is strictly
     conservative; every margin actually emitted is the measured model's.
  2. Within-week information. The bakeoff's online harness updated on
     earlier same-week games (id order) before predicting later ones. The
     shipped weekly table is frozen at week-open: it sees games strictly
     BEFORE (season, week), like `ratings.solve`. Less information, never
     more — no leakage is possible by construction, and predictions match
     what would have been published at week-open.

SCOPE: bakeoff evidence covers in-season weeks >= 4. Preseason / early-week
projections (`scripts/predict_2026_wk01.py`) deliberately stay on the
ridge + `preseason_prior` path, which is the only machinery that ingests
talent / returning production. Revisit only with a measured wk1-3 comparison.

Adapter notes: `KalmanTable` exposes the same consumed surface as
`ratings.RatingsTable` (predict_margin, veto/require_usable, hfa, off/def_,
net_band, connectivity, to_frame, fingerprint) so cfb_sim / explain /
weekly_loop run unchanged. The filter only knows a NET strength, so the
off/def split is symmetric (off = +net/2, def_ = -net/2) — same identity
side_edges needs; the conference blocks are empty (the state-space has no
conference term; `explain.decompose_margin` renders the kalman shape).
`net_sd`/`net_band` are the filter's own posterior sd — unlike ridge there
is no prior-misfit widening because the posterior variance already carries
time-drift and observation-noise uncertainty.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import ratings as R
from .bakeoff import KalmanModel, KalmanParams

ROUND = 6

#: Frozen 2019-2020 burn-in grid fit — recorded in _worklog.md 2026-07-30
#: ("kalman frozen: obs_sd 14 / week_sd 0.8 / season_sd 6 / hfa 2.4",
#: prior_sd held at 10.0 by the registered grid). These shipped the bakeoff
#: selection AND holdout numbers; they are not tunables.
FROZEN = KalmanParams(obs_sd=14.0, week_sd=0.8, season_sd=6.0,
                      prior_sd=10.0, hfa=2.4)


class KalmanTable:
    """RatingsTable-compatible view of the filtered state at (season, week)."""

    engine = "kalman"

    def __init__(self, *, season: int, week: int, params: KalmanParams,
                 mu_state: Dict[str, float], var_state: Dict[str, float],
                 n_games: Dict[str, int], team_conf: Dict[str, str],
                 n_games_fit: int):
        self.season = int(season)
        self.week = int(week)
        self.params = params
        self.mu = 0.0                      # league mean of net (states are deviations)
        self.hfa_league = float(params.hfa)
        self.hfa_team: Dict[str, float] = {}   # no per-team HFA in the registered model
        self.conf_off: Dict[str, float] = {}   # no conference block in the state-space
        self.conf_def: Dict[str, float] = {}
        self.conf_games: Dict[str, int] = {}
        self.team_conf = dict(team_conf)
        self.net = {t: round(float(v), ROUND) for t, v in mu_state.items()}
        self.net_sd = {t: round(float(np.sqrt(var_state[t])), ROUND)
                       for t in var_state}
        self.net_band = dict(self.net_sd)  # posterior sd IS the band (see module doc)
        # Symmetric split so predict_margin/side_edges identities hold with
        # only a net strength: off - def_ view of the same number.
        self.off = {t: round(v / 2.0, ROUND) for t, v in self.net.items()}
        self.def_ = {t: round(-v / 2.0, ROUND) for t, v in self.net.items()}
        self.off_dev = dict(self.off)      # dev == effective: no conference block
        self.def_dev = dict(self.def_)
        self.prior: Dict[str, float] = {}
        self.prior_misfit_sd = 0.0
        self.connectivity = R.Connectivity(
            hops={t: None for t in self.net},
            n_games=dict(n_games),
            component_size={t: len(self.net) for t in self.net},
            score={t: (1.0 if n_games.get(t, 0) > 0 else 0.0)
                   for t in self.net},
            reference=[])
        self.veto = {t: (t == R.FCS_TEAM or n_games.get(t, 0) == 0)
                     for t in self.net}
        self.n_games_fit = int(n_games_fit)
        self.condition = float("nan")      # no linear system, no condition number
        self.residual_sd = float(params.obs_sd)

    # ---- accessors (same contracts as RatingsTable) ----------------------- #
    def def_rating(self, team: str) -> float:
        return round(-self.def_.get(team, 0.0), ROUND)

    def hfa(self, team: str, neutral: bool = False) -> float:
        if neutral:
            return 0.0
        return round(self.hfa_league + self.hfa_team.get(team, 0.0), ROUND)

    def require_usable(self, team: str) -> None:
        if self.veto.get(team, True):
            raise R.DisconnectedTeamError(
                f"{team!r} is not ratable at season={self.season} "
                f"week={self.week}: the kalman filter has observed "
                f"{self.connectivity.n_games.get(team, 0)} games for it. "
                "Refusing to emit a rating rather than emit the raw prior.")

    def predict_margin(self, home: str, away: str, neutral: bool = False) -> float:
        self.require_usable(home)
        self.require_usable(away)
        return round(self.hfa(home, neutral)
                     + (self.off.get(home, 0.0) + self.def_.get(away, 0.0))
                     - (self.off.get(away, 0.0) + self.def_.get(home, 0.0)), ROUND)

    def to_frame(self) -> pd.DataFrame:
        """Same columns as RatingsTable.to_frame — parquet schema stability."""
        teams = sorted(self.net)
        return pd.DataFrame({
            "season": self.season, "week": self.week, "team": teams,
            "off": [self.off[t] for t in teams],
            "def": [self.def_rating(t) for t in teams],
            "net": [self.net[t] for t in teams],
            "net_sd": [self.net_sd[t] for t in teams],
            "net_band": [self.net_band[t] for t in teams],
            "hfa": [round(self.hfa_league, ROUND) for _ in teams],
            "prior": [0.0 for _ in teams],
            "connectivity": [self.connectivity.score.get(t, 0.0) for t in teams],
            "hops_to_reference": [None for _ in teams],
            "fbs_games": [self.connectivity.n_games.get(t, 0) for t in teams],
            "veto": [bool(self.veto.get(t, True)) for t in teams],
        }).sort_values("net", ascending=False).reset_index(drop=True)

    def fingerprint(self) -> str:
        payload = {
            "engine": "kalman",
            "season": int(self.season), "week": int(self.week),
            "params": {k: float(v) for k, v in sorted(vars(self.params).items())},
            "hfa_league": round(self.hfa_league, ROUND),
            "net": {k: round(v, ROUND) for k, v in sorted(self.net.items())},
            "net_sd": {k: round(v, ROUND) for k, v in sorted(self.net_sd.items())},
            "n_games": {k: int(v) for k, v in
                        sorted(self.connectivity.n_games.items())},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def solve_kalman(games: pd.DataFrame, season: int, week: int,
                 params: Optional[KalmanParams] = None) -> KalmanTable:
    """Run the frozen filter over every game strictly BEFORE (season, week).

    Walk-forward by construction (`< week`, never `<= week`), identical to
    `ratings.solve`'s discipline; `tests/test_kalman_engine.py` poisons the
    future and asserts it. State evolution is `bakeoff.KalmanModel` itself —
    the measured candidate, not a reimplementation.
    """
    params = params or FROZEN
    hist = games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))]
    if hist.empty:
        raise R.RatingsError(f"no games before season={season} week={week}")

    model = KalmanModel(params)
    n_games: Dict[str, int] = {}
    ordered = hist.sort_values(["season", "week", "id"], kind="mergesort")
    for g in ordered.itertuples():
        model.update(g)
        for name, fbs in ((g.home_team, g.home_fbs), (g.away_team, g.away_fbs)):
            t = name if bool(fbs) else R.FCS_TEAM
            n_games[t] = n_games.get(t, 0) + 1

    # bakeoff.KalmanModel pools non-FBS sides under its own literal
    # "FCS_TEAM"; the product convention (ratings, consumers, frames) is
    # R.FCS_TEAM ("__FCS__"). Re-key so veto/to_frame speak one language.
    for d in (model.mu, model.var, model.last_week):
        if "FCS_TEAM" in d:
            d[R.FCS_TEAM] = d.pop("FCS_TEAM")

    # Age every state to the as-of week so the bands reflect elapsed time
    # (a team on bye still drifts). Mirrors KalmanModel._age's charging.
    for t in list(model.mu):
        model._age(t, int(season), int(week))
        model.last_week[t] = (int(season), int(week))

    conf: Dict[str, str] = {}
    for g in ordered.itertuples():
        if bool(g.home_fbs):
            conf[str(g.home_team)] = str(getattr(g, "home_conf", "") or "")
        if bool(g.away_fbs):
            conf[str(g.away_team)] = str(getattr(g, "away_conf", "") or "")

    return KalmanTable(
        season=season, week=week, params=params,
        mu_state=dict(model.mu),
        var_state={t: model.var.get(t, params.prior_sd ** 2) for t in model.mu},
        n_games=n_games, team_conf=conf, n_games_fit=len(ordered))


_KALMAN_CACHE: Dict[str, KalmanTable] = {}


def _cache_key(games: pd.DataFrame, season: int, week: int,
               params: KalmanParams) -> str:
    hist = games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))]
    cols = ["id", "season", "week", "home_team", "away_team",
            "home_points", "away_points", "neutral_site", "home_fbs", "away_fbs"]
    blob = hist[cols].to_csv(index=False).encode()
    h = hashlib.sha256(blob).hexdigest()
    pkey = json.dumps({k: float(v) for k, v in sorted(vars(params).items())})
    return hashlib.sha256((h + pkey).encode()).hexdigest()


def solve_kalman_cached(games: pd.DataFrame, season: int, week: int,
                        params: Optional[KalmanParams] = None) -> KalmanTable:
    """Content-keyed memoised weekly solve — same policy as solve_cached."""
    params = params or FROZEN
    key = _cache_key(games, season, week, params)
    if key not in _KALMAN_CACHE:
        _KALMAN_CACHE[key] = solve_kalman(games, season, week, params)
    return _KALMAN_CACHE[key]


def clear_cache() -> None:
    _KALMAN_CACHE.clear()
