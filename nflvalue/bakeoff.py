"""Workstream G — model bakeoff candidates and the shared walk-forward harness.

Everything here executes the protocol PRE-REGISTERED in _worklog.md
(2026-07-30) — folds, candidates, hyperparameter policy, decision metric and
tie rule were fixed before any candidate ran. Read that entry first; this
docstring only restates the load-bearing parts:

  * identical folds: eval = FBS-vs-FBS, 2021-2024, weeks >= 4, walk-forward;
    graded on the intersection of games every candidate prices. 2025 is the
    final holdout, touched once by the selection winner + incumbent.
  * Elo / G-Elo / Kalman hyperparameters: grid-fit on 2019-2020 ONLY, frozen.
  * sklearn models: fixed defaults + fixed seed, refit before each eval week
    on strictly-prior games. No tuning — a tuned tree is a different,
    unregistered candidate.
  * primary metric: pooled margin MSE. Tie rule: within 1% of the incumbent
    ridge = tie, ridge stays.

The Glickman-Stern-style candidate uses a DIAGONAL-covariance Kalman
approximation (per-team mean/variance, opponent uncertainty folded into the
observation noise). That is an approximation and is recorded as one — the
full coupled filter is future work if this candidate wins on the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import ingest, ratings as R


# --------------------------------------------------------------------------- #
# Elo family
# --------------------------------------------------------------------------- #
@dataclass
class EloParams:
    k: float = 20.0
    hfa_elo: float = 55.0          # Elo points of home edge
    pts_per_elo: float = 0.04      # margin points per Elo point of diff
    mov: bool = False              # G-Elo margin-of-victory multiplier
    season_regress: float = 0.30   # offseason regression toward the mean


class EloModel:
    """Classic Elo over the FBS graph; FCS opponents pooled to one pseudo-team.

    Margin prediction = (elo_home - elo_away + hfa_elo) * pts_per_elo.
    G-Elo (mov=True) scales K by the 538 MOV multiplier
    ln(|margin|+1) * 2.2 / (0.001*elo_winner_diff + 2.2).
    """

    def __init__(self, params: EloParams):
        self.p = params
        self.elo: Dict[str, float] = {}
        self.season_seen: Optional[int] = None

    def _team(self, name: str, is_fbs: bool) -> str:
        return name if is_fbs else "FCS_TEAM"

    def _get(self, t: str) -> float:
        return self.elo.get(t, 1500.0)

    def _new_season(self) -> None:
        for t in self.elo:
            self.elo[t] = 1500.0 + (self.elo[t] - 1500.0) * (1 - self.p.season_regress)

    def predict(self, home: str, away: str, home_fbs: bool, away_fbs: bool,
                neutral: bool) -> float:
        h = self._get(self._team(home, home_fbs))
        a = self._get(self._team(away, away_fbs))
        hfa = 0.0 if neutral else self.p.hfa_elo
        return (h - a + hfa) * self.p.pts_per_elo

    def update(self, game) -> None:
        if self.season_seen is not None and game.season != self.season_seen:
            self._new_season()
        self.season_seen = game.season
        th = self._team(game.home_team, bool(game.home_fbs))
        ta = self._team(game.away_team, bool(game.away_fbs))
        h, a = self._get(th), self._get(ta)
        hfa = 0.0 if bool(game.neutral_site) else self.p.hfa_elo
        exp_home = 1.0 / (1.0 + 10 ** (-((h - a + hfa) / 400.0)))
        margin = float(game.margin)
        score_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        k = self.p.k
        if self.p.mov and margin != 0:
            winner_diff = (h + hfa - a) if margin > 0 else (a - h - hfa)
            k = k * np.log(abs(margin) + 1) * 2.2 / (0.001 * winner_diff + 2.2)
            k = float(np.clip(k, 0.0, 8 * self.p.k))
        delta = k * (score_home - exp_home)
        self.elo[th] = h + delta
        self.elo[ta] = a - delta


# --------------------------------------------------------------------------- #
# Diagonal-covariance Kalman (Glickman-Stern-style state-space, approximated)
# --------------------------------------------------------------------------- #
@dataclass
class KalmanParams:
    obs_sd: float = 16.0           # game noise
    week_sd: float = 0.8           # week-to-week strength drift
    season_sd: float = 6.0         # season-to-season drift
    prior_sd: float = 10.0         # new-team prior spread
    hfa: float = 2.8               # points


class KalmanModel:
    """Per-team (mean, var) filtered on margins. Opponent uncertainty enters
    the observation variance — the diagonal approximation, recorded as such."""

    def __init__(self, p: KalmanParams):
        self.p = p
        self.mu: Dict[str, float] = {}
        self.var: Dict[str, float] = {}
        self.last_week: Dict[str, Tuple[int, int]] = {}

    def _team(self, name: str, is_fbs: bool) -> str:
        return name if is_fbs else "FCS_TEAM"

    def _state(self, t: str) -> Tuple[float, float]:
        return self.mu.get(t, 0.0), self.var.get(t, self.p.prior_sd ** 2)

    def _age(self, t: str, season: int, week: int) -> None:
        """Inflate variance for elapsed time since the team's last game."""
        if t not in self.mu:
            return
        s0, w0 = self.last_week.get(t, (season, week))
        seasons = max(season - s0, 0)
        weeks = max((week - w0) if seasons == 0 else week, 0)
        self.var[t] = (self.var[t] + seasons * self.p.season_sd ** 2
                       + weeks * self.p.week_sd ** 2)

    def predict(self, home: str, away: str, home_fbs: bool, away_fbs: bool,
                neutral: bool) -> float:
        mh, _ = self._state(self._team(home, home_fbs))
        ma, _ = self._state(self._team(away, away_fbs))
        return mh - ma + (0.0 if neutral else self.p.hfa)

    def update(self, game) -> None:
        season, week = int(game.season), int(game.week)
        th = self._team(game.home_team, bool(game.home_fbs))
        ta = self._team(game.away_team, bool(game.away_fbs))
        for t in (th, ta):
            self._age(t, season, week)
            self.last_week[t] = (season, week)
        (mh, vh), (ma, va) = self._state(th), self._state(ta)
        hfa = 0.0 if bool(game.neutral_site) else self.p.hfa
        resid = float(game.margin) - (mh - ma + hfa)
        # Observation variance sees the opponent's uncertainty (diagonal appx)
        s_h = vh + va + self.p.obs_sd ** 2
        gain_h = vh / s_h
        gain_a = va / s_h
        self.mu[th] = mh + gain_h * resid
        self.mu[ta] = ma - gain_a * resid
        self.var[th] = vh * (1 - gain_h)
        self.var[ta] = va * (1 - gain_a)


# --------------------------------------------------------------------------- #
# Sequential evaluation for the online models (Elo/G-Elo/Kalman)
# --------------------------------------------------------------------------- #
def run_online_model(model, games: pd.DataFrame,
                     eval_seasons: Iterable[int],
                     first_week: int = 4) -> pd.DataFrame:
    """Feed games chronologically; before each game in the eval window,
    record the model's prediction, then update. Walk-forward by construction."""
    eval_set = set(int(s) for s in eval_seasons)
    rows = []
    ordered = games.sort_values(["season", "week", "id"], kind="mergesort")
    for g in ordered.itertuples():
        if (int(g.season) in eval_set and int(g.week) >= first_week
                and bool(g.home_fbs) and bool(g.away_fbs)):
            rows.append({"id": int(g.id), "season": int(g.season),
                         "week": int(g.week),
                         "pred": model.predict(g.home_team, g.away_team,
                                               bool(g.home_fbs),
                                               bool(g.away_fbs),
                                               bool(g.neutral_site)),
                         "margin": float(g.margin)})
        model.update(g)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tier-1 walk-forward feature frame for the sklearn candidates
# --------------------------------------------------------------------------- #
def feature_frame(seasons: List[int]) -> pd.DataFrame:
    """One row per FBS-vs-FBS game with strictly-prior-information features.

    All rolling values are expanding means over EARLIER weeks of the same
    season (cumsum-minus-current inside the team group — the same no-leak
    construction test_leakage's factor work uses); season-level values
    (talent, returning, prev-season net) are known preseason.
    """
    games = R.game_frame(seasons)
    plays = ingest.load_plays_clean(seasons)
    drives = ingest.load_drives(seasons)

    # per (season, offense, week) play efficiency
    pw = (plays.groupby(["season", "offense", "week"], observed=True)
          .agg(n=("ppa", "size"), ppa_sum=("ppa", "sum")).reset_index()
          .sort_values(["season", "offense", "week"]))
    go = pw.groupby(["season", "offense"], observed=True)
    for c in ("n", "ppa_sum"):
        pw[f"c_{c}"] = go[c].cumsum() - pw[c]
    pw["off_ppa"] = pw["c_ppa_sum"] / pw["c_n"].replace(0, np.nan)
    off_key = {(int(r.season), str(r.offense), int(r.week)): r.off_ppa
               for r in pw.itertuples() if r.c_n >= 60}

    dw = (plays.groupby(["season", "defense", "week"], observed=True)
          .agg(n=("ppa", "size"), ppa_sum=("ppa", "sum")).reset_index()
          .sort_values(["season", "defense", "week"]))
    gd = dw.groupby(["season", "defense"], observed=True)
    for c in ("n", "ppa_sum"):
        dw[f"c_{c}"] = gd[c].cumsum() - dw[c]
    dw["def_ppa"] = dw["c_ppa_sum"] / dw["c_n"].replace(0, np.nan)
    def_key = {(int(r.season), str(r.defense), int(r.week)): r.def_ppa
               for r in dw.itertuples() if r.c_n >= 60}

    from .cfb_sim import classify_drives
    d, _ = classify_drives(drives)
    d = d[d["start_period"] <= 4]
    d["trip"] = d["end_yards_to_goal"] <= 40
    d["trip_pts"] = np.where(d["trip"], d["off_pts"], np.nan)
    tw = (d.groupby(["season", "offense", "week"], observed=True)
          .agg(nd=("off_pts", "size"), trips=("trip", "sum"),
               tp=("trip_pts", "sum"), sytg=("start_yards_to_goal", "sum"))
          .reset_index().sort_values(["season", "offense", "week"]))
    gt = tw.groupby(["season", "offense"], observed=True)
    for c in ("nd", "trips", "tp", "sytg"):
        tw[f"c_{c}"] = gt[c].cumsum() - tw[c]
    tw["fin"] = tw["c_tp"] / tw["c_trips"].replace(0, np.nan)
    tw["fp"] = -(tw["c_sytg"] / tw["c_nd"].replace(0, np.nan))
    drv_key = {(int(r.season), str(r.offense), int(r.week)): (r.fin, r.fp)
               for r in tw.itertuples() if r.c_nd >= 30}

    talent = ingest.load_talent(seasons)
    tal = {}
    for s, grp in talent.groupby("season"):
        vals = grp.dropna(subset=["talent"])
        mu, sd = vals["talent"].mean(), max(vals["talent"].std(), 1e-9)
        for r in vals.itertuples():
            tal[(int(s), str(r.team))] = (float(r.talent) - mu) / sd
    returning = ingest.load_returning(seasons)
    ret = {(int(r.season), str(r.team)): float(r.percent_ppa)
           for r in returning.itertuples() if r.percent_ppa == r.percent_ppa}

    def latest(key, season, team, week):
        for w in range(week, 0, -1):
            v = key.get((season, team, w))
            if v is not None:
                return v
        return None

    rows = []
    for g in games.itertuples():
        if not (g.home_fbs and g.away_fbs):
            continue
        s, w = int(g.season), int(g.week)
        h, a = str(g.home_team), str(g.away_team)
        oh = latest(off_key, s, h, w)
        oa = latest(off_key, s, a, w)
        dh = latest(def_key, s, h, w)
        da = latest(def_key, s, a, w)
        dr_h = latest(drv_key, s, h, w) or (None, None)
        dr_a = latest(drv_key, s, a, w) or (None, None)
        rows.append({
            "id": int(g.id), "season": s, "week": w, "margin": float(g.margin),
            "off_ppa_diff": (oh - oa) if oh is not None and oa is not None else np.nan,
            "def_ppa_diff": (da - dh) if dh is not None and da is not None else np.nan,
            "fin_diff": (dr_h[0] - dr_a[0])
            if dr_h[0] is not None and dr_a[0] is not None else np.nan,
            "fp_diff": (dr_h[1] - dr_a[1])
            if dr_h[1] is not None and dr_a[1] is not None else np.nan,
            "talent_diff": tal.get((s, h), 0.0) - tal.get((s, a), 0.0),
            "returning_diff": ret.get((s, h), 0.5) - ret.get((s, a), 0.5),
            "neutral": float(bool(g.neutral_site)),
            "week_num": w,
        })
    return pd.DataFrame(rows)


def run_sklearn_model(make_model, feats: pd.DataFrame,
                      eval_seasons: Iterable[int],
                      first_week: int = 4) -> pd.DataFrame:
    """Refit before each eval week on all strictly-prior games. Fixed
    defaults, fixed seed — no tuning (registered policy)."""
    cols = ["off_ppa_diff", "def_ppa_diff", "fin_diff", "fp_diff",
            "talent_diff", "returning_diff", "neutral", "week_num"]
    df = feats.dropna(subset=["off_ppa_diff", "def_ppa_diff"]).copy()
    for c in ("fin_diff", "fp_diff"):
        df[c] = df[c].fillna(0.0)
    eval_set = set(int(s) for s in eval_seasons)
    rows = []
    weeks = (df[df["season"].isin(eval_set) & (df["week"] >= first_week)]
             [["season", "week"]].drop_duplicates()
             .sort_values(["season", "week"]))
    for _, (s, w) in weeks.iterrows():
        train = df[(df["season"] < s)
                   | ((df["season"] == s) & (df["week"] < w))]
        test = df[(df["season"] == s) & (df["week"] == w)]
        if len(train) < 500 or test.empty:
            continue
        model = make_model()
        model.fit(train[cols], train["margin"])
        pred = model.predict(test[cols])
        for (_, r), p in zip(test.iterrows(), pred):
            rows.append({"id": int(r["id"]), "season": int(s), "week": int(w),
                         "pred": float(p), "margin": float(r["margin"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Burn-in grid fits for the online models (2019-2020 only, then frozen)
# --------------------------------------------------------------------------- #
def fit_online_params(games: pd.DataFrame, make_model, grid: List[dict],
                      burn_seasons=(2019, 2020), first_week: int = 4,
                      min_preds: int = 100):
    """Choose the grid point minimizing MAE on the burn-in seasons' own
    walk-forward predictions. Never sees 2021+."""
    burn = games[games["season"].isin(burn_seasons)]
    best, best_mae = None, np.inf
    for cfg in grid:
        model = make_model(cfg)
        preds = run_online_model(model, burn, eval_seasons=list(burn_seasons),
                                 first_week=first_week)
        if len(preds) < min_preds:
            continue
        mae = float((preds["pred"] - preds["margin"]).abs().mean())
        if mae < best_mae:
            best, best_mae = cfg, mae
    if best is None:
        raise RuntimeError("no grid point produced enough burn-in predictions")
    return best, best_mae
