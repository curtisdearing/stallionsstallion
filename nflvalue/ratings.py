"""CFB ratings engine — hierarchical ridge solve over the whole game graph.

Workstream B. Replaces `build_ratings.py`'s sequential single-game Elo-style
update, which assumed NFL schedule density (16-17 games, everyone connected to
everyone within a couple of hops). College football does not have that: in the
2023 Mountain West slice, 12 teams play 12-13 games each and **39 other teams
appear exactly once**. A sequential updater walks that graph one edge at a time
and never propagates strength through chains; a joint solve does.

WHAT THIS MODULE IS
-------------------
One regularised least-squares problem, refit from scratch every week over every
game played *strictly before* that week. Two rows per game (points scored by
each side), so offence and defence are separately identified — a margin-only
design identifies `off + def` and nothing finer.

    points_scored(s vs o) = mu
                          + hfa_league + hfa_dev[s]      (home, non-neutral only)
                          + conf_off[conf(s)] + conf_def[conf(o)]
                          + off[s] + def[o]

`def[o]` is signed as *points allowed above average*, so a good defence is
negative. The exported `def_rating` is `-def[o]` so that higher is better
everywhere in the output and `net = off_rating + def_rating`. Subtracting the
two team rows reproduces the spec's margin identity exactly:

    margin = hfa + (off[h] + def[a]) - (off[a] + def[h])

WHY HIERARCHICAL RIDGE RATHER THAN "SOLVE, THEN SHRINK"
-------------------------------------------------------
Conference strength is an **explicit parameter block**, penalised more weakly
than team deviations. A team with one game therefore cannot pull its own
coefficient far from zero, and what it does explain gets absorbed by its
conference term — i.e. its rating *is* the conference level plus a small
deviation. That is the conference-mean shrinkage BUILD_SPEC asks for, obtained
as a property of the estimator rather than bolted on afterwards.

The conference penalty is itself scaled by the number of **cross-conference**
games, because those are the only games that can identify it. Counting a
conference's internal games instead was a real bug during the build: it made a
large conference's term the cheapest parameter in the model, so a single
cross-conference blowout was paid for by swinging a 49-game conference 28
points rather than by moving the one team that played. It also happens to be
the behaviour the spec describes — conference strength "re-estimated as
non-conference games accumulate".

Every penalised column is centred against the free intercept before solving.
Without that, `mu` and a conference term appearing in nearly every row are two
intercepts fighting over the same level, the split is arbitrary, and the
exported ratings become meaningless in absolute terms while the margins stay
fine — which is the worst kind of bug, because nothing looks broken.

CONNECTIVITY IS AN INPUT, NOT A LABEL
-------------------------------------
`connectivity_table()` runs a multi-source BFS from a stable reference set
through the FBS-only game graph. The resulting score does three things, none of
them cosmetic:

  1. it **scales the ridge penalty per team** — a badly connected team is
     shrunk harder toward its conference mean, so it literally changes the
     fitted number;
  2. it feeds the per-team standard error via the ridge posterior covariance,
     which is the uncertainty band Workstream C/F consume; and
  3. it sets `veto`, which marks a team as not-ratable at all.

A team in a component that never reaches the reference set gets `connectivity=0`
and `veto=True`; asking the table for a usable rating on such a team raises
`DisconnectedTeamError` rather than handing back a garbage number.

FCS OPPONENTS
-------------
Never fed at face value. Every non-FBS team collapses into a single pseudo-team
`__FCS__` in its own pseudo-conference, and those rows carry a weight < 1. A
70-3 result therefore informs one pooled FCS coefficient instead of setting an
FBS team's rating, and FCS edges are excluded from the connectivity graph
entirely (beating an FCS team tells you nothing about where you sit in the FBS
graph).

TALENT SENTINELS
----------------
`ingest` already converts CFBD's `talent == 0` to NULL + `unranked_recruiting`
for Army, Navy and Air Force. This module must never let a NULL become a low
number: `preseason_prior()` substitutes the **conference mean z-score**, and
`tests/test_ratings.py` asserts Air Force does not land in the bottom of the
prior. Air Force led the 2023 Mountain West in net PPA; treating it as the
worst-recruited team in FBS would be a modelling error, not a rounding one.

NO CONSTANT IS GUESSED
----------------------
Every free parameter lives in `RatingParams`, and `walk_forward_search()` fits
them by predicting each week from the weeks before it. Nothing here inherits an
NFL magnitude — HFA in particular is estimated from the CFB rows, not set to
the NFL parent's ~2.0.

DEPENDENCIES: numpy + pandas only. BUILD_SPEC suggests sparse matrices; at 56
teams the normal equations are a ~130x130 dense solve that runs in under a
millisecond, and scipy is listed in requirements.txt but is not installed in
this environment. Dense is used deliberately, not by oversight. Likewise the
suggested "warm start from last week's solution" is a no-op for a direct
Cholesky solve — there is no iterate to warm-start — so the caching requirement
is met by content-hash memoisation of whole weekly solutions instead, and this
paragraph exists so nobody files that as a missing feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, asdict, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import ingest

FCS_TEAM = "__FCS__"
FCS_CONF = "__FCS__"

#: Rounding used for every exported number and for the reproducibility
#: fingerprint. Ridge solves are deterministic, but BLAS summation order can
#: differ in the last couple of ULPs across builds; 6dp is far finer than any
#: decision this model makes and keeps the fingerprint honest.
ROUND = 6


class RatingsError(RuntimeError):
    """Base class for anything the ratings engine refuses to do."""


class IllConditionedError(RatingsError):
    """The normal-equation matrix is too ill-conditioned to trust."""


class DisconnectedTeamError(RatingsError):
    """A rating was requested for a team the game graph cannot reach."""


# --------------------------------------------------------------------------- #
# Parameters — every one of these is fit, not chosen
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RatingParams:
    """Free parameters of the ratings engine.

    Defaults are *starting points for a search*, not beliefs. `walk_forward_search`
    picks the values; anything left at a default in a shipped run should be
    reported as unfit.
    """

    #: ridge penalty on team offence/defence deviations
    lambda_team: float = 12.0
    #: ridge penalty on conference terms — much weaker, so shared strength
    #: flows into the conference block instead of into one-game teams
    lambda_conf: float = 1.0
    #: ridge penalty on per-team HFA deviations (league HFA itself is free)
    lambda_hfa: float = 40.0
    #: exponential recency decay per week of game age (0 = flat)
    decay: float = 0.0
    #: row weight on games against pooled FCS opposition
    fcs_weight: float = 0.25
    #: multiplier on lambda_team for explosiveness units (spec: explosiveness
    #: gets materially more shrinkage than efficiency)
    explosiveness_shrinkage: float = 4.0
    #: points of net rating per 1 sd of the composite preseason prior
    prior_strength: float = 6.0
    #: relative weights inside the preseason prior
    prior_w_talent: float = 1.0
    prior_w_returning: float = 1.0
    prior_w_prev_rating: float = 1.0
    #: connectivity: games needed before a team is "self-supporting"
    conn_half_games: float = 4.0
    #: connectivity decay per BFS hop away from the reference set
    conn_hop_decay: float = 0.6
    #: a team needs this many FBS games to join the reference set
    conn_ref_min_games: int = 5
    #: floor on the connectivity multiplier so the penalty stays finite
    conn_floor: float = 0.05
    #: Below this connectivity a team is vetoed outright. The default is not a
    #: magic number: with conn_half_games=4 and conn_hop_decay=0.6 it is exactly
    #: the rule "a team needs >=2 FBS games AND must sit within one hop of the
    #: reference set" (1 game at 1 hop scores 0.200; 2 games at 1 hop, 0.333).
    #: Change the connectivity parameters and this stops meaning that, so it is
    #: fit alongside them rather than pinned.
    conn_veto_below: float = 0.25
    #: fail loud above this condition number
    max_condition: float = 1e10

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def game_frame(seasons: Iterable[int]) -> pd.DataFrame:
    """Played games with scores, conferences and FCS flags resolved.

    Conference comes from the per-season `/teams` pull, never a hardcoded map.
    """
    seasons = sorted({int(s) for s in seasons})
    games = ingest.load_games(seasons)
    teams = ingest.load_teams(seasons)

    conf = {(int(r.season), str(r.team)): (str(r.conference) or "Independent")
            for r in teams.itertuples()}
    fbs = {(int(r.season), str(r.team)) for r in teams.itertuples() if bool(r.is_fbs)}

    g = games.copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].astype(str)
    g = g.dropna(subset=["home_points", "away_points"]).copy()
    g["home_points"] = g["home_points"].astype(float)
    g["away_points"] = g["away_points"].astype(float)
    g["margin"] = g["home_points"] - g["away_points"]

    # FCS flags come from ingest; recompute conference off the teams table so
    # an empty CFBD conference string never silently becomes its own bucket.
    g["home_fbs"] = [(int(s), t) in fbs for s, t in zip(g["season"], g["home_team"])]
    g["away_fbs"] = [(int(s), t) in fbs for s, t in zip(g["season"], g["away_team"])]
    g["home_conf"] = [conf.get((int(s), t), "Independent")
                      for s, t in zip(g["season"], g["home_team"])]
    g["away_conf"] = [conf.get((int(s), t), "Independent")
                      for s, t in zip(g["season"], g["away_team"])]
    g["neutral_site"] = g["neutral_site"].astype(bool)
    return g.sort_values(["season", "week", "id"], kind="mergesort").reset_index(drop=True)


def _pool_fcs(team: str, is_fbs: bool) -> str:
    return team if is_fbs else FCS_TEAM


def _pool_conf(conf: str, is_fbs: bool) -> str:
    return conf if is_fbs else FCS_CONF


# --------------------------------------------------------------------------- #
# Connectivity — a real input
# --------------------------------------------------------------------------- #
@dataclass
class Connectivity:
    hops: Dict[str, Optional[int]]
    n_games: Dict[str, int]
    component_size: Dict[str, int]
    score: Dict[str, float]
    reference: List[str]

    def penalty_scale(self, team: str, floor: float) -> float:
        """Multiplier on the ridge penalty. Badly connected => penalised more."""
        s = max(self.score.get(team, 0.0), floor)
        return 1.0 / s


def connectivity_table(games: pd.DataFrame, params: RatingParams) -> Connectivity:
    """Shortest path through the FBS game graph to a stable reference set.

    FCS edges are excluded on purpose: beating an FCS team does not connect you
    to anybody in the FBS graph, and counting it as if it did is exactly the
    kind of flattery that makes a September rating look sturdier than it is.
    """
    adj: Dict[str, set] = {}
    n_games: Dict[str, int] = {}
    for r in games.itertuples():
        h, a = r.home_team, r.away_team
        if not (r.home_fbs and r.away_fbs):
            # still note the FBS side exists, but the edge is not usable
            for t, ok in ((h, r.home_fbs), (a, r.away_fbs)):
                if ok:
                    adj.setdefault(t, set())
            continue
        adj.setdefault(h, set()).add(a)
        adj.setdefault(a, set()).add(h)
        n_games[h] = n_games.get(h, 0) + 1
        n_games[a] = n_games.get(a, 0) + 1

    teams = sorted(adj)
    for t in teams:
        n_games.setdefault(t, 0)

    reference = sorted(t for t in teams if n_games[t] >= params.conn_ref_min_games)
    if not reference and teams:
        # Early season: nobody has hit the bar yet. Fall back to the best-connected
        # teams so the metric degrades smoothly instead of dividing by nothing.
        best = max(n_games[t] for t in teams)
        reference = sorted(t for t in teams if n_games[t] == best)

    hops: Dict[str, Optional[int]] = {t: None for t in teams}
    dq = deque()
    for t in reference:
        hops[t] = 0
        dq.append(t)
    while dq:
        cur = dq.popleft()
        for nxt in sorted(adj[cur]):
            if hops.get(nxt) is None:
                hops[nxt] = hops[cur] + 1
                dq.append(nxt)

    # connected components (for reporting / fail-loud)
    comp_size: Dict[str, int] = {}
    seen = set()
    for t in teams:
        if t in seen:
            continue
        stack, comp = [t], []
        seen.add(t)
        while stack:
            c = stack.pop()
            comp.append(c)
            for nxt in adj[c]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        for c in comp:
            comp_size[c] = len(comp)

    score: Dict[str, float] = {}
    for t in teams:
        h = hops[t]
        if h is None:
            score[t] = 0.0
            continue
        games_factor = n_games[t] / (n_games[t] + params.conn_half_games)
        hop_factor = params.conn_hop_decay ** max(h - 1, 0)
        score[t] = round(float(games_factor * hop_factor), 10)

    return Connectivity(hops=hops, n_games=n_games, component_size=comp_size,
                        score=score, reference=reference)


# --------------------------------------------------------------------------- #
# Preseason prior
# --------------------------------------------------------------------------- #
def preseason_prior(season: int, teams: Sequence[str], conf_of: Dict[str, str],
                    params: RatingParams,
                    prev_net: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Net-points prior per team from talent + returning production (+ last year).

    LOAD-BEARING: talent is NULL for Army, Navy and Air Force because CFBD
    serves 0 as a missing-data sentinel for service academies. A NULL here
    means *unmeasured*, never *bad*. Missing components fall back to the team's
    **conference mean**, and if the whole conference is missing, to 0 (the
    league mean of the z-score) — never to a low value.
    """
    talent = ingest.load_talent([season])
    returning = ingest.load_returning([season])

    tal = {str(r.team): (float(r.talent) if pd.notna(r.talent) else None)
           for r in talent.itertuples()}
    ret = {str(r.team): (float(r.percent_ppa) if pd.notna(r.percent_ppa) else None)
           for r in returning.itertuples()}

    def _z(raw: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        vals = [v for v in raw.values() if v is not None]
        if len(vals) < 2:
            return {k: (0.0 if v is not None else None) for k, v in raw.items()}
        mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        if sd == 0:
            return {k: (0.0 if v is not None else None) for k, v in raw.items()}
        return {k: ((v - mu) / sd if v is not None else None) for k, v in raw.items()}

    z_tal, z_ret = _z(tal), _z(ret)
    z_prev = _z({t: (prev_net.get(t) if prev_net else None) for t in teams})

    def _with_conf_fallback(z: Dict[str, Optional[float]]) -> Dict[str, float]:
        by_conf: Dict[str, List[float]] = {}
        for t in teams:
            v = z.get(t)
            if v is not None:
                by_conf.setdefault(conf_of.get(t, "Independent"), []).append(v)
        conf_mean = {c: float(np.mean(v)) for c, v in by_conf.items()}
        out = {}
        for t in teams:
            v = z.get(t)
            if v is None:
                v = conf_mean.get(conf_of.get(t, "Independent"), 0.0)
            out[t] = float(v)
        return out

    zt = _with_conf_fallback(z_tal)
    zr = _with_conf_fallback(z_ret)
    zp = _with_conf_fallback(z_prev)

    w_t, w_r = params.prior_w_talent, params.prior_w_returning
    w_p = params.prior_w_prev_rating if prev_net else 0.0
    tot = w_t + w_r + w_p
    if tot <= 0:
        return {t: 0.0 for t in teams}

    return {t: round(float(params.prior_strength
                           * (w_t * zt[t] + w_r * zr[t] + w_p * zp[t]) / tot), 10)
            for t in teams}


# --------------------------------------------------------------------------- #
# The solve
# --------------------------------------------------------------------------- #
@dataclass
class RatingsTable:
    season: int
    week: int
    params: RatingParams
    mu: float
    hfa_league: float
    hfa_team: Dict[str, float]
    conf_off: Dict[str, float]
    conf_def: Dict[str, float]
    conf_games: Dict[str, int]
    off: Dict[str, float]            # EFFECTIVE: team deviation + its conference term
    def_: Dict[str, float]           # EFFECTIVE points allowed above avg (lower better)
    off_dev: Dict[str, float]        # raw team-deviation block, for inspection
    def_dev: Dict[str, float]
    team_conf: Dict[str, str]
    net: Dict[str, float]
    net_sd: Dict[str, float]         # posterior sd of the fitted coefficients
    net_band: Dict[str, float]       # net_sd widened by prior misfit x (1-connectivity)
    connectivity: Connectivity
    prior: Dict[str, float]
    prior_misfit_sd: float
    veto: Dict[str, bool]
    n_games_fit: int
    condition: float
    residual_sd: float

    # ---- accessors -------------------------------------------------------- #
    def def_rating(self, team: str) -> float:
        """Higher is better (sign-flipped from the raw solve coefficient)."""
        return round(-self.def_.get(team, 0.0), ROUND)

    def hfa(self, team: str, neutral: bool = False) -> float:
        """Per-team HFA. Exactly zero at a neutral site, by construction."""
        if neutral:
            return 0.0
        return round(self.hfa_league + self.hfa_team.get(team, 0.0), ROUND)

    def require_usable(self, team: str) -> None:
        if self.veto.get(team, True):
            raise DisconnectedTeamError(
                f"{team!r} is not ratable at season={self.season} week={self.week}: "
                f"connectivity={self.connectivity.score.get(team, 0.0):.4f}, "
                f"hops={self.connectivity.hops.get(team)}, "
                f"fbs_games={self.connectivity.n_games.get(team, 0)}. "
                "Refusing to emit a rating rather than emit a garbage one.")

    def predict_margin(self, home: str, away: str, neutral: bool = False) -> float:
        """Home-minus-away expected margin. Raises on unratable teams."""
        self.require_usable(home)
        self.require_usable(away)
        return round(self.hfa(home, neutral)
                     + (self.off.get(home, 0.0) + self.def_.get(away, 0.0))
                     - (self.off.get(away, 0.0) + self.def_.get(home, 0.0)), ROUND)

    def to_frame(self) -> pd.DataFrame:
        teams = sorted(self.net)
        return pd.DataFrame({
            "season": self.season, "week": self.week, "team": teams,
            "off": [round(self.off[t], ROUND) for t in teams],
            "def": [self.def_rating(t) for t in teams],
            "net": [round(self.net[t], ROUND) for t in teams],
            "net_sd": [round(self.net_sd[t], ROUND) for t in teams],
            "net_band": [round(self.net_band[t], ROUND) for t in teams],
            "hfa": [round(self.hfa_league + self.hfa_team.get(t, 0.0), ROUND)
                    for t in teams],
            "prior": [round(self.prior.get(t, 0.0), ROUND) for t in teams],
            "connectivity": [round(self.connectivity.score.get(t, 0.0), ROUND)
                             for t in teams],
            "hops_to_reference": [self.connectivity.hops.get(t) for t in teams],
            "fbs_games": [self.connectivity.n_games.get(t, 0) for t in teams],
            "veto": [bool(self.veto.get(t, True)) for t in teams],
        }).sort_values("net", ascending=False).reset_index(drop=True)

    def fingerprint(self) -> str:
        """Stable content hash of every exported number. Run-to-run equality."""
        payload = {
            # int() not the raw value: a numpy int64 week (which is what you get
            # from `df["week"].unique()`) is not JSON-serialisable, and the
            # fingerprint must not depend on how the caller happened to type it.
            "season": int(self.season), "week": int(self.week),
            "params": self.params.key(),
            "mu": round(self.mu, ROUND), "hfa_league": round(self.hfa_league, ROUND),
            "off": {k: round(v, ROUND) for k, v in sorted(self.off.items())},
            "def": {k: round(v, ROUND) for k, v in sorted(self.def_.items())},
            "hfa_team": {k: round(v, ROUND) for k, v in sorted(self.hfa_team.items())},
            "conf_off": {k: round(v, ROUND) for k, v in sorted(self.conf_off.items())},
            "conf_def": {k: round(v, ROUND) for k, v in sorted(self.conf_def.items())},
            "net_sd": {k: round(v, ROUND) for k, v in sorted(self.net_sd.items())},
            "net_band": {k: round(v, ROUND) for k, v in sorted(self.net_band.items())},
            "conn": {k: round(v, ROUND) for k, v in sorted(self.connectivity.score.items())},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


#: Weeks charged for crossing a season boundary when computing recency decay.
#: A game 12 months ago is older than a game 12 weeks ago in every sense that
#: matters (roster turnover, coaching change, portal), so the gap is explicit
#: rather than implied by week numbers restarting at 1.
WEEKS_PER_SEASON = 20


def _row_weights(games: pd.DataFrame, as_of_season: int, as_of_week: int,
                 params: RatingParams) -> np.ndarray:
    age = ((as_of_season - games["season"].to_numpy(dtype=float)) * WEEKS_PER_SEASON
           + (as_of_week - games["week"].to_numpy(dtype=float)))
    w = np.exp(-params.decay * np.maximum(age, 0.0))
    fcs_game = ~(games["home_fbs"].to_numpy() & games["away_fbs"].to_numpy())
    w = np.where(fcs_game, w * params.fcs_weight, w)
    return w


def solve(games: pd.DataFrame, season: int, week: int, params: RatingParams,
          prev_net: Optional[Dict[str, float]] = None,
          prior: Optional[Dict[str, float]] = None) -> RatingsTable:
    """Fit ratings from every game strictly BEFORE `week` of `season`.

    Walk-forward by construction: the filter is `< week`, never `<= week`, and
    nothing in here reads a column the games table would not have had before
    kickoff. `tests/test_leakage.py` poisons the future and asserts this.
    """
    hist = games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))].copy()
    if hist.empty:
        raise RatingsError(f"no games before season={season} week={week}")

    conn = connectivity_table(hist, params)

    # ---- label spaces (sorted => deterministic column order) --------------- #
    def _sides(g):
        for r in g.itertuples():
            yield (_pool_fcs(r.home_team, r.home_fbs), _pool_conf(r.home_conf, r.home_fbs),
                   _pool_fcs(r.away_team, r.away_fbs), _pool_conf(r.away_conf, r.away_fbs),
                   r.home_points, r.away_points, bool(r.neutral_site))

    sides = list(_sides(hist))
    teams = sorted({s[0] for s in sides} | {s[2] for s in sides})
    confs = sorted({s[1] for s in sides} | {s[3] for s in sides})
    fbs_teams = [t for t in teams if t != FCS_TEAM]

    ti = {t: i for i, t in enumerate(teams)}
    ci = {c: i for i, c in enumerate(confs)}
    nT, nC = len(teams), len(confs)

    # column layout
    c_mu = 0
    c_hfa = 1
    c_hfa_dev = 2                     # + nT
    c_conf_off = c_hfa_dev + nT       # + nC
    c_conf_def = c_conf_off + nC      # + nC
    c_off = c_conf_def + nC           # + nT
    c_def = c_off + nT                # + nT
    P = c_def + nT

    nrows = 2 * len(sides)
    X = np.zeros((nrows, P), dtype=float)
    y = np.zeros(nrows, dtype=float)
    w = np.repeat(_row_weights(hist, season, week, params), 2)

    for k, (ht, hc, at, ac, hp, ap, neutral) in enumerate(sides):
        r_home, r_away = 2 * k, 2 * k + 1
        # home scoring row
        X[r_home, c_mu] = 1.0
        if not neutral:
            X[r_home, c_hfa] = 1.0
            X[r_home, c_hfa_dev + ti[ht]] = 1.0
        X[r_home, c_conf_off + ci[hc]] = 1.0
        X[r_home, c_conf_def + ci[ac]] = 1.0
        X[r_home, c_off + ti[ht]] = 1.0
        X[r_home, c_def + ti[at]] = 1.0
        y[r_home] = hp
        # away scoring row (no HFA term at all — the away side never gets one,
        # and at a neutral site neither side does)
        X[r_away, c_mu] = 1.0
        X[r_away, c_conf_off + ci[ac]] = 1.0
        X[r_away, c_conf_def + ci[hc]] = 1.0
        X[r_away, c_off + ti[at]] = 1.0
        X[r_away, c_def + ti[ht]] = 1.0
        y[r_away] = ap

    # ---- preseason prior as the ridge PRIOR MEAN (not a post-hoc blend) ---- #
    # Conference membership for EXPORT is taken from the most recent season in
    # the fit window. The design matrix already uses each game's own-season
    # conference row by row, so realignment is handled correctly in the fit;
    # this map only decides which conference term a team's exported rating is
    # credited with, and "the one it plays in now" is the right answer there.
    conf_of: Dict[str, str] = {}
    latest = int(hist["season"].max())
    for r in hist.itertuples():
        if int(r.season) != latest:
            continue
        conf_of[_pool_fcs(r.home_team, r.home_fbs)] = _pool_conf(r.home_conf, r.home_fbs)
        conf_of[_pool_fcs(r.away_team, r.away_fbs)] = _pool_conf(r.away_conf, r.away_fbs)
    for (ht, hc, at, ac, *_rest) in sides:
        conf_of.setdefault(ht, hc)
        conf_of.setdefault(at, ac)
    if prior is None:
        prior_net = preseason_prior(season, fbs_teams, conf_of, params,
                                    prev_net=prev_net)
    else:
        # Explicit prior injection exists so tests can exercise the solve
        # without a parquet cache. Production callers leave it None.
        prior_net = {t: float(prior.get(t, 0.0)) for t in fbs_teams}
    prior_net[FCS_TEAM] = prior_net.get(FCS_TEAM, 0.0)

    theta0 = np.zeros(P, dtype=float)
    for t in teams:
        if t == FCS_TEAM:
            continue
        half = prior_net.get(t, 0.0) / 2.0
        theta0[c_off + ti[t]] = half
        theta0[c_def + ti[t]] = -half     # def is "allowed above avg": good => negative

    # ---- penalties -------------------------------------------------------- #
    lam = np.zeros(P, dtype=float)
    lam[c_mu] = 0.0                       # intercept unpenalised
    lam[c_hfa] = 0.0                      # league HFA free — never inherited from NFL
    if not X[:, c_hfa].any():
        # Every game in the window was at a neutral site (a bowl-only or
        # championship-week slice). The HFA column is structurally zero, so
        # leaving it unpenalised makes the normal equations singular. Pin it
        # to exactly 0 — which is the right answer — instead of failing.
        lam[c_hfa] = 1.0
    for t in teams:
        scale = conn.penalty_scale(t, params.conn_floor)
        lam[c_hfa_dev + ti[t]] = params.lambda_hfa * scale
        lam[c_off + ti[t]] = params.lambda_team * scale
        lam[c_def + ti[t]] = params.lambda_team * scale
    # A conference term is only as trustworthy as the number of games that can
    # IDENTIFY it, and only CROSS-conference games can. Counting a conference's
    # internal games here was a bug worth spelling out: on a synthetic league
    # where one conference held 49 of 50 games it made that conference's term
    # the cheapest parameter in the model, so a single 80-0 cross-conference
    # result was paid for by swinging the 49-game conference 28 points rather
    # than by moving the one team that actually played the game. This is also
    # exactly the behaviour BUILD_SPEC describes — conference strength
    # "re-estimated as non-conference games accumulate".
    conf_games: Dict[str, int] = {c: 0 for c in confs}
    for (ht, hc, at, ac, *_r) in sides:
        if hc == ac:
            continue
        conf_games[hc] = conf_games.get(hc, 0) + 1
        conf_games[ac] = conf_games.get(ac, 0) + 1
    for c in confs:
        n = conf_games.get(c, 0)
        # monotone in n, finite at n=0 (a conference nobody has played outside
        # of is not measurable, so its term is pinned near the league mean)
        scale = 1.0 + params.conn_half_games / max(float(n), 0.5)
        lam[c_conf_off + ci[c]] = params.lambda_conf * scale
        lam[c_conf_def + ci[c]] = params.lambda_conf * scale

    # ---- centre every penalised column against the free intercept --------- #
    # Without this the model is very nearly degenerate: `mu` is unpenalised and
    # a conference term that appears in almost every row is a second intercept,
    # so the level splits between them arbitrarily. Observed before the fix, on
    # a synthetic league where one conference covered 49 of 50 games: mu drifted
    # to 37.7 and conf_off to -20.3, and the exported `net` levels became
    # nonsense while the margins stayed fine. Subtracting each penalised
    # column's weighted mean is an exact reparameterisation — every fitted value
    # and every predicted margin is unchanged, because the shift is constant
    # across rows and is absorbed by `mu` — but it makes the penalty mean what
    # it says: shrink toward the league average, not toward zero-on-an-arbitrary-scale.
    penalised = lam > 0
    col_mean = np.zeros(P)
    wsum = float(w.sum())
    if wsum > 0:
        col_mean[penalised] = (w @ X[:, penalised]) / wsum
    X = X - col_mean[None, :]

    theta, A_inv, cond = _ridge(X, y, w, lam, theta0, params.max_condition)

    resid = y - X @ theta
    dof = max(nrows - np.count_nonzero(lam == 0) - 1, 1)
    sigma2 = float((w * resid ** 2).sum() / dof)

    off_dev = {t: float(theta[c_off + ti[t]]) for t in teams}
    def_dev = {t: float(theta[c_def + ti[t]]) for t in teams}
    conf_off = {c: float(theta[c_conf_off + ci[c]]) for c in confs}
    conf_def = {c: float(theta[c_conf_def + ci[c]]) for c in confs}
    hfa_team = {t: float(theta[c_hfa_dev + ti[t]]) for t in teams}

    # EFFECTIVE ratings fold the conference term into the team. A rating that
    # ignored it would make a heavily shrunk one-game team look league-average
    # regardless of whether it plays in the SEC or the MAC, and — worse —
    # predict_margin would then price a cross-conference game as if the two
    # conferences were equal, which is the exact error the explicit conference
    # term exists to prevent.
    off, dff, net, net_sd = {}, {}, {}, {}
    for t in teams:
        c = conf_of.get(t, "Independent")
        off[t] = off_dev[t] + conf_off.get(c, 0.0)
        dff[t] = def_dev[t] + conf_def.get(c, 0.0)
        net[t] = off[t] - dff[t]
        io, idf = c_off + ti[t], c_def + ti[t]
        var = sigma2 * (A_inv[io, io] + A_inv[idf, idf] - 2 * A_inv[io, idf])
        net_sd[t] = float(math.sqrt(max(var, 0.0)))

    # ---- how wrong is the preseason prior, empirically? ------------------- #
    # `net_sd` alone is the posterior sd of the coefficients GIVEN the prior,
    # and for a one-game team that is small precisely because the prior is
    # doing all the work — which reports high confidence for the teams we know
    # least about. Backwards. So measure the prior's own error on the teams
    # that do have data (well-connected ones), and widen every team's band by
    # that amount scaled by how little the graph constrains it. This is
    # estimated from the fit, not chosen.
    anchored = [t for t in teams
                if t != FCS_TEAM and conn.score.get(t, 0.0) >= params.conn_veto_below]
    misfits = [net[t] - prior_net.get(t, 0.0) for t in anchored]
    prior_misfit_sd = float(np.std(misfits, ddof=1)) if len(misfits) >= 3 else 0.0

    net_band = {}
    for t in teams:
        slack = (1.0 - min(conn.score.get(t, 0.0), 1.0)) * prior_misfit_sd
        net_band[t] = float(math.sqrt(net_sd[t] ** 2 + slack ** 2))

    veto = {t: (t == FCS_TEAM or conn.score.get(t, 0.0) < params.conn_veto_below)
            for t in teams}

    return RatingsTable(
        season=season, week=week, params=params,
        mu=float(theta[c_mu]), hfa_league=float(theta[c_hfa]),
        hfa_team=hfa_team, conf_off=conf_off, conf_def=conf_def,
        conf_games=conf_games,
        off=off, def_=dff, off_dev=off_dev, def_dev=def_dev, team_conf=conf_of,
        net=net, net_sd=net_sd, net_band=net_band,
        connectivity=conn, prior=prior_net, prior_misfit_sd=prior_misfit_sd,
        veto=veto, n_games_fit=len(hist), condition=cond,
        residual_sd=float(math.sqrt(sigma2)),
    )


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: np.ndarray,
           theta0: np.ndarray, max_condition: float):
    """Weighted ridge with a non-zero prior mean. Deterministic, dense."""
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(lam)
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > max_condition:
        raise IllConditionedError(
            f"design matrix condition number {cond:.3e} exceeds {max_condition:.3e}; "
            "refusing to emit ratings from an unidentifiable fit. Usually this means "
            "a whole conference is present in only one game, or lambda_conf is 0.")
    b = Xw.T @ (y - X @ theta0)
    delta = np.linalg.solve(A, b)
    A_inv = np.linalg.inv(A)
    return theta0 + delta, A_inv, cond


# --------------------------------------------------------------------------- #
# Unit-level ratings (8 numbers)
# --------------------------------------------------------------------------- #
RUSH_TYPES = {"rush", "rushing touchdown", "sack", "fumble recovery (own)",
              "fumble recovery (opponent)", "fumble return touchdown"}
PASS_TYPES = {"pass reception", "pass incompletion", "passing touchdown",
              "interception", "pass interception return",
              "interception return touchdown", "pass completion"}

UNIT_NAMES = [
    "rush_off_eff", "rush_off_exp", "pass_off_eff", "pass_off_exp",
    "rush_def_eff", "rush_def_exp", "pass_def_eff", "pass_def_exp",
]


def _success(down, distance, gained) -> Optional[bool]:
    try:
        d, dist, g = int(down), float(distance), float(gained)
    except (TypeError, ValueError):
        return None
    if d < 1 or d > 4 or dist <= 0:
        return None
    need = {1: 0.5, 2: 0.7}.get(d, 1.0) * dist
    return g >= need


def play_units(plays: pd.DataFrame) -> pd.DataFrame:
    """Per (game, offense, phase) efficiency + explosiveness from clean plays.

    Connelly's definitions: **efficiency** = success rate; **explosiveness** =
    mean PPA on successful plays only. Sacks count as rush plays here (that is
    how CFBD types them) and are left in — removing them would flatter every
    pass offence that cannot protect.

    Input MUST be `plays_clean` (garbage time already removed at ingest).
    """
    p = plays.copy()
    pt = p["play_type"].astype(str).str.lower()
    p["phase"] = np.where(pt.isin(RUSH_TYPES), "rush",
                          np.where(pt.isin(PASS_TYPES), "pass", ""))
    p = p[p["phase"] != ""].copy()
    p["success"] = [_success(d, dist, g) for d, dist, g
                    in zip(p["down"], p["distance"], p["yards_gained"])]
    p = p[p["success"].notna()].copy()
    p["success"] = p["success"].astype(bool)
    p["ppa"] = pd.to_numeric(p["ppa"], errors="coerce")

    rows = []
    for (gid, off, dfn, phase), grp in p.groupby(
            ["game_id", "offense", "defense", "phase"], observed=True, sort=True):
        succ = grp[grp["success"]]
        expl = succ["ppa"].mean()
        rows.append({
            "game_id": int(gid), "offense": str(off), "defense": str(dfn),
            "phase": str(phase), "n": int(len(grp)),
            "eff": float(grp["success"].mean()),
            "exp": float(expl) if pd.notna(expl) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["game_id", "offense", "phase"], kind="mergesort").reset_index(drop=True)


def solve_units(games: pd.DataFrame, units: pd.DataFrame, season: int, week: int,
                params: RatingParams) -> Dict[str, Dict[str, float]]:
    """Eight opponent-adjusted unit ratings, same hierarchical ridge machinery.

    Explosiveness is penalised `explosiveness_shrinkage`x harder than efficiency.
    That is not a stylistic choice: `research/win-factors-literature-scan.md`
    records the externally confirmed finding that explosiveness has near-zero
    week-to-week stickiness. An unshrunk explosiveness rating is mostly a
    record of which team happened to hit a 70-yard run last Saturday.
    """
    hist = games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))]
    if hist.empty:
        raise RatingsError(f"no games before season={season} week={week}")

    meta = {}
    for r in hist.itertuples():
        meta[int(r.id)] = {
            r.home_team: (_pool_fcs(r.home_team, r.home_fbs),
                          _pool_conf(r.home_conf, r.home_fbs),
                          _pool_fcs(r.away_team, r.away_fbs),
                          _pool_conf(r.away_conf, r.away_fbs),
                          bool(r.home_fbs and r.away_fbs), int(r.week)),
            r.away_team: (_pool_fcs(r.away_team, r.away_fbs),
                          _pool_conf(r.away_conf, r.away_fbs),
                          _pool_fcs(r.home_team, r.home_fbs),
                          _pool_conf(r.home_conf, r.home_fbs),
                          bool(r.home_fbs and r.away_fbs), int(r.week)),
        }

    conn = connectivity_table(hist, params)
    out: Dict[str, Dict[str, float]] = {}

    for phase in ("rush", "pass"):
        sub = units[units["phase"] == phase]
        for metric in ("eff", "exp"):
            recs = []
            for r in sub.itertuples():
                m = meta.get(int(r.game_id))
                if not m or r.offense not in m:
                    continue
                val = getattr(r, metric)
                if not np.isfinite(val):
                    continue
                team, tconf, opp, oconf, both_fbs, wk = m[r.offense]
                recs.append((team, tconf, opp, oconf, float(val), both_fbs, wk, int(r.n)))
            if len(recs) < 4:
                out[f"{phase}_off_{metric}"] = {}
                out[f"{phase}_def_{metric}"] = {}
                continue

            teams = sorted({r[0] for r in recs} | {r[2] for r in recs})
            confs = sorted({r[1] for r in recs} | {r[3] for r in recs})
            ti = {t: i for i, t in enumerate(teams)}
            ci = {c: i for i, c in enumerate(confs)}
            nT, nC = len(teams), len(confs)
            c_mu, c_co, c_cd = 0, 1, 1 + nC
            c_o, c_d = 1 + 2 * nC, 1 + 2 * nC + nT
            P = c_d + nT

            X = np.zeros((len(recs), P))
            y = np.zeros(len(recs))
            wt = np.zeros(len(recs))
            for i, (team, tconf, opp, oconf, val, both_fbs, wk, n) in enumerate(recs):
                X[i, c_mu] = 1.0
                X[i, c_co + ci[tconf]] = 1.0
                X[i, c_cd + ci[oconf]] = 1.0
                X[i, c_o + ti[team]] = 1.0
                X[i, c_d + ti[opp]] = 1.0
                y[i] = val
                base = math.exp(-params.decay * max(week - wk, 0))
                wt[i] = base * (1.0 if both_fbs else params.fcs_weight) * math.sqrt(n)

            spread = float(np.std(y)) or 1.0
            base_lam = params.lambda_team / max(spread ** 2, 1e-9) * 0.01
            mult = params.explosiveness_shrinkage if metric == "exp" else 1.0
            lam = np.zeros(P)
            for t in teams:
                s = conn.penalty_scale(t, params.conn_floor)
                lam[c_o + ti[t]] = base_lam * mult * s
                lam[c_d + ti[t]] = base_lam * mult * s
            for c in confs:
                lam[c_co + ci[c]] = params.lambda_conf / max(spread ** 2, 1e-9) * 0.01
                lam[c_cd + ci[c]] = params.lambda_conf / max(spread ** 2, 1e-9) * 0.01

            theta, _, _ = _ridge(X, y, wt, lam, np.zeros(P), params.max_condition)
            out[f"{phase}_off_{metric}"] = {
                t: round(float(theta[c_o + ti[t]]), ROUND) for t in teams}
            # defence: negative coefficient = holds opponents below average = good.
            # Sign-flipped on export so higher is better everywhere.
            out[f"{phase}_def_{metric}"] = {
                t: round(-float(theta[c_d + ti[t]]), ROUND) for t in teams}

    return {k: out.get(k, {}) for k in UNIT_NAMES}


# --------------------------------------------------------------------------- #
# Weekly refit + content-hash cache
# --------------------------------------------------------------------------- #
_SOLVE_CACHE: Dict[str, RatingsTable] = {}


def _cache_key(games: pd.DataFrame, season: int, week: int, params: RatingParams) -> str:
    hist = games[(games["season"] < season)
                 | ((games["season"] == season) & (games["week"] < week))]
    cols = ["id", "season", "week", "home_team", "away_team",
            "home_points", "away_points", "neutral_site", "home_fbs", "away_fbs"]
    blob = hist[cols].to_csv(index=False).encode()
    h = hashlib.sha256(blob).hexdigest()
    return hashlib.sha256((h + params.key()).encode()).hexdigest()


def solve_cached(games: pd.DataFrame, season: int, week: int,
                 params: RatingParams) -> RatingsTable:
    """Memoised weekly solve, keyed by the CONTENT of the games used.

    Content-hash, not `(season, week)` — if a score is corrected upstream the
    key changes and the cache misses, which is the whole point. A key built
    from the week number alone would happily serve a rating computed from data
    that has since been revised.
    """
    key = _cache_key(games, season, week, params)
    if key not in _SOLVE_CACHE:
        _SOLVE_CACHE[key] = solve(games, season, week, params)
    return _SOLVE_CACHE[key]


def clear_cache() -> None:
    _SOLVE_CACHE.clear()


def run_season(season: int, params: Optional[RatingParams] = None,
               games: Optional[pd.DataFrame] = None,
               first_week: int = 2) -> Dict[int, RatingsTable]:
    """Refit every week of a season. Week `w` sees only weeks < `w`."""
    params = params or RatingParams()
    games = game_frame([season]) if games is None else games
    weeks = sorted(int(w) for w in games.loc[games["season"] == season, "week"].unique())
    out = {}
    for w in weeks:
        if w < first_week:
            continue
        out[w] = solve_cached(games, season, w, params)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward parameter search — no constant is guessed
# --------------------------------------------------------------------------- #
def walk_forward_error(games: pd.DataFrame, season: int, params: RatingParams,
                       first_week: int = 4) -> Tuple[float, int]:
    """Mean absolute margin error predicting each week from the weeks before it.

    Games where either side is vetoed (or is FCS) are skipped, not guessed at —
    a model that refuses to price a game it cannot see is behaving correctly,
    and scoring it on those games would reward overconfidence.
    """
    weeks = sorted(int(w) for w in games.loc[games["season"] == season, "week"].unique())
    errs: List[float] = []
    for w in weeks:
        if w < first_week:
            continue
        try:
            tbl = solve_cached(games, season, w, params)
        except (RatingsError, np.linalg.LinAlgError):
            continue
        wk = games[(games["season"] == season) & (games["week"] == w)]
        for r in wk.itertuples():
            if not (r.home_fbs and r.away_fbs):
                continue
            try:
                pred = tbl.predict_margin(r.home_team, r.away_team, bool(r.neutral_site))
            except DisconnectedTeamError:
                continue
            errs.append(abs(pred - float(r.margin)))
    if not errs:
        return float("inf"), 0
    return float(np.mean(errs)), len(errs)


def walk_forward_unit_error(games: pd.DataFrame, units: pd.DataFrame, season: int,
                            unit: str, params: RatingParams,
                            first_week: int = 4) -> Tuple[float, int]:
    """MAE of a unit rating predicting the NEXT week's observed unit value.

    Exists because `walk_forward_error` cannot fit `explosiveness_shrinkage` at
    all — that parameter only touches `solve_units`, which the margin objective
    never calls. Left in the margin grid it looked fit while actually being
    chosen by tie-break, and the tie-break picked 1.0, i.e. the value that turns
    the explosiveness guardrail off. Found by inspecting the search trace and
    noticing every row for that parameter had an identical MAE.

    Prediction: observed(team, week) ~= league_mean + off_rating[team]
    + def_rating[opponent], with both ratings fit strictly before `week`.
    """
    if unit not in UNIT_NAMES:
        raise RatingsError(f"unknown unit {unit!r}")
    phase, side, metric = unit.split("_")
    metric = {"eff": "eff", "exp": "exp"}[metric]

    weeks = sorted(int(w) for w in games.loc[games["season"] == season, "week"].unique())
    errs: List[float] = []
    for w in weeks:
        if w < first_week:
            continue
        try:
            rated = solve_units(games, units, season, w, params)
        except (RatingsError, np.linalg.LinAlgError):
            continue
        off_r = rated[f"{phase}_off_{metric}"]
        def_r = rated[f"{phase}_def_{metric}"]
        if not off_r:
            continue
        prior_rows = units.merge(
            games.loc[games["season"] == season, ["id", "week"]],
            left_on="game_id", right_on="id", how="inner")
        seen = prior_rows[(prior_rows["week"] < w) & (prior_rows["phase"] == phase)]
        base = float(seen[metric].mean()) if len(seen) else 0.0
        this_week = prior_rows[(prior_rows["week"] == w) & (prior_rows["phase"] == phase)]
        for r in this_week.itertuples():
            o, d = str(r.offense), str(r.defense)
            if o not in off_r or d not in def_r:
                continue
            val = getattr(r, metric)
            if not np.isfinite(val):
                continue
            pred = base + off_r[o] - def_r[d]
            errs.append(abs(pred - float(val)))
    if not errs:
        return float("inf"), 0
    return float(np.mean(errs)), len(errs)


def unit_stickiness(games: pd.DataFrame, units: pd.DataFrame,
                    season: int, min_games: int = 6) -> pd.DataFrame:
    """Week-to-week lag-1 autocorrelation of each raw unit value, per team.

    This is the claim from `research/win-factors-literature-scan.md` measured
    directly instead of assumed: explosiveness is supposed to have near-zero
    week-to-week stickiness. Reported alongside the ratings because a unit with
    no stickiness has no business driving a projection, however good it looks
    as a season-long descriptor.

    NOT opponent-adjusted, and small-n: this is a diagnostic, not a finding.
    """
    m = units.merge(games.loc[games["season"] == season, ["id", "week"]],
                    left_on="game_id", right_on="id", how="inner")
    rows = []
    for phase in ("rush", "pass"):
        for metric in ("eff", "exp"):
            sub = m[m["phase"] == phase].sort_values(["offense", "week"])
            rs = []
            for _, grp in sub.groupby("offense", observed=True):
                x = grp[metric].to_numpy(dtype=float)
                if len(x) < min_games or not np.isfinite(x).all():
                    continue
                if np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
                    continue
                rs.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
            rows.append({"phase": phase, "metric": metric, "n_teams": len(rs),
                         "lag1_r": round(float(np.mean(rs)), 4) if rs else np.nan})
    return pd.DataFrame(rows)


def fit_explosiveness_shrinkage(games: pd.DataFrame, units: pd.DataFrame,
                                season: int, base: RatingParams,
                                values: Sequence[float] = (1.0, 2.0, 4.0, 8.0,
                                                           16.0, 32.0, 64.0, 128.0),
                                first_week: int = 4) -> Tuple[float, pd.DataFrame]:
    """Fit the explosiveness multiplier against next-week explosiveness itself."""
    rows = []
    for v in values:
        cand = replace(base, explosiveness_shrinkage=v)
        for unit in [u for u in UNIT_NAMES if u.endswith("_exp")]:
            err, n = walk_forward_unit_error(games, units, season, unit, cand,
                                             first_week)
            rows.append({"value": v, "unit": unit, "mae": err, "n": n})
    df = pd.DataFrame(rows)
    agg = df.groupby("value")["mae"].mean()
    return float(agg.idxmin()), df


#: The grid `fit_params` searches. Two of these ranges are DELIBERATELY
#: constrained and the constraint is a guardrail, not a convenience:
#:
#:   * `fcs_weight` stops at 0.5. An unconstrained search on the MWC 2023 slice
#:     picks 1.0 — it wants FCS results at full weight — for a 0.12-point MAE
#:     gain on 42 gradable games, i.e. noise. BUILD_SPEC and AGENT_BUILD_PROMPT
#:     both say FCS results must not enter the FBS fit at face value. The
#:     optimiser does not get to overrule that, and the preference is recorded
#:     here rather than quietly accommodated.
#:   * `prior_strength` stops at 20. Beyond that the "rating" is the preseason
#:     prior wearing a rating's clothes.
#:
#: `explosiveness_shrinkage` is NOT in this grid and that is deliberate: it has
#: no effect whatsoever on the margin objective, so including it produced a row
#: per candidate with an identical MAE and a tie-break "win" for 1.0 — the value
#: that switches the guardrail off. It is fit separately by
#: `fit_explosiveness_shrinkage`, against next-week explosiveness.
DEFAULT_GRID: Dict[str, Sequence[float]] = {
    "lambda_team": (0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
    "lambda_conf": (0.25, 0.5, 1, 2, 4),
    "fcs_weight": (0.0, 0.1, 0.25, 0.5),
    "prior_strength": (0, 3, 6, 9, 12, 16, 20),
    "decay": (0.0, 0.02, 0.05, 0.1, 0.2),
    "lambda_hfa": (10, 40, 100, 400),
}


def params_path(root: Optional[str] = None) -> str:
    import os
    root = root or ingest.ROOT
    return os.path.join(root, "data", "rating_params.json")


def save_params(params: RatingParams, path: Optional[str] = None) -> str:
    import os
    path = path or params_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=2, sort_keys=True)
    return path


def load_params(path: Optional[str] = None) -> RatingParams:
    """Load fitted parameters. Raises if they were never fit — by design.

    Silently falling back to the dataclass defaults would let an unfit model
    ship looking exactly like a fit one.
    """
    import os
    path = path or params_path()
    if not os.path.exists(path):
        raise RatingsError(
            f"no fitted rating parameters at {path}. Run `python build_ratings.py "
            "--fit` first; the RatingParams defaults are search starting points, "
            "not fitted values, and must not be used as if they were.")
    with open(path) as f:
        raw = json.load(f)
    known = {f_.name for f_ in RatingParams.__dataclass_fields__.values()}
    return RatingParams(**{k: v for k, v in raw.items() if k in known})


def walk_forward_search(games: pd.DataFrame, season: int,
                        grid: Dict[str, Sequence[float]],
                        base: Optional[RatingParams] = None,
                        first_week: int = 4) -> pd.DataFrame:
    """Coordinate-wise walk-forward grid search over `grid`.

    Coordinate-wise (one parameter at a time, keeping the best so far) rather
    than a full product: the full grid over six parameters is thousands of
    season refits, and with a single 98-game season the extra resolution would
    be fitting noise. Reported honestly as what it is.
    """
    params = base or RatingParams()
    rows = []
    for name, values in grid.items():
        best_val, best_err = getattr(params, name), float("inf")
        for v in values:
            cand = replace(params, **{name: v})
            err, n = walk_forward_error(games, season, cand, first_week)
            rows.append({"param": name, "value": v, "mae": err, "n": n})
            if err < best_err:
                best_err, best_val = err, v
        params = replace(params, **{name: best_val})
    df = pd.DataFrame(rows)
    df.attrs["best"] = params
    return df
