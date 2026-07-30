"""Workstream C — drive-level CFB Monte Carlo with matchup-conditional rates.

Why this is a NEW module and not a rework of `nflvalue/montecarlo.py`: the NFL
parent's engine has NFL magnitudes baked in as literals (BASE_OFF_PPD=1.971,
92% XP, DMAX=15, a hand-set TD-probability cap) and BUILD_SPEC section 0
forbids inheriting NFL constants — CFB numbers must be measured, not assumed.
Every rate here is FIT, walk-forward, from the CFBD drives cache. The old
module stays untouched as reference (vault files are never deleted).

The model
---------
A game is a sequence of drives. For each simulated game we:

  1. sample a (home_drives, away_drives) pair from the EMPIRICAL joint
     distribution of regulation drive counts (they correlate ~0.84 — sampling
     sides independently would overstate pace variance);
  2. for every drive, sample a joint outcome (offense_points,
     defense_points) — 0/3/6/7/8 for the offense, 0/2/6/7/8 for the defense
     (pick-sixes, safeties) — from rates conditioned on the offense's
     RATING-DERIVED EDGE for that game. This is the matchup-conditional
     requirement of BUILD_SPEC section 4: a single league-wide rate would
     under-simulate blowouts, and CFB's talent spread makes that fatal;
  3. if regulation ends tied, run the 2021+ CFB overtime module
     (`nflvalue/cfb_ot.py`). College games cannot tie.

The offense edge `e` is in margin points: for the home side
    e_home = off[home] + def_[away] + hfa(home)/2
and symmetrically (minus half HFA) for the away side, so that
    e_home - e_away == RatingsTable.predict_margin(home, away).

Drive-outcome rates are fit on ALL regulation FBS-vs-FBS drives INCLUDING
garbage time — deliberately inverted from the ratings layer's clean-plays
rule, and worth being precise about why: ratings estimate latent strength, so
garbage possessions are noise there; this layer reproduces REALIZED score
distributions, and garbage-time dynamics are part of how real blowouts
finish. The e-conditioning puts those drives in the mismatch bins where they
actually occur, which is exactly the composition the simulator must emit.

Walk-forward discipline: `walk_forward_backtest` fits the rate model at week
W from drives strictly before W (labelled with each drive's OWN-WEEK rating
table, never a later one), predicts week W, and only then labels week W's
drives and adds them to the training pool. `tests/test_leakage.py` poisons
this loop; keep it that way.

Full distributions are the output — margin and total arrays, not point
estimates. Uncertainty is a first-class product per non-negotiable #7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import ratings as R
from .cfb_ot import OTParams, simulate_ot

#: Joint drive outcomes: (offense_points, defense_points). Every cell was
#: observed in the 2019-2025 cache; anything else (~0.7% of drives, mostly
#: multi-score data glitches like a 14) is dropped AND counted by the
#: classifier, never coerced into a neighbouring class.
OFF_PTS = (0, 3, 6, 7, 8)
DEF_PTS = (0, 2, 6, 7, 8)
CLASSES: List[Tuple[int, int]] = [(o, d) for o in OFF_PTS for d in DEF_PTS]
_CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


class SimError(RuntimeError):
    pass


def classify_drives(drives: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Score-delta classification. Returns (classified, n_dropped).

    Deltas are used instead of the `drive_result` strings on purpose: the
    string vocabulary has 20+ variants ('PUNT TD', 'MISSED FG TD', ...) whose
    meanings would have to be guessed, while end-minus-start scores are the
    points that actually went on the board, XP misses and 2-pt tries included.
    """
    d = drives.copy()
    d["off_pts"] = (d["end_offense_score"] - d["start_offense_score"])
    d["def_pts"] = (d["end_defense_score"] - d["start_defense_score"])
    ok = d["off_pts"].isin(OFF_PTS) & d["def_pts"].isin(DEF_PTS)
    dropped = int((~ok).sum())
    d = d.loc[ok].copy()
    d["off_pts"] = d["off_pts"].astype(int)
    d["def_pts"] = d["def_pts"].astype(int)
    return d, dropped


@dataclass
class DriveRateModel:
    """P(joint drive outcome | offense edge e), fit as quantile bins over e
    with linear interpolation between bin centres.

    Binned rather than parametric so the data — not a chosen link function —
    decides how blowout drive mixes differ from even-game ones (BUILD_SPEC 4
    allows either). `n_bins`/`min_bin_n` are structural resolution choices,
    not tuned constants: they were not searched against any accuracy metric.
    """
    bin_centers: np.ndarray = field(default_factory=lambda: np.zeros(0))
    probs: np.ndarray = field(default_factory=lambda: np.zeros((0, len(CLASSES))))
    n_fit: int = 0
    n_dropped: int = 0

    @classmethod
    def fit(cls, labeled: pd.DataFrame, n_bins: int = 8,
            min_bin_n: int = 400) -> "DriveRateModel":
        """`labeled` needs columns e, off_pts, def_pts (already classified)."""
        if labeled.empty:
            raise SimError("no labelled drives to fit on")
        e = labeled["e"].to_numpy(dtype=float)
        cls_idx = np.array([_CLASS_INDEX[(o, d)] for o, d in
                            zip(labeled["off_pts"], labeled["def_pts"])])
        n_bins = max(1, min(n_bins, len(e) // max(min_bin_n, 1)))
        if n_bins < 2:
            raise SimError(
                f"only {len(e)} labelled drives — not enough for even 2 bins "
                f"of {min_bin_n}; refusing to fit a 'matchup-conditional' "
                "model that is actually unconditional")
        qs = np.quantile(e, np.linspace(0, 1, n_bins + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        which = np.clip(np.searchsorted(qs, e, side="right") - 1, 0, n_bins - 1)
        centers, probs = [], []
        for b in range(n_bins):
            m = which == b
            if m.sum() == 0:
                continue
            counts = np.bincount(cls_idx[m], minlength=len(CLASSES)).astype(float)
            centers.append(float(e[m].mean()))
            probs.append(counts / counts.sum())
        return cls(bin_centers=np.array(centers), probs=np.array(probs),
                   n_fit=int(len(e)), n_dropped=0)

    def class_probs(self, e: float) -> np.ndarray:
        """Interpolated joint-class probabilities at edge e (clamped at ends)."""
        c = self.bin_centers
        if len(c) == 0:
            raise SimError("model not fit")
        if e <= c[0]:
            return self.probs[0]
        if e >= c[-1]:
            return self.probs[-1]
        j = int(np.searchsorted(c, e))
        w = (e - c[j - 1]) / (c[j] - c[j - 1])
        p = (1 - w) * self.probs[j - 1] + w * self.probs[j]
        return p / p.sum()

    def p_td(self, e: float) -> float:
        p = self.class_probs(e)
        return float(sum(p[i] for i, (o, _) in enumerate(CLASSES) if o >= 6))

    def expected_net_pts(self, e: float) -> float:
        """E[offense - defense points] per drive at edge e (diagnostic)."""
        p = self.class_probs(e)
        return float(sum(p[i] * (o - d) for i, (o, d) in enumerate(CLASSES)))


@dataclass
class PaceModel:
    """Empirical joint (home_drives, away_drives) pairs; sampled whole so the
    ~0.84 correlation between the sides survives into the simulation."""
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))

    @classmethod
    def fit(cls, drives: pd.DataFrame) -> "PaceModel":
        reg = drives[drives["start_period"] <= 4]
        cnt = (reg.groupby(["game_id", "is_home_offense"]).size()
               .unstack(fill_value=0))
        if cnt.empty or True not in cnt.columns or False not in cnt.columns:
            raise SimError("cannot fit pace: no two-sided games")
        # NB: cnt[[True, False]] would be read by pandas as a row MASK, not a
        # column pick, because the column labels are booleans. Pick as Series.
        pairs = np.stack([cnt[True].to_numpy(dtype=int),
                          cnt[False].to_numpy(dtype=int)], axis=1)  # (home, away)
        pairs = pairs[(pairs[:, 0] >= 6) & (pairs[:, 1] >= 6)]  # broken games out
        if len(pairs) < 50:
            raise SimError(f"only {len(pairs)} games with drive counts")
        return cls(pairs=pairs)

    def sample(self, rng, n: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self.pairs), n)
        return self.pairs[idx, 0], self.pairs[idx, 1]


# --------------------------------------------------------------------------- #
# Labelling: attach each drive's own-week rating edge
# --------------------------------------------------------------------------- #
def label_week_drives(drives_wk: pd.DataFrame, games_wk: pd.DataFrame,
                      tbl: "R.RatingsTable") -> pd.DataFrame:
    """Label one week's FBS-vs-FBS drives with the offense edge e implied by
    that week's OWN rating table (which saw only earlier games). Drives in
    games with a vetoed/unratable side are skipped, not guessed at."""
    meta = {int(r.id): (r.home_team, r.away_team, bool(r.neutral_site))
            for r in games_wk.itertuples()
            if bool(r.home_fbs) and bool(r.away_fbs)}
    out = []
    for r in drives_wk.itertuples():
        gm = meta.get(int(r.game_id))
        if gm is None:
            continue
        home, away, neutral = gm
        if tbl.veto.get(home, True) or tbl.veto.get(away, True):
            continue
        offense = home if bool(r.is_home_offense) else away
        defense = away if bool(r.is_home_offense) else home
        half_hfa = tbl.hfa(home, neutral) / 2.0
        e = (tbl.off.get(offense, 0.0) + tbl.def_.get(defense, 0.0)
             + (half_hfa if offense == home else -half_hfa))
        out.append({"e": e, "off_pts": int(r.off_pts), "def_pts": int(r.def_pts),
                    "season": int(r.season), "week": int(r.week)})
    return pd.DataFrame(out)


def side_edges(tbl: "R.RatingsTable", home: str, away: str,
               neutral: bool) -> Tuple[float, float]:
    """(e_home, e_away) such that e_home - e_away == tbl.predict_margin(...)."""
    tbl.require_usable(home)
    tbl.require_usable(away)
    half_hfa = tbl.hfa(home, neutral) / 2.0
    e_home = tbl.off.get(home, 0.0) + tbl.def_.get(away, 0.0) + half_hfa
    e_away = tbl.off.get(away, 0.0) + tbl.def_.get(home, 0.0) - half_hfa
    return e_home, e_away


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
_OFF_LOOKUP = np.array([o for o, _ in CLASSES], dtype=float)
_DEF_LOOKUP = np.array([d for _, d in CLASSES], dtype=float)


def _sample_side(rng, n: int, max_d: int, counts: np.ndarray,
                 probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (off_points_total, def_points_total) for one side's drives."""
    cum = np.cumsum(probs)
    cum[-1] = 1.0
    idx = np.searchsorted(cum, rng.random((n, max_d)), side="right")
    idx = np.clip(idx, 0, len(CLASSES) - 1)
    mask = np.arange(max_d)[None, :] < counts[:, None]
    off = (_OFF_LOOKUP[idx] * mask).sum(axis=1)
    dfn = (_DEF_LOOKUP[idx] * mask).sum(axis=1)
    return off, dfn


def simulate_game(rng, e_home: float, e_away: float, rates: DriveRateModel,
                  pace: PaceModel, ot: OTParams, n: int = 4000) -> Dict:
    """Simulate one matchup n times. Returns full margin/total arrays plus a
    summary. Margins are home-minus-away and never zero (OT resolves ties)."""
    nh, na = pace.sample(rng, n)
    max_d = int(max(nh.max(), na.max()))
    p_home = rates.class_probs(e_home)
    p_away = rates.class_probs(e_away)
    h_off, h_gave = _sample_side(rng, n, max_d, nh, p_home)
    a_off, a_gave = _sample_side(rng, n, max_d, na, p_away)
    home_score = h_off + a_gave
    away_score = a_off + h_gave
    margin = home_score - away_score
    total = home_score + away_score

    tied = margin == 0
    if tied.any():
        base = max(rates.p_td(0.0), 1e-6)
        ot_m, ot_pts = simulate_ot(rng, int(tied.sum()), ot,
                                   tilt_home=rates.p_td(e_home) / base,
                                   tilt_away=rates.p_td(e_away) / base)
        margin = margin.copy()
        total = total.copy()
        margin[tied] = ot_m
        total[tied] = total[tied] + ot_pts

    return {
        "margin": margin, "total": total,
        "margin_mean": float(margin.mean()), "margin_sd": float(margin.std()),
        "total_mean": float(total.mean()), "total_sd": float(total.std()),
        "p_home_win": float((margin > 0).mean()),
        "p_ot": float(tied.mean()),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Walk-forward backtest driver (gate 3)
# --------------------------------------------------------------------------- #
def walk_forward_backtest(games: pd.DataFrame, drives: pd.DataFrame,
                          params: "R.RatingParams", ot: OTParams,
                          eval_seasons: Iterable[int],
                          burn_in_seasons: Iterable[int] = (),
                          first_week: int = 4, n_sims: int = 4000,
                          min_train_drives: int = 5000,
                          n_bins: int = 8, min_bin_n: int = 400,
                          seed: int = 20260730,
                          progress=None) -> pd.DataFrame:
    """Walk every week in chronological order; fit rates on the past only.

    For each eval week W: fit DriveRateModel on all drives labelled so far
    (all strictly before W), simulate W's FBS-vs-FBS games, THEN label W's
    drives with W's own table and add them to the pool. Burn-in seasons are
    walked for labelling only. Returns one row per simulated game.
    """
    drives_c, _ = classify_drives(drives)
    drives_c = drives_c[drives_c["start_period"] <= 4]
    pace = PaceModel.fit(drives_c)

    all_seasons = sorted(set(int(s) for s in burn_in_seasons)
                         | set(int(s) for s in eval_seasons))
    eval_set = set(int(s) for s in eval_seasons)
    pool: List[pd.DataFrame] = []
    pool_n = 0
    rows = []
    rng = np.random.default_rng(seed)

    for season in all_seasons:
        weeks = sorted(int(w) for w in
                       games.loc[games["season"] == season, "week"].unique())
        for w in weeks:
            try:
                tbl = R.solve_cached(games, season, w, params)
            except (R.RatingsError, np.linalg.LinAlgError):
                continue
            gw = games[(games["season"] == season) & (games["week"] == w)]
            dw = drives_c[(drives_c["season"] == season)
                          & (drives_c["week"] == w)]
            if (season in eval_set and w >= first_week
                    and pool_n >= min_train_drives):
                train = pd.concat(pool, ignore_index=True)
                rates = DriveRateModel.fit(train, n_bins=n_bins,
                                           min_bin_n=min_bin_n)
                for g in gw.itertuples():
                    if not (g.home_fbs and g.away_fbs):
                        continue
                    try:
                        e_h, e_a = side_edges(tbl, g.home_team, g.away_team,
                                              bool(g.neutral_site))
                    except R.DisconnectedTeamError:
                        continue
                    sim = simulate_game(rng, e_h, e_a, rates, pace, ot,
                                        n=n_sims)
                    rows.append({
                        "season": season, "week": w, "id": int(g.id),
                        "home": g.home_team, "away": g.away_team,
                        "pred_b": tbl.predict_margin(
                            g.home_team, g.away_team, bool(g.neutral_site)),
                        "mc_margin": sim["margin_mean"],
                        "mc_margin_sd": sim["margin_sd"],
                        "mc_total": sim["total_mean"],
                        "mc_total_sd": sim["total_sd"],
                        "p_home_win": sim["p_home_win"],
                        "p_ot": sim["p_ot"],
                        "margin": float(g.margin),
                        "total": float(g.home_points + g.away_points),
                        "n_train_drives": pool_n,
                    })
            labeled = label_week_drives(dw, gw, tbl)
            if len(labeled):
                pool.append(labeled)
                pool_n += len(labeled)
            if progress is not None:
                progress(season, w, pool_n, len(rows))
    return pd.DataFrame(rows)
