"""Drive-by-drive Monte Carlo game simulator.

Given two teams' power ratings (points off/def) and league priors, we simulate a
game thousands of times, one possession at a time, scoring touchdowns (6-8),
field goals (3), and the occasional defensive score. Summing real integer scores
reproduces the spiky NFL margin distribution (the pushes on 3 and 7 that a smooth
bell curve misses), which matters for spreads and totals.

From the simulated score distribution we read off:
    P(home win), P(home covers a spread), P(over a total), and percentiles.

Uses numpy for speed (a backtest runs ~1,400 games x thousands of sims). If numpy
isn't installed, the rest of the app still works — MC features just switch off.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

DMAX = 15           # max possessions per team simulated
BASE_OFF_PPD = 1.971  # league points/drive a baseline offense scores itself


def expected_points(home: Dict, away: Dict, priors: Dict):
    """Expected points for each team from ratings (off/def in points/game)."""
    lg = priors["league_ppg"]
    hfa = priors.get("hfa_points", 1.5)
    exp_home = lg + home["off"] - away["def"] + hfa / 2.0
    exp_away = lg + away["off"] - home["def"] - hfa / 2.0
    return max(exp_home, 3.0), max(exp_away, 3.0)


def _sim_team(rng, n, target_ppd, drives, priors):
    """Simulate `drives[i]` possessions for `n` games. Returns (off_pts, pts_given)."""
    do = priors["drive_outcomes"]
    m = float(np.clip(target_ppd / BASE_OFF_PPD, 0.30, 2.5))
    p_td = min(do["td"] * m, 0.55)
    p_fg = min(do["fg"] * m, 0.32)
    p_dtd = do["def_td"]          # this offense turns it over for a defensive TD
    p_saf = do["safety"]

    u = rng.random((n, DMAX))
    v = rng.random((n, DMAX))
    td = u < p_td
    fg = (u >= p_td) & (u < p_td + p_fg)
    dtd = (u >= p_td + p_fg) & (u < p_td + p_fg + p_dtd)
    saf = (u >= p_td + p_fg + p_dtd) & (u < p_td + p_fg + p_dtd + p_saf)

    td_pts = 6 + np.where(v < 0.92, 1, np.where(v < 0.97, 0, 2))  # XP / miss / 2pt
    off = td * td_pts + fg * 3
    given = dtd * 7 + saf * 2

    mask = np.arange(DMAX)[None, :] < drives[:, None]
    return (off * mask).sum(axis=1), (given * mask).sum(axis=1)


def simulate(home: Dict, away: Dict, priors: Dict,
             spread_line: Optional[float] = None, total_line: Optional[float] = None,
             n: int = 20000, seed: Optional[int] = None) -> Dict:
    """Simulate a matchup `n` times.

    spread_line: home favored by this many (nflfastR convention; home covers if
                 final margin > spread_line). total_line: points for over/under.
    """
    rng = np.random.default_rng(seed)
    exp_home, exp_away = expected_points(home, away, priors)

    dmean = priors.get("drives_mean", 11.2)
    dsd = max(priors.get("drives_sd", 1.7) * 0.6, 0.5)
    drives = np.clip(np.round(rng.normal(dmean, dsd, n)), 8, DMAX).astype(int)

    # Each team also scores defensive/ST points off the opponent (~1.9/game), so
    # the offense must target the rest to keep the total calibrated.
    do = priors["drive_outcomes"]
    def_pts = dmean * (do["def_td"] * 7 + do["safety"] * 2)
    target_home = max(exp_home - def_pts, 3.0) / dmean
    target_away = max(exp_away - def_pts, 3.0) / dmean

    h_off, h_give = _sim_team(rng, n, target_home, drives, priors)
    a_off, a_give = _sim_team(rng, n, target_away, drives, priors)
    home_score = h_off + a_give
    away_score = a_off + h_give

    margin = home_score - away_score          # >0 = home wins
    total = home_score + away_score

    out = {
        "exp_home": round(exp_home, 2), "exp_away": round(exp_away, 2),
        "home_mean": round(float(home_score.mean()), 1),
        "away_mean": round(float(away_score.mean()), 1),
        "margin_mean": round(float(margin.mean()), 2),
        "margin_sd": round(float(margin.std()), 2),
        "total_mean": round(float(total.mean()), 1),
        "total_sd": round(float(total.std()), 2),
        "p_home_win": round(float((margin > 0).mean()), 4),
        "p_away_win": round(float((margin < 0).mean()), 4),
        "median_home": int(np.median(home_score)),
        "median_away": int(np.median(away_score)),
        "n": n,
    }
    if spread_line is not None:
        out["spread_line"] = spread_line
        out["p_home_cover"] = round(float((margin > spread_line).mean()), 4)
        out["p_away_cover"] = round(float((margin < spread_line).mean()), 4)
        out["p_push_spread"] = round(float((margin == spread_line).mean()), 4)
    if total_line is not None:
        out["total_line"] = total_line
        out["p_over"] = round(float((total > total_line).mean()), 4)
        out["p_under"] = round(float((total < total_line).mean()), 4)
    # compact margin histogram (home perspective) for the dashboard
    bins = np.arange(-35, 36, 5)
    hist, _ = np.histogram(np.clip(margin, -34, 34), bins=bins)
    out["margin_hist"] = {"bins": bins[:-1].tolist(), "counts": (hist / n).round(4).tolist()}
    return out


def fair_moneyline_odds(p_win: float) -> int:
    """Convert a win probability to fair American odds (no vig)."""
    p_win = min(max(p_win, 1e-3), 1 - 1e-3)
    dec = 1.0 / p_win
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100.0 / (dec - 1)))
