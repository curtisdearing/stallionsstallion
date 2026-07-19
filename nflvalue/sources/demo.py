"""Realistic demo data so the whole system works with zero API keys.

It mimics what the live feed looks like: several games, 5-6 books with slightly
different lines (so line-shopping and mispriced outliers appear), weather,
injuries, revenge spots, and player props.

Crucially, ``simulate_results`` plays the games out using a HIDDEN "true" model
in which the situational factors genuinely matter. That gives the learning loop
real signal: over simulated weeks, the factor weights drift toward the truth and
the dashboard's calibration / ROI improve. It's synthetic, but it demonstrates
exactly how the live system learns from finished games.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from .. import factors as factmod
from .. import oddsmath

BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars", "bovada"]

TEAMS = list(factmod.STADIUMS.keys())

# Hidden power ratings (points better/worse than average). Demo-only.
POWER = {
    "Kansas City Chiefs": 6.5, "Philadelphia Eagles": 6.0, "Buffalo Bills": 5.5,
    "Baltimore Ravens": 5.5, "San Francisco 49ers": 5.0, "Detroit Lions": 4.5,
    "Cincinnati Bengals": 3.5, "Green Bay Packers": 3.5, "Dallas Cowboys": 2.5,
    "Houston Texans": 2.5, "Miami Dolphins": 2.0, "Los Angeles Rams": 2.0,
    "Minnesota Vikings": 1.5, "Tampa Bay Buccaneers": 1.0, "Pittsburgh Steelers": 1.0,
    "Los Angeles Chargers": 0.5, "Seattle Seahawks": 0.5, "Jacksonville Jaguars": 0.0,
    "Indianapolis Colts": -0.5, "Atlanta Falcons": -0.5, "New York Jets": -1.0,
    "Chicago Bears": -1.0, "Denver Broncos": -1.5, "Cleveland Browns": -2.0,
    "Arizona Cardinals": -2.5, "New Orleans Saints": -2.5, "Tennessee Titans": -3.0,
    "Las Vegas Raiders": -3.5, "Washington Commanders": -3.5,
    "New York Giants": -4.5, "New England Patriots": -4.5, "Carolina Panthers": -5.5,
}

# A couple of star players per team for player props (name, position, type, mean).
STARS = {
    "Kansas City Chiefs": [("P. Mahomes", "QB", "pass_yds", 280), ("X. Worthy", "WR", "reception_yds", 64)],
    "Buffalo Bills": [("J. Allen", "QB", "pass_yds", 255), ("J. Cook", "RB", "rush_yds", 78)],
    "Philadelphia Eagles": [("J. Hurts", "QB", "pass_yds", 230), ("S. Barkley", "RB", "rush_yds", 98)],
    "Baltimore Ravens": [("L. Jackson", "QB", "pass_yds", 235), ("D. Henry", "RB", "rush_yds", 95)],
    "San Francisco 49ers": [("B. Purdy", "QB", "pass_yds", 250), ("C. McCaffrey", "RB", "rush_yds", 88)],
    "Detroit Lions": [("J. Goff", "QB", "pass_yds", 265), ("J. Gibbs", "RB", "rush_yds", 82)],
    "Cincinnati Bengals": [("J. Burrow", "QB", "pass_yds", 275), ("J. Chase", "WR", "reception_yds", 86)],
    "Green Bay Packers": [("J. Love", "QB", "pass_yds", 245), ("J. Jacobs", "RB", "rush_yds", 80)],
    "Dallas Cowboys": [("D. Prescott", "QB", "pass_yds", 260), ("C. Lamb", "WR", "reception_yds", 84)],
    "Miami Dolphins": [("T. Tagovailoa", "QB", "pass_yds", 255), ("T. Hill", "WR", "reception_yds", 82)],
    "Houston Texans": [("C. Stroud", "QB", "pass_yds", 250), ("N. Collins", "WR", "reception_yds", 78)],
}
GENERIC_STARS = [("QB1", "QB", "pass_yds", 235), ("RB1", "RB", "rush_yds", 72),
                 ("WR1", "WR", "reception_yds", 68)]

# The hidden truth the learner is trying to recover.
TRUE_WEIGHTS = {
    "g_injury_diff": 0.65, "g_revenge": 0.18, "g_rest_diff": 0.35,
    "g_matchup": 0.70, "g_home_field": 0.22,
    "t_weather": 0.85, "t_injuries": 0.30, "t_pace": 0.45,
    "p_usage_trend": 0.60, "p_opp_defense": 0.60,
    "p_weather_pass": 0.75, "p_weather_rush": 0.35,
}

# How much of each situational factor the market already prices in.
# 0 = market ignores factors (huge edges); 1 = perfectly efficient (no edge).
# Real books are very efficient (~0.9+). The demo leaves a deliberately larger
# exploitable residual so you can SEE the learner find it and improve ROI;
# raise this toward 0.9 for a more realistic (and humbling) simulation.
MARKET_EFFICIENCY = 0.62


def _g_effect(ctx) -> float:
    """True effect on home margin, in points (+ favors home)."""
    feat = factmod.game_side_features(ctx, "home")
    return 6.0 * sum(TRUE_WEIGHTS.get(k, 0) * v for k, v in feat.items())


def _t_effect(ctx) -> float:
    """True effect on the game total, in points (+ raises total)."""
    feat = factmod.total_features(ctx, "over")
    return 7.0 * sum(TRUE_WEIGHTS.get(k, 0) * v for k, v in feat.items())


def _p_effect(pctx, ptype) -> float:
    """True fractional effect on a player's prop mean."""
    pf = factmod.prop_features(pctx, ptype, "over")
    return 0.18 * sum(TRUE_WEIGHTS.get(k, 0) * v for k, v in pf.items())


def _half(x: float) -> float:
    return round(x * 2) / 2.0


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _two_way_decimals(p_a: float, margin_per_side: float, rng: random.Random):
    """Fair prob -> a book's quoted decimals for both sides (with vig + noise)."""
    p_a = min(max(p_a, 0.03), 0.97)
    qa = p_a * (1 + margin_per_side) * (1 + rng.uniform(-0.02, 0.02))
    qb = (1 - p_a) * (1 + margin_per_side) * (1 + rng.uniform(-0.02, 0.02))
    return round(1 / qa, 3), round(1 / qb, 3)


def _rand_injuries(rng: random.Random) -> List[Dict]:
    injs = []
    for _ in range(rng.randint(0, 3)):
        pos = rng.choice(["WR", "RB", "CB", "S", "LB", "OT", "QB", "TE", "EDGE"])
        status = rng.choice(["questionable", "questionable", "doubtful", "out"])
        injs.append({"position": pos, "status": status, "name": f"{pos} starter"})
    return injs


def _make_context(home: str, away: str, rng: random.Random) -> Dict:
    st = factmod.STADIUMS.get(home, {"dome": True})
    if st.get("dome"):
        weather = {"dome": True}
    else:
        rough = rng.random() < 0.35
        weather = {
            "dome": False,
            "wind_mph": round(rng.uniform(12, 32), 1) if rough else round(rng.uniform(0, 10), 1),
            "precip_mm": round(rng.uniform(1, 7), 1) if rough and rng.random() < 0.6 else 0.0,
            "temp_f": round(rng.uniform(10, 35)) if rough and rng.random() < 0.5 else round(rng.uniform(45, 75)),
        }
    sev = factmod.weather_severity(weather)
    inj_home = _rand_injuries(rng)
    inj_away = _rand_injuries(rng)
    revenge_home = 1.0 if rng.random() < 0.18 else 0.0
    revenge_away = 1.0 if rng.random() < 0.18 else 0.0
    matchup_edge = (POWER.get(home, 0) - POWER.get(away, 0)) / 12.0
    return {
        "weather": weather,
        "weather_sev": sev,
        "inj_home": factmod.injury_severity(inj_home),
        "inj_away": factmod.injury_severity(inj_away),
        "injuries_home": inj_home,
        "injuries_away": inj_away,
        "revenge_home": revenge_home,
        "revenge_away": revenge_away,
        "rest_home": rng.choice([7, 7, 7, 10, 6]),
        "rest_away": rng.choice([7, 7, 7, 10, 6, 4]),
        "matchup_edge": round(matchup_edge, 4),
        "pace_edge": round(rng.uniform(-0.4, 0.4), 4),
    }


def _make_props(home, away, ctx, rng):
    props = []
    for team in (home, away):
        roster = STARS.get(team, GENERIC_STARS)
        for (name, pos, ptype, mean) in roster:
            pctx = {
                "weather_sev": ctx["weather_sev"],
                "opp_def_rating": round(rng.uniform(-0.6, 0.6), 3),
                "usage_trend": round(rng.uniform(-0.6, 0.6), 3),
            }
            frac = _p_effect(pctx, ptype)
            truth_mean = mean * (1 + frac)
            line = _half(mean * (1 + MARKET_EFFICIENCY * frac))  # book prices most of it
            book_prices = {}
            for bk in BOOKS:
                da, db = _two_way_decimals(0.5, 0.045, rng)  # priced ~50/50 at the line
                book_prices[bk] = {"over": da, "under": db}
            if rng.random() < 0.5:  # soft price outlier -> a line-shopping edge
                bk = rng.choice(BOOKS)
                sd = rng.choice(["over", "under"])
                book_prices[bk][sd] = round(book_prices[bk][sd] * rng.uniform(1.02, 1.05), 3)
            label = {"pass_yds": "Pass Yds", "rush_yds": "Rush Yds",
                     "reception_yds": "Rec Yds"}.get(ptype, ptype)
            props.append({
                "player": f"{name} ({team.split()[-1]})",
                "team": team, "prop_type": ptype, "label": label,
                "line": line, "books": book_prices, "ctx": pctx,
                "_mean": mean, "_truth_mean": truth_mean,
            })
    rng.shuffle(props)
    return props[:rng.randint(3, 6)]


def generate_slate(base_dt: datetime, n_games: int = 11, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    teams = TEAMS[:]
    rng.shuffle(teams)
    games = []
    for i in range(min(n_games, len(teams) // 2)):
        home, away = teams[2 * i], teams[2 * i + 1]
        ctx = _make_context(home, away, rng)

        base_margin = POWER.get(home, 0) - POWER.get(away, 0) + 2.0  # +2 home field
        base_total = _half(rng.uniform(40, 50))
        e_margin = _g_effect(ctx)            # true factor effect on margin
        e_total = _t_effect(ctx)             # true factor effect on total

        # The market prices in MOST of the factor effect; a small residual remains.
        posted_margin = base_margin + MARKET_EFFICIENCY * e_margin
        posted_total = base_total + MARKET_EFFICIENCY * e_total
        spread_home = _half(-posted_margin)
        truth = {"exp_margin": base_margin + e_margin,     # full truth for simulation
                 "exp_total": base_total + e_total}

        p_home_ml = oddsmath.sigmoid(posted_margin / 7.0)  # ML priced off posted line
        commence = base_dt + timedelta(days=rng.choice([0, 1, 2, 3]),
                                       hours=rng.choice([13, 16, 20]))
        gid = f"demo-{seed}-{i}-{home.split()[-1]}-{away.split()[-1]}"

        h2h, spreads, totals = {}, {}, {}
        for bk in BOOKS:
            mh, ma = _two_way_decimals(p_home_ml, 0.025, rng)
            h2h[bk] = {"home": mh, "away": ma}
            sh, sa = _two_way_decimals(0.5, 0.045, rng)
            spreads[bk] = {"home": {"point": spread_home, "price": sh},
                           "away": {"point": -spread_home, "price": sa}}
            oh, ou = _two_way_decimals(0.5, 0.045, rng)
            totals[bk] = {"over": {"point": posted_total, "price": oh},
                          "under": {"point": posted_total, "price": ou}}
        bk = rng.choice(BOOKS)  # inject a soft outlier so line-shopping shows value
        side = rng.choice(["home", "away"])
        h2h[bk][side] = round(h2h[bk][side] * rng.uniform(1.02, 1.05), 3)

        games.append({
            "id": gid,
            "commence_time": commence.replace(tzinfo=timezone.utc).isoformat(),
            "home_team": home, "away_team": away,
            "books": {"h2h": h2h, "spreads": spreads, "totals": totals},
            "context": ctx,
            "props": _make_props(home, away, ctx, rng),
            "_truth": truth,
        })
    return games


def simulate_results(games: List[Dict], seed: int = 0) -> Dict[str, Dict]:
    """Play the games out under the HIDDEN true model and return results."""
    rng = random.Random(seed * 7919 + 13)
    results = {}
    for g in games:
        truth = g.get("_truth", {})
        exp_margin = truth.get("exp_margin", 0.0)   # full hidden truth (incl. residual)
        exp_total = truth.get("exp_total", 44.0)

        margin = rng.gauss(exp_margin, 10.5)        # NFL margins ~ sd 10-13
        total = max(20.0, rng.gauss(exp_total, 9.0))
        home_score = int(round(_clip((total + margin) / 2.0, 0, 80)))
        away_score = int(round(_clip((total - margin) / 2.0, 0, 80)))

        prop_actuals = {}
        for p in g.get("props", []):
            mean = p.get("_truth_mean", p.get("_mean", p["line"]))
            actual = max(0.0, rng.gauss(mean, p.get("_mean", mean) * 0.20))
            prop_actuals[p["player"]] = round(actual, 1)

        results[g["id"]] = {
            "home_score": home_score, "away_score": away_score,
            "prop_actuals": prop_actuals,
            "settled_ts": g["commence_time"],
            "completed": True,
        }
    return results
