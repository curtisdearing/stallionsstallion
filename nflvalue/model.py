"""Turn raw multi-book odds + context into ranked value bets.

Pipeline per market:
  1. de-vig every book and build a weighted consensus fair probability,
  2. find the BEST price available across all books (line shopping),
  3. nudge the fair probability with situational factors (learned weights),
  4. compute EV at the best price and a (fractional) Kelly stake,
  5. emit a candidate record for every side so the learner can calibrate.

``run.py`` decides which candidates clear the EV threshold and become picks.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from . import factors, oddsmath


# --------------------------------------------------------------------------- #
def _cid(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def apply_factors(
    p_consensus: float,
    features: Dict[str, float],
    weights: Dict[str, float],
    max_shift: float = 0.65,
) -> float:
    """Shift the market fair prob by the learned factor signal (in log-odds).

    The total shift is clamped so we never wander absurdly far from a sharp
    market on the strength of a few situational features.
    """
    shift = sum(weights.get(k, 0.0) * v for k, v in features.items())
    shift = max(-max_shift, min(max_shift, shift))
    return oddsmath.sigmoid(oddsmath.logit(p_consensus) + shift)


def _explain(features: Dict[str, float], weights: Dict[str, float], top: int = 3):
    """Human-readable list of the biggest factor contributions for a pick."""
    contribs = []
    for k, v in features.items():
        c = weights.get(k, 0.0) * v
        if abs(c) > 1e-4:
            contribs.append((k, c))
    contribs.sort(key=lambda x: -abs(x[1]))
    label = {
        "g_injury_diff": "injury edge", "g_revenge": "revenge spot",
        "g_rest_diff": "rest edge", "g_matchup": "matchup edge",
        "g_home_field": "home field", "t_weather": "weather", "t_injuries":
        "injuries", "t_pace": "pace", "p_usage_trend": "usage trend",
        "p_opp_defense": "opp defense", "p_weather_pass": "weather (pass)",
        "p_weather_rush": "weather (rush)",
    }
    out = []
    for k, c in contribs[:top]:
        out.append({"factor": label.get(k, k), "impact": round(c, 3),
                    "dir": "+" if c > 0 else "-"})
    return out


def _make_candidate(game, market, outcome_label, side_key, p_consensus,
                    best_price, best_book, point, features, weights, cfg, extra=None):
    # Raw model probability (used for learning + calibration).
    p_model = apply_factors(p_consensus, features, weights)
    # Betting probability: shrink the model's disagreement with the market toward
    # the market. Markets are sharp and our edge estimate is noisy, so we only
    # act on a fraction of the gap. This keeps EV honest (less overconfident).
    shrink = cfg.get("edge_shrinkage", 0.5)
    p_bet = min(max(p_consensus + shrink * (p_model - p_consensus), 1e-4), 1 - 1e-4)

    ev = oddsmath.ev_pct(p_bet, best_price)
    edge = oddsmath.prob_edge(p_bet, best_price)
    kelly = oddsmath.kelly_fraction(p_bet, best_price)
    stake = min(kelly * cfg.get("kelly_multiplier", 0.25),
                cfg.get("max_stake_pct", 0.03) / 1.0)
    rec = {
        "id": _cid(game["id"], market, side_key, point),
        "type": "prop" if market == "prop" else "game",
        "game_id": game["id"],
        "commence_time": game.get("commence_time"),
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "market": market,
        "outcome": outcome_label,
        "side_key": side_key,
        "point": point,
        "best_book": best_book,
        "price_decimal": round(best_price, 3),
        "price_american": oddsmath.american_str(oddsmath.decimal_to_american(best_price)),
        "p_consensus": round(p_consensus, 4),
        "p_model": round(p_model, 4),
        "p_bet": round(p_bet, 4),
        "ev": round(ev, 4),
        "edge": round(edge, 4),
        "kelly": round(kelly, 4),
        "stake_units": round(stake * 100, 2),       # in % of bankroll
        "features": features,
        "why": _explain(features, weights),
        "status": "pending",
    }
    if extra:
        rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
# Game lines: moneyline, spread, total
# --------------------------------------------------------------------------- #
def build_game_candidates(game: Dict, weights: Dict[str, float], cfg: Dict) -> List[Dict]:
    out: List[Dict] = []
    ctx = game.get("context", {})
    books = game.get("books", {})

    # ---- Moneyline (h2h) ----
    h2h = books.get("h2h", {})
    bp = {b: (p.get("home"), p.get("away")) for b, p in h2h.items()
          if p.get("home") and p.get("away")}
    con = oddsmath.consensus_two_way(bp, cfg.get("sharp_books", ["pinnacle"]),
                                     cfg.get("sharp_weight", 2.0))
    if con:
        out.append(_make_candidate(
            game, "moneyline", f"{game['home_team']} ML", "home",
            con["p_a"], con["best_a"], con["best_a_book"], None,
            factors.game_side_features(ctx, "home"), weights, cfg))
        out.append(_make_candidate(
            game, "moneyline", f"{game['away_team']} ML", "away",
            con["p_b"], con["best_b"], con["best_b_book"], None,
            factors.game_side_features(ctx, "away"), weights, cfg))

    # ---- Spread ----
    spreads = books.get("spreads", {})
    bp = {}
    pts = {"home": [], "away": []}
    for b, p in spreads.items():
        h, a = p.get("home"), p.get("away")
        if h and a and h.get("price") and a.get("price"):
            bp[b] = (h["price"], a["price"])
            pts["home"].append(h.get("point"))
            pts["away"].append(a.get("point"))
    con = oddsmath.consensus_two_way(bp, cfg.get("sharp_books", ["pinnacle"]),
                                     cfg.get("sharp_weight", 2.0))
    if con:
        ph = _mode(pts["home"])
        pa = _mode(pts["away"])
        out.append(_make_candidate(
            game, "spread", f"{game['home_team']} {_fmt_pt(ph)}", "home",
            con["p_a"], con["best_a"], con["best_a_book"], ph,
            factors.game_side_features(ctx, "home"), weights, cfg))
        out.append(_make_candidate(
            game, "spread", f"{game['away_team']} {_fmt_pt(pa)}", "away",
            con["p_b"], con["best_b"], con["best_b_book"], pa,
            factors.game_side_features(ctx, "away"), weights, cfg))

    # ---- Total ----
    totals = books.get("totals", {})
    bp = {}
    pts = []
    for b, p in totals.items():
        o, u = p.get("over"), p.get("under")
        if o and u and o.get("price") and u.get("price"):
            bp[b] = (o["price"], u["price"])
            pts.append(o.get("point"))
    con = oddsmath.consensus_two_way(bp, cfg.get("sharp_books", ["pinnacle"]),
                                     cfg.get("sharp_weight", 2.0))
    if con:
        pt = _mode(pts)
        out.append(_make_candidate(
            game, "total", f"Over {pt}", "over",
            con["p_a"], con["best_a"], con["best_a_book"], pt,
            factors.total_features(ctx, "over"), weights, cfg))
        out.append(_make_candidate(
            game, "total", f"Under {pt}", "under",
            con["p_b"], con["best_b"], con["best_b_book"], pt,
            factors.total_features(ctx, "under"), weights, cfg))
    return out


# --------------------------------------------------------------------------- #
# Player props
# --------------------------------------------------------------------------- #
def build_prop_candidates(game: Dict, weights: Dict[str, float], cfg: Dict) -> List[Dict]:
    out: List[Dict] = []
    for prop in game.get("props", []):
        bp = {}
        for b, pr in prop.get("books", {}).items():
            o, u = pr.get("over"), pr.get("under")
            if o and u:
                bp[b] = (o, u)
        con = oddsmath.consensus_two_way(bp, cfg.get("sharp_books", ["pinnacle"]),
                                         cfg.get("sharp_weight", 2.0))
        if not con:
            continue
        pctx = prop.get("ctx", {})
        ptype = prop.get("prop_type", "")
        line = prop.get("line")
        label = f"{prop.get('player')} {prop.get('label', ptype)} {line}"
        extra = {"player": prop.get("player"), "team": prop.get("team"),
                 "prop_type": ptype}
        out.append(_make_candidate(
            game, "prop", f"{label} Over", f"{prop.get('player')}|over",
            con["p_a"], con["best_a"], con["best_a_book"], line,
            factors.prop_features(pctx, ptype, "over"), weights, cfg, extra))
        out.append(_make_candidate(
            game, "prop", f"{label} Under", f"{prop.get('player')}|under",
            con["p_b"], con["best_b"], con["best_b_book"], line,
            factors.prop_features(pctx, ptype, "under"), weights, cfg, extra))
    return out


# --------------------------------------------------------------------------- #
def _mode(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(set(vals), key=vals.count)


def _fmt_pt(pt):
    if pt is None:
        return ""
    return f"+{pt}" if pt > 0 else str(pt)
