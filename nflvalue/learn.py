"""The self-improving part: grade finished games and adjust factor weights.

Workflow (driven by update_results.py):
  1. Match finished games to the predictions we logged.
  2. Grade each backed outcome win / loss / push and record the profit.
  3. Run one online logistic-regression step per graded prediction so factor
     weights drift toward whatever actually predicted winners.
  4. Recompute performance metrics (ROI, win rate, Brier, calibration) that the
     dashboard shows, so you can watch the model learn.
"""

from __future__ import annotations

from typing import Dict, List

from . import config

# Starting weights: small, conservative priors. Learning moves them from here
# toward whatever actually predicts winners (the exploitable residual edge).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "g_injury_diff": 0.12, "g_revenge": 0.03, "g_rest_diff": 0.06,
    "g_matchup": 0.15, "g_home_field": 0.05,
    "t_weather": 0.18, "t_injuries": 0.05, "t_pace": 0.08,
    "p_usage_trend": 0.12, "p_opp_defense": 0.12,
    "p_weather_pass": 0.15, "p_weather_rush": 0.08,
}

WEIGHT_CLIP = 2.5


def load_weights() -> Dict[str, float]:
    w = config.load_json(config.WEIGHTS_PATH, None)
    if not w:
        return dict(DEFAULT_WEIGHTS)
    # make sure newly added factors get a prior
    for k, v in DEFAULT_WEIGHTS.items():
        w.setdefault(k, v)
    return w


def save_weights(weights: Dict[str, float]) -> None:
    config.save_json(config.WEIGHTS_PATH, weights)


def load_history() -> List[Dict]:
    return config.load_json(config.HISTORY_PATH, [])


def save_history(history: List[Dict]) -> None:
    config.save_json(config.HISTORY_PATH, history)


# --------------------------------------------------------------------------- #
# Logging predictions
# --------------------------------------------------------------------------- #
def log_candidates(history: List[Dict], candidates: List[Dict], is_bet_ids) -> List[Dict]:
    """Upsert this slate's candidates into history (so we can grade them later).

    Every candidate is stored for calibration; the ones that cleared the EV
    threshold are flagged ``is_bet`` so we can also track a real bankroll curve.
    """
    by_id = {r["id"]: r for r in history}
    for c in candidates:
        rec = dict(c)
        rec["is_bet"] = rec["id"] in is_bet_ids
        existing = by_id.get(rec["id"])
        if existing and existing.get("status") == "graded":
            continue  # never overwrite a settled result
        if existing:
            existing.update({
                "price_decimal": rec["price_decimal"],
                "price_american": rec["price_american"],
                "best_book": rec["best_book"],
                "p_consensus": rec["p_consensus"],
                "p_model": rec["p_model"],
                "ev": rec["ev"], "edge": rec["edge"],
                "kelly": rec["kelly"], "stake_units": rec["stake_units"],
                "features": rec["features"], "why": rec["why"],
                "is_bet": rec["is_bet"],
            })
        else:
            history.append(rec)
            by_id[rec["id"]] = rec
    return history


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def _did_win(rec: Dict, res: Dict):
    """Return (y, profit_per_unit) or None for a push, given a game result.

    ``res`` keys: home_score, away_score, and for props: actuals {player_key: value}.
    """
    market = rec["market"]
    side = rec["side_key"]
    hs, as_ = res.get("home_score"), res.get("away_score")

    if market == "moneyline":
        if hs == as_:
            return None
        won = (side == "home" and hs > as_) or (side == "away" and as_ > hs)
    elif market == "spread":
        margin = hs - as_
        pt = rec.get("point") or 0
        if side == "home":
            diff = margin + pt
        else:
            diff = (-margin) + pt
        if abs(diff) < 1e-9:
            return None
        won = diff > 0
    elif market == "total":
        pt = rec.get("point") or 0
        total = hs + as_
        if abs(total - pt) < 1e-9:
            return None
        won = (side == "over" and total > pt) or (side == "under" and total < pt)
    elif market == "prop":
        actuals = res.get("prop_actuals", {})
        key = rec.get("player")
        actual = actuals.get(key)
        if actual is None:
            return "skip"
        pt = rec.get("point") or 0
        if abs(actual - pt) < 1e-9:
            return None
        over = "over" in rec["side_key"]
        won = (over and actual > pt) or (not over and actual < pt)
    else:
        return "skip"

    profit = (rec["price_decimal"] - 1.0) if won else -1.0
    return (1 if won else 0, profit)


def grade(history: List[Dict], results: Dict[str, Dict]) -> int:
    """Settle pending predictions whose games have a result. Returns count graded."""
    graded = 0
    for rec in history:
        if rec.get("status") == "graded":
            continue
        res = results.get(rec["game_id"])
        if not res:
            continue
        outcome = _did_win(rec, res)
        if outcome == "skip":
            continue
        rec["status"] = "graded"
        rec["settled_ts"] = res.get("settled_ts")
        if outcome is None:
            rec["result"] = "push"
            rec["profit"] = 0.0
            rec["y"] = None
        else:
            y, profit = outcome
            rec["result"] = "win" if y == 1 else "loss"
            rec["profit"] = round(profit, 4)
            rec["y"] = y
        graded += 1
    return graded


# --------------------------------------------------------------------------- #
# Learning: online logistic update on the factor weights
# --------------------------------------------------------------------------- #
def update_weights(weights: Dict[str, float], history: List[Dict],
                   cfg: Dict) -> Dict[str, float]:
    lr = cfg.get("learning_rate", 0.06)
    l2 = cfg.get("l2", 0.01)
    # Train only on records not yet used, in settle order.
    fresh = [r for r in history
             if r.get("status") == "graded" and r.get("y") is not None
             and not r.get("trained")]
    fresh.sort(key=lambda r: str(r.get("settled_ts")))
    for rec in fresh:
        x = rec.get("features", {})
        p = rec.get("p_model", 0.5)
        y = rec["y"]
        err = y - p                      # gradient of log-loss wrt the logit shift
        for k, xk in x.items():
            if xk == 0:
                continue
            w = weights.get(k, 0.0)
            w += lr * (err * xk - l2 * w)
            weights[k] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, round(w, 5)))
        rec["trained"] = True
    return weights


# --------------------------------------------------------------------------- #
# Performance metrics for the dashboard
# --------------------------------------------------------------------------- #
def metrics(history: List[Dict], cfg: Dict) -> Dict:
    graded = [r for r in history if r.get("status") == "graded"]
    settled = [r for r in graded if r.get("y") is not None]   # exclude pushes
    bets = [r for r in graded if r.get("is_bet")]

    def _roi(recs):
        staked = sum(1.0 for r in recs if r.get("y") is not None)
        profit = sum(r.get("profit", 0.0) for r in recs)
        return (profit / staked) if staked else 0.0

    n = len(settled)
    wins = sum(1 for r in settled if r["y"] == 1)
    brier = sum((r["p_model"] - r["y"]) ** 2 for r in settled) / n if n else 0.0

    # calibration: bucket model probabilities into 10 bins
    bins = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        b = [r for r in settled if lo <= r["p_model"] < hi or (i == 9 and r["p_model"] == 1.0)]
        if b:
            bins.append({
                "bucket": f"{int(lo*100)}-{int(hi*100)}%",
                "predicted": round(sum(r["p_model"] for r in b) / len(b), 4),
                "actual": round(sum(r["y"] for r in b) / len(b), 4),
                "n": len(b),
            })

    # bankroll equity curve from recommended bets, using fractional-Kelly stakes
    bankroll = cfg.get("bankroll_units", 100.0)
    equity = [round(bankroll, 2)]
    for r in sorted([b for b in bets if b.get("y") is not None],
                    key=lambda r: str(r.get("settled_ts"))):
        stake = bankroll * (r.get("stake_units", 0.0) / 100.0)
        bankroll += stake * r.get("profit", 0.0)
        equity.append(round(bankroll, 2))

    bet_profit = sum(r.get("profit", 0.0) for r in bets if r.get("y") is not None)
    bet_n = sum(1 for r in bets if r.get("y") is not None)

    return {
        "graded_total": len(graded),
        "settled_total": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "roi_all": round(_roi(settled), 4),
        "brier": round(brier, 4),
        "calibration": bins,
        "bets_settled": bet_n,
        "bets_win_rate": round(sum(1 for r in bets if r.get("y") == 1) / bet_n, 4) if bet_n else 0.0,
        "bets_roi": round(bet_profit / bet_n, 4) if bet_n else 0.0,
        "equity_curve": equity,
        "bankroll": round(bankroll, 2),
        "start_bankroll": cfg.get("bankroll_units", 100.0),
    }
