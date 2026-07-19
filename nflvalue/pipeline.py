"""High-level orchestration used by run.py and update_results.py."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from . import config, dashboard, learn, model, oddsmath
from .sources import demo, live

try:  # Monte Carlo needs numpy; the rest of the app runs fine without it.
    from . import montecarlo as _mc
    _HAVE_MC = True
except Exception:  # noqa: BLE001
    _HAVE_MC = False

import os as _os

DISCLAIMER = (
    "<b>For research & entertainment.</b> Sportsbook lines are highly efficient; "
    "this tool finds and sizes <i>potential</i> edges and learns from results — it is "
    "not a guarantee of profit. Bet only what you can afford to lose. If gambling "
    "stops being fun, call 1-800-GAMBLER."
)


def _eligible(c: Dict, cfg: Dict) -> bool:
    return cfg.get("min_odds", 1.4) <= c["price_decimal"] <= cfg.get("max_odds", 6.0)


def _all_games_view(games: List[Dict], cfg: Dict) -> List[Dict]:
    rows = []
    for g in games:
        books = g.get("books", {})
        ctx = g.get("context", {})
        # moneyline best prices
        h2h = {b: (p.get("home"), p.get("away")) for b, p in books.get("h2h", {}).items()
               if p.get("home") and p.get("away")}
        con = oddsmath.consensus_two_way(h2h, cfg.get("sharp_books", []), cfg.get("sharp_weight", 2.0))
        ml = "—"
        if con:
            ml = (f"{g['home_team'].split()[-1]} "
                  f"{oddsmath.american_str(oddsmath.decimal_to_american(con['best_a']))} / "
                  f"{g['away_team'].split()[-1]} "
                  f"{oddsmath.american_str(oddsmath.decimal_to_american(con['best_b']))}")
        # spread / total modal points
        sp = next(iter(books.get("spreads", {}).values()), None)
        spread = ("%+g" % sp["home"]["point"]) if sp and sp["home"].get("point") is not None else "—"
        tot = next(iter(books.get("totals", {}).values()), None)
        total = ("%g" % tot["over"]["point"]) if tot and tot["over"].get("point") is not None else "—"

        wx = ctx.get("weather", {})
        if wx.get("dome"):
            wxs = "dome"
        elif wx:
            wxs = f"{round(wx.get('wind_mph',0))}mph wind, {round(wx.get('temp_f',60))}°F"
            if wx.get("precip_mm"):
                wxs += f", {wx.get('precip_mm')}mm rain"
        else:
            wxs = ""
        inj = (f"inj {len(ctx.get('injuries_home',[]))}-{len(ctx.get('injuries_away',[]))}"
               if ctx else "")
        rev = " • revenge spot" if (ctx.get("revenge_home") or ctx.get("revenge_away")) else ""
        try:
            kickoff = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00")).strftime("%a %m/%d %H:%M")
        except Exception:  # noqa: BLE001
            kickoff = g.get("commence_time", "")

        rows.append({
            "home_team": g["home_team"], "away_team": g["away_team"],
            "kickoff": kickoff, "spread": spread, "total": total, "ml": ml,
            "context": " • ".join([s for s in [wxs, inj] if s]) + rev,
        })
    return rows


def _build_candidates(games: List[Dict], weights: Dict, cfg: Dict):
    game_c, prop_c = [], []
    for g in games:
        game_c += model.build_game_candidates(g, weights, cfg)
        prop_c += model.build_prop_candidates(g, weights, cfg)
    return game_c, prop_c


def _market_lines(game: Dict):
    """Return (home_spread_point, total_point) from the books, if present."""
    books = game.get("books", {})
    sp = next(iter(books.get("spreads", {}).values()), None)
    home_point = sp["home"].get("point") if sp and sp.get("home") else None
    tot = next(iter(books.get("totals", {}).values()), None)
    total_point = tot["over"].get("point") if tot and tot.get("over") else None
    return home_point, total_point


def attach_montecarlo(games: List[Dict], n: int = 8000):
    """Run the drive-based Monte Carlo for each game using historical ratings.

    Returns a dashboard section, or None if numpy / ratings aren't available.
    The MC is a calibrated fair-value second opinion — NOT an edge vs the close
    (see the backtest). We surface projections, not bet signals.
    """
    if not _HAVE_MC:
        return None
    priors = config.load_json(_os.path.join(config.DATA_DIR, "league_priors.json"), None)
    ratings = config.load_json(_os.path.join(config.DATA_DIR, "ratings_current.json"), None)
    if not priors or not ratings:
        return None

    sims = []
    for g in games:
        h = ratings.get(g["home_team"])
        a = ratings.get(g["away_team"])
        if not h or not a:
            continue
        home_point, total_point = _market_lines(g)
        spread_line = (-home_point) if home_point is not None else None
        r = _mc.simulate(h, a, priors, spread_line=spread_line, total_line=total_point, n=n)
        g["mc"] = r  # available to the model/dashboard
        sims.append({
            "home_team": g["home_team"], "away_team": g["away_team"],
            "commence_time": g.get("commence_time"),
            "proj_home": r["median_home"], "proj_away": r["median_away"],
            "exp_home": r["exp_home"], "exp_away": r["exp_away"],
            "p_home_win": r["p_home_win"], "p_away_win": r["p_away_win"],
            "fair_home_ml": _mc.fair_moneyline_odds(r["p_home_win"]),
            "fair_away_ml": _mc.fair_moneyline_odds(r["p_away_win"]),
            "market_home_point": home_point, "market_total": total_point,
            "p_home_cover": r.get("p_home_cover"), "p_over": r.get("p_over"),
            "margin_mean": r["margin_mean"], "margin_sd": r["margin_sd"],
            "total_mean": r["total_mean"], "margin_hist": r["margin_hist"],
        })
    if not sims:
        return None
    return {"seasons": priors.get("seasons", []), "n_sims": n, "games": sims}


def run(cfg: Dict = None, mode: str = None, games: List[Dict] = None) -> Dict:
    """Build the current slate, find value, log predictions, render dashboard."""
    cfg = cfg or config.load_config()
    if mode is None:
        mode = "live" if cfg.get("odds_api_key") else "demo"

    if games is None:
        if mode == "live":
            try:
                games = live.build_live_slate(cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[run] live fetch failed ({exc}); falling back to demo")
                mode = "demo"
                games = demo.generate_slate(datetime.now(timezone.utc))
        else:
            games = demo.generate_slate(datetime.now(timezone.utc))

    weights = learn.load_weights()
    game_c, prop_c = _build_candidates(games, weights, cfg)
    thr = cfg.get("ev_threshold", 0.03)

    value_bets = sorted([c for c in game_c if c["ev"] >= thr and _eligible(c, cfg)],
                        key=lambda c: -c["ev"])[:40]
    value_props = sorted([c for c in prop_c if c["ev"] >= thr and _eligible(c, cfg)],
                         key=lambda c: -c["ev"])[:40]

    is_bet_ids = {c["id"] for c in value_bets + value_props}
    history = learn.load_history()
    learn.log_candidates(history, game_c + prop_c, is_bet_ids)
    learn.save_history(history)
    metrics = learn.metrics(history, cfg)

    mc_section = attach_montecarlo(games)
    backtest = config.load_json(_os.path.join(config.DATA_DIR, "backtest.json"), None)
    weekly = config.load_json(_os.path.join(config.DATA_DIR, "weekly.json"), None)

    best_ev = max([c["ev"] for c in value_bets + value_props], default=None)
    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": mode,
        "refresh_seconds": cfg.get("refresh_seconds", 90),
        "disclaimer": DISCLAIMER,
        "summary": {
            "n_games": len(games),
            "n_value_bets": len(value_bets),
            "n_value_props": len(value_props),
            "best_ev": best_ev,
            "mc_on": mc_section is not None,
        },
        "value_bets": value_bets,
        "value_props": value_props,
        "all_games": _all_games_view(games, cfg),
        "metrics": metrics,
        "weights": weights,
        "monte_carlo": mc_section,
        "backtest": backtest,
        "weekly": weekly,
    }
    config.save_json(config.LATEST_PATH, data)
    dashboard.write_dashboard(data)
    return data


def grade_and_learn(cfg: Dict, results: Dict[str, Dict]) -> Dict:
    """Settle results, run the learning update, refresh dashboard metrics."""
    weights = learn.load_weights()
    history = learn.load_history()
    n = learn.grade(history, results)
    learn.update_weights(weights, history, cfg)
    learn.save_weights(weights)
    learn.save_history(history)
    metrics = learn.metrics(history, cfg)

    data = config.load_json(config.LATEST_PATH, None)
    if data:
        data["metrics"] = metrics
        data["weights"] = weights
        data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        config.save_json(config.LATEST_PATH, data)
        dashboard.write_dashboard(data)
    return {"graded": n, "metrics": metrics}


def simulate_learning(cfg: Dict, weeks: int = 8) -> Dict:
    """DEMO bootstrap: play out N past weeks so the model learns before today.

    For each week: build a slate, log predictions with the current weights,
    simulate the games under the hidden true model, grade, and update weights.
    """
    weights = learn.load_weights()
    history = learn.load_history()
    base = datetime.now(timezone.utc)
    total_graded = 0
    for wk in range(weeks):
        seed = 1000 + wk
        games = demo.generate_slate(base, n_games=12, seed=seed)
        game_c, prop_c = _build_candidates(games, weights, cfg)
        thr = cfg.get("ev_threshold", 0.03)
        is_bet = {c["id"] for c in (game_c + prop_c)
                  if c["ev"] >= thr and _eligible(c, cfg)}
        learn.log_candidates(history, game_c + prop_c, is_bet)
        results = demo.simulate_results(games, seed=seed)
        total_graded += learn.grade(history, results)
        learn.update_weights(weights, history, cfg)
    learn.save_weights(weights)
    learn.save_history(history)
    return {"weeks": weeks, "graded": total_graded,
            "metrics": learn.metrics(history, cfg)}
