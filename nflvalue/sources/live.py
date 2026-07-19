"""Assemble a live slate from The Odds API + Open-Meteo + ESPN.

Returns the SAME game structure the demo generator produces, so the rest of the
pipeline (model, learning, dashboard) doesn't care whether data is live or demo.
"""

from __future__ import annotations

from typing import Dict, List

from .. import factors as factmod
from . import espn, oddsapi, weather


def _match_team(name: str, injuries: Dict[str, List[Dict]]):
    if name in injuries:
        return injuries[name]
    last = name.split()[-1].lower()
    for k, v in injuries.items():
        if k.split()[-1].lower() == last:
            return v
    return []


def build_live_slate(cfg: Dict) -> List[Dict]:
    games = oddsapi.fetch_game_odds(cfg)
    print(f"[live] {len(games)} games from The Odds API")
    injuries = espn.fetch_injuries()
    power = espn.fetch_power()

    # props for the first N games (credit-conscious)
    prop_games = games[: cfg.get("max_prop_games_per_run", 4)] if cfg.get("fetch_props") else []
    prop_ids = {g["id"] for g in prop_games}

    for g in games:
        home, away = g["home_team"], g["away_team"]
        wx = weather.forecast_for_game(home, g["commence_time"]) or {"dome": True}
        sev = factmod.weather_severity(wx)
        inj_h = _match_team(home, injuries)
        inj_a = _match_team(away, injuries)
        matchup = 0.0
        if power:
            matchup = (power.get(home, 0.0) - power.get(away, 0.0)) / 12.0
        g["context"] = {
            "weather": wx, "weather_sev": sev,
            "inj_home": factmod.injury_severity(inj_h),
            "inj_away": factmod.injury_severity(inj_a),
            "injuries_home": inj_h, "injuries_away": inj_a,
            "revenge_home": 0.0, "revenge_away": 0.0,   # set manually if known
            "rest_home": 7, "rest_away": 7,
            "matchup_edge": round(matchup, 4),
            "pace_edge": 0.0,
        }
        if g["id"] in prop_ids:
            props = oddsapi.fetch_event_props(cfg, g["id"])
            for p in props:
                p["ctx"]["weather_sev"] = sev
            g["props"] = props
            print(f"[live]   {len(props)} props for {away} @ {home}")
    return games


def fetch_live_results(cfg: Dict) -> Dict[str, Dict]:
    return oddsapi.fetch_scores(cfg, days_from=3)
