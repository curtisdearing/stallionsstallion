"""Situational factors that move a game away from the market price.

Each factor turns raw context (weather forecast, injury report, revenge spot,
matchup ratings) into a small signed number. The model multiplies those
numbers by learned weights and adds them to the market's fair log-odds.

The weights start small and conservative on purpose: the market is sharp, so
we only drift away from it a little, and the learning loop (learn.py) grows or
shrinks each weight based on whether it actually predicted winners.
"""

from __future__ import annotations

from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# NFL stadiums: (lat, lon, is_dome). Weather is ignored for domes / fixed roofs.
# --------------------------------------------------------------------------- #
STADIUMS: Dict[str, Dict] = {
    "Arizona Cardinals":      {"lat": 33.5277, "lon": -112.2626, "dome": True},
    "Atlanta Falcons":        {"lat": 33.7554, "lon": -84.4008,  "dome": True},
    "Baltimore Ravens":       {"lat": 39.2780, "lon": -76.6227,  "dome": False},
    "Buffalo Bills":          {"lat": 42.7738, "lon": -78.7870,  "dome": False},
    "Carolina Panthers":      {"lat": 35.2258, "lon": -80.8528,  "dome": False},
    "Chicago Bears":          {"lat": 41.8623, "lon": -87.6167,  "dome": False},
    "Cincinnati Bengals":     {"lat": 39.0954, "lon": -84.5160,  "dome": False},
    "Cleveland Browns":       {"lat": 41.5061, "lon": -81.6995,  "dome": False},
    "Dallas Cowboys":         {"lat": 32.7473, "lon": -97.0945,  "dome": True},
    "Denver Broncos":         {"lat": 39.7439, "lon": -105.0201, "dome": False},
    "Detroit Lions":          {"lat": 42.3400, "lon": -83.0456,  "dome": True},
    "Green Bay Packers":      {"lat": 44.5013, "lon": -88.0622,  "dome": False},
    "Houston Texans":         {"lat": 29.6847, "lon": -95.4107,  "dome": True},
    "Indianapolis Colts":     {"lat": 39.7601, "lon": -86.1639,  "dome": True},
    "Jacksonville Jaguars":   {"lat": 30.3239, "lon": -81.6373,  "dome": False},
    "Kansas City Chiefs":     {"lat": 39.0489, "lon": -94.4839,  "dome": False},
    "Las Vegas Raiders":      {"lat": 36.0909, "lon": -115.1833, "dome": True},
    "Los Angeles Chargers":   {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "Los Angeles Rams":       {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "Miami Dolphins":         {"lat": 25.9580, "lon": -80.2389,  "dome": False},
    "Minnesota Vikings":      {"lat": 44.9737, "lon": -93.2581,  "dome": True},
    "New England Patriots":   {"lat": 42.0909, "lon": -71.2643,  "dome": False},
    "New Orleans Saints":     {"lat": 29.9511, "lon": -90.0812,  "dome": True},
    "New York Giants":        {"lat": 40.8135, "lon": -74.0745,  "dome": False},
    "New York Jets":          {"lat": 40.8135, "lon": -74.0745,  "dome": False},
    "Philadelphia Eagles":    {"lat": 39.9008, "lon": -75.1675,  "dome": False},
    "Pittsburgh Steelers":    {"lat": 40.4468, "lon": -80.0158,  "dome": False},
    "San Francisco 49ers":    {"lat": 37.4030, "lon": -121.9700, "dome": False},
    "Seattle Seahawks":       {"lat": 47.5952, "lon": -122.3316, "dome": False},
    "Tampa Bay Buccaneers":   {"lat": 27.9759, "lon": -82.5033,  "dome": False},
    "Tennessee Titans":       {"lat": 36.1665, "lon": -86.7713,  "dome": False},
    "Washington Commanders":  {"lat": 38.9078, "lon": -76.8645,  "dome": False},
}

# Positional importance weights for injury impact (QB dominates everything).
POSITION_WEIGHT = {
    "QB": 1.00, "RB": 0.30, "WR": 0.28, "TE": 0.18, "OT": 0.22, "OL": 0.18,
    "G": 0.15, "C": 0.15, "EDGE": 0.25, "DE": 0.22, "DT": 0.18, "LB": 0.18,
    "CB": 0.24, "S": 0.18, "K": 0.10, "DEF": 0.20,
}
STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.75, "questionable": 0.35, "ir": 1.0}


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def weather_severity(weather: Optional[Dict]) -> float:
    """0 (perfect / dome) .. ~1 (brutal). High wind & precip hurt passing/scoring.

    ``weather`` keys: wind_mph, precip_mm, temp_f (all optional).
    """
    if not weather or weather.get("dome"):
        return 0.0
    wind = float(weather.get("wind_mph", 0) or 0)
    precip = float(weather.get("precip_mm", 0) or 0)
    temp = float(weather.get("temp_f", 60) or 60)

    wind_s = min(wind / 30.0, 1.0)                 # 30+ mph = max
    precip_s = min(precip / 8.0, 1.0)              # 8+ mm/hr = max
    cold_s = min(max(20.0 - temp, 0) / 30.0, 1.0)  # below 20F starts to bite
    sev = 0.55 * wind_s + 0.30 * precip_s + 0.15 * cold_s
    return round(min(sev, 1.0), 4)


# --------------------------------------------------------------------------- #
# Injuries
# --------------------------------------------------------------------------- #
def injury_severity(injuries) -> float:
    """Sum a team's injury impact. ``injuries`` is a list of dicts with
    keys: position, status. Returns roughly 0 (healthy) .. ~2 (decimated)."""
    if not injuries:
        return 0.0
    total = 0.0
    for inj in injuries:
        pos = str(inj.get("position", "")).upper()
        status = str(inj.get("status", "")).lower()
        total += POSITION_WEIGHT.get(pos, 0.12) * STATUS_WEIGHT.get(status, 0.3)
    return round(total, 4)


# --------------------------------------------------------------------------- #
# Feature vectors
# --------------------------------------------------------------------------- #
def game_side_features(ctx: Dict, side: str) -> Dict[str, float]:
    """Features for backing ``side`` ('home' or 'away') on the moneyline/spread.

    ``ctx`` carries pre-computed numbers for the game (see model.build_*).
    Features are signed from the perspective of the backed side.
    """
    sign = 1.0 if side == "home" else -1.0
    inj_home = ctx.get("inj_home", 0.0)
    inj_away = ctx.get("inj_away", 0.0)
    # Positive when the OTHER team is more banged up than ours.
    inj_diff = sign * (inj_away - inj_home)

    revenge = ctx.get("revenge_home", 0.0) if side == "home" else ctx.get("revenge_away", 0.0)
    rest_diff = sign * (ctx.get("rest_home", 0) - ctx.get("rest_away", 0)) / 7.0
    matchup = sign * ctx.get("matchup_edge", 0.0)   # >0 favours home offense/defense net
    home_field = 1.0 if side == "home" else 0.0

    return {
        "g_injury_diff": round(inj_diff, 4),
        "g_revenge": round(revenge, 4),
        "g_rest_diff": round(rest_diff, 4),
        "g_matchup": round(matchup, 4),
        "g_home_field": home_field,
    }


def total_features(ctx: Dict, side: str) -> Dict[str, float]:
    """Features for totals. side = 'over' or 'under'. Bad weather favours under."""
    sign = 1.0 if side == "under" else -1.0
    sev = ctx.get("weather_sev", 0.0)
    inj_off = ctx.get("inj_home", 0.0) + ctx.get("inj_away", 0.0)  # injuries lower scoring
    pace = ctx.get("pace_edge", 0.0)  # >0 = faster/higher scoring expected
    return {
        "t_weather": round(sign * sev, 4),
        "t_injuries": round(sign * 0.5 * inj_off, 4),
        "t_pace": round(-sign * pace, 4),
    }


def prop_features(ctx: Dict, prop_type: str, side: str) -> Dict[str, float]:
    """Features for a player prop. side = 'over' or 'under'.

    ``ctx`` keys used: weather_sev, opp_def_rating (vs this prop type, >0 = tough
    defense), usage_trend (>0 = trending up), is_pass_prop / is_rush_prop bools.
    """
    sign = 1.0 if side == "over" else -1.0
    sev = ctx.get("weather_sev", 0.0)
    opp_def = ctx.get("opp_def_rating", 0.0)
    usage = ctx.get("usage_trend", 0.0)

    feats = {
        "p_usage_trend": round(sign * usage, 4),
        "p_opp_defense": round(-sign * opp_def, 4),
    }
    if prop_type.startswith("pass") or prop_type.startswith("recept") or prop_type.startswith("rec"):
        # Wind/rain suppress passing & receiving.
        feats["p_weather_pass"] = round(-sign * sev, 4)
    elif prop_type.startswith("rush"):
        # Bad weather slightly boosts rushing volume.
        feats["p_weather_rush"] = round(sign * 0.5 * sev, 4)
    return feats
