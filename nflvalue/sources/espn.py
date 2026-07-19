"""ESPN's free, unofficial NFL endpoints (no API key required).

Used for injuries and a light matchup signal from standings. These endpoints
are undocumented and can change; every call is wrapped so a failure just means
that factor is treated as neutral rather than crashing the pipeline.
"""

from __future__ import annotations

from typing import Dict, List

from ._http import get_json

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def fetch_injuries() -> Dict[str, List[Dict]]:
    """Return {team_display_name: [{position, status, name}, ...]}."""
    out: Dict[str, List[Dict]] = {}
    try:
        data = get_json(f"{SITE}/injuries")
    except Exception as exc:  # noqa: BLE001
        print(f"[espn] injuries fetch failed: {exc}")
        return out
    for team in data.get("injuries", []):
        name = team.get("displayName") or team.get("team", {}).get("displayName")
        entries = []
        for it in team.get("injuries", []):
            ath = it.get("athlete", {}) or {}
            pos = (ath.get("position", {}) or {}).get("abbreviation", "")
            entries.append({
                "position": pos,
                "status": str(it.get("status", "")).lower(),
                "name": ath.get("displayName", ""),
            })
        if name:
            out[name] = entries
    return out


def fetch_power() -> Dict[str, float]:
    """Light team-strength proxy from standings win%, centered at 0."""
    out: Dict[str, float] = {}
    try:
        data = get_json("https://site.api.espn.com/apis/v2/sports/football/nfl/standings")
    except Exception as exc:  # noqa: BLE001
        print(f"[espn] standings fetch failed: {exc}")
        return out
    try:
        for child in data.get("children", []):
            for entry in child.get("standings", {}).get("entries", []):
                team = entry.get("team", {}).get("displayName")
                winpct = 0.5
                for st in entry.get("stats", []):
                    if st.get("name") == "winPercent":
                        winpct = float(st.get("value", 0.5))
                if team:
                    out[team] = (winpct - 0.5) * 12.0  # ~ points scale
    except Exception:  # noqa: BLE001
        return out
    return out
