"""Per-team-game measurements from the cached plays table.

One function, `team_game_measures(plays_clean)`, turns a plays frame into a tidy
long frame with one row per `(game_id, team, measure_id)` carrying `value`, its
numerator and its denominator. Everything downstream — rolling summaries, opponent
adjustment, notes — consumes that frame, so there is exactly one place where a
definition lives.

ATTRIBUTION, VERIFIED NOT ASSUMED
---------------------------------
CFBD sets `offense` to the team that had possession, on every play type. Checked
empirically against `plays_2026`, game 401867916: row 24 is a Robert Morris passing
touchdown and row 25 is a `Kickoff` with `offense = Robert Morris`, i.e. the KICKING
team. On punts, `offense` is the PUNTING team (row 4, `Punt Return`, 4th-and-4,
offense = Wagner). That makes special-teams COUNTS trustworthy.

It does not rescue special-teams YARDAGE. `yards_gained` on those rows is not the
return distance — a `Punt` row reads 0 and a `Punt Return Touchdown` reads 22 on a
play that plainly gained far more. So coverage and return yardage are registered
`blocked` with reason `unverified_field_semantics` and this module never computes
them. Counts and event rates, which are unambiguous, are computed.

GARBAGE TIME
------------
Input is `plays_clean`, from which `ingest` has already removed garbage time. That
is a documented choice, not an oversight: a fourth-quarter kneel-down sequence is
not evidence about a team's short-yardage offence.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CFBD play-type vocabulary. Lower-cased at the boundary, matched exactly, never
# by substring — "Interception Return Touchdown" and "Interception" are different
# events and a `str.contains` would double-count them.
# --------------------------------------------------------------------------- #
RUSH_TYPES = {"rush", "rushing touchdown"}
SACK_TYPES = {"sack"}
PASS_TYPES = {"pass reception", "pass incompletion", "passing touchdown",
              "pass completion"}
INT_TYPES = {"interception", "pass interception return",
             "interception return touchdown"}
FUMBLE_ALL = {"fumble recovery (own)", "fumble recovery (opponent)",
              "fumble return touchdown"}
FUMBLE_LOST = {"fumble recovery (opponent)", "fumble return touchdown"}

FG_GOOD = {"field goal good"}
FG_MISS = {"field goal missed"}
FG_BLOCKED = {"blocked field goal", "blocked field goal touchdown"}
PUNT_TYPES = {"punt", "punt return", "blocked punt", "blocked punt touchdown",
              "punt return touchdown"}
PUNT_BLOCKED = {"blocked punt", "blocked punt touchdown"}
PUNT_RETURN_TD = {"punt return touchdown"}
KICKOFF_TYPES = {"kickoff", "kickoff return (offense)", "kickoff return touchdown"}
KICK_RETURN_TD = {"kickoff return touchdown"}

#: Explosive-play thresholds. A stated convention, registered as such — see the
#: `caveat` on the registry entries. Not a validated cut point.
EXPLOSIVE_RUSH_YARDS = 12
EXPLOSIVE_PASS_YARDS = 16

#: A field-goal attempt from N yards to goal is kicked from N + 17 (7 yards of
#: snap depth plus the 10-yard end zone).
FG_DISTANCE_OFFSET = 17

_FG_BUCKETS = (("lt30", 0, 30), ("30_39", 30, 40), ("40_49", 40, 50),
               ("50_plus", 50, 999))


def _success(down, distance, gained) -> Optional[bool]:
    """Down-and-distance success. Identical thresholds to `ratings._success`.

    Duplicated rather than imported because `nflvalue/ratings.py` is sha256-pinned
    by `data/evidence_manifest.json` and backs the repo's only selected benchmark;
    `tests/test_team_evidence_measures.py::test_success_matches_the_ratings_definition`
    asserts the two agree on every input, so the copy cannot drift silently.
    """
    try:
        d, dist, g = int(down), float(distance), float(gained)
    except (TypeError, ValueError):
        return None
    if d < 1 or d > 4 or dist <= 0:
        return None
    need = {1: 0.5, 2: 0.7}.get(d, 1.0) * dist
    return g >= need


def annotate(plays: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean/derived columns every measure reads. Pure; no aggregation."""
    p = plays.copy()
    pt = p["play_type"].astype(str).str.strip().str.lower()
    p["_pt"] = pt
    p["_is_rush"] = pt.isin(RUSH_TYPES)
    p["_is_sack"] = pt.isin(SACK_TYPES)
    p["_is_pass"] = pt.isin(PASS_TYPES)
    p["_is_int"] = pt.isin(INT_TYPES)
    # A dropback is any pass-typed play, a sack, or an interception. QB scrambles
    # are typed Rush upstream and cannot be separated (playText is dropped at
    # ingest), so this denominator understates true dropbacks by the scramble
    # count. That caveat is carried on every sack-rate registry entry.
    p["_is_dropback"] = p["_is_pass"] | p["_is_sack"] | p["_is_int"]
    p["_is_fumble"] = pt.isin(FUMBLE_ALL)
    p["_is_fumble_lost"] = pt.isin(FUMBLE_LOST)
    p["_is_scrimmage"] = (p["_is_rush"] | p["_is_sack"] | p["_is_pass"]
                          | p["_is_int"] | p["_is_fumble"])

    yards = pd.to_numeric(p["yards_gained"], errors="coerce")
    p["_yards"] = yards
    p["_success"] = [_success(d, dist, g) for d, dist, g
                     in zip(p["down"], p["distance"], yards)]
    p["_ppa"] = pd.to_numeric(p["ppa"], errors="coerce")

    down = pd.to_numeric(p["down"], errors="coerce")
    dist = pd.to_numeric(p["distance"], errors="coerce")
    p["_short_yardage"] = down.isin([3, 4]) & (dist > 0) & (dist <= 2)
    p["_short_converted"] = p["_short_yardage"] & (yards >= dist)

    p["_fg_good"] = pt.isin(FG_GOOD)
    p["_fg_miss"] = pt.isin(FG_MISS)
    p["_fg_blocked"] = pt.isin(FG_BLOCKED)
    p["_fg_att"] = p["_fg_good"] | p["_fg_miss"] | p["_fg_blocked"]
    ytg = pd.to_numeric(p["yards_to_goal"], errors="coerce")
    p["_fg_distance"] = np.where(p["_fg_att"], ytg + FG_DISTANCE_OFFSET, np.nan)

    p["_is_punt"] = pt.isin(PUNT_TYPES)
    p["_punt_blocked"] = pt.isin(PUNT_BLOCKED)
    p["_punt_return_td"] = pt.isin(PUNT_RETURN_TD)
    p["_is_kickoff"] = pt.isin(KICKOFF_TYPES)
    p["_kick_return_td"] = pt.isin(KICK_RETURN_TD)
    return p


def _rate(num: float, den: float) -> Optional[float]:
    return float(num) / float(den) if den else None


def _rows_for(gid: int, season: int, week: int, team: str, opponent: str,
              own: pd.DataFrame, opp: pd.DataFrame) -> List[Dict]:
    """Every measure for one team in one game.

    `own` = plays where this team had possession. `opp` = plays where the opponent
    had possession. Both sides come from the same frame, so a "created" and an
    "allowed" measure never disagree about which plays happened.
    """
    out: List[Dict] = []

    def add(measure_id, value, num, den):
        out.append({"game_id": int(gid), "season": int(season), "week": int(week),
                    "team": str(team), "opponent": str(opponent),
                    "measure_id": measure_id,
                    "value": None if value is None or (isinstance(value, float)
                                                       and not np.isfinite(value))
                    else float(value),
                    "num": float(num), "den": float(den)})

    for side, frame in (("off", own), ("def", opp)):
        for phase, mask_col in (("rush", "_is_rush"), ("pass", "_is_pass")):
            sub = frame[frame[mask_col]]
            graded = sub[sub["_success"].notna()]
            add(f"{phase}_success_rate_{side}",
                _rate(graded["_success"].astype(bool).sum(), len(graded)),
                graded["_success"].astype(bool).sum(), len(graded))
            ppa = sub["_ppa"].dropna()
            add(f"{phase}_ppa_per_play_{side}",
                float(ppa.mean()) if len(ppa) else None, float(ppa.sum()), len(ppa))
            thr = EXPLOSIVE_RUSH_YARDS if phase == "rush" else EXPLOSIVE_PASS_YARDS
            with_yards = sub[sub["_yards"].notna()]
            expl = int((with_yards["_yards"] >= thr).sum())
            add(f"explosive_play_rate_{phase}_{side}",
                _rate(expl, len(with_yards)), expl, len(with_yards))

        drop = int(frame["_is_dropback"].sum())
        sacks = int(frame["_is_sack"].sum())
        add(f"sack_rate_per_dropback_{side}", _rate(sacks, drop), sacks, drop)

        rushes = frame[frame["_is_rush"] & frame["_yards"].notna()]
        stuffed = int((rushes["_yards"] <= 0).sum())
        add(f"rush_stuff_rate_proxy_{side}", _rate(stuffed, len(rushes)),
            stuffed, len(rushes))

        sy = frame[frame["_short_yardage"]]
        conv = int(sy["_short_converted"].sum())
        add(f"short_yardage_conversion_rate_{side}", _rate(conv, len(sy)),
            conv, len(sy))

    # ------------------------------------------------------------ turnovers --
    own_drop = int(own["_is_dropback"].sum())
    opp_drop = int(opp["_is_dropback"].sum())
    own_plays = int(own["_is_scrimmage"].sum())
    opp_plays = int(opp["_is_scrimmage"].sum())

    ints_thrown = int(own["_is_int"].sum())
    ints_forced = int(opp["_is_int"].sum())
    add("interceptions_thrown", ints_thrown, ints_thrown, own_drop)
    add("interception_rate_per_dropback", _rate(ints_thrown, own_drop),
        ints_thrown, own_drop)
    add("interceptions_forced", ints_forced, ints_forced, opp_drop)
    add("interception_rate_forced_per_opp_dropback", _rate(ints_forced, opp_drop),
        ints_forced, opp_drop)

    fum_all = int(own["_is_fumble"].sum())
    fum_lost = int(own["_is_fumble_lost"].sum())
    add("fumbles_total_off", fum_all, fum_all, own_plays)
    add("fumbles_lost_off", fum_lost, fum_lost, fum_all)
    add("fumble_retention_rate_off", _rate(fum_all - fum_lost, fum_all),
        fum_all - fum_lost, fum_all)
    add("fumble_rate_per_play_off", _rate(fum_all, own_plays), fum_all, own_plays)

    opp_fum_all = int(opp["_is_fumble"].sum())
    opp_fum_lost = int(opp["_is_fumble_lost"].sum())
    add("fumbles_forced_def", opp_fum_all, opp_fum_all, opp_plays)
    add("fumble_recovery_share_def", _rate(opp_fum_lost, opp_fum_all),
        opp_fum_lost, opp_fum_all)
    add("takeaway_rate_per_opp_play", _rate(ints_forced + opp_fum_lost, opp_plays),
        ints_forced + opp_fum_lost, opp_plays)
    add("giveaway_rate_per_play", _rate(ints_thrown + fum_lost, own_plays),
        ints_thrown + fum_lost, own_plays)

    # ----------------------------------------------------------------- pace --
    add("plays_per_game_off", own_plays, own_plays, 1)
    add("plays_per_game_def", opp_plays, opp_plays, 1)

    # ------------------------------------------------------- special teams --
    fg_att = int(own["_fg_att"].sum())
    fg_good = int(own["_fg_good"].sum())
    add("fg_attempts", fg_att, fg_att, 1)
    add("fg_made_rate", _rate(fg_good, fg_att), fg_good, fg_att)
    dists = own.loc[own["_fg_att"], "_fg_distance"].dropna()
    add("fg_distance_median", float(dists.median()) if len(dists) else None,
        float(dists.sum()), len(dists))
    add("fg_distance_iqr",
        float(dists.quantile(0.75) - dists.quantile(0.25)) if len(dists) else None,
        0.0, len(dists))
    long_att = own[own["_fg_att"] & (own["_fg_distance"] >= 40)]
    add("fg_made_rate_40_plus", _rate(int(long_att["_fg_good"].sum()), len(long_att)),
        int(long_att["_fg_good"].sum()), len(long_att))
    for name, lo, hi in _FG_BUCKETS:
        b = own[own["_fg_att"] & (own["_fg_distance"] >= lo) & (own["_fg_distance"] < hi)]
        out.append({"game_id": int(gid), "season": int(season), "week": int(week),
                    "team": str(team), "opponent": str(opponent),
                    "measure_id": f"fg_distance_by_bucket::{name}",
                    "value": float(int(b["_fg_good"].sum())),
                    "num": float(int(b["_fg_good"].sum())), "den": float(len(b))})

    punts = int(own["_is_punt"].sum())
    kickoffs = int(own["_is_kickoff"].sum())
    add("punt_count", punts, punts, 1)
    blocked = int(own["_fg_blocked"].sum()) + int(own["_punt_blocked"].sum())
    add("kick_blocked_rate_allowed", _rate(blocked, fg_att + punts),
        blocked, fg_att + punts)

    # Coverage: the team is kicking/punting, the opponent returned it for a score.
    add("punt_return_td_allowed_per_punt",
        _rate(int(own["_punt_return_td"].sum()), punts),
        int(own["_punt_return_td"].sum()), punts)
    add("kick_return_td_allowed_per_kickoff",
        _rate(int(own["_kick_return_td"].sum()), kickoffs),
        int(own["_kick_return_td"].sum()), kickoffs)
    # Returns: the opponent was kicking/punting and this team returned it.
    opp_punts = int(opp["_is_punt"].sum())
    opp_kicks = int(opp["_is_kickoff"].sum())
    add("punt_return_td_per_punt_faced",
        _rate(int(opp["_punt_return_td"].sum()), opp_punts),
        int(opp["_punt_return_td"].sum()), opp_punts)
    add("kick_return_td_per_kickoff_faced",
        _rate(int(opp["_kick_return_td"].sum()), opp_kicks),
        int(opp["_kick_return_td"].sum()), opp_kicks)
    return out


def team_game_measures(plays_clean: pd.DataFrame) -> pd.DataFrame:
    """Tidy long frame: one row per (game_id, team, measure_id).

    Columns: game_id, season, week, team, opponent, measure_id, value, num, den.
    `num`/`den` are carried so a season-to-date aggregate can be a ratio of sums
    rather than a mean of ratios — which is the difference between a real rate and
    an average of small-sample rates.
    """
    if plays_clean is None or len(plays_clean) == 0:
        return pd.DataFrame(columns=["game_id", "season", "week", "team",
                                     "opponent", "measure_id", "value",
                                     "num", "den"])
    p = annotate(plays_clean)
    rows: List[Dict] = []
    for gid, g in p.groupby("game_id", sort=True, observed=True):
        teams = sorted({str(t) for t in g["offense"].dropna().unique()}
                       | {str(t) for t in g["defense"].dropna().unique()})
        if len(teams) != 2:
            # A game whose play rows name anything other than exactly two teams is
            # a data defect. Skip it loudly rather than half-attributing it.
            continue
        season = int(g["season"].iloc[0])
        week = int(g["week"].iloc[0])
        for team in teams:
            opponent = teams[0] if teams[1] == team else teams[1]
            own = g[g["offense"].astype(str) == team]
            opp = g[g["offense"].astype(str) == opponent]
            rows.extend(_rows_for(gid, season, week, team, opponent, own, opp))
    if not rows:
        return pd.DataFrame(columns=["game_id", "season", "week", "team",
                                     "opponent", "measure_id", "value",
                                     "num", "den"])
    out = pd.DataFrame(rows)
    return out.sort_values(["season", "week", "game_id", "team", "measure_id"],
                           kind="mergesort").reset_index(drop=True)


#: measure ids this module produces that aggregate as a MEAN of per-game values
#: rather than a ratio of summed numerator to summed denominator
MEAN_AGGREGATED = frozenset({
    "plays_per_game_off", "plays_per_game_def",
})

#: measure ids that aggregate as a SUM of per-game counts
SUM_AGGREGATED = frozenset({
    "interceptions_thrown", "interceptions_forced", "fumbles_total_off",
    "fumbles_lost_off", "fumbles_forced_def", "fg_attempts", "punt_count",
})

#: measure ids whose season-to-date value must be recomputed from the POOLED raw
#: observations, because the statistic is not composable. A median of medians is
#: not a median, and averaging four one-attempt IQRs of zero yields zero.
POOLED_AGGREGATED = frozenset({"fg_distance_median", "fg_distance_iqr"})


def base_measure_id(measure_id: str) -> str:
    """`fg_distance_by_bucket::40_49` -> `fg_distance_by_bucket`.

    Distribution measures carry a component suffix so one registry entry covers
    the whole distribution rather than one entry per bucket.
    """
    return str(measure_id).split("::", 1)[0]


def robust_fg_summary(annotated: pd.DataFrame, team: str) -> Dict[str, object]:
    """Pooled, robust field-goal distance summary for one team.

    Median and IQR rather than a mean and an SD: a single 58-yard try moves a mean
    and does not move a median, and a kicking profile built on means is mostly a
    record of how often the coach sent the unit out in desperation.
    """
    own = annotated[(annotated["offense"].astype(str) == str(team))
                    & annotated["_fg_att"]]
    dists = pd.to_numeric(own["_fg_distance"], errors="coerce").dropna()
    buckets = {}
    for name, lo, hi in _FG_BUCKETS:
        b = own[(own["_fg_distance"] >= lo) & (own["_fg_distance"] < hi)]
        buckets[name] = {"attempts": int(len(b)), "made": int(b["_fg_good"].sum())}
    return {
        "n_attempts": int(len(dists)),
        "median": float(dists.median()) if len(dists) else None,
        "iqr": (float(dists.quantile(0.75) - dists.quantile(0.25))
                if len(dists) else None),
        "buckets": buckets,
    }
