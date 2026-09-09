"""As-of-week team-season profiles.

`build_profiles(season, week, ...)` returns one `TeamSeasonProfile` per team,
built from games strictly earlier than `(season, week)` and from nothing else.

THE TEMPORAL RULE
-----------------
`_strictly_before` is the repo's standard predicate:

    (season < S) | ((season == S) & (week < W))

Never `<=`. It is duplicated here rather than imported because the two modules
that host it — `nflvalue/ratings.py:832` and `nflvalue/cfb_sim.py:347` — are both
sha256-pinned by `data/evidence_manifest.json`; a shared helper would have to live
in one of them. `tests/test_team_evidence_leakage.py::
test_the_local_predicate_agrees_with_cfb_sim` asserts the copy cannot drift.

OPPONENT ADJUSTMENT
-------------------
Exactly one adjustment is applied, and it is the EXISTING one:
`ratings.solve_units`, whose design matrix carries the opponent's column on every
row. It covers rush/pass efficiency and explosiveness. Everything else ships
opponent-UNadjusted with `opponent_adjusted=False`, and the profile carries
`opponents_faced` so a consumer can condition on schedule itself.

WHY NOT ADJUST THE REST
-----------------------
Because adjusting the rest means writing a second rating engine, and because
opponent strength is ALREADY inside the shipped ratings three times over (the
per-game opponent term, the conference blocks, and connectivity). A raw
season-to-date aggregate fed beside a rating-based prediction re-injects the same
schedule signal. So does an "average opponent rating" SOS feature. So does a
common-opponent difference, which is the pairwise special case of what the ridge
solves globally with less information and no regularisation.

`common_opponents()` below therefore returns a DESCRIPTIVE listing and nothing
else — no summed differential, no rating. `tests/test_team_evidence_leakage.py::
test_common_opponents_is_descriptive_only` asserts no profile field combines
common-opponent results into a strength number.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .. import ingest, ratings
from . import measures as MEAS
from .contracts import (AvailabilityClass, EvidenceAvailability, Missingness,
                        Observation, TeamSeasonProfile)
from .registry import MeasureRegistry, MeasureSpec
from .sources import EvidenceSources, SourceState

#: the source a unit rating is attributed to (approved; `nflvalue.engine.solve_table`)
UNIT_SOURCE_ID = "model_ratings"

#: which `solve_units` output maps to which registered measure
UNIT_MEASURES = {
    "rush_off_eff": "rush_off_eff_adj", "rush_off_exp": "rush_off_exp_adj",
    "pass_off_eff": "pass_off_eff_adj", "pass_off_exp": "pass_off_exp_adj",
    "rush_def_eff": "rush_def_eff_adj", "rush_def_exp": "rush_def_exp_adj",
    "pass_def_eff": "pass_def_eff_adj", "pass_def_exp": "pass_def_exp_adj",
}


def _strictly_before(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows from before `(season, week)`. Week granularity, never season."""
    if frame is None or len(frame) == 0:
        return frame
    s = pd.to_numeric(frame["season"], errors="coerce")
    w = pd.to_numeric(frame["week"], errors="coerce")
    return frame[(s < int(season)) | ((s == int(season)) & (w < int(week)))]


def _kickoff_of(games: pd.DataFrame, game_ids: Sequence[int]) -> Optional[dt.datetime]:
    """The latest kickoff among the games a value is built from — the value's
    `effective_through`. A value cannot be effective for anything earlier."""
    if not len(game_ids) or "start_date" not in games.columns:
        return None
    sub = games[games["id"].isin(list(game_ids))]
    if sub.empty:
        return None
    stamps = pd.to_datetime(sub["start_date"], errors="coerce", utc=True).dropna()
    if stamps.empty:
        return None
    return stamps.max().to_pydatetime()


def _aggregate(rows: pd.DataFrame, spec: MeasureSpec) -> Tuple[Optional[float], int, int]:
    """Season-to-date value, denominator, and number of team-games.

    Rates are a ratio of summed numerator to summed denominator — never a mean of
    per-game rates, which over-weights a three-snap game against a seventy-snap
    one.
    """
    n_games = int(rows["game_id"].nunique())
    if spec.measure_id in MEAS.SUM_AGGREGATED:
        return float(rows["num"].sum()), int(rows["den"].sum()), n_games
    if spec.measure_id in MEAS.MEAN_AGGREGATED:
        vals = rows["value"].dropna()
        return (float(vals.mean()) if len(vals) else None, int(len(vals)), n_games)
    den = float(rows["den"].sum())
    if den <= 0:
        return None, 0, n_games
    return float(rows["num"].sum()) / den, int(den), n_games


def _missing(spec: MeasureSpec, team: str, season: int, week: int, *,
             availability: EvidenceAvailability, missingness: Missingness,
             reason: str, n: int = 0, n_games: int = 0,
             state: Optional[SourceState] = None) -> Observation:
    return Observation.missing(
        measure_id=spec.measure_id, team=team, season=season, as_of_week=week,
        label=spec.label, unit=spec.unit, availability=availability,
        missingness=missingness, reason=reason,
        availability_class=AvailabilityClass(spec.availability_class),
        n=n, n_games=n_games,
        source_id=spec.source_id,
        source_url_or_id=state.source_url_or_id if state else None,
        source_version=state.version if state else None,
        retrieved_at=state.retrieved_at if state else None,
        caveat=spec.caveat)


def build_profiles(season: int, week: int, *, data_dir: str,
                   seasons_of_history: Optional[Sequence[int]] = None,
                   teams: Optional[Iterable[str]] = None,
                   sources: Optional[EvidenceSources] = None,
                   registry: Optional[MeasureRegistry] = None,
                   plays: Optional[pd.DataFrame] = None,
                   games: Optional[pd.DataFrame] = None,
                   with_opponent_adjustment: bool = True,
                   now: Optional[dt.datetime] = None,
                   ) -> Dict[str, TeamSeasonProfile]:
    """Build one profile per team, as of `(season, week)`.

    `plays`/`games` may be injected (tests do); otherwise they are loaded through
    the existing `nflvalue.ingest` loaders. Current-season profiles use only the
    current season's completed games — a team-season identity does not inherit its
    own prior year — while the opponent adjustment is fitted on whatever history
    is passed, because that is what `solve_units` needs to identify the graph.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    registry = registry or MeasureRegistry.load(data_dir)
    sources = sources or EvidenceSources(data_dir)
    hist_seasons = tuple(sorted(set(seasons_of_history or [season])))

    if plays is None:
        plays = ingest.load_plays_clean([season])
    if games is None:
        games = ingest.load_games(hist_seasons)

    # ---- gate every source ONCE, before any number is computed ---------------
    states: Dict[str, SourceState] = {
        sid: sources.state(sid, seasons=(season,)) for sid in registry.source_ids()}
    # The unit ratings are computed by `ratings.solve_units` from the plays cache;
    # they never read data/ratings_{season}.parquet, so they are versioned against
    # what actually produced them rather than against a file they do not open.
    if UNIT_SOURCE_ID in states:
        states[UNIT_SOURCE_ID] = sources.derived_state(
            UNIT_SOURCE_ID, [states.get("model_drive_features")])

    # ---- current-season, strictly-earlier plays -----------------------------
    season_plays = plays[pd.to_numeric(plays["season"], errors="coerce") == int(season)]
    earlier = _strictly_before(season_plays, season, week)
    per_game = MEAS.team_game_measures(earlier)
    annotated = MEAS.annotate(earlier) if len(earlier) else MEAS.annotate(season_plays.head(0))

    all_teams = sorted({str(t) for t in per_game["team"].unique()}) if len(per_game) else []
    if teams is not None:
        wanted = {str(t) for t in teams}
        all_teams = sorted(wanted | set(all_teams)) if wanted else all_teams

    # ---- the EXISTING opponent adjustment -----------------------------------
    unit_ratings: Dict[str, Dict[str, float]] = {}
    unit_error: Optional[str] = None
    if with_opponent_adjustment:
        try:
            unit_ratings = _opponent_adjusted_units(hist_seasons, season, week)
        except Exception as exc:                              # noqa: BLE001
            unit_error = f"{type(exc).__name__}: {exc}"

    profiles: Dict[str, TeamSeasonProfile] = {}
    for team in all_teams:
        rows = per_game[per_game["team"] == team] if len(per_game) else per_game
        game_ids = tuple(sorted(int(g) for g in rows["game_id"].unique())) if len(rows) else ()
        opponents = tuple(sorted({str(o) for o in rows["opponent"].unique()})) if len(rows) else ()
        effective_through = _kickoff_of(games, game_ids)

        obs: List[Observation] = []
        for spec in sorted(registry.measures.values(), key=lambda s: s.measure_id):
            obs.append(_observe(spec, team, season, week, rows, states,
                                effective_through, unit_ratings, unit_error,
                                annotated))

        neutral = ()
        if len(games) and game_ids and "neutral_site" in games.columns:
            g = games[games["id"].isin(list(game_ids))]
            neutral = tuple(sorted(int(i) for i in
                                   g.loc[g["neutral_site"].astype(bool), "id"]))

        profiles[team] = TeamSeasonProfile(
            team=team, season=int(season), as_of_week=int(week),
            generated_at_utc=now, observations=tuple(obs), games_used=game_ids,
            opponents_faced=opponents, neutral_site_games=neutral)
    return profiles


def _observe(spec: MeasureSpec, team: str, season: int, week: int,
             rows: pd.DataFrame, states: Dict[str, SourceState],
             effective_through: Optional[dt.datetime],
             unit_ratings: Dict[str, Dict[str, float]], unit_error: Optional[str],
             annotated: pd.DataFrame) -> Observation:
    state = states.get(spec.source_id)

    # 1. registry-level block always wins — a blocked field never reaches the data
    if spec.status == "blocked":
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.BLOCKED_UNVERIFIED_AVAILABILITY,
                        missingness=(Missingness.SOURCE_NOT_APPROVED
                                     if spec.source_id in ("qb_availability_feed",
                                                           "unit_availability_feed",
                                                           "cfbd_portal")
                                     else Missingness.UNVERIFIED_FIELD_SEMANTICS),
                        reason=spec.blocked_reason or "blocked", state=state)
    if spec.status == "gated_pending_source":
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NOT_APPROVED,
                        missingness=Missingness.SOURCE_NOT_APPROVED,
                        reason=spec.blocked_reason or "source not approved",
                        state=state)

    # 2. source gate
    if state is None or not state.ok:
        return _missing(spec, team, season, week,
                        availability=(state.availability if state else
                                      EvidenceAvailability.UNAVAILABLE_NO_SOURCE),
                        missingness=(state.missingness if state else
                                     Missingness.SOURCE_NOT_DECLARED),
                        reason=(state.reason if state and state.reason else
                                f"source {spec.source_id!r} unavailable"),
                        state=state)

    # 3. the opponent-adjusted family comes from ratings.solve_units, not from the
    #    per-game frame
    if spec.opponent_adjusted:
        return _observe_unit(spec, team, season, week, state, effective_through,
                             unit_ratings, unit_error)

    # 4. pooled statistics are recomputed from the raw observations
    if spec.measure_id in MEAS.POOLED_AGGREGATED:
        summary = MEAS.robust_fg_summary(annotated, team)
        key = "median" if spec.measure_id.endswith("median") else "iqr"
        n = int(summary["n_attempts"])
        if n < max(spec.min_n, 1) or summary[key] is None:
            return _missing(spec, team, season, week,
                            availability=EvidenceAvailability.INSUFFICIENT_SAMPLE,
                            missingness=Missingness.BELOW_SAMPLE_GATE,
                            reason=(f"{n} field-goal attempts before wk{week}; the "
                                    f"registered gate is {spec.min_n}"),
                            n=n, state=state)
        return Observation.present(
            measure_id=spec.measure_id, team=team, season=season, as_of_week=week,
            label=spec.label, unit=spec.unit, value=float(summary[key]),
            n=n, n_games=int(rows["game_id"].nunique()) if len(rows) else 0,
            availability_class=AvailabilityClass(spec.availability_class),
            source_id=spec.source_id, source_url_or_id=state.source_url_or_id,
            source_version=state.version, retrieved_at=state.retrieved_at,
            effective_through=effective_through, caveat=spec.caveat)

    # 5. everything else aggregates from the per-game frame
    sub = rows[rows["measure_id"] == spec.measure_id] if len(rows) else rows
    if sub is None or len(sub) == 0:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                        missingness=Missingness.NO_PRIOR_GAMES,
                        reason=(f"no completed games for {team} before "
                                f"{season} wk{week}"),
                        state=state)
    value, n, n_games = _aggregate(sub, spec)
    if value is None:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                        missingness=Missingness.NO_OBSERVATIONS,
                        reason=(f"{spec.measure_id} has an empty denominator over "
                                f"{n_games} game(s) — the opportunity never arose"),
                        n=0, n_games=n_games, state=state)
    if n < spec.min_n:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.INSUFFICIENT_SAMPLE,
                        missingness=Missingness.BELOW_SAMPLE_GATE,
                        reason=(f"denominator {n} is below the registered gate "
                                f"{spec.min_n} — a rate over {n} is not a rate"),
                        n=n, n_games=n_games, state=state)
    if effective_through is None:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                        missingness=Missingness.AVAILABILITY_UNVERIFIABLE,
                        reason=("no kickoff timestamp for the games this value is "
                                "built from, so effective_through cannot be set"),
                        n=n, n_games=n_games, state=state)
    return Observation.present(
        measure_id=spec.measure_id, team=team, season=season, as_of_week=week,
        label=spec.label, unit=spec.unit, value=value, n=n, n_games=n_games,
        availability_class=AvailabilityClass(spec.availability_class),
        source_id=spec.source_id, source_url_or_id=state.source_url_or_id,
        source_version=state.version, retrieved_at=state.retrieved_at,
        effective_through=effective_through,
        game_ids=tuple(sorted(int(g) for g in sub["game_id"].unique())),
        caveat=spec.caveat)


def _observe_unit(spec: MeasureSpec, team: str, season: int, week: int,
                  state: SourceState, effective_through: Optional[dt.datetime],
                  unit_ratings: Dict[str, Dict[str, float]],
                  unit_error: Optional[str]) -> Observation:
    key = next((k for k, v in UNIT_MEASURES.items() if v == spec.measure_id), None)
    if unit_error is not None or key is None or key not in unit_ratings:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                        missingness=Missingness.NO_PRIOR_GAMES,
                        reason=(unit_error or "ratings.solve_units produced no "
                                f"{spec.measure_id} — usually too little history "
                                f"before {season} wk{week} to identify the graph"),
                        state=state)
    table = unit_ratings[key]
    if team not in table:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                        missingness=Missingness.NO_PRIOR_GAMES,
                        reason=(f"{team} is not in the opponent-adjusted solve "
                                "(no qualifying earlier games, or pooled as FCS)"),
                        state=state)
    if effective_through is None:
        return _missing(spec, team, season, week,
                        availability=EvidenceAvailability.UNAVAILABLE_NO_DATA,
                        missingness=Missingness.AVAILABILITY_UNVERIFIABLE,
                        reason="no kickoff timestamp to set effective_through",
                        state=state)
    return Observation.present(
        measure_id=spec.measure_id, team=team, season=season, as_of_week=week,
        label=spec.label, unit=spec.unit, value=float(table[team]), n=1, n_games=0,
        availability_class=AvailabilityClass(spec.availability_class),
        source_id=spec.source_id, source_url_or_id=state.source_url_or_id,
        source_version=state.version, retrieved_at=state.retrieved_at,
        effective_through=effective_through, opponent_adjusted=True,
        opponent_adjustment_method=spec.opponent_adjustment_method,
        caveat=spec.caveat)


def _opponent_adjusted_units(hist_seasons: Sequence[int], season: int,
                             week: int) -> Dict[str, Dict[str, float]]:
    """Call the EXISTING adjustment. This function builds no rating of its own.

    `ratings.game_frame` resolves conferences and FCS pooling; `ratings.play_units`
    produces the per-(game, offence, defence, phase) rows; `ratings.solve_units`
    performs the hierarchical-ridge solve and applies its own strictly-earlier
    filter at `ratings.py:832`. Nothing here is written back.
    """
    gf = ratings.game_frame(hist_seasons)
    plays = ingest.load_plays_clean(sorted(set(hist_seasons)))
    units = ratings.play_units(_strictly_before(plays, season, week))
    if units.empty:
        raise ValueError("no unit rows before the cutoff")
    return ratings.solve_units(gf, units, int(season), int(week),
                               ratings.load_params())


def common_opponents(profiles: Dict[str, TeamSeasonProfile], home: str,
                     away: str) -> Dict[str, object]:
    """DESCRIPTIVE listing of shared opponents. Deliberately returns no number
    that could be read as a strength differential.

    Opponent strength is already inside the ratings. "A beat X by 20, B beat X by
    3, so A is 17 better" is the pairwise special case of the global ridge solve,
    with less information and no regularisation, and its errors correlate with the
    rating's own — so it looks predictive in-sample and is not. This function
    exists so the note can SAY who both teams played, and stops there.
    """
    ph, pa = profiles.get(home), profiles.get(away)
    if ph is None or pa is None:
        return {"shared": [], "usage": "descriptive_only", "note": "unknown team"}
    shared = sorted(set(ph.opponents_faced) & set(pa.opponents_faced))
    return {
        "shared": shared,
        "home_opponents": list(ph.opponents_faced),
        "away_opponents": list(pa.opponents_faced),
        "usage": "descriptive_only",
        "note": ("Listing only. Opponent strength is already absorbed by "
                 "ratings.solve_units and by the shipped margin engine; summing "
                 "these results into a differential would be a second "
                 "strength-of-schedule adjustment on top of the first."),
    }
