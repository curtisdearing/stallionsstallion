"""Independent review counterexamples: known bugs remain strict expected failures.

These tests do not approve sources, change a registry, or activate an evidence
feature. A fix produces XPASS (and fails CI) until its review marker is removed.
Synthetic fixtures need no CFBD caches or network.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from nflvalue.context.contracts import Availability
from nflvalue.context.sources import SourceError
from nflvalue.team_evidence import measures as M, news as N, profile as P
from nflvalue.team_evidence.contracts import (
    AvailabilityClass, EvidenceAvailability, EvidenceContractError, Missingness,
    Observation,
)
from nflvalue.team_evidence.sources import EvidenceSources, SourceState

DATA = str(Path(__file__).resolve().parents[1] / "data")
NOW = dt.datetime(2026, 9, 9, tzinfo=dt.timezone.utc)
KICK = dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc)


def play(gid=1, week=1, offense="A", defense="B", play_type="Rush", down=1,
         distance=10, yards_gained=6, yards_to_goal=23):
    return dict(game_id=gid, season=2026, week=week, period=1, offense=offense,
                defense=defense, offense_score=0, defense_score=0, down=down,
                distance=distance, yards_to_goal=yards_to_goal,
                yards_gained=yards_gained, play_type=play_type, ppa=0.2,
                garbage_time=False)


class FixtureSources(EvidenceSources):
    def state(self, source_id, *, seasons=()):
        if source_id in {"model_drive_features", "model_ratings", "cfbd_drives"}:
            return SourceState(source_id, EvidenceAvailability.OK,
                               Missingness.PRESENT, None,
                               "internal:review-fixture", "sha256:fixture", NOW,
                               AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR)
        return super().state(source_id, seasons=seasons)


@pytest.fixture
def frames():
    plays = pd.DataFrame([play(gid=w, week=w, offense=t, defense=o)
                          for w in (1, 2) for t, o in (("A", "B"), ("B", "A"))
                          for _ in range(30)])
    games = pd.DataFrame([dict(id=w, season=2026, week=w,
                              start_date=f"2026-09-0{w}T12:00:00Z",
                              neutral_site=False, home_team="A", away_team="B",
                              home_points=24, away_points=17)
                          for w in (1, 2)])
    return plays, games


def profile(plays, games, *, week=3, **kwargs):
    kwargs.setdefault("with_opponent_adjustment", False)
    return P.build_profiles(2026, week, data_dir=DATA, plays=plays, games=games,
                            sources=FixtureSources(DATA), now=NOW, **kwargs)


def observed(rows, measure_id):
    out = M.team_game_measures(pd.DataFrame(rows))
    return out[(out.team == "A") & (out.measure_id == measure_id)].iloc[0]


@pytest.mark.xfail(strict=True, reason="ER01: short-yardage counts punts/FGs/timeouts")
def test_short_yardage_excludes_non_scrimmage_opportunities():
    row = observed([play(down=3, distance=2, yards_gained=2),
                    play(play_type="Punt", down=4, distance=2, yards_gained=0),
                    play(play_type="Field Goal Good", down=4, distance=2, yards_gained=0),
                    play(play_type="Timeout", down=3, distance=2, yards_gained=0),
                    play(offense="B", defense="A")],
                   "short_yardage_conversion_rate_off")
    assert row.den == 1
    assert row.num == 1
    assert row.value == 1.0


@pytest.mark.xfail(strict=True, reason="ER09: missing yards silently become a failure")
def test_missing_yards_are_not_a_failed_graded_play():
    row = observed([play(yards_gained=float("nan")),
                    play(offense="B", defense="A")], "rush_success_rate_off")
    assert row.den == 0
    assert pd.isna(row.value)


@pytest.mark.xfail(strict=True, reason="ER05: population is built only from observed plays")
def test_known_schedule_teams_remain_present_before_first_play(frames):
    plays, games = frames
    out = profile(plays, games, week=1)
    assert set(out) == {"A", "B"}
    assert all(o.value is None for p in out.values() for o in p.observations)


@pytest.mark.xfail(strict=True, reason="ER07: suffixed FG bucket IDs never reach profiles")
def test_fg_bucket_profile_exists_when_attempts_exist(frames):
    plays, games = frames
    plays.loc[plays.offense == "A", "play_type"] = "Field Goal Good"
    observations = profile(plays, games)["A"].by_measure()
    assert observations["fg_attempts"].value == 60
    assert observations["fg_distance_by_bucket"].is_present


@pytest.mark.xfail(strict=True, reason="ER08: Observation.present accepts NaN")
def test_nonfinite_present_values_are_rejected():
    with pytest.raises(EvidenceContractError):
        Observation.present(measure_id="test", team="A", season=2026, as_of_week=2,
                            label="test", unit="rate", value=float("nan"), n=1,
                            n_games=1, availability_class=AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR,
                            source_id="fixture", source_url_or_id="internal:fixture",
                            source_version="sha256:fixture", retrieved_at=NOW,
                            effective_through=KICK)


@pytest.mark.xfail(strict=True, reason="ER08: dataclass bypass lacks invariant validation")
def test_raw_dataclass_replacement_cannot_drop_present_value_provenance(frames):
    plays, games = frames
    obs = profile(plays, games)["A"].by_measure()["rush_success_rate_off"]
    with pytest.raises(EvidenceContractError):
        dataclasses.replace(obs, source_id=None, source_version=None)


class RequestedFeedApproved:
    """Only the requested feed is approved; other IDs are deliberately unknown."""
    def gate(self, source_id):
        if source_id == "qb_availability_feed":
            return Availability.OK, None
        raise SourceError("undeclared source: " + source_id)

    def url(self, source_id):
        return "https://example.test/requested-feed"


def fact(*, source_id="qb_availability_feed", published=None):
    published = published or KICK - dt.timedelta(days=1)
    return N.PersonnelFact(
        fact_id="review-fact", subject_type="player", subject_id="player:1",
        subject_name="J. Smith", team="A", season=2026, week=2,
        claim="availability", status="out", status_confidence="reported",
        published_at=published, retrieved_at=published + dt.timedelta(minutes=1),
        source_id=source_id, source_url="https://example.test/report",
        citation_quote="Smith will not play.", uncertainty="single unverified report")


@pytest.mark.xfail(strict=True, reason="ER04: supplied facts bypass their own source gate")
def test_each_fact_source_requires_approval():
    try:
        _, facts = N.collect("A", 2026, 2, data_dir=DATA,
                             manifest=RequestedFeedApproved(),
                             facts=[fact(source_id="undeclared_source")], kickoff=KICK)
    except (N.NewsContractError, SourceError):
        return
    assert not facts


@pytest.mark.xfail(strict=True, reason="ER04: collect has an optional temporal cutoff")
def test_news_collection_without_cutoff_fails_closed():
    try:
        _, facts = N.collect("A", 2026, 2, data_dir=DATA,
                             manifest=RequestedFeedApproved(),
                             facts=[fact(published=dt.datetime(2030, 1, 1,
                                                              tzinfo=dt.timezone.utc))])
    except N.NewsContractError:
        return
    assert not facts


@pytest.mark.xfail(strict=True, reason="ER02: unit support n=1/n_games=0 is hardcoded")
def test_adjusted_unit_observation_carries_actual_training_support(frames, monkeypatch):
    plays, games = frames
    monkeypatch.setattr(P, "_opponent_adjusted_units",
                        lambda *_: {key: {"A": 0.2} for key in P.UNIT_MEASURES})
    obs = profile(plays, games, with_opponent_adjustment=True)["A"].by_measure()["rush_off_eff_adj"]
    assert obs.is_present
    assert obs.n_games > 0
    assert obs.game_ids


@pytest.mark.xfail(strict=True, reason="ER06: approved drive source still unconditionally blocked")
def test_approved_drive_source_does_not_still_report_not_approved(frames):
    plays, games = frames
    obs = profile(plays, games)["A"].by_measure()["drives_per_game"]
    assert obs.availability is not EvidenceAvailability.UNAVAILABLE_NOT_APPROVED


@pytest.mark.parametrize("ridge,decay", [(3, 0.0), (3000, 0.05)])
def test_score_adapter_matches_every_centered_solver_training_total(monkeypatch, ridge, decay):
    """Independent check against the solver's actual design, not its mean only."""
    import numpy as np
    from nflvalue import final_score_research as F, ratings as R

    rng = np.random.default_rng(1024)
    rows = []
    for season in (2019, 2020, 2021):
        for week in range(1, 7):
            for home, away in (("A", "B"), ("B", "C"), ("C", "A")):
                rows.append(dict(id=len(rows), season=season, week=week,
                                 home_team=home, away_team=away,
                                 home_points=int(rng.integers(7, 60)),
                                 away_points=int(rng.integers(0, 60)),
                                 home_fbs=True, away_fbs=True,
                                 home_conf="Big Ten" if season == 2019 and home == "A" else "SEC",
                                 away_conf="ACC", neutral_site=bool(rng.uniform() < .35)))
    captured = {}
    original = R._ridge

    def record_actual_design(X, y, w, lam, theta0, max_condition):
        result = original(X, y, w, lam, theta0, max_condition)
        captured["fitted"] = X @ result[0]
        return result

    monkeypatch.setattr(R, "_ridge", record_actual_design)
    fitted = F.fit_total(pd.DataFrame(rows), 2021, 3, window=3, population="p4",
                         ridge=ridge, decay=decay)
    reconstructed = np.asarray([
        F.ridge_total(fitted, r.home_team, r.away_team, r.neutral_site,
                      r.home_conf, r.away_conf)[0]
        for r in fitted.games.itertuples()])
    expected = captured["fitted"].reshape(-1, 2).sum(axis=1)
    assert np.allclose(reconstructed, expected, rtol=0, atol=1e-10)
