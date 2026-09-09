"""Leakage, identity and double-counting.

The technique is the repo's own (`tests/test_leakage.py`): poison the future, then
assert the past did not move. Poisoning covers both directions that matter —
corrupting later rows AND appending rows that did not exist — because a whole-frame
aggregate survives the first and fails the second.
"""
from __future__ import annotations

import ast
import datetime as dt
import os

import pandas as pd
import pytest

from nflvalue import cfb_sim
from nflvalue.team_evidence import news as NEWS
from nflvalue.team_evidence import profile as TP
from nflvalue.team_evidence.contracts import (AvailabilityClass,
                                              EvidenceAvailability, Missingness)
from nflvalue.team_evidence.registry import MeasureRegistry
from nflvalue.team_evidence.sources import EvidenceSources, SourceState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NOW = dt.datetime(2026, 9, 9, 12, 0, tzinfo=dt.timezone.utc)

TEAMS = ("A", "B", "C", "D")


class FakeSources(EvidenceSources):
    """Approve the cache-backed sources without needing the gitignored parquet.

    The gate itself is tested separately, against the real manifest, in
    `test_an_unapproved_source_cannot_emit_a_value` and
    `test_missing_caches_fail_closed`.
    """

    APPROVED = ("model_drive_features", "model_ratings")

    def state(self, source_id, *, seasons=()):
        if source_id in self.APPROVED:
            return SourceState(
                source_id=source_id, availability=EvidenceAvailability.OK,
                missingness=Missingness.PRESENT, reason=None,
                source_url_or_id=f"internal:test:{source_id}",
                version="sha256:0123456789abcdef", retrieved_at=NOW,
                availability_class=AvailabilityClass.RECONSTRUCTED_WEEK_GRANULAR)
        return super().state(source_id, seasons=seasons)


def play(gid, week, off, dfn, play_type="Rush", down=1, distance=10,
         yards_to_goal=50, yards_gained=5, ppa=0.2, season=2026):
    return dict(game_id=gid, season=season, week=week, period=1, offense=off,
                defense=dfn, offense_score=0, defense_score=0, down=down,
                distance=distance, yards_to_goal=yards_to_goal,
                yards_gained=yards_gained, play_type=play_type, ppa=ppa,
                garbage_time=False)


def season_frames(weeks=(1, 2, 3, 4), season=2026):
    """A tiny two-game-per-week season with stable, week-dependent play values."""
    plays, games = [], []
    gid = 1000
    for week in weeks:
        for home, away in (("A", "B"), ("C", "D")):
            gid += 1
            for i in range(30):
                plays.append(play(gid, week, home, away, yards_gained=4 + week))
                plays.append(play(gid, week, away, home, yards_gained=3))
                plays.append(play(gid, week, home, away, "Pass Reception",
                                  yards_gained=9))
                plays.append(play(gid, week, away, home, "Pass Reception",
                                  yards_gained=6))
            games.append(dict(
                id=gid, season=season, week=week,
                start_date=f"2026-09-{week:02d}T23:00:00.000Z", neutral_site=False,
                home_team=home, away_team=away, home_points=24, away_points=17,
                home_conference="X", away_conference="Y",
                home_fcs=False, away_fcs=False))
    return pd.DataFrame(plays), pd.DataFrame(games)


def build(plays, games, week, **kw):
    kw.setdefault("with_opponent_adjustment", False)
    return TP.build_profiles(2026, week, data_dir=DATA, plays=plays, games=games,
                             sources=FakeSources(DATA), now=NOW, **kw)


def snapshot(profiles):
    return {team: tuple(sorted((o.measure_id, o.value, o.n, o.n_games,
                                o.availability.value)
                               for o in p.observations))
            for team, p in sorted(profiles.items())}


# --------------------------------------------------------------- the predicate
def test_the_local_predicate_agrees_with_cfb_sim():
    """`_strictly_before` is duplicated because `ratings.py` and `cfb_sim.py` are
    both hash-pinned. This asserts the copy cannot drift."""
    frame = pd.DataFrame([{"season": s, "week": w, "id": s * 100 + w}
                          for s in (2024, 2025, 2026) for w in range(0, 17)])
    for season in (2024, 2025, 2026):
        for week in range(0, 17):
            mine = TP._strictly_before(frame, season, week)
            theirs = cfb_sim._strictly_before(frame, season, week)
            assert sorted(mine["id"]) == sorted(theirs["id"]), (season, week)


def test_the_predicate_is_strict_in_both_directions():
    frame = pd.DataFrame([{"season": 2026, "week": w, "id": w} for w in range(1, 6)])
    kept = set(TP._strictly_before(frame, 2026, 4)["id"])
    assert kept == {1, 2, 3}, "week 4 itself must be excluded"
    assert 4 in set(TP._strictly_before(frame, 2026, 5)["id"]), \
        "and week 4 must appear once the cutoff moves"


# ------------------------------------------------------------------ poisoning
def test_a_poisoned_future_week_cannot_move_an_earlier_profile():
    plays, games = season_frames()
    clean = snapshot(build(plays, games, 3))
    poisoned = plays.copy()
    later = poisoned["week"] >= 3
    poisoned.loc[later, "yards_gained"] = 99
    poisoned.loc[later, "ppa"] = 42.0
    poisoned.loc[later, "play_type"] = "Interception"
    assert snapshot(build(poisoned, games, 3)) == clean


def test_appending_future_games_cannot_move_an_earlier_profile():
    """The failure the poison test alone misses: a whole-frame aggregate survives
    corrupted rows and breaks on rows that did not exist."""
    plays, games = season_frames(weeks=(1, 2))
    clean = snapshot(build(plays, games, 3))
    extra_plays, extra_games = season_frames(weeks=(3, 4, 5))
    both = pd.concat([plays, extra_plays], ignore_index=True)
    both_games = pd.concat([games, extra_games], ignore_index=True)
    assert snapshot(build(both, both_games, 3)) == clean


def test_the_cutoff_actually_bites_in_the_other_direction():
    """A test that only proves 'nothing moved' would also pass if the layer read
    nothing at all."""
    plays, games = season_frames()
    at3 = build(plays, games, 3)["A"].by_measure()["rush_success_rate_off"]
    at4 = build(plays, games, 4)["A"].by_measure()["rush_success_rate_off"]
    assert at3.n_games == 2 and at4.n_games == 3
    assert at3.n != at4.n


def test_a_later_report_cannot_alter_an_earlier_feature():
    kickoff = dt.datetime(2026, 9, 5, 23, 0, tzinfo=dt.timezone.utc)
    before = _fact("f1", published=kickoff - dt.timedelta(days=1))
    after = _fact("f2", published=kickoff + dt.timedelta(hours=2))
    kept = NEWS.facts_before_kickoff([before, after], kickoff)
    assert [f.fact_id for f in kept] == ["f1"]
    at_kickoff = _fact("f3", published=kickoff)
    assert NEWS.facts_before_kickoff([at_kickoff], kickoff) == []


def _fact(fact_id, published, *, team="A", status="out", quote="Smith will not play."):
    return NEWS.PersonnelFact(
        fact_id=fact_id, subject_type="player", subject_id="p1",
        subject_name="Smith", team=team, season=2026, week=2,
        claim="availability", status=status, status_confidence="reported",
        published_at=published, retrieved_at=published + dt.timedelta(minutes=5),
        source_id="qb_availability_feed", source_url=f"https://example.test/{fact_id}",
        citation_quote=quote, uncertainty="single beat report, not confirmed by the school")


# ---------------------------------------------------------------- sample counts
def test_sample_counts_are_the_summed_denominator_not_a_count_of_games():
    plays, games = season_frames(weeks=(1, 2))
    obs = build(plays, games, 3)["A"].by_measure()["rush_success_rate_off"]
    # A is home in both weeks: 30 rushes per game as offence, 2 games
    assert obs.n_games == 2
    assert obs.n == 60


def test_a_rate_is_a_ratio_of_sums_not_a_mean_of_rates():
    """A three-snap game must not weigh the same as a seventy-snap one."""
    plays = pd.DataFrame(
        [play(1, 1, "A", "B", yards_gained=9) for _ in range(60)]
        + [play(1, 1, "B", "A", yards_gained=1)]
        + [play(2, 2, "A", "B", yards_gained=1) for _ in range(2)]
        + [play(2, 2, "B", "A", yards_gained=1)])
    games = pd.DataFrame([
        dict(id=1, season=2026, week=1, start_date="2026-09-01T23:00:00.000Z",
             neutral_site=False, home_team="A", away_team="B", home_points=1,
             away_points=0, home_conference="X", away_conference="Y",
             home_fcs=False, away_fcs=False),
        dict(id=2, season=2026, week=2, start_date="2026-09-08T23:00:00.000Z",
             neutral_site=False, home_team="A", away_team="B", home_points=1,
             away_points=0, home_conference="X", away_conference="Y",
             home_fcs=False, away_fcs=False)])
    obs = build(plays, games, 3)["A"].by_measure()["rush_success_rate_off"]
    assert obs.n == 62
    assert obs.value == pytest.approx(60 / 62)      # not (1.0 + 0.0) / 2


def test_a_denominator_below_the_registered_gate_is_refused_not_rendered():
    plays = pd.DataFrame([play(1, 1, "A", "B", yards_gained=9) for _ in range(3)]
                         + [play(1, 1, "B", "A", yards_gained=1) for _ in range(3)])
    games = pd.DataFrame([dict(
        id=1, season=2026, week=1, start_date="2026-09-01T23:00:00.000Z",
        neutral_site=False, home_team="A", away_team="B", home_points=1,
        away_points=0, home_conference="X", away_conference="Y",
        home_fcs=False, away_fcs=False)])
    obs = build(plays, games, 2)["A"].by_measure()["rush_success_rate_off"]
    assert obs.availability is EvidenceAvailability.INSUFFICIENT_SAMPLE
    assert obs.value is None
    assert "gate" in obs.reason


# ------------------------------------------------------- identity and venues
def test_team_season_identity_does_not_bleed_across_seasons():
    plays_2025, games_2025 = season_frames(weeks=(1, 2, 3), season=2025)
    plays_2025["season"] = 2025
    games_2025["season"] = 2025
    plays_2026, games_2026 = season_frames(weeks=(1, 2))
    only = build(plays_2026, games_2026, 3)
    both = build(pd.concat([plays_2025, plays_2026], ignore_index=True),
                 pd.concat([games_2025, games_2026], ignore_index=True), 3)
    assert snapshot(only) == snapshot(both)


def test_neutral_site_games_are_carried_and_not_counted_as_home():
    plays, games = season_frames(weeks=(1, 2))
    games = games.copy()
    games.loc[games["week"] == 2, "neutral_site"] = True
    prof = build(plays, games, 3)["A"]
    neutral_ids = set(prof.neutral_site_games)
    assert neutral_ids
    assert neutral_ids <= set(prof.games_used)
    assert all(int(g) in set(games.loc[games["neutral_site"].astype(bool), "id"])
               for g in neutral_ids)


def test_effective_through_is_the_latest_kickoff_used():
    plays, games = season_frames(weeks=(1, 2))
    obs = build(plays, games, 3)["A"].by_measure()["rush_success_rate_off"]
    latest = pd.to_datetime(games["start_date"], utc=True).max().to_pydatetime()
    assert obs.effective_through == latest


# --------------------------------------------------- the double-count guard
def test_common_opponents_is_descriptive_only():
    plays, games = season_frames()
    profiles = build(plays, games, 4)
    out = TP.common_opponents(profiles, "A", "B")
    assert out["usage"] == "descriptive_only"
    numeric = [v for v in out.values() if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    assert not numeric, "a common-opponent differential is a second SOS adjustment"
    assert "strength" in out["note"].lower()


def test_no_profile_field_combines_common_opponent_results_into_a_number():
    plays, games = season_frames()
    profiles = build(plays, games, 4)
    for prof in profiles.values():
        for obs in prof.observations:
            assert "common_opponent" not in obs.measure_id
            assert "sos" not in obs.measure_id.split("_")
            assert "schedule_strength" not in obs.measure_id


def test_exactly_one_opponent_adjustment_exists_and_it_is_the_existing_one():
    reg = MeasureRegistry.load(DATA)
    methods = {s.opponent_adjustment_method for s in reg.measures.values()
               if s.opponent_adjusted}
    assert len(methods) == 1
    assert "solve_units" in methods.pop()


def test_unadjusted_measures_say_they_are_unadjusted():
    plays, games = season_frames()
    for prof in build(plays, games, 4).values():
        for obs in prof.observations:
            if obs.measure_id.endswith("_adj"):
                continue
            assert obs.opponent_adjusted is False
            assert obs.opponent_adjustment_method is None


def test_the_opponent_adjusted_family_comes_from_ratings_solve_units(monkeypatch):
    """Wiring test: the adjusted values must arrive from the existing solver, not
    from anything this package fits."""
    called = {}

    def fake(hist_seasons, season, week):
        called["args"] = (tuple(hist_seasons), season, week)
        return {k: {"A": 0.25, "B": -0.25} for k in TP.UNIT_MEASURES}

    monkeypatch.setattr(TP, "_opponent_adjusted_units", fake)
    plays, games = season_frames(weeks=(1, 2))
    profiles = TP.build_profiles(2026, 3, data_dir=DATA, plays=plays, games=games,
                                 sources=FakeSources(DATA), now=NOW,
                                 with_opponent_adjustment=True)
    assert called["args"][1:] == (2026, 3)
    obs = profiles["A"].by_measure()["rush_off_eff_adj"]
    assert obs.is_present and obs.value == pytest.approx(0.25)
    assert obs.opponent_adjusted is True
    assert "solve_units" in obs.opponent_adjustment_method


def test_this_package_defines_no_ridge_or_rating_solver():
    """A second rating engine would have to fit something. Nothing here does."""
    pkg = os.path.join(ROOT, "nflvalue", "team_evidence")
    banned = ("lstsq", "linalg.solve", "np.linalg", "Ridge", "def solve(",
              "def fit(", "KalmanModel")
    for name in sorted(os.listdir(pkg)):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(pkg, name), encoding="utf-8").read()
        for token in banned:
            assert token not in src, f"{name} contains {token!r}"


# ----------------------------------------------------------- fail-closed gate
def test_an_unapproved_source_cannot_emit_a_value():
    """Against the REAL manifest: drives are pending and the personnel feeds are
    not requested, so those measures must render as refusals."""
    plays, games = season_frames()
    profiles = TP.build_profiles(2026, 4, data_dir=DATA, plays=plays, games=games,
                                 with_opponent_adjustment=False, now=NOW)
    m = profiles["A"].by_measure()
    for mid in ("drives_per_game", "points_per_drive_off",
                "avg_start_yards_to_goal_off"):
        assert m[mid].availability is EvidenceAvailability.UNAVAILABLE_NOT_APPROVED
        assert m[mid].value is None
    for mid in ("personnel_status_qb", "personnel_status_unit",
                "roster_changes_verified"):
        assert m[mid].value is None
        assert m[mid].availability is not EvidenceAvailability.OK


def test_missing_caches_fail_closed_rather_than_defaulting(tmp_path):
    """An approved source whose artifact is not on disk yields NO value.

    Runs against a data dir holding only the two registries, so the result does not
    depend on whether the gitignored parquet caches happen to exist on this
    machine. Never zero, never a mean, never a silent omission.
    """
    import shutil
    data = tmp_path / "data"
    data.mkdir()
    for name in ("context_sources.json", "team_evidence_registry.json"):
        shutil.copy(os.path.join(DATA, name), data / name)
    plays, games = season_frames()
    profiles = TP.build_profiles(2026, 4, data_dir=str(data), plays=plays,
                                 games=games, with_opponent_adjustment=False,
                                 now=NOW)
    obs = profiles["A"].by_measure()["rush_success_rate_off"]
    assert obs.value is None
    assert obs.availability is EvidenceAvailability.UNAVAILABLE_NO_DATA
    assert obs.missingness is Missingness.NO_OBSERVATIONS
    assert "no cached artifact" in (obs.reason or "")


def test_no_observation_is_ever_silently_omitted():
    """Every registered measure appears in every profile, present or refused. A
    missing row and a forgotten row look identical, so there are no missing rows."""
    plays, games = season_frames()
    reg = MeasureRegistry.load(DATA)
    for prof in build(plays, games, 4).values():
        assert {o.measure_id for o in prof.observations} == set(reg.ids())


# ------------------------------------------------------------ module isolation
FORECAST_WORDS = ("projected_home_margin", "p_home_win", "projected_total",
                  "forecast_margin_sd")
FORECAST_MODULES = ("engine", "kalman_engine", "cfb_sim", "cfb_ot", "montecarlo",
                    "gameline", "bakeoff", "forecast_contract", "producers",
                    "snapshots", "model")


def test_no_calculation_module_names_a_forecast_field():
    """`tests/test_factor_isolation.py` forbids a forecast module from reading a
    registry. This package reads two, so it must not be a forecast module.

    `vault.py` is exempt from the word check and covered by the two tests below:
    it ECHOES a frozen forecast row into the immutable pregame record, which is
    the whole point of that record, and it computes nothing.
    """
    pkg = os.path.join(ROOT, "nflvalue", "team_evidence")
    for name in sorted(os.listdir(pkg)):
        if not name.endswith(".py") or name == "vault.py":
            continue
        src = open(os.path.join(pkg, name), encoding="utf-8").read()
        for word in FORECAST_WORDS:
            assert word not in src, f"{name} writes {word!r}"


def test_the_vault_writer_imports_no_forecast_machinery():
    src = open(os.path.join(ROOT, "nflvalue", "team_evidence", "vault.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        for mod in names:
            tail = mod.rsplit(".", 1)[-1]
            assert tail not in FORECAST_MODULES, f"vault.py imports {mod!r}"


def test_the_pregame_record_echoes_and_invents_nothing():
    """Every number in the rendered record must be present in the row it was
    handed. A pregame record that computes anything is not a record."""
    from nflvalue.team_evidence import vault as V
    row = {"status": "priced", "model_family": "ridge_cross_division",
           "model_version": "abc123", "projected_home_margin": 40.534,
           "forecast_margin_sd": 17.896, "p_home_win": 0.9882,
           "forecast_sd_basis": "cross_division_ridge_wk1_15_2021_2025",
           "generated_at_utc": "20260902T234416Z", "snapshot_id": "s1",
           "source_digest": "d1", "config_hash": "c1"}
    body = V.pregame_body(row, snapshot_stamp="20260902T234416Z")
    import re
    supplied = {str(v) for v in row.values()} | {"20260902T234416Z"}
    for token in re.findall(r"-?\d+\.\d+", body):
        assert token in supplied, f"{token!r} was invented, not echoed"
    for key, value in row.items():
        assert str(value) in body, f"{key} was dropped from the record"

    for name in sorted(os.listdir(os.path.join(ROOT, "nflvalue"))):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(ROOT, "nflvalue", name),
                              encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "team_evidence" not in node.module, f"{name} imports the layer"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "team_evidence" not in alias.name, f"{name} imports the layer"
