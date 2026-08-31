"""The demoted early-season renderer, and the shared producer behind it.

`scripts/predict_2026_wk01.py` used to be the SECOND writer of the live
artifact `data/predictions_{season}_wk{NN}.csv`: it and `scripts/weekly_loop.py`
emitted different schemas into one filename and whichever ran last won. It is a
demo/fixture renderer now -- a thin wrapper over
`nflvalue.producers.build_early_week`, the same producer the weekly loop calls
for weeks 0-3, confined to `data/demo/`.

Every invariant the old writer carried is therefore re-asserted here against
the shared producer instead of against a private assembly path: one row per
scheduled game, cross-division matchups refused rather than priced by
substituting the pooled FCS pseudo-team, no numbers on a refused row, the
preseason prior assembled inside `ratings.solve` rather than injected through
its test-only hook, a row set that is not its schedule failing the run, and the
Eastern calendar deciding `slate`/`public_date`.

The demotion itself is only worth something if three further things hold, so
they are locked too:
  1. `check_destination` refuses the live artifact's filename, `data/snapshots/`,
     `data/site/` and the `data/` root -- this script cannot become a publisher
     again by being pointed at one;
  2. the demo and the weekly loop's early-season path emit identical contract
     rows, because there is one producer and one artifact builder;
  3. the early-season `forecast_margin_sd` is the registered out-of-sample
     basis from `data/forecast_sd_basis.json`, never the rating band --
     `rating_parameter_sd` is the separate, much smaller technical quantity.
"""

import contextlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
sys.path.insert(0, _ROOT)

from nflvalue import engine as EG, forecast_contract as FC  # noqa: E402
from nflvalue import ingest, producers as PR, ratings as R  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PW = _load("predict_2026_wk01")
WL = _load("weekly_loop")

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _history() -> pd.DataFrame:
    """Two prior seasons of a tiny connected FBS round robin."""
    margins = {("AAA", "BBB"): 14, ("CCC", "DDD"): -3, ("AAA", "CCC"): 10,
               ("BBB", "DDD"): 7, ("AAA", "DDD"): 21, ("BBB", "CCC"): -6}
    rows, gid = [], 0
    for season in (2024, 2025):
        wk = 1
        for (home, away), margin in margins.items():
            gid += 1
            rows.append({
                "id": gid, "season": season, "week": wk,
                "home_team": home, "away_team": away,
                "home_conf": "X" if home in ("AAA", "BBB") else "Y",
                "away_conf": "X" if away in ("AAA", "BBB") else "Y",
                "home_points": 24 + margin, "away_points": 24,
                "neutral_site": False, "home_fbs": True, "away_fbs": True,
                "margin": float(margin)})
            if gid % 2 == 0:
                wk += 1
    return pd.DataFrame(rows)


SCHEDULE = pd.DataFrame([
    {"id": 501, "week": 1, "start_date": "2026-08-29T23:00:00Z",
     "home_team": "AAA", "away_team": "BBB", "neutral_site": False},
    {"id": 502, "week": 1, "start_date": "2026-09-05T16:00:00Z",
     "home_team": "CCC", "away_team": "Tiny State", "neutral_site": False},
    {"id": 503, "week": 1, "start_date": "2026-09-05T18:00:00Z",
     "home_team": "Tiny State", "away_team": "Smaller State",
     "neutral_site": False},
    {"id": 504, "week": 1, "start_date": "2026-09-05T20:00:00Z",
     "home_team": "New FBS", "away_team": "DDD", "neutral_site": False},
    {"id": 505, "week": 2, "start_date": "2026-09-12T16:00:00Z",
     "home_team": "AAA", "away_team": "CCC", "neutral_site": False},
])

TEAMS_2026 = pd.DataFrame(
    [{"team": t, "conference": "X" if t in ("AAA", "BBB") else "Y",
      "is_fbs": True} for t in TEAMS]
    + [{"team": "New FBS", "conference": "Y", "is_fbs": True},
       {"team": "Tiny State", "conference": "FCS", "is_fbs": False},
       {"team": "Smaller State", "conference": "FCS", "is_fbs": False}])

#: The sandbox's own basis manifest. Deliberately NOT the shipped numbers: the
#: SD and the id are distinctive, so a row carrying them can only have got them
#: from the registry -- which is the claim, not a coincidence a shared constant
#: would also satisfy. The shipped file gets its own test below.
BASIS_MANIFEST = {
    "schema_version": 1,
    "generated_at_utc": "2026-08-30T00:00:00Z",
    "prereg": "2026-08-30T17:20Z",
    "bases": [{
        "basis_id": "test_early_ridge_basis",
        "model_family": "ridge_preseason_prior",
        "covers_weeks": [0, 1, 2, 3],
        "sd_source": "pooled_out_of_sample_residual",
        "forecast_margin_sd": 17.25,
        "usable": True,
        "n": 724,
        "window": "synthetic fixture: stands in for the registered early-week "
                  "out-of-sample residual SD",
        "source": "tests/test_predict_wk01_writer.py",
    }],
}

SEASON, WEEK = 2026, 1


def _phi(z: float) -> float:
    """Normal CDF, re-derived here rather than imported from the contract.

    The point of these checks is that the published probability belongs to the
    published distribution; borrowing the contract's own helper would let one
    bug satisfy both sides.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _loader(schedule: pd.DataFrame, teams: pd.DataFrame = TEAMS_2026):
    """Stand in for the cached CFBD tables `ingest.load` would read."""
    def fake_load(kind, seasons, **kw):
        if kind == "teams":
            return teams
        if kind == "games":
            return schedule
        return pd.DataFrame()
    return fake_load


@pytest.fixture()
def demo(tmp_path, monkeypatch):
    """The demo renderer with every cached input mocked and `data/` sandboxed.

    `DATA` and `DEMO_DIR` both move into the sandbox so `check_destination`
    still judges paths by the same relative shapes it judges the real tree by.
    """
    data_dir = tmp_path / "data"
    (data_dir / "demo").mkdir(parents=True)
    (data_dir / "forecast_sd_basis.json").write_text(json.dumps(BASIS_MANIFEST))

    monkeypatch.setattr(PW, "HISTORY", [2024, 2025])
    monkeypatch.setattr(PW, "DATA", str(data_dir))
    monkeypatch.setattr(PW, "DEMO_DIR", str(data_dir / "demo"))
    monkeypatch.setattr(R, "game_frame", lambda seasons: _history())
    monkeypatch.setattr(ingest, "load", _loader(SCHEDULE))
    R.clear_cache()
    yield PW
    R.clear_cache()


@contextlib.contextmanager
def _argv(*args):
    """The demo renderer is a CLI; its destination policy is an --out flag."""
    saved = sys.argv
    sys.argv = ["predict_2026_wk01", *args]
    try:
        yield
    finally:
        sys.argv = saved


def _records(frame: pd.DataFrame, columns) -> list:
    """Rows as plain records, with null spelled one way.

    Used to compare a frame that has been through a CSV against one that has
    not: an all-null column comes back as float NaN from disk where the
    in-memory frame holds None. Every value still has to match; only the two
    spellings of "no value" are reconciled.
    """
    return [{k: (None if pd.isna(v) else v) for k, v in record.items()}
            for record in frame[list(columns)].to_dict("records")]


def _demo_out(demo, season: int = SEASON, week: int = WEEK) -> str:
    return os.path.join(demo.DEMO_DIR,
                        f"early_{season}_wk{week:02d}_demo.csv")


def _run(demo, *argv) -> pd.DataFrame:
    """Render, then read the rendered rows back THROUGH the contract reader.

    `read_artifact` validates, so every test below starts from rows that have
    already had to satisfy the public contract to be visible at all.
    """
    with _argv(*argv):
        assert demo.main() == 0
    return FC.read_artifact(_demo_out(demo))


# --------------------------------------------------------------------------- #
# What the old writer had to prove, re-expressed against the shared producer
# --------------------------------------------------------------------------- #
def test_producer_emits_every_scheduled_game_with_stable_identity(demo):
    df = _run(demo).set_index("game_id")

    assert sorted(df.index) == [501, 502, 503, 504], \
        "every week-1 schedule row must survive; week 2 is not this artifact"
    assert len(df) == 4, "one row per scheduled game, never two"
    assert set(df["season"]) == {SEASON} and set(df["week"]) == {WEEK}

    scheduled = SCHEDULE[SCHEDULE["week"] == WEEK].set_index("id")
    for game_id, game in scheduled.iterrows():
        row = df.loc[game_id]
        assert (row["home"], row["away"]) == (game.home_team, game.away_team), \
            "identity travels with the id; a row must name the game it prices"
        assert row["kickoff_utc"] == game.start_date
        assert bool(row["neutral_site"]) is bool(game.neutral_site)

    assert set(df["model_family"]) == {"ridge_preseason_prior"}, \
        "weeks 0-3 are priced by the early-season ridge policy"
    assert PR.production_family(WEEK) == "ridge_preseason_prior"
    assert len(set(df["model_version"])) == 1, \
        "one fitted table produced the week, and every row names it"
    assert df["model_version"].notna().all()
    assert set(df["status"]) <= {s.value for s in FC.Status}


def test_cross_division_matchups_are_refused_instead_of_priced(demo):
    df = _run(demo).set_index("game_id")

    assert df.loc[502, "status"] == FC.Status.REFUSED.value
    assert "FBS-vs-non-FBS" in df.loc[502, "refusal_reason"]
    assert df.loc[503, "status"] == FC.Status.REFUSED.value
    assert "Non-FBS" in df.loc[503, "refusal_reason"]
    assert df.loc[504, "status"] == FC.Status.REFUSED.value
    assert "NO RATING" in df.loc[504, "refusal_reason"]

    assert R.FCS_TEAM not in set(df["home"]) | set(df["away"]), \
        "the pooled FCS pseudo-team may never stand in for a real opponent"
    assert (df.loc[502, "away"], df.loc[503, "home"]) == ("Tiny State",
                                                          "Tiny State"), \
        "a refused row still names the real opponent it refused"


def test_refused_rows_publish_no_numbers(demo):
    df = _run(demo).set_index("game_id")

    for game_id in (502, 503, 504):
        for col in FC.FORECAST_COLUMNS:
            assert pd.isna(df.loc[game_id, col]), \
                f"game {game_id} was refused, so {col} must be empty"
        assert pd.isna(df.loc[game_id, "forecast_sd_basis"]), \
            "a refusal cites no out-of-sample basis: it publishes no number"

    for col in FC.REQUIRED_FORECAST_COLUMNS:
        assert not pd.isna(df.loc[501, col]), \
            "a fully rated FBS matchup must still be priced"


def test_producer_never_injects_a_prior(demo, monkeypatch):
    """`prior=` is `ratings.solve`'s test-only injection hook.

    Production must go through the branch that assembles the prior inside
    `solve` (talent + returning production + prior-year net), because that
    function is the one place documenting how a published prior is built.
    Routing round it would mean the shipped prior is assembled somewhere the
    reader of `preseason_prior` will never look.
    """
    solves, priors = [], []
    real_solve, real_prior = R.solve, R.preseason_prior

    def spy_solve(games, season, week, params, prev_net=None, prior=None):
        solves.append({"season": int(season), "prior": prior,
                       "prev_net": prev_net})
        return real_solve(games, season, week, params, prev_net=prev_net,
                          prior=prior)

    def spy_prior(season, teams, conf_of, params, prev_net=None):
        priors.append({"season": int(season), "prev_net": prev_net})
        return real_prior(season, teams, conf_of, params, prev_net=prev_net)

    monkeypatch.setattr(R, "solve", spy_solve)
    monkeypatch.setattr(R, "preseason_prior", spy_prior)
    df = _run(demo)

    target = [c for c in solves if c["season"] == SEASON]
    assert target, "the producer must solve the forecast season"
    for call in target:
        assert call["prior"] is None, \
            "prior= is the test-only injection hook, never a production path"
        assert call["prev_net"], \
            "prior-year team strength must reach the 2026 solve"

    built = [p for p in priors if p["season"] == SEASON]
    assert built, "the 2026 prior must be assembled inside solve, not handed in"
    assert all(p["prev_net"] for p in built), \
        "solve's own prior branch must receive last season's net ratings"
    assert (df["status"] == FC.Status.PRICED.value).any(), \
        "the production branch must still produce a priced row"


def test_rows_that_are_not_the_schedule_fail_loud(demo, monkeypatch):
    """The artifact is exactly its schedule, and the builder enforces it.

    Two ways in: a schedule that would really produce a duplicated row, and a
    coverage verdict injected at `engine.id_coverage_problem` (the one place
    that decides). Both must stop the run BEFORE anything reaches disk.
    """
    doubled = pd.concat([SCHEDULE, SCHEDULE[SCHEDULE["id"] == 501]],
                        ignore_index=True)
    monkeypatch.setattr(ingest, "load", _loader(doubled))
    with pytest.raises(FC.ContractError, match="duplicate row ids"):
        with _argv():
            demo.main()
    assert not os.path.exists(_demo_out(demo)), \
        "a run that cannot state its own schedule must publish nothing"

    monkeypatch.setattr(ingest, "load", _loader(SCHEDULE))
    monkeypatch.setattr(EG, "id_coverage_problem",
                        lambda rows, schedule: "row ids do not match")
    with pytest.raises(FC.ContractError, match="do not match"):
        with _argv():
            demo.main()
    assert not os.path.exists(_demo_out(demo))


def test_slate_and_public_date_are_right_at_the_utc_eastern_boundary(
        demo, monkeypatch):
    """A kickoff at 2026-09-01T02:00:00Z is still 2026-08-31 evening in
    America/New_York. It must publish as wk0/2026-08-31 -- not silently roll
    over to wk1/2026-09-01 just because the UTC calendar date already
    changed. `slate`/`public_date` are the v2 names for that pair, and they
    come off `cfb_calendar`, the one definition of the Eastern game day."""
    boundary = pd.concat([SCHEDULE, pd.DataFrame([
        {"id": 506, "week": 1, "start_date": "2026-09-01T02:00:00Z",
         "home_team": "AAA", "away_team": "CCC", "neutral_site": False},
    ])], ignore_index=True)
    monkeypatch.setattr(ingest, "load", _loader(boundary))

    df = _run(demo).set_index("game_id")

    assert df.loc[506, "slate"] == "wk0", \
        "the UTC calendar date rolled over but the Eastern one had not"
    assert df.loc[506, "public_date"] == "2026-08-31"
    assert df.loc[506, "kickoff_utc"] == "2026-09-01T02:00:00Z", \
        "the instant is published unchanged; only the label is Eastern"
    assert df.loc[506, "status"] == FC.Status.PRICED.value, \
        "AAA vs CCC is a rated FBS matchup -- the boundary must not affect " \
        "pricing"
    assert df.loc[502, "slate"] == "wk1", \
        "a September kickoff in the same CFBD week is not a wk0 game"


def test_no_roster_or_context_field_reaches_a_public_row(demo):
    """The contract is closed, so leakage is a schema question, not a habit.

    The forbidden list keeps the old writer's personnel/context ban and adds
    the actionability words the retired `forecast_status` column used to carry
    in prose: this artifact is paper-only, and there is no column in which a
    stake, an edge or a pick could ride along.
    """
    df = _run(demo)
    assert list(df.columns) == list(FC.COLUMNS), \
        "a field no reader knows how to interpret must not ride along"

    forbidden = ("roster", "nil", "starter", "player", "depth", "injur",
                 "context", "personnel", "bet", "edge", "stake", "pick",
                 "wager", "kelly")
    for col in df.columns:
        assert not any(f in col.lower() for f in forbidden), \
            f"no personnel/context/actionable input reaches a row ({col})"


# --------------------------------------------------------------------------- #
# The demotion itself
# --------------------------------------------------------------------------- #
def test_demo_script_cannot_write_the_live_artifact(demo):
    """`check_destination` is the demotion. Refusals, not warnings.

    The name check is location-independent on purpose: the two-writer problem
    was a filename with two owners, so `predictions_YYYY_wkNN.csv` is refused
    wherever it is pointed, including inside the demo directory.
    """
    live_shapes = [
        os.path.join(demo.DATA, "predictions_2026_wk01.csv"),
        os.path.join(demo.DEMO_DIR, "predictions_2026_wk01.csv"),
        os.path.join(_ROOT, "data", "predictions_2026_wk01.csv"),
    ]
    for path in live_shapes:
        with pytest.raises(demo.NotAPublisherError, match="live artifact"):
            demo.check_destination(path)

    with pytest.raises(demo.NotAPublisherError,
                       match=r"data/snapshots, which only"):
        demo.check_destination(
            os.path.join(demo.DATA, "snapshots", "2026", "wk01", "rows.csv"))
    with pytest.raises(demo.NotAPublisherError,
                       match=r"data/site, which only"):
        demo.check_destination(
            os.path.join(demo.DATA, "site", "api", "hub.json"))
    with pytest.raises(demo.NotAPublisherError, match="data/ root"):
        demo.check_destination(os.path.join(demo.DATA, "scratch.csv"))

    accepted = os.path.join(demo.DEMO_DIR, "early_2026_wk01_demo.csv")
    assert demo.check_destination(accepted) == os.path.abspath(accepted), \
        "the demo directory is the one destination this script owns"


def test_demo_refuses_a_live_destination_before_writing_anything(demo):
    """End to end: the refusal lands on the way in, not once the file exists."""
    target = os.path.join(demo.DATA, "predictions_2026_wk01.csv")
    with pytest.raises(demo.NotAPublisherError):
        with _argv("--out", target):
            demo.main()
    assert not os.path.exists(target), \
        "a refused destination must not be written even once"
    assert not os.path.exists(_demo_out(demo)), \
        "and the run must not quietly redirect itself somewhere else"


def test_demo_and_weekly_loop_early_path_publish_identical_rows(demo,
                                                                monkeypatch):
    """One producer, one artifact builder, therefore one schema.

    The loop's early-season branch is called directly and its publication step
    is captured: freezing, snapshotting and the site are the loop's own later
    stages, and what is on trial here is the rows both routes hand to them.
    `now` sits before every kickoff in the fixture so nothing freezes and the
    two frames are compared as the pregame rows they both are.
    """
    published = {}

    def capture(season, week, rows, *args, **kwargs):
        published["rows"] = rows
        return {"n": int(len(rows))}

    monkeypatch.setattr(WL, "DATA", demo.DATA)
    monkeypatch.setattr(WL, "_publish_week", capture)

    demo_rows = _run(demo)

    # Exactly how `stage_simulate` reaches the early-season branch.
    games = R.game_frame(list(range(WL.HISTORY_START, SEASON + 1)))
    schedule = ingest.load_games([SEASON])
    schedule = schedule[schedule["week"] == WEEK]
    WL._publish_early_week(
        SEASON, WEEK, schedule, games, R.load_params(),
        FC.BasisRegistry.load(WL.DATA),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        "test-digest", "test-window", False, None)
    loop_rows = FC.coerce_frame(published["rows"])

    FC.validate_frame(loop_rows, where="weekly loop early path")

    # Everything except the provenance stamps that are per-run by definition
    # (`generated_at_utc`, `snapshot_id`, `source_digest`, `config_hash`).
    shared = [c for c in FC.COLUMNS
              if c not in ("generated_at_utc", "snapshot_id", "source_digest",
                           "config_hash")]
    assert _records(loop_rows, shared) == _records(demo_rows, shared), \
        "one producer and one fitted model: the two routes cannot differ, " \
        "down to the model_version fingerprint"


def test_early_forecast_sd_is_the_registered_basis_not_the_rating_band(demo):
    """The forecast SD is registered out-of-sample evidence for weeks 0-3.

    The rating band is a different quantity -- how unsure the ratings are, a
    couple of points -- and publishing it as the forecast's spread is what made
    the old artifact's win probabilities describe a distribution nobody held.
    It still travels, as `rating_parameter_sd`, under its own name.
    """
    registry = FC.BasisRegistry.load(demo.DATA)
    basis = registry.require("ridge_preseason_prior", WEEK)
    registered_sd = float(basis["forecast_margin_sd"])

    priced = FC.forecast_bearing(_run(demo))
    assert len(priced), "nothing was priced, so nothing is under test"

    for row in priced.to_dict("records"):
        assert row["forecast_sd_basis"] == basis["basis_id"], \
            "a published number names the evidence its spread comes from"
        assert float(row["forecast_margin_sd"]) == registered_sd, \
            "the forecast SD is the registered early-week basis, verbatim"

        band = float(row["rating_parameter_sd"])
        margin = float(row["projected_home_margin"])
        assert band > 0, "the technical rating band still travels, named"
        assert band != registered_sd and band < registered_sd, \
            "the rating band is the smaller, separate technical quantity"
        published_p = float(row["p_home_win"])
        assert abs(published_p - _phi(margin / registered_sd)) < 1e-3, \
            "p_home_win belongs to the distribution the row publishes"
        # Had the band been passed off as the forecast spread, the published
        # probability would sit outside the contract's own coherence tolerance
        # -- which is exactly the incoherence version 2 exists to prevent.
        assert abs(_phi(margin / band) - published_p) > FC.P_COHERENCE_TOL


def test_the_shipped_basis_manifest_registers_real_early_week_evidence():
    """`data/forecast_sd_basis.json` must actually cover weeks 0-3.

    The default registry the demo and the weekly loop both load is the shipped
    file, so the sandbox manifest above proves the plumbing and this proves the
    plumbing has something valid to carry. The week-4-15 anchor is kept in the
    same file as the record of what the early route used to borrow; the
    registered early basis must not be that number.
    """
    with open(os.path.join(_ROOT, "data", "forecast_sd_basis.json")) as fh:
        payload = json.load(fh)
    basis = FC.BasisRegistry(payload).require("ridge_preseason_prior", WEEK)

    assert basis["covers_weeks"] == list(EG.EARLY_SEASON_WEEKS)
    assert basis["sd_source"] == "pooled_out_of_sample_residual"
    assert float(basis["forecast_margin_sd"]) > 0
    assert int(basis["n"]) > 0, "registered means measured on real games"
    assert float(basis["forecast_margin_sd"]) != float(
        payload["anchor_week4_15"]["sd"]), \
        "the week-4-15 residual SD is an extrapolation for weeks 0-3, not " \
        "evidence about them"
