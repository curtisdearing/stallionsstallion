"""The versioned public forecast contract, and the ambiguity it removes.

Before schema v2 the same filename could hold either of two schemas: the
early-season writer's `pred_margin`/`band_pm`, or the weekly loop's
`mc_margin`/`mc_margin_sd`/`mc_total` (plus those two as aliases). `band_pm`
meant rating-parameter uncertainty in one route (~3 points) and simulation
outcome uncertainty in the other (~16-18), under one column name at one
`schema_version`, and the hub feed copied a ridge point estimate into
`mc_margin` whenever the Monte-Carlo column was missing.

These tests exist to make each of those specific things impossible, and to
keep them impossible.
"""

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import producers as PR  # noqa: E402
from nflvalue import snapshots as SN  # noqa: E402


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WL = _load("weekly_loop", "scripts/weekly_loop.py")

SEASON, EARLY_WEEK, LATE_WEEK = 2026, 1, 8

BASIS = {
    "schema_version": 1,
    "bases": [
        {"basis_id": "test_early", "model_family": "ridge_preseason_prior",
         "covers_weeks": [0, 1, 2, 3], "sd_source": "pooled_out_of_sample_residual",
         "forecast_margin_sd": 16.9, "usable": True, "n": 724},
        {"basis_id": "test_sim", "model_family": "kalman_montecarlo",
         "covers_weeks": list(range(4, 21)),
         "sd_source": "per_game_predictive_simulation",
         "forecast_margin_sd": None, "usable": True},
    ],
}


@pytest.fixture()
def registry():
    return FC.BasisRegistry(BASIS, source="<test>")


def _schedule(week=LATE_WEEK):
    return pd.DataFrame([
        {"id": 101, "home_team": "Ohio State", "away_team": "Michigan",
         "neutral_site": False, "start_date": "2026-10-17T16:00:00Z",
         "home_fcs": False, "away_fcs": False, "week": week},
        {"id": 102, "home_team": "Alabama", "away_team": "Mercer",
         "neutral_site": False, "start_date": "2026-10-17T20:00:00Z",
         "home_fcs": False, "away_fcs": True, "week": week},
    ])


def _provenance(family, week):
    return FC.Provenance(model_family=family, model_version=f"fp-{family}",
                         generated_at_utc="2026-10-14T12:00:00Z",
                         source_digest="dig", config_hash="cfg")


def _ridge_rows(registry, schedule=None):
    schedule = _schedule(EARLY_WEEK) if schedule is None else schedule
    sd = registry.margin_sd("ridge_preseason_prior", EARLY_WEEK)
    basis = registry.basis_id("ridge_preseason_prior", EARLY_WEEK)
    forecasts = {101: FC.Forecast(
        predictive=FC.PredictiveMargin.from_normal(7.5, sd, basis),
        rating_parameter_sd=3.0)}
    return FC.build_artifact(
        SEASON, EARLY_WEEK, schedule, forecasts,
        {102: FC.DEFAULT_REFUSALS["cross_division"]},
        _provenance("ridge_preseason_prior", EARLY_WEEK))


def _sim_rows(registry, schedule=None, **kwargs):
    schedule = _schedule(LATE_WEEK) if schedule is None else schedule
    simulated = {101: {"mc_margin": 7.5, "mc_margin_sd": 17.2,
                       "mc_total": 51.3, "p_home_win": 0.6683,
                       "param_sd": 2.1}}
    return WL.simulation_artifact(
        SEASON, LATE_WEEK, schedule, simulated, {},
        _provenance("kalman_montecarlo", LATE_WEEK), registry, **kwargs)


# --------------------------------------------------------------------------- #
# One contract, two routes
# --------------------------------------------------------------------------- #
def test_both_routes_emit_the_identical_public_contract(registry):
    """The requirement in one assertion: same columns, same rules, both valid."""
    ridge, sim = _ridge_rows(registry), _sim_rows(registry)
    assert list(ridge.columns) == list(FC.COLUMNS)
    assert list(sim.columns) == list(FC.COLUMNS)
    FC.validate_frame(ridge, where="ridge")
    FC.validate_frame(sim, where="sim")
    assert set(ridge["status"]) <= {s.value for s in FC.Status}
    assert set(sim["status"]) <= {s.value for s in FC.Status}
    # And the same reader gets the same shape of answer from either.
    assert (set(FC.forecast_bearing(ridge).columns)
            == set(FC.forecast_bearing(sim).columns))


def test_the_two_routes_are_genuinely_different_models(registry):
    """Guards the test above from passing because both routes are the same.

    They must agree on the CONTRACT while disagreeing on the model -- that is
    the whole point. If this ever fails, the previous test has stopped
    proving anything.
    """
    ridge = FC.forecast_bearing(_ridge_rows(registry)).iloc[0]
    sim = FC.forecast_bearing(_sim_rows(registry)).iloc[0]
    assert ridge["model_family"] == "ridge_preseason_prior"
    assert sim["model_family"] == "kalman_montecarlo"
    assert ridge["forecast_margin_sd"] != sim["forecast_margin_sd"]
    assert ridge["forecast_sd_basis"] != sim["forecast_sd_basis"]
    # The ridge route runs no simulation, so it projects no total.
    assert pd.isna(ridge["projected_total"])
    assert sim["projected_total"] == pytest.approx(51.3)


# --------------------------------------------------------------------------- #
# No ridge estimate may wear a Monte-Carlo name
# --------------------------------------------------------------------------- #
def test_the_contract_has_no_monte_carlo_name_left_to_relabel_into():
    assert set(FC.COLUMNS).isdisjoint(FC.BANNED_COLUMNS)
    for name in ("mc_margin", "mc_margin_sd", "mc_total", "pred_margin",
                 "band_pm", "param_sd"):
        assert name in FC.BANNED_COLUMNS


def test_a_ridge_row_cannot_acquire_a_simulation_quantity(registry):
    """A total is a simulation output. The ridge family may not publish one."""
    basis = registry.basis_id("ridge_preseason_prior", EARLY_WEEK)
    predictive = FC.PredictiveMargin(mean=7.5, sd=16.9, p_home_win=0.67,
                                     basis_id=basis, total=51.3)
    with pytest.raises(FC.ContractError, match="cannot project a total"):
        FC.build_artifact(SEASON, EARLY_WEEK, _schedule(EARLY_WEEK),
                          {101: FC.Forecast(predictive=predictive)}, {102: "x"},
                          _provenance("ridge_preseason_prior", EARLY_WEEK))


def test_no_ridge_artifact_column_is_a_monte_carlo_name(registry):
    ridge = _ridge_rows(registry)
    assert not set(ridge.columns) & set(FC.BANNED_COLUMNS)


def test_the_simulator_key_is_translated_exactly_once(registry):
    """The VALUE survives the hop; the NAME does not.

    `mc_margin` is still the simulator's own dictionary key -- that is its
    internal vocabulary and it is fine. What must never happen again is that
    key reaching a published column. `mc_forecasts` is the single place the
    translation happens, so this pins it.
    """
    forecasts = WL.mc_forecasts(
        {101: {"mc_margin": 7.5, "mc_margin_sd": 17.2, "mc_total": 51.3,
               "p_home_win": 0.6683, "param_sd": 2.1}}, registry, LATE_WEEK)
    predictive = forecasts[101].predictive
    assert predictive.mean == 7.5           # the number is carried through
    assert predictive.sd == 17.2
    assert forecasts[101].rating_parameter_sd == 2.1


#: The v1 names that carried the ambiguity, as opposed to the ones that were
#: merely renamed. A static scan can only be run against these: `id`, `date`,
#: `neutral`, `engine` and `veto` are also banned as CONTRACT columns, but
#: they are ordinary column names on other frames -- `id` is the schedule's
#: own identity column, which `scoreboard.py` legitimately merges on. Banning
#: those strings from source would be a scan that fires on the wrong thing.
#: They stay covered where it matters, by
#: `test_every_v1_column_is_rejected_by_name`, which rejects them as columns
#: of a published artifact.
AMBIGUOUS_V1_FIELDS = frozenset({
    "mc_margin", "mc_margin_sd", "mc_total", "pred_margin", "band_pm",
    "param_sd", "forecast_status",
})


def test_the_ambiguous_names_are_all_still_banned_columns():
    """Keeps the narrowed static scan honest about what it covers."""
    assert AMBIGUOUS_V1_FIELDS <= set(FC.BANNED_COLUMNS)


@pytest.mark.parametrize("module", ["nflvalue/producers.py",
                                    "scripts/build_hub_feed.py",
                                    "nflvalue/scoreboard.py"])
def test_public_writers_name_no_v1_forecast_field_in_code(module):
    """Static: the modules that build public output never mention a v1
    forecast name in executable code.

    Docstrings are excluded -- the history is worth writing down. Executable
    string constants are not: one of them is how the relabel got in.
    """
    tree = ast.parse((ROOT / module).read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    offenders = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings and node.value in AMBIGUOUS_V1_FIELDS]
    assert not offenders, f"{module} names v1 forecast fields in code: {offenders}"


# --------------------------------------------------------------------------- #
# Refusals and degradation publish no numbers
# --------------------------------------------------------------------------- #
def test_refused_rows_carry_no_forecast_numbers(registry):
    refused = _ridge_rows(registry)
    refused = refused[refused["status"] == FC.Status.REFUSED.value]
    assert len(refused) == 1
    for column in FC.FORECAST_COLUMNS:
        assert refused[column].isna().all(), column
    assert refused["forecast_sd_basis"].isna().all()
    assert refused["refusal_reason"].notna().all()


def test_degraded_rows_carry_no_forecast_numbers(registry):
    rows = _sim_rows(registry, degraded=True,
                     degraded_reason="games feed stale")
    assert set(rows["status"]) == {FC.Status.DEGRADED.value}
    for column in FC.FORECAST_COLUMNS:
        assert rows[column].isna().all(), column
    assert (rows["refusal_reason"] == "games feed stale").all()


def test_a_degraded_run_must_say_what_degraded_it(registry):
    with pytest.raises(FC.ContractError, match="must say what degraded it"):
        _sim_rows(registry, degraded=True, degraded_reason=None)


def test_a_refusal_carrying_a_number_is_rejected(registry):
    rows = _ridge_rows(registry)
    rows.loc[rows["status"] == FC.Status.REFUSED.value,
             "projected_home_margin"] = 3.0
    with pytest.raises(FC.ContractError, match="carry values"):
        FC.validate_frame(rows, where="mutated")


def test_a_priced_row_carrying_a_refusal_reason_is_rejected(registry):
    rows = _ridge_rows(registry)
    rows.loc[rows["status"] == FC.Status.PRICED.value,
             "refusal_reason"] = "not really"
    with pytest.raises(FC.ContractError, match="is not a refusal"):
        FC.validate_frame(rows, where="mutated")


# --------------------------------------------------------------------------- #
# Uncertainty has to be coherent, and evidenced
# --------------------------------------------------------------------------- #
def test_p_home_win_must_belong_to_the_published_distribution(registry):
    rows = _ridge_rows(registry)
    bearing = rows["status"] == FC.Status.PRICED.value
    rows.loc[bearing, "p_home_win"] = 0.95      # from some other distribution
    with pytest.raises(FC.ContractError, match="does not belong"):
        FC.validate_frame(rows, where="mutated")


def test_there_is_no_way_to_supply_an_independent_probability():
    """The constructors derive it; a caller cannot hand one in.

    This is the structural half of the fix. The published early-season
    `p_home_win` used to be formed from the week-4-15 residual SD and
    published beside a rating band that described something else entirely --
    possible only because the three numbers came from three places.
    """
    normal = FC.PredictiveMargin.from_normal(7.5, 16.9, "b")
    assert normal.p_home_win == pytest.approx(
        0.5 * (1 + math.erf(7.5 / (16.9 * math.sqrt(2)))))
    samples = FC.PredictiveMargin.from_samples([3.0, -1.0, 9.0, 5.0], "b")
    assert samples.p_home_win == pytest.approx(0.75)
    assert samples.mean == pytest.approx(4.0)
    with pytest.raises(FC.ContractError, match="not coherent"):
        FC.PredictiveMargin.from_summary(7.5, 16.9, 0.95, "b")


def test_a_week_with_no_registered_basis_cannot_be_priced(registry):
    with pytest.raises(FC.ContractError, match="no registered out-of-sample"):
        registry.require("ridge_preseason_prior", 9)
    with pytest.raises(FC.ContractError, match="no registered out-of-sample"):
        registry.require("kalman_montecarlo", 1)


def test_an_unusable_basis_cannot_be_priced():
    payload = {"bases": [{"basis_id": "underpowered",
                          "model_family": "ridge_preseason_prior",
                          "covers_weeks": [0, 1, 2, 3], "usable": False,
                          "forecast_margin_sd": None,
                          "note": "n below the pre-registered floor"}]}
    with pytest.raises(FC.ContractError, match="marked unusable"):
        FC.BasisRegistry(payload).require("ridge_preseason_prior", 1)


def test_the_shipped_basis_is_early_week_evidence_not_the_rating_band():
    """The early SD must be the out-of-sample residual SD, not the band.

    `band_pm` used to publish a summed rating band of about 3 points beside a
    win probability drawn from a ~16.7-point distribution. The shipped basis
    has to be the latter kind of quantity, and it has to cover the weeks it
    prices.
    """
    manifest = json.loads((ROOT / "data" / "forecast_sd_basis.json").read_text())
    registry = FC.BasisRegistry(manifest, source="shipped")
    basis = registry.require("ridge_preseason_prior", 1)
    assert basis["usable"] is True
    assert set(basis["covers_weeks"]) >= {0, 1, 2, 3}
    assert basis["sd_source"] == "pooled_out_of_sample_residual"
    assert basis["n"] >= 300              # the pre-registered power floor
    assert basis["forecast_margin_sd"] > 10.0   # a forecast SD, not a band
    assert "weeks 1-3" in basis["window"]


# --------------------------------------------------------------------------- #
# Mixed schemas are rejected, loudly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("legacy", sorted(FC.BANNED_COLUMNS))
def test_every_v1_column_is_rejected_by_name(registry, legacy):
    rows = _ridge_rows(registry)
    rows[legacy] = None
    with pytest.raises(FC.ContractError) as exc:
        FC.validate_frame(rows, where="v1")
    assert legacy in str(exc.value)
    assert FC.BANNED_COLUMNS[legacy].split()[0] in str(exc.value)


def test_mixed_schema_versions_are_rejected(registry):
    rows = _ridge_rows(registry)
    rows.loc[0, "schema_version"] = 1
    with pytest.raises(FC.ContractError, match="Mixed-schema data is rejected"):
        FC.validate_frame(rows, where="mixed")


def test_an_unrecognised_column_is_rejected(registry):
    rows = _ridge_rows(registry)
    rows["qb_availability"] = 0.9
    with pytest.raises(FC.ContractError, match="unrecognised columns"):
        FC.validate_frame(rows, where="widened")


def test_fractional_identity_is_named_by_the_contract_not_by_pandas(tmp_path):
    """A bad id must fail as a contract error, not as a float64 cast error."""
    path = tmp_path / "predictions_2026_wk08.csv"
    frame = pd.DataFrame([{c: None for c in FC.COLUMNS}])
    frame.loc[0, "game_id"] = 101.5
    frame.to_csv(path, index=False)
    with pytest.raises(FC.ContractError, match="fractional"):
        FC.read_artifact(str(path))


# --------------------------------------------------------------------------- #
# A mismatch fails BEFORE any public output exists
# --------------------------------------------------------------------------- #
def test_write_artifact_leaves_no_file_when_the_frame_is_invalid(registry,
                                                                tmp_path):
    rows = _ridge_rows(registry)
    rows["band_pm"] = 3.0
    target = tmp_path / "predictions_2026_wk01.csv"
    with pytest.raises(FC.ContractError):
        FC.write_artifact(rows, str(target))
    assert not target.exists()
    assert not (tmp_path / "predictions_2026_wk01.csv.tmp").exists()


def test_write_snapshot_creates_no_stamp_when_the_frame_is_invalid(registry,
                                                                   tmp_path):
    rows = _ridge_rows(registry)
    rows.loc[rows["status"] == FC.Status.REFUSED.value, "p_home_win"] = 0.5
    with pytest.raises(FC.ContractError):
        SN.write_snapshot(str(tmp_path), SEASON, EARLY_WEEK, "20261014T120000Z",
                          rows, source_digest="d", window="w")
    week = SN.week_dir(str(tmp_path), SEASON, EARLY_WEEK)
    assert not Path(week).exists() or not list(Path(week).glob("2026*"))


def test_an_existing_published_file_is_not_touched_by_a_failed_write(registry,
                                                                    tmp_path):
    """The previous week's honest artifact must survive a bad rebuild."""
    target = tmp_path / "predictions_2026_wk01.csv"
    good = _ridge_rows(registry)
    FC.write_artifact(good, str(target))
    before = target.read_text()

    bad = good.copy()
    bad["mc_margin"] = 7.5
    with pytest.raises(FC.ContractError):
        FC.write_artifact(bad, str(target))
    assert target.read_text() == before


# --------------------------------------------------------------------------- #
# The full row-class table, in one mixed frame
# --------------------------------------------------------------------------- #
def _mixed_row_class_frame(registry):
    """One row per row class the public contract distinguishes: valid
    forecast (priced) / refusal (refused) / malformed input (degraded -- the
    run could not stand behind these numbers, which is what malformed input
    produces)."""
    basis = registry.basis_id("ridge_preseason_prior", EARLY_WEEK)
    sd = registry.margin_sd("ridge_preseason_prior", EARLY_WEEK)
    predictive = FC.PredictiveMargin.from_normal(7.5, sd, basis)
    provenance = _provenance("ridge_preseason_prior", EARLY_WEEK)
    base = {c: None for c in FC.COLUMNS}
    base.update({
        "schema_version": FC.SCHEMA_VERSION, "season": SEASON, "week": EARLY_WEEK,
        "kickoff_utc": "2026-09-05T16:00:00Z", "neutral_site": False,
        "slate": "wk1", "public_date": "2026-09-05",
        "model_family": "ridge_preseason_prior",
        "model_version": provenance.model_version,
        "generated_at_utc": provenance.generated_at_utc,
        "snapshot_id": provenance.snapshot_id,
        "source_digest": provenance.source_digest,
        "config_hash": provenance.config_hash,
    })
    valid = dict(base, game_id=201, home="Ohio State", away="Michigan",
                status=FC.Status.PRICED.value,
                projected_home_margin=round(predictive.mean, 3),
                forecast_margin_sd=round(predictive.sd, 3),
                p_home_win=round(predictive.p_home_win, 4),
                forecast_sd_basis=basis, rating_parameter_sd=3.0)
    refusal = dict(base, game_id=202, home="Tiny State", away="Smaller State",
                  status=FC.Status.REFUSED.value,
                  refusal_reason=FC.DEFAULT_REFUSALS["non_fbs"])
    malformed = dict(base, game_id=203, home="Bad Row U", away="Also Bad U",
                     status=FC.Status.DEGRADED.value,
                     refusal_reason="malformed input: schedule row failed to "
                                   "parse cleanly")
    return FC.validate_frame(
        pd.DataFrame([valid, refusal, malformed], columns=list(FC.COLUMNS)),
        where="mixed row classes")


def test_the_full_row_class_table_holds_across_a_mixed_frame(registry):
    """Valid forecast / refusal / malformed input, asserted against the whole
    table this contract enforces: visible, numeric-or-null, status+reason,
    eligible for grading or not."""
    frame = _mixed_row_class_frame(registry)
    by_id = frame.set_index("game_id")
    assert sorted(by_id.index) == [201, 202, 203], \
        "every row class is visible -- none dropped"

    valid, refusal, malformed = by_id.loc[201], by_id.loc[202], by_id.loc[203]

    # valid forecast: numeric fields required and finite; explicit status.
    assert valid["status"] == FC.Status.PRICED.value
    for column in FC.REQUIRED_FORECAST_COLUMNS:
        assert math.isfinite(float(valid[column])), column
    assert pd.isna(valid["refusal_reason"])

    # refusal: numeric fields null; machine-readable nonempty reason.
    assert refusal["status"] == FC.Status.REFUSED.value
    for column in FC.FORECAST_COLUMNS:
        assert pd.isna(refusal[column]), column
    assert isinstance(refusal["refusal_reason"], str) \
        and refusal["refusal_reason"].strip()

    # malformed input: numeric fields null; explicit status IS the
    # machine-readable signal (a closed enum value, not free-text prose), plus
    # its own nonempty reason.
    assert malformed["status"] == FC.Status.DEGRADED.value
    assert malformed["status"] in {s.value for s in FC.Status}
    for column in FC.FORECAST_COLUMNS:
        assert pd.isna(malformed[column]), column
    assert isinstance(malformed["refusal_reason"], str) \
        and malformed["refusal_reason"].strip()

    # eligibility: only the valid forecast enters grading/metrics.
    bearing = FC.forecast_bearing(frame)
    assert set(bearing["game_id"]) == {201}


# --------------------------------------------------------------------------- #
# Uncertainty without a validated basis is an honest status, not a crash and
# not a rating band standing in for a forecast SD
# --------------------------------------------------------------------------- #
def test_uncertainty_unvalidated_is_reason_bearing_not_forecast_bearing():
    status = FC.Status.UNCERTAINTY_UNVALIDATED.value
    assert status not in FC.FORECAST_BEARING
    assert status in FC.REASON_BEARING
    assert status in FC.RUN_DEGRADED_STATUSES


def test_degraded_status_must_be_a_run_degraded_status(registry):
    with pytest.raises(FC.ContractError,
                       match="not a whole-run degraded status"):
        FC.build_artifact(SEASON, EARLY_WEEK, _schedule(EARLY_WEEK), {}, {},
                          _provenance("ridge_preseason_prior", EARLY_WEEK),
                          degraded=True, degraded_reason="x",
                          degraded_status=FC.Status.REFUSED)


def test_ridge_route_degrades_to_uncertainty_unvalidated_when_no_basis_is_registered(
        registry, monkeypatch):
    """The catch, isolated from the ridge fit itself: no basis -> an honest
    status reaching the artifact, not an unhandled exception reaching the
    weekly loop."""
    def _boom(*a, **k):
        raise PR.UncertaintyUnvalidatedError(
            "no registered out-of-sample basis covers week 1 for "
            "model_family 'ridge_preseason_prior'")
    monkeypatch.setattr(PR, "early_ridge_forecasts", _boom)

    class _StubTable:
        def fingerprint(self):
            return "stub-fp"
    monkeypatch.setattr(PR.EG, "solve_production_table",
                        lambda *a, **k: _StubTable())

    rows = PR.build_early_week(
        SEASON, EARLY_WEEK, games=pd.DataFrame(), schedule=_schedule(EARLY_WEEK),
        generated_at_utc="2026-10-14T12:00:00Z", data_dir="unused",
        registry=registry)
    assert set(rows["status"]) == {FC.Status.UNCERTAINTY_UNVALIDATED.value}
    for column in FC.FORECAST_COLUMNS:
        assert rows[column].isna().all(), column
    assert rows["refusal_reason"].str.startswith("UNCERTAINTY UNVALIDATED").all()


def test_kalman_route_degrades_to_uncertainty_unvalidated_when_no_basis_is_registered():
    """The same honest degrade on the in-season route: one status, both
    routes -- not a second, divergent failure mode."""
    empty_registry = FC.BasisRegistry({"bases": []}, source="<empty>")
    schedule = _schedule(LATE_WEEK)
    simulated = {101: {"mc_margin": 7.5, "mc_margin_sd": 17.2, "mc_total": 51.3,
                       "p_home_win": 0.6683, "param_sd": 2.1}}
    rows = WL.simulation_artifact(
        SEASON, LATE_WEEK, schedule, simulated, {},
        _provenance("kalman_montecarlo", LATE_WEEK), empty_registry)
    assert set(rows["status"]) == {FC.Status.UNCERTAINTY_UNVALIDATED.value}
    for column in FC.FORECAST_COLUMNS:
        assert rows[column].isna().all(), column
    assert rows["refusal_reason"].str.startswith("UNCERTAINTY UNVALIDATED").all()


def test_write_snapshot_accepts_uncertainty_unvalidated_as_run_degraded(tmp_path):
    """`write_snapshot`'s degraded/rows consistency check must recognise the
    new status as a run-degraded one, not only the literal `degraded` value."""
    empty_registry = FC.BasisRegistry({"bases": []}, source="<empty>")
    schedule = _schedule(LATE_WEEK)
    simulated = {101: {"mc_margin": 7.5, "mc_margin_sd": 17.2, "mc_total": 51.3,
                       "p_home_win": 0.6683, "param_sd": 2.1}}
    rows = WL.simulation_artifact(
        SEASON, LATE_WEEK, schedule, simulated, {},
        _provenance("kalman_montecarlo", LATE_WEEK), empty_registry)
    SN.write_snapshot(str(tmp_path), SEASON, LATE_WEEK, "20261014T120000Z",
                      rows, source_digest="d", window="w", degraded=True,
                      degraded_reason="no validated out-of-sample basis")


# --------------------------------------------------------------------------- #
# The hub keeps per-row provenance and status
# --------------------------------------------------------------------------- #
def test_hub_feed_keeps_every_row_s_status_and_provenance(registry, tmp_path,
                                                          monkeypatch):
    hub = _load("build_hub_feed", "scripts/build_hub_feed.py")
    data = tmp_path / "data"
    (data / "site" / "api").mkdir(parents=True)
    FC.write_artifact(_ridge_rows(registry),
                      str(data / "predictions_2026_wk01.csv"))
    monkeypatch.setattr(hub, "ROOT", str(tmp_path))
    out = data / "site" / "api" / "hub.json"
    monkeypatch.setattr(sys, "argv", ["build_hub_feed.py", "--out", str(out)])
    assert hub.main() == 0

    feed = json.loads(out.read_text())
    assert feed["schema_version"] == FC.SCHEMA_VERSION
    assert feed["model_family"] == "ridge_preseason_prior"
    for game in feed["week"]["games"]:
        assert set(game) == set(FC.COLUMNS)
        assert game["status"] in {s.value for s in FC.Status}
        assert game["model_family"] and game["model_version"]
    priced = [g for g in feed["week"]["games"]
              if g["status"] in FC.FORECAST_BEARING]
    assert all(g["forecast_sd_basis"] for g in priced)
    assert feed["week"]["by_status"]["refused"] == 1
