"""The cross-division (FBS-vs-FCS) route: what it may publish, and what it
must not touch.

Pre-registered 2026-09-02T23:25Z (`_worklog.md`). The basis
`cross_division_ridge_wk1_15_2021_2025` came out usable (n=599, pooled SD
17.8957, bias -4.64 recorded and NOT corrected). These tests pin:

* the projection is the table's own arithmetic against the pooled `__FCS__`
  pseudo-team, and refuses an unratable FBS side or a table with no FCS
  coefficient;
* nothing is priced without a usable registered basis -- the standing
  refusal comes back word for word;
* a priced cross-division row names its own family, version and basis, and
  can never carry a total;
* the per-row family override is all-or-nothing and validated;
* the card decomposes the number against the pseudo-team, says so in the
  label, and carries the cross-division caution above the early-season one;
* NEGATIVE CONTROL on the real week-1 artifact: the 43 published FBS-vs-FBS
  numbers are identical before and after the route exists, and every
  previously refused cross-division game is now priced;
* the basis merge never drops another basis.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import cross_division as XD  # noqa: E402
from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import insight as IN  # noqa: E402
from nflvalue import producers as PR  # noqa: E402
from nflvalue import ratings as R  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_ratings import synth_games, FLAT_PRIOR, P  # noqa: E402


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BT = _load("backtest_cross_division", "scripts/backtest_cross_division.py")

SEASON, WEEK = 2023, 9
XD_SD = 17.8957

BASIS_WITH = {"schema_version": 1, "bases": [
    {"basis_id": "test_early", "model_family": "ridge_preseason_prior",
     "covers_weeks": [0, 1, 2, 3], "sd_source": "pooled_out_of_sample_residual",
     "forecast_margin_sd": 16.9, "usable": True, "n": 724},
    {"basis_id": "test_xd", "model_family": "ridge_cross_division",
     "covers_weeks": list(range(0, 21)),
     "sd_source": "pooled_out_of_sample_residual",
     "forecast_margin_sd": XD_SD, "usable": True, "n": 599},
]}
BASIS_WITHOUT = {"schema_version": 1, "bases": BASIS_WITH["bases"][:1]}


@pytest.fixture()
def table():
    g = synth_games(fcs_games=[(2, "A", "Tiny State", 70.0, 3.0),
                               (3, "B", "Other Tiny", 63.0, 7.0)])
    return R.solve(g, SEASON, WEEK, P, prior=FLAT_PRIOR)


def _schedule():
    return pd.DataFrame([
        {"id": 1, "home_team": "A", "away_team": "B", "neutral_site": False,
         "start_date": "2023-10-28T16:00:00Z", "week": WEEK},
        {"id": 2, "home_team": "A", "away_team": "Tiny State",
         "neutral_site": False, "start_date": "2023-10-28T20:00:00Z",
         "week": WEEK},
        {"id": 3, "home_team": "Tiny State", "away_team": "C",
         "neutral_site": True, "start_date": "2023-10-28T23:00:00Z",
         "week": WEEK},
        {"id": 4, "home_team": "Tiny State", "away_team": "Other Tiny",
         "neutral_site": False, "start_date": "2023-10-29T00:00:00Z",
         "week": WEEK},
    ])


FBS = {"A", "B", "C", "D"}


# --------------------------------------------------------------------------- #
# The projection
# --------------------------------------------------------------------------- #
def test_projection_is_the_tables_own_arithmetic_against_the_pseudo_team(table):
    F = R.FCS_TEAM
    expected = (table.hfa("A", False)
                + (table.off["A"] + table.def_[F])
                - (table.off[F] + table.def_["A"]))
    assert XD.predict_margin(table, "A", fbs_is_home=True) == pytest.approx(
        expected, abs=1e-6)
    # Away FBS side, neutral: no HFA, sign flipped.
    away = XD.predict_margin(table, "A", fbs_is_home=False, neutral=True)
    assert away == pytest.approx(-(expected - table.hfa("A", False)), abs=1e-6)


def test_fcs_pseudo_team_stays_vetoed_as_a_rating(table):
    """The bypass is for the far side of a matchup only. Asking the table
    for the pseudo-team as a team still refuses."""
    assert table.veto[R.FCS_TEAM] is True
    with pytest.raises(R.DisconnectedTeamError):
        table.predict_margin("A", R.FCS_TEAM)


def test_projection_refuses_an_unratable_fbs_side(table):
    with pytest.raises(R.DisconnectedTeamError):
        XD.predict_margin(table, "Nowhere U", fbs_is_home=True)


def test_projection_refuses_a_table_that_never_saw_an_fcs_game():
    t = R.solve(synth_games(), SEASON, WEEK, P, prior=FLAT_PRIOR)
    assert R.FCS_TEAM not in t.off
    with pytest.raises(R.RatingsError, match="no pooled FCS coefficient"):
        XD.predict_margin(t, "A", fbs_is_home=True)


# --------------------------------------------------------------------------- #
# The producer
# --------------------------------------------------------------------------- #
def test_without_a_registered_basis_the_standing_refusal_comes_back(table):
    reg = FC.BasisRegistry(BASIS_WITHOUT, source="<test>")
    forecasts, refusals = PR.cross_division_forecasts(
        table, _schedule(), reg, WEEK, fbs=FBS)
    assert forecasts == {}
    assert refusals == {2: FC.DEFAULT_REFUSALS["cross_division"],
                        3: FC.DEFAULT_REFUSALS["cross_division"]}


def test_with_a_basis_only_cross_division_games_are_priced_and_say_so(table):
    reg = FC.BasisRegistry(BASIS_WITH, source="<test>")
    forecasts, refusals = PR.cross_division_forecasts(
        table, _schedule(), reg, WEEK, fbs=FBS)
    assert set(forecasts) == {2, 3} and refusals == {}
    for fc in forecasts.values():
        assert fc.model_family == "ridge_cross_division"
        assert fc.model_version == table.fingerprint()
        assert fc.predictive.basis_id == "test_xd"
        assert fc.predictive.sd == XD_SD
        assert fc.predictive.total is None
    assert forecasts[2].predictive.mean == XD.predict_margin(
        table, "A", fbs_is_home=True)
    assert forecasts[3].predictive.mean == XD.predict_margin(
        table, "C", fbs_is_home=False, neutral=True)


def test_p_home_win_is_read_off_the_same_distribution(table):
    reg = FC.BasisRegistry(BASIS_WITH, source="<test>")
    forecasts, _ = PR.cross_division_forecasts(
        table, _schedule(), reg, WEEK, fbs=FBS)
    p = forecasts[2].predictive
    assert p.p_home_win == pytest.approx(FC._phi(p.mean / p.sd))


def test_only_ids_narrows_to_still_pregame_games(table):
    reg = FC.BasisRegistry(BASIS_WITH, source="<test>")
    forecasts, refusals = PR.cross_division_forecasts(
        table, _schedule(), reg, WEEK, fbs=FBS, only_ids={3})
    assert set(forecasts) == {3} and refusals == {}


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
def _pm():
    return FC.PredictiveMargin.from_normal(30.0, XD_SD, "test_xd")


def test_per_row_family_override_is_all_or_nothing():
    with pytest.raises(FC.ContractError, match="BOTH model_family"):
        FC.Forecast(predictive=_pm(), model_family="ridge_cross_division")
    with pytest.raises(FC.ContractError, match="BOTH model_family"):
        FC.Forecast(predictive=_pm(), model_version="fp")


def test_per_row_family_override_must_be_a_registered_family():
    with pytest.raises(FC.ContractError, match="unknown per-row model_family"):
        FC.Forecast(predictive=_pm(), model_family="ridge_of_my_dreams",
                    model_version="fp")


def test_artifact_rows_carry_the_overriding_family_and_version(table):
    reg = FC.BasisRegistry(BASIS_WITH, source="<test>")
    forecasts, _ = PR.cross_division_forecasts(
        table, _schedule(), reg, WEEK, fbs=FBS)
    prov = FC.Provenance(model_family="kalman_montecarlo",
                         model_version="fp-kalman",
                         generated_at_utc="2023-10-25T12:00:00Z")
    rows = FC.build_artifact(SEASON, WEEK, _schedule(), forecasts,
                             {1: "x", 4: FC.DEFAULT_REFUSALS["non_fbs"]}, prov)
    by_id = rows.set_index("game_id")
    assert by_id.loc[2, "model_family"] == "ridge_cross_division"
    assert by_id.loc[2, "model_version"] == table.fingerprint()
    assert by_id.loc[2, "forecast_sd_basis"] == "test_xd"
    assert by_id.loc[2, "status"] == FC.Status.PRICED.value
    # The main-route family still names the rows it did not price.
    assert by_id.loc[1, "model_family"] == "kalman_montecarlo"
    assert by_id.loc[1, "model_version"] == "fp-kalman"
    assert by_id.loc[4, "status"] == FC.Status.REFUSED.value
    FC.validate_frame(rows, where="test")


def test_a_cross_division_row_cannot_borrow_the_simulations_total():
    """Inside a Kalman-week artifact, a ridge row is judged by ITS family."""
    prov = FC.Provenance(model_family="kalman_montecarlo",
                         model_version="fp-kalman",
                         generated_at_utc="2023-10-25T12:00:00Z")
    forecasts = {2: FC.Forecast(
        predictive=FC.PredictiveMargin.from_summary(
            30.0, XD_SD, FC._phi(30.0 / XD_SD), "test_xd", total=55.0),
        model_family="ridge_cross_division", model_version="fp-xd")}
    with pytest.raises(FC.ContractError, match="runs no simulation"):
        FC.build_artifact(SEASON, WEEK, _schedule(), forecasts, {}, prov)


# --------------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------------- #
def test_card_components_close_on_the_projection_and_name_the_pseudo_team(table):
    reg = FC.BasisRegistry(BASIS_WITH, source="<test>")
    forecasts, _ = PR.cross_division_forecasts(
        table, _schedule(), reg, WEEK, fbs=FBS)
    prov = FC.Provenance(model_family="ridge_preseason_prior",
                         model_version="fp", generated_at_utc="2023-10-25T12:00:00Z")
    rows = FC.build_artifact(SEASON, WEEK, _schedule(), forecasts,
                             {1: "x", 4: "y"}, prov)
    priced = rows[rows["status"] == FC.Status.PRICED.value]
    comps = IN.components_for_rows(priced, table, table, fbs=FBS)
    assert set(comps) == {2, 3}
    for row in priced.to_dict("records"):
        why = IN.build_why(comps[int(row["game_id"])], row)
        assert "pooled FCS pseudo-team" in why[0]["label"]
    # neutral-site cross-division game: no home-field row at all
    assert [c["key"] for c in IN.build_why(comps[3], priced.set_index(
        "game_id").loc[3].to_dict() | {"game_id": 3})] == ["rating_differential"]


def test_card_caution_is_the_cross_division_one_even_in_an_early_week():
    row = {"status": "priced", "model_family": "ridge_cross_division",
           "week": 1, "projected_home_margin": 30.0,
           "forecast_margin_sd": XD_SD}
    caution = IN.build_caution(row)
    assert caution["code"] == "cross_division_pooled"
    assert "not corrected" in caution["text"]
    # The vocabulary guard must accept the new sentence.
    IN.assert_non_actionable({"caution": caution}, proper_nouns=("A", "B"))


# --------------------------------------------------------------------------- #
# Negative control on the real published week
# --------------------------------------------------------------------------- #
#: The last week-1 artifact published BEFORE the cross-division route
#: existed: 43 priced FBS-vs-FBS rows, 48 cross-division refusals. It is an
#: immutable, tracked snapshot, so the control does not move when the live
#: artifact is regenerated with the route in place.
PRE_ROUTE_SNAPSHOT = (ROOT / "data" / "snapshots" / "2026_wk01"
                      / "20260902T150740Z" / "predictions.csv")


def _published_wk1():
    if not PRE_ROUTE_SNAPSHOT.exists():
        pytest.skip("no pre-route week-1 snapshot in this checkout")
    return FC.read_artifact(str(PRE_ROUTE_SNAPSHOT))


def test_real_week1_fbs_numbers_are_identical_and_cross_division_is_priced():
    """Rebuild 2026 week 1 through the production early route with the REAL
    registry. Every FBS-vs-FBS number the pre-route snapshot carries must
    come back identical -- the route added a set of rows, it did not move
    one. And every row that snapshot refused as cross-division must now be
    priced under the registered basis, naming its family (bar a new-to-FBS
    side with no rating, refused as such)."""
    from nflvalue import ingest
    published = _published_wk1()
    games = R.game_frame(list(range(2019, 2027)))
    upcoming = ingest.load_games([2026])
    schedule = upcoming[upcoming["week"] == 1]
    rows = PR.build_early_week(
        2026, 1, games=games, schedule=schedule,
        generated_at_utc="2026-09-02T23:00:00Z", data_dir=str(ROOT / "data"))
    new = rows.set_index("game_id")
    old = published.set_index("game_id")

    was_priced = old[old["status"] == FC.Status.PRICED.value]
    assert len(was_priced) == 43
    for gid, row in was_priced.iterrows():
        for col in ("projected_home_margin", "forecast_margin_sd",
                    "p_home_win", "rating_parameter_sd"):
            assert new.loc[gid, col] == pytest.approx(row[col], abs=1e-9), (gid, col)
        assert new.loc[gid, "forecast_sd_basis"] == row["forecast_sd_basis"]
        assert new.loc[gid, "model_family"] == "ridge_preseason_prior"
        assert new.loc[gid, "model_version"] == row["model_version"]

    was_xd = old[old["refusal_reason"] == FC.DEFAULT_REFUSALS["cross_division"]]
    assert len(was_xd) == 48
    now_xd = new.loc[was_xd.index]
    # Two of the 48 involve a programme classified FBS in the 2026 teams
    # pull that has no FBS rating yet (Sacramento State at home to an FCS
    # side; a new-to-FBS visitor at North Dakota State): refused with a NO
    # RATING reason, which is the honest answer, not a cross-division one.
    refused = now_xd[now_xd["status"] == FC.Status.REFUSED.value]
    assert refused["refusal_reason"].str.startswith("NO RATING").all()
    assert len(refused) <= 2
    priced = now_xd[now_xd["status"] == FC.Status.PRICED.value]
    assert len(priced) == len(now_xd) - len(refused) >= 46
    assert (priced["model_family"] == "ridge_cross_division").all()
    assert (priced["forecast_sd_basis"]
            == "cross_division_ridge_wk1_15_2021_2025").all()
    assert ((priced["forecast_margin_sd"] - XD_SD).abs() < 1e-3).all()
    assert priced["projected_total"].isna().all()
    # Every FBS home side is projected ahead of the pooled FCS opponent.
    assert (priced["p_home_win"] > 0.5).all()

    was_non_fbs = old[old["refusal_reason"] == FC.DEFAULT_REFUSALS["non_fbs"]]
    assert (new.loc[was_non_fbs.index, "status"] == FC.Status.REFUSED.value).all()


# --------------------------------------------------------------------------- #
# The basis manifest
# --------------------------------------------------------------------------- #
def test_merge_never_drops_another_basis(tmp_path):
    path = tmp_path / "forecast_sd_basis.json"
    path.write_text(json.dumps(BASIS_WITHOUT))
    merged = BT.merge_into_manifest(str(path), {"basis_id": BT.BASIS_ID,
                                                "model_family": BT.FAMILY})
    ids = [b["basis_id"] for b in merged["bases"]]
    assert ids == ["test_early", BT.BASIS_ID]
    # Re-merging replaces, never duplicates.
    path.write_text(json.dumps(merged))
    again = BT.merge_into_manifest(str(path), {"basis_id": BT.BASIS_ID,
                                               "model_family": BT.FAMILY,
                                               "n": 1})
    assert [b["basis_id"] for b in again["bases"]] == ["test_early", BT.BASIS_ID]
    assert again["bases"][1]["n"] == 1


def test_registered_basis_is_the_pre_registered_shape():
    reg = FC.BasisRegistry.load(str(ROOT / "data"))
    basis = reg.require("ridge_cross_division", 1)
    assert basis["basis_id"] == BT.BASIS_ID
    assert basis["prereg"] == BT.PREREG
    assert basis["n"] >= BT.MIN_N
    assert basis["forecast_margin_sd"] > basis["fbs_vs_fbs_early_sd_for_comparison"]
    assert any(f.startswith("BIAS") for f in basis["findings"])
    # the two pre-existing bases are still there
    assert reg.require("ridge_preseason_prior", 1)["basis_id"] == \
        "early_ridge_wk1_3_2021_2025"
    assert reg.require("kalman_montecarlo", 8)["basis_id"] == "sim_kalman_wk4_plus"
