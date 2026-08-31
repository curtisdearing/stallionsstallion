"""Hub feed contract — identity in, identity out, under forecast contract v2.

The feed is the last hop before the public site, so it is the last place a
row can silently lose its identity (which model priced it, under what status,
for which scheduled game). It recomputes nothing: every number is copied from
the artifact, and anything the contract cannot vouch for is a loud failure
rather than a quiet ``None`` that reads as "no forecast".

Schema version 2 moved the strictness out of the script and into
``nflvalue.forecast_contract``, so these tests exercise the same failure modes
through the contract rather than through hand-rolled parsing: a violating
artifact must fail the build, never be translated into something publishable.
Two of them now assert the OPPOSITE of what they used to. The feed once copied
the ridge point estimate ``pred_margin`` into ``mc_margin`` whenever the
Monte-Carlo column was absent — a number produced by no simulation, published
under a simulation's name. That relabel is gone, and the tests that blessed it
are the tests that now forbid it.
"""

import importlib.util
import json
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
sys.path.insert(0, _ROOT)

from nflvalue import forecast_contract as FC  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HF = _load("build_hub_feed")

RIDGE = "ridge_preseason_prior"
KALMAN = "kalman_montecarlo"

#: The registered early-week out-of-sample basis. A forecast-bearing row has to
#: name one, so the fixtures name the real one rather than inventing a label.
BASIS = "early_ridge_wk1_3_2021_2025"
BASIS_SD = 16.8982


def _row(**overrides):
    """One contract row: every column present, defaults that validate."""
    row = {column: None for column in FC.COLUMNS}
    row.update({
        "schema_version": FC.SCHEMA_VERSION,
        "season": 2026, "week": 1, "game_id": 401,
        "home": "Home U", "away": "Away U",
        "kickoff_utc": "2026-08-30T00:00:00Z", "neutral_site": False,
        "slate": "wk0", "public_date": "2026-08-29",
        "status": FC.Status.REFUSED.value,
        "refusal_reason": FC.DEFAULT_REFUSALS["unrated"],
        "model_family": RIDGE, "model_version": "ridge-2026w1-abc123",
        "generated_at_utc": "2026-08-30T12:00:00Z",
        "snapshot_id": "snap-2026-08-30T12:00:00Z",
        "source_digest": "sha256:0f0f", "config_hash": "cfg-01",
    })
    row.update(overrides)
    return row


def _forecast_row(margin=4.5, sd=BASIS_SD, *, family=RIDGE, total=None,
                  **overrides):
    """A forecast-bearing row whose three numbers come off ONE distribution.

    They are read off :class:`PredictiveMargin` instead of being typed in,
    because a fixture that hand-picks ``p_home_win`` beside an unrelated SD is
    the exact incoherence the contract exists to reject: the fixture would be
    testing the coherence check rather than the feed.
    """
    predictive = FC.PredictiveMargin.from_normal(margin, sd, BASIS)
    row = _row(status=FC.Status.PRICED.value)
    row.update({
        "refusal_reason": None,
        "model_family": family,
        "projected_home_margin": round(predictive.mean, 3),
        "forecast_margin_sd": round(predictive.sd, 3),
        "p_home_win": round(predictive.p_home_win, 4),
        "projected_total": total,
        "rating_parameter_sd": 3.1,
        "forecast_sd_basis": BASIS,
    })
    row.update(overrides)
    return row


PRICED = _forecast_row()
FROZEN = _forecast_row(margin=-2.25, game_id=402, home="Frozen U",
                       away="Away U", status=FC.Status.FROZEN.value,
                       frozen_from_stamp="2026-08-29T18:00:00Z")
REFUSED = _row(game_id=403, home="FCS A", away="FCS B",
               refusal_reason=FC.DEFAULT_REFUSALS["non_fbs"])
DEGRADED = _row(game_id=404, home="Degraded U", away="Away U",
                status=FC.Status.DEGRADED.value,
                refusal_reason=("ratings refresh failed; the run could not "
                                "stand behind a number"))
ALL_FOUR = [PRICED, FROZEN, REFUSED, DEGRADED]

#: The version-1 artifact, kept verbatim as the shape that is now REFUSED.
#: `band_pm` meant two different things in two routes and `mc_margin` was
#: whatever the writer felt like putting there, which is why no reader may
#: translate this file — it can only reject it.
V1_COLUMNS = ["season", "week", "id", "slate", "date", "home", "away",
              "neutral", "mc_margin", "mc_margin_sd", "mc_total", "p_home_win",
              "param_sd", "pred_margin", "band_pm", "veto", "note", "engine",
              "forecast_status"]
V1_PRICED = {"season": 2026, "week": 1, "id": 401, "slate": "wk0",
             "date": "2026-08-29", "home": "Home U", "away": "Away U",
             "neutral": "False", "mc_margin": "", "mc_margin_sd": "",
             "mc_total": "", "p_home_win": "0.61", "param_sd": "",
             "pred_margin": "4.5", "band_pm": "17.2", "veto": "False",
             "note": "", "engine": "ridge",
             "forecast_status": "early-season paper-only research forecast"}


def _artifact_path(tmp_path):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return data / "predictions_2026_wk01.csv"


def _arm(tmp_path, monkeypatch):
    """Point main() at a sandboxed ROOT; return the feed path it would write."""
    out = tmp_path / "hub.json"
    monkeypatch.setattr(HF, "ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["build_hub_feed", "--out", str(out)])
    return out


def _stage(tmp_path, rows, monkeypatch):
    """Stage a contract-valid artifact through the contract's own writer."""
    FC.write_artifact(pd.DataFrame(rows, columns=list(FC.COLUMNS)),
                      str(_artifact_path(tmp_path)))
    return _arm(tmp_path, monkeypatch)


def _stage_violating(tmp_path, rows, monkeypatch, columns=None):
    """Stage an artifact the contract would refuse to write.

    Deliberately bypasses :func:`FC.write_artifact` — that is the only way to
    hand the feed a violating file, and what the feed does with one is the
    thing under test. Returns the feed path so a test can also assert that
    nothing was published.
    """
    frame = pd.DataFrame(rows, columns=list(columns or FC.COLUMNS))
    frame.to_csv(_artifact_path(tmp_path), index=False)
    return _arm(tmp_path, monkeypatch)


def _published_keys(value):
    """Every key the feed publishes, at any depth."""
    if isinstance(value, dict):
        for key, sub in value.items():
            yield key
            yield from _published_keys(sub)
    elif isinstance(value, list):
        for item in value:
            yield from _published_keys(item)


def _build(tmp_path, rows, monkeypatch):
    """Run the real main() against a sandboxed ROOT; return the feed dict."""
    out = _stage(tmp_path, rows, monkeypatch)
    assert HF.main() == 0
    with open(out) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Identity survives the CSV -> hub hop
# --------------------------------------------------------------------------- #
def test_hub_preserves_every_row_and_its_identity(tmp_path, monkeypatch):
    feed = _build(tmp_path, ALL_FOUR, monkeypatch=monkeypatch)
    games = feed["week"]["games"]

    assert [g["game_id"] for g in games] == [401, 402, 403, 404], \
        "a game's id is its identity across the CSV -> hub hop"
    assert feed["schema_version"] == FC.SCHEMA_VERSION == 2
    assert feed["week"]["n_games"] == 4

    priced = games[0]
    assert (priced["season"], priced["week"]) == (2026, 1)
    assert priced["status"] == "priced"
    assert priced["model_family"] == RIDGE
    assert priced["projected_home_margin"] == 4.5
    assert priced["forecast_margin_sd"] == round(BASIS_SD, 3)
    assert priced["p_home_win"] == PRICED["p_home_win"]
    assert priced["refusal_reason"] is None
    assert priced["neutral_site"] is False

    refused = games[2]
    assert refused["status"] == "refused"
    assert refused["refusal_reason"].startswith("Non-FBS matchup")
    for field in FC.FORECAST_COLUMNS:
        assert refused[field] is None, \
            f"a refused game must publish no {field}"
    assert refused["forecast_sd_basis"] is None, \
        "a row carrying no forecast may not cite a forecast SD basis"


def test_per_row_provenance_and_status_survive_into_the_hub(tmp_path,
                                                            monkeypatch):
    """The hub carries the whole contract row, not a lossy summary.

    A reader of the feed alone must be able to say which model, which
    registered SD basis and which snapshot produced each number, and to tell a
    priced row from a republished one from a refusal. Dropping provenance at
    the last hop would make the public artifact less answerable than the
    private one it was copied from.
    """
    feed = _build(tmp_path, ALL_FOUR, monkeypatch=monkeypatch)
    games = {g["game_id"]: g for g in feed["week"]["games"]}

    for game in games.values():
        assert set(game) == set(FC.COLUMNS), \
            "the hub publishes exactly the contract's columns, no more, no less"
        assert set(HF.ROW_FIELDS) == set(FC.COLUMNS)

    priced = games[401]
    assert priced["model_family"] == RIDGE
    assert priced["model_version"] == "ridge-2026w1-abc123"
    assert priced["forecast_sd_basis"] == BASIS
    assert priced["snapshot_id"] == "snap-2026-08-30T12:00:00Z"
    assert priced["frozen_from_stamp"] is None

    frozen = games[402]
    assert frozen["status"] == "frozen"
    assert frozen["frozen_from_stamp"] == "2026-08-29T18:00:00Z", \
        "a republished pre-kickoff forecast must keep the stamp it came from"
    assert frozen["projected_home_margin"] == -2.25

    degraded = games[404]
    assert degraded["status"] == "degraded"
    assert degraded["refusal_reason"].startswith("ratings refresh failed")
    assert degraded["projected_home_margin"] is None


def test_week_block_counts_by_status_and_never_says_vetoed(tmp_path,
                                                           monkeypatch):
    """`n_vetoed` was a two-valued reading of a four-valued world.

    A refusal and a degraded run are not the same event — one is the model
    declining a game it was never validated on, the other is the run being
    unable to stand behind any number — so the week block counts the statuses
    it actually has instead of collapsing them into "vetoed".
    """
    feed = _build(tmp_path, ALL_FOUR, monkeypatch=monkeypatch)
    block = feed["week"]

    assert "n_vetoed" not in block, "the veto vocabulary is gone with v1"
    assert block["n_games"] == 4
    assert block["n_priced"] == 2, "priced and frozen rows both carry numbers"
    assert block["n_unpriced"] == 2
    assert block["by_status"] == {"priced": 1, "frozen": 1, "refused": 1,
                                 "degraded": 1, "uncertainty_unvalidated": 0}
    assert set(block["by_status"]) == {s.value for s in FC.Status}
    assert sum(block["by_status"].values()) == block["n_games"]
    assert block["n_priced"] == sum(block["by_status"][s]
                                    for s in FC.FORECAST_BEARING)
    assert "unpriced by design" in block["status_note"]


def test_top_level_model_family_is_derived_from_the_rows(tmp_path,
                                                         monkeypatch):
    feed = _build(tmp_path, [PRICED, REFUSED], monkeypatch=monkeypatch)
    assert feed["model_family"] == RIDGE, \
        "the feed must name the family that actually priced the week"
    assert "engine" not in feed, "v1's `engine` key does not survive into v2"
    assert "kalman" not in json.dumps(feed).lower()


def test_mixed_model_families_never_produce_a_false_family_claim(tmp_path,
                                                                 monkeypatch):
    """Two families in one week means the artifact was assembled from two runs.

    The feed cannot name one of them without hiding the other, so it names
    neither.
    """
    simulated = _forecast_row(margin=1.5, game_id=405, family=KALMAN,
                              total=52.5, home="Sim U")
    feed = _build(tmp_path, [PRICED, simulated], monkeypatch=monkeypatch)
    assert feed["model_family"] == HF.UNRESOLVED_FAMILY == "unresolved"
    assert feed["week"]["games"][1]["projected_total"] == 52.5, \
        "a family that runs a simulation may publish a total; ridge may not"


def test_family_from_rows_never_invents_a_family():
    """`unresolved` is the answer to every question the rows cannot settle."""
    assert not hasattr(HF, "_engine_from_rows")
    assert not hasattr(HF, "KNOWN_ENGINES")
    assert not hasattr(HF, "UNRESOLVED_ENGINE"), \
        "the engine vocabulary was replaced, not renamed alongside"

    assert HF._family_from_rows([]) == HF.UNRESOLVED_FAMILY
    assert HF._family_from_rows([{"model_family": None}]) == \
        HF.UNRESOLVED_FAMILY
    assert HF._family_from_rows([{"model_family": "wizard"}]) == \
        HF.UNRESOLVED_FAMILY
    assert HF._family_from_rows([{"model_family": RIDGE},
                                 {"model_family": KALMAN}]) == \
        HF.UNRESOLVED_FAMILY
    assert HF._family_from_rows([{"model_family": RIDGE},
                                 {"model_family": RIDGE}]) == RIDGE


def test_an_unnameable_producer_never_reaches_the_feed(tmp_path, monkeypatch):
    """The feed no longer has to guess at a missing or unknown producer.

    Under v1 an artifact could simply omit the `engine` column and the feed
    had to publish it as `unresolved`. The contract closes that off upstream:
    a row whose producer cannot be named is not a row anything may publish, so
    the build fails instead of shipping an anonymous forecast.
    """
    out = _stage_violating(tmp_path, [_forecast_row(model_family="wizard")],
                           monkeypatch)
    with pytest.raises(HF.HubFeedError, match="model_family"):
        HF.main()
    assert not out.exists()

    columns = [c for c in FC.COLUMNS if c != "model_family"]
    out = _stage_violating(tmp_path, [PRICED, REFUSED], monkeypatch,
                           columns=columns)
    with pytest.raises(HF.HubFeedError, match="model_family"):
        HF.main()
    assert not out.exists()


def test_no_artifact_still_never_claims_a_model_family(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    out = _arm(tmp_path, monkeypatch)
    assert HF.main() == 0
    feed = json.load(open(out))
    assert feed["week"] is None
    assert feed["record"] is None
    assert feed["model_family"] == HF.UNRESOLVED_FAMILY
    assert feed["schema_version"] == FC.SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# No ridge estimate under a Monte-Carlo name — the relabel that is now banned
# --------------------------------------------------------------------------- #
def test_hub_never_publishes_a_ridge_estimate_under_a_monte_carlo_name(
        tmp_path, monkeypatch):
    """Reversal of `..._counts_and_backfills_mc_margin_for_the_early_season_schema`.

    That test blessed the backfill: when the early-season artifact carried no
    `mc_margin` column, the feed copied `pred_margin` into it so the row would
    not "present mc_margin as a refusal". The number it published had run no
    simulation, and the name it published it under said it had. The counts
    that test guarded still matter, so they are still asserted here — but the
    relabel is now the failure, not the feature: the ridge estimate appears
    once, under a name that says which model made it, and no Monte-Carlo name
    exists anywhere in the feed for it to be copied into.
    """
    feed = _build(tmp_path, [PRICED, REFUSED], monkeypatch=monkeypatch)

    assert feed["week"]["n_games"] == 2
    assert feed["week"]["n_priced"] == 1 and feed["week"]["n_unpriced"] == 1
    assert feed["model_family"] == RIDGE

    # No key the feed publishes, at any depth, may be a v1 name.
    assert set(_published_keys(feed)).isdisjoint(FC.BANNED_COLUMNS)
    # And the simulation names must not appear anywhere in the document at
    # all, key or string. (The short v1 names `id`, `date` and `neutral` live
    # on inside `game_id`, `public_date` and `neutral_site`, so those three are
    # covered by the key check above rather than by a substring search.)
    body = json.dumps(feed)
    for banned in ("mc_margin", "mc_margin_sd", "mc_total", "pred_margin",
                   "band_pm", "param_sd", "veto", "engine", "forecast_status"):
        assert banned in FC.BANNED_COLUMNS
        assert banned not in body, \
            f"v1 name {banned!r} reached the published feed"
    assert set(FC.COLUMNS).isdisjoint(FC.BANNED_COLUMNS), \
        "the contract keeps no Monte-Carlo name for a ridge number to land in"

    priced = feed["week"]["games"][0]
    assert [k for k, v in priced.items() if v == 4.5] == \
        ["projected_home_margin"], \
        "the ridge estimate may appear once, under the one name that fits it"
    assert priced["model_family"] == RIDGE and priced["forecast_sd_basis"]
    assert priced["projected_total"] is None, \
        "the ridge route runs no simulation, so it projects no total"

    refused = feed["week"]["games"][1]
    assert refused["projected_home_margin"] is None, \
        "a refused game stays empty; there is nothing to back-fill it from"


def test_a_v1_artifact_is_rejected_rather_than_translated(tmp_path,
                                                          monkeypatch):
    """Reversal of `..._never_backfills_a_genuinely_blank_mc_margin_...`.

    That test drew a line between a structurally absent Monte-Carlo column
    (translate it) and a present-but-blank one (leave it null). Both halves of
    that distinction are gone: a v1 artifact is not a thing to be read
    carefully, it is a thing to be refused, because `band_pm` does not record
    which of its two meanings it carried and no amount of care recovers it.
    """
    out = _stage_violating(tmp_path, [V1_PRICED], monkeypatch,
                           columns=V1_COLUMNS)
    with pytest.raises(HF.HubFeedError) as caught:
        HF.main()
    message = str(caught.value)
    assert "schema-version-1" in message
    assert "band_pm" in message and "pred_margin" in message
    assert not out.exists(), "a v1 artifact must publish nothing at all"


def test_a_monte_carlo_column_grafted_onto_v2_is_rejected_not_backfilled(
        tmp_path, monkeypatch):
    """A single v1 name is enough to make the whole file unreadable.

    This is the case the old backfill test called "the column exists and this
    row is blank". A mixed-schema artifact is not a v2 artifact with one extra
    field — it is one where a reader cannot tell which contract any number was
    written under, so it fails the build instead of being partly trusted.
    """
    grafted = {**PRICED, "mc_margin": 4.5}
    out = _stage_violating(tmp_path, [grafted], monkeypatch,
                           columns=list(FC.COLUMNS) + ["mc_margin"])
    with pytest.raises(HF.HubFeedError, match="mc_margin"):
        HF.main()
    assert not out.exists()


# --------------------------------------------------------------------------- #
# A violating artifact fails the build — it is never published as `null`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", list(FC.REQUIRED_FORECAST_COLUMNS))
@pytest.mark.parametrize("value", ["N/A", "four point five", "", "   "])
def test_priced_row_without_a_readable_number_fails_loud(tmp_path, monkeypatch,
                                                         field, value):
    """A number the reader cannot parse is not a refusal.

    Publishing it as `null` would read on the site as "the model declined to
    forecast this game", which is a different and untrue statement. The
    contract makes the two impossible to confuse: a row that says `priced`
    must carry all three numbers, so an unparseable or missing one fails the
    build by name.
    """
    out = _stage_violating(tmp_path, [{**PRICED, field: value}], monkeypatch)
    with pytest.raises(HF.HubFeedError, match=field):
        HF.main()
    assert not out.exists()


def test_a_refused_row_publishes_nulls_and_never_zeros(tmp_path, monkeypatch):
    """Blank is blank. A zero margin is a forecast; a refusal is not."""
    feed = _build(tmp_path, [REFUSED], monkeypatch=monkeypatch)
    game = feed["week"]["games"][0]
    for field in FC.FORECAST_COLUMNS:
        assert game[field] is None and game[field] != 0


@pytest.mark.parametrize("field", ["season", "week", "game_id"])
@pytest.mark.parametrize("value", ["", "   ", "abc", "nan"])
def test_unreadable_identity_fails_the_build_by_name(tmp_path, monkeypatch,
                                                     field, value):
    """season/week/game_id are identity, not forecasts.

    A blank one cannot become `null`: the published row would name a game that
    is not the game the artifact priced.
    """
    out = _stage_violating(tmp_path, [{**PRICED, field: value}], monkeypatch)
    with pytest.raises(HF.HubFeedError, match=field):
        HF.main()
    assert not out.exists()


@pytest.mark.parametrize("field", ["season", "week", "game_id"])
@pytest.mark.parametrize("value", ["1.5", "inf", "-inf"])
def test_fractional_or_infinite_identity_fails_the_build(tmp_path, monkeypatch,
                                                         field, value):
    """A fractional or infinite identity is refused one layer earlier.

    Whole-number identity is a dtype in v2, so these values die in
    `FC.coerce_frame` before `validate_frame` gets to phrase an opinion, and
    the build fails with pandas' cast error rather than a `HubFeedError`. What
    matters — and what is asserted — is that such a row is never truncated
    into a neighbouring game and never reaches the feed.
    """
    out = _stage_violating(tmp_path, [{**PRICED, field: value}], monkeypatch)
    with pytest.raises((HF.HubFeedError, TypeError, OverflowError)):
        HF.main()
    assert not out.exists()


@pytest.mark.parametrize("field,names_it", [
    # `projected_home_margin` is named as `margin` when the coherence check is
    # the one that catches it: an infinite margin implies p(home win) 0 or 1.
    ("projected_home_margin", "margin"),
    ("forecast_margin_sd", "forecast_margin_sd"),
    ("p_home_win", "p_home_win"),
])
@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_nonfinite_forecast_number_fails_loud(tmp_path, monkeypatch, field,
                                              names_it, value):
    """NaN/Inf are not numbers the site can render, and JSON cannot hold
    them: publishing one emits invalid JSON, so it fails the build."""
    out = _stage_violating(tmp_path, [{**PRICED, field: value}], monkeypatch)
    with pytest.raises(HF.HubFeedError, match=names_it):
        HF.main()
    assert not out.exists()


def test_duplicate_game_ids_fail_loud(tmp_path, monkeypatch):
    """Two rows claiming one game publish an ambiguous identity."""
    twin = {**PRICED, "home": "Other U"}
    out = _stage_violating(tmp_path, [PRICED, twin], monkeypatch)
    with pytest.raises(HF.HubFeedError, match="more than once|duplicat"):
        HF.main()
    assert not out.exists()


def test_row_season_week_must_match_the_artifact_filename(tmp_path,
                                                          monkeypatch):
    """The feed labels the week from the filename; a row that disagrees
    would be published under a week it was not priced for.

    Both rows here are contract-valid on their own — the contract has no view
    on which file a row was written to — so this check belongs to the feed and
    is tested through a validly-written artifact.
    """
    stray = {**REFUSED, "season": 2025}
    out = _stage(tmp_path, [PRICED, stray], monkeypatch)
    with pytest.raises(HF.HubFeedError, match="filename"):
        HF.main()
    assert not out.exists()

    stray_week = {**REFUSED, "week": 7}
    out = _stage(tmp_path, [PRICED, stray_week], monkeypatch)
    with pytest.raises(HF.HubFeedError, match="filename"):
        HF.main()
    assert not out.exists()


def test_valid_identity_and_numbers_still_pass_through_exactly(tmp_path,
                                                               monkeypatch):
    """The strictness must not cost the honest rows anything."""
    feed = _build(tmp_path, ALL_FOUR, monkeypatch=monkeypatch)
    games = feed["week"]["games"]
    assert [(g["season"], g["week"], g["game_id"]) for g in games] == \
        [(2026, 1, 401), (2026, 1, 402), (2026, 1, 403), (2026, 1, 404)]
    assert games[0]["projected_home_margin"] == 4.5
    assert games[2]["projected_home_margin"] is None
    assert feed["model_family"] == RIDGE
    assert feed["schema_version"] == FC.SCHEMA_VERSION
    body = json.dumps(feed)
    assert "NaN" not in body and "Infinity" not in body


# --------------------------------------------------------------------------- #
# Nothing actionable, nothing personal
# --------------------------------------------------------------------------- #
def test_feed_rows_carry_no_roster_or_context_field(tmp_path, monkeypatch):
    feed = _build(tmp_path, ALL_FOUR, monkeypatch=monkeypatch)
    forbidden = ("roster", "nil", "starter", "player", "depth", "injur",
                 "context", "personnel")
    for game in feed["week"]["games"]:
        for key in game:
            assert not any(f in key.lower() for f in forbidden), \
                f"no personnel/context input may reach the feed ({key})"


def test_the_feed_stays_non_actionable(tmp_path, monkeypatch):
    """v1 carried the paper-only framing in a free-text `forecast_status`
    column, which is gone: the column was a sentence a producer could rewrite,
    and it sat in the same row as the numbers it disclaimed.

    The invariant it protected does not depend on that column, so it is
    asserted where it actually lives now. `status` is a closed four-value
    enum — a producer cannot phrase it into a recommendation. The framing is
    stated once, at the top of the feed, in the disclaimer. And the rows
    themselves carry no actionable vocabulary anywhere, in any key or value:
    the feed says what the model projected, never what to do about it.
    """
    feed = _build(tmp_path, ALL_FOUR, monkeypatch=monkeypatch)

    disclaimer = feed["disclaimer"].lower()
    assert "paper-only" in disclaimer
    assert "nothing here is a bet recommendation" in disclaimer

    for game in feed["week"]["games"]:
        assert "forecast_status" not in game
        assert game["status"] in {s.value for s in FC.Status}, \
            "status is an enum, not a sentence a producer can rewrite"

    # Everything the feed publishes except the one disclaimer that names the
    # thing it is not. If an actionable word turns up in here, a row has
    # started giving advice.
    published = json.dumps({k: v for k, v in feed.items()
                            if k != "disclaimer"}).lower()
    for word in ("confidence", "pick", "edge", "stake", "bankroll", "clv",
                 "recommend", "spread", "wager", "bet", "parlay"):
        assert word not in published, \
            f"actionable vocabulary reached the feed ({word})"
