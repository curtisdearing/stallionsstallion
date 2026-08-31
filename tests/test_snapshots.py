"""Game-window snapshot system — the immutability, idempotency, freeze and
degraded-mode contracts, tested offline.

Written test-first. Every assertion here is a rule the weekly loop must not be
able to violate by accident:

  * a successful pregame run leaves an IMMUTABLE, timestamped snapshot;
  * `latest` is a POINTER, never a rewrite of history;
  * a rerun with identical inputs and an unchanged kickoff window produces no
    new meaningful snapshot (and, because the pregame job has no refresh stage,
    costs zero CFBD calls);
  * a changed source fingerprint, or an advanced kickoff window, DOES produce
    a new snapshot — two pregame runs in one week are normal, not an error;
  * once a game kicks off its final pregame row is frozen and can only ever be
    re-published verbatim from a snapshot taken BEFORE kickoff (this is the
    structural no-post-kickoff-leakage guarantee, not a convention);
  * a stale source degrades the output visibly instead of publishing
    confident-looking numbers over old data.

Everything the store handles is a `forecast_contract` schema-2 row, built by
`FC.build_artifact` and validated on the way in and on the way out. That is
why "frozen" and "degraded" are read here as row STATUS rather than as extra
boolean columns: the public contract is closed, so the snapshot writer has
nowhere to bolt a column of its own, and a row states exactly one thing about
itself.

PAPER-ONLY throughout: nothing here produces or implies a bet.
"""

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from nflvalue import forecast_contract as FC
from nflvalue import snapshots as SN


UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture()
def data_dir(tmp_path):
    return str(tmp_path)


def _games(rows):
    """Schedule frame in the shape ingest.load_games() returns."""
    return pd.DataFrame(rows)


def _week_games(*starts, played=()):
    rows = []
    for i, s in enumerate(starts, start=101):
        rows.append({
            "id": i, "season": 2026, "week": 2, "start_date": s,
            "home_team": f"Home{i}", "away_team": f"Away{i}",
            "neutral_site": False, "home_fcs": False, "away_fcs": False,
            "home_points": 24 if i in played else None,
            "away_points": 17 if i in played else None,
        })
    return _games(rows)


#: The fixture week. Week 2 is the early-season route, so these rows name
#: `ridge_preseason_prior` — which runs no simulation and therefore may not
#: publish a `projected_total`. Matching the real production family for the
#: week matters: `freeze_kicked_off` stamps a structural refusal with
#: `producers.production_family(week)`, so anything else here would leave one
#: artifact naming two producers for one week.
SEASON, WEEK = 2026, 2
MODEL_FAMILY = "ridge_preseason_prior"
MODEL_VERSION = "test-fixture-ridge"
#: The registered out-of-sample basis that covers weeks 0-3 for this family.
BASIS = "early_ridge_wk1_3_2021_2025"
FORECAST_SD = 16.5
RATING_PARAM_SD = 1.9
REFUSAL = FC.DEFAULT_REFUSALS["unrated"]

#: Fixed, so two builds of the same margins are byte-identical.
PROVENANCE = FC.Provenance(model_family=MODEL_FAMILY,
                           model_version=MODEL_VERSION,
                           generated_at_utc="2026-09-01T12:00:00Z")

#: The fixture week's kickoffs. A row's `kickoff_utc` has to name a real
#: instant, and these are the week's canonical noon / afternoon / night slots.
KICKOFFS = {101: "2026-09-05T16:00:00Z",
            102: "2026-09-05T19:30:00Z",
            103: "2026-09-05T23:00:00Z"}

#: Several tests use a deliberately absurd margin (99, 500) as a sentinel: a
#: later number that must never reach a frozen row. Against a 16.5-point SD
#: those imply a certainty the contract will not print — `p_home_win` must lie
#: strictly inside (0, 1) — so the probability is read at a capped z. It stays
#: coherent by a wide margin (beyond |z|=3.5 the true and capped probabilities
#: differ by <0.0003, against FC.P_COHERENCE_TOL of 0.05) and `from_summary`
#: re-checks it, so the sentinels stay publishable rows without the fixture
#: ever publishing a probability that does not belong to its own distribution.
_Z_CAP = 3.5


def _predictive(margin):
    z = max(-_Z_CAP, min(_Z_CAP, float(margin) / FORECAST_SD))
    return FC.PredictiveMargin.from_summary(
        mean=float(margin), sd=FORECAST_SD,
        p_home_win=0.5 * (1.0 + math.erf(z / math.sqrt(2.0))),
        basis_id=BASIS)


def _rows(margins, *, degraded_reason=None):
    """Artifact rows in the one public contract (see forecast_contract).

    `{game_id: margin}`, where a `None` margin is a game the model declined.
    Pass `degraded_reason` for a run that cannot stand behind any number:
    degradation is a property of the ROWS now, so the same builder produces
    it and every row comes back `status="degraded"`, reasoned, and numberless.
    """
    schedule = pd.DataFrame([
        {"id": gid, "start_date": KICKOFFS[gid], "home_team": f"Home{gid}",
         "away_team": f"Away{gid}", "neutral_site": False}
        for gid in margins
    ])
    forecasts = {gid: FC.Forecast(_predictive(m),
                                  rating_parameter_sd=RATING_PARAM_SD)
                 for gid, m in margins.items() if m is not None}
    refusals = {gid: REFUSAL for gid, m in margins.items() if m is None}
    return FC.build_artifact(SEASON, WEEK, schedule, forecasts, refusals,
                             PROVENANCE, degraded=degraded_reason is not None,
                             degraded_reason=degraded_reason)


def _write(data_dir, stamp, rows, *, digest="d0", window="w0",
           degraded=False, reason=None, taken_at=None):
    return SN.write_snapshot(
        data_dir, 2026, 2, stamp, rows,
        source_digest=digest, window=window,
        degraded=degraded, degraded_reason=reason,
        taken_at=taken_at or datetime(2026, 9, 1, 12, 0, tzinfo=UTC))


# --------------------------------------------------------------------------- #
# 1. Immutable snapshots + a `latest` that never overwrites history
# --------------------------------------------------------------------------- #
def test_snapshot_write_creates_timestamped_artifact_and_manifest(data_dir):
    res = _write(data_dir, "20260901T120000Z", _rows({101: 3.5}))

    snap = SN.snapshot_dir(data_dir, 2026, 2, "20260901T120000Z")
    assert os.path.isfile(os.path.join(snap, "predictions.csv"))
    meta = json.load(open(os.path.join(snap, "snapshot.json")))
    for key in ("stamp", "season", "week", "source_digest", "window",
                "degraded", "taken_at", "n_rows", "sha256"):
        assert key in meta, f"snapshot manifest missing {key} — must trace"
    assert res["path"] == snap


def test_second_snapshot_does_not_overwrite_the_first(data_dir):
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}), digest="d1")
    _write(data_dir, "20260902T120000Z", _rows({101: 9.5}), digest="d2")

    stamps = [s["stamp"] for s in SN.list_snapshots(data_dir, 2026, 2)]
    assert stamps == ["20260901T120000Z", "20260902T120000Z"]

    first = pd.read_csv(os.path.join(
        SN.snapshot_dir(data_dir, 2026, 2, "20260901T120000Z"),
        "predictions.csv"))
    assert float(first.loc[0, "projected_home_margin"]) == 3.5, \
        "history must survive a later run verbatim"


def test_rewriting_an_existing_stamp_is_refused(data_dir):
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}))
    with pytest.raises(SN.SnapshotError, match="immutable"):
        _write(data_dir, "20260901T120000Z", _rows({101: 99.0}))


def test_latest_pointer_tracks_newest_without_copying_history(data_dir):
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}), digest="d1")
    _write(data_dir, "20260902T120000Z", _rows({101: 9.5}), digest="d2")

    latest = SN.read_latest(data_dir, 2026, 2)
    assert latest["stamp"] == "20260902T120000Z"
    assert latest["source_digest"] == "d2"
    assert len(SN.list_snapshots(data_dir, 2026, 2)) == 2, \
        "the latest pointer must not consume a historical snapshot"


# --------------------------------------------------------------------------- #
# 2. Deterministic idempotency — fingerprint + kickoff window, not a sentinel
# --------------------------------------------------------------------------- #
def test_unchanged_inputs_do_not_create_a_duplicate_snapshot(data_dir):
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}),
           digest="abc", window="2026-09-05T16:00:00Z")

    d = SN.decide(data_dir, 2026, 2, source_digest="abc",
                  window="2026-09-05T16:00:00Z")

    assert d["run"] is False
    assert "unchanged" in d["reason"]
    assert len(SN.list_snapshots(data_dir, 2026, 2)) == 1


def test_changed_source_digest_causes_a_new_snapshot(data_dir):
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}),
           digest="abc", window="2026-09-05T16:00:00Z")

    d = SN.decide(data_dir, 2026, 2, source_digest="CHANGED",
                  window="2026-09-05T16:00:00Z")
    assert d["run"] is True and "digest" in d["reason"]

    _write(data_dir, "20260902T120000Z", _rows({101: 7.5}),
           digest="CHANGED", window="2026-09-05T16:00:00Z")
    assert len(SN.list_snapshots(data_dir, 2026, 2)) == 2


def test_two_same_week_pregame_runs_both_execute_when_the_window_advances(data_dir):
    """The old one-run-per-week sentinel made this impossible."""
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}),
           digest="same", window="2026-09-05T16:00:00Z")

    # Identical inputs, but the noon slate has kicked off: window advanced.
    d = SN.decide(data_dir, 2026, 2, source_digest="same",
                  window="2026-09-05T19:00:00Z")
    assert d["run"] is True
    assert "window" in d["reason"]

    _write(data_dir, "20260905T170000Z", _rows({101: 3.5}),
           digest="same", window="2026-09-05T19:00:00Z")
    assert len(SN.list_snapshots(data_dir, 2026, 2)) == 2, \
        "Tuesday-through-Saturday means several snapshots per week"


def test_first_run_of_a_week_always_runs(data_dir):
    d = SN.decide(data_dir, 2026, 2, source_digest="abc", window="w")
    assert d["run"] is True and "no prior snapshot" in d["reason"]


def test_recovering_from_a_degraded_snapshot_reruns(data_dir):
    _write(data_dir, "20260901T120000Z",
           _rows({101: 3.5}, degraded_reason="games feed stale"), digest="abc",
           window="w", degraded=True, reason="games feed stale")

    d = SN.decide(data_dir, 2026, 2, source_digest="abc", window="w",
                  degraded=False)
    assert d["run"] is True and "degraded" in d["reason"], \
        "a degraded publication must be replaced as soon as data is fresh"


def test_decision_is_pure_and_spends_no_api_calls(data_dir, monkeypatch):
    """decide() must never touch a source. The pregame job has no refresh
    stage precisely so a re-run cannot burn CFBD budget."""
    import nflvalue.ingest as ingest
    calls = []
    monkeypatch.setattr(ingest, "calls_used_this_month",
                        lambda: calls.append("read") or 0)
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}),
           digest="abc", window="w")
    SN.decide(data_dir, 2026, 2, source_digest="abc", window="w")
    assert calls == []


# --------------------------------------------------------------------------- #
# 3. Source fingerprint — what legitimately forces a re-run
# --------------------------------------------------------------------------- #
def test_digest_is_stable_and_order_independent(data_dir):
    a = SN.source_digest({"z": 1, "a": {"k": [1, 2]}})
    b = SN.source_digest({"a": {"k": [1, 2]}, "z": 1})
    assert a == b == SN.source_digest({"z": 1, "a": {"k": [1, 2]}})
    assert a != SN.source_digest({"z": 2, "a": {"k": [1, 2]}})


def test_schedule_change_moves_the_digest(data_dir):
    g1 = _week_games("2026-09-05T16:00:00Z")
    g2 = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    d1, _ = SN.collect_source_digest(data_dir, 2026, 2, g1, version={"c": "1"})
    d2, _ = SN.collect_source_digest(data_dir, 2026, 2, g2, version={"c": "1"})
    assert d1 != d2, "a newly scheduled game must force a re-run"


def test_kickoff_time_change_moves_the_digest(data_dir):
    g1 = _week_games("2026-09-05T16:00:00Z")
    g2 = _week_games("2026-09-05T19:30:00Z")
    d1, _ = SN.collect_source_digest(data_dir, 2026, 2, g1, version={"c": "1"})
    d2, _ = SN.collect_source_digest(data_dir, 2026, 2, g2, version={"c": "1"})
    assert d1 != d2


def test_code_or_config_version_change_moves_the_digest(data_dir):
    g = _week_games("2026-09-05T16:00:00Z")
    d1, _ = SN.collect_source_digest(data_dir, 2026, 2, g, version={"c": "1"})
    d2, _ = SN.collect_source_digest(data_dir, 2026, 2, g, version={"c": "2"})
    assert d1 != d2


def test_supported_source_freshness_moves_the_digest(data_dir):
    """rosters / availability / weather / lines are supported inputs: when one
    of their cached artifacts changes, the forecast must be allowed to move."""
    g = _week_games("2026-09-05T16:00:00Z")
    cfbd = os.path.join(data_dir, "cfbd")
    os.makedirs(cfbd, exist_ok=True)
    src = os.path.join(cfbd, "lines_2026.parquet")
    open(src, "wb").write(b"one")
    d1, parts1 = SN.collect_source_digest(data_dir, 2026, 2, g,
                                          version={"c": "1"})
    assert "lines_2026.parquet" in parts1["sources"]

    open(src, "wb").write(b"two-different-length")
    d2, _ = SN.collect_source_digest(data_dir, 2026, 2, g, version={"c": "1"})
    assert d1 != d2, "a changed source artifact must invalidate the snapshot"


# --------------------------------------------------------------------------- #
# 4. Kickoff windows — the Tuesday-through-Saturday cadence
# --------------------------------------------------------------------------- #
def test_window_is_the_next_pending_kickoff_floored(data_dir):
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T19:30:00Z",
                        "2026-09-05T23:00:00Z")
    now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    assert SN.kickoff_window(games, now) == "2026-09-05T16:00:00Z"


def test_saturday_yields_three_successive_windows(data_dir):
    """noon / afternoon / night — three re-runs, driven by the schedule
    itself rather than by hardcoded clock bands."""
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T19:30:00Z",
                        "2026-09-05T23:00:00Z")
    seen = []
    for now in (datetime(2026, 9, 5, 14, 0, tzinfo=UTC),
                datetime(2026, 9, 5, 17, 0, tzinfo=UTC),
                datetime(2026, 9, 5, 21, 0, tzinfo=UTC)):
        seen.append(SN.kickoff_window(games, now))
    assert seen == ["2026-09-05T16:00:00Z", "2026-09-05T19:00:00Z",
                    "2026-09-05T23:00:00Z"]
    assert len(set(seen)) == 3


def test_window_ignores_games_that_already_kicked_off(data_dir):
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    now = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)
    assert SN.kickoff_window(games, now) == "2026-09-05T23:00:00Z"


def test_window_is_none_when_every_game_has_kicked_off(data_dir):
    games = _week_games("2026-09-05T16:00:00Z")
    now = datetime(2026, 9, 6, 4, 0, tzinfo=UTC)
    assert SN.kickoff_window(games, now) == SN.WINDOW_NONE


def test_pending_game_ids_exclude_kicked_off_games(data_dir):
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    now = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)
    assert SN.pending_game_ids(games, now) == [102]


# --------------------------------------------------------------------------- #
# 5. Kickoff freeze — a kicked-off game can never be revised
# --------------------------------------------------------------------------- #
def test_a_kicked_off_game_cannot_be_revised(data_dir):
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    _write(data_dir, "20260905T150000Z", _rows({101: 3.5, 102: -6.0}),
           digest="d1", window="2026-09-05T16:00:00Z",
           taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))

    now = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)   # 101 has kicked off
    frz = SN.freeze_kicked_off(data_dir, 2026, 2, games, now)
    assert frz["newly_frozen"] == [101]

    # A later run recomputes BOTH games with different numbers.
    revised = _rows({101: 99.0, 102: -2.0})
    out, n = SN.apply_frozen(revised, SN.load_frozen(data_dir, 2026, 2))

    assert n == 1
    row101 = out.set_index("game_id").loc[101]
    assert float(row101["projected_home_margin"]) == 3.5, \
        "the kicked-off game must keep its final PREGAME number, verbatim"
    # `frozen` is a status, not a side column: a republished pre-kickoff row
    # says so about itself, and names the stamp it was republished from.
    assert row101["status"] == FC.Status.FROZEN.value
    assert row101["frozen_from_stamp"] == "20260905T150000Z"
    row102 = out.set_index("game_id").loc[102]
    assert float(row102["projected_home_margin"]) == -2.0, \
        "a game that has not kicked off is still free to move"
    assert row102["status"] == FC.Status.PRICED.value
    assert pd.isna(row102["frozen_from_stamp"])


def test_freeze_uses_the_last_snapshot_taken_before_kickoff(data_dir):
    """The structural no-leakage guarantee: a frozen number can only come
    from a snapshot that predates the kickoff it belongs to."""
    games = _week_games("2026-09-05T16:00:00Z")
    _write(data_dir, "20260903T120000Z", _rows({101: 1.0}), digest="d1",
           window="w", taken_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC))
    _write(data_dir, "20260905T153000Z", _rows({101: 4.0}), digest="d2",
           window="w", taken_at=datetime(2026, 9, 5, 15, 30, tzinfo=UTC))
    # Taken AFTER kickoff — must never be the freeze source.
    _write(data_dir, "20260905T170000Z", _rows({101: 50.0}), digest="d3",
           window="w", taken_at=datetime(2026, 9, 5, 17, 0, tzinfo=UTC))

    SN.freeze_kicked_off(data_dir, 2026, 2, games,
                         datetime(2026, 9, 5, 18, 0, tzinfo=UTC))
    frozen = SN.load_frozen(data_dir, 2026, 2)
    assert frozen[101]["from_stamp"] == "20260905T153000Z"
    assert float(frozen[101]["row"]["projected_home_margin"]) == 4.0


def test_freezing_is_idempotent_and_append_only(data_dir):
    games = _week_games("2026-09-05T16:00:00Z")
    _write(data_dir, "20260905T150000Z", _rows({101: 3.5}), digest="d1",
           window="w", taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))
    now = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)

    first = SN.freeze_kicked_off(data_dir, 2026, 2, games, now)
    second = SN.freeze_kicked_off(data_dir, 2026, 2, games, now)

    assert first["newly_frozen"] == [101]
    assert second["newly_frozen"] == [], "a frozen row is never re-frozen"
    assert float(SN.load_frozen(data_dir, 2026, 2)[101]["row"]
                 ["projected_home_margin"]) == 3.5


def test_a_game_with_no_pre_kickoff_snapshot_is_refused_not_invented(data_dir):
    games = _week_games("2026-09-05T16:00:00Z")
    # Only snapshot is AFTER kickoff.
    _write(data_dir, "20260905T170000Z", _rows({101: 12.0}), digest="d1",
           window="w", taken_at=datetime(2026, 9, 5, 17, 0, tzinfo=UTC))

    SN.freeze_kicked_off(data_dir, 2026, 2, games,
                         datetime(2026, 9, 5, 18, 0, tzinfo=UTC))
    frozen = SN.load_frozen(data_dir, 2026, 2)

    assert frozen[101]["from_stamp"] is None
    assert frozen[101]["row"]["status"] == FC.Status.REFUSED.value
    assert frozen[101]["row"]["refusal_reason"] == \
        SN.NO_PREGAME_SNAPSHOT.format(version=FC.SCHEMA_VERSION), \
        "an unforecast game stays an explicit refusal — never back-filled"
    assert frozen[101]["row"]["projected_home_margin"] is None


def test_frozen_rows_never_vanish_from_the_artifact(data_dir):
    """The full-schedule invariant survives freezing."""
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    _write(data_dir, "20260905T150000Z", _rows({101: 3.5, 102: -6.0}),
           digest="d1", window="w",
           taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))
    SN.freeze_kicked_off(data_dir, 2026, 2, games,
                         datetime(2026, 9, 5, 18, 0, tzinfo=UTC))

    out, _ = SN.apply_frozen(_rows({101: 1.0, 102: 1.0}),
                            SN.load_frozen(data_dir, 2026, 2))
    assert sorted(out["game_id"]) == [101, 102]


# --------------------------------------------------------------------------- #
# 6. Degraded mode — stale input degrades visibly, never silently
# --------------------------------------------------------------------------- #
def test_stale_source_degrades_visibly_instead_of_publishing_confident_data(data_dir):
    res = _write(data_dir, "20260901T120000Z",
                 _rows({101: 3.5}, degraded_reason="games feed 9 days stale"),
                 degraded=True, reason="games feed 9 days stale")

    meta = json.load(open(os.path.join(res["path"], "snapshot.json")))
    assert meta["degraded"] is True
    assert "stale" in meta["degraded_reason"]

    written = pd.read_csv(os.path.join(res["path"], "predictions.csv"))
    assert written.loc[0, "status"] == FC.Status.DEGRADED.value, \
        "every published row must carry the degraded marker"
    assert "stale" in str(written.loc[0, "refusal_reason"]), \
        "the reason must say so where a reader will actually see it"
    # The marker is not decoration: a degraded row publishes NO number, so
    # there is nothing confident-looking left to mistake for a forecast.
    assert pd.isna(written.loc[0, "projected_home_margin"])
    assert pd.isna(written.loc[0, "p_home_win"])

    assert SN.read_latest(data_dir, 2026, 2)["degraded"] is True


def test_a_healthy_snapshot_is_not_marked_degraded(data_dir):
    res = _write(data_dir, "20260901T120000Z", _rows({101: 3.5}))
    written = pd.read_csv(os.path.join(res["path"], "predictions.csv"))
    assert written.loc[0, "status"] == FC.Status.PRICED.value
    assert pd.isna(written.loc[0, "refusal_reason"])
    assert SN.read_latest(data_dir, 2026, 2)["degraded"] is False


def test_a_degraded_run_must_degrade_its_own_rows(data_dir):
    """The writer cannot stamp degradation on from outside any more.

    Marking the manifest degraded while every row still publishes a confident
    number is the exact disagreement between artifact and manifest that the
    old bolt-on column allowed, so the store refuses the write instead.
    """
    with pytest.raises(SN.SnapshotError, match="degraded"):
        _write(data_dir, "20260901T120000Z", _rows({101: 3.5}),
               degraded=True, reason="games feed stale")


# --------------------------------------------------------------------------- #
# 7. Reproducibility — a snapshot is checkable against its own manifest
# --------------------------------------------------------------------------- #
def test_snapshot_digest_detects_tampering(data_dir):
    res = _write(data_dir, "20260901T120000Z", _rows({101: 3.5}))
    assert SN.verify_snapshot(res["path"]) is True

    csv_path = os.path.join(res["path"], "predictions.csv")
    body = open(csv_path).read().replace("3.5", "9.9")
    open(csv_path, "w").write(body)
    assert SN.verify_snapshot(res["path"]) is False


def test_snapshots_are_listed_oldest_first_for_replay(data_dir):
    for stamp, m in (("20260903T120000Z", 1.0), ("20260901T120000Z", 2.0),
                     ("20260902T120000Z", 3.0)):
        _write(data_dir, stamp, _rows({101: m}), digest=stamp)
    assert [s["stamp"] for s in SN.list_snapshots(data_dir, 2026, 2)] == [
        "20260901T120000Z", "20260902T120000Z", "20260903T120000Z"]


# --------------------------------------------------------------------------- #
# 8. Cadence capstone — one CFB week, Tuesday through Saturday night
# --------------------------------------------------------------------------- #
def test_a_full_week_of_ticks_produces_one_snapshot_per_real_change(data_dir):
    """Tue open, Wed no-op, Fri injury news, then the three Saturday windows.

    Proves the whole contract end to end: no duplicate work, a snapshot for
    every genuine change, history intact, and every started game frozen to a
    number that predates its own kickoff.
    """
    games = _week_games("2026-09-05T16:00:00Z",      # noon
                        "2026-09-05T19:30:00Z",      # afternoon
                        "2026-09-05T23:00:00Z")      # night
    margins = {101: 3.5, 102: -6.0, 103: 10.0}

    def tick(now, digest, stamp):
        window = SN.kickoff_window(games, now)
        d = SN.decide(data_dir, 2026, 2, source_digest=digest, window=window)
        if not d["run"]:
            return None
        SN.freeze_kicked_off(data_dir, 2026, 2, games, now)
        rows, _ = SN.apply_frozen(_rows(margins),
                                  SN.load_frozen(data_dir, 2026, 2))
        _write(data_dir, stamp, rows, digest=digest, window=window,
               taken_at=now)
        return window

    assert tick(datetime(2026, 9, 1, 14, tzinfo=UTC), "d1",
                "20260901T140000Z") == "2026-09-05T16:00:00Z"      # Tue open
    assert tick(datetime(2026, 9, 2, 14, tzinfo=UTC), "d1",
                "20260902T140000Z") is None                        # Wed no-op
    margins = {101: 4.0, 102: -7.0, 103: 11.0}      # Friday injury report
    assert tick(datetime(2026, 9, 4, 14, tzinfo=UTC), "d2",
                "20260904T140000Z") == "2026-09-05T16:00:00Z"      # Fri news

    # Saturday. The model keeps moving; each game must stop moving at ITS OWN
    # kickoff, not at some shared cutoff.
    margins = {101: 99.0, 102: -8.0, 103: 12.0}
    assert tick(datetime(2026, 9, 5, 17, tzinfo=UTC), "d2",
                "20260905T170000Z") == "2026-09-05T19:00:00Z"      # post-noon
    margins = {101: 99.0, 102: 99.0, 103: 99.0}
    assert tick(datetime(2026, 9, 5, 21, tzinfo=UTC), "d2",
                "20260905T210000Z") == "2026-09-05T23:00:00Z"      # post-aft.

    stamps = [s["stamp"] for s in SN.list_snapshots(data_dir, 2026, 2)]
    assert stamps == ["20260901T140000Z", "20260904T140000Z",
                      "20260905T170000Z", "20260905T210000Z"], \
        "one snapshot per real change — and the Wednesday no-op left none"

    final = pd.read_csv(os.path.join(
        SN.snapshot_dir(data_dir, 2026, 2, "20260905T210000Z"),
        "predictions.csv")).set_index("game_id")
    assert float(final.loc[101, "projected_home_margin"]) == 4.0, \
        "the noon game froze at the last value published before ITS kickoff"
    assert float(final.loc[102, "projected_home_margin"]) == -8.0, \
        "the afternoon game kept moving until 19:30, then froze — not earlier"
    assert float(final.loc[103, "projected_home_margin"]) == 99.0, \
        "the night game has not kicked off and may still move"
    assert list(final["status"]) == [FC.Status.FROZEN.value,
                                     FC.Status.FROZEN.value,
                                     FC.Status.PRICED.value]

    frozen = SN.load_frozen(data_dir, 2026, 2)
    assert frozen[101]["from_stamp"] == "20260904T140000Z"
    assert frozen[102]["from_stamp"] == "20260905T170000Z"
    assert 103 not in frozen


# --------------------------------------------------------------------------- #
# 9. Hardening — a frozen row's provenance survives every later publish
# --------------------------------------------------------------------------- #
def test_a_frozen_degraded_row_keeps_its_degraded_marker_when_a_later_snapshot_is_healthy(data_dir):
    """A game frozen from a DEGRADED pregame snapshot carries that provenance
    forever. A later, healthy run for the rest of the week must not relabel
    it — a row's own `status` and `refusal_reason` describe what its OWN
    source looked like, not what this week's most recent run looked like.

    Degradation is a row state now rather than a side column, which is what
    makes "keeps its marker" checkable at all: the splice either reproduces
    the frozen row's `status="degraded"` and its reason verbatim, or it
    silently republishes a degraded game as this run's healthy one.
    """
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    _write(data_dir, "20260905T150000Z",
          _rows({101: 3.5, 102: -1.0}, degraded_reason="games feed stale"),
          digest="d1", window="w", degraded=True, reason="games feed stale",
          taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))

    SN.freeze_kicked_off(data_dir, 2026, 2, games,
                         datetime(2026, 9, 5, 18, 0, tzinfo=UTC))   # 101 froze
    frozen = SN.load_frozen(data_dir, 2026, 2)
    assert frozen[101]["row"]["status"] == FC.Status.DEGRADED.value
    assert frozen[101]["row"]["refusal_reason"] == "games feed stale"

    # A later, HEALTHY run recomputes the still-pending game and republishes.
    later_rows, _ = SN.apply_frozen(_rows({101: None, 102: 8.0}), frozen)
    res = SN.write_snapshot(data_dir, 2026, 2, "20260905T190000Z", later_rows,
                            source_digest="d2", window="w2", degraded=False,
                            taken_at=datetime(2026, 9, 5, 19, 0, tzinfo=UTC))

    published = pd.read_csv(os.path.join(res["path"], "predictions.csv")
                            ).set_index("game_id")
    assert published.loc[101, "status"] == FC.Status.DEGRADED.value, \
        "the frozen game's own degraded provenance must survive a healthy rerun"
    assert published.loc[101, "refusal_reason"] == "games feed stale"
    assert published.loc[101, "frozen_from_stamp"] == "20260905T150000Z", \
        "and the freeze still records which snapshot it came from"
    assert pd.isna(published.loc[101, "projected_home_margin"]), \
        "a healthy rerun may not fill in the number the degraded run withheld"
    assert published.loc[102, "status"] == FC.Status.PRICED.value, \
        "a game that is not frozen must reflect the CURRENT run's health"
    assert pd.isna(published.loc[102, "refusal_reason"])


def test_refused_frozen_row_is_never_altered_by_a_later_publish(data_dir):
    """A kicked-off game with no qualifying pregame snapshot becomes a
    structural refusal. A later publish for the rest of the week — even a
    DEGRADED one — must reproduce that refusal's status and reason unchanged.
    Restating it as this run's degradation, or flipping it back to a priced
    row, is backfill wearing a different hat."""
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T23:00:00Z")
    # Only 102 (the later kickoff) ever gets a pregame snapshot; 101 never does.
    _write(data_dir, "20260905T150000Z", _rows({102: 5.0}), digest="d1",
          window="w", taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))

    SN.freeze_kicked_off(data_dir, 2026, 2, games,
                         datetime(2026, 9, 5, 17, 0, tzinfo=UTC))   # 101 froze
    frozen = SN.load_frozen(data_dir, 2026, 2)
    refusal_reason = frozen[101]["row"]["refusal_reason"]
    assert frozen[101]["from_stamp"] is None

    # The recompute would have priced 102 at 9.0, but a degraded run stands
    # behind no number, so the rows it builds carry the reason instead.
    later_rows, _ = SN.apply_frozen(
        _rows({101: None, 102: 9.0}, degraded_reason="games feed stale"),
        frozen)
    res = SN.write_snapshot(data_dir, 2026, 2, "20260905T173000Z", later_rows,
                            source_digest="d2", window="w2", degraded=True,
                            degraded_reason="games feed stale",
                            taken_at=datetime(2026, 9, 5, 17, 30, tzinfo=UTC))

    published = pd.read_csv(os.path.join(res["path"], "predictions.csv")
                            ).set_index("game_id")
    assert published.loc[101, "status"] == FC.Status.REFUSED.value
    assert published.loc[101, "refusal_reason"] == refusal_reason, \
        "a structural refusal must not be restated as this run's degradation " \
        "when it was never part of that run"
    assert pd.isna(published.loc[101, "projected_home_margin"]), \
        "a game with no pregame snapshot may never be back-filled with a number"
    assert published.loc[102, "status"] == FC.Status.DEGRADED.value, \
        "a genuinely live row this run must still show the current health"
    assert published.loc[102, "refusal_reason"] == "games feed stale"


# --------------------------------------------------------------------------- #
# 10. Hardening — tampered and forged-duplicate snapshots are never trusted
# --------------------------------------------------------------------------- #
def test_freeze_never_trusts_a_tampered_pre_kickoff_snapshot(data_dir):
    """A tampered snapshot stops the run; it is never quietly stepped over.

    This used to assert that `freeze_kicked_off` skipped the tampered file and
    froze from an older valid one. That is not enough. A store containing an
    altered immutable record is DAMAGED, and silently continuing over it makes
    the week read as though it simply had fewer snapshots — the same shape as
    a week with none, which `decide` treats as "nothing published yet, go
    ahead". Damaged state now raises, and nothing is frozen from a store the
    pipeline cannot vouch for.
    """
    games = _week_games("2026-09-05T16:00:00Z", "2026-09-05T16:00:00Z")

    only = _write(data_dir, "20260905T150000Z", _rows({101: 3.5}), digest="d1",
                 window="w", taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))
    csv_path = os.path.join(only["path"], "predictions.csv")
    open(csv_path, "w").write(open(csv_path).read().replace("3.5", "999.0"))
    assert SN.verify_snapshot(only["path"]) is False

    with pytest.raises(SN.SnapshotIntegrityError, match="sha256"):
        SN.list_snapshots(data_dir, 2026, 2)

    with pytest.raises(SN.SnapshotIntegrityError):
        SN.freeze_kicked_off(data_dir, 2026, 2, games,
                             datetime(2026, 9, 5, 18, 0, tzinfo=UTC))

    # And the damage did not leak into the freeze record: nothing was written,
    # so no game is recorded as frozen from a snapshot nobody can trust.
    assert not os.path.exists(SN.frozen_path(data_dir, 2026, 2))


def test_a_forged_duplicate_stamp_fails_loudly(data_dir):
    """snapshot.json's declared `stamp` is untrusted input.

    A manifest claiming a stamp that belongs to a DIFFERENT directory is a
    forged or duplicated identity. Excluding it from the replay list kept it
    from winning, but left a store with a forgery in it looking merely
    smaller. It raises now, naming the stamp, so the forgery gets looked at
    rather than routed around.
    """
    _write(data_dir, "20260905T140000Z", _rows({101: 4.0}), digest="d1",
          window="w", taken_at=datetime(2026, 9, 5, 14, 0, tzinfo=UTC))

    forged_dir = SN.snapshot_dir(data_dir, 2026, 2, "20260905T150000Z")
    os.makedirs(forged_dir)
    _rows({101: 50.0}).to_csv(os.path.join(forged_dir, "predictions.csv"),
                              index=False)
    forged_meta = {
        "stamp": "20260905T140000Z",   # claims the EARLIER snapshot's identity
        "season": 2026, "week": 2, "source_digest": "d-forged", "window": "w",
        "degraded": False, "degraded_reason": None,
        "taken_at": datetime(2026, 9, 5, 15, 0, tzinfo=UTC).isoformat(),
        "n_rows": 1, "n_priced": 1,
        "sha256": SN._sha256_file(os.path.join(forged_dir, "predictions.csv")),
        "paper_only": True,
    }
    with open(os.path.join(forged_dir, "snapshot.json"), "w") as fh:
        json.dump(forged_meta, fh)

    with pytest.raises(SN.SnapshotIntegrityError, match="20260905T140000Z"):
        SN.list_snapshots(data_dir, 2026, 2)

def test_replaying_a_snapshot_reproduces_it_byte_for_byte(data_dir):
    """BUILD_SPEC section 13 DoD: re-running the same manifest reproduces
    identical output. Same rows in, same bytes out."""
    rows = _rows({101: 3.5, 102: None})
    a = _write(data_dir, "20260901T120000Z", rows, digest="d", window="w")
    b = _write(data_dir, "20260901T120001Z", rows, digest="d", window="w")
    assert a["sha256"] == b["sha256"]
    assert SN.verify_snapshot(a["path"]) and SN.verify_snapshot(b["path"])
