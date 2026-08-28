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

PAPER-ONLY throughout: nothing here produces or implies a bet.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import pytest

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


def _rows(margins):
    """Artifact rows in the published contract shape (see week_prediction_rows)."""
    out = []
    for gid, m in margins.items():
        vetoed = m is None
        out.append({
            "season": 2026, "week": 2, "id": gid, "slate": "wk2",
            "date": "2026-09-05", "home": f"Home{gid}", "away": f"Away{gid}",
            "neutral": False,
            "mc_margin": None if vetoed else m,
            "mc_margin_sd": None if vetoed else 16.5,
            "mc_total": None if vetoed else 51.0,
            "p_home_win": None if vetoed else 0.58,
            "param_sd": None if vetoed else 1.9,
            "pred_margin": None if vetoed else m,
            "band_pm": None if vetoed else 16.5,
            "veto": vetoed,
            "note": "refused" if vetoed else None,
        })
    return pd.DataFrame(out)


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
    assert float(first.loc[0, "mc_margin"]) == 3.5, \
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
    _write(data_dir, "20260901T120000Z", _rows({101: 3.5}), digest="abc",
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
    row101 = out.set_index("id").loc[101]
    assert float(row101["mc_margin"]) == 3.5, \
        "the kicked-off game must keep its final PREGAME number, verbatim"
    assert bool(row101["frozen"]) is True
    row102 = out.set_index("id").loc[102]
    assert float(row102["mc_margin"]) == -2.0, \
        "a game that has not kicked off is still free to move"
    assert bool(row102["frozen"]) is False


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
    assert float(frozen[101]["row"]["mc_margin"]) == 4.0


def test_freezing_is_idempotent_and_append_only(data_dir):
    games = _week_games("2026-09-05T16:00:00Z")
    _write(data_dir, "20260905T150000Z", _rows({101: 3.5}), digest="d1",
           window="w", taken_at=datetime(2026, 9, 5, 15, 0, tzinfo=UTC))
    now = datetime(2026, 9, 5, 18, 0, tzinfo=UTC)

    first = SN.freeze_kicked_off(data_dir, 2026, 2, games, now)
    second = SN.freeze_kicked_off(data_dir, 2026, 2, games, now)

    assert first["newly_frozen"] == [101]
    assert second["newly_frozen"] == [], "a frozen row is never re-frozen"
    assert float(SN.load_frozen(data_dir, 2026, 2)[101]["row"]["mc_margin"]) == 3.5


def test_a_game_with_no_pre_kickoff_snapshot_is_refused_not_invented(data_dir):
    games = _week_games("2026-09-05T16:00:00Z")
    # Only snapshot is AFTER kickoff.
    _write(data_dir, "20260905T170000Z", _rows({101: 12.0}), digest="d1",
           window="w", taken_at=datetime(2026, 9, 5, 17, 0, tzinfo=UTC))

    SN.freeze_kicked_off(data_dir, 2026, 2, games,
                         datetime(2026, 9, 5, 18, 0, tzinfo=UTC))
    frozen = SN.load_frozen(data_dir, 2026, 2)

    assert frozen[101]["from_stamp"] is None
    assert frozen[101]["row"]["veto"] is True
    assert "no pregame snapshot" in frozen[101]["row"]["note"].lower(), \
        "an unforecast game stays an explicit refusal — never back-filled"
    assert frozen[101]["row"]["mc_margin"] is None


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
    assert sorted(out["id"]) == [101, 102]


# --------------------------------------------------------------------------- #
# 6. Degraded mode — stale input degrades visibly, never silently
# --------------------------------------------------------------------------- #
def test_stale_source_degrades_visibly_instead_of_publishing_confident_data(data_dir):
    res = _write(data_dir, "20260901T120000Z", _rows({101: 3.5}),
                 degraded=True, reason="games feed 9 days stale")

    meta = json.load(open(os.path.join(res["path"], "snapshot.json")))
    assert meta["degraded"] is True
    assert "stale" in meta["degraded_reason"]

    written = pd.read_csv(os.path.join(res["path"], "predictions.csv"))
    assert bool(written.loc[0, "degraded"]) is True, \
        "every published row must carry the degraded marker"
    assert "DEGRADED" in str(written.loc[0, "note"]), \
        "the note must say so where a reader will actually see it"

    assert SN.read_latest(data_dir, 2026, 2)["degraded"] is True


def test_a_healthy_snapshot_is_not_marked_degraded(data_dir):
    res = _write(data_dir, "20260901T120000Z", _rows({101: 3.5}))
    written = pd.read_csv(os.path.join(res["path"], "predictions.csv"))
    assert bool(written.loc[0, "degraded"]) is False
    assert SN.read_latest(data_dir, 2026, 2)["degraded"] is False


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
        "predictions.csv")).set_index("id")
    assert float(final.loc[101, "mc_margin"]) == 4.0, \
        "the noon game froze at the last value published before ITS kickoff"
    assert float(final.loc[102, "mc_margin"]) == -8.0, \
        "the afternoon game kept moving until 19:30, then froze — not earlier"
    assert float(final.loc[103, "mc_margin"]) == 99.0, \
        "the night game has not kicked off and may still move"
    assert [bool(x) for x in final["frozen"]] == [True, True, False]

    frozen = SN.load_frozen(data_dir, 2026, 2)
    assert frozen[101]["from_stamp"] == "20260904T140000Z"
    assert frozen[102]["from_stamp"] == "20260905T170000Z"
    assert 103 not in frozen


def test_replaying_a_snapshot_reproduces_it_byte_for_byte(data_dir):
    """BUILD_SPEC section 13 DoD: re-running the same manifest reproduces
    identical output. Same rows in, same bytes out."""
    rows = _rows({101: 3.5, 102: None})
    a = _write(data_dir, "20260901T120000Z", rows, digest="d", window="w")
    b = _write(data_dir, "20260901T120001Z", rows, digest="d", window="w")
    assert a["sha256"] == b["sha256"]
    assert SN.verify_snapshot(a["path"]) and SN.verify_snapshot(b["path"])
