"""The weekly pipeline must not be able to publish misleading numbers.

Six things used to make that possible, and there is a test here for each:

* the freshness gate checked one file (games) and hardcoded `schema_ok=True`,
  so a run could pass it with no drives on disk at all;
* failing the gate marked the run `degraded` and then carried on into
  simulation and publication anyway;
* `build_site.py` swallowed hub-feed failures and `run_weekly.sh` logged them,
  so the runner reported success while the public surface went stale;
* `run_weekly.sh` committed `data/site` and then ran an UNSCOPED
  `git push origin master`, unattended, on a schedule;
* the lock was exists-then-write, and the `latest` pointer was a plain write;
* corrupt snapshot metadata read as "no snapshot", which is the same shape as
  "nothing published yet, go ahead and publish".
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import atomic  # noqa: E402
from nflvalue import data_health as DH  # noqa: E402
from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import snapshots as SN  # noqa: E402

UTC = timezone.utc
SEASON = 2026


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WL = _load("weekly_loop", "scripts/weekly_loop.py")


# --------------------------------------------------------------------------- #
# A synthetic data directory in which every declared input is healthy
# --------------------------------------------------------------------------- #
def _games_frame(season, weeks=(1, 2, 8)):
    return pd.DataFrame([
        {"id": season * 1000 + w * 10 + i, "season": season, "week": w,
         "start_date": f"{season}-09-0{min(w, 9)}T16:00:00Z",
         "neutral_site": False, "home_team": f"H{i}", "away_team": f"A{i}",
         "home_points": None, "away_points": None,
         "home_conference": "C", "away_conference": "C",
         "home_fcs": False, "away_fcs": False}
        for w in weeks for i in (1, 2)])


def _drives_frame(season):
    return pd.DataFrame([
        {"season": season, "week": w, "offense": f"H{i}", "defense": f"A{i}",
         "start_period": 1}
        for w in (1, 2) for i in (1, 2)])


@pytest.fixture()
def healthy(tmp_path):
    """Every input `data_health` declares, present and current."""
    data = tmp_path / "data"
    (data / "cfbd").mkdir(parents=True)
    for season in range(DH.HISTORY_START, SEASON + 1):
        _games_frame(season).to_parquet(data / "cfbd" / f"games_{season}.parquet")
        _drives_frame(season).to_parquet(
            data / "cfbd" / f"drives_{season}.parquet")
    pd.DataFrame([{"team": f"T{i}", "is_fbs": True} for i in range(12)]
                 ).to_parquet(data / "cfbd" / f"teams_{SEASON}.parquet")
    pd.DataFrame([{"game_id": 1, "spread": -3.0}]
                 ).to_parquet(data / "cfbd" / f"lines_{SEASON}.parquet")

    pd.DataFrame([{"slice": "all", "clv_pts": 0.1}] * 120
                 ).to_csv(data / "clv_log_backtest.csv", index=False)
    # The benchmark input is the evidence MANIFEST, not a backtest file. What
    # the pipeline needs in order to make a benchmark claim is the selection
    # gate; which artifact (if any) is selected is a display decision, not a
    # reason to block a forecast.
    shutil.copy(ROOT / "data" / "evidence_manifest.json",
                data / "evidence_manifest.json")
    pd.DataFrame([{"season": 2021, "week": 1, "residual": 1.0}] * 400
                 ).to_csv(data / "backtest_early.csv", index=False)
    shutil.copy(ROOT / "data" / "forecast_sd_basis.json",
                data / "forecast_sd_basis.json")
    return data


def _evaluate(data, week=8, job="week-open", family="kalman_montecarlo", **kw):
    return DH.evaluate(str(data), SEASON, week, job, family, **kw)


def _age(path, hours):
    stamp = time.time() - hours * 3600
    os.utime(path, (stamp, stamp))


# --------------------------------------------------------------------------- #
# 1. Dependency-aware freshness
# --------------------------------------------------------------------------- #
def test_a_fully_stocked_data_dir_is_healthy(healthy):
    report = _evaluate(healthy)
    assert report.healthy, report.reasons
    assert report.blocked == ()


def test_missing_drives_block_drive_simulation(healthy):
    """The headline gap: the old gate could not see this at all."""
    os.remove(healthy / "cfbd" / f"drives_{SEASON}.parquet")
    report = _evaluate(healthy)
    assert not report.healthy
    assert "drive_simulation" in report.blocked
    assert not report.inputs["drives"].ok
    assert any("drives" in reason for reason in report.reasons)


def test_stale_drives_block_drive_simulation(healthy):
    _age(healthy / "cfbd" / f"drives_{SEASON}.parquet",
         DH.CURRENT_SEASON_STALENESS_HOURS + 24)
    report = _evaluate(healthy)
    assert not report.healthy
    assert "drive_simulation" in report.blocked


def test_missing_drive_history_blocks_drive_simulation(healthy):
    os.remove(healthy / "cfbd" / "drives_2022.parquet")
    report = _evaluate(healthy)
    assert not report.healthy
    assert "drive_simulation" in report.blocked


def test_drives_are_irrelevant_to_the_early_season_route(healthy):
    """Engine-aware, not one global list.

    Weeks 0-3 are priced by ridge + preseason prior, which runs no simulation.
    Blocking those weeks on a drives cache they never read would be a false
    alarm, and false alarms are how a gate gets ignored.
    """
    os.remove(healthy / "cfbd" / f"drives_{SEASON}.parquet")
    report = _evaluate(healthy, week=1, family="ridge_preseason_prior")
    assert report.healthy, report.reasons
    assert "drive_simulation" not in report.required


def test_lines_are_optional_and_only_suppress_the_insight(healthy):
    """No forecast depends on a market line; a displayed comparison does."""
    os.remove(healthy / "cfbd" / f"lines_{SEASON}.parquet")
    report = _evaluate(healthy)
    assert report.healthy, report.reasons
    assert report.blocked == ()
    assert "market_insight" in report.optional_unavailable
    assert DH.status_of(report) == DH.STATUS_LIMITED


def test_missing_benchmark_evidence_blocks_the_claims_that_cite_it(healthy):
    """The explainer quotes these back at the reader as evidence."""
    os.remove(healthy / "evidence_manifest.json")
    report = _evaluate(healthy)
    assert not report.healthy
    assert "benchmark_claims" in report.blocked


def test_missing_sd_basis_blocks_pricing(healthy):
    os.remove(healthy / "forecast_sd_basis.json")
    report = _evaluate(healthy)
    assert not report.healthy
    assert "forecast_sd_basis" in report.blocked


def test_a_schema_violation_is_caught(healthy):
    """`schema_ok` was hardcoded True; now the columns are actually read."""
    bad = _games_frame(SEASON).drop(columns=["start_date"])
    bad.to_parquet(healthy / "cfbd" / f"games_{SEASON}.parquet")
    report = _evaluate(healthy)
    assert not report.healthy
    assert "start_date" in report.inputs["games"].reason


def test_an_empty_input_is_caught(healthy):
    _games_frame(SEASON).iloc[0:0].to_parquet(
        healthy / "cfbd" / f"games_{SEASON}.parquet")
    report = _evaluate(healthy)
    assert not report.healthy
    assert not report.inputs["games"].ok


def test_a_week_the_source_does_not_cover_is_caught(healthy):
    """Coverage, not just freshness: a current file can still lack the week."""
    report = _evaluate(healthy, week=13)
    assert not report.healthy
    assert "coverage" in report.inputs["games"].reason


def test_a_future_dated_input_is_treated_as_leakage(healthy):
    path = healthy / "cfbd" / f"games_{SEASON}.parquet"
    _age(path, -48)                       # 48 hours in the future
    report = _evaluate(healthy)
    assert report.leakage_suspected
    assert not report.healthy


def test_historical_caches_never_go_stale(healthy):
    """A finished season is an immutable record, not a stale feed."""
    _age(healthy / "cfbd" / "games_2021.parquet", 24 * 3650)
    _age(healthy / "cfbd" / "drives_2021.parquet", 24 * 3650)
    report = _evaluate(healthy)
    assert report.healthy, report.reasons


# --------------------------------------------------------------------------- #
# 2. Fail-closed publication
# --------------------------------------------------------------------------- #
@pytest.fixture()
def loop_sandbox(healthy, monkeypatch):
    """`weekly_loop` pointed at the synthetic data dir."""
    manifests = healthy / "manifests"
    manifests.mkdir()
    monkeypatch.setattr(WL, "DATA", str(healthy))
    monkeypatch.setattr(WL, "MANIFESTS", str(manifests))
    monkeypatch.setattr(WL, "LOCK_PATH", str(healthy / ".weekly.lock"))
    monkeypatch.setattr(WL.ingest, "calls_used_this_month", lambda: 10)
    monkeypatch.setattr(WL.ingest, "load_games",
                        lambda seasons, **kw: _games_frame(SEASON))
    monkeypatch.setattr(WL, "current_season_week",
                        lambda now=None: (SEASON, 8, True))
    monkeypatch.setattr(WL, "notify", lambda manifest: {"status": "skipped"})
    monkeypatch.setattr(WL, "data_as_of", lambda: {})
    return healthy


def _latest_manifest(data_dir):
    paths = sorted((data_dir / "manifests").glob("*.json"))
    assert paths, "the run must leave a manifest even when it halts"
    return json.loads(paths[-1].read_text())


def test_unhealthy_inputs_halt_before_anything_is_published(loop_sandbox,
                                                            monkeypatch):
    """The core requirement, end to end.

    The old flow set `degraded=True` here and then ran simulate anyway, which
    advanced `latest` and overwrote the published artifact with numbers built
    on data nothing had checked.
    """
    os.remove(loop_sandbox / "cfbd" / f"drives_{SEASON}.parquet")

    published = loop_sandbox / f"predictions_{SEASON}_wk08.csv"
    published.write_text("PRE-EXISTING VERIFIED ARTIFACT\n")
    week_dir = Path(SN.week_dir(str(loop_sandbox), SEASON, 8))
    week_dir.mkdir(parents=True)
    atomic.write_json(SN.latest_path(str(loop_sandbox), SEASON, 8),
                      {"stamp": "PRIOR"})

    simulated = []
    monkeypatch.setattr(WL, "stage_simulate",
                        lambda *a, **k: simulated.append(1) or {"n": 1})

    code = WL.run("pregame", force_week=8)

    assert code == WL.EXIT_UNHEALTHY_INPUTS
    assert not simulated, "simulation must not run on unhealthy inputs"
    assert published.read_text() == "PRE-EXISTING VERIFIED ARTIFACT\n", \
        "a verified artifact must not be overwritten by a halted run"
    assert json.loads(
        Path(SN.latest_path(str(loop_sandbox), SEASON, 8)).read_text()
    )["stamp"] == "PRIOR", "the latest pointer must not advance"
    assert not list(week_dir.glob("2026*")), "no snapshot may be written"

    manifest = _latest_manifest(loop_sandbox)
    assert manifest["halted"] is True
    assert "drive_simulation" in manifest["blocked_capabilities"]
    assert "drives" in manifest["halt_reason"]
    assert manifest["stages"]["simulate"]["skipped"] is True


def test_the_halt_status_is_public_and_carries_no_numbers(loop_sandbox):
    """"Data unavailable" must not be able to look like a forecast."""
    os.remove(loop_sandbox / "cfbd" / f"drives_{SEASON}.parquet")
    WL.run("pregame", force_week=8)

    status = json.loads(
        (loop_sandbox / DH.PUBLIC_STATUS_FILE).read_text())
    assert status["status"] == DH.STATUS_UNAVAILABLE
    assert status["numeric_forecast_published"] is False
    assert "unavailable" in status["headline"].lower()
    assert "drive_simulation" in status["blocked_capabilities"]

    # Nothing in the status may be mistakable for a forecast: no field in it
    # is a bare number except the season/week identity and the schema version.
    numeric = {k: v for k, v in status.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert set(numeric) <= {"season", "week", "schema_version"}
    for banned in FC.FORECAST_COLUMNS:
        assert banned not in json.dumps(status)


def test_a_healthy_run_records_that_it_published(loop_sandbox, monkeypatch):
    monkeypatch.setattr(
        WL, "stage_simulate",
        lambda *a, **k: {"n": 2, "n_schedule": 2, "skipped": False})
    monkeypatch.setattr(WL, "pregame_decision",
                        lambda *a, **k: {"run": True, "reason": "test",
                                         "window": "w", "source_digest": "d",
                                         "n_pending": 2})
    assert WL.run("pregame", force_week=8) == 0
    status = json.loads((loop_sandbox / DH.PUBLIC_STATUS_FILE).read_text())
    assert status["status"] == DH.STATUS_OK
    assert status["numeric_forecast_published"] is True


# --------------------------------------------------------------------------- #
# 3. Snapshot integrity
# --------------------------------------------------------------------------- #
def test_corrupt_frozen_metadata_fails_loudly(tmp_path):
    """The leakage-critical one.

    `{}` for an unreadable freeze record told the caller "no game has kicked
    off", so a started game would be re-forecast and a post-kickoff number
    published as a pregame one.
    """
    week = Path(SN.week_dir(str(tmp_path), 2026, 2))
    week.mkdir(parents=True)
    (week / "frozen.json").write_text("{not json")
    with pytest.raises(SN.SnapshotIntegrityError, match="unreadable|could not be read"):
        SN.load_frozen(str(tmp_path), 2026, 2)


def test_a_missing_frozen_file_is_still_legitimately_empty(tmp_path):
    """Absent is a real state; damaged is not. They must not be conflated."""
    assert SN.load_frozen(str(tmp_path), 2026, 2) == {}


def test_corrupt_latest_pointer_fails_loudly(tmp_path):
    week = Path(SN.week_dir(str(tmp_path), 2026, 2))
    week.mkdir(parents=True)
    (week / "latest.json").write_text("{truncated")
    with pytest.raises(SN.SnapshotIntegrityError):
        SN.read_latest(str(tmp_path), 2026, 2)


def test_a_pointer_naming_a_missing_stamp_fails_loudly(tmp_path):
    """Otherwise `decide` reads it as "no prior snapshot" and republishes."""
    week = Path(SN.week_dir(str(tmp_path), 2026, 2))
    week.mkdir(parents=True)
    atomic.write_json(str(week / "latest.json"), {"stamp": "20260905T140000Z"})
    with pytest.raises(SN.SnapshotIntegrityError, match="not on disk"):
        SN.read_latest(str(tmp_path), 2026, 2)


def test_corrupt_snapshot_manifest_fails_loudly(tmp_path):
    stamp_dir = Path(SN.snapshot_dir(str(tmp_path), 2026, 2, "20260905T140000Z"))
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "snapshot.json").write_text("{oops")
    with pytest.raises(SN.SnapshotIntegrityError):
        SN.list_snapshots(str(tmp_path), 2026, 2)


def test_the_latest_pointer_is_written_atomically(tmp_path):
    target = tmp_path / "latest.json"
    atomic.write_json(str(target), {"stamp": "A"})
    assert json.loads(target.read_text())["stamp"] == "A"
    assert not list(tmp_path.glob("*.tmp*")), "no temp file may be left behind"

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        atomic.write_json(str(target), {"stamp": Unserialisable()})
    assert json.loads(target.read_text())["stamp"] == "A", \
        "a failed write must leave the previous pointer intact"
    assert not list(tmp_path.glob("*.tmp*"))


def test_the_lock_is_taken_atomically(tmp_path, monkeypatch):
    """One syscall creates and claims it, so two runs cannot both proceed."""
    lock = tmp_path / ".weekly.lock"
    monkeypatch.setattr(WL, "DATA", str(tmp_path))
    monkeypatch.setattr(WL, "LOCK_PATH", str(lock))

    assert WL.acquire_lock("pregame") is True
    assert WL.acquire_lock("pregame") is False, \
        "a second run must not share the single-writer lock"
    WL.release_lock()
    assert WL.acquire_lock("pregame") is True


def test_a_stale_lock_is_broken_but_a_live_one_is_not(tmp_path, monkeypatch):
    lock = tmp_path / ".weekly.lock"
    monkeypatch.setattr(WL, "DATA", str(tmp_path))
    monkeypatch.setattr(WL, "LOCK_PATH", str(lock))
    old = (datetime.now(UTC)
           - timedelta(hours=WL.LOCK_STALE_HOURS + 1)).isoformat()
    atomic.write_json(str(lock), {"pid": 1, "job": "pregame", "utc": old})
    assert WL.acquire_lock("pregame") is True


def test_lock_acquisition_uses_an_exclusive_create(tmp_path):
    """Structural: exists-then-write has a window; O_EXCL does not."""
    source = (ROOT / "scripts" / "weekly_loop.py").read_text()
    body = source[source.index("def acquire_lock("):]
    body = body[:body.index("\ndef ")]
    assert "O_EXCL" in body and "O_CREAT" in body
    assert "os.path.exists(LOCK_PATH)" not in body


# --------------------------------------------------------------------------- #
# 4. The runner publishes nothing
# --------------------------------------------------------------------------- #
def _executable_lines(path):
    """Shell lines with comments and blank lines removed."""
    out = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(line)
    return out


def test_the_runner_invokes_no_git_command_at_all():
    lines = _executable_lines(ROOT / "scripts" / "run_weekly.sh")
    offenders = [l for l in lines if re.search(r"\bgit\b", l)]
    assert not offenders, (
        "run_weekly.sh must never run git; publishing is a manual command:\n"
        + "\n".join(offenders))


@pytest.mark.parametrize("verb", ["commit", "push", "add"])
def test_the_runner_contains_no_publish_verb(verb):
    """Tolerant of options between `git` and the verb.

    A literal `"git push"` check misses `git -C "$REPO" push origin master`,
    which is exactly the form the old runner used — verified by mutation:
    appending that line left this test green until the pattern was widened.
    """
    lines = _executable_lines(ROOT / "scripts" / "run_weekly.sh")
    pattern = re.compile(r"\bgit\b(?:\s+-\S+(?:\s+\S+)?)*\s+" + verb + r"\b")
    offenders = [l for l in lines if pattern.search(l)]
    assert not offenders, "\n".join(offenders)


def test_no_launchd_job_publishes():
    """Scans what launchd would EXECUTE, not the comments.

    The plists document the removed publish step on purpose -- an operator
    reading one should learn why the automatic push is gone. Scanning raw text
    would flag that explanation, so this parses the plist and checks the
    values launchd actually acts on.
    """
    import plistlib
    for path in sorted((ROOT / "ops" / "launchd").glob("*.plist")):
        with open(path, "rb") as handle:
            plist = plistlib.load(handle)
        acted_on = json.dumps({k: v for k, v in plist.items()
                               if k != "Label"}, default=str)
        assert "git" not in acted_on, f"{path.name} would run git"
        assert "publish_site.sh" not in acted_on, (
            f"{path.name} schedules the manual publish command; publishing "
            "must not be automated")
        assert "STALLIONS_NO_PUBLISH" not in json.dumps(
            plist.get("EnvironmentVariables", {})), (
            f"{path.name} sets a kill switch for a publish step that no "
            "longer exists")


def test_the_manual_publish_command_exists_and_guards_itself():
    script = ROOT / "scripts" / "publish_site.sh"
    assert script.exists()
    text = script.read_text()
    assert "git push origin master" in text, "it is the publisher"
    # ...and it refuses in the four ways the old automatic publish could not.
    assert 'BRANCH" = "master"' in text or '"$BRANCH" = "master"' in text
    assert "origin/master..master" in text, "must show what the push carries"
    assert "data_unavailable" in text, "must refuse after a halted run"
    assert "read -r REPLY" in text, "must require confirmation"


def test_the_runner_reports_a_build_failure_as_unsuccessful():
    text = (ROOT / "scripts" / "run_weekly.sh").read_text()
    assert "CODE=4" in text
    assert 'exit "$CODE"' in text


# --------------------------------------------------------------------------- #
# 5. A failed hub/site build is a failed run
# --------------------------------------------------------------------------- #
def test_build_site_propagates_a_hub_failure(monkeypatch):
    site = _load("build_site", "scripts/build_site.py")

    class Result:
        returncode = 2

    # `subprocess` is imported inside the function, so patch the real module.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    assert site._also_build_hub_feed() == 2


def test_build_site_exit_status_includes_the_hub():
    """Structural: the hub's code must reach SystemExit, not be discarded."""
    text = (ROOT / "scripts" / "build_site.py").read_text()
    assert "raise SystemExit(_code or _hub)" in text
    assert "check=False" not in text or "returncode" in text


def _mini_repo(tmp_path):
    """A throwaway tree the hub script can run in, with its own data dir."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "data" / "site" / "api").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "build_hub_feed.py",
                repo / "scripts" / "build_hub_feed.py")
    os.symlink(ROOT / "nflvalue", repo / "nflvalue")
    return repo


def test_the_hub_build_exits_nonzero_on_a_contract_violation(tmp_path):
    """End to end: a v1 artifact makes the hub build FAIL, and publish nothing."""
    repo = _mini_repo(tmp_path)
    (repo / "data" / f"predictions_{SEASON}_wk08.csv").write_text(
        "season,week,id,pred_margin,band_pm,veto\n"
        f"{SEASON},8,101,6.0,3.0,False\n")
    out = repo / "data" / "site" / "api" / "hub.json"
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "build_hub_feed.py"),
         "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "forecast contract" in (proc.stderr + proc.stdout)
    assert not out.exists(), "a failed hub build must publish nothing"


# --------------------------------------------------------------------------- #
# 6. The hub carries the state a reader needs
# --------------------------------------------------------------------------- #
def _contract_rows():
    base = {c: None for c in FC.COLUMNS}
    common = {"schema_version": FC.SCHEMA_VERSION, "season": SEASON, "week": 8,
              "kickoff_utc": f"{SEASON}-10-17T16:00:00Z", "neutral_site": False,
              "slate": "wk8", "public_date": f"{SEASON}-10-17",
              "model_family": "kalman_montecarlo", "model_version": "fp",
              "generated_at_utc": f"{SEASON}-10-14T12:00:00Z",
              "snapshot_id": "20261014T120000Z", "source_digest": "digest-abc",
              "config_hash": "cfg"}
    frozen = {**base, **common, "game_id": 101, "home": "H1", "away": "A1",
              "status": FC.Status.FROZEN.value,
              "projected_home_margin": 6.0, "forecast_margin_sd": 17.0,
              "p_home_win": 0.638, "projected_total": 51.0,
              "rating_parameter_sd": 2.0,
              "forecast_sd_basis": "sim_kalman_wk4_plus",
              "frozen_from_stamp": "20261013T120000Z"}
    degraded = {**base, **common, "game_id": 102, "home": "H2", "away": "A2",
                "status": FC.Status.DEGRADED.value,
                "refusal_reason": "games feed stale"}
    return pd.DataFrame([frozen, degraded], columns=list(FC.COLUMNS))


def test_the_hub_preserves_frozen_degraded_snapshot_id_and_digest(tmp_path):
    repo = _mini_repo(tmp_path)
    FC.write_artifact(_contract_rows(),
                      str(repo / "data" / f"predictions_{SEASON}_wk08.csv"))
    DH_status = {"schema_version": 1, "status": DH.STATUS_LIMITED,
                 "headline": "x", "numeric_forecast_published": True}
    atomic.write_json(str(repo / "data" / DH.PUBLIC_STATUS_FILE), DH_status)

    out = repo / "data" / "site" / "api" / "hub.json"
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "build_hub_feed.py"),
         "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    feed = json.loads(out.read_text())
    rows = {g["game_id"]: g for g in feed["week"]["games"]}
    assert rows[101]["status"] == FC.Status.FROZEN.value
    assert rows[101]["frozen_from_stamp"] == "20261013T120000Z"
    assert rows[102]["status"] == FC.Status.DEGRADED.value
    assert rows[102]["refusal_reason"] == "games feed stale"
    for game in rows.values():
        assert game["snapshot_id"] == "20261014T120000Z"
        assert game["source_digest"] == "digest-abc"
    assert feed["week"]["by_status"]["frozen"] == 1
    assert feed["week"]["by_status"]["degraded"] == 1
    assert feed["health"]["status"] == DH.STATUS_LIMITED


def test_a_corrupt_health_file_fails_the_hub_build(tmp_path):
    repo = _mini_repo(tmp_path)
    FC.write_artifact(_contract_rows(),
                      str(repo / "data" / f"predictions_{SEASON}_wk08.csv"))
    (repo / "data" / DH.PUBLIC_STATUS_FILE).write_text("{broken")
    out = repo / "data" / "site" / "api" / "hub.json"
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "build_hub_feed.py"),
         "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert not out.exists()


# --------------------------------------------------------------------------- #
# 7. The audit trail is reproducible off this machine
# --------------------------------------------------------------------------- #
def test_the_snapshot_store_is_tracked_not_machine_local():
    ignore = (ROOT / ".gitignore").read_text()
    lines = [l.strip() for l in ignore.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert "data/snapshots/" not in lines, (
        "the pre-kickoff record must not be machine-local: a frozen row whose "
        "proof exists on one Mac cannot be audited by anyone")
    assert "data/snapshots/**/explainer.html" in lines, (
        "the rendered explainer is a presentation of predictions.csv and is "
        "excluded on purpose")
    assert "config.local.json" in lines, "credentials stay ignored"


# --------------------------------------------------------------------------- #
# 8. The manual publish path cannot silently push unrelated local work
# --------------------------------------------------------------------------- #
def _publish_sandbox(tmp_path):
    """A throwaway local origin + working repo. Pushing here is real but
    harmless -- nothing leaves tmp_path, and nothing is the real repo."""
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(remote)],
                   check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ}

    def _run(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True,
                              capture_output=True, text=True, env=env)

    _run("init")
    _run("symbolic-ref", "HEAD", "refs/heads/master")
    _run("config", "user.email", "t@example.com")
    _run("config", "user.name", "Test")
    _run("remote", "add", "origin", str(remote))
    (repo / "README.md").write_text("x")
    _run("add", "README.md")
    _run("commit", "-m", "init")
    _run("push", "-u", "origin", "master")

    (repo / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "publish_site.sh",
               repo / "scripts" / "publish_site.sh")
    os.chmod(repo / "scripts" / "publish_site.sh", 0o755)
    (repo / "data" / "site" / "api").mkdir(parents=True)
    (repo / "data" / "site" / "index.html").write_text("v1")
    _run("add", "data/site")
    _run("commit", "-m", "site v1")
    _run("push", "origin", "master")
    return repo, remote, _run


def _remote_master_sha(remote):
    out = subprocess.run(["git", "rev-parse", "master"], cwd=remote,
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def test_publish_refuses_under_yes_when_unrelated_commits_are_unpushed(
        tmp_path):
    """`--yes` must not be a way to silently push local work the operator
    never looked at. This is the exact defect the script's own header names
    (`git cannot push one commit without its ancestors`) -- disclosed as a
    warning but, before this fix, never actually gated.
    """
    repo, remote, _run = _publish_sandbox(tmp_path)
    before = _remote_master_sha(remote)

    # An UNRELATED local commit, never pushed -- this is the "unrelated local
    # work" the push must not carry silently.
    (repo / "NOTES.md").write_text("private wip")
    _run("add", "NOTES.md")
    _run("commit", "-m", "unrelated local work")

    # A real reason to publish: the site changed.
    (repo / "data" / "site" / "index.html").write_text("v2")

    proc = subprocess.run(
        ["bash", "scripts/publish_site.sh", "--yes"], cwd=repo,
        capture_output=True, text=True, input="", timeout=30)

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "REFUSED" in proc.stdout + proc.stderr
    assert _remote_master_sha(remote) == before, (
        "nothing may reach origin/master while an unrelated commit would "
        "have ridden along")


def test_publish_yes_still_works_with_nothing_unrelated_ahead(tmp_path):
    """The fix must not make the common case (no ahead commits) need a
    human at the keyboard."""
    repo, remote, _run = _publish_sandbox(tmp_path)
    (repo / "data" / "site" / "index.html").write_text("v2")

    proc = subprocess.run(
        ["bash", "scripts/publish_site.sh", "--yes"], cwd=repo,
        capture_output=True, text=True, input="", timeout=30)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    pushed = subprocess.run(["git", "log", "-1", "--format=%s", "master"],
                            cwd=remote, check=True, capture_output=True,
                            text=True).stdout
    assert "publish site" in pushed.lower()


def test_publish_with_ahead_commits_still_succeeds_on_explicit_confirmation(
        tmp_path):
    """The gate is a forced prompt, not a hard refusal -- an operator who
    reads the warning and types 'publish' can still push, ancestors and
    all."""
    repo, remote, _run = _publish_sandbox(tmp_path)
    (repo / "NOTES.md").write_text("private wip")
    _run("add", "NOTES.md")
    _run("commit", "-m", "unrelated local work")
    (repo / "data" / "site" / "index.html").write_text("v2")

    proc = subprocess.run(
        ["bash", "scripts/publish_site.sh", "--yes"], cwd=repo,
        capture_output=True, text=True, input="publish\n", timeout=30)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    pushed = subprocess.run(["git", "log", "-1", "--format=%s", "master"],
                            cwd=remote, check=True, capture_output=True,
                            text=True).stdout
    assert "publish site" in pushed.lower()


def test_the_docs_describe_the_actual_publish_behaviour():
    runbook = (ROOT / "RUNBOOK_WEEKLY.md").read_text()
    load_me = (ROOT / "ops" / "LOAD_ME.md").read_text()
    for text, name in ((runbook, "RUNBOOK_WEEKLY.md"),
                       (load_me, "ops/LOAD_ME.md")):
        assert "publish_site.sh" in text, f"{name} must name the manual command"
    assert "no git command" in runbook.lower(), \
        "the runbook must say plainly that the runner does not publish"
    assert "exit 3" in runbook and "exit 4" in runbook, \
        "the runbook must document the new exit statuses"
    assert "data/snapshots/` is **tracked**" in load_me or \
        "is **tracked**" in load_me, \
        "ops/LOAD_ME.md must record the reversed storage policy"
