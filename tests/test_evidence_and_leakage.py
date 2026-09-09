"""Benchmark evidence: what may back a claim, and what the future may touch.

Two separate failure modes are guarded here.

**Leakage.** The pre-correction walk-forward fitted `PaceModel` on the full
pooled multi-season frame and OT parameters on the whole 2021-2025 window,
both in-window relative to the seasons being graded. A model graded on data it
could not have had is not measured, it is flattered. The corrected fitting
filters every input through one predicate, `_strictly_before`, and these tests
poison the future to prove nothing upstream of it can move.

**Provenance.** A benchmark file on disk is not evidence about the current
model; it is evidence about whatever code produced it. `data/backtest_c.csv`
kept feeding the site, the explainer, the calibration report and the factor
battery after the method it measured was replaced. The evidence manifest is
now the only way to choose one, archival records can never be chosen, and
every read re-hashes the artifact, its producing code and its inputs.
"""

import ast
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import atomic  # noqa: E402
from nflvalue import cfb_ot, cfb_sim  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402
from nflvalue import ratings as R  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "cfbd"
HAS_CACHE = (CACHE / "drives_2023.parquet").exists()
cache_only = pytest.mark.skipif(
    not HAS_CACHE, reason="needs the cached parquet drives/games")


# --------------------------------------------------------------------------- #
# Synthetic world: small, deterministic, and rich enough to walk
# --------------------------------------------------------------------------- #
TEAMS = [f"T{i:02d}" for i in range(16)]


def _synthetic(seasons=(2020, 2021), weeks=12, poison_from=None):
    """Games + drives for a toy league.

    `poison_from=(season, week)` corrupts everything at or after that point:
    scores blown up, drive outcomes flipped, pace doubled. Anything fitted
    strictly before the cutoff must be identical with and without it.
    """
    rng = np.random.default_rng(11)
    strength = {t: float(rng.normal(0, 7)) for t in TEAMS}
    games, drives, gid = [], [], 1000
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(TEAMS)
            rng.shuffle(order)
            for i in range(0, len(order), 2):
                home, away = order[i], order[i + 1]
                poisoned = (poison_from is not None
                            and (season, week) >= tuple(poison_from))
                margin = strength[home] - strength[away] + 2.5
                margin += float(rng.normal(0, 10))
                hp = 24 + margin / 2
                ap = 24 - margin / 2
                if poisoned:
                    hp, ap = 900.0, 0.0
                games.append({
                    "id": gid, "season": season, "week": week,
                    "start_date": f"{season}-09-01T16:00:00Z",
                    "neutral_site": False, "home_team": home,
                    "away_team": away,
                    "home_points": round(hp, 1), "away_points": round(ap, 1),
                    "home_conference": "C", "away_conference": "C",
                    "home_fcs": False, "away_fcs": False})
                # The drive schema matches the cached CFBD one: classify_drives
                # reads end-minus-start scores, not the result strings.
                n_drives = 24 if not poisoned else 48
                off_score = def_score = 0
                for d in range(n_drives):
                    off, dfn = (home, away) if d % 2 == 0 else (away, home)
                    scored = (d % 3 == 0) != poisoned
                    points = 7 if scored else 0
                    drives.append({
                        "game_id": gid, "season": season, "week": week,
                        "drive_number": d + 1,
                        "offense": off, "defense": dfn,
                        "is_home_offense": off == home,
                        "offense_conference": "C", "defense_conference": "C",
                        "start_period": 1 + (d % 4),
                        "drive_result": "TD" if scored else "PUNT",
                        "scoring": bool(scored),
                        "start_yards_to_goal": 75, "end_yards_to_goal": 0,
                        "plays": 6, "yards": 30, "garbage_time": False,
                        "start_offense_score": off_score,
                        "start_defense_score": def_score,
                        "end_offense_score": off_score + points,
                        "end_defense_score": def_score})
                    off_score, def_score = def_score, off_score + points
                gid += 1
    g = pd.DataFrame(games)
    g["margin"] = g["home_points"] - g["away_points"]
    g["home_fbs"] = True
    g["away_fbs"] = True
    g["home_conf"] = "C"
    g["away_conf"] = "C"
    return g, pd.DataFrame(drives)


# --------------------------------------------------------------------------- #
# 1. The future cannot reach backwards
# --------------------------------------------------------------------------- #
def test_strictly_before_never_returns_the_present_or_future():
    """The one predicate every pace/OT/rate input is filtered through."""
    frame = pd.DataFrame([{"season": s, "week": w}
                          for s in (2020, 2021, 2022) for w in range(1, 16)])
    kept = cfb_sim._strictly_before(frame, 2021, 8)
    assert not kept.empty
    assert ((kept["season"] < 2021)
            | ((kept["season"] == 2021) & (kept["week"] < 8))).all()
    assert not ((kept["season"] == 2021) & (kept["week"] >= 8)).any()
    assert not (kept["season"] > 2021).any()


@cache_only
def test_poisoned_future_cannot_move_an_earlier_pace_or_ot_fit():
    """Real cached data, real fitters, one cutoff.

    Pace and OT are the two the pre-correction code fitted over the whole
    window; this is the direct proof that they no longer can be.
    """
    games = R.game_frame([2019, 2020, 2021, 2022, 2023])
    drives = pd.read_parquet(CACHE / "drives_2023.parquet")
    for season in (2019, 2020, 2021, 2022):
        drives = pd.concat(
            [pd.read_parquet(CACHE / f"drives_{season}.parquet"), drives],
            ignore_index=True)
    regulation, _ = cfb_sim.classify_drives(drives)
    regulation = regulation[regulation["start_period"] <= 4]

    season, week = 2023, 10
    clean_pace = cfb_sim.PaceModel.fit(
        cfb_sim._strictly_before(regulation, season, week))
    clean_ot = cfb_ot.fit_ot_params(
        cfb_sim._strictly_before(drives, season, week))

    # Poison everything from the cutoff onward: duplicate it 50x with mangled
    # outcomes. If any fit reaches forward, these numbers move.
    future = drives[(drives["season"] > season)
                    | ((drives["season"] == season)
                       & (drives["week"] >= week))].copy()
    assert len(future), "the poison must actually have something to poison"
    future = pd.concat([future] * 50, ignore_index=True)
    future["scoring"] = True
    if "drive_result" in future.columns:
        future["drive_result"] = "TD"
    poisoned = pd.concat([drives, future], ignore_index=True)
    poisoned_regulation, _ = cfb_sim.classify_drives(poisoned)
    poisoned_regulation = poisoned_regulation[
        poisoned_regulation["start_period"] <= 4]

    dirty_pace = cfb_sim.PaceModel.fit(
        cfb_sim._strictly_before(poisoned_regulation, season, week))
    dirty_ot = cfb_ot.fit_ot_params(
        cfb_sim._strictly_before(poisoned, season, week))

    assert clean_ot.as_dict() == dirty_ot.as_dict(), (
        "an OT parameter moved when only FUTURE drives changed")
    clean_pairs = np.asarray(clean_pace.pairs)
    dirty_pairs = np.asarray(dirty_pace.pairs)
    assert clean_pairs.shape == dirty_pairs.shape, (
        "the pace model saw a different amount of history when only FUTURE "
        "drives changed")
    assert np.array_equal(clean_pairs, dirty_pairs), (
        "a pace parameter moved when only FUTURE drives changed")
    del games


#: A fixed OT object for the synthetic tests. Supplying `ot=` is the
#: documented legacy/test mode ("the caller is asserting it was already fit
#: safely"), which isolates the rates/pace/ratings path here; the OT fit's own
#: walk-forward guarantee is proved against real cached data above.
FIXED_OT = cfb_ot.OTParams(p_td=0.30, p_fg=0.17,
                           td_pts={7: 0.70, 6: 0.26, 8: 0.04}, p_two=0.17,
                           n_possessions=223, n_td_ot1=61, n_two=6)


def _walk(games, drives, **kw):
    params = R.load_params()
    return cfb_sim.walk_forward_backtest(
        games, drives, params, ot=kw.pop("ot"), eval_seasons=[2021],
        burn_in_seasons=[2020], first_week=4, n_sims=200,
        min_train_drives=200, n_bins=2, min_bin_n=10, engine="ridge", **kw)


def test_poisoned_future_cannot_move_an_earlier_evaluation_row():
    """End to end on the synthetic world.

    OT is supplied fixed here (the documented `ot=` mode) so this test isolates
    the rates/pace/ratings path; the OT fit's own walk-forward guarantee is
    proved against real data in the test above.
    """
    ot = FIXED_OT

    clean_games, clean_drives = _synthetic()
    poisoned_games, poisoned_drives = _synthetic(poison_from=(2021, 9))

    clean = _walk(clean_games, clean_drives, ot=ot)
    dirty = _walk(poisoned_games, poisoned_drives, ot=ot)

    early = lambda df: df[df["week"] < 9].sort_values("id").reset_index(drop=True)
    a, b = early(clean), early(dirty)
    assert len(a) and len(a) == len(b)
    for column in ("mc_margin", "mc_margin_sd", "mc_total", "p_home_win",
                   "pred_b", "n_train_drives"):
        pd.testing.assert_series_equal(
            a[column], b[column], check_exact=True,
            obj=f"{column} changed for a week evaluated BEFORE the poison")
    # And the poison really did land somewhere, or this proves nothing.
    late = clean[clean["week"] >= 9]
    if len(late):
        assert not late["mc_margin"].equals(
            dirty[dirty["week"] >= 9]["mc_margin"])


def test_a_games_simulation_does_not_depend_on_what_else_is_in_the_frame():
    """Per-game RNG keyed on (seed, game id) — never on iteration order.

    This is what makes a rerun byte-identical: no row's distribution can
    depend on which other rows are present or in what order they arrive.
    """
    ot = FIXED_OT
    games, drives = _synthetic()
    full = _walk(games, drives, ot=ot)
    assert len(full)

    keep = set(full["id"].tolist()[::2])
    subset_games = games[~games["id"].isin(
        [i for i in full["id"] if i not in keep])]
    subset = _walk(subset_games, drives, ot=ot)

    merged = full.merge(subset, on="id", suffixes=("_full", "_subset"))
    common = merged[merged["id"].isin(keep)]
    if len(common):
        for column in ("mc_margin", "p_home_win"):
            assert np.allclose(common[f"{column}_full"],
                               common[f"{column}_subset"]), (
                f"{column} moved when unrelated games left the frame")


def test_rerunning_the_walk_is_byte_identical():
    ot = FIXED_OT
    games, drives = _synthetic()
    first = _walk(games, drives, ot=ot)
    second = _walk(games.copy(), drives.copy(), ot=ot)
    assert first.to_csv(index=False) == second.to_csv(index=False)


@cache_only
def test_the_shipped_candidate_matches_the_hash_its_run_recorded():
    """The artifact on disk is the one the producing run wrote."""
    records = {r["artifact_id"]: r for r in EV.load_run_records(str(ROOT))}
    assert records, "no run records; produce a candidate first"
    for artifact_id, record in records.items():
        path = ROOT / record["path"]
        assert path.exists(), f"{artifact_id}: {record['path']} missing"
        assert EV.sha256_file(str(path)) == record["content_sha256"], (
            f"{artifact_id} has changed since the run that produced it")


# --------------------------------------------------------------------------- #
# 2. Old artifacts cannot feed current claims
# --------------------------------------------------------------------------- #
@pytest.fixture()
def manifest():
    return EV.Manifest.load(str(ROOT))


def test_the_legacy_artifact_is_present_and_labelled_archival(manifest):
    """Preserved untouched, and unusable as a current claim."""
    record = manifest.records["backtest_c_ridge_20260730"]
    assert record.path == "data/backtest_c.csv"
    assert (ROOT / record.path).exists(), "the legacy artifact must be kept"
    assert record.status == EV.Status.ARCHIVAL.value
    assert not record.selectable
    assert "pre" in record.not_selected_reason.lower()


def test_select_never_returns_an_archival_record(sandbox):
    # Exercise real selection AND full hash verification in a miniature repo;
    # the status rule must not depend on gitignored historical CFBD caches.
    payload=json.loads((sandbox / 'data' / EV.MANIFEST_FILE).read_text())
    archival=dict(payload['artifacts'][0],artifact_id='archival',
                  status=EV.Status.ARCHIVAL.value,
                  not_selected_reason='superseded test fixture')
    payload['artifacts'].append(archival)
    manifest=EV.Manifest(payload,str(sandbox))
    selected=[]
    for role in EV.ROLES:
        try:
            chosen = manifest.select(role)
        except EV.BenchmarkUnavailable:
            continue
        assert chosen.status == EV.Status.SELECTED.value
        assert chosen.status != EV.Status.ARCHIVAL.value
        selected.append(chosen.artifact_id)
    assert selected==['bench']


def test_a_candidate_is_not_a_claim(manifest):
    candidates = [r for r in manifest.for_role(EV.ROLE_WK4PLUS)
                  if r.status == EV.Status.CANDIDATE.value]
    assert candidates, "the corrected run should be recorded as a candidate"
    for record in candidates:
        assert not record.selectable
        assert record.not_selected_reason


def test_no_selected_benchmark_yields_a_notice_not_a_number(manifest):
    frame, record, notice = EV.benchmark_or_notice(str(ROOT), EV.ROLE_WK4PLUS)
    if record is not None:            # a human has since selected one
        pytest.skip("a benchmark is selected; the notice path is not active")
    assert frame is None
    assert "awaiting revalidation" in notice.lower()
    for number in ("mae", "brier"):
        assert number not in notice.lower().split("—")[0]


@pytest.mark.parametrize("script", [
    "scripts/build_site.py", "scripts/render_explainer.py",
    "scripts/weekly_loop.py", "scripts/calibration_o.py",
    "scripts/factor_battery.py", "scripts/factor_battery_context.py",
    "scripts/factor_qb_pregame.py",
])
def test_no_public_consumer_reads_a_benchmark_file_directly(script):
    """Static: the manifest is the only door.

    A consumer that can open `data/backtest_c.csv` itself will eventually do
    so again, whatever the manifest says.
    """
    tree = ast.parse((ROOT / script).read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef,
                                       ast.AsyncFunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)
                  and isinstance(node.body[0].value.value, str)}
    offenders = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
        and node.value.startswith("backtest_c")]
    assert not offenders, (
        f"{script} names a benchmark file in code: {offenders}. Select it "
        "through nflvalue.evidence instead")


def test_the_site_payload_shows_the_notice_when_nothing_is_selected(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_site", ROOT / "scripts" / "build_site.py")
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)
    reg, clv, bt, recaps, record, notice = site.load_inputs()
    if record is not None:
        pytest.skip("a benchmark is selected")
    assert bt is None
    assert notice and "awaiting revalidation" in notice.lower()


def test_the_public_calibration_report_publishes_no_stale_metric():
    report = DATA / "calibration_report.txt"
    if not report.exists():
        pytest.skip("no calibration report generated yet")
    text = report.read_text()
    manifest = EV.Manifest.load(str(ROOT))
    try:
        manifest.select(EV.ROLE_WK4PLUS)
    except EV.BenchmarkUnavailable:
        assert "AWAITING REVALIDATION" in text.upper()
        # The report may EXPLAIN that no Brier score is reported; what it may
        # not do is report one. Look for a number, not the word.
        import re as _re
        quoted = _re.findall(r"Brier\s+\d", text)
        assert not quoted, (
            f"the public report quotes a Brier value {quoted} with no "
            "selected benchmark")
        assert not _re.findall(r"\bMAE\s+\d", text)


# --------------------------------------------------------------------------- #
# 3. The manifest rejects mismatched code / data / schema hashes
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sandbox(tmp_path):
    """A miniature repo with one artifact and a manifest describing it."""
    (tmp_path / "data").mkdir()
    (tmp_path / "nflvalue").mkdir()
    artifact = tmp_path / "data" / "bench.csv"
    artifact.write_text("season,week,id,mc_margin,margin\n2023,7,1,3.0,7.0\n")
    code = tmp_path / "nflvalue" / "producer.py"
    code.write_text("# producing code\n")
    inputs = tmp_path / "data" / "source.csv"
    inputs.write_text("a,b\n1,2\n")

    record = {
        "artifact_id": "bench", "role": EV.ROLE_WK4PLUS,
        "status": EV.Status.SELECTED.value, "path": "data/bench.csv",
        "method": "test_v1",
        "code_hashes": {"nflvalue/producer.py": EV.sha256_file(str(code))},
        "input_hashes": {"data/source.csv": EV.sha256_file(str(inputs))},
        "schema": EV.csv_header(str(artifact)),
        "content_sha256": EV.sha256_file(str(artifact)),
    }
    atomic.write_json(str(tmp_path / "data" / EV.MANIFEST_FILE), {
        "schema_version": EV.SCHEMA_VERSION, "artifacts": [record]})
    return tmp_path


def test_a_matching_manifest_selects_cleanly(sandbox):
    assert EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS).artifact_id \
        == "bench"


def test_the_manifest_rejects_a_content_mismatch(sandbox):
    (sandbox / "data" / "bench.csv").write_text(
        "season,week,id,mc_margin,margin\n2023,7,1,99.0,7.0\n")
    with pytest.raises(EV.EvidenceError, match="content sha256 mismatch"):
        EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS)


def test_the_manifest_rejects_a_code_mismatch(sandbox):
    """The artifact is untouched; the code that made it is not."""
    (sandbox / "nflvalue" / "producer.py").write_text("# changed\n")
    with pytest.raises(EV.EvidenceError, match="code hash mismatch"):
        EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS)


def test_the_manifest_rejects_an_input_mismatch(sandbox):
    (sandbox / "data" / "source.csv").write_text("a,b\n9,9\n")
    with pytest.raises(EV.EvidenceError, match="input hash mismatch"):
        EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS)


def test_the_manifest_rejects_a_schema_mismatch(sandbox):
    """Same rows, renamed column: a reader would silently mean something else."""
    path = sandbox / "data" / "bench.csv"
    path.write_text("season,week,id,pred_margin,margin\n2023,7,1,3.0,7.0\n")
    manifest = json.loads((sandbox / "data" / EV.MANIFEST_FILE).read_text())
    manifest["artifacts"][0]["content_sha256"] = EV.sha256_file(str(path))
    atomic.write_json(str(sandbox / "data" / EV.MANIFEST_FILE), manifest)
    with pytest.raises(EV.EvidenceError, match="schema mismatch"):
        EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS)


def test_a_missing_declared_input_fails_closed(sandbox):
    os.remove(sandbox / "data" / "source.csv")
    with pytest.raises(EV.EvidenceError, match="provenance cannot be checked"):
        EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS)


def test_a_missing_manifest_fails_closed(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(EV.EvidenceError, match="no evidence manifest"):
        EV.Manifest.load(str(tmp_path))


def test_two_selected_artifacts_for_one_role_is_refused(sandbox):
    manifest = json.loads((sandbox / "data" / EV.MANIFEST_FILE).read_text())
    twin = dict(manifest["artifacts"][0])
    twin["artifact_id"] = "bench_twin"
    manifest["artifacts"].append(twin)
    atomic.write_json(str(sandbox / "data" / EV.MANIFEST_FILE), manifest)
    with pytest.raises(EV.EvidenceError, match="selected artifacts"):
        EV.Manifest.load(str(sandbox))


def test_an_archival_artifact_cannot_be_selected_by_the_builder():
    """The builder refuses, rather than trusting the operator's typing."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_evidence_manifest.py"),
         "--select", "backtest_c_ridge_20260730",
         "--acknowledge", "attempting the forbidden thing"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "archival" in (proc.stdout + proc.stderr)


def test_selecting_requires_an_acknowledgement():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_evidence_manifest.py"),
         "--select", "backtest_c_corrected_wf_kalman_20260830"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "acknowledge" in (proc.stdout + proc.stderr).lower()


def test_the_run_record_captures_everything_the_manifest_needs():
    """Code hash, input hashes, engine policy, window, exclusions, seed…"""
    records = EV.load_run_records(str(ROOT))
    if not records:
        pytest.skip("no candidate run records present")
    for record in records:
        for field in ("artifact_id", "path", "method", "engine_policy",
                      "seasons", "weeks", "seed", "exclusions", "code_hashes",
                      "input_hashes", "schema", "content_sha256",
                      "generated_at_utc"):
            assert field in record, f"{record.get('artifact_id')}: no {field}"
        assert record["code_hashes"], "code provenance is the point"
        assert record["input_hashes"], "input provenance is the point"
        assert record["engine_policy"]["policy_changed_by_this_run"] is False


def test_the_corrected_run_did_not_change_engine_policy():
    """A run may not promote its own engine, however good it looks."""
    from nflvalue import engine as EG
    config = json.loads((ROOT / "config.json").read_text())
    assert config["margin_engine"] == "kalman", (
        "the registered wk4+ engine must still be kalman")
    assert EG.EARLY_SEASON_ENGINE == "ridge"
    assert EG.EARLY_SEASON_WEEKS == (0, 1, 2, 3)


def test_a_hand_edited_status_cannot_promote_an_unverifiable_artifact(sandbox):
    """`status` lives in a JSON file, so it is only as strong as the last edit.

    The builder refuses to select an archival record. This is the second lock:
    an artifact with no recorded code provenance is unselectable whatever the
    file says, which is exactly the state every pre-correction artifact is in.
    """
    payload = json.loads((sandbox / "data" / EV.MANIFEST_FILE).read_text())
    payload["artifacts"][0]["code_hashes"] = {}
    payload["artifacts"][0]["input_hashes"] = {}
    atomic.write_json(str(sandbox / "data" / EV.MANIFEST_FILE), payload)
    with pytest.raises(EV.EvidenceError, match="no code provenance"):
        EV.Manifest.load(str(sandbox)).select(EV.ROLE_WK4PLUS)


def test_the_shipped_archival_records_are_structurally_unselectable(manifest):
    for record in manifest.records.values():
        if record.status == EV.Status.ARCHIVAL.value:
            assert not record.code_hashes, (
                f"{record.artifact_id}: an archival record with code "
                "provenance could be promoted by a hand-edit")
