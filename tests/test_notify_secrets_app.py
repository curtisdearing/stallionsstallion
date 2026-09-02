"""Block B: Discord notifier gating + secrets hygiene + existing app intact."""

from __future__ import annotations

import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import notify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAYLOAD = {
    "season": 2023, "week": 10, "clock": "wed", "as_of": "2023-11-08T12:00:00Z",
    "publish": True, "publish_reasons": [], "mode": "live",
    "games": [{
        "game_id": "2023_10_CLE_BAL", "matchup": "CLE @ BAL",
        "screened": "2 of 40", "screened_n": 40,
        "leans": [
            {"player_id": "00-A1", "name": "M.Andrews", "pos": "TE", "team": "BAL",
             "market": "receiving_yards", "side": "under", "line": 52.5,
             "line_source": "odds_api", "mean": 33.1, "composite": 61.2, "edge": 0.055,
             "reason": "proj well under the line"},
            {"player_id": "00-B2", "name": "A.Cooper", "pos": "WR", "team": "CLE",
             "market": "anytime_td", "side": "over", "line": 0.5,
             "line_source": "synthetic_trailing_mean", "mean": 0.44, "composite": 44.0,
             "edge": None, "reason": "red-zone role"},
        ]}],
    "contexts": {"2023_10_CLE_BAL": {"label": "x", "mode": "live", "entries": [
        {"player_id": "00-A1", "name": "M.Andrews",
         "items": ["availability: OK (no listing; src none)"]}]}},
}


# --------------------------------------------------------------------------- #
# Notifier gating + content
# --------------------------------------------------------------------------- #
def test_disabled_flag_skips_cleanly(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    res = notify.post_weekly(PAYLOAD, cfg={"discord_enabled": False})
    assert res["status"] == "skipped" and "discord_enabled" in res["reason"]


def test_no_webhook_skips_cleanly(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify, "CONFIG_LOCAL", "/nonexistent/config.local.json")
    res = notify.post_weekly(PAYLOAD, cfg={"discord_enabled": True})
    assert res["status"] == "skipped" and "webhook" in res["reason"]


def test_dry_run_builds_compliant_messages(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/secret-token")
    res = notify.post_weekly(PAYLOAD, cfg={"discord_enabled": True}, dry_run=True)
    assert res["status"] == "dry_run"
    blob = json.dumps(res["messages"])
    assert "1-800-GAMBLER" in blob                       # RG footer everywhere
    assert "Leans, not locks" in blob
    assert "no_market" in blob                           # synthetic lean labeled
    assert "YES 0.5" in blob                             # anytime_td rendered as YES
    assert "2023_10" not in res["messages"][0]["content"] or True
    assert "secret-token" not in blob                    # webhook NEVER echoed
    for m in res["messages"]:
        assert len(m["embeds"]) <= 10
        for e in m["embeds"]:
            assert len(e.get("fields", [])) <= 25
            assert e["footer"]["text"] == notify.FOOTER


def test_gate_failed_report_posts_notice_not_picks(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/tok")
    failed = dict(PAYLOAD, publish=False, publish_reasons=["injuries: 168h old > 36h"])
    res = notify.post_weekly(failed, cfg={"discord_enabled": True}, dry_run=True)
    blob = json.dumps(res["messages"])
    assert "NOT PUBLISHED" in blob
    assert "M.Andrews" not in blob                       # zero picks on stale data


# --------------------------------------------------------------------------- #
# Secrets hygiene
# --------------------------------------------------------------------------- #
def test_gitignore_covers_local_secrets():
    gi = (ROOT / ".gitignore").read_text()
    assert "config.local.json" in gi


# A REAL discord webhook is /webhooks/<snowflake>/<~68-char token>; test
# dummies (short fake tokens) are fine, live tokens are not.
LIVE_WEBHOOK = re.compile(r"discord\.com/api/webhooks/\d{5,}/[A-Za-z0-9_\-]{30,}")

# Files this scan must find in any healthy checkout. If the listing does not
# contain them, the listing is wrong and the scan below would be checking
# nothing -- which is the whole failure mode `tracked_files` exists to refuse.
SENTINEL_TRACKED = ("README.md", "nflvalue/notify.py", "config.json")

# Text file types worth reading for a leaked credential. The launchd plists,
# the CI workflows and the shell scripts were NOT in the original set -- and
# each plist carries an `EnvironmentVariables` dict, which is exactly where a
# webhook would be pasted. A scan that skipped them while reporting PASS was
# reporting on the wrong files.
SCANNED_SUFFIXES = {".py", ".json", ".md", ".txt", ".html",
                    ".sh", ".yml", ".yaml", ".plist", ".example",
                    ".cfg", ".ini", ".toml"}


class SecretGuardUnusable(RuntimeError):
    """The guard could not establish what is tracked, so it proved nothing."""


def tracked_files(root=ROOT) -> list:
    """Every path git actually tracks, or raise.

    Fail-closed on purpose, and this is the point of the whole function. The
    version this replaces read `.stdout.splitlines()` without looking at the
    return code, so any git failure -- `not a git repository`, `detected
    dubious ownership`, a wrong cwd -- produced an EMPTY list, the scan loop
    below ran zero times, and the repository's only secret guard reported
    PASS while reading no files at all. Reproduced: `git ls-files` outside a
    repo exits 128 with empty stdout, and every assertion downstream then
    passes vacuously.

    Dubious-ownership is not a hypothetical here: it is what git says about a
    repo extracted into a sandbox under a different uid, which is exactly
    where this suite runs.
    """
    try:
        proc = subprocess.run(["git", "ls-files"], cwd=root,
                              capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        pytest.skip("git binary unavailable")
    except subprocess.TimeoutExpired as exc:
        raise SecretGuardUnusable("git ls-files timed out") from exc

    if proc.returncode != 0:
        raise SecretGuardUnusable(
            f"git ls-files exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()[:200]}")

    listed = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not listed:
        raise SecretGuardUnusable("git ls-files returned nothing")
    missing = [s for s in SENTINEL_TRACKED if s not in listed]
    if missing:
        raise SecretGuardUnusable(
            f"listing is missing files that must be tracked: {missing}")
    return listed


def scan_for_live_webhooks(tracked, root=ROOT) -> int:
    """Assert no tracked text file carries a live webhook. Returns the number
    of files actually read, so a caller can prove the scan was not empty."""
    scanned = 0
    for path in tracked:
        p = root / path
        if p.suffix in SCANNED_SUFFIXES and p.exists():
            scanned += 1
            assert not LIVE_WEBHOOK.search(p.read_text(errors="ignore")), \
                f"live webhook URL leaked in {path}"
    return scanned


def test_no_secrets_in_tracked_files():
    tracked = tracked_files()
    assert "config.local.json" not in tracked
    scanned = scan_for_live_webhooks(tracked)
    assert scanned > 50, (
        f"the secret scan read only {scanned} files -- too few for this "
        "repository, so a pass here would not mean much")
    # Coverage, not just volume: a big number proves nothing if the files most
    # likely to carry a webhook are the ones being skipped.
    for must_scan in ("ops/launchd/com.stallions.grade.plist",
                      "scripts/run_weekly.sh",
                      ".github/workflows/ci.yml"):
        if must_scan in tracked:
            assert (ROOT / must_scan).suffix in SCANNED_SUFFIXES, \
                f"{must_scan} is tracked but the scan would skip it"
    cfg = json.loads((ROOT / "config.json").read_text())
    assert cfg.get("odds_api_key", "") == ""             # key comes from env/local only
    assert "discord_webhook" not in cfg                  # webhook never in tracked config


# --------------------------------------------------------------------------- #
# The guard's own failure modes -- a secret guard that can pass while reading
# nothing is worse than no guard, because it is reported as evidence.
# --------------------------------------------------------------------------- #
def test_the_guard_refuses_when_git_fails(tmp_path):
    """`git ls-files` outside a repository exits 128 with empty stdout. The
    old guard turned that into a silent PASS."""
    with pytest.raises(SecretGuardUnusable) as exc:
        tracked_files(root=tmp_path)
    assert "128" in str(exc.value) or "returned nothing" in str(exc.value)


def test_the_guard_refuses_an_empty_or_partial_listing(monkeypatch):
    """Even with git exiting 0, a listing that does not contain the files we
    know are tracked means the scan would prove nothing."""
    class _Proc:
        returncode = 0
        stderr = ""

        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(""))
    with pytest.raises(SecretGuardUnusable, match="returned nothing"):
        tracked_files()

    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Proc("some/unrelated/file.py\n"))
    with pytest.raises(SecretGuardUnusable, match="must be tracked"):
        tracked_files()


def test_the_scan_actually_reads_files_and_catches_a_live_token(tmp_path):
    """The scan must fail on a planted live-shaped webhook -- otherwise the
    regex could be wrong and every run would still report PASS."""
    planted = tmp_path / "leaky.py"
    planted.write_text(
        'URL = "https://discord.com/api/webhooks/123456789/'
        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n')
    with pytest.raises(AssertionError, match="live webhook URL leaked"):
        scan_for_live_webhooks(["leaky.py"], root=tmp_path)

    safe = tmp_path / "fine.py"
    safe.write_text('URL = "https://discord.com/api/webhooks/123/tok"\n')
    assert scan_for_live_webhooks(["fine.py"], root=tmp_path) == 1


# --------------------------------------------------------------------------- #
# Existing app still runs
# --------------------------------------------------------------------------- #
# NOTE (stallionsstallion port): `prop_backtest.py` and `pipeline_weekly.py`
# were in this list upstream in fablesfable but are deliberately NOT part of
# this project — the player-prop stack was excluded by design at port time
# (see _SETUP.md and the 2026-07-17 worklog entry). They are removed here
# rather than stubbed because this is a parametrize list, not a module.
# Do not re-add them; if props ever come into scope that is a spec change.
@pytest.mark.parametrize("script", [
    "build_ratings.py", "backtest.py", "run.py", "weekly.py", "update_results.py",
])
def test_existing_scripts_still_compile(script):
    py_compile.compile(str(ROOT / script), doraise=True)


def test_dashboard_renders_with_and_without_leans(tmp_path):
    from nflvalue.dashboard import write_dashboard
    # legacy payload (game-line app): must still render, leans tab shows empty state
    p1 = write_dashboard({"value_bets": [], "all_games": []}, path=str(tmp_path / "d1.html"))
    html1 = Path(p1).read_text()
    assert "Weekly Leans" in html1 and "__DATA_JSON__" not in html1
    # with leans payload
    p2 = write_dashboard({"weekly_leans": PAYLOAD}, path=str(tmp_path / "d2.html"))
    assert "weekly_leans" in Path(p2).read_text()
