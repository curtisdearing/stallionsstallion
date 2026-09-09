"""Obsidian generation: bounded blocks, immutability, and the no-prose-reading rule.

The failure this suite exists to prevent is a generator that eats a paragraph
somebody wrote, or a calculation that reads a number back out of a note it printed
there itself.
"""
from __future__ import annotations

import ast
import datetime as dt
import os

import pytest

from nflvalue.team_evidence import profile as TP
from nflvalue.team_evidence import vault as V
from nflvalue.team_evidence.registry import MeasureRegistry

from tests.test_team_evidence_leakage import (FakeSources, NOW, season_frames,
                                              snapshot)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PKG = os.path.join(ROOT, "nflvalue", "team_evidence")


def a_profile(week=3):
    plays, games = season_frames(weeks=(1, 2))
    return TP.build_profiles(2026, week, data_dir=DATA, plays=plays, games=games,
                             sources=FakeSources(DATA), now=NOW,
                             with_opponent_adjustment=False)["A"]


# ------------------------------------------------------------ bounded blocks
def test_a_new_note_gets_the_block_appended():
    out = V.render_bounded("", "team-season-profile", "numbers here")
    assert "BEGIN GENERATED team-season-profile v1" in out
    assert "END GENERATED team-season-profile v1" in out
    assert "numbers here" in out


def test_only_the_block_is_replaced():
    first = V.render_bounded("# Title\n\nMy own paragraph.\n",
                             "team-season-profile", "old numbers")
    second = V.render_bounded(first, "team-season-profile", "new numbers")
    assert "My own paragraph." in second
    assert "new numbers" in second
    assert "old numbers" not in second
    assert second.startswith("# Title\n\nMy own paragraph.")


def test_text_after_the_block_survives_too():
    text = V.render_bounded("# T\n", "team-season-profile", "old")
    text += "\n## Notes\n\nSomething I wrote afterwards.\n"
    out = V.render_bounded(text, "team-season-profile", "new")
    assert "Something I wrote afterwards." in out
    assert out.index("new") < out.index("Something I wrote afterwards.")


def test_a_start_marker_without_an_end_raises_rather_than_guessing():
    broken = ("# T\n"
              + V.GENERATED_START.format(block_id="team-season-profile", version=1)
              + "\nhalf a block\n")
    with pytest.raises(V.MalformedBlockError):
        V.render_bounded(broken, "team-season-profile", "new")


def test_regeneration_is_idempotent():
    a = V.render_bounded("# T\n", "team-season-profile", "same")
    assert V.render_bounded(a, "team-season-profile", "same") == a


def test_manual_prose_survives_regeneration(tmp_path):
    root = str(tmp_path)
    prof = a_profile()
    path = V.write_team_note(root, prof, artifact_sha256="a" * 64, commit="abc1234",
                             registry=MeasureRegistry.load(DATA))
    text = open(path, encoding="utf-8").read()
    assert V.MANUAL_HEADING in text
    hand = "\nRun defence looked soft on the perimeter against C. Watch the edge.\n"
    open(path, "w", encoding="utf-8").write(text + hand)

    V.write_team_note(root, prof, artifact_sha256="b" * 64, commit="def5678",
                      registry=MeasureRegistry.load(DATA))
    after = open(path, encoding="utf-8").read()
    assert hand.strip() in after
    assert "def5678" in after and "abc1234" not in after


# ------------------------------------------------------------------ identity
def test_the_slug_is_a_filename_and_the_id_is_the_cfbd_string():
    assert V.team_slug("San José State") == "san-jose-state"
    assert V.team_slug("Hawai'i") == "hawai-i"
    assert V.team_id(2026, "San José State") == "2026:San José State"
    assert V.team_id(2026, "Hawai'i") == "2026:Hawai'i"


def test_note_paths_lead_with_the_stable_id():
    p = V.game_note_path("/v", 2026, 2, 401858423, "Rutgers", "Massachusetts")
    assert p.endswith("games/2026/wk02/401858423--massachusetts-at-rutgers.md")
    t = V.team_note_path("/v", 2026, "Ohio State")
    assert t.endswith("teams/2026/ohio-state.md")


def test_the_note_carries_the_artifact_version_and_the_commit(tmp_path):
    path = V.write_team_note(str(tmp_path), a_profile(), artifact_sha256="c" * 64,
                             commit="9f8e7d6", registry=MeasureRegistry.load(DATA))
    text = open(path, encoding="utf-8").read()
    assert "sha256:cccccccccccccccc" in text
    assert "9f8e7d6" in text
    assert "team_id: \"2026:A\"" in text


def test_blocked_measures_are_printed_with_their_reason_not_omitted(tmp_path):
    path = V.write_team_note(str(tmp_path), a_profile(), artifact_sha256="d" * 64,
                             commit="x", registry=MeasureRegistry.load(DATA))
    text = open(path, encoding="utf-8").read()
    assert "seconds_per_play" in text or "Seconds per play" in text
    assert "blocked_unverified_availability" in text
    assert "unavailable_not_approved" in text


def test_the_note_states_the_availability_class(tmp_path):
    path = V.write_team_note(str(tmp_path), a_profile(), artifact_sha256="e" * 64,
                             commit="x", registry=MeasureRegistry.load(DATA))
    text = open(path, encoding="utf-8").read()
    assert "reconstructed_week_granular" in text
    assert "00_contract.md" in text


# ------------------------------------------- immutable pregame, append postgame
def test_a_pregame_record_is_written_once(tmp_path):
    path = str(tmp_path / "g.md")
    V.write_pregame_record(path, 401858423, "Pick: A by 7. Reason: rating gap.")
    first = open(path, encoding="utf-8").read()
    V.write_pregame_record(path, 401858423, "Pick: A by 7. Reason: rating gap.")
    assert open(path, encoding="utf-8").read() == first, "identical is a no-op"


def test_rewriting_a_pregame_record_raises(tmp_path):
    path = str(tmp_path / "g.md")
    V.write_pregame_record(path, 401858423, "Pick: A by 7.")
    with pytest.raises(V.ImmutablePregameError):
        V.write_pregame_record(path, 401858423, "Pick: B by 3. (knew the result)")


def test_postgame_append_never_rewrites_the_pregame_record(tmp_path):
    path = str(tmp_path / "g.md")
    V.write_pregame_record(path, 401858423, "Pick: A by 7. Reason: rating gap.")
    start = V.PREGAME_START.format(game_id=401858423)
    end = V.PREGAME_END.format(game_id=401858423)
    before = open(path, encoding="utf-8").read().partition(start)[2].partition(end)[0]

    for i in range(3):
        V.append_postgame_review(path, 401858423, f"Review {i}: it lost by 4.",
                                 reviewed_at=dt.datetime(2026, 9, 10 + i,
                                                         tzinfo=dt.timezone.utc))
    text = open(path, encoding="utf-8").read()
    after = text.partition(start)[2].partition(end)[0]
    assert after == before, "the original rationale was rewritten"
    assert text.count(V.POSTGAME_HEADING) == 1
    for i in range(3):
        assert f"Review {i}: it lost by 4." in text


def test_a_postgame_review_with_no_pregame_record_is_refused(tmp_path):
    with pytest.raises(V.VaultError):
        V.append_postgame_review(str(tmp_path / "missing.md"), 1, "review")


# -------------------------------------------- machine never reads the prose
def test_no_calculation_module_reads_the_vault():
    """Structural, both directions: the calculation modules may not import the
    vault writer, and no module but the writer may name a vault directory."""
    for name in ("measures.py", "profile.py", "registry.py", "contracts.py",
                 "sources.py", "news.py"):
        src = open(os.path.join(PKG, name), encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "vault" not in node.module, f"{name} imports vault"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "vault" not in alias.name, f"{name} imports vault"
        for token in ('"teams/', "'teams/", '"games/', "'games/", "picks"):
            assert token not in src, f"{name} names a vault path ({token})"


def test_the_vault_writer_does_not_import_the_calculation_modules_for_numbers():
    """`vault.py` may render an Observation; it may not recompute one."""
    src = open(os.path.join(PKG, "vault.py"), encoding="utf-8").read()
    assert "from .measures import" not in src
    assert "import measures" not in src
    assert "team_game_measures" not in src
    assert "build_profiles" not in src


def test_poisoned_note_prose_cannot_change_a_profile(tmp_path):
    """Functional proof. Write the note, replace every number in it with nonsense,
    rebuild from canonical data, assert byte-identical."""
    plays, games = season_frames(weeks=(1, 2))
    kw = dict(data_dir=DATA, plays=plays, games=games, sources=FakeSources(DATA),
              now=NOW, with_opponent_adjustment=False)
    before = TP.build_profiles(2026, 3, **kw)
    root = str(tmp_path)
    reg = MeasureRegistry.load(DATA)
    path = V.write_team_note(root, before["A"], artifact_sha256="f" * 64,
                             commit="c0ffee", registry=reg)

    text = open(path, encoding="utf-8").read()
    import re
    poisoned = re.sub(r"\d+\.\d+", "999.999", text)
    poisoned += "\n\nRush success rate is actually 0.001 and the team has 40 games.\n"
    open(path, "w", encoding="utf-8").write(poisoned)

    after = TP.build_profiles(2026, 3, **kw)
    assert snapshot(after) == snapshot(before)


def test_the_layer_writes_no_note_outside_its_registered_layout(tmp_path):
    root = str(tmp_path)
    V.write_team_note(root, a_profile(), artifact_sha256="0" * 64, commit="x",
                      registry=MeasureRegistry.load(DATA))
    written = [os.path.relpath(os.path.join(dp, f), root)
               for dp, _, fs in os.walk(root) for f in fs]
    assert written == ["teams/2026/a.md"]
    assert not any(p.startswith("picks") for p in written), \
        "picks/ belongs to the existing generators and to Curtis"


def test_vault_root_falls_back_inside_the_repo_when_there_is_no_vault(tmp_path):
    repo = tmp_path / "repo"
    (repo / "nflvalue").mkdir(parents=True)
    assert V.vault_root(str(repo)).endswith(os.path.join("repo", "data",
                                                         "vault_preview"))
    (tmp_path / "picks").mkdir()
    assert V.vault_root(str(repo)) == str(tmp_path)
