"""The release contract: the tree that SHIPS must import on its own.

A clone, a CI runner, or a fresh checkout receives tracked content only.  A
module that exists solely as an untracked file in someone's working tree is
invisible to `git archive` and to `git clone`: the loop imports fine for the
one person who has that stray file and fails for everyone, and everything,
else.  Checking importability in place cannot see this, because in place is
exactly where the stray file lives.

So the subject of every assertion here is the **candidate tree** -- HEAD plus
the staged index -- materialised by `git write-tree` into a real tree object
and extracted clean.  That is what a commit of the current index would publish.
Two consequences worth stating:

* The gate is checkable BEFORE the commit exists.  `write-tree` creates no
  commit, moves no ref and does not touch the working tree, so this runs
  safely inside a dirty repository that has to stay dirty.
* Once the candidate set is committed, the candidate tree IS HEAD's tree, so
  these same assertions keep covering the released commit unchanged.

Concretely, this is the test that fails when `nflvalue/cfb_calendar.py` --
imported at module scope by `scripts/weekly_loop.py` -- is present in the
working tree but absent from version control.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import release_gate as RG  # noqa: E402

pytestmark = pytest.mark.skipif(
    not RG.is_git_checkout(ROOT),
    reason="release-integrity gates need a git checkout (no .git here)")



def test_candidate_tree_tracks_every_required_release_file():
    tracked = RG.tree_paths(ROOT)
    missing = sorted(p for p in RG.REQUIRED_TRACKED_PATHS if p not in tracked)
    assert not missing, (
        "these files are in the working tree but not in the candidate change "
        "set, so a clone would not get them:\n  "
        + "\n  ".join(missing)
        + "\n\nfix: git add " + " ".join(missing))


def test_cfb_calendar_is_tracked_not_merely_present():
    """Named regression for the blocker this module exists to catch."""
    tracked = RG.tree_paths(ROOT)
    assert "nflvalue/cfb_calendar.py" in tracked, (
        "scripts/weekly_loop.py imports nflvalue.cfb_calendar at module "
        "scope. The file is untracked, so `git archive` omits it and the "
        "shipped loop raises ImportError on a clean checkout.")


def test_weekly_loop_imports_from_a_clean_candidate_tree():
    with RG.extracted_tree(ROOT) as tree:
        probe = RG.import_probe(tree, RG.SUPPORTED_ENTRY_POINT)
    assert probe.returncode == 0, (
        f"{RG.SUPPORTED_ENTRY_POINT} failed to import from a clean extraction "
        f"of the candidate tree:\n{probe.stderr}\n\n"
        "This usually means an import target (e.g. nflvalue.cfb_calendar) "
        "exists only as an untracked file in the working tree.")


def test_extraction_contains_tracked_content_only():
    """The extraction must not quietly inherit the working tree's strays.

    If untracked files could leak into the scratch tree, every check above
    would pass for the wrong reason -- which is precisely the failure mode
    being guarded against.
    """
    tracked = RG.tree_paths(ROOT)
    with RG.extracted_tree(ROOT) as tree:
        extracted = {
            str(p.relative_to(tree)) for p in tree.rglob("*") if p.is_file()}
    assert not (extracted - tracked), (
        "untracked files leaked into the extracted candidate tree: "
        + ", ".join(sorted(extracted - tracked)[:20]))
    assert not (tree / ".git").exists()
