"""Release-integrity gate: validate the CANDIDATE tree, not the dirty worktree.

The tree that actually ships to a clone, a CI runner, or a fresh checkout is
never the working tree.  It is HEAD plus whatever is staged in the git index --
the *candidate change set*.  A module that exists only as an untracked file
imports fine for the one person who has it sitting in their tree and fails for
everyone and everything else.  That failure mode is invisible to any check that
runs in place, which is why every check here runs against a clean extraction of
the candidate tree instead.

``git write-tree`` is what makes this checkable *before* a commit exists: it
records the index as a real tree object without creating a commit, moving a
branch, or touching the working tree.  ``git archive`` then materialises that
tree into a scratch directory containing exactly what a clone would contain.

When the index and HEAD agree -- i.e. once the candidate set has been committed
-- the candidate tree *is* HEAD's tree, so the same checks keep covering the
released commit with no change in meaning.

Stdlib only, on purpose: this module is imported by the test suite and by
``scripts/release_check.py``, and a gate that needs pandas to decide whether the
release is sound is a gate that can go dark exactly when it is needed.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The single supported operational entry point.
SUPPORTED_ENTRY_POINT = "scripts/weekly_loop.py"

#: Paths the candidate tree must track for the release to stand on its own.
#: Each one is here because its absence has produced, or would produce, a tree
#: that imports in the author's worktree and nowhere else.
REQUIRED_TRACKED_PATHS: Tuple[str, ...] = (
    "README.md",
    "scripts/weekly_loop.py",
    "scripts/release_check.py",
    "nflvalue/cfb_calendar.py",
    "nflvalue/legacy_guard.py",
    "nflvalue/release_gate.py",
    "nflvalue/forecast_contract.py",
    "nflvalue/producers.py",
    # Data, not code, and therefore invisible to the import probe: the
    # production path refuses to price any week without a registered
    # out-of-sample basis covering it, so an untracked manifest ships a tree
    # that imports cleanly and then declines to forecast anything. The CSV is
    # the evidence the manifest summarises; without it the basis is a number
    # nobody can audit.
    "data/forecast_sd_basis.json",
    "data/backtest_early.csv",
    "tests/test_cfb_calendar.py",
    "tests/test_clean_checkout_importability.py",
    "tests/test_legacy_entry_points_fail_closed.py",
)

#: Legacy entry points that must refuse to run.  Kept as historical source;
#: made inert by nflvalue.legacy_guard.
LEGACY_ENTRY_POINTS: Tuple[str, ...] = (
    "run.py",
    "weekly.py",
    "scripts/auto_weekly.py",
)

#: (entry point, argv) pairs probed by the gate.  The argv variants matter:
#: a guard placed after argument parsing would let ``--live`` reach live-pricing
#: code before refusing, so each documented invocation is probed separately.
LEGACY_INVOCATIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("run.py", ()),
    ("run.py", ("--live",)),
    ("run.py", ("--demo",)),
    ("weekly.py", ()),
    ("weekly.py", ("--live",)),
    ("scripts/auto_weekly.py", ()),
    ("scripts/auto_weekly.py", ("--job", "wed")),
    ("scripts/auto_weekly.py", ("--job", "t90")),
    ("scripts/auto_weekly.py", ("--job", "tuesday")),
    ("scripts/auto_weekly.py", ("--job", "deploy")),
)

_IMPORT_PROBE = textwrap.dedent("""
    import importlib.util, os, sys
    sys.path.insert(0, os.path.abspath('.'))
    spec = importlib.util.spec_from_file_location('weekly_loop', %r)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('imported', mod.__name__)
""")


class GitError(RuntimeError):
    """A git plumbing call the gate depends on did not succeed."""


class Check(NamedTuple):
    """One release invariant and how it came out."""

    name: str
    ok: bool
    detail: str


# --------------------------------------------------------------------------- #
# git plumbing
# --------------------------------------------------------------------------- #
def _git(root: Path, *args: str, timeout: int = 120) -> str:
    proc = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def is_git_checkout(root: Path = REPO_ROOT) -> bool:
    """Is ``root`` a real git checkout?

    Source tarballs and the archive extractions this module itself produces
    carry no ``.git``.  The gate is meaningless there -- there is no index and
    no HEAD to compare against -- so callers skip rather than fail, which keeps
    a tarball user from seeing a red suite for a check that does not apply.
    """
    try:
        _git(root, "rev-parse", "--git-dir", timeout=30)
    except (GitError, OSError, subprocess.SubprocessError):
        return False
    return True


def candidate_tree_id(root: Path = REPO_ROOT) -> str:
    """SHA of the tree that WOULD ship if the index were committed right now.

    ``git write-tree`` writes tree objects only.  It creates no commit, moves
    no ref, and does not touch the working tree, so calling it is safe inside a
    dirty repository that must stay dirty.
    """
    return _git(root, "write-tree").strip()


def head_tree_id(root: Path = REPO_ROOT) -> str:
    """SHA of HEAD's tree -- what a clone gets *today*, before any commit."""
    return _git(root, "rev-parse", "HEAD^{tree}").strip()


def tree_paths(root: Path = REPO_ROOT, tree_id: str = "") -> frozenset:
    """Every file path recorded in ``tree_id`` (default: the candidate tree)."""
    tree_id = tree_id or candidate_tree_id(root)
    out = _git(root, "ls-tree", "-r", "--name-only", tree_id)
    return frozenset(line for line in out.splitlines() if line)


@contextlib.contextmanager
def extracted_tree(root: Path = REPO_ROOT, tree_id: str = "") -> Iterator[Path]:
    """Materialise ``tree_id`` into a scratch dir; yield its path.

    What lands in the directory is exactly what ``git clone`` + checkout of
    that tree would produce: tracked content only, no untracked strays, no
    ``.git``.  Anything that works in here works for everyone.
    """
    tree_id = tree_id or candidate_tree_id(root)
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / "candidate.tar"
        _git(root, "archive", "--format=tar", "-o", str(archive), tree_id)
        target = tmpdir / "tree"
        target.mkdir()
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(target, filter="data")
            except TypeError:  # Python < 3.11.4 has no extraction filter
                tar.extractall(target)
        yield target


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _clean_env() -> Dict[str, str]:
    """Environment with nothing that could smuggle the worktree back in."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONSTARTUP", "GIT_DIR", "GIT_WORK_TREE",
                "GIT_INDEX_FILE"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def import_probe(tree: Path, module_path: str = SUPPORTED_ENTRY_POINT,
                 python: str = sys.executable,
                 timeout: int = 180) -> subprocess.CompletedProcess:
    """Import ``module_path`` from inside ``tree`` in a fresh interpreter."""
    return subprocess.run(
        [python, "-c", _IMPORT_PROBE % module_path], cwd=str(tree),
        capture_output=True, text=True, timeout=timeout, env=_clean_env())


def legacy_probe(tree: Path, entry_point: str, argv: Sequence[str] = (),
                 python: str = sys.executable,
                 timeout: int = 180) -> subprocess.CompletedProcess:
    """Run a legacy entry point inside ``tree`` and capture what it did.

    Runs against the extracted candidate tree rather than the repository so a
    guard regression can never write an artifact into the user's checkout: if
    the refusal ever stops working, the damage is confined to a scratch dir
    that is deleted on the way out, and the test still fails.
    """
    return subprocess.run(
        [python, entry_point, *argv], cwd=str(tree), capture_output=True,
        text=True, timeout=timeout, env=_clean_env())


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def run_checks(root: Path = REPO_ROOT,
               python: str = sys.executable) -> List[Check]:
    """Every release invariant, evaluated against the candidate tree."""
    from . import legacy_guard  # local: keeps module import side-effect free

    checks: List[Check] = []
    tree_id = candidate_tree_id(root)
    checks.append(Check("candidate tree materialises", True,
                        f"git write-tree -> {tree_id}"))

    paths = tree_paths(root, tree_id)
    missing = [p for p in REQUIRED_TRACKED_PATHS if p not in paths]
    checks.append(Check(
        "required files tracked in candidate tree", not missing,
        "all present" if not missing else
        "untracked/unstaged, so a clone would not get them: "
        + ", ".join(missing) + "  (fix: git add " + " ".join(missing) + ")"))

    with extracted_tree(root, tree_id) as tree:
        probe = import_probe(tree, SUPPORTED_ENTRY_POINT, python=python)
        checks.append(Check(
            f"clean candidate tree imports {SUPPORTED_ENTRY_POINT}",
            probe.returncode == 0,
            "imported" if probe.returncode == 0 else
            (probe.stderr.strip() or probe.stdout.strip())[-1200:]))

        readme = tree / "README.md"
        if not readme.exists():
            checks.append(Check("README names one supported product", False,
                                "README.md is not in the candidate tree"))
        else:
            text = readme.read_text(errors="ignore")
            gaps = []
            if SUPPORTED_ENTRY_POINT not in text:
                gaps.append(f"does not name {SUPPORTED_ENTRY_POINT}")
            if "paper-only" not in text:
                gaps.append("does not state the paper-only scope")
            for legacy in LEGACY_ENTRY_POINTS:
                if legacy not in text:
                    gaps.append(f"does not mark {legacy} unsupported")
            checks.append(Check("README names one supported product",
                                not gaps, "ok" if not gaps else "; ".join(gaps)))

        for entry_point, argv in LEGACY_INVOCATIONS:
            label = " ".join([entry_point, *argv])
            if entry_point not in paths:
                checks.append(Check(f"legacy fails closed: {label}", False,
                                    "not in the candidate tree"))
                continue
            proc = legacy_probe(tree, entry_point, argv, python=python)
            problems = []
            if proc.returncode != legacy_guard.EXIT_CODE:
                problems.append(
                    f"exit {proc.returncode}, expected "
                    f"{legacy_guard.EXIT_CODE}")
            if proc.stdout.strip():
                problems.append(f"wrote to stdout: {proc.stdout.strip()[:200]!r}")
            if legacy_guard.REFUSAL_MARKER not in proc.stderr:
                problems.append("no refusal on stderr")
            for artifact in ("dashboard.html", "data/latest.json",
                             "data/weekly.json"):
                if (tree / artifact).exists():
                    problems.append(f"produced {artifact}")
            checks.append(Check(f"legacy fails closed: {label}", not problems,
                                "refused" if not problems
                                else "; ".join(problems)))

    return checks


def head_import_status(root: Path = REPO_ROOT,
                       python: str = sys.executable) -> Check:
    """Advisory: does HEAD *alone* import, i.e. is the candidate committed yet?

    Deliberately not part of :func:`run_checks`.  Before the candidate change
    set is committed, HEAD failing here is the expected state and says only
    "there is a commit still to make", not "the release is broken".
    """
    try:
        tree_id = head_tree_id(root)
    except GitError as exc:
        return Check("HEAD imports (advisory)", False, str(exc))
    with extracted_tree(root, tree_id) as tree:
        probe = import_probe(tree, SUPPORTED_ENTRY_POINT, python=python)
    if probe.returncode == 0:
        return Check("HEAD imports (advisory)", True,
                     "HEAD already stands alone")
    return Check(
        "HEAD imports (advisory)", False,
        "HEAD does not import on its own; the candidate change set closes "
        "this and becomes HEAD when it is committed. "
        + (probe.stderr.strip().splitlines() or [""])[-1][:300])
