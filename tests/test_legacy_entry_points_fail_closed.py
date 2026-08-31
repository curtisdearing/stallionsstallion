"""The legacy NFL entry points must be inert, and provably so.

`run.py`, `weekly.py` and `scripts/auto_weekly.py` are inherited from the NFL
project this repository was ported from.  Their source is deliberately kept --
the worklog and decision records still reference it -- but none of them may
run, because behind each one sits pricing, bankroll, staking, player-prop and
notification code that is unmaintained here and out of scope.

"Cannot produce wagering-oriented output" is asserted in the strongest form
available: **stdout is empty**.  Not "no forbidden words were observed" -- no
output at all.  The refusal goes to stderr, the exit status is a distinctive
non-zero code, and no artifact appears on disk.

Everything runs against a clean extraction of the candidate tree, never the
repository, for two reasons: it proves the guard is in the tree that ships
rather than only in the author's worktree, and a guard regression can then
only scribble inside a scratch directory that is deleted on the way out.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import legacy_guard as LG  # noqa: E402
from nflvalue import release_gate as RG  # noqa: E402

pytestmark = pytest.mark.skipif(
    not RG.is_git_checkout(ROOT),
    reason="release-integrity gates need a git checkout (no .git here)")


#: Output strings the legacy entry points print when they actually run.  Taken
#: verbatim from their source, so a regression that revives any of them trips
#: this list even if the exit status were somehow made to look clean.
WAGERING_OUTPUT_MARKERS = (
    "Top edges",
    "Value bets",
    "value_bets",
    "value props",
    "No +EV plays",
    "Season-to-date",
    "Saved: data/weekly.json",
    "Dashboard:",
    "avg margin error",
)

#: Files the legacy paths write when they complete.  None may exist afterwards.
LEGACY_ARTIFACTS = ("dashboard.html", "data/latest.json", "data/weekly.json")


@pytest.fixture(scope="module")
def candidate_tree():
    """Shared extraction for the read-only checks (source inspection, import)."""
    with RG.extracted_tree(ROOT) as tree:
        yield tree


@pytest.fixture
def pristine_tree():
    """A fresh extraction per executed invocation.

    Execution tests get their own tree so an artifact found afterwards belongs
    unambiguously to the entry point under test -- otherwise one leaking guard
    would fail every later invocation's artifact assertion and point at the
    wrong file.
    """
    with RG.extracted_tree(ROOT) as tree:
        yield tree


def _module_ast(tree: Path, rel: str) -> ast.Module:
    return ast.parse((tree / rel).read_text())


def _first_statement(fn: ast.FunctionDef) -> ast.stmt:
    """The first statement that actually executes, skipping any docstring."""
    body = fn.body
    if ast.get_docstring(fn) is not None:
        body = body[1:]
    return body[0]


def _is_refusal(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "refuse")


@pytest.mark.parametrize(
    "entry_point,argv", RG.LEGACY_INVOCATIONS,
    ids=["_".join([e, *a]).replace("/", ".") for e, a in RG.LEGACY_INVOCATIONS])
def test_legacy_invocation_fails_closed(pristine_tree, entry_point, argv):
    before = {a for a in LEGACY_ARTIFACTS if (pristine_tree / a).exists()}
    proc = RG.legacy_probe(pristine_tree, entry_point, argv)

    assert proc.stdout.strip() == "", (
        f"{entry_point} {' '.join(argv)} wrote to stdout; a refused entry "
        f"point must produce no output at all:\n{proc.stdout[:2000]}")
    assert proc.returncode == LG.EXIT_CODE, (
        f"expected the refusal exit status {LG.EXIT_CODE}, got "
        f"{proc.returncode}. stderr:\n{proc.stderr[:2000]}")
    assert LG.REFUSAL_MARKER in proc.stderr
    assert LG.SUPPORTED_ENTRY_POINT in proc.stderr, (
        "the refusal must point the reader at the one supported entry point")

    for marker in WAGERING_OUTPUT_MARKERS:
        assert marker not in proc.stdout, f"{marker!r} reached stdout"
    produced = {a for a in LEGACY_ARTIFACTS
                if (pristine_tree / a).exists()} - before
    assert not produced, (
        f"{entry_point} {' '.join(argv)} produced {sorted(produced)}")


@pytest.mark.parametrize("rel", ["run.py", "weekly.py"])
def test_guard_runs_before_any_heavy_import(candidate_tree, rel):
    """Module-scope refusal, ahead of the imports it is protecting.

    A guard placed after `from nflvalue import pipeline` would already have
    executed that package's import side effects before refusing.
    """
    module = _module_ast(candidate_tree, rel)

    refusal_lines = [n.lineno for n in module.body if _is_refusal(n)]
    assert refusal_lines, f"{rel} has no module-scope refusal"

    heavy_lines = []
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            names = {a.name for a in node.names}
            if node.module.startswith("nflvalue") and names != {"legacy_guard"}:
                heavy_lines.append(node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"numpy", "pandas", "scipy",
                                                "sklearn", "nflvalue"}:
                    heavy_lines.append(node.lineno)

    assert heavy_lines, f"{rel} no longer imports anything heavy -- update this test"
    assert min(refusal_lines) < min(heavy_lines), (
        f"{rel}: the refusal at line {min(refusal_lines)} must precede the "
        f"first heavy import at line {min(heavy_lines)}")


@pytest.mark.parametrize(
    "func", ["main", "job_wed", "job_t90", "job_tuesday", "job_deploy"])
def test_auto_weekly_refuses_on_every_operational_path(candidate_tree, func):
    """auto_weekly cannot guard at module scope, so it guards every path.

    The existing suite imports this module for its pure clock helpers
    (`current_week` / `last_completed_week`), which touch nothing operational.
    The refusal therefore sits at the top of `main()` and of every `job_*`
    function -- every route to pricing, staking, prop or notification work,
    whether entered from the command line or by a direct call.
    """
    module = _module_ast(candidate_tree, "scripts/auto_weekly.py")
    fn = next((n for n in module.body
               if isinstance(n, ast.FunctionDef) and n.name == func), None)
    assert fn is not None, f"auto_weekly.{func} is gone -- update this test"
    assert _is_refusal(_first_statement(fn)), (
        f"auto_weekly.{func} does not refuse as its first executed statement")


def test_legacy_guard_is_stdlib_only(candidate_tree):
    """A guard that can fail to import is not a guard.

    If `nflvalue/legacy_guard.py` grew a third-party import, a missing
    dependency would raise before the refusal and hand control back to the
    very code paths this module exists to stop.
    """
    module = _module_ast(candidate_tree, "nflvalue/legacy_guard.py")
    imported = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    outside = sorted(imported - set(sys.stdlib_module_names) - {"__future__"})
    assert not outside, f"legacy_guard imports non-stdlib modules: {outside}"


def test_supported_entry_point_still_runs_in_the_candidate_tree(candidate_tree):
    """Fail-closed must not have closed the door on the supported path."""
    probe = RG.import_probe(candidate_tree, RG.SUPPORTED_ENTRY_POINT)
    assert probe.returncode == 0, (
        "the one supported entry point must still import:\n" + probe.stderr)
