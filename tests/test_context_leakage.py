"""Leakage tests for the pregame-context layer (written BEFORE the
implementation).

Non-negotiable #5 says context is display-only *by construction*, not by
convention. These tests are the construction:

  * the context package cannot import the modelling modules, and the
    modelling modules cannot import it — so no code path exists by which a
    context value could reach a projection;
  * an observation timestamped at or after kickoff is REFUSED, not used;
  * closing-line fields are denied to this layer by name;
  * no provider may read a post-kickoff column;
  * poisoning every context input leaves the projection byte-identical.
"""
from __future__ import annotations

import ast
import datetime as dt
import glob
import os

import pytest

from nflvalue.context import contracts as C
from nflvalue.context import providers as P
from nflvalue.context import assemble as ASM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "nflvalue", "context")
DATA = os.path.join(ROOT, "data")

#: modules that compute a projection. The context layer may not touch them.
MODEL_MODULES = {
    "ratings", "cfb_sim", "engine", "kalman_engine", "montecarlo",
    "gameline", "bakeoff", "learn", "model", "factors", "oddsmath",
    "killcheck", "clv",
}
#: columns that only exist once a game has been played.
POSTGAME_DENYLIST = {
    "home_points", "away_points", "points", "margin", "total_actual",
    "result", "final", "winner", "score", "home_score", "away_score",
}
#: closing-line field names, denied to this layer by contract.
CLOSE_DENYLIST = {
    "spread_close", "close", "closing_spread", "over_under_close",
    "total_close", "formatted_spread_close", "closing_line", "clv_pts",
}

KICK = dt.datetime(2026, 8, 29, 23, 0, tzinfo=dt.timezone.utc)

# --------------------------------------------------------------------- #
# fixtures — synthetic frames, so these tests never need the real cache
# --------------------------------------------------------------------- #
from nflvalue.context import fixtures as FX  # noqa: E402


@pytest.fixture
def demo_game():
    return FX.demo_game()


@pytest.fixture
def demo_frames():
    return FX.demo_frames()


@pytest.fixture
def demo_context(demo_game, demo_frames):
    return ASM.assemble_game_context(
        game=demo_game, frames=demo_frames, data_dir=DATA,
        now=dt.datetime(2026, 8, 28, 12, 0, tzinfo=dt.timezone.utc))



def _imported_names(path, absolute_only=False):
    """Names this file imports. `absolute_only` drops `from . import x`,
    which is an intra-package reference, not a dependency."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[-1])
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and absolute_only:
                continue          # relative import: same package
            if node.module:
                out.add(node.module.split(".")[-1])
                out.add(node.module)
            if not node.level:
                for a in node.names:
                    out.add(a.name)
    return out


# --------------------------------------------------------------------- #
# F8-style architectural guarantee, both directions
# --------------------------------------------------------------------- #
def test_context_package_never_imports_a_modelling_module():
    files = sorted(glob.glob(os.path.join(PKG, "*.py")))
    assert files, "context package not found"
    for path in files:
        names = _imported_names(path)
        bad = names & MODEL_MODULES
        assert not bad, (
            f"{os.path.basename(path)} imports {sorted(bad)} — the context "
            "layer must be unable to reach the projection")


def test_modelling_modules_never_import_the_context_package():
    for path in sorted(glob.glob(os.path.join(ROOT, "nflvalue", "*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in MODEL_MODULES:
            continue
        names = _imported_names(path)
        assert not any("context" == n or n.endswith(".context") for n in names), (
            f"{stem}.py imports the context layer — that would make a "
            "display value reachable from a projection")


def test_context_package_declares_no_new_third_party_dependency():
    allowed = {"pandas", "numpy"}
    stdlib_ok = {"__future__", "dataclasses", "datetime", "enum", "json",
                 "math", "os", "typing", "html", "collections", "functools",
                 "hashlib", "re", "types", "zoneinfo", "ast", "glob"}
    for path in sorted(glob.glob(os.path.join(PKG, "*.py"))):
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                mods = [(node.module or "").split(".")[0]]
            for top in mods:
                if not top or top == "nflvalue":
                    continue
                assert top in allowed or top in stdlib_ok, (
                    f"{os.path.basename(path)} imports {top!r} — a new "
                    "dependency needs a documented approval gate, per "
                    "AGENT_BUILD_PROMPT section 8")


# --------------------------------------------------------------------- #
# time-window guards
# --------------------------------------------------------------------- #
def test_observation_at_or_after_kickoff_is_refused():
    for observed in (KICK, KICK + dt.timedelta(minutes=1)):
        av, note = C.kickoff_guard(observed_at=observed,
                                   effective_for_game=KICK, kickoff=KICK)
        assert av is C.Availability.REFUSED_POST_KICKOFF, observed
        assert "kickoff" in note.lower()


def test_observation_before_kickoff_passes_the_guard():
    av, note = C.kickoff_guard(observed_at=KICK - dt.timedelta(hours=1),
                               effective_for_game=KICK, kickoff=KICK)
    assert av is C.Availability.OK and note is None


def test_effective_for_game_after_kickoff_is_refused():
    av, _ = C.kickoff_guard(observed_at=KICK - dt.timedelta(days=1),
                            effective_for_game=KICK + dt.timedelta(hours=3),
                            kickoff=KICK)
    assert av is C.Availability.REFUSED_POST_KICKOFF


def test_post_kickoff_observation_survives_assembly_as_a_refusal(demo_game):
    """A cache refreshed AFTER kickoff must not become pregame context."""
    late = FX.demo_frames(observed_at=KICK + dt.timedelta(hours=1))
    ctx = ASM.assemble_game_context(game=demo_game, frames=late, data_dir=DATA,
                                    now=KICK + dt.timedelta(hours=5))
    refused = [v for v in ctx.values
               if v.availability is C.Availability.REFUSED_POST_KICKOFF]
    assert refused, "a post-kickoff observation must be visibly refused"
    assert all(v.value is None for v in refused)
    assert all(v.usage is C.Usage.MISSING for v in refused
               if v.registry_status != "excluded")
    # and it is still rendered, not dropped
    assert {v.key for v in ctx.values} == {s.key for s in C.FIELDS}


# --------------------------------------------------------------------- #
# market: opening prices only, diagnostic only
# --------------------------------------------------------------------- #
def test_market_provider_declares_open_only():
    prov = P.MarketOpenDiagnostic()
    assert prov.line_kind == "open"
    assert not (set(prov.reads) & CLOSE_DENYLIST)


def test_closing_line_columns_are_denied_by_name():
    for col in CLOSE_DENYLIST:
        with pytest.raises(C.LeakageError):
            C.assert_column_allowed(col)


def test_no_provider_reads_a_postgame_column():
    for prov in P.all_providers():
        bad = set(prov.reads) & POSTGAME_DENYLIST
        assert not bad, f"{prov.name} reads post-kickoff columns {sorted(bad)}"


def test_market_context_is_diagnostic_never_used_by_model(demo_context):
    mk = [v for v in demo_context.values if v.family == "market_open_diagnostic"]
    assert mk, "the market family must be represented"
    for v in mk:
        assert v.usage is not C.Usage.USED_BY_MODEL
        assert "diagnostic" in (v.note or "").lower() or v.value is None


def test_poisoned_closing_lines_do_not_change_context(demo_frames, demo_game):
    """`lines_*.parquet` carries `spread`/`over_under` beside the opening
    price. Those columns BECOME the closing line as kickoff approaches, so
    the provider must ignore them even though their names look innocent."""
    a = ASM.assemble_game_context(game=demo_game, frames=demo_frames,
                                  data_dir=DATA, now=KICK - dt.timedelta(days=1))
    poisoned = {k: v.copy() for k, v in demo_frames.items()}
    lines = poisoned["lines"]
    for col in ("spread", "over_under", "spread_close", "over_under_close",
                "closing_spread", "home_moneyline"):
        lines[col] = -999.0
    b = ASM.assemble_game_context(game=demo_game, frames=poisoned,
                                  data_dir=DATA, now=KICK - dt.timedelta(days=1))
    assert a.to_dict() == b.to_dict(), "a live/closing price moved a context value"


def test_market_provider_drops_non_opening_price_columns(demo_frames):
    prov = P.MarketOpenDiagnostic()
    lines = demo_frames["lines"].copy()
    lines["spread"] = -999.0
    lines["over_under"] = -999.0
    safe = prov.opening_only(lines)
    assert "spread" not in safe.columns and "over_under" not in safe.columns
    assert "spread_open" in safe.columns


# --------------------------------------------------------------------- #
# the projection cannot be touched
# --------------------------------------------------------------------- #
def test_assembling_leaves_the_projection_untouched(demo_frames, demo_game):
    projection = {"mc_margin": 9.783, "mc_total": 51.1295, "p_home_win": 0.70225}
    before = dict(projection)
    ASM.assemble_game_context(game=demo_game, frames=demo_frames, data_dir=DATA,
                              now=KICK - dt.timedelta(days=1),
                              projection=projection)
    assert projection == before, "the assembler mutated the projection"


def test_poisoning_every_input_leaves_the_projection_identical(
        demo_frames, demo_game):
    projection = {"mc_margin": 9.783, "mc_total": 51.1295, "p_home_win": 0.70225}
    baseline = dict(projection)
    poisoned = {}
    for k, df in demo_frames.items():
        d = df.copy()
        for col in d.columns:
            if d[col].dtype.kind in "fi":
                d[col] = 10_000.0
        poisoned[k] = d
    ctx = ASM.assemble_game_context(game=demo_game, frames=poisoned, data_dir=DATA,
                                    now=KICK - dt.timedelta(days=1),
                                    projection=projection)
    assert projection == baseline
    assert ctx.projection == baseline, (
        "the context carries the projection through verbatim or not at all")


def test_context_view_of_the_projection_is_read_only(demo_frames, demo_game):
    projection = {"mc_margin": 1.0}
    ctx = ASM.assemble_game_context(game=demo_game, frames=demo_frames, data_dir=DATA,
                                    now=KICK - dt.timedelta(days=1),
                                    projection=projection)
    with pytest.raises(TypeError):
        ctx.projection["mc_margin"] = 99.0


# --------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------- #
def test_derived_features_never_read_the_described_game(
        demo_frames, demo_game):
    prov = P.FiveFactorsWalkForward()
    assert prov.excludes_current_game is True
    used = prov.rows_considered(demo_frames, demo_game)
    assert demo_game["id"] not in set(used.get("game_ids", []))
    assert all(w < demo_game["week"] for w in used.get("weeks", []))
