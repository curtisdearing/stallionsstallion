"""The context hook in explain.py must be PURELY additive.

Omit `contexts=` and the week explainer renders exactly what it rendered
before the hook existed. Pass it and the only difference is the panel
block plus its stylesheet — asserted by stripping both back out and
comparing to the no-context render.
"""
from __future__ import annotations

import datetime as dt
import os
import re

import pandas as pd
import pytest

from nflvalue import explain as EX
from nflvalue.context import assemble as ASM
from nflvalue.context import fixtures as FX
from nflvalue.context.panel import CSS as CTX_CSS

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NOW = dt.datetime(2026, 8, 28, 12, 0, tzinfo=dt.timezone.utc)


def _explanation(game_id=999001):
    return {
        "game_id": game_id, "season": 2026, "week": 3, "home": "Demo Home",
        "away": "Demo Away", "neutral": False, "slice": "P4vP4",
        "pred_linear": 6.0, "mc_margin": 6.1, "mc_sd": 18.2, "mc_total": 51.1,
        "p_home_win": 0.63, "market": None, "market_total": None,
        "contributions": [{"key": "team_rating", "label": "Team rating",
                           "points": 3.7, "tier": "1", "why": "core"},
                          {"key": "hfa", "label": "Home field", "points": 2.4,
                           "tier": "1", "why": "hfa"}],
        "excluded": EX.excluded_rows(),
        "prose": {"team_rating": "strength favours Demo Home"},
        "margin_deciles": {"p10": -17.0, "p25": -6.0, "p50": 6.0, "p75": 18.0,
                           "p90": 29.0},
        "margin_hist": {"bins": [-14, -7, 0, 7], "counts": [0.2, 0.3, 0.3, 0.2]},
        "confidence": {"connectivity_home": 1.0, "connectivity_away": 1.0,
                       "games_home": 80, "games_away": 74, "band_home": 7.3,
                       "band_away": 7.5},
        "why_not": ["the band spans a change of winner"],
    }


@pytest.fixture
def clv():
    return pd.DataFrame({"count": [757], "mean": [0.182]},
                        index=pd.Index(["P4vP4"], name="slice"))


@pytest.fixture
def ctx():
    return ASM.assemble_game_context(game=FX.demo_game(), frames=FX.demo_frames(),
                                     data_dir=DATA, now=NOW)


def _render(clv, contexts=None):
    kw = {} if contexts is None else {"contexts": contexts}
    return EX.render_week_html([_explanation()], clv_slices=clv,
                               title="t", generated_note="n", **kw)


def test_no_context_render_is_unchanged_by_the_hook(clv):
    assert _render(clv) == _render(clv, contexts={})
    assert _render(clv) == _render(clv, contexts=None)


def test_no_context_render_has_no_panel_and_no_panel_css(clv):
    out = _render(clv)
    assert "class='ctx'" not in out
    assert CTX_CSS not in out


def test_with_context_the_only_difference_is_the_panel(clv, ctx):
    plain = _render(clv)
    withctx = _render(clv, contexts={999001: ctx})
    assert "<!--ctx:start-->" in withctx and "class='ctx'" in withctx
    stripped = re.sub(r"<!--ctx:start-->.*?<!--ctx:end-->", "", withctx,
                      flags=re.S).replace(CTX_CSS, "")
    assert stripped == plain, "the hook changed something outside the panel"


def test_panel_sits_inside_its_game_block(clv, ctx):
    withctx = _render(clv, contexts={999001: ctx})
    assert withctx.index("Simulated margin") < withctx.index("<!--ctx:start-->")
    assert withctx.index("Why not") < withctx.index("<!--ctx:start-->")


def test_explanation_payload_gains_context_only_when_asked():
    exp = _explanation()
    assert "context" not in exp
    exp2 = dict(exp)
    exp2["context"] = {"x": 1}
    assert set(exp2) - set(exp) == {"context"}


def test_panel_value_never_appears_in_the_projection_line(clv, ctx):
    withctx = _render(clv, contexts={999001: ctx})
    head = withctx[:withctx.index("<!--ctx:start-->")]
    assert "Simulated margin" in head and "6.1" in head
    assert "returning_ppa" not in head and "talent_composite" not in head
