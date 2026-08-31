"""The explainer panel must separate four things and never blur them:
used by the model, display-only, excluded, and missing.
"""
from __future__ import annotations

import datetime as dt
import os
import re

import pytest

from nflvalue.context import assemble as ASM
from nflvalue.context import contracts as C
from nflvalue.context import fixtures as FX
from nflvalue.context import panel as PANEL

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NOW = dt.datetime(2026, 8, 28, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def ctx():
    return ASM.assemble_game_context(game=FX.demo_game(), frames=FX.demo_frames(),
                                     data_dir=DATA, now=NOW,
                                     projection={"mc_margin": 6.1})


def test_panel_renders_all_four_buckets_with_headings(ctx):
    html = PANEL.render_context_panel(ctx)
    for heading in ("Used by the model", "Display-only", "Excluded", "Missing"):
        assert heading in html, f"panel is missing the {heading!r} section"


def test_every_field_appears_exactly_once(ctx):
    html = PANEL.render_context_panel(ctx)
    for spec in C.FIELDS:
        assert html.count(f'data-field="{spec.key}"') == 1, spec.key


def test_panel_shows_provenance_for_available_values(ctx):
    html = PANEL.render_context_panel(ctx)
    for v in ctx.values:
        if v.availability is C.Availability.OK:
            assert v.source_url_or_id in html
            assert v.observed_at.strftime("%Y-%m-%d") in html


def test_missing_values_state_why(ctx):
    html = PANEL.render_context_panel(ctx)
    for v in ctx.by_usage(C.Usage.MISSING):
        assert v.availability.value in html
    assert "unavailable" in html.lower()


def test_panel_never_prints_a_number_for_a_missing_field(ctx):
    for v in ctx.by_usage(C.Usage.MISSING):
        assert v.value is None
        assert not re.search(r"-?\d", v.display.replace("unavailable", ""))


def test_panel_names_the_open_approval_gates(ctx):
    html = PANEL.render_context_panel(ctx)
    assert "Approval gates" in html
    for src in ("weather_forecast", "qb_availability_feed",
                "unit_availability_feed", "cfbd_venues", "cfbd_portal"):
        assert src in html


def test_panel_states_the_display_only_guarantee(ctx):
    html = PANEL.render_context_panel(ctx)
    assert "cannot" in html.lower() and "projection" in html.lower()


def test_panel_escapes_hostile_strings():
    game = FX.demo_game()
    game["home_team"] = "<script>alert(1)</script>"
    frames = FX.demo_frames()
    ctx = ASM.assemble_game_context(game=game, frames=frames, data_dir=DATA, now=NOW)
    html = PANEL.render_context_panel(ctx)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_excluded_section_lists_standing_exclusions(ctx):
    from nflvalue.context.registry import PRESERVED_EXCLUSIONS
    html = PANEL.render_context_panel(ctx)
    for fid in PRESERVED_EXCLUSIONS:
        assert fid in html


def test_panel_is_self_contained(ctx):
    html = PANEL.render_context_panel(ctx)
    assert "<script" not in html.lower()
    assert "http://" not in html and "src=" not in html
