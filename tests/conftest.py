"""STUB — not part of this project's scope.

This file was copied from fablesfable/tests/conftest.py during the initial
game-line-only duplication (2026-07-17), then found to depend on
fablesfable's player-prop stack (candidates.py/composite.py/features.py/
projection.py/context_features.py/context_study.py/prop_learning.py/
advanced_features.py — all deliberately excluded here; see
../_SETUP.md and the stallionsstallion MOC for the scope decision).

It can't be deleted from this vault mount, so it's stubbed instead of
left silently broken. Its fixtures (pbp_fast/pbp_tiny/backtest_report_fast) all load player-level pbp via nflvalue.features.load_pbp or run prop_backtest.py, neither of which exists here. A fresh conftest.py belongs here once the CFBD data pipeline (Phase 0 of the portability plan) exists.

Original, working version: nfl-sim/fablesfable/tests/conftest.py (still lives there,
untouched).
"""
