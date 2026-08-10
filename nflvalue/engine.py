"""Margin-engine dispatch — ONE place decides which model prices games.

`config.json`'s `"margin_engine"` selects the engine for every in-season
consumer (weekly loop, gate-3 sim harness, explainer). The registered
default is `"kalman"` (Workstream G decision, 2026-07-30 — see
`kalman_engine`'s module docstring for the record). `"ridge"` remains fully
wired on purpose: it is the H-spec drift-comparison baseline, and flipping
the config key back is the entire rollback story.

Consumers call `solve_table(games, season, week, params)` exactly where they
used to call `ratings.solve_cached(...)`. The ridge `RatingParams` argument
is accepted (and ignored) under the kalman engine so call sites don't fork.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union

import pandas as pd

from . import ratings as R
from . import kalman_engine as K

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The registered default (bakeoff G, 2026-07-30). config.json may override.
DEFAULT_ENGINE = "kalman"

Table = Union[R.RatingsTable, K.KalmanTable]


def margin_engine() -> str:
    """Engine name from config.json (`margin_engine`), else the default."""
    for name in ("config.local.json", "config.json"):
        path = os.path.join(_ROOT, name)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    cfg = json.load(fh)
            except Exception:  # noqa: BLE001 — a broken config must not
                continue       # silently flip the engine; fall through.
            if "margin_engine" in cfg:
                return str(cfg["margin_engine"])
    return DEFAULT_ENGINE


def solve_table(games: pd.DataFrame, season: int, week: int,
                params: Optional[R.RatingParams] = None,
                engine: Optional[str] = None) -> Table:
    """Weekly table from the configured engine. Fails loud on unknown names."""
    eng = engine or margin_engine()
    if eng == "kalman":
        return K.solve_kalman_cached(games, season, week)
    if eng == "ridge":
        return R.solve_cached(games, season, week, params or R.load_params())
    raise ValueError(
        f"unknown margin_engine {eng!r} — expected 'kalman' or 'ridge'. "
        "Refusing to guess which model prices games.")
