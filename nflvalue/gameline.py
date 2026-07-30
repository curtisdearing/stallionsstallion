"""Workstream D — game-line market layer: blend, CLV log, slice tags.

This is the actual product; everything upstream (ratings, simulation) is an
input to it. Three deliberate design decisions, per BUILD_SPEC section 5 and
the standing non-negotiables:

* **The model is never bet straight.** The projection that leaves this module
  is an error-weighted blend of the model margin and the market margin, with
  the weight FIT on accumulated out-of-sample squared error (nfelo-style
  inverse-MSE weighting), never chosen by hand. Walk-forward: the weight used
  in week W is computed from prediction/outcome pairs strictly before W.

* **This is a NEW schema, not killcheck/clv.py's.** Those files inherited
  fablesfable's PLAYER-PROP shape (a `leans` table keyed on player_id+market
  with prop-tool sample bars) — flagged in _SETUP.md and
  FACTOR_REVALIDATION_SPEC §4. A game-line bet candidate is
  (game, market, side, price); it gets its own log and, later, its own
  kill-check bars re-derived for game lines (Workstream K), not the props
  n=150/52% inherited silently.

* **CLV is the scoreboard.** The log records the OPENING price taken and the
  closing price, in points, with a market slice tag, from day one. Sign
  conventions are hand-worked in tests/test_gameline.py — the classic silent
  bug in this layer is a flipped spread sign that makes everything look
  brilliant.

CFBD spread convention (verified on the cache): home-perspective, NEGATIVE
when the home team is favoured. Implied home margin = -spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

#: Power-conference NAMES (membership always resolved per season from the
#: teams table — never a hardcoded team list; realignment moves teams, not
#: usually conference names). Pac-12 stays: it existed through 2023 and was
#: rebuilt in 2026.
POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}

#: Minimum accumulated OOS pairs before a blend weight is trusted. Below this
#: the week is SKIPPED and recorded, not guessed. Structural floor, not tuned.
MIN_BLEND_HISTORY = 150


class GameLineError(RuntimeError):
    pass


def blend_weight(model_pred: np.ndarray, market_pred: np.ndarray,
                 actual: np.ndarray) -> float:
    """nfelo-style inverse-MSE weight on the MODEL (market gets 1-w).

    w = (1/mse_model) / (1/mse_model + 1/mse_market), computed on
    out-of-sample pairs only — the caller is responsible for walk-forward
    discipline (and the leakage test checks the caller).
    """
    mse_model = float(np.mean((model_pred - actual) ** 2))
    mse_market = float(np.mean((market_pred - actual) ** 2))
    if mse_model <= 0 or mse_market <= 0:
        raise GameLineError("degenerate MSE in blend fit — inputs look wrong")
    inv_mo, inv_mk = 1.0 / mse_model, 1.0 / mse_market
    return inv_mo / (inv_mo + inv_mk)


def walk_forward_blend(df: pd.DataFrame,
                       model_col: str = "mc_margin",
                       market_col: str = "market",
                       actual_col: str = "margin",
                       min_history: int = MIN_BLEND_HISTORY
                       ) -> pd.DataFrame:
    """Add `blend` and `blend_w` columns, weight fit strictly on earlier weeks.

    Rows are ordered by (season, week). For each (season, week) group the
    weight comes from ALL rows strictly before it (pooled across seasons —
    the error structure is a property of the model/market pair, not of one
    season). Weeks with fewer than `min_history` accumulated pairs get NaN
    blend (skipped, recorded), never a guessed weight.
    """
    need = [model_col, market_col, actual_col, "season", "week"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise GameLineError(f"walk_forward_blend: missing columns {missing}")
    out = df.sort_values(["season", "week"], kind="mergesort").copy()
    out["blend"] = np.nan
    out["blend_w"] = np.nan
    usable = out.dropna(subset=[model_col, market_col, actual_col])
    for (season, week), idx in out.groupby(["season", "week"]).groups.items():
        past = usable[(usable["season"] < season)
                      | ((usable["season"] == season) & (usable["week"] < week))]
        if len(past) < min_history:
            continue
        w = blend_weight(past[model_col].to_numpy(),
                         past[market_col].to_numpy(),
                         past[actual_col].to_numpy())
        rows = out.loc[idx]
        out.loc[idx, "blend_w"] = w
        out.loc[idx, "blend"] = (w * rows[model_col]
                                 + (1.0 - w) * rows[market_col])
    return out


def disagreement_beta(model_pred: np.ndarray, market_pred: np.ndarray,
                      actual: np.ndarray) -> float:
    """OLS of (actual - market) on (model - market): how much of the model's
    DISAGREEMENT with the market is signal?

    Added for the D-part-2 protocol (pre-registered in the 2026-07-30 worklog
    entry BEFORE any holdout number was seen): inverse-MSE weighting failed
    the Phase-2 gate because model and market errors correlate 0.90, which
    that rule ignores. Regressing on the disagreement handles the correlation
    by construction: beta=0 collapses to the market, beta=1 to the model.
    """
    x = model_pred - market_pred
    y = actual - market_pred
    vx = float(np.dot(x, x))
    if vx <= 1e-9:
        raise GameLineError("model never disagrees with the market — "
                            "nothing to regress on")
    return float(np.dot(x, y) / vx)


def walk_forward_disagreement_blend(df: pd.DataFrame,
                                    model_col: str = "mc_margin",
                                    market_col: str = "market",
                                    actual_col: str = "margin",
                                    min_history: int = MIN_BLEND_HISTORY
                                    ) -> pd.DataFrame:
    """blend = market + beta * (model - market), beta fit strictly on earlier
    weeks' OOS pairs. Same skip-don't-guess floor as walk_forward_blend."""
    need = [model_col, market_col, actual_col, "season", "week"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise GameLineError(f"walk_forward_disagreement_blend: missing {missing}")
    out = df.sort_values(["season", "week"], kind="mergesort").copy()
    out["blend"] = np.nan
    out["blend_beta"] = np.nan
    usable = out.dropna(subset=[model_col, market_col, actual_col])
    for (season, week), idx in out.groupby(["season", "week"]).groups.items():
        past = usable[(usable["season"] < season)
                      | ((usable["season"] == season) & (usable["week"] < week))]
        if len(past) < min_history:
            continue
        beta = disagreement_beta(past[model_col].to_numpy(),
                                 past[market_col].to_numpy(),
                                 past[actual_col].to_numpy())
        rows = out.loc[idx]
        out.loc[idx, "blend_beta"] = beta
        out.loc[idx, "blend"] = (rows[market_col]
                                 + beta * (rows[model_col] - rows[market_col]))
    return out


# --------------------------------------------------------------------------- #
# Slice tagging — per-season conference membership, never hardcoded teams
# --------------------------------------------------------------------------- #
def conf_tier(conference: str) -> str:
    return "P4" if conference in POWER_CONFS else "G5"


def slice_tag(home_conf: str, away_conf: str, week: int,
              neutral: bool) -> str:
    """One primary tag per game, most specific first (mirrors the spec's
    P4 / G5 / week 0-1 / neutral slices)."""
    if week <= 1:
        return "week0-1"
    if neutral:
        return "neutral"
    tiers = sorted((conf_tier(home_conf), conf_tier(away_conf)))
    return "v".join(tiers)          # G5vG5, G5vP4, P4vP4


# --------------------------------------------------------------------------- #
# The CLV log — game-line schema, opening price taken vs closing price
# --------------------------------------------------------------------------- #
def clv_points_spread(side: str, spread_open: float,
                      spread_close: float) -> float:
    """Points of closing-line value for a spread bet taken at the open.

    Home bet: positive when the line moved TOWARD home (home more favoured
    at close than at open). Hand-worked: open -7, close -9, home taken ->
    holding -7 beats the -9 close by 2.0 points.
    """
    if side == "home":
        return float(spread_open - spread_close)
    if side == "away":
        return float(spread_close - spread_open)
    raise GameLineError(f"unknown side {side!r}")


def build_clv_log(games: pd.DataFrame, lines: pd.DataFrame,
                  blend_by_game: Dict[int, float],
                  min_edge_pts: float) -> pd.DataFrame:
    """Paper CLV log: for every (game, book) with BOTH an opening and closing
    spread, if the blended margin disagrees with the OPEN by at least
    `min_edge_pts`, log a candidate taken at the open and score its CLV
    against that book's close.

    PAPER-ONLY by construction: the output is a research log. Nothing in it
    is a recommendation; gate 7 (kill-check, slice-specific, n-gated) is the
    only thing that can ever change that, and it has not passed.

    **THE SIGNAL MUST BE CLOSE-FREE — retraction on record (2026-07-30).**
    `blend_by_game` must contain NO closing-line information. The first CLV
    run passed the model-market blend here; because the blend embeds the
    same week's closing median, "edge vs the open" degenerated into "the
    line moved" and CLV became mechanically positive — the pure-artifact
    control (median close passed as the 'model') measures +2.9 to +3.5 pts
    of fake CLV, which is what that run reported. The honest raw-model
    signal measures +0.08 (2021-23, t=1.5) / +0.27 (2024, t=2.9).
    tests/test_gameline.py::test_clv_signal_must_be_close_free documents
    the trap. Pass raw model output, or a blend built against the OPEN.
    """
    meta = {int(r.id): r for r in games.itertuples()}
    recs: List[Dict] = []
    usable = lines.dropna(subset=["spread", "spread_open"])
    for r in usable.itertuples():
        gid = int(r.game_id)
        g = meta.get(gid)
        blend = blend_by_game.get(gid)
        if g is None or blend is None or np.isnan(blend):
            continue
        open_margin = -float(r.spread_open)
        edge = blend - open_margin
        if abs(edge) < min_edge_pts:
            continue
        side = "home" if edge > 0 else "away"
        recs.append({
            "game_id": gid, "season": int(g.season), "week": int(g.week),
            "book": str(r.provider), "market": "spread", "side": side,
            "line_open": float(r.spread_open), "line_close": float(r.spread),
            "blend_margin": float(blend), "edge_at_open": float(edge),
            "clv_pts": clv_points_spread(side, float(r.spread_open),
                                         float(r.spread)),
            "slice": slice_tag(str(g.home_conf), str(g.away_conf),
                               int(g.week), bool(g.neutral_site)),
        })
    return pd.DataFrame(recs)
