"""Weekly scoreboard & recap — WEEKLY_SCOREBOARD_SPEC, Phase 1(+).

The load-bearing design decision (spec §0): **the recap is read-only with
respect to the model.** It grades the projections the weekly loop PUBLISHED
(read from data/predictions_{season}_wk{week}.csv — never recomputed after
outcomes are known), and "trends that followed" live in an append-only
hypothesis ledger that is INTAKE for Workstreams E/H, never an adjustment.
Nothing in this module may write anything the ratings, simulation, or blend
read — tests/test_scoreboard.py enforces the import direction and the write
surface (data/recaps/ + the ledger file only).

Phase-2-lite note (2026-07-30): Workstream C exists, so the graded
projections are the drive-level MC margins; Workstream D's blend FAILED its
gate twice, so there is no blend and no bet-candidate language here. The
weekly close-free CLV diagnostic (raw published model vs same-book open)
appears when data/clv_live_{season}.csv has rows for the week, labelled
diagnostic-only — slice labels come from gameline.slice_tag, the same
constant F6 uses, per the spec's shared-taxonomy rule.

A good week is noise until gate 7 says otherwise; the recap says so itself
whenever the model beats the market on the week (tested, not eyeballed).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RECAP_DIR = os.path.join(DATA, "recaps")
LEDGER_PATH = os.path.join(DATA, "hypothesis_ledger.json")

TOP_N_MISSES = 5
REVISIT_MIN_OBS = 3
#: Historical baseline the weekly gap is judged against (gate-3 backtest,
#: pooled 2021-2023): model 13.24 vs market 12.24 MAE.
BASELINE_MODEL_MAE = 13.24
BASELINE_MARKET_MAE = 12.24


class ScoreboardError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Hypothesis ledger — append-only, never expiring, never load-bearing
# --------------------------------------------------------------------------- #
def load_ledger(path: Optional[str] = None) -> List[Dict]:
    p = path or LEDGER_PATH
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return json.load(fh)


def save_ledger(entries: List[Dict], path: Optional[str] = None) -> None:
    p = path or LEDGER_PATH
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, p)


def propose_hypotheses(misses: pd.DataFrame, season: int,
                       week: int) -> List[Dict]:
    """Zero or more candidates from this week's misses, from the display-only
    context columns. Mechanical rules, no cleverness: a pattern is proposed
    when it marks >= 2 of the top misses."""
    out = []
    for col, text in (("first_season_coach",
                       "biggest misses involved a first-season head coach"),
                      ("qb_change",
                       "biggest misses involved a team whose last game was "
                       "started by a non-primary QB")):
        if col not in misses.columns:
            continue
        hits = misses[misses[col].fillna(False)]
        if len(hits) >= 2:
            out.append({
                "id": f"{season}wk{week}-{col.replace('_', '-')}",
                "logged": f"{season}-week-{week} recap",
                "pattern": col,
                "hypothesis": (f"{len(hits)} of this week's "
                               f"{len(misses)} " + text +
                               "; display-only context, model does not "
                               "adjust for it (registry status governs)"),
                "evidence_games": [int(g) for g in hits["id"]],
                "status": "pending",
                "revisits": [],
                "promoted_to": None,
            })
    return out


def revisit_ledger(entries: List[Dict], misses: pd.DataFrame, season: int,
                   week: int) -> List[Dict]:
    """Score open entries against this week's misses. An observation supports
    an entry when its pattern marks >= 2 of the week's misses again, and
    contradicts when it marks none. After REVISIT_MIN_OBS observations the
    status moves; nothing ever changes a rating (see module docstring)."""
    for e in entries:
        if e.get("status") != "pending":
            continue
        col = e.get("pattern")
        if not col or col not in misses.columns:
            continue
        n_hits = int(misses[col].fillna(False).sum())
        support = n_hits >= 2
        contradict = n_hits == 0
        if not (support or contradict):
            continue                      # 1 hit: ambiguous, not an observation
        e["revisits"].append({"week": f"{season}wk{week}",
                              "support": bool(support)})
        if len(e["revisits"]) >= REVISIT_MIN_OBS:
            n_sup = sum(1 for r in e["revisits"] if r["support"])
            e["status"] = ("supported"
                           if n_sup * 2 > len(e["revisits"]) else "contradicted")
    return entries


# --------------------------------------------------------------------------- #
# Context columns (display-only, from cached factor feeds)
# --------------------------------------------------------------------------- #
def attach_context(graded: pd.DataFrame, season: int) -> pd.DataFrame:
    from . import ingest
    g = graded.copy()
    try:
        co = ingest.load_coaches([season])
        first = {str(r.team) for r in co.itertuples() if bool(r.first_season)}
    except Exception:  # noqa: BLE001
        first = set()
    g["first_season_coach"] = [
        (h in first) or (a in first) for h, a in zip(g["home"], g["away"])]
    try:
        qb = ingest.load_qb_starts([season])
        starters = qb[qb["is_week_starter"]].sort_values(["season", "week"])
        changed = set()
        for (s, team), grp in starters.groupby(["season", "team"],
                                               observed=True):
            seq = grp[["week", "athlete_id"]].drop_duplicates("week") \
                                             .sort_values("week")
            weeks = list(seq.itertuples())
            counts: dict = {}
            for i, r in enumerate(weeks):
                if i > 0 and counts:
                    primary = max(counts, key=counts.get)
                    if (counts[primary] >= 3
                            and weeks[i - 1].athlete_id != primary):
                        changed.add((int(r.week), str(team)))
                counts[r.athlete_id] = counts.get(r.athlete_id, 0) + 1
    except Exception:  # noqa: BLE001
        changed = set()
    g["qb_change"] = [((int(w), h) in changed) or ((int(w), a) in changed)
                      for w, h, a in zip(g["week"], g["home"], g["away"])]
    return g


# --------------------------------------------------------------------------- #
# The recap
# --------------------------------------------------------------------------- #
def grade_week(season: int, week: int,
               data_dir: Optional[str] = None) -> pd.DataFrame:
    """Join the PUBLISHED projections with finals + the market close.

    Reads data/predictions_{season}_wk{week}.csv — the artifact the loop
    published pregame. It is never regenerated here; grading a hindsight
    re-run is the leakage-in-a-narrative-costume the spec forbids.
    """
    from . import ingest
    d = data_dir or DATA
    pub_path = os.path.join(d, f"predictions_{season}_wk{week:02d}.csv")
    if not os.path.exists(pub_path):
        raise ScoreboardError(
            f"{os.path.basename(pub_path)} missing — the recap grades what "
            "was PUBLISHED; without a published file there is nothing "
            "honest to grade")
    pub = pd.read_csv(pub_path)
    games = ingest.load_games([season])
    finals = games.dropna(subset=["home_points", "away_points"])
    finals = finals[finals["week"] == week][["id", "home_points", "away_points"]]
    lines = ingest.load_lines([season])
    med = (lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median()
           if len(lines) else pd.Series(dtype=float))
    g = pub.merge(finals, left_on="id", right_on="id", how="inner")
    if g.empty:
        return g
    g["margin"] = g["home_points"] - g["away_points"]
    g["market"] = [(-float(med.get(i)) if i in med.index else np.nan)
                   for i in g["id"]]
    g["model_err"] = g["mc_margin"] - g["margin"]
    g["market_err"] = g["market"] - g["margin"]
    g["closer"] = np.where(g["market_err"].isna(), "model (no line)",
                           np.where(g["model_err"].abs() < g["market_err"].abs(),
                                    "model", "market"))
    return g


def build_recap(season: int, week: int, data_dir: Optional[str] = None,
                ledger_path: Optional[str] = None) -> Dict:
    """Produce the week's recap artifacts. Returns the structured dict."""
    d = data_dir or DATA
    graded = grade_week(season, week, d)
    if graded.empty:
        raise ScoreboardError(f"no graded games for {season} wk{week} — "
                              "either the week isn't final or nothing was "
                              "published")
    graded = attach_context(graded, season)
    both = graded.dropna(subset=["market"])
    model_mae = float(graded["model_err"].abs().mean())
    market_mae = float(both["market_err"].abs().mean()) if len(both) else None
    model_mae_h2h = (float(both["model_err"].abs().mean())
                     if len(both) else None)
    misses = graded.reindex(
        graded["model_err"].abs().sort_values(ascending=False).index
    ).head(TOP_N_MISSES)

    ledger = load_ledger(ledger_path)
    ledger = revisit_ledger(ledger, misses, season, week)
    new_entries = propose_hypotheses(misses, season, week)
    existing_ids = {e["id"] for e in ledger}
    new_entries = [e for e in new_entries if e["id"] not in existing_ids]
    ledger.extend(new_entries)
    save_ledger(ledger, ledger_path)

    beat_market = (market_mae is not None and model_mae_h2h is not None
                   and model_mae_h2h < market_mae)
    noise_note = (
        "The model beat the closing line this week. That is NOISE until "
        "proven otherwise: single-week gaps of this size are inside normal "
        "variance, no slice has a kill-check GO, and the Phase-2 blend gate "
        "FAILED on every window tested. Diagnostic, not an edge."
        if beat_market else
        "The market was closer this week, consistent with the historical "
        f"baseline (model {BASELINE_MODEL_MAE} vs market "
        f"{BASELINE_MARKET_MAE} MAE, pooled 2021-2023).")

    recap = {
        "season": season, "week": week,
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_graded": len(graded), "n_with_market": len(both),
        "model_mae": round(model_mae, 2),
        "model_mae_h2h": (None if model_mae_h2h is None
                          else round(model_mae_h2h, 2)),
        "market_mae": None if market_mae is None else round(market_mae, 2),
        "baseline": {"model": BASELINE_MODEL_MAE,
                     "market": BASELINE_MARKET_MAE},
        "model_beat_market": bool(beat_market),
        "noise_note": noise_note,
        "misses": [{
            "id": int(m.id), "matchup": f"{m.away} @ {m.home}",
            "published": round(float(m.mc_margin), 1),
            "actual": round(float(m.margin), 1),
            "error": round(float(m.model_err), 1),
            "market": (None if m.market != m.market
                       else round(float(m.market), 1)),
            "context": [c for c, flag in
                        (("first-season head coach", m.first_season_coach),
                         ("recent QB change", m.qb_change)) if flag] or
                       ["no cached display-only context flagged"],
        } for m in misses.itertuples()],
        "ledger_new": [e["id"] for e in new_entries],
        "ledger_open": [e["id"] for e in ledger if e["status"] == "pending"],
        "ledger_resolved": [{"id": e["id"], "status": e["status"]}
                            for e in ledger
                            if e["status"] in ("supported", "contradicted")],
        "paper_note": ("PAPER-ONLY diagnostic. Nothing here recommends a bet "
                       "or implies the model is improving in a load-bearing "
                       "sense; promotion of any hypothesis runs through the "
                       "full Workstream E gate. 1-800-GAMBLER."),
    }

    os.makedirs(os.path.join(d, "recaps"), exist_ok=True)
    jpath = os.path.join(d, "recaps", f"{season}_wk{week:02d}.json")
    with open(jpath, "w") as fh:
        json.dump(recap, fh, indent=2)
    with open(os.path.join(d, "recaps", f"{season}_wk{week:02d}.md"), "w") as fh:
        fh.write(render_markdown(recap))
    return recap


def render_markdown(r: Dict) -> str:
    lines = [
        f"# {r['season']} week {r['week']} — weekly scoreboard",
        "",
        f"_{r['paper_note']}_",
        "",
        f"Graded **{r['n_graded']}** published projections "
        f"({r['n_with_market']} with a market close).",
        "",
        f"**Model MAE {r['model_mae']}** vs **market {r['market_mae']}** "
        f"(same-game head-to-head: model {r['model_mae_h2h']}); historical "
        f"baseline {r['baseline']['model']} vs {r['baseline']['market']}.",
        "",
        r["noise_note"],
        "",
        "## This week's misses",
        "",
    ]
    for m in r["misses"]:
        mk = "no line" if m["market"] is None else f"market {m['market']:+}"
        lines.append(f"- **{m['matchup']}** — published {m['published']:+}, "
                     f"actual {m['actual']:+} (err {m['error']:+}; {mk}). "
                     f"Context: {'; '.join(m['context'])}.")
    resolved = ", ".join("{} ({})".format(e["id"], e["status"])
                         for e in r["ledger_resolved"]) or "none"
    lines += ["", "## Hypothesis ledger",
              "",
              f"New this week: {', '.join(r['ledger_new']) or 'none'}. "
              f"Open: {len(r['ledger_open'])}. "
              f"Resolved: {resolved}.",
              "",
              "_The ledger is intake for Workstreams E/H only. A ledger full "
              "of pending and contradicted entries is a healthy ledger._"]
    return "\n".join(lines) + "\n"
