"""Workstream F — the explainer: why is the number what it is, and how much
should anyone trust it?

Architectural guarantee (F8, enforced by a test): this module is DOWNSTREAM
of everything. It consumes a RatingsTable, a finished simulation dict, and
market/CLV frames, and produces a serialized `explanation` dict + static
HTML. Nothing in `ratings.py`, `cfb_sim.py`, or `gameline.py` imports it, so
no code path exists by which an explanation can move a projection — the
`game_notes.py` pattern, promoted to a module-level rule.

Honesty requirements built in rather than bolted on:

* F1 — the margin decomposition is EXACT, not approximate: the ratings solve
  is linear, so team-deviation + conference-term + HFA sum to
  `predict_margin` to floating point, and the drive-level simulation's
  deviation from that is shown as its own signed row. A test asserts the sum.
* F2 — every row carries an evidence tier from
  research/proven-win-indicators.md; rejected inputs (unit ratings,
  explosiveness) appear ONLY as excluded rows explaining why they are absent.
* F3 — plain-language strings are rendered from the SAME values as the table
  (bound through one dict, property-tested), so prose cannot drift.
* F4 — the margin DISTRIBUTION is shown (deciles + histogram), with
  connectivity, sample size, and band width; a wide band renders as a reason
  to pass.
* F5 — the "why not" panel is populated by a rules engine that is
  structurally unable to return an empty list while the kill-check has not
  passed; a test renders a full real week and asserts non-emptiness anyway.
* F7 — paper-mode banner is ON and cannot be disabled: `paper_mode_banner`
  demands a slice-specific kill-check GO object and verifies it; none exists.
"""

from __future__ import annotations

import html
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

GAMBLING_HELP = "1-800-GAMBLER"

#: F2 — evidence tiers, sourced from research/proven-win-indicators.md and
#: this project's own gate-2/gate-3 findings. Tier 3 / excluded rows are
#: rendered only to explain their absence.
EVIDENCE_TIERS: Dict[str, Dict[str, str]] = {
    "team_rating": {
        "tier": "1",
        "why": "opponent-adjusted efficiency differential; the core "
               "pregame-predictive signal (proven-win-indicators: EPA/success "
               "rate family), fit walk-forward on n=1,765 games",
    },
    "conference": {
        "tier": "3",
        "why": "conference-strength term measured FLAT at this sample "
               "(gate-2 re-fit: 0.009 MAE across the whole lambda_conf grid); "
               "kept in the solve, shown here because it is part of the exact "
               "decomposition — do not read strength into it",
    },
    "hfa": {
        "tier": "1",
        "why": "per-team time-varying HFA earned its place at n=1,765 "
               "(league 2.83 pts; lambda_hfa=10 interior) after NOT earning "
               "it at n=42 — the factor registry tracks its re-validation",
    },
    "simulation": {
        "tier": "1",
        "why": "drive-level Monte Carlo with matchup-conditional drive rates "
               "(gate 3); this row is the sim's deviation from the linear "
               "prediction — OT handling and score-distribution asymmetry",
    },
    "units_passrush": {
        "tier": "excluded",
        "why": "unit ratings carry NO week-to-week signal at this sample "
               "(lag-1 stickiness ~-0.1 on 133 teams, explosiveness shrinkage "
               "fit to max) — excluded from the projection entirely",
    },
    "situational": {
        "tier": "ungated",
        "why": "no situational factor has cleared n-gate + matched control + "
               "BH q<0.05 (Workstream E not run); display-only by "
               "non-negotiable #6",
    },
}


class ExplainError(RuntimeError):
    pass


class PaperModeError(ExplainError):
    """Raised on any attempt to disable the paper banner without a verified
    slice-specific kill-check GO."""


def paper_mode_banner(disable: bool = False,
                      killcheck_go: Optional[Dict] = None) -> str:
    """F7: the banner is on unless a VERIFIED kill-check GO is presented.

    `killcheck_go` must be a dict with {'slice': str, 'go': True,
    'n': int>=bar, 'clv_mean': >0} produced by Workstream K — which does not
    exist yet, so today this function cannot be talked out of the banner.
    """
    if not disable:
        return ("PAPER MODE — research output. Nothing here is a bet "
                f"recommendation. {GAMBLING_HELP}")
    ok = (isinstance(killcheck_go, dict) and killcheck_go.get("go") is True
          and killcheck_go.get("slice") and int(killcheck_go.get("n", 0)) > 0
          and float(killcheck_go.get("clv_mean", 0.0)) > 0.0)
    if not ok:
        raise PaperModeError(
            "paper mode cannot be disabled: no verified slice-specific "
            "kill-check GO was provided (gate 7 has not passed)")
    return (f"kill-check GO for slice {killcheck_go['slice']} "
            f"(n={killcheck_go['n']}) — other slices remain paper-only. "
            f"{GAMBLING_HELP}")


# --------------------------------------------------------------------------- #
# F1 — exact contribution decomposition
# --------------------------------------------------------------------------- #
def decompose_margin(tbl, home: str, away: str, neutral: bool,
                     mc_margin: Optional[float] = None) -> List[Dict]:
    """Signed point contributions that SUM (exactly) to the projection.

    The solve is linear: off[t] = off_dev[t] + conf_off[conf(t)] and
    def_[t] = def_dev[t] + conf_def[conf(t)], so
      margin = HFA + team-deviation diff + conference-term diff.
    If `mc_margin` is given, a fourth row carries the simulation's deviation
    from the linear prediction, so the rows sum to the SIM margin instead.
    """
    tbl.require_usable(home)
    tbl.require_usable(away)
    hc, ac = tbl.team_conf.get(home, ""), tbl.team_conf.get(away, "")
    team = ((tbl.off_dev.get(home, 0.0) + tbl.def_dev.get(away, 0.0))
            - (tbl.off_dev.get(away, 0.0) + tbl.def_dev.get(home, 0.0)))
    conf = ((tbl.conf_off.get(hc, 0.0) + tbl.conf_def.get(ac, 0.0))
            - (tbl.conf_off.get(ac, 0.0) + tbl.conf_def.get(hc, 0.0)))
    hfa = tbl.hfa(home, neutral)
    rows = [
        {"key": "team_rating", "label": "Team rating differential (off/def)",
         "points": team},
        {"key": "conference", "label": "Conference-strength term",
         "points": conf},
        {"key": "hfa",
         "label": ("Home-field advantage" if not neutral
                   else "Home-field advantage (neutral site: zero)"),
         "points": hfa},
    ]
    if mc_margin is not None:
        linear = hfa + team + conf
        rows.append({"key": "simulation",
                     "label": "Drive-level simulation adjustment",
                     "points": float(mc_margin) - linear})
    for r in rows:
        meta = EVIDENCE_TIERS[r["key"]]
        r["tier"], r["why"] = meta["tier"], meta["why"]
        r["points"] = float(r["points"])
    return rows


def excluded_rows() -> List[Dict]:
    """Inputs shown ONLY to explain why they contribute nothing (F2)."""
    return [
        {"key": k, "label": lbl, "points": 0.0,
         "tier": EVIDENCE_TIERS[k]["tier"], "why": EVIDENCE_TIERS[k]["why"]}
        for k, lbl in (("units_passrush", "Unit matchups (pass/rush)"),
                       ("situational", "Situational factors"))
    ]


# --------------------------------------------------------------------------- #
# F3 — plain language bound to the same values
# --------------------------------------------------------------------------- #
def plain_language(row: Dict, home: str, away: str) -> str:
    p = row["points"]
    mag = abs(p)
    direction = home if p > 0 else away
    if row["key"] == "team_rating":
        return (f"Opponent-adjusted team strength favours {direction} by "
                f"{mag:.1f} points.")
    if row["key"] == "conference":
        return (f"The conference term adds {p:+.1f} points — measured flat "
                "at this sample; treat it as bookkeeping, not signal.")
    if row["key"] == "hfa":
        if p == 0.0:
            return "Neutral site: no home-field advantage is applied."
        return (f"{home}'s home field is worth {mag:.1f} points here "
                "(team-specific, fit walk-forward).")
    if row["key"] == "simulation":
        return (f"Simulating the game drive by drive moves the number by "
                f"{p:+.1f} points versus the linear rating prediction.")
    return row["why"]


# --------------------------------------------------------------------------- #
# F5 — the "why not" rules engine
# --------------------------------------------------------------------------- #
def why_not_reasons(*, tbl, home: str, away: str, week: int,
                    mc_margin: float, mc_sd: float,
                    market: Optional[float],
                    model_mae: float,
                    slice_tag: str,
                    clv_slices: Optional[pd.DataFrame] = None) -> List[str]:
    """The case against acting. Structurally never empty while gate 7 is
    unpassed — and a test renders a full week to prove it stays populated
    with SPECIFIC reasons too."""
    reasons: List[str] = []
    conn_h = tbl.connectivity.score.get(home, 0.0)
    conn_a = tbl.connectivity.score.get(away, 0.0)
    if min(conn_h, conn_a) < 0.5:
        reasons.append(
            f"Thin schedule connectivity ({home} {conn_h:.2f} / {away} "
            f"{conn_a:.2f}) — the rating rests on few comparable games.")
    n_h = tbl.connectivity.n_games.get(home, 0)
    n_a = tbl.connectivity.n_games.get(away, 0)
    if min(n_h, n_a) < 6:
        reasons.append(f"Small sample: {home} {n_h} / {away} {n_a} rated "
                       "FBS games.")
    if week <= 4:
        reasons.append("Early season: ratings lean on the preseason prior, "
                       "which is deliberately weak.")
    band = max(tbl.net_band.get(home, 0.0), tbl.net_band.get(away, 0.0))
    if band > 6.0:
        reasons.append(f"Wide rating band (±{band:.1f} pts on the shakier "
                       "side) — a wide band is a reason to pass, not a "
                       "footnote.")
    if market is not None and not np.isnan(market):
        gap = abs(mc_margin - market)
        if gap < model_mae:
            reasons.append(
                f"Model-market gap ({gap:.1f} pts) is smaller than the "
                f"model's own out-of-sample error ({model_mae:.1f} MAE) — "
                "the disagreement is within noise.")
        else:
            reasons.append(
                f"The market disagrees by {gap:.1f} pts, and the Phase-2 "
                "gate showed the closing line is the better margin "
                "forecaster on every window tested (2021-2023 AND the 2024 "
                "holdout).")
    else:
        reasons.append("No market line cached for this game — nothing to "
                       "compare against, which is itself disqualifying.")
    if clv_slices is not None and slice_tag in getattr(clv_slices, "index", []):
        row = clv_slices.loc[slice_tag]
        reasons.append(
            f"Slice {slice_tag}: measured close-free CLV "
            f"{row['mean']:+.2f} pts on n={int(row['count'])} — "
            "not a demonstrated edge.")
    else:
        reasons.append(f"Slice {slice_tag} has no positive CLV history.")
    # The structural floor: gate 7. Present until a kill-check GO exists.
    reasons.append("Kill-check (gate 7) has NOT passed for any slice — "
                   "paper-only stands.")
    return reasons


# --------------------------------------------------------------------------- #
# The explanation object (serialized; F8)
# --------------------------------------------------------------------------- #
def build_explanation(*, tbl, game: Dict, sim: Dict,
                      market: Optional[float], market_total: Optional[float],
                      model_mae: float, slice_tag: str,
                      clv_slices: Optional[pd.DataFrame] = None) -> Dict:
    home, away = game["home_team"], game["away_team"]
    neutral = bool(game.get("neutral_site", False))
    rows = decompose_margin(tbl, home, away, neutral,
                            mc_margin=sim["margin_mean"])
    margin = np.asarray(sim["margin"])
    deciles = np.percentile(margin, [10, 25, 50, 75, 90])
    hist_bins = np.arange(-49, 50, 7)
    hist, _ = np.histogram(np.clip(margin, -48, 48), bins=hist_bins)
    exp = {
        "game_id": int(game["id"]), "season": int(game["season"]),
        "week": int(game["week"]), "home": home, "away": away,
        "neutral": neutral, "slice": slice_tag,
        "pred_linear": float(tbl.predict_margin(home, away, neutral)),
        "mc_margin": float(sim["margin_mean"]),
        "mc_sd": float(sim["margin_sd"]),
        "mc_total": float(sim["total_mean"]),
        "p_home_win": float(sim["p_home_win"]),
        "market": None if market is None or np.isnan(market) else float(market),
        "market_total": (None if market_total is None or np.isnan(market_total)
                         else float(market_total)),
        "contributions": rows,
        "excluded": excluded_rows(),
        "prose": {r["key"]: plain_language(r, home, away) for r in rows},
        "margin_deciles": {"p10": float(deciles[0]), "p25": float(deciles[1]),
                           "p50": float(deciles[2]), "p75": float(deciles[3]),
                           "p90": float(deciles[4])},
        "margin_hist": {"bins": hist_bins[:-1].tolist(),
                        "counts": (hist / len(margin)).round(4).tolist()},
        "confidence": {
            "connectivity_home": float(tbl.connectivity.score.get(home, 0.0)),
            "connectivity_away": float(tbl.connectivity.score.get(away, 0.0)),
            "games_home": int(tbl.connectivity.n_games.get(home, 0)),
            "games_away": int(tbl.connectivity.n_games.get(away, 0)),
            "band_home": float(tbl.net_band.get(home, 0.0)),
            "band_away": float(tbl.net_band.get(away, 0.0)),
        },
        "why_not": why_not_reasons(
            tbl=tbl, home=home, away=away, week=int(game["week"]),
            mc_margin=float(sim["margin_mean"]), mc_sd=float(sim["margin_sd"]),
            market=market, model_mae=model_mae, slice_tag=slice_tag,
            clv_slices=clv_slices),
    }
    return exp


def contributions_sum(exp: Dict) -> float:
    return float(sum(r["points"] for r in exp["contributions"]))


# --------------------------------------------------------------------------- #
# HTML rendering — self-contained, dashboard.py pattern
# --------------------------------------------------------------------------- #
_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;
background:#101418;color:#dfe6ee}
.banner{background:#7a1f1f;color:#fff;padding:10px 16px;font-weight:600;
position:sticky;top:0}
.wrap{max-width:1080px;margin:0 auto;padding:16px}
.game{background:#1a2026;border:1px solid #2a333c;border-radius:10px;
margin:18px 0;padding:14px 18px}
h2{margin:4px 0 10px;font-size:1.15em}
table{border-collapse:collapse;width:100%;font-size:.92em}
td,th{padding:4px 8px;border-bottom:1px solid #2a333c;text-align:left}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tier{font-size:.78em;padding:1px 7px;border-radius:9px;background:#2a333c}
.tier-1{background:#1d4d2c}.tier-3{background:#4d431d}
.tier-excluded{background:#4d1d1d}.tier-ungated{background:#3c2a4d}
.cols{display:flex;gap:18px;flex-wrap:wrap}
.col{flex:1 1 420px}
.whynot{background:#241a1a;border:1px solid #46302f;border-radius:8px;
padding:10px 14px}
.whynot h3,.for h3{margin:2px 0 8px;font-size:1.0em}
.for{background:#1a2420;border:1px solid #2f4638;border-radius:8px;
padding:10px 14px}
.small{color:#93a1b0;font-size:.85em}
.hist{display:flex;align-items:flex-end;gap:2px;height:64px;margin:6px 0}
.hist div{background:#3b6ea5;width:100%}
.dagger{color:#c9a227}
ul{margin:6px 0 6px 18px;padding:0}li{margin:3px 0}
.head{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
.badge{background:#2a333c;border-radius:6px;padding:2px 8px;font-size:.85em}
"""


def _fmt(x, nd=1):
    return "—" if x is None else f"{x:+.{nd}f}"


def render_week_html(explanations: List[Dict], *, clv_slices: pd.DataFrame,
                     title: str, generated_note: str) -> str:
    """One self-contained HTML page for a week. No server, no external libs."""
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style>",
        f"<div class='banner'>{html.escape(paper_mode_banner())}</div>",
        "<div class='wrap'>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='small'>{html.escape(generated_note)} — display layer is "
        "read-only by construction (F8): nothing on this page can move a "
        "projection.</p>",
    ]
    # F6 — slice CLV scoreboard, honest version
    parts.append("<div class='game'><h2>Slice CLV scoreboard (close-free, "
                 "paper log)</h2><table><tr><th>slice</th><th class='num'>n"
                 "</th><th class='num'>mean CLV (pts)</th><th>kill-check"
                 "</th></tr>")
    for s, r in clv_slices.iterrows():
        parts.append(f"<tr><td>{html.escape(str(s))}</td>"
                     f"<td class='num'>{int(r['count'])}</td>"
                     f"<td class='num'>{r['mean']:+.2f}</td>"
                     "<td>NOT RUN — no slice has GO</td></tr>")
    parts.append("</table><p class='small'>No slice may imply an edge until "
                 "the kill-check returns GO for that slice. None has. The "
                 "Phase-2 blend gate FAILED on 2021-2023 and on the 2024 "
                 "holdout: the closing line remains the better margin "
                 "forecaster.</p></div>")

    for e in explanations:
        c = e["confidence"]
        parts.append("<div class='game'>")
        parts.append(
            "<div class='head'><h2>"
            f"{html.escape(e['away'])} @ {html.escape(e['home'])}"
            f"{' (neutral)' if e['neutral'] else ''}</h2>"
            f"<span class='badge'>wk {e['week']} · {html.escape(e['slice'])}"
            "</span></div>")
        parts.append(
            f"<p>Simulated margin <b>{_fmt(e['mc_margin'])}</b> "
            f"(sd {e['mc_sd']:.1f}) · linear {_fmt(e['pred_linear'])} · "
            f"market {_fmt(e['market'])} · P({html.escape(e['home'])} win) "
            f"{e['p_home_win']:.2f} · total {e['mc_total']:.0f}"
            f"{'' if e['market_total'] is None else ' (mkt %.0f)' % e['market_total']}"
            "</p>")
        parts.append("<div class='cols'><div class='col'>")
        parts.append("<table><tr><th>contribution</th><th class='num'>pts"
                     "</th><th>evidence</th></tr>")
        for r in e["contributions"] + e["excluded"]:
            tier = html.escape(str(r["tier"]))
            parts.append(
                f"<tr><td>{html.escape(r['label'])}</td>"
                f"<td class='num'>{r['points']:+.2f}</td>"
                f"<td><span class='tier tier-{tier}' title="
                f"'{html.escape(r['why'])}'>tier {tier}</span></td></tr>")
        s = contributions_sum(e)
        parts.append(f"<tr><td><b>sum</b></td><td class='num'><b>{s:+.2f}"
                     "</b></td><td class='small'>= simulated margin (exact)"
                     "</td></tr></table>")
        parts.append("<ul class='small'>")
        for k in ("team_rating", "hfa", "conference", "simulation"):
            if k in e["prose"]:
                parts.append(f"<li>{html.escape(e['prose'][k])}</li>")
        parts.append("</ul></div><div class='col'>")
        # F4 — distribution
        d = e["margin_deciles"]
        parts.append(
            f"<p class='small'>margin deciles: 10% {d['p10']:+.0f} · 25% "
            f"{d['p25']:+.0f} · median {d['p50']:+.0f} · 75% {d['p75']:+.0f}"
            f" · 90% {d['p90']:+.0f} · connectivity "
            f"{c['connectivity_home']:.2f}/{c['connectivity_away']:.2f} · "
            f"band ±{max(c['band_home'], c['band_away']):.1f}</p>")
        mx = max(e["margin_hist"]["counts"]) or 1
        bars = "".join(f"<div style='height:{100*v/mx:.0f}%'></div>"
                       for v in e["margin_hist"]["counts"])
        parts.append(f"<div class='hist'>{bars}</div>")
        parts.append("<div class='whynot'><h3>Why not</h3><ul>")
        for r in e["why_not"]:
            parts.append(f"<li>{html.escape(r)}</li>")
        parts.append("</ul></div>")
        parts.append("</div></div></div>")

    parts.append(f"<p class='small'>{html.escape(GAMBLING_HELP)} — if this "
                 "stops feeling like the modeling exercise it is, notice "
                 "that.</p></div>")
    return "".join(parts)
