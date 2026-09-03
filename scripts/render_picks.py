#!/usr/bin/env python3
"""Every FBS game of a published week as one page: projected winner, win
probability, 80% / 95% predictive intervals on the margin, the rationale, and
the caution -- written into the vault's `picks/` folder, outside the repo.

This is Curtis's reading surface, not the published one. The published card
(`nflvalue/insight.py`) refuses actionable vocabulary by construction; this
page is allowed to say "pick" because it is a private research note in the
vault, and it says so in its own header. Every number on it is read from the
contract artifact and the week's insight cards -- nothing is recomputed, no
model is re-solved, no clock is read. Re-running it on the same inputs
produces the same bytes.

What "every FBS game" means here: every scheduled game with at least one FBS
side (the season's `/teams` classification, the same set the producer uses).
FCS-vs-FCS games are not FBS games and are left out. An FBS game the model
refused (no pre-kickoff record, a new-to-FBS side with no rating) is listed
with its reason rather than dropped -- a missing row would read as a missing
game.

Intervals: the contract publishes one predictive distribution per game,
Normal(projected margin, forecast_margin_sd). The 80% interval is mean
+/- 1.2816 SD, the 95% mean +/- 1.9600 SD, expressed from the projected
winner's side. Those are wide -- roughly +/- 22 and +/- 33 points at the
registered early-season SD -- and that width is the honest answer, not a
formatting choice.

Usage: python3 scripts/render_picks.py --season 2026 --week 1 [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from nflvalue import cfb_calendar  # noqa: E402
from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import ingest  # noqa: E402
from nflvalue import producers as PR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

Z80 = 1.2816
Z95 = 1.9600

FAMILY_LABEL = {
    "ridge_preseason_prior": "ridge + preseason prior (FBS-vs-FBS)",
    "kalman_montecarlo": "Kalman ratings + drive-level Monte Carlo",
    "ridge_cross_division": "ridge vs pooled FCS pseudo-team (FBS-vs-FCS)",
}

#: The four traditional power conferences, for framing an FBS-vs-FBS
#: rationale in plain terms (a blueblood against a small-conference team).
#: Descriptive only -- conference is never an input to any forecast, so
#: getting this set wrong changes nothing about a number, only a sentence.
POWER_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}


def load_conferences(season: int) -> Dict[str, str]:
    """team -> conference name for the season, from the same /teams pull
    `producers.fbs_teams` uses. Purely descriptive: used only to phrase the
    rationale in plain terms, never to price anything. {} if the pull has
    no conference column -- the casual rationale just skips that framing."""
    teams = ingest.load("teams", [int(season)])
    if teams.empty or "conference" not in teams.columns:
        return {}
    return {str(r.team): str(r.conference) for r in teams.itertuples()}


def _wagered_team(pick: str) -> str:
    """The team actually being bet, parsed off the shown pick string --
    'Team ML' / 'Team ML -500' -> 'Team'; 'Team +10.0' / 'Team -7.0' ->
    'Team'. This is the only thing that should ever decide whose case the
    rationale argues; the model's own straight-up favourite is a different
    question and the two are not always the same team (Curtis, 2026-09-03:
    a huge spread on an FCS or Group-of-Five mismatch is very often a bet
    that the underdog *covers*, not that the favourite is good, and a
    rationale that spends its whole length on the favourite while the pick
    sits on the other team is not a rationale for the pick)."""
    if " ML" in pick:
        return pick.split(" ML")[0].strip()
    team, _, _num = pick.rpartition(" ")
    return team.strip()


def casual_rationale_parts(p: Dict[str, Any], call: Dict[str, Any],
                           conferences: Dict[str, str],
                           bases: Dict[str, dict]) -> List[str]:
    """The same rationale `pick_row` already built, said the way you'd say
    it out loud instead of the technical version -- but argued for the team
    actually being bet (`call['pick']`), which is not always the model's own
    straight-up favourite (`p['pick']`). A wide spread on a mismatch is
    routinely a bet that the big favourite *doesn't cover* that many points,
    not that the underdog wins; when that's the case this leads with the
    number, not the favourite, and the historical cross-division bias -- the
    FBS side tends to beat the projection, not fall short of it -- is flipped
    to say what it actually means for a points-taking bet instead of reading
    like blanket support for the favourite. No new facts either way: the
    rating gap and home-field points are `p['why_components']` (the same
    numbers that sum to the published margin), the framing is
    `p['model_family']` plus conference tier, and the bias is the same
    registered residual -- nothing here is invented or estimated."""
    if not p["pick"] or not call.get("pick"):
        return []
    home, away = p["home"], p["away"]
    model_fav = p["pick"]
    model_dog = away if model_fav == home else home
    wagered = _wagered_team(call["pick"])
    on_model_side = (wagered == model_fav)
    comps = p.get("why_components") or []
    rating = next((comp["points"] for comp in comps
                  if str(comp.get("label", "")).startswith("Rating differential")),
                 None)
    hfa = next((comp["points"] for comp in comps
               if str(comp.get("label", "")).startswith("Home-field")), None)
    family = p.get("model_family")
    hfa_clause = (f", plus {home}'s usual home-field bump of about {hfa:+.1f}"
                 if hfa is not None else "")
    bits = []

    if not on_model_side:
        # The pick argues against the model's own favourite -- lead with
        # why THIS side, in the actual bet's own terms, not the favourite's.
        margin = p.get("margin_pick")
        if call["kind"] == "spread" and margin is not None:
            line = float(call["pick"].rsplit(" ", 1)[-1])
            cmp = ("more cushion than" if abs(line) > margin
                  else "not far off")
            bits.append(
                f"The model likes {model_fav} to win this by about "
                f"{margin:.0f} points{hfa_clause} — but the number gives "
                f"{wagered} {line:+.1f}, {cmp} that projected gap, so the "
                f"bet is that {model_fav} doesn't cover, not that "
                f"{wagered} wins outright.")
        elif call["kind"] == "moneyline" and margin is not None:
            price = call.get("price")
            price_bit = (f" ({price:+.0f})" if price is not None else "")
            bits.append(
                f"The model still expects {model_fav} to win this by about "
                f"{margin:.0f} points{hfa_clause} and gives {wagered} only "
                f"about {call['p']:.0%} to win outright — but the price on "
                f"{wagered}{price_bit} is generous enough that the number, "
                "not the model's own pick, is where the value sits.")

    if family == "ridge_cross_division":
        bits.append(
            f"{model_fav} is the big name in this one — {model_dog} is a "
            "small school making the trip for a payday, a classic money "
            "game.")
    else:
        fav_conf, dog_conf = conferences.get(model_fav), conferences.get(model_dog)
        if fav_conf and dog_conf:
            fav_power, dog_power = (fav_conf in POWER_CONFERENCES,
                                    dog_conf in POWER_CONFERENCES)
            if fav_power and not dog_power:
                bits.append(
                    f"{model_fav} ({fav_conf}) is the power-conference name "
                    f"here against {model_dog} out of the {dog_conf} — a "
                    "step down in competition.")
            elif dog_power and not fav_power:
                bits.append(
                    f"{model_dog} ({dog_conf}) is the bigger name on paper, "
                    f"but the model still likes {model_fav} ({fav_conf}) "
                    "here.")

    if on_model_side:
        if rating is not None:
            talent = (f"{model_fav} grades out about {abs(rating):.0f} "
                     f"points better than {model_dog} on talent")
            if hfa is not None:
                talent += (f", and {home} gets the usual home-field bump "
                          f"of about {hfa:+.1f} on top of that")
            bits.append(talent + ".")
        elif hfa is not None:
            bits.append(f"{home} gets the usual home-field bump of about "
                        f"{hfa:+.1f} points.")

    if family == "ridge_cross_division":
        basis = bases.get(str(p.get("forecast_sd_basis")) or "", {})
        if _finite(basis.get("mean_residual")):
            bias = abs(float(basis["mean_residual"]))
            if on_model_side:
                bits.append(
                    f"One thing to flag: in games like this historically, "
                    f"the FBS side has actually beaten this projection by "
                    f"about {bias:.1f} more points on average — a known "
                    "bias we haven't corrected for, so if anything this "
                    "margin runs a little conservative.")
            else:
                bits.append(
                    f"One thing to flag against this side: in games like "
                    f"this historically, the FBS side has beaten this "
                    f"projection by about {bias:.1f} more points on "
                    "average, on top of the margin already priced in here "
                    "— a known bias we haven't corrected for, and it cuts "
                    "against the points-taking side, not for it.")
    return bits


def casual_rationale(p: Dict[str, Any], call: Dict[str, Any],
                     conferences: Dict[str, str],
                     bases: Dict[str, dict]) -> str:
    """`casual_rationale_parts` as one line, for anything that wants it flat."""
    return " ".join(casual_rationale_parts(p, call, conferences, bases))


def default_out_dir() -> str:
    """The vault's picks/ folder when the repo sits inside the vault, else
    data/picks/ in the repo. Never data/site -- this page is not published."""
    vault = os.path.join(os.path.dirname(ROOT), "picks")
    return vault if os.path.isdir(vault) else os.path.join(DATA, "picks")


def kickoff_et(kickoff_utc: str) -> str:
    ts = pd.to_datetime(kickoff_utc, utc=True).tz_convert(
        cfb_calendar.PUBLIC_TIMEZONE)
    return ts.strftime("%a %b %d %I:%M%p ET").replace(" 0", " ")


def load_inputs(season: int, week: int, artifact: Optional[str] = None,
                insights: Optional[str] = None) -> tuple:
    art = artifact or os.path.join(DATA, f"predictions_{season}_wk{week:02d}.csv")
    ins = insights or os.path.join(DATA, f"insights_{season}_wk{week:02d}.json")
    if not os.path.exists(art):
        raise SystemExit(f"no published artifact at {art}")
    if not os.path.exists(ins):
        raise SystemExit(f"no insight cards at {ins}; run "
                         f"scripts/render_insights.py --season {season} "
                         f"--week {week} first")
    rows = FC.read_artifact(art)
    with open(ins) as fh:
        cards = {int(c["game_id"]): c for c in json.load(fh)["cards"]}
    with open(os.path.join(DATA, "forecast_sd_basis.json")) as fh:
        bases = {b["basis_id"]: b for b in json.load(fh)["bases"]}
    return rows, cards, bases


def load_ledger(season: int, week: int, path: Optional[str] = None
                ) -> Dict[int, dict]:
    """The frozen market leans for the week, by game id, or {}.

    Read from `data/board_ledger.csv` -- the first-write-wins ledger whose
    lines were frozen before kickoff -- never from the live lines cache, so a
    spread shown here is the one the lean was actually recorded against.
    """
    path = path or os.path.join(DATA, "board_ledger.csv")
    if not os.path.exists(path):
        return {}
    frame = pd.read_csv(path)
    frame = frame[(frame["season"] == season) & (frame["week"] == week)]
    out = {}
    for row in frame.to_dict("records"):
        out[int(row["game_id"])] = {
            "ats_side": str(row["side_raw"]),
            "ats_p": float(row["p_cover_raw"]),
            "ats_p_adj": float(row["p_cover_adj"]),
            "ats_line_home": float(row["line_taken_home"]),
            "ats_book": str(row.get("book", "")),
            "ats_taken_at": str(row.get("taken_at_utc", "")),
        }
    return out


#: Above this market spread the registered lean is inside the pre-registered
#: scale question (model = 0.807 x market + 0.92, 2026-09-02): the model runs
#: ~19% compressed against the market, which mechanically produces an
#: underdog lean on every large spread. Flagged, never hidden.
LARGE_SPREAD = 14.0


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _why_text(card: Dict[str, Any]) -> str:
    parts = []
    for comp in card.get("why") or []:
        parts.append(f"{comp['label']} {comp['points']:+.1f}")
    return "; ".join(parts)


def pick_row(row: Dict[str, Any], card: Dict[str, Any],
             bases: Dict[str, dict], fbs: set,
             ledger: Optional[Dict[int, dict]] = None) -> Dict[str, Any]:
    """One game -> one line of the page. Refused games keep their reason."""
    home, away = str(row["home"]), str(row["away"])
    lean = (ledger or {}).get(int(row["game_id"]), {})
    out = {
        "ats_side": lean.get("ats_side"), "ats_p": lean.get("ats_p"),
        "ats_p_adj": lean.get("ats_p_adj"),
        "ats_line_home": lean.get("ats_line_home"),
        "ats_book": lean.get("ats_book"),
        "ats_large_spread": (None if not lean else
                             abs(lean["ats_line_home"]) > LARGE_SPREAD),
        "game_id": int(row["game_id"]),
        "kickoff_utc": row["kickoff_utc"],
        "kickoff_et": kickoff_et(row["kickoff_utc"]),
        "public_date": row["public_date"],
        "away": away, "home": home,
        "neutral": bool(row["neutral_site"]),
        "status": row["status"],
        "home_is_fbs": home in fbs, "away_is_fbs": away in fbs,
        "model_family": row.get("model_family"),
        "forecast_sd_basis": row.get("forecast_sd_basis"),
        "why_components": card.get("why") or [],
        "pick": None, "p_pick": None, "margin_pick": None,
        "ci80_lo": None, "ci80_hi": None, "ci95_lo": None, "ci95_hi": None,
        "sd": None, "rationale": "", "caution": "", "result": "",
    }
    caution = card.get("caution") or {}
    out["caution"] = caution.get("text", "")
    if row["status"] not in FC.FORECAST_BEARING or not _finite(
            row.get("projected_home_margin")):
        out["rationale"] = f"Not forecast: {row.get('refusal_reason') or caution.get('detail') or 'declined'}"
        return out
    mean = float(row["projected_home_margin"])
    sd = float(row["forecast_margin_sd"])
    p_home = float(row["p_home_win"])
    if mean >= 0:
        pick, p_pick, margin = home, p_home, mean
    else:
        pick, p_pick, margin = away, 1.0 - p_home, -mean
    out.update({
        "pick": pick, "p_pick": round(p_pick, 3),
        "margin_pick": round(margin, 1), "sd": round(sd, 1),
        "ci80_lo": round(margin - Z80 * sd, 1),
        "ci80_hi": round(margin + Z80 * sd, 1),
        "ci95_lo": round(margin - Z95 * sd, 1),
        "ci95_hi": round(margin + Z95 * sd, 1),
    })
    family = str(row.get("model_family"))
    bits = [FAMILY_LABEL.get(family, family)]
    why = _why_text(card)
    if why:
        bits.append(why)
    elif row["status"] == FC.Status.FROZEN.value:
        bits.append("frozen before kickoff; breakdown not recomputed")
    basis = bases.get(str(row.get("forecast_sd_basis")) or "", {})
    if family == "ridge_cross_division" and _finite(basis.get("mean_residual")):
        bias = float(basis["mean_residual"])
        bits.append(
            f"registered bias: out of sample the FBS side beat this "
            f"projection by {abs(bias):.1f} pts on average (n={basis.get('n')}); "
            "not corrected")
    out["rationale"] = " — ".join(bits)
    return out


def build(rows: pd.DataFrame, cards: Dict[int, dict], bases: Dict[str, dict],
          fbs: set, ledger: Optional[Dict[int, dict]] = None
          ) -> List[Dict[str, Any]]:
    out = []
    for row in rows.sort_values(["kickoff_utc", "game_id"]).to_dict("records"):
        home, away = str(row["home"]), str(row["away"])
        if home not in fbs and away not in fbs:
            continue  # not an FBS game
        card = cards.get(int(row["game_id"]))
        if card is None:
            raise SystemExit(f"game {row['game_id']} has no insight card; "
                             "the cards and the artifact are out of step")
        out.append(pick_row(row, card, bases, fbs, ledger))
    return out


def _fmt_p(p: Optional[float]) -> str:
    """Whole percent in the middle; one decimal at the ends, so a 0.9986
    does not print as a certainty."""
    if p is None:
        return ""
    return f"{100 * p:.1f}%" if (p > 0.99 or p < 0.01) else f"{100 * p:.0f}%"


def render_markdown(picks: List[Dict[str, Any]], season: int, week: int,
                    rows: pd.DataFrame, bases: Dict[str, dict]) -> str:
    forecast = [p for p in picks if p["pick"]]
    refused = [p for p in picks if not p["pick"]]
    generated = sorted(set(rows["generated_at_utc"].dropna().astype(str)))
    snapshot = sorted(set(rows["snapshot_id"].dropna().astype(str)))
    early = bases.get("early_ridge_wk1_3_2021_2025", {})
    xd = bases.get("cross_division_ridge_wk1_15_2021_2025", {})

    lines = [
        f"# {season} week {week} — projected winner, every FBS game",
        "",
        f"Source: `data/predictions_{season}_wk{week:02d}.csv` (contract v2) and "
        f"`data/insights_{season}_wk{week:02d}.json`. Artifact generated "
        f"{', '.join(generated)}; snapshot {', '.join(snapshot)}. "
        "Rendered by `scripts/render_picks.py`; nothing here is recomputed.",
        "",
        "**Paper-only research output.** A projected winner is the side the "
        "model's own predictive distribution puts ahead; the probability is "
        "read off that same distribution, and so are the intervals. This is "
        "a private vault note and may use the word *pick*; the published "
        "card does not, on purpose. No forecast, row or lean here is a bet, "
        "an edge or a recommendation.",
        "",
        "## How to read it",
        "",
        "- **Pick** — the projected winner. **p** — its win probability from "
        "Phi(margin / SD).",
        "- **Margin** — projected margin from the pick's side. **80% / 95%** — "
        "predictive intervals on that margin (mean ± 1.28 SD, ± 1.96 SD). A "
        "negative lower bound means the interval includes the pick losing.",
        f"- FBS-vs-FBS games use the registered early-season SD "
        f"{early.get('forecast_margin_sd')} (n={early.get('n')}, weeks 1-3, "
        "2021-2025) in weeks 0-3, and the simulation's own per-game SD from "
        "week 4.",
        f"- FBS-vs-FCS games use the cross-division route registered "
        f"2026-09-02: SD {xd.get('forecast_margin_sd')} (n={xd.get('n')}), "
        f"with a recorded, uncorrected bias of {xd.get('mean_residual')} pts "
        "(the FBS side has beaten the projection by about that much on "
        "average). The FCS opponent is projected as the pooled FCS "
        "pseudo-team, not as itself — so a strong FCS programme is "
        "under-rated here and a weak one over-rated.",
        "- **Result** column is blank until graded; grade straight-up against "
        "the pick, never against the interval.",
        "",
        f"**Coverage:** {len(picks)} FBS games in the artifact — "
        f"{len(forecast)} with a projected winner, {len(refused)} not "
        "forecast (listed at the end with the reason).",
        "",
    ]

    def table(items: List[Dict[str, Any]], with_date: bool = True) -> List[str]:
        head = ["Kickoff (ET)", "Game", "Pick", "p", "Margin", "80%", "95%",
                "Route", "Result"]
        out = ["| " + " | ".join(head) + " |",
               "|" + "|".join("---" for _ in head) + "|"]
        for p in items:
            site = " (N)" if p["neutral"] else ""
            game = f"{p['away']} @ {p['home']}{site}"
            route = {"ridge_preseason_prior": "FBS ridge",
                     "kalman_montecarlo": "FBS sim",
                     "ridge_cross_division": "vs FCS pool"}.get(
                         str(p["model_family"]), str(p["model_family"]))
            out.append("| " + " | ".join([
                p["kickoff_et"], game, p["pick"], _fmt_p(p["p_pick"]),
                f"{p['margin_pick']:+.1f}",
                f"{p['ci80_lo']:+.1f} … {p['ci80_hi']:+.1f}",
                f"{p['ci95_lo']:+.1f} … {p['ci95_hi']:+.1f}",
                route, p["result"] or ""]) + " |")
        return out

    if forecast:
        strongest = sorted(forecast, key=lambda p: (-p["p_pick"], p["game_id"]))[:10]
        closest = sorted(forecast, key=lambda p: (p["p_pick"], p["game_id"]))[:10]
        lines += ["## Strongest calls (top 10 by win probability)", ""]
        lines += table(strongest)
        lines += ["", "## Closest calls (10 nearest to a coin flip)", ""]
        lines += table(closest)
        lines += ["", "## Every game, in kickoff order", ""]
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for p in forecast:
            by_date.setdefault(p["public_date"], []).append(p)
        for date in sorted(by_date):
            lines += [f"### {date}", ""]
            lines += table(by_date[date])
            lines += [""]
        lines += ["## Rationale and caution, per game", ""]
        for p in forecast:
            site = " (neutral site)" if p["neutral"] else ""
            lines += [
                f"**{p['away']} @ {p['home']}{site}** — {p['kickoff_et']} · "
                f"game {p['game_id']}",
                "",
                f"- Pick **{p['pick']}** ({_fmt_p(p['p_pick'])}), projected by "
                f"{p['margin_pick']:+.1f}; 80% {p['ci80_lo']:+.1f} … "
                f"{p['ci80_hi']:+.1f}, 95% {p['ci95_lo']:+.1f} … "
                f"{p['ci95_hi']:+.1f} (SD {p['sd']}).",
                f"- Why: {p['rationale']}.",
                f"- Caution: {p['caution']}",
                "",
            ]
    if refused:
        lines += ["## Not forecast", ""]
        for p in refused:
            lines += [f"- {p['kickoff_et']} — {p['away']} @ {p['home']}: "
                      f"{p['rationale']}"]
        lines += [""]
    return "\n".join(lines) + "\n"


def best_call(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The one call to show for a game: the frozen spread lean when the model
    holds it with MORE confidence than the straight-up winner, else the
    straight-up winner. Totals never appear: the early-season ridge publishes
    no total by contract, and no registered basis backs an over/under
    probability in any week yet, so there is no honest number to show."""
    if not p["pick"]:
        return None
    if p["ats_side"] and p["ats_p"] is not None and p["ats_p"] > p["p_pick"]:
        return {"kind": "spread", "pick": p["ats_side"], "p": p["ats_p"],
                "flag": bool(p["ats_large_spread"]), "p_adj": p["ats_p_adj"]}
    return {"kind": "straight_up", "pick": f"{p['pick']} (straight up)",
            "p": p["p_pick"], "flag": False, "p_adj": None}


def render_compact(picks: List[Dict[str, Any]], season: int, week: int,
                   rows: pd.DataFrame) -> str:
    """Game / pick / confidence, one line per game, most confident first."""
    calls = [(p, best_call(p)) for p in picks]
    called = [(p, c) for p, c in calls if c]
    called.sort(key=lambda pc: (-pc[1]["p"], pc[0]["game_id"]))
    n_spread = sum(1 for _, c in called if c["kind"] == "spread")
    n_flag = sum(1 for _, c in called if c["flag"])
    n_lines = sum(1 for p in picks if p["ats_side"])
    snapshot = sorted(set(rows["snapshot_id"].dropna().astype(str)))
    lines = [
        f"# {season} week {week} — picks",
        "",
        f"{len(called)} calls: {len(called) - n_spread} straight up, "
        f"{n_spread} against the spread (shown where the model holds the "
        f"spread lean with more confidence than the winner; {n_lines} games "
        "had a frozen line). Over/under: none — the early-season route "
        "publishes no total by contract, and no registered basis backs an "
        "over/under probability yet. Paper-only; private vault note. "
        f"Snapshot {', '.join(snapshot)}; spread leans from "
        "`data/board_ledger.csv` (frozen before kickoff).",
        "",
        "| Game | Pick | Confidence |",
        "|---|---|---|",
    ]
    for p, c in called:
        site = " (N)" if p["neutral"] else ""
        conf = _fmt_p(c["p"]) + (r"\*" if c["flag"] else "")
        lines.append(f"| {p['away']} @ {p['home']}{site} · "
                     f"{p['kickoff_et']} | {c['pick']} | {conf} |")
    if n_flag:
        lines += [
            "",
            rf"\* {n_flag} spread leans sit on a market spread over "
            f"{LARGE_SPREAD:.0f} points. Those are inside the open, "
            "pre-registered scale question (the model runs ~19% compressed "
            "against the market, which produces an underdog lean on every "
            "large spread — one bias, reported many times). The confidence "
            "shown is the registered number; the scale-adjusted diagnostic "
            "for each is in the CSV (`ats_p_adj`). Treat these as one "
            "position, not many.",
        ]
    refused = [p for p in picks if not p["pick"]]
    if refused:
        lines += ["", "Not called: " + "; ".join(
            f"{p['away']} @ {p['home']}" for p in refused) + "."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# The value page: model probability against the PRICE, not against 50%
# --------------------------------------------------------------------------- #
#: Standard spread juice. The ledger froze lines, not prices; -110 is the
#: assumption the break-even below is built on and is stated on the page.
SPREAD_PRICE = -110.0

#: Curtis's rule (2026-09-03): a play the model gives less than an even
#: chance is cut, whatever its expected value. A 26% long shot at 40-to-1
#: has a large expected profit and is still a pick that loses three times
#: in four; this page is for picks, not for lottery tickets.
MIN_PLAY_PROB = 0.5

#: Units as confidence (Curtis, 2026-09-03). Sized on the SMALLER of the two
#: expected profits -- the model's own and the scale-adjusted one -- because
#: the scale question is open and a pick that only works under one view has
#: not earned the other's confidence. Capped at 3: nothing here has been
#: graded yet. A pick negative under either view gets 0 and is listed as a
#: pass.
UNIT_LADDER = ((40.0, 3.0), (25.0, 2.0), (10.0, 1.0), (0.0, 0.5))

#: Curtis's rule (2026-09-03): the list he actually acts on is not "every
#: call" -- it's the plays that clear BOTH bars at once. A longshot value
#: swap at 51% is not confidence; a -20000 moneyline "lock" worth 0 units is
#: not value. Neither belongs on the list he bets from and grades against
#: real outcomes -- that's the only way to learn whether the pricing logic
#: (the EV/units layer sitting on top of the model) is any good.
TOP_MIN_CONFIDENCE = 0.65
TOP_MIN_UNITS = 1.0


def units_for(play: Dict[str, Any]) -> float:
    """Units from the conservative (smaller) expected profit per $100."""
    evs = [play["ev"]]
    if play.get("ev_adj") is not None:
        evs.append(play["ev_adj"])
    worst = 100.0 * min(evs)
    if worst <= 0:
        return 0.0
    for floor, units in UNIT_LADDER:
        if worst >= floor:
            return units
    return 0.0
PROVIDER_PRIORITY = ("DraftKings", "Draft Kings", "Bovada", "ESPN Bet", "consensus")


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def payout(american: float) -> float:
    """Net profit per 1 unit staked at American odds."""
    american = float(american)
    return american / 100.0 if american > 0 else 100.0 / abs(american)


def implied(american: float) -> float:
    """Break-even probability at these odds (vig included -- that IS the hurdle)."""
    return 1.0 / (1.0 + payout(american))


def expected_value(p: float, american: float) -> float:
    """EV per 1 unit staked: p * profit - (1 - p) * 1."""
    return p * payout(american) - (1.0 - p)


def load_scale(season: int, week: int, path: Optional[str] = None
               ) -> Optional[dict]:
    """The registered model-vs-market scale diagnostic from the week's board
    (slope, intercept fitted on the FBS-vs-FBS games), or None. Reported
    beside the registered number, never substituted for it."""
    path = path or os.path.join(DATA, f"board_{season}_wk{week:02d}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        diag = (json.load(fh).get("scale_diagnostic") or {})
    slope, intercept = diag.get("model_vs_market_slope"), diag.get(
        "model_vs_market_intercept")
    if not (_finite(slope) and _finite(intercept)) or float(slope) <= 0:
        return None
    return {"slope": float(slope), "intercept": float(intercept)}


def rescale(mean: float, scale: Optional[dict]) -> Optional[float]:
    """The model margin put on the market's scale by the registered fit."""
    if scale is None:
        return None
    return (mean - scale["intercept"]) / scale["slope"]


def load_market(season: int, week: int, path: Optional[str] = None
                ) -> tuple:
    """One cached market row per game (spread + moneylines) and the cache's
    own timestamp. These are the LIVE cached prices, not frozen leans; the
    page labels them as such and says when the cache was written."""
    path = path or os.path.join(DATA, "cfbd", f"lines_{season}.parquet")
    if not os.path.exists(path):
        return {}, None
    lines = pd.read_parquet(path)
    lines = lines[lines["week"] == week]
    as_of = pd.Timestamp(os.path.getmtime(path), unit="s", tz="UTC").strftime(
        "%Y-%m-%dT%H:%MZ")
    rank = {name: i for i, name in enumerate(PROVIDER_PRIORITY)}
    out = {}
    for gid, sub in lines.groupby("game_id"):
        sub = sub.assign(_rank=sub["provider"].map(
            lambda name: rank.get(str(name), len(rank)))).sort_values("_rank")
        row = sub.iloc[0]
        out[int(gid)] = {
            "provider": str(row["provider"]),
            "spread": (None if pd.isna(row["spread"]) else float(row["spread"])),
            "home_ml": (None if pd.isna(row["home_moneyline"])
                        else float(row["home_moneyline"])),
            "away_ml": (None if pd.isna(row["away_moneyline"])
                        else float(row["away_moneyline"])),
        }
    return out, as_of


def value_plays(p: Dict[str, Any], row: Dict[str, Any], market: dict,
                scale: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Every priced play for one game with its EV, from the model's own
    distribution. Spread: the frozen ledger lean when one exists (registered
    p_cover), else the cached spread (unfrozen, labelled). Moneyline: the
    cached price for the side the model favours AND the other side -- value
    can sit on the dog."""
    if not p["pick"] or not _finite(row.get("projected_home_margin")):
        return []
    mean = float(row["projected_home_margin"])
    sd = float(row["forecast_margin_sd"])
    p_home = float(row["p_home_win"])
    home, away = p["home"], p["away"]
    mean_adj = rescale(mean, scale)
    p_home_adj = None if mean_adj is None else _phi(mean_adj / sd)
    plays = []
    # ---- spread ----
    if p["ats_side"]:
        line_home = float(p["ats_line_home"])
        frozen = True
        p_cover = float(p["ats_p"])
        side = p["ats_side"]
    elif market and market.get("spread") is not None:
        line_home = float(market["spread"])      # quoted from the HOME side
        frozen = False
        market_margin = -line_home
        edge = mean - market_margin
        p_cover = _phi(edge / sd) if edge >= 0 else _phi(-edge / sd)
        side = (f"{home} {line_home:+.1f}" if edge >= 0
                else f"{away} {-line_home:+.1f}")
    else:
        line_home, frozen, p_cover, side = None, None, None, None
    if side:
        if frozen and p["ats_p_adj"] is not None:
            p_cover_adj = float(p["ats_p_adj"])
        elif mean_adj is not None:
            e_adj = mean_adj - (-line_home)
            # the adjusted view may sit on the OTHER side; then the taken
            # side's cover probability is the complement
            same_side = (e_adj >= 0) == (side.startswith(home))
            p_cover_adj = _phi(abs(e_adj) / sd) if same_side else _phi(-abs(e_adj) / sd)
        else:
            p_cover_adj = None
        plays.append({
            "kind": "spread", "pick": side, "p": p_cover,
            "price": SPREAD_PRICE, "break_even": implied(SPREAD_PRICE),
            "ev": expected_value(p_cover, SPREAD_PRICE),
            "ev_adj": (None if p_cover_adj is None
                       else expected_value(p_cover_adj, SPREAD_PRICE)),
            "frozen": frozen, "flag": abs(line_home) > LARGE_SPREAD,
            "p_adj": p_cover_adj,
        })
    # ---- moneyline, both sides ----
    if market:
        for team, prob, prob_adj, price in (
                (home, p_home, p_home_adj, market.get("home_ml")),
                (away, 1.0 - p_home,
                 None if p_home_adj is None else 1.0 - p_home_adj,
                 market.get("away_ml"))):
            if price is None:
                continue
            plays.append({
                "kind": "moneyline", "pick": f"{team} ML {price:+.0f}",
                "p": prob, "price": price, "break_even": implied(price),
                "ev": expected_value(prob, price),
                "ev_adj": (None if prob_adj is None
                           else expected_value(prob_adj, price)),
                "frozen": False,
                "flag": (line_home is not None and abs(line_home) > LARGE_SPREAD),
                "p_adj": prob_adj,
            })
    return plays


def best_value_play(plays: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The single highest-EV play on a game that clears the profitability
    bar: at least `MIN_PLAY_PROB` and a positive registered expected value.
    None if nothing on the game clears it -- that is the honest answer on
    some games, not a case to work around. The one selection rule behind
    both the value page's ranking and the next-N-days page's value swap;
    factored out so they cannot drift into two different bars."""
    candidates = [x for x in plays if x["ev"] > 0 and x["p"] >= MIN_PLAY_PROB]
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["ev"], -x["break_even"]))


def top_plays(called: List[tuple], min_p: float = TOP_MIN_CONFIDENCE,
             min_units: float = TOP_MIN_UNITS) -> List[tuple]:
    """The `(pick, call)` pairs from a `render_upcoming` pass that clear
    BOTH bars: real confidence (the play's own probability -- the side
    actually being bet, not the model's headline number when this swapped
    to a value play) AND real value (units sized off actual expected profit
    against the price on offer, not just "positive"). This is what filters
    out the two things that are not what Curtis asked for: a longshot swap
    with value but low confidence, and a heavily-priced "lock" with
    confidence but no units. Sorted date, then confidence, then value --
    the three axes this is meant to track by."""
    picks = [(p, c) for p, c in called
            if c["units"] is not None and c["units"] >= min_units
            and c["p"] >= min_p]
    picks.sort(key=lambda pc: (pc[0]["public_date"], -pc[1]["p"],
                               -pc[1]["units"], pc[0]["game_id"]))
    return picks


def render_value(picks: List[Dict[str, Any]], rows: pd.DataFrame,
                 markets: Dict[int, dict], market_as_of: Optional[str],
                 season: int, week: int, scale: Optional[dict] = None) -> str:
    """Best play per game by expected value against the price, positive EV
    only, ranked. A 99% favourite at -10000 does not appear: there is no
    value in being right about something the price already says."""
    by_id = {int(r["game_id"]): r for r in rows.to_dict("records")}
    best = []
    n_priced_games = 0
    for p in picks:
        plays = value_plays(p, by_id[p["game_id"]], markets.get(p["game_id"]),
                            scale)
        if plays:
            n_priced_games += 1
        play = best_value_play(plays)
        if play is not None:
            best.append((p, play))
    best.sort(key=lambda pb: (-pb[1]["ev"], pb[0]["game_id"]))
    both = [(p, b) for p, b in best
            if b.get("ev_adj") is not None and b["ev_adj"] > 0]
    both.sort(key=lambda pb: (-min(pb[1]["ev"], pb[1]["ev_adj"]), pb[0]["game_id"]))
    n_ml = sum(1 for _, b in best if b["kind"] == "moneyline")
    n_unfrozen = sum(1 for _, b in best if b["frozen"] is False and b["kind"] == "spread")
    n_flag = sum(1 for _, b in best if b["flag"])
    lines = [
        f"# {season} week {week} — value",
        "",
        "One pick per game, only where the model gives it at least a "
        f"{100 * MIN_PLAY_PROB:.0f}% chance AND it pays better than that chance "
        "deserves. Best pick per game, ranked by expected profit.",
        "",
        "**How to read it.** *Chance it hits* is the model's probability the "
        "pick wins (covers, for a spread). *Needs to hit* is how often a pick "
        "at that price has to win just to break even — a −110 spread needs "
        f"{100 * implied(SPREAD_PRICE):.1f}%, a +150 underdog needs 40%. The "
        "gap between those two is the whole case for the pick. *Expected "
        "profit per $100* is what that gap is worth on average: chance of "
        "winning × payout, minus the chance of losing × $100. It is an "
        "average over many bets, not what happens on one.",
        "",
        "Spread picks assume a standard −110 price. Moneylines and any spread "
        "marked (u) use the cached prices as of "
        f"{market_as_of or 'unknown'} — not frozen; the frozen spreads come "
        "from `data/board_ledger.csv`. Paper-only; private vault note; gate 7 "
        "has not run and nothing here is staked.",
        "",
        f"{len(best)} picks from {n_priced_games} priced games with a market "
        f"({n_ml} moneyline, {len(best) - n_ml} spread, {n_unfrozen} of the "
        f"spreads unfrozen). Cut: every play the model gives under "
        f"{100 * MIN_PLAY_PROB:.0f}%. Over/under: none — no total is published "
        "by contract in weeks 0-3 and no registered basis backs an over/under "
        "probability yet.",
        "",
    ]

    def _ev(value: Optional[float]) -> str:
        return "n/a" if value is None else f"${100 * value:+.0f}"

    def _units(b) -> str:
        u = units_for(b)
        return "pass" if u == 0 else (f"{u:g}u")

    def table(items: List[tuple]) -> List[str]:
        out = ["| Game | Pick | Units | Chance it hits | Needs to hit | "
               "Expected profit per $100 | Same, scale-adjusted |",
               "|---|---|---|---|---|---|---|"]
        for p, b in items:
            site = " (N)" if p["neutral"] else ""
            tags = (("(u) " if b["frozen"] is False and b["kind"] == "spread"
                     else "") + (r"\* " if b["flag"] else ""))
            out.append(
                f"| {p['away']} @ {p['home']}{site} · {p['kickoff_et']} | "
                f"{tags}{b['pick']} | {_units(b)} | {_fmt_p(b['p'])} | "
                f"{_fmt_p(b['break_even'])} | {_ev(b['ev'])} | "
                f"{_ev(b.get('ev_adj'))} |")
        return out

    # ---- by day, units first: the page a reader actually uses ----------
    by_day: Dict[str, List[tuple]] = {}
    for p, b in best:
        by_day.setdefault(p["public_date"], []).append((p, b))
    lines += [
        "## Picks by day, sized in units",
        "",
        "Units are confidence: sized on the SMALLER of the two expected "
        "profits (the model's own and the scale-adjusted one), so a pick that "
        "only works under one view is not trusted. 3u = both views say $40+ "
        "per $100; 2u = $25+; 1u = $10+; 0.5u = anything positive; pass = "
        "negative under either view. Capped at 3u because nothing has been "
        "graded yet. Starred picks are one opinion repeated (see below) -- "
        "size the group, not each line.",
        "",
    ]
    for day in sorted(by_day):
        items = sorted(by_day[day],
                       key=lambda pb: (-units_for(pb[1]), -pb[1]["ev"],
                                       pb[0]["game_id"]))
        total = sum(units_for(b) for _, b in items)
        lines += [f"### {day} — {total:g} units across "
                  f"{sum(1 for _, b in items if units_for(b) > 0)} picks", ""]
        lines += table(items)
        lines += [""]

    if scale:
        lines += [
            "## Picks that survive both views",
            "",
            "Profitable under the model's own numbers AND after the "
            "scale adjustment (the model's margins are known to run about "
            "19% smaller than the market's; this view stretches them back "
            f"out — slope {scale['slope']:.3f}, intercept "
            f"{scale['intercept']:.2f}, fitted on this week's FBS-vs-FBS "
            "games, so for FBS-vs-FCS it is an extrapolation). Ranked by "
            "the smaller of the two profits. The adjusted view is a "
            "diagnostic shown beside the registered number, never in place "
            "of it.",
            "",
        ]
        lines += table(both) if both else ["(none)"]
        lines += ["", "## Every pick, model's own numbers", ""]
    lines += table(best)
    notes = []
    if n_flag:
        notes.append(
            rf"\* {n_flag} picks sit on a market spread over {LARGE_SPREAD:.0f} "
            "points. The model's margins run about 19% smaller than the "
            "market's, which makes it lean the underdog on every big spread "
            "for the same reason — so these are one opinion repeated, not "
            "independent picks. Week-1 results will say whether that opinion "
            "is right (result_raw vs result_adj).")
    if n_unfrozen:
        notes.append(
            f"(u) {n_unfrozen} spread picks use the cached line rather than a "
            "frozen ledger lean (the ledger holds only the 43 FBS-vs-FBS "
            "games recorded 2026-09-02T01:00Z). To grade them honestly, "
            "freeze them first: `python3 scripts/board_ledger.py --record`.")
    if notes:
        lines += [""] + notes
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# The upcoming page: one call per game, sorted by date then confidence
# --------------------------------------------------------------------------- #

def call_for_game(p: Dict[str, Any], row: Dict[str, Any],
                  market: Optional[dict],
                  scale: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """One call for one game: the model's most-confident side (a spread pick
    -- frozen ledger lean, else the cached market line, standard -110 either
    way -- when its cover probability beats the straight-up win probability,
    else the projected winner as a moneyline call at the cached price) UNLESS
    that call is a pass on units (priced in, no value against the price), in
    which case this reaches for the best play anywhere on the game --
    including the other side -- that clears the value bar (`best_value_play`:
    at least 50% and positive EV), and marks it `swapped`. Only when nothing
    on the game clears that bar does a pass stand, honestly, rather than
    being papered over with a play that does not actually pay. No over/under:
    the early-season route publishes no total by contract, and no registered
    basis backs an over/under probability in any week yet -- so none is
    shown, rather than made up.

    Carries the same expected-value pair every value play does (registered
    and scale-adjusted) and the units they size to, so this page does not
    need a second pass through `value_plays` to answer "is this worth
    anything" -- units are None (n/a) only when there is no price to check
    any pick against at all."""
    if not p["pick"]:
        return None
    plays = value_plays(p, row, market, scale)
    favoured = [x for x in plays
               if x["kind"] == "spread"
               or x["pick"].startswith(f"{p['pick']} ML")]
    confident = max(favoured, key=lambda x: x["p"]) if favoured else None
    swapped = False
    chosen = confident
    if confident is not None:
        ev = confident.get("ev")
        units = None if ev is None else units_for(
            {"ev": ev, "ev_adj": confident.get("ev_adj")})
        if units == 0.0:
            alt = best_value_play(plays)
            if alt is not None and alt["pick"] != confident["pick"]:
                chosen, swapped = alt, True
    elif plays:
        chosen = best_value_play(plays)
        swapped = chosen is not None
    if chosen is None:
        return {"kind": "moneyline", "pick": f"{p['pick']} ML", "p": p["p_pick"],
                "price": None, "flag": False, "frozen": None,
                "has_market": False, "ev": None, "ev_adj": None, "units": None,
                "swapped": False}
    ev, ev_adj = chosen.get("ev"), chosen.get("ev_adj")
    return {"kind": chosen["kind"], "pick": chosen["pick"], "p": chosen["p"],
            "price": chosen.get("price"), "flag": chosen["flag"],
            "frozen": chosen.get("frozen"), "has_market": True,
            "ev": ev, "ev_adj": ev_adj,
            "units": None if ev is None else units_for(
                {"ev": ev, "ev_adj": ev_adj}),
            "swapped": swapped}


def _render_pick_blocks(pairs: List[tuple], conferences: Dict[str, str],
                        bases: Dict[str, dict]) -> List[str]:
    """One block per game -- a day heading when the day changes, the game
    and kickoff, the pick with its units and chance on one line, then the
    rationale as its own spaced paragraphs, one sentence per paragraph.
    Curtis (2026-09-03): the table cell was cramming the rationale into a
    single wrapped line; he wants it wide, spaced, and read in full. Both
    the top-plays list and the full-window list render through here so
    they cannot drift into two formats."""
    lines = []
    last_day = None
    for p, c in pairs:
        if p["public_date"] != last_day:
            last_day = p["public_date"]
            day_label = pd.Timestamp(last_day).strftime("%A %b %d").replace(" 0", " ")
            lines += [f"### {day_label}", ""]
        site = " (N)" if p["neutral"] else ""
        tags = (("(value) " if c.get("swapped") else "")
               + ("(u) " if c["kind"] == "spread" and c["frozen"] is False else "")
               + (r"\* " if c["flag"] else "")
               + ("(no market) " if not c["has_market"] else ""))
        units = "n/a" if c["units"] is None else (
            "pass" if c["units"] == 0 else f"{c['units']:g}u")
        lines.append(f"#### {p['away']} @ {p['home']}{site} · {p['kickoff_et']}")
        lines.append("")
        lines.append(f"**Pick: {tags}{c['pick']}** — {units} — "
                     f"{_fmt_p(c['p'])} chance it hits")
        lines.append("")
        for part in casual_rationale_parts(p, c, conferences, bases):
            lines += [part, ""]
    return lines


def render_upcoming(picks: List[Dict[str, Any]], rows: pd.DataFrame,
                    markets: Dict[int, dict], market_as_of: Optional[str],
                    season: int, week: int, days: int = 3,
                    today: Optional[str] = None,
                    scale: Optional[dict] = None,
                    conferences: Optional[Dict[str, str]] = None,
                    bases: Optional[Dict[str, dict]] = None,
                    min_confidence: float = TOP_MIN_CONFIDENCE,
                    min_units: float = TOP_MIN_UNITS) -> str:
    """Every FBS game kicking off in the next `days` calendar days (public,
    America/New_York dates), one call per game, grouped by date and sorted
    by confidence within the day -- team + market (moneyline or spread),
    the units it is worth, and the rationale behind it in plain English.

    Leads with a second, curated table -- the plays that clear BOTH the
    confidence floor and the value floor at once (`top_plays`), which is
    the actionable list: not a longshot swap with value and not a priced-in
    "lock" with none. That's the list meant to be logged with
    `scripts/bet_log.py` and graded once games are final."""
    conferences, bases = (conferences or {}), (bases or {})
    by_id = {int(r["game_id"]): r for r in rows.to_dict("records")}
    start = (pd.Timestamp(today) if today else
            pd.Timestamp.now(tz=cfb_calendar.PUBLIC_TIMEZONE)).normalize()
    window = {(start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(days)}
    called = []
    for p in picks:
        if p["public_date"] not in window:
            continue
        call = call_for_game(p, by_id[p["game_id"]], markets.get(p["game_id"]),
                             scale)
        if call:
            called.append((p, call))
    called.sort(key=lambda pc: (pc[0]["public_date"], -pc[1]["p"],
                                pc[0]["game_id"]))
    top = top_plays(called, min_p=min_confidence, min_units=min_units)
    not_called = [p for p in picks if p["public_date"] in window and not p["pick"]]
    n_flag = sum(1 for _, c in called if c["flag"])
    n_no_market = sum(1 for _, c in called if not c["has_market"])
    lo = min(window) if window else (today or "")
    hi = max(window) if window else (today or "")
    lines = [
        f"# {season} week {week} — picks, {lo} to {hi}",
        "",
        f"{len(called)} calls across {len(window)} days, one per game, "
        "sorted by date then by how confident the model is. Spread picks "
        "use the frozen line where one was recorded, else the current "
        "cached market line (both assume standard -110 juice); moneyline "
        "picks use the cached price when the book has one. Over/under: "
        "none -- the early-season route publishes no total by contract, "
        "and no registered basis backs an over/under probability in any "
        "week yet. Paper-only; private vault note; nothing here is staked.",
        "",
        "## The list: confidence AND value, not either alone",
        "",
        f"{len(top)} of {len(called)} calls clear both bars: at least "
        f"{min_confidence:.0%} on the side actually being bet (its own "
        "chance, not the model's headline number when this swapped to a "
        "value play) AND at least "
        f"{min_units:g} unit of sized value against the price actually on "
        "offer. Not a longshot that happens to have value, not a "
        "'guaranteed' favourite worth nothing at its price -- both, "
        "together. Sorted date, then confidence, then value. This is the "
        "list meant to be logged with `scripts/bet_log.py --add` and "
        "graded with `--grade` once the games are final -- that's the only "
        "way to learn whether the pricing logic on top of the model (the "
        "EV/units layer, not the model's own win probability) is actually "
        "any good.",
        "",
    ]
    if top:
        lines += _render_pick_blocks(top, conferences, bases)
    else:
        lines += ["*Nothing this window clears both bars.*"]
    lines += [
        "",
        "## Every call this window",
        "",
        "**Units** are confidence sized against the price, not just the "
        "win probability: sized on the smaller of the model's own expected "
        "profit per $100 and the scale-adjusted view (the model's margins "
        "run about 19% smaller than the market's; see the value page for "
        "the fit), ladder 3u/2u/1u/0.5u, capped at 3u since nothing has "
        "been graded yet, and 'n/a' where there is no price to check the "
        "pick against at all. When the model's most-confident side is a "
        "pass (priced in, no value), this page reaches for the best play "
        "anywhere on that game -- including the other side -- that clears "
        "the bar (50%+ and positive EV), tagged **(value)**; if nothing on "
        "the game clears it, the pass stands rather than being papered "
        "over. **Rationale** always argues for the team actually being "
        "bet, not the model's own straight-up favourite when the two "
        "differ -- a wide spread on a mismatch is routinely a bet that the "
        "favourite doesn't cover, not that the underdog wins, and the "
        "rationale leads with that math (the model's own margin against "
        "the number on the board) before the background on why the number "
        "is that big. The talent gap, home-field number, money-game or "
        "power-conference framing, and the registered cross-division bias "
        "are the same facts either way -- nothing invented, just argued "
        "for the side actually on the ticket.",
        "",
    ]
    lines += _render_pick_blocks(called, conferences, bases)
    notes = []
    n_swapped = sum(1 for _, c in called if c.get("swapped"))
    if n_swapped:
        notes.append(
            f"(value) {n_swapped} picks are not the model's most-confident "
            "side on that game -- the confident side priced in with no "
            "value, so this shows whichever play on the game clears the "
            "50%+ and positive-EV bar instead. Read the confidence column "
            "literally: it is that play's own chance, not the model's "
            "headline win probability.")
    if n_flag:
        notes.append(
            rf"\* {n_flag} picks sit on a market spread over "
            f"{LARGE_SPREAD:.0f} points -- inside the open, pre-registered "
            "scale question (the model runs ~19% compressed against the "
            "market, which leans it toward the underdog on every large "
            "spread). One opinion repeated, not independent picks.")
    if n_no_market:
        notes.append(
            f"(no market) {n_no_market} picks have no cached line or price "
            "-- the confidence shown is the model's own win probability, "
            "not measured against anything a book is offering, and units "
            "are n/a for the same reason.")
    n_unfrozen_spread = sum(1 for _, c in called
                            if c["kind"] == "spread" and c["frozen"] is False)
    if n_unfrozen_spread:
        notes.append(
            f"(u) {n_unfrozen_spread} spread picks use the cached line "
            "rather than a frozen ledger lean; the market can move before "
            "kickoff on those.")
    if notes:
        lines += [""] + notes
    if not_called:
        lines += ["", "Not called: " + "; ".join(
            f"{p['away']} @ {p['home']}" for p in not_called) + "."]
    return "\n".join(lines) + "\n"


CSV_COLUMNS = ("game_id", "kickoff_et", "kickoff_utc", "public_date", "away",
               "home", "neutral", "status", "model_family", "pick", "p_pick",
               "margin_pick", "sd", "ci80_lo", "ci80_hi", "ci95_lo", "ci95_hi",
               "rationale", "caution", "result")


def write_outputs(picks: List[Dict[str, Any]], season: int, week: int,
                  out_dir: str, text: str,
                  compact_text: Optional[str] = None,
                  value_text: Optional[str] = None,
                  upcoming_text: Optional[str] = None,
                  upcoming_days: int = 3) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{season}_wk{week:02d}_every_fbs_game")
    md_path, csv_path = stem + ".md", stem + ".csv"
    with open(md_path, "w") as fh:
        fh.write(text)
    if compact_text is not None:
        with open(os.path.join(out_dir, f"{season}_wk{week:02d}_picks.md"),
                  "w") as fh:
            fh.write(compact_text)
    if value_text is not None:
        with open(os.path.join(out_dir, f"{season}_wk{week:02d}_value.md"),
                  "w") as fh:
            fh.write(value_text)
    if upcoming_text is not None:
        with open(os.path.join(
                out_dir, f"{season}_wk{week:02d}_next{upcoming_days}days.md"),
                  "w") as fh:
            fh.write(upcoming_text)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for p in picks:
            writer.writerow(p)
    return md_path, csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--auto", action="store_true",
                    help="season and week from the weekly loop's own "
                         "current_season_week() -- the week the loop is IN")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--insights", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--upcoming-days", type=int, default=3,
                    help="window for the next-N-days picks page (default 3)")
    ap.add_argument("--today", default=None,
                    help="override 'today' (public ET date, YYYY-MM-DD) for "
                         "the next-N-days window; default is the real clock")
    ap.add_argument("--min-confidence", type=float, default=TOP_MIN_CONFIDENCE,
                    help="confidence floor for the top-plays table on the "
                         f"next-N-days page (default {TOP_MIN_CONFIDENCE:g})")
    ap.add_argument("--min-units", type=float, default=TOP_MIN_UNITS,
                    help="units floor for the top-plays table on the "
                         f"next-N-days page (default {TOP_MIN_UNITS:g})")
    args = ap.parse_args()
    if args.auto:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "weekly_loop", os.path.join(ROOT, "scripts", "weekly_loop.py"))
        loop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loop)
        args.season, args.week, in_season = loop.current_season_week()
        if not in_season:
            print("[picks] offseason: nothing to render")
            return 0
    if args.season is None or args.week is None:
        ap.error("--season and --week are required unless --auto is given")

    rows, cards, bases = load_inputs(args.season, args.week, args.artifact,
                                     args.insights)
    fbs = PR.fbs_teams(args.season)
    ledger = load_ledger(args.season, args.week)
    picks = build(rows, cards, bases, fbs, ledger)
    text = render_markdown(picks, args.season, args.week, rows, bases)
    compact = render_compact(picks, args.season, args.week, rows)
    markets, market_as_of = load_market(args.season, args.week)
    scale = load_scale(args.season, args.week)
    value = render_value(picks, rows, markets, market_as_of,
                         args.season, args.week, scale=scale)
    upcoming = render_upcoming(picks, rows, markets, market_as_of,
                               args.season, args.week,
                               days=args.upcoming_days, today=args.today,
                               scale=scale, conferences=load_conferences(args.season),
                               bases=bases, min_confidence=args.min_confidence,
                               min_units=args.min_units)
    out_dir = args.out_dir or default_out_dir()
    md_path, csv_path = write_outputs(
        picks, args.season, args.week, out_dir, text, compact, value,
        upcoming, args.upcoming_days)
    n_pick = sum(1 for p in picks if p["pick"])
    print(f"[picks] {len(picks)} FBS games, {n_pick} with a projected winner, "
          f"{len(picks) - n_pick} not forecast")
    print(f"[picks] wrote {md_path}")
    print(f"[picks] wrote {csv_path}")
    print(f"[picks] wrote {os.path.join(out_dir, f'{args.season}_wk{args.week:02d}_picks.md')}")
    print(f"[picks] wrote {os.path.join(out_dir, f'{args.season}_wk{args.week:02d}_value.md')}")
    print(f"[picks] wrote {os.path.join(out_dir, f'{args.season}_wk{args.week:02d}_next{args.upcoming_days}days.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
