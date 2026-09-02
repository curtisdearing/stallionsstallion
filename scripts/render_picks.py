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
             bases: Dict[str, dict], fbs: set) -> Dict[str, Any]:
    """One game -> one line of the page. Refused games keep their reason."""
    home, away = str(row["home"]), str(row["away"])
    out = {
        "game_id": int(row["game_id"]),
        "kickoff_utc": row["kickoff_utc"],
        "kickoff_et": kickoff_et(row["kickoff_utc"]),
        "public_date": row["public_date"],
        "away": away, "home": home,
        "neutral": bool(row["neutral_site"]),
        "status": row["status"],
        "home_is_fbs": home in fbs, "away_is_fbs": away in fbs,
        "model_family": row.get("model_family"),
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
          fbs: set) -> List[Dict[str, Any]]:
    out = []
    for row in rows.sort_values(["kickoff_utc", "game_id"]).to_dict("records"):
        home, away = str(row["home"]), str(row["away"])
        if home not in fbs and away not in fbs:
            continue  # not an FBS game
        card = cards.get(int(row["game_id"]))
        if card is None:
            raise SystemExit(f"game {row['game_id']} has no insight card; "
                             "the cards and the artifact are out of step")
        out.append(pick_row(row, card, bases, fbs))
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


CSV_COLUMNS = ("game_id", "kickoff_et", "kickoff_utc", "public_date", "away",
               "home", "neutral", "status", "model_family", "pick", "p_pick",
               "margin_pick", "sd", "ci80_lo", "ci80_hi", "ci95_lo", "ci95_hi",
               "rationale", "caution", "result")


def write_outputs(picks: List[Dict[str, Any]], season: int, week: int,
                  out_dir: str, text: str) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{season}_wk{week:02d}_every_fbs_game")
    md_path, csv_path = stem + ".md", stem + ".csv"
    with open(md_path, "w") as fh:
        fh.write(text)
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
    picks = build(rows, cards, bases, fbs)
    text = render_markdown(picks, args.season, args.week, rows, bases)
    md_path, csv_path = write_outputs(picks, args.season, args.week,
                                      args.out_dir or default_out_dir(), text)
    n_pick = sum(1 for p in picks if p["pick"])
    print(f"[picks] {len(picks)} FBS games, {n_pick} with a projected winner, "
          f"{len(picks) - n_pick} not forecast")
    print(f"[picks] wrote {md_path}")
    print(f"[picks] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
