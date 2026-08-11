#!/usr/bin/env python3
"""Build the slim machine-readable feed the Dearing Football hub consumes.

Writes ``data/site/api/hub.json`` — a COMMITTED artifact, like the rest of
``data/site/`` (the Pages workflow publishes the committed site only; the
weekly loop regenerates and pushes it from the Mac, so the CFBD key never
touches CI — see the 2026-08-10 launch checkpoint).

Honesty rules:
  * Numbers are copied verbatim from the pipeline's own artifacts (the
    newest ``data/predictions_*.csv`` and the season's ``data/recaps/``) —
    nothing is recomputed, so the feed cannot disagree with the site.
  * Vetoed / unpriced games ride along with their veto note; the model
    refusing to price a game is information, not an omission.
  * Missing inputs -> null sections, never a crash: a broken feed build
    must not block the site rebuild.
  * The feed carries only DERIVED analysis (model outputs, graded records)
    — no CollegeFootballData source data is redistributed, per the
    recorded terms position.

Usage: python scripts/build_hub_feed.py [--out data/site/api/hub.json]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISCLAIMER = ("Paper-only research: nothing here is a bet recommendation. "
              "The kill-check (gate 7) has not returned GO for any slice.")


def _latest_predictions_path():
    """Newest predictions file by (season, week) parsed from the name."""
    best = None
    for path in glob.glob(os.path.join(ROOT, "data", "predictions_*.csv")):
        m = re.search(r"predictions_(\d{4})_wk(\d+)", os.path.basename(path))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if best is None or key > best[0]:
            best = (key, path)
    return best  # ((season, week), path) or None


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _load_games(path):
    games = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            veto = str(row.get("veto", "")).strip().lower() == "true"
            games.append({
                "slate": row.get("slate") or None,
                "date": row.get("date") or None,
                "home": row.get("home"),
                "away": row.get("away"),
                "neutral": str(row.get("neutral", "")).strip().lower() == "true",
                "pred_margin": _num(row.get("pred_margin")),
                "p_home_win": _num(row.get("p_home_win")),
                "band_pm": _num(row.get("band_pm")),
                "veto": veto,
                "note": (row.get("note") or "").strip() or None,
            })
    return games


def _season_record(season):
    """Aggregate the season's graded recaps verbatim; None if none exist."""
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "recaps",
                                          f"{season}_wk*.json")))
    if not paths:
        return None
    weeks, graded, beat = [], 0, 0
    mae_num = mae_den = mkt_num = mkt_den = 0.0
    for p in paths:
        try:
            with open(p) as fh:
                r = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        n = int(r.get("n_graded") or 0)
        graded += n
        if r.get("model_beat_market"):
            beat += 1
        if r.get("model_mae") is not None and n:
            mae_num += float(r["model_mae"]) * n
            mae_den += n
        nm = int(r.get("n_with_market") or 0)
        if r.get("market_mae") is not None and nm:
            mkt_num += float(r["market_mae"]) * nm
            mkt_den += nm
        weeks.append({"week": r.get("week"), "n_graded": n,
                      "model_mae": r.get("model_mae"),
                      "market_mae": r.get("market_mae"),
                      "model_beat_market": r.get("model_beat_market")})
    if not weeks:
        return None
    return {
        "season": season,
        "weeks_graded": len(weeks),
        "games_graded": graded,
        "model_mae": round(mae_num / mae_den, 2) if mae_den else None,
        "market_mae": round(mkt_num / mkt_den, 2) if mkt_den else None,
        "weeks_model_beat_market": beat,
        "by_week": weeks,
        "note": ("The model is NOT expected to beat the closing line; "
                 "if it does, suspect a bug before an edge."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "data", "site", "api", "hub.json"))
    args = ap.parse_args()

    feed = {
        "schema_version": 1,
        "project": "stallionsstallion",
        "engine": "kalman state-space (registered bakeoff winner, 2026-07-30)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "attribution": ("Derived analysis of CollegeFootballData.com — "
                        "no source data redistributed."),
    }

    latest = _latest_predictions_path()
    if latest:
        (season, week), path = latest
        games = _load_games(path)
        priced = [g for g in games if not g["veto"]]
        feed["week"] = {
            "season": season, "week": week,
            "n_games": len(games), "n_priced": len(priced),
            "n_vetoed": len(games) - len(priced),
            "veto_note": ("Vetoed games are refused, not guessed at: "
                          "FBS-vs-FCS and unratable teams stay unpriced "
                          "by design."),
            "games": games,
        }
        feed["record"] = _season_record(season)
    else:
        feed["week"] = None
        feed["record"] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(feed, fh, separators=(",", ":"))
    os.replace(tmp, args.out)
    print(f"[hub-feed] wrote {args.out} ({os.path.getsize(args.out)} bytes; "
          f"week={'%s wk%s' % (feed['week']['season'], feed['week']['week']) if feed['week'] else None}; "
          f"record={'yes' if feed['record'] else None})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
