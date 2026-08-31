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
    must not block the site rebuild.  A *malformed* input is different: a
    number the feed cannot parse fails the build, because publishing it as
    ``null`` would read as "the model declined to forecast this game".
  * The model family named at the top of the feed is DERIVED from the rows,
    never asserted here.  A hardcoded engine string is a claim this file is in
    no position to make: the week's own artifact knows what priced it.
  * The artifact is VALIDATED against `nflvalue.forecast_contract` before a
    single row is copied.  This file used to do the opposite: when the
    `mc_margin` column was absent it copied the ridge point estimate
    `pred_margin` into `mc_margin`, so a number produced by no simulation was
    published under a Monte-Carlo name.  That relabel is gone, and cannot come
    back -- the contract has no `mc_margin` to relabel into.
  * The feed carries only DERIVED analysis (model outputs, graded records)
    — no CollegeFootballData source data is redistributed, per the
    recorded terms position.

Usage: python scripts/build_hub_feed.py [--out data/site/api/hub.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import data_health as DH  # noqa: E402
from nflvalue import forecast_contract as FC  # noqa: E402

DISCLAIMER = ("Paper-only research: nothing here is a bet recommendation. "
              "The kill-check (gate 7) has not returned GO for any slice.")

#: What the feed says when the rows do not agree on one known model family.
UNRESOLVED_FAMILY = "unresolved"

#: Exactly the contract's own columns, carried through verbatim. Per-row
#: provenance and status ride along on purpose: a reader of the hub must be
#: able to tell a priced row from a refusal, and to see which model, which
#: registered SD basis and which snapshot produced each number.
ROW_FIELDS = tuple(FC.COLUMNS)


class HubFeedError(RuntimeError):
    """The artifact could not be carried into the feed without inventing."""


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


def _jsonable(value):
    """NaN/NA -> None; numpy scalars -> plain Python. JSON carries neither."""
    if value is None or value is getattr(__import__("pandas"), "NA", None):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
        return _jsonable(value)
    return value


def _load_games(path):
    """Every published row, contract-validated, copied verbatim.

    Validation happens before anything is copied, so a mixed-schema artifact
    fails the build instead of being published under whichever reading
    happened to parse. Nothing is recomputed, renamed or back-filled here --
    in particular a ridge point estimate is NOT copied into a Monte-Carlo
    field, which is what the previous version of this function did.
    """
    try:
        frame = FC.read_artifact(path)
    except FC.ContractError as exc:
        raise HubFeedError(
            f"{os.path.basename(path)} does not satisfy forecast contract "
            f"v{FC.SCHEMA_VERSION} and will not be published: {exc}") from None

    games, seen_ids = [], {}
    for position, record in enumerate(frame.to_dict("records"), start=1):
        where = f"{os.path.basename(path)} row {position}"
        game = {field: _jsonable(record.get(field)) for field in ROW_FIELDS}
        game_id = game["game_id"]
        if game_id is None:
            raise HubFeedError(f"{where}: game_id is blank")
        game["game_id"] = int(game_id)
        first = seen_ids.setdefault(game["game_id"], position)
        if first != position:
            raise HubFeedError(
                f"{where}: game_id={game['game_id']} is duplicated (first seen "
                f"at row {first}). Two rows claiming one game publish an "
                "ambiguous identity")
        games.append(game)
    return games


def _family_from_rows(games):
    """The model family the ROWS agree on, or `unresolved`. Never a bare claim.

    A week may legitimately be priced by one family only; a feed that found
    two would be describing an artifact assembled from two runs, which the
    contract does not permit and this feed will not paper over.
    """
    named = [g.get("model_family") for g in games]
    if named and all(name in FC.FAMILIES for name in named) \
            and len(set(named)) == 1:
        return named[0]
    return UNRESOLVED_FAMILY


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


def _insight_cards(season, week):
    """The four-section cards for the week, if they have been rendered.

    Only forecast-bearing rows carry a card into the feed: a refusal already
    states itself completely in its own row (`status` + `refusal_reason`), and
    duplicating 400 of them would multiply the feed for no reader benefit.
    Absent cards are absent, never an empty list of blank frameworks.
    """
    path = os.path.join(ROOT, "data", f"insights_{season}_wk{week:02d}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise HubFeedError(
            f"{os.path.basename(path)} could not be read ({exc}); a damaged "
            "insight file fails the build rather than publishing a week with "
            "silently missing cards") from None
    cards = [c for c in payload.get("cards", [])
             if c.get("status") in FC.FORECAST_BEARING]
    return {"schema_version": payload.get("schema_version"),
            "n_cards_total": payload.get("n_cards"),
            "by_status": payload.get("by_status"),
            "n_cards_in_feed": len(cards),
            "feed_note": ("Cards are included for forecast-bearing rows only; "
                          "refused and degraded rows state themselves in the "
                          "row."),
            "cards": cards}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "data", "site", "api", "hub.json"))
    args = ap.parse_args()

    feed = {
        "schema_version": FC.SCHEMA_VERSION,
        "project": "stallionsstallion",
        # Filled in from the artifact's own rows below — never asserted here.
        "model_family": UNRESOLVED_FAMILY,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "attribution": ("Derived analysis of CollegeFootballData.com — "
                        "no source data redistributed."),
    }

    # Pipeline health rides at the top of the feed, verbatim and non-numeric.
    # A consumer must be able to tell "these are this week's forecasts" from
    # "the pipeline halted and these are the last ones it stood behind"
    # WITHOUT inspecting timestamps and guessing. A corrupt health file raises
    # rather than reading as healthy.
    feed["health"] = DH.read_status(os.path.join(ROOT, "data"))


    latest = _latest_predictions_path()
    if latest:
        (season, week), path = latest
        games = _load_games(path)
        # The feed labels the week from the filename, so a row that claims a
        # different one would be published under a week it was not priced for.
        for game in games:
            if (game["season"], game["week"]) != (season, week):
                raise HubFeedError(
                    f"{os.path.basename(path)}: game_id={game['game_id']} "
                    f"claims season={game['season']} wk{game['week']}, but the "
                    f"filename says season={season} wk{week}. Refusing to "
                    "publish a row under a week it was not priced for")
        priced = [g for g in games
                  if g["status"] in FC.FORECAST_BEARING]
        feed["model_family"] = _family_from_rows(games)
        feed["week"] = {
            "season": season, "week": week,
            "n_games": len(games), "n_priced": len(priced),
            "n_unpriced": len(games) - len(priced),
            "by_status": {status.value: sum(1 for g in games
                                            if g["status"] == status.value)
                          for status in FC.Status},
            "status_note": ("Refused and degraded games are refused, not "
                            "guessed at, and carry no forecast numbers: "
                            "unsupported cross-/lower-division matchups and "
                            "unratable teams stay unpriced by design."),
            "games": games,
        }
        cards = _insight_cards(season, week)
        if cards is not None:
            feed["week"]["insights"] = cards
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
