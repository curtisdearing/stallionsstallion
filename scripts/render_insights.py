#!/usr/bin/env python3
"""Build the insight cards for a published week. Deterministic; no prose.

    python3 scripts/render_insights.py --season 2026 --week 1
    python3 scripts/render_insights.py --season 2026 --week 1 --text

Reads the PUBLISHED contract artifact — never a re-simulation — so every
number on a card is the number that was published, and re-running this
changes nothing. The rating decomposition behind the "why" section is derived
from the production table for the same week; a frozen row gets no
decomposition, because its reasoning is not re-derived after kickoff.

Writes `data/insights_{season}_wk{NN}.json`, which is what a hub or API card
renders. Cached data only; makes no API call and reads no clock (every
timestamp on a card comes from the artifact).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from nflvalue import atomic, engine as EG  # noqa: E402
from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import insight as IN  # noqa: E402
from nflvalue import ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HISTORY_START = 2019


def artifact_path(season: int, week: int) -> str:
    return os.path.join(DATA, f"predictions_{season}_wk{week:02d}.csv")


def out_path(season: int, week: int) -> str:
    return os.path.join(DATA, f"insights_{season}_wk{week:02d}.json")


def data_as_of(season: int) -> str | None:
    """Modification time of the schedule cache the week was priced from."""
    path = os.path.join(DATA, "cfbd", f"games_{season}.parquet")
    if not os.path.exists(path):
        return None
    from datetime import datetime, timezone
    return (datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"))


def components_for(rows: pd.DataFrame, season: int, week: int) -> dict:
    """Rating decomposition per PRICED game, from the production table.

    Frozen rows are deliberately absent: the table has moved since they were
    frozen, so a decomposition built now would explain the published number
    with reasoning that did not produce it.
    """
    priced = rows[rows["status"] == FC.Status.PRICED.value]
    if priced.empty:
        return {}
    params = R.load_params()
    games = R.game_frame(list(range(HISTORY_START, season + 1)))
    table = EG.solve_production_table(games, season, week, params)
    family = FC.FAMILIES[str(priced.iloc[0]["model_family"])]
    out = {}
    for row in priced.to_dict("records"):
        simulated = (float(row["projected_home_margin"])
                     if family.supports_totals else None)
        try:
            out[int(row["game_id"])] = IN.components_from_table(
                table, home=str(row["home"]), away=str(row["away"]),
                neutral=bool(row["neutral_site"]),
                model_version=str(row["model_version"]),
                simulated_margin=simulated)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"game {row['game_id']}: could not rebuild the rating "
                f"decomposition ({type(exc).__name__}: {exc}). Refusing to "
                "publish a card that cannot say why.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--text", action="store_true", help="also print the cards")
    ap.add_argument("--artifact", default=None,
                    help="read a specific contract artifact instead of the "
                         "published one (for demo/fixture renders)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.artifact or artifact_path(args.season, args.week)
    if not os.path.exists(path):
        raise SystemExit(f"no published artifact at {os.path.relpath(path, ROOT)}")
    rows = FC.read_artifact(path)

    payload = IN.build_week(
        rows.to_dict("records"),
        components_by_game=components_for(rows, args.season, args.week),
        data_as_of_utc=data_as_of(args.season))

    destination = args.out or out_path(args.season, args.week)
    atomic.write_json(destination, payload, sort_keys=True)
    print(f"[insights] {os.path.relpath(destination, ROOT)}: "
          f"{payload['n_cards']} cards {payload['by_status']}")
    if args.text:
        for card in payload["cards"]:
            print()
            print(IN.render_card_text(card))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
