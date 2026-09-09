#!/usr/bin/env python3
"""Render team-season profiles and game logs into the Obsidian vault.

    .venv/bin/python scripts/render_team_notes.py --season 2026 --week 2
    .venv/bin/python scripts/render_team_notes.py --season 2026 --week 2 --dry-run

Reads the artifact written by `scripts/build_team_evidence.py` and writes:

    teams/{season}/{team_slug}.md                       generated block only
    games/{season}/wk{NN}/{game_id}--away-at-home.md     immutable pregame record

Anything outside a generated block is preserved byte-for-byte, including a
hand-written `## Notes` section. The pregame record is written once and refuses to
be rewritten; postgame reviews are appended by
`nflvalue.team_evidence.vault.append_postgame_review`, never merged into it.

This script writes private notes into the vault. It publishes nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nflvalue import ingest                                    # noqa: E402
from nflvalue.team_evidence import measures as MEAS             # noqa: E402
from nflvalue.team_evidence import vault as V                  # noqa: E402
from nflvalue.team_evidence.contracts import (AvailabilityClass,  # noqa: E402
                                              EvidenceAvailability,
                                              Missingness, Observation,
                                              TeamSeasonProfile)
from nflvalue.team_evidence.registry import MeasureRegistry     # noqa: E402

DATA = os.path.join(ROOT, "data")


def artifact_path(season: int, week: int) -> str:
    return os.path.join(DATA, "team_evidence", f"{season}_wk{int(week):02d}.json")


def _obs(d: dict) -> Observation:
    def ts(v):
        return dt.datetime.fromisoformat(v) if v else None
    return Observation(
        measure_id=d["measure_id"], team=d["team"], season=int(d["season"]),
        as_of_week=int(d["as_of_week"]), label=d["label"], unit=d.get("unit"),
        value=d.get("value"), n=int(d.get("n") or 0),
        n_games=int(d.get("n_games") or 0),
        availability=EvidenceAvailability(d["availability"]),
        missingness=Missingness(d["missingness"]),
        availability_class=AvailabilityClass(d["availability_class"]),
        reason=d.get("reason"), source_id=d.get("source_id"),
        source_url_or_id=d.get("source_url_or_id"),
        source_version=d.get("source_version"),
        retrieved_at=ts(d.get("retrieved_at")),
        effective_through=ts(d.get("effective_through")),
        opponent_adjusted=bool(d.get("opponent_adjusted")),
        opponent_adjustment_method=d.get("opponent_adjustment_method"),
        game_ids=tuple(d.get("game_ids") or ()), caveat=d.get("caveat"))


def _profile(d: dict) -> TeamSeasonProfile:
    return TeamSeasonProfile(
        team=d["team"], season=int(d["season"]), as_of_week=int(d["as_of_week"]),
        generated_at_utc=dt.datetime.fromisoformat(d["generated_at_utc"]),
        observations=tuple(_obs(o) for o in d["observations"]),
        games_used=tuple(int(g) for g in d.get("games_used") or ()),
        opponents_faced=tuple(d.get("opponents_faced") or ()),
        neutral_site_games=tuple(int(g) for g in d.get("neutral_site_games") or ()),
        personnel_facts=tuple(d.get("personnel_facts") or ()),
        notes=tuple(d.get("notes") or ()))


def _frozen_rows(season: int, week: int):
    """Read the immutable pregame freeze, if one exists. READ ONLY.

    `data/snapshots/{season}_wk{NN}/frozen.json` is written by
    `nflvalue.snapshots.freeze_kicked_off` and is the only artifact in this repo
    that proves what was published before a kickoff. Nothing here writes to it.
    """
    path = os.path.join(DATA, "snapshots", f"{season}_wk{int(week):02d}",
                        "frozen.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {int(k): v for k, v in raw.items() if isinstance(v, dict)}


def _write_game_notes(root, payload, args, sha, commit) -> int:
    season = int(payload["season"])
    used = sorted({int(g) for t in payload["teams"].values()
                   for g in (t.get("games_used") or [])})
    if not used:
        return 0
    plays = ingest.load_plays_clean([season])
    if plays is None or len(plays) == 0:
        print("no plays cache; skipping game notes", file=sys.stderr)
        return 0
    per_game = MEAS.team_game_measures(plays[plays["game_id"].isin(used)])
    if per_game.empty:
        return 0
    games = ingest.load_games([season])
    meta = {int(r.id): r for r in games.itertuples()} if len(games) else {}

    latest_week = int(per_game["week"].max())
    target = per_game[per_game["week"] == latest_week]
    frozen = _frozen_rows(season, latest_week)

    count = 0
    for gid, grp in target.groupby("game_id", sort=True):
        g = meta.get(int(gid))
        home = str(getattr(g, "home_team", "")) or str(grp["team"].iloc[0])
        away = str(getattr(g, "away_team", "")) or str(grp["opponent"].iloc[0])
        entry = frozen.get(int(gid)) or {}
        row = entry.get("row") if isinstance(entry.get("row"), dict) else None
        if args.dry_run:
            count += 1
            continue
        V.write_game_note(
            root, season, latest_week, int(gid), home, away,
            grp.to_dict("records"), artifact_sha256=sha, commit=commit,
            kickoff_utc=str(getattr(g, "start_date", "")) or None,
            neutral_site=bool(getattr(g, "neutral_site", False)),
            pregame=row, pregame_stamp=entry.get("from_stamp"))
        count += 1
    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--root", default=None,
                    help="vault root; defaults to the folder containing the repo")
    ap.add_argument("--limit", type=int, default=0, help="write at most N team notes")
    ap.add_argument("--all-teams", action="store_true",
                    help="also write notes for non-FBS teams that appear in the "
                         "artifact (default: FBS only)")
    ap.add_argument("--games", action="store_true",
                    help="also write a game log per completed game in the most "
                         "recent week the artifact used, with the frozen pregame "
                         "record when data/snapshots/ has one")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    path = args.artifact or artifact_path(args.season, args.week)
    if not os.path.exists(path):
        print(f"no artifact at {os.path.relpath(path, ROOT)} — run "
              "scripts/build_team_evidence.py first", file=sys.stderr)
        return 2
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    root = args.root or V.vault_root(ROOT)
    registry = MeasureRegistry.load(DATA)
    sha = payload.get("artifact_sha256", "")
    commit = payload.get("commit", "unknown")

    teams = sorted(payload["teams"].items())
    if not args.all_teams:
        kept = [(k, v) for k, v in teams if v.get("is_fbs")]
        if kept:
            teams = kept
        elif payload.get("fbs_classification_known") is False:
            print("no FBS classification in the artifact; writing every team",
                  file=sys.stderr)
    if args.limit:
        teams = teams[:args.limit]

    written = []
    for _, raw in teams:
        prof = _profile(raw)
        target = V.team_note_path(root, prof.season, prof.team)
        if args.dry_run:
            written.append(target)
            continue
        written.append(V.write_team_note(root, prof, artifact_sha256=sha,
                                         commit=commit, registry=registry))

    n_games = 0
    if args.games:
        n_games = _write_game_notes(root, payload, args, sha, commit)

    verb = "would write" if args.dry_run else "wrote"
    if args.games:
        print(f"{verb} {n_games} game note(s)")
    print(f"{verb} {len(written)} team note(s) under {root}")
    for p in written[:5]:
        print(f"  {os.path.relpath(p, root)}")
    if len(written) > 5:
        print(f"  ... and {len(written) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
