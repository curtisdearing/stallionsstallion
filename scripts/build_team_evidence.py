#!/usr/bin/env python3
"""Build team-season evidence profiles as of a (season, week) and write the artifact.

    .venv/bin/python scripts/build_team_evidence.py --season 2026 --week 2

Writes `data/team_evidence/{season}_wk{NN}.json`, atomically, containing one
profile per team plus a run header carrying the registry version, the source
states and the sha256 of the artifact itself.

This script prices nothing, publishes nothing and touches no snapshot. It is an
evidence build; the weekly loop does not depend on it and is not modified by it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nflvalue import atomic, ingest                      # noqa: E402
from nflvalue.team_evidence import profile as TP         # noqa: E402
from nflvalue.team_evidence.registry import MeasureRegistry  # noqa: E402
from nflvalue.team_evidence.sources import EvidenceSources   # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(DATA, "team_evidence")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def out_path(season: int, week: int) -> str:
    return os.path.join(OUT_DIR, f"{season}_wk{int(week):02d}.json")


def build(season: int, week: int, history: int) -> dict:
    seasons = list(range(season - history, season + 1))
    registry = MeasureRegistry.load(DATA)
    sources = EvidenceSources(DATA)
    states = {sid: sources.state(sid, seasons=(season,))
              for sid in registry.source_ids()}
    # Match what build_profiles actually uses: the unit ratings are computed from
    # the plays cache, not read from data/ratings_{season}.parquet, so the header
    # must not report a dependency the values do not have.
    if TP.UNIT_SOURCE_ID in states:
        states[TP.UNIT_SOURCE_ID] = sources.derived_state(
            TP.UNIT_SOURCE_ID, [states.get("model_drive_features")])

    profiles = TP.build_profiles(season, week, data_dir=DATA,
                                 seasons_of_history=seasons,
                                 sources=sources, registry=registry)

    # FBS membership from the per-season /teams pull, never a hardcoded map. A
    # team absent from that table is marked unknown rather than guessed non-FBS.
    teams_tbl = ingest.load_teams([season])
    fbs = ({str(r.team) for r in teams_tbl.itertuples() if bool(r.is_fbs)}
           if len(teams_tbl) else set())
    known = {str(r.team) for r in teams_tbl.itertuples()} if len(teams_tbl) else set()

    payload = {
        "schema": "stallionsstallion.team_evidence.v1",
        "season": int(season),
        "as_of_week": int(week),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": git_commit(),
        "registry_path": os.path.relpath(registry.path, ROOT),
        "registry_measures": len(registry.measures),
        "seasons_of_history": seasons,
        "source_states": {
            sid: {"availability": st.availability.value,
                  "missingness": st.missingness.value,
                  "reason": st.reason,
                  "version": st.version,
                  "retrieved_at": (st.retrieved_at.isoformat()
                                   if st.retrieved_at else None),
                  "availability_class": st.availability_class.value}
            for sid, st in sorted(states.items())},
        "blocked_measures": {
            s.measure_id: s.blocked_reason for s in sorted(
                registry.blocked(), key=lambda s: s.measure_id)},
        "availability_note": (
            "Every value here is reconstructed_week_granular. Nothing in "
            "data/cfbd/*.parquet records when a row became knowable, so pregame "
            "availability is the assumption that everything from week <= W-1 was "
            "knowable at week W. Retrospective PREDICTIVE claims off these values "
            "are blocked by team_evidence.contracts."
            "assert_retrospective_claim_allowed."),
        "fbs_classification_known": bool(known),
        "teams": {t: {**p.to_dict(),
                      "is_fbs": (t in fbs) if t in known else None}
                  for t, p in sorted(profiles.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["artifact_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True,
                    help="build from games STRICTLY EARLIER than this week")
    ap.add_argument("--history", type=int, default=2,
                    help="prior seasons fed to the opponent-adjustment solve")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    payload = build(args.season, args.week, args.history)
    path = args.out or out_path(args.season, args.week)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic.write_json(path, payload)

    teams = payload["teams"]
    present = sum(1 for t in teams.values()
                  for o in t["observations"] if o["availability"] == "ok")
    total = sum(len(t["observations"]) for t in teams.values())
    print(f"wrote {os.path.relpath(path, ROOT)}")
    n_fbs = sum(1 for t in teams.values() if t.get("is_fbs"))
    print(f"  teams        {len(teams)} ({n_fbs} FBS)")
    print(f"  observations {present}/{total} present")
    print(f"  blocked      {len(payload['blocked_measures'])} measures")
    print(f"  artifact     sha256:{payload['artifact_sha256'][:16]}")
    for sid, st in payload["source_states"].items():
        if st["availability"] != "ok":
            print(f"  source {sid}: {st['availability']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
