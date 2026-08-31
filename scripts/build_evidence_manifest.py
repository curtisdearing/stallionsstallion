#!/usr/bin/env python3
"""Assemble data/evidence_manifest.json — the benchmark selection gate.

    python3 scripts/build_evidence_manifest.py               # rebuild + verify
    python3 scripts/build_evidence_manifest.py --verify-only
    python3 scripts/build_evidence_manifest.py --select <artifact_id> \
        --role benchmark_wk4plus --acknowledge "<why the rule is satisfied>"

Assembles three kinds of record:

* **archival** — the pre-correction artifacts. Declared here by hand because
  their provenance was never captured at production time: there is no run
  record naming the code and inputs behind `data/backtest_c.csv`, which is
  itself part of why it cannot back a claim about the current model. Their
  content hash and schema ARE computed, so at least tampering is detectable.
* **candidate** — read from the run records the producing script wrote at the
  moment it produced the artifact, so provenance is recorded by the run rather
  than typed in afterwards.
* **selected** — never assigned automatically. `--select` requires an
  `--acknowledge` string that is stored in the manifest, so accepting a
  benchmark is always a deliberate, attributable act. A better-looking metric
  is not a reason and this script will not treat it as one.

The file it writes is the ONLY thing the site, the insight/explainer system,
the calibration report and the factor battery may use to choose a benchmark.
Selecting nothing is a valid, honest state; consumers render
`evidence.AWAITING_REVALIDATION` for it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import atomic, evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

PRE_CORRECTION_METHOD = "walk_forward_pace_ot_pooled_v1_20260730"

#: The pre-correction record, preserved and labelled. Numbers unchanged; the
#: files are never rewritten by this script.
ARCHIVAL = [
    {
        "artifact_id": "backtest_c_ridge_20260730",
        "path": "data/backtest_c.csv",
        "description": ("REGISTERED 2026-07-30 gate-3 ridge record. Produced "
                        "before the pace/OT walk-forward correction: PaceModel "
                        "was fitted on the full pooled multi-season frame and "
                        "OT parameters on the whole 2021-2025 window, both "
                        "in-window relative to the seasons graded. Preserved "
                        "as history; it describes a method the model no longer "
                        "uses."),
        "seasons": [2021, 2022, 2023],
        "weeks": {"eval_seasons": [2021, 2022, 2023], "first_week": 4},
    },
    {
        "artifact_id": "backtest_c_kalman_20260730",
        "path": "data/backtest_c_kalman.csv",
        "description": ("Pre-correction kalman companion to the registered "
                        "ridge record; same pooled pace/OT exposure."),
        "seasons": [2021, 2022, 2023],
        "weeks": {"eval_seasons": [2021, 2022, 2023], "first_week": 4},
    },
    {
        "artifact_id": "backtest_c_2024_holdout",
        "path": "data/backtest_c_2024.csv",
        "description": ("Pre-correction 2024 holdout extension of the gate-3 "
                        "record."),
        "seasons": [2024],
        "weeks": {"eval_seasons": [2024], "first_week": 4},
    },
]

#: The weeks 0-3 route. Unaffected by the pace/OT correction because it runs
#: no simulation at all -- there is no pace model and no OT model in it -- so
#: its registered evidence is not pre-correction in the sense the archival
#: records are, and it stays selectable.
EARLY = {
    "artifact_id": "backtest_early_20260830",
    "path": "data/backtest_early.csv",
    "method": "walk_forward_ridge_preseason_prior_v1_20260830",
    "description": ("Pre-registered 2026-08-30 early-week out-of-sample "
                    "residuals: seasons 2021-2025, CFBD weeks 1-3, "
                    "FBS-vs-FBS, walk-forward through "
                    "engine.solve_production_table. The ridge route runs no "
                    "simulation, so the pace/OT correction does not touch it."),
    "seasons": [2021, 2022, 2023, 2024, 2025],
    "weeks": {"eval_seasons": [2021, 2022, 2023, 2024, 2025],
              "first_week": 1, "last_week": 3},
    "seed": None,
    "selection_rule": ("selectable while it remains the artifact the "
                       "registered forecast-SD basis is derived from and its "
                       "producing method is unchanged"),
}

#: Code behind the early-week artifact. Recorded because it CAN be: that run
#: happened under the current tree and nothing has changed since, so hashing
#: now records the state that produced it. The archival records get no code
#: hashes for the opposite reason -- hashing today's `scripts/backtest_c.py`
#: would assert a provenance for a 2026-07-30 artifact that is simply untrue,
#: and a false provenance is worse than a missing one. `evidence.select`
#: refuses any artifact with no code provenance, so that absence is also what
#: makes the archival records structurally unselectable rather than merely
#: labelled.
EARLY_CODE_FILES = (
    "scripts/backtest_early.py", "nflvalue/engine.py", "nflvalue/ratings.py",
    "nflvalue/kalman_engine.py", "config.json",
)
EARLY_INPUT_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)


def _base(path: str) -> dict:
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        raise SystemExit(f"declared artifact {path} is not on disk")
    frame_rows = max(sum(1 for _ in open(full)) - 1, 0)
    return {"content_sha256": EV.sha256_file(full),
            "schema": EV.csv_header(full), "n_rows": frame_rows}


def _metrics_for(artifact_id: str) -> dict | None:
    path = os.path.join(DATA, "evidence", f"eval_{artifact_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    return {k: payload[k] for k in
            ("n", "seasons", "weeks", "margin", "brier", "reliability_bins",
             "interval_coverage", "evaluated_at_utc") if k in payload}


def build(previous: dict | None) -> dict:
    prior = {a["artifact_id"]: a for a in (previous or {}).get("artifacts", [])}
    artifacts = []

    for entry in ARCHIVAL:
        record = {
            **entry, "role": EV.ROLE_WK4PLUS,
            "status": EV.Status.ARCHIVAL.value,
            "method": PRE_CORRECTION_METHOD,
            "engine_policy": {}, "seed": None,
            "exclusions": {}, "code_hashes": {}, "input_hashes": {},
            "generated_at_utc": "2026-07-30T00:00:00Z",
            "metrics": None,
            "selection_rule": ("never selectable: produced before the pace/OT "
                               "walk-forward correction, and with no recorded "
                               "code or input provenance to verify it against"),
            "not_selected_reason": ("pre-correction method "
                                    f"{PRE_CORRECTION_METHOD}"),
            "supersedes": [],
            **_base(entry["path"]),
        }
        artifacts.append(record)

    for run in EV.load_run_records(ROOT):
        artifact_id = run["artifact_id"]
        record = dict(run)
        record["metrics"] = _metrics_for(artifact_id)
        record["supersedes"] = record.get("supersedes") or []
        # A prior manifest may have SELECTED this artifact. Rebuilding must not
        # silently revoke a human decision, nor silently keep one whose
        # artifact has changed underneath it.
        was = prior.get(artifact_id, {})
        if was.get("status") == EV.Status.SELECTED.value \
                and was.get("content_sha256") == record.get("content_sha256"):
            record["status"] = EV.Status.SELECTED.value
            record["not_selected_reason"] = ""
        else:
            record["status"] = EV.Status.CANDIDATE.value
            if not record.get("not_selected_reason"):
                record["not_selected_reason"] = (
                    "awaiting acceptance against the selection rule; no human "
                    "has recorded that the rule is satisfied")
        artifacts.append(record)

    early_inputs = {}
    for season in EARLY_INPUT_SEASONS:
        rel = os.path.join("data", "cfbd", f"games_{season}.parquet")
        if os.path.exists(os.path.join(ROOT, rel)):
            early_inputs[rel] = EV.sha256_file(os.path.join(ROOT, rel))
    early = {**EARLY, "role": EV.ROLE_EARLY,
             "status": EV.Status.SELECTED.value,
             "engine_policy": {"early_season_engine": "ridge",
                               "policy_changed_by_this_run": False},
             "exclusions": {},
             "code_hashes": {rel: EV.sha256_file(os.path.join(ROOT, rel))
                             for rel in EARLY_CODE_FILES
                             if os.path.exists(os.path.join(ROOT, rel))},
             "input_hashes": early_inputs,
             "generated_at_utc": "2026-08-30T17:20:00Z",
             "metrics": _metrics_for(EARLY["artifact_id"]),
             "not_selected_reason": "", "supersedes": [],
             **_base(EARLY["path"])}
    artifacts.append(early)

    return {
        "schema_version": EV.SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "note": ("The only permitted way to choose a benchmark artifact. "
                 "Archival records are preserved history and can never be "
                 "selected. A candidate becomes selected only by an explicit, "
                 "acknowledged human decision recorded here — never because a "
                 "metric improved. Where no artifact is selected for a role, "
                 "consumers must show the awaiting-revalidation state instead "
                 "of any earlier number."),
        "artifacts": artifacts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--select", metavar="ARTIFACT_ID")
    ap.add_argument("--role", default=EV.ROLE_WK4PLUS)
    ap.add_argument("--acknowledge", default="",
                    help="why the selection rule is satisfied; stored verbatim")
    args = ap.parse_args()

    path = EV.Manifest.path_for(ROOT)

    if args.verify_only:
        manifest = EV.Manifest.load(ROOT)
        results = manifest.verify_all()
        for artifact_id, status in results.items():
            print(f"{'ok  ' if status == 'ok' else 'FAIL'}  {artifact_id}  "
                  f"{'' if status == 'ok' else status}")
        return 0 if all(v == "ok" for v in results.values()) else 1

    previous = None
    if os.path.exists(path):
        with open(path) as handle:
            previous = json.load(handle)
    payload = build(previous)

    if args.select:
        if not args.acknowledge:
            raise SystemExit(
                "--select requires --acknowledge: selecting a benchmark is a "
                "decision, and the manifest records who decided and why. A "
                "better metric is not a reason.")
        found = False
        for record in payload["artifacts"]:
            if record["artifact_id"] != args.select:
                continue
            if record["status"] == EV.Status.ARCHIVAL.value:
                raise SystemExit(
                    f"{args.select} is archival and can never be selected; it "
                    "was produced by a method the model no longer uses")
            if record["role"] != args.role:
                raise SystemExit(
                    f"{args.select} has role {record['role']}, not {args.role}")
            record["status"] = EV.Status.SELECTED.value
            record["not_selected_reason"] = ""
            record["selection_acknowledgement"] = args.acknowledge
            found = True
        if not found:
            raise SystemExit(f"no artifact {args.select!r} in the manifest")

    atomic.write_json(path, payload)
    manifest = EV.Manifest.load(ROOT)          # parses + validates what we wrote
    results = manifest.verify_all()
    print(f"wrote {os.path.relpath(path, ROOT)} "
          f"({len(payload['artifacts'])} artifacts)")
    for role in EV.ROLES:
        try:
            record = manifest.select(role)
            print(f"  {role:20s} SELECTED {record.artifact_id} "
                  f"({record.n_rows} rows)")
        except EV.BenchmarkUnavailable:
            statuses = ", ".join(
                f"{r.artifact_id}={r.status}" for r in manifest.for_role(role))
            print(f"  {role:20s} NONE SELECTED -> consumers show the "
                  f"awaiting-revalidation state [{statuses}]")
    bad = {k: v for k, v in results.items() if v != "ok"}
    if bad:
        for artifact_id, reason in bad.items():
            print(f"  VERIFY FAIL {artifact_id}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
