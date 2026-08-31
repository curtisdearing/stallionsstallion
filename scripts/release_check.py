#!/usr/bin/env python3
"""Release-integrity check for the CANDIDATE tree.

    python3 scripts/release_check.py            # human-readable report
    python3 scripts/release_check.py --json     # machine-readable
    python3 scripts/release_check.py --python /path/to/python3.11

Every invariant is evaluated against a clean extraction of the *candidate
tree* -- HEAD plus the staged index, materialised with ``git write-tree`` and
``git archive`` -- never against the dirty working tree.  That is the whole
point: the working tree contains untracked files that ship to nobody, and a
check that runs in place cannot tell the difference.

The check is read-only with respect to git history and the working tree.  It
creates tree objects (write-tree) and a scratch directory, and it makes no
commit, moves no ref, stages nothing, and deletes nothing.

Exit status: 0 if every invariant holds, 1 otherwise.  The HEAD line is
advisory and never changes the exit status -- before the candidate set is
committed, a failing HEAD means "there is a commit still to make".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import release_gate as RG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the report as JSON")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used to run the in-tree probes "
                         "(default: this one)")
    ap.add_argument("--no-advisory", action="store_true",
                    help="skip the advisory HEAD-import line")
    args = ap.parse_args()

    root = RG.REPO_ROOT
    checks = RG.run_checks(root, python=args.python)
    advisory = None if args.no_advisory else RG.head_import_status(
        root, python=args.python)
    failed = [c for c in checks if not c.ok]

    if args.as_json:
        payload = {
            "root": str(root),
            "candidate_tree": RG.candidate_tree_id(root),
            "python": args.python,
            "ok": not failed,
            "checks": [c._asdict() for c in checks],
        }
        if advisory is not None:
            payload["advisory"] = advisory._asdict()
        print(json.dumps(payload, indent=2))
        return 1 if failed else 0

    width = max(len(c.name) for c in checks)
    print(f"release check -- candidate tree of {root}")
    print(f"probe interpreter: {args.python}")
    print("-" * (width + 12))
    for check in checks:
        print(f"{'PASS' if check.ok else 'FAIL'}  {check.name:<{width}}  "
              f"{check.detail}")
    print("-" * (width + 12))
    if advisory is not None:
        print(f"{'note' if advisory.ok else 'NOTE'}  {advisory.name:<{width}}  "
              f"{advisory.detail}")
        print("-" * (width + 12))
    if failed:
        print(f"FAILED: {len(failed)} of {len(checks)} invariants "
              "do not hold for the candidate tree.")
        return 1
    print(f"OK: all {len(checks)} invariants hold for the candidate tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
