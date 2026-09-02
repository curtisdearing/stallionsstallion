#!/bin/bash
# Zero-credit weekly runner — runs the loop natively on this machine.
# No AI, no Cowork session, no tokens: plain Python on a schedule.
#
# Usage:  ./scripts/run_weekly.sh week-open|pregame|grade
# Wire-up (launchd/cron): see RUNBOOK_WEEKLY.md next to this file.
#
# Premortem hardening (each of these failure modes is handled, not hoped away):
#  * cron/launchd's bare environment (no PATH/locale) -> absolute paths, own venv
#  * missing deps after an OS update            -> verified before running, loud fail
#  * no network / CFBD down                     -> loop marks manifest degraded, exits nonzero
#  * two jobs overlapping                       -> loop's single-writer lock
#  * crash mid-run                              -> loop's per-stage sentinels resume
#  * runaway API spend                          -> loop's budget guard hard-refuses
#  * offseason                                  -> clean exit 0, manifest says so
#  * silent failure                             -> exit code + dated log + (optional) Discord on fail
#  * unhealthy inputs                           -> loop halts BEFORE publishing, exit 3
#
# THIS RUNNER NEVER PUBLISHES. It builds local artifacts and reports whether
# they are ready. It runs no `git add`, no `git commit` and no `git push` --
# it did, and the push was unscoped (`git push origin master`), so a run could
# publish every unpushed commit on the branch as a side effect of grading a
# week. Publishing is now one explicit human command:
#
#     ./scripts/publish_site.sh
#
# which shows what would go out and requires confirmation.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
LOG_DIR="$REPO/data/logs"
JOB="${1:?usage: run_weekly.sh week-open|pregame|grade}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/${JOB}_${STAMP}.log"
mkdir -p "$LOG_DIR"

# launchd resolves /usr/bin/python3 on macOS (currently 3.9), while this CFB
# pipeline uses Python 3.11 syntax.  Permit an explicit override but prefer
# the Homebrew interpreter installed for this Mac and fail before any API work.
PYTHON="${STALLIONS_PYTHON:-/opt/homebrew/opt/python@3.11/bin/python3.11}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3.11 || true)"
fi
if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "[runner] requires Python 3.11+; install Homebrew python@3.11 or set STALLIONS_PYTHON" | tee -a "$LOG"
  exit 1
fi

# One-time venv bootstrap; self-heals if an OS update breaks the interpreter.
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[runner] creating venv" | tee -a "$LOG"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$REPO/requirements-weekly.txt"
fi
# A prior runner release could have created this venv with macOS's Python 3.9.
# Recreate rather than discovering an incompatible interpreter after scheduling.
if ! "$VENV/bin/python3" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "[runner] recreating venv with Python 3.11+" | tee -a "$LOG"
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$REPO/requirements-weekly.txt"
fi
# Dep sanity — fail loud BEFORE touching the API budget.
"$VENV/bin/python3" - <<'PY' || { echo "[runner] deps broken — reinstalling" | tee -a "$LOG"; "$VENV/bin/pip" install -q -r "$REPO/requirements-weekly.txt"; }
import pandas, pyarrow, numpy
PY

cd "$REPO"
echo "[runner] $JOB starting $STAMP (log: $LOG)" | tee -a "$LOG"
set +e
"$VENV/bin/python3" scripts/weekly_loop.py --job "$JOB" >>"$LOG" 2>&1
CODE=$?
set -e
echo "[runner] $JOB finished with exit $CODE" | tee -a "$LOG"

# After a successful grade pass, regenerate the public site from fresh data.
# A build failure is a FAILED RUN. It used to be logged and swallowed, so the
# runner reported success while the site and the hub feed silently kept
# serving the previous week's numbers -- the operator had no signal at all.
if [ "$JOB" = "grade" ] && [ "$CODE" -eq 0 ]; then
  # build_site.py builds the site AND the hub feed, and now exits non-zero if
  # either fails, so one call covers both and attributes the failure in the log.
  if ! "$VENV/bin/python3" scripts/build_site.py >>"$LOG" 2>&1; then
    echo "[runner] SITE/HUB BUILD FAILED — local artifacts are not ready to publish" | tee -a "$LOG"
    CODE=4
  else
    echo "[runner] local site + hub feed rebuilt; READY TO PUBLISH" | tee -a "$LOG"
    echo "[runner] publish with: ./scripts/publish_site.sh   (nothing was pushed)" | tee -a "$LOG"
  fi
fi

# After a successful forecast pass, refresh the vault's every-FBS-game page
# (picks/<season>_wk<NN>_every_fbs_game.md, outside the repo). It is a
# read-only render of the artifact and the cards -- nothing is recomputed --
# and it is a private note, so a failure here is logged, not a failed run.
if { [ "$JOB" = "pregame" ] || [ "$JOB" = "week-open" ]; } && [ "$CODE" -eq 0 ]; then
  if "$VENV/bin/python3" scripts/render_picks.py --auto >>"$LOG" 2>&1; then
    echo "[runner] vault picks page refreshed" | tee -a "$LOG"
  else
    echo "[runner] vault picks page NOT refreshed (see log); the run itself is unaffected" | tee -a "$LOG"
  fi
fi

case "$CODE" in
  0) echo "[runner] $JOB OK" | tee -a "$LOG" ;;
  3) echo "[runner] $JOB HALTED: inputs failed their health check; nothing was published" | tee -a "$LOG" ;;
  4) echo "[runner] $JOB UNSUCCESSFUL: local artifact build failed" | tee -a "$LOG" ;;
  *) echo "[runner] $JOB FAILED (exit $CODE)" | tee -a "$LOG" ;;
esac

# Keep the last 60 logs; never let the log dir grow unbounded.
ls -t "$LOG_DIR" | tail -n +61 | while read -r f; do rm -f "$LOG_DIR/$f"; done

exit "$CODE"
