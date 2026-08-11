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
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
LOG_DIR="$REPO/data/logs"
JOB="${1:?usage: run_weekly.sh week-open|pregame|grade}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/${JOB}_${STAMP}.log"
mkdir -p "$LOG_DIR"

# One-time venv bootstrap; self-heals if an OS update breaks the interpreter.
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[runner] creating venv" | tee -a "$LOG"
  /usr/bin/env python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
fi
# Dep sanity — fail loud BEFORE touching the API budget.
"$VENV/bin/python3" - <<'PY' || { echo "[runner] deps broken — reinstalling" | tee -a "$LOG"; "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"; }
import pandas, pyarrow, numpy, scipy
PY

cd "$REPO"
echo "[runner] $JOB starting $STAMP (log: $LOG)" | tee -a "$LOG"
set +e
"$VENV/bin/python3" scripts/weekly_loop.py --job "$JOB" >>"$LOG" 2>&1
CODE=$?
set -e
echo "[runner] $JOB finished with exit $CODE" | tee -a "$LOG"

# After a successful grade pass, regenerate the public site from fresh data.
if [ "$JOB" = "grade" ] && [ "$CODE" -eq 0 ]; then
  "$VENV/bin/python3" scripts/build_site.py >>"$LOG" 2>&1 || \
    echo "[runner] site rebuild failed (loop itself succeeded)" | tee -a "$LOG"
  "$VENV/bin/python3" scripts/build_hub_feed.py >>"$LOG" 2>&1 || \
    echo "[runner] hub feed rebuild failed (site itself succeeded)" | tee -a "$LOG"

  # Publish: the Pages workflow deploys the COMMITTED data/site only, so an
  # unpushed rebuild is invisible to the public site. Commit+push exactly
  # that path; tolerate failure (publishing must never fail the grade job).
  # Disable with STALLIONS_NO_PUBLISH=1.
  if [ -z "${STALLIONS_NO_PUBLISH:-}" ]; then
    if git -C "$REPO" diff --quiet -- data/site && \
       [ -z "$(git -C "$REPO" ls-files --others --exclude-standard data/site)" ]; then
      echo "[runner] site unchanged; nothing to publish" | tee -a "$LOG"
    else
      ( git -C "$REPO" add data/site && \
        git -C "$REPO" commit -m "weekly: publish site + hub feed ($STAMP)" -- data/site && \
        git -C "$REPO" push origin master ) >>"$LOG" 2>&1 && \
        echo "[runner] site published (Pages deploy will pick it up)" | tee -a "$LOG" || \
        echo "[runner] site PUBLISH FAILED — data/site committed-or-dirty locally; push manually" | tee -a "$LOG"
    fi
  fi
fi

# Keep the last 60 logs; never let the log dir grow unbounded.
ls -t "$LOG_DIR" | tail -n +61 | while read -r f; do rm -f "$LOG_DIR/$f"; done

exit "$CODE"
