#!/bin/bash
# The ONLY thing in this repository that publishes. Run it by hand.
#
#     ./scripts/publish_site.sh            # shows what would go out, then asks
#     ./scripts/publish_site.sh --yes      # skip the prompt (still runs every check)
#
# Why this is separate from the weekly runner
# -------------------------------------------
# The runner used to commit `data/site` and then run `git push origin master`
# as the tail of a grade job. Four things were wrong with that, and all four
# are checked below rather than commented about:
#
#   1. The push was UNSCOPED. The commit was pathspec-limited to data/site,
#      but git cannot push one commit without its ancestors, so every unpushed
#      local commit on the branch went public as a side effect of grading.
#   2. On a feature branch the commit landed on that branch while the push
#      targeted master, so the runner logged "site published" and published
#      nothing.
#   3. The "nothing to publish" guard compared the worktree to the index, so a
#      single failed commit left data/site staged and wedged publishing
#      silently and permanently.
#   4. It ran unattended, from launchd, with no human looking at what went out.
#
# This script never runs from a scheduler. It refuses unless it is on master,
# it shows every commit the push would carry, and it stops on anything it does
# not understand.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
ASSUME_YES="${1:-}"

fail() { echo "[publish] REFUSED: $*" >&2; exit 1; }

# 1. Branch. A publish from anywhere but master is the silent no-op above.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "master" ] || fail "on branch '$BRANCH', not master.
  The Pages workflow deploys committed data/site from master only. Publishing
  from here would commit to '$BRANCH' and push nothing visible."

# 2. Is there anything to publish at all?
if git diff --quiet HEAD -- data/site && \
   [ -z "$(git ls-files --others --exclude-standard data/site)" ]; then
  echo "[publish] data/site matches HEAD and nothing is untracked — nothing to do."
  exit 0
fi

# 3. Everything the push would carry, not just the site commit.
AHEAD="$(git log --oneline origin/master..master || true)"
echo "[publish] repository: $REPO"
echo "[publish] branch:     $BRANCH"
echo
echo "[publish] changes to data/site that would be committed:"
git status --short -- data/site
echo
if [ -n "$AHEAD" ]; then
  echo "[publish] WARNING — these local commits are ALREADY unpushed and would"
  echo "          go public with this push, because git cannot push one commit"
  echo "          without its ancestors:"
  echo "$AHEAD" | sed 's/^/            /'
  echo
fi

# 4. Do not publish numbers the pipeline itself does not stand behind.
HEALTH="$REPO/data/pipeline_health.json"
if [ -f "$HEALTH" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$HEALTH" 2>/dev/null || echo "")"
  if [ "$STATUS" = "data_unavailable" ]; then
    fail "the last pipeline run halted on unhealthy inputs
  (data/pipeline_health.json says status=data_unavailable). Publishing now
  would push whatever the site last built over data the pipeline refused to
  stand behind. Fix the inputs and re-run the loop first."
  fi
  echo "[publish] pipeline health: ${STATUS:-unknown}"
fi

# 5. Confirm. --yes may only skip this when the push carries nothing but the
#    site commit: if local commits are already ahead of origin/master, this
#    push would carry them too (git cannot push one commit without its
#    ancestors), and that is exactly the unattended-unscoped-push failure
#    mode this script exists to prevent. A human must see the warning above
#    and type the confirmation themselves -- --yes does not stand in for that.
if [ -n "$AHEAD" ] && [ "$ASSUME_YES" = "--yes" ]; then
  echo "[publish] --yes cannot skip confirmation here: this push would carry"
  echo "          local commits unrelated to the site update (listed above)."
  ASSUME_YES=""
fi
if [ "$ASSUME_YES" != "--yes" ]; then
  printf "[publish] type 'publish' to commit data/site and push master: "
  REPLY=""
  read -r REPLY || true          # EOF (no operator present) must refuse, not error
  [ "$REPLY" = "publish" ] || fail "not confirmed"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
git add -- data/site
git commit -m "weekly: publish site + hub feed ($STAMP)" -- data/site
git push origin master
echo "[publish] pushed. GitHub Pages will deploy the committed data/site."
