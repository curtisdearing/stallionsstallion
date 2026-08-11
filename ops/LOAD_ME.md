# Load the weekly loop into launchd — one paste

Ready-to-load agents for the three jobs in `RUNBOOK_WEEKLY.md`. All paths are
absolute for this Mac (`/Users/curtisdearing/...`); nothing needs editing.
The plists live in `ops/launchd/`; copies go to `~/Library/LaunchAgents/`
(launchd loads from there, so the repo copies stay the versioned source).

| Job | Label | Schedule (local) |
|---|---|---|
| week-open | `com.stallions.weekopen` | Tue 10:00 |
| pregame | `com.stallions.pregame` | Fri 16:00 |
| grade | `com.stallions.grade` | Sun 09:00 |

## Load (paste this whole block)

```bash
cd ~/Documents/Side-Projects-Vault/20-projects/stallionsstallion/stallionsstallion
chmod +x scripts/run_weekly.sh
mkdir -p data/logs ~/Library/LaunchAgents
cp ops/launchd/com.stallions.{weekopen,pregame,grade}.plist ~/Library/LaunchAgents/
for j in weekopen pregame grade; do
  launchctl bootout  gui/$(id -u)/com.stallions.$j 2>/dev/null || true   # idempotent reload
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.stallions.$j.plist
done
launchctl list | grep stallions
```

Expected: three `com.stallions.*` lines (status `-` / exit `0` until first run).

## Verify the first run (don't wait for Tuesday)

```bash
launchctl kickstart -k gui/$(id -u)/com.stallions.grade
sleep 20
ls -t data/logs | head -3
tail -5 "data/logs/$(ls -t data/logs | grep '^grade_' | head -1)"
python3 -c "import glob,json;p=sorted(glob.glob('data/manifests/*.json'))[-1];print(p);print(json.dumps(json.load(open(p))['stages'],indent=1))"
```

What "it worked" looks like right now (offseason): the grade log ends with
`[runner] grade finished with exit 0`, and the newest manifest's stages show
`offseason_check: {"ok": true, "note": "no games within 7 days — clean
offseason exit"}`. In season the same command shows every stage `ok: true`.
(Verified 2026-08-11 in a sandbox dry-run: exit 0, manifest says offseason.)

## Where logs land

- `data/logs/{job}_{UTCSTAMP}.log` — one dated log per run, written by
  `run_weekly.sh`, auto-pruned to the newest 60.
- `data/logs/launchd-{weekopen,pregame,grade}.log` — launchd's own
  stdout/err capture (venv bootstrap chatter, crashes before logging starts).
- `data/manifests/{season}_wk{NN}_{stamp}.json` — the authoritative per-run
  record: per-stage pass/fail, degraded flag, call tally before/after.

## How the grade job publishes

After a successful grade pass, `run_weekly.sh` rebuilds the site + hub feed,
then **commits and pushes exactly `data/site/`** to `origin master`
(message `weekly: publish site + hub feed (<stamp>)`). The push triggers
`pages.yml`, which deploys the committed `data/site/` — that is the only
publish path; the CFBD key never touches CI. A publish failure logs loudly
but never fails the grade job. Opt out by adding
`<key>STALLIONS_NO_PUBLISH</key><string>1</string>` to the
`EnvironmentVariables` dict in `com.stallions.grade.plist` (then reload it).
The push uses your normal git credentials (keychain) — run one manual
`git push` from a terminal first if the keychain hasn't cached them.

## Manage

```bash
launchctl kickstart -k gui/$(id -u)/com.stallions.weekopen   # run any job now
launchctl bootout gui/$(id -u)/com.stallions.grade           # unload one job
launchctl print gui/$(id -u)/com.stallions.grade | head -20  # inspect state
```

Missed-run semantics (RUNBOOK_WEEKLY.md): launchd skips a trigger if the Mac
is asleep; the jobs are idempotent and sentinel-guarded, so a missed run is
absorbed by the next one — no double-counting. ~25 CFBD calls/week; the
loop's budget guard hard-refuses within 50 calls of the 1,000/month cap.
Paper-only throughout: nothing in these jobs stakes or recommends anything.
