# Load the weekly loop into launchd — one paste

Ready-to-load agents for the three jobs in `RUNBOOK_WEEKLY.md`. All paths are
absolute for this Mac (`/Users/curtisdearing/...`); nothing needs editing.
The plists live in `ops/launchd/`; copies go to `~/Library/LaunchAgents/`
(launchd loads from there, so the repo copies stay the versioned source).

| Job | Label | Schedule (local ET) | CFBD calls per fire |
|---|---|---|---|
| week-open | `com.stallions.weekopen` | Tue 10:00 | up to 25 |
| pregame | `com.stallions.pregame` | Tue/Wed/Thu 16:00 · Fri 12:00 & 16:00 · **Sat 08:00, 13:00, 17:00** | **0** |
| grade | `com.stallions.grade` | Sun 09:00 | up to 25 |

The three Saturday fires are the final windows before the noon, afternoon and
night kickoff slates. Eight pregame fires a week sounds expensive and is not:
the pregame job has no refresh stage, so it spends **no CFBD budget at all**,
and it does real work only when the source fingerprint or the pending kickoff
window has actually moved. An unchanged tick writes a manifest that says it
skipped and exits 0. Monthly burn is unchanged from the old one-fire schedule:
roughly 200 of the 1,000 free calls.

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

## The grade job does NOT publish (changed)

`run_weekly.sh` runs no `git add`, no `git commit` and no `git push`. After a
successful grade pass it rebuilds `data/site/` and the hub feed **locally** and
reports whether they are ready:

```
[runner] local site + hub feed rebuilt; READY TO PUBLISH
[runner] publish with: ./scripts/publish_site.sh   (nothing was pushed)
```

A site or hub build failure now makes the run **unsuccessful** (exit 4) instead
of being logged and swallowed. `STALLIONS_NO_PUBLISH` no longer exists — there
is nothing to opt out of.

Publishing is one explicit human command, `./scripts/publish_site.sh`, which
refuses unless it is on `master`, prints every unpushed commit the push would
carry, refuses outright if the last pipeline run halted on unhealthy inputs,
and asks for typed confirmation before it commits `data/site` and pushes.

> **Why this changed.** The runner used to commit `data/site` and then run an
> unscoped `git push origin master` at 09:00 on a Sunday, unattended. The
> commit was pathspec-scoped but the push was not, and git cannot push one
> commit without its ancestors — so **every unpushed local commit went public
> as a side effect of grading a week**. Two further sharp edges, both
> reproduced: the commit landed on whatever branch was checked out while the
> push always targeted `master` (on a feature branch the runner logged "site
> published" and published nothing), and the "nothing to publish" guard
> compared worktree to index, so one failed commit left `data/site` staged and
> silently wedged publishing forever. `publish_site.sh` checks all three.

## Exit statuses

| Code | Meaning | What to do |
|---|---|---|
| 0 | job completed | nothing; publish by hand if you want the site live |
| 1 | a stage failed | re-run the same job — it resumes from its sentinels |
| 3 | **halted: inputs failed their health check** | read `halt_reason` in the manifest and `data/pipeline_health.json`; nothing was published and nothing was overwritten |
| 4 | local site/hub build failed | the loop itself succeeded; the artifacts are not publishable |

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


---

## What a pregame fire actually does (new 2026-08-28)

The old behaviour was one run per week: the first pregame fire wrote
`data/manifests/_done_{season}_wk{NN}_pregame_simulate` and every later fire
that week skipped. A Saturday morning run could not react to a Friday injury
report. That sentinel no longer applies to `pregame`
(`weekly_loop.SENTINEL_JOBS`); repeat-suppression is now decided from the
snapshot store instead.

Each fire:

1. computes a **source fingerprint** over the week's schedule, the cached
   source artifacts (games / lines / drives / teams / rosters / availability /
   weather), and the code+config version;
2. computes the **pending kickoff window** — the next kickoff that has not
   happened yet, floored to the hour;
3. runs only if either moved since the last snapshot (or if the last
   publication was degraded and sources are healthy again);
4. **freezes** every game that has kicked off, copying its row from the newest
   snapshot taken *strictly before that game's own kickoff*;
5. writes an immutable snapshot and advances the `latest` pointer.

Layout under `data/snapshots/{season}_wk{NN}/`:

```
{stamp}/predictions.csv     immutable — rewriting a stamp raises
{stamp}/snapshot.json       digest, window, degraded, sha256, taken_at
{stamp}/explainer.html      optional
latest.json                 POINTER to the newest stamp (never a copy)
frozen.json                 append-only game_id -> final pregame row
```

`data/predictions_{season}_wk{NN}.csv` is still written for the site and hub
feed; it is now just a copy of the newest snapshot, written tmp+`os.replace`.

**Storage policy — REVERSED, and this is now the single source of truth.**
`data/snapshots/` is **tracked**. An earlier revision put it in `.gitignore`
and described it as MACHINE-LOCAL: never committed, never pushed. That made the
only pre-kickoff record exist on exactly one Mac, which defeats the point of
freezing at kickoff — the value of a frozen row is that it can be *proved*
afterwards, and a proof nobody else can see is not one. It also meant a fresh
clone started with no frozen record, so any already-started game there
published as a refusal rather than the forecast that really was made.

What is tracked: `predictions.csv` (contract rows), `snapshot.json` (stamp,
source digest, window, sha256, taken_at, status counts), `frozen.json` (the
per-game pre-kickoff rows and their provenance), `latest.json` (the pointer).
Roughly 60 KB per snapshot, a few MB a season.

What is NOT tracked: `explainer.html` inside each stamp
(`data/snapshots/**/explainer.html` in `.gitignore`) — it is a rendering of
`predictions.csv`, regenerable from it, and it is most of the bytes.

**No credential goes in.** `snapshot.json` stores the source digest, which is a
hash — never the config it hashes. The CFBD key and any webhook live in
`config.local.json`, which stays ignored. Nothing in a snapshot is
secret-bearing, and nothing in it is CFBD source data: the rows are derived
analysis, consistent with the recorded terms position.

**What is on this Mac right now does not match that policy** (checked
2026-09-02T02:09Z; `git ls-files data/snapshots` returns nothing).

- Nothing under `data/snapshots/` is tracked. The policy above is the intent;
  no snapshot has been committed under it yet.
- There is no `data/snapshots/2026_wk01/` directory at all, so no `latest.json`
  and no `frozen.json` for the published week. The committed artifact
  `data/predictions_2026_wk01.csv` carries `snapshot_id 20260902T005703Z` on
  all 455 rows — a stamp with no store behind it here, because the run that
  produced it executed in a sandbox whose store was never brought back. Until
  that store is recovered or the week is re-run on this Mac, the published
  rows cannot be verified against the snapshot they name, and
  `SN.list_snapshots('data', 2026, 1)` returns nothing.
- What is in `data/snapshots/` is four loose files written between
  2026-08-28T05:07Z and 06:16Z, in the pre-store layout: `pregame_2026-08-29_20260828T050735Z.json`
  (`stallionsstallion.pregame_snapshot.v1` — 74 games for the 08-29 slate, 6
  forecast, 68 refused, plus a whole-week 408/49/359 count), its
  `.provenance-addendum.json` (records that the snapshot's `git_sha d3bd4e9`
  was superseded by the committed `2d491c2`, without rewriting the immutable
  file), the human packet `stallionsstallion_2026-08-29_forecast_packet.md`,
  and `context_panel_2026-08-29.html`. They predate the
  `{season}_wk{NN}/{stamp}/` layout the policy describes and are not read by
  `nflvalue/snapshots.py`. They are an archival record of the 08-29 slate, not
  a store this code can load. **Nothing here moves, renames or deletes them** —
  whether they are committed as-is, or moved into the store layout, is
  Curtis's decision.

## Inspecting a week without running anything

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
from nflvalue import snapshots as SN
for s in SN.list_snapshots('data', 2026, 1):
    print(s['stamp'], s['window'], 'degraded' if s['degraded'] else 'ok',
          s['n_rows'], 'rows', 'verified' if SN.verify_snapshot(s['path']) else 'TAMPERED')
print('frozen:', len(SN.load_frozen('data', 2026, 1)))
PY
```
