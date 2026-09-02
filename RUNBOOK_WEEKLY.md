# Weekly loop runbook — zero-credit operation

The weekly loop is plain Python. It does not need Claude, Cowork, or any
tokens to run — those were only used to *build* it. This runbook wires it to
macOS's native scheduler so the 2026 season runs itself for free.

## What runs when (CFB-shaped cadence)

| Job | When | What it does |
|---|---|---|
| `week-open` | Tue 10:00 | refresh cache (~20 calls), refit ratings, honesty report, simulate the upcoming week, render the explainer |
| `pregame` | Fri 16:00 | freshness check; re-simulate only if the cache moved |
| `grade` | Sun 09:00 | pull finals, honesty report, **weekly scoreboard recap + hypothesis ledger**, **close-free CLV capture**, rebuild the site + hub feed **locally** |

**Which week `grade` grades.** The loop's week is the lowest week with a
kickoff still in the future, and the grade job recaps and captures CLV for the
week *before* that one. A week ending in a Monday game is therefore still
current on the Sunday before it, and that Sunday's run grades the week before
— quietly. What that costs depends on the week: in week 1 the target is week
0 and nothing is graded; in any later week the target is a week already
recapped the previous Sunday, so the run re-grades it, reports a plausible
non-zero n, and the week that just finished waits another seven days. Every
recap and CLV stage return names its `target_week` in the manifest, so check
there rather than inferring from n. To grade week N by hand, run
`--job grade --week N+1`: `--week` sets the week the loop is in, not the week
to grade. The refresh stage pulls finals either way.

Every job starts with a dependency-aware **input health check**
(`nflvalue/data_health.py`) and **halts before publishing anything** if the
inputs its engine actually depends on are missing, malformed, empty, stale,
future-dated, or do not cover the target week. Nothing in this pipeline
publishes to the internet on its own — see "Publishing" below.

Costs ~25 CFBD calls/week against the 1,000/month free cap; the loop's
budget guard hard-refuses within 50 calls of the cap. In the offseason every
job exits 0 immediately (manifest says "offseason").

## One-time setup (5 minutes)

```bash
cd ~/Documents/Side-Projects-Vault/20-projects/stallionsstallion/stallionsstallion
chmod +x scripts/run_weekly.sh
./scripts/run_weekly.sh week-open        # first run builds the venv, then executes
```

Requirements already in place: `config.local.json` holds the CFBD key
(gitignored — never commit it); the parquet cache ships with the repo dir.
Optional: set `discord_webhook` in `config.local.json` to get a message on
every run (including failures).

## Scheduling with launchd (survives reboots; preferred over cron on macOS)

Create three files in `~/Library/LaunchAgents/`, changing only the label,
the job argument, and the calendar block. Template
(`com.stallions.weekopen.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.stallions.weekopen</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string>
    <string>/Users/curtisdearing/Documents/Side-Projects-Vault/20-projects/stallionsstallion/stallionsstallion/scripts/run_weekly.sh</string>
    <string>week-open</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>2</integer>
        <key>Hour</key><integer>10</integer>
        <key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>/tmp/stallions-weekopen.log</string>
  <key>StandardErrorPath</key><string>/tmp/stallions-weekopen.log</string>
</dict></plist>
```

Then: `launchctl load ~/Library/LaunchAgents/com.stallions.weekopen.plist`
(repeat for `pregame` → Weekday 5 / Hour 16, and `grade` → Weekday 0 /
Hour 9). `launchctl list | grep stallions` confirms. Note launchd skips a
run if the Mac is asleep at the trigger minute — the `grade` job is
idempotent, so a missed Sunday run just executes Tuesday inside `week-open`'s
window with no double-counting (sentinels + the lock handle it).

## What "it worked" looks like

- `data/manifests/{season}_wk{NN}_*.json` — one per run, every stage ok
- `data/recaps/{season}_wk{NN}.md` — Sunday's human-readable recap
- `data/clv_live_{season}.csv` — the live paper CLV log growing weekly
- `data/site/index.html` — the website, regenerated with fresh numbers
- `data/logs/` — dated logs, auto-pruned to the last 60
- `../picks/{season}_wk{NN}_every_fbs_game.md` (+ `.csv`) — the vault's
  every-FBS-game page: projected winner, win probability, 80% / 95%
  predictive intervals, rationale, caution. Rendered read-only from the
  artifact and the cards by `scripts/render_picks.py --auto` after every
  successful `pregame` / `week-open`; a private note, never published.
  Re-render by hand with `python3 scripts/render_picks.py --season S --week W`
  (needs `data/insights_S_wkNN.json`; `scripts/render_insights.py` makes it).

## What "it broke" looks like, and what to do

| Symptom | Meaning | Action |
|---|---|---|
| exit 3 + `halted: true` | the input health gate blocked a capability this job needs | read `halt_reason` in the manifest and `data/pipeline_health.json`; **nothing was published, `latest` did not move, no artifact was overwritten** |
| exit 4 | the loop succeeded but the local site/hub build failed | the artifacts are not publishable; fix the build before running `publish_site.sh` |
| exit 1 + a FAILED stage in the manifest | one stage died; earlier stages are checkpointed | re-run the same job — it resumes, not restarts |
| "budget guard: refusing" | monthly API cap nearly spent | wait for the month to roll or check the tally for a runaway |
| "live lock held" | a previous run is still going (or crashed <2h ago) | nothing — the next scheduled run proceeds after the stale-lock window |

Nothing in this loop stakes anything, recommends anything, or needs
babysitting. It accumulates the live, out-of-sample record that the
kill-check (gate 7) will eventually be run against — once, pre-registered.


---

## Snapshot cadence and idempotency (added 2026-08-28)

`pregame` is now a Tuesday-through-Saturday job with eight fires a week
(see `ops/LOAD_ME.md`), and it is idempotent by fingerprint rather than by a
once-per-week sentinel.

**When a pregame fire does real work**

| Trigger | Example |
|---|---|
| no snapshot yet this week | the Tuesday open |
| source fingerprint moved | schedule change, new kickoff time, refreshed lines / rosters / availability / weather cache, changed config or git sha |
| pending kickoff window advanced | the noon slate started, so the next pending kickoff is now the afternoon one |
| last publication was degraded and sources are healthy again | recovery after a stale feed |

Anything else exits 0 having written a manifest whose `simulate` stage reads
`{"ok": true, "skipped": true, "reason": "source fingerprint and kickoff
window unchanged since <stamp>"}`.

**What can and cannot change after kickoff**

A game's row freezes the moment its own kickoff passes. The frozen row is
copied from the newest snapshot taken *strictly before that kickoff*, and the
provenance (`from_stamp`) is recorded in `frozen.json`. Later fires re-publish
it verbatim. A game that kicked off with no pre-kickoff snapshot is frozen as
an explicit refusal — never back-filled, because a number computed after
kickoff is not a forecast. This is the structural form of BUILD_SPEC §0.4.

**Grade resumability**

The grade sentinel now carries a fingerprint of which games in the target week
have final scores. When another game goes final the fingerprint changes and the
stage reopens, so a grade run that fired before the night slate finished is
resumed rather than suppressed for the rest of the week.

**Unhealthy inputs (rewritten)**

A failed health check no longer publishes anything at all. The previous
behaviour — write the snapshot anyway, mark it `degraded`, bannered — still
advanced the `latest` pointer and overwrote the published artifact, which is
the one thing a run over unchecked data must not do.

Now the run halts at the health stage and:

- writes the manifest with `halted: true`, `halt_reason`, and
  `blocked_capabilities`, and marks every later stage explicitly skipped;
- writes `data/pipeline_health.json` — a deliberately **non-numeric** public
  status (`status: data_unavailable`, a headline, the blocked capabilities and
  the exact reasons; no margin, no probability, no counts of "current"
  forecasts). The hub feed forwards it verbatim as `health`;
- notifies, and exits **3**.

The `latest` pointer does not move, no snapshot is written, and
`data/predictions_{season}_wk{NN}.csv` is left exactly as it was.

**What is checked, and what merely suppresses a claim**

| Input | Role |
|---|---|
| `games`, `teams` | load-bearing for every job: schedule, results, and the FBS classification the refusal rule reads |
| `games` history | load-bearing: prior seasons the ratings and the preseason prior are fitted on |
| `drives` + history | **hard block for drive simulation** (weeks 4+). Missing or stale drives mean the Monte Carlo would be fitted on a stale picture of the season |
| `forecast_sd_basis.json`, `backtest_early.csv` | hard block: no row may carry a forecast SD without its registered out-of-sample basis |
| `backtest_c.csv`, `clv_log_backtest.csv` | **hard block for the claims that cite them** — the explainer quotes the registered model MAE and CLV slices back at the reader as evidence |
| `lines` | **optional**. No forecast depends on a market line, so its absence never blocks a run; it suppresses the market comparison and CLV capture, and the status file lists them under `suppressed_insights` |

Historical caches carry no staleness threshold: a finished season is an
immutable record, and calling `games_2021` stale after eight days would block
every run forever. Only the current season ages.

**Publishing (changed — read this)**

`run_weekly.sh` runs **no git command**. It builds local artifacts and reports
readiness. Publishing is one explicit manual command:

```bash
./scripts/publish_site.sh          # shows what would go out, then asks
```

It refuses unless you are on `master`, prints every unpushed commit the push
would carry, refuses outright if the last run halted on unhealthy inputs, and
requires typed confirmation. The old behaviour — an unattended, unscoped
`git push origin master` at the tail of a Sunday grade job — could publish
every unpushed local commit as a side effect of grading a week.

A site or hub build failure now makes the run **unsuccessful** (exit 4). It
used to be logged and swallowed, so the runner reported success while the
public surface silently kept serving the previous week.

**Reading the record**

(These paths exist once a run on this Mac has written a store for the week;
see the note below on the current state of `data/snapshots/`.)

```bash
ls data/snapshots/2026_wk01/                 # every snapshot for the week
cat data/snapshots/2026_wk01/latest.json     # pointer to the newest
cat data/snapshots/2026_wk01/frozen.json     # per-game final pregame rows
cat data/pipeline_health.json                # last input-health status
```

**The snapshot store is tracked in git** — that is the policy, and
`ops/LOAD_ME.md` carries it in full, including what is deliberately excluded
(the rendered `explainer.html`, regenerable from `predictions.csv`) and why
nothing in the store is secret-bearing. It is the immutable pre-kickoff record
and the reason a frozen row can be proved after the fact; a record that lives
on one Mac only cannot be audited by anyone.

**Nothing is tracked under it yet, and week 1's store is missing** (checked
2026-09-02). `data/snapshots/2026_wk01/` does not exist on this Mac even
though the committed week-1 artifact names `snapshot_id 20260902T005703Z`, so
that week's published rows cannot currently be verified against the snapshot
they cite. `ops/LOAD_ME.md` records the exact state and what is Curtis's call.

**Integrity is fail-closed.** Corrupt or missing snapshot metadata raises
`SnapshotIntegrityError` instead of reading as "no snapshot": an unreadable
`frozen.json` used to mean "no game has kicked off", which would let a started
game be re-forecast. The single-writer lock is taken with one atomic
`O_CREAT|O_EXCL` syscall, and the `latest` pointer moves by `os.replace`.

Still paper-only. Gate 7 has not run; nothing here is a bet recommendation.
