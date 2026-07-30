# Weekly loop runbook — zero-credit operation

The weekly loop is plain Python. It does not need Claude, Cowork, or any
tokens to run — those were only used to *build* it. This runbook wires it to
macOS's native scheduler so the 2026 season runs itself for free.

## What runs when (CFB-shaped cadence)

| Job | When | What it does |
|---|---|---|
| `week-open` | Tue 10:00 | refresh cache (~20 calls), refit ratings, honesty report, simulate the upcoming week, render the explainer |
| `pregame` | Fri 16:00 | freshness check; re-simulate only if the cache moved |
| `grade` | Sun 09:00 | pull finals, honesty report, **weekly scoreboard recap + hypothesis ledger**, **close-free CLV capture**, rebuild the public site |

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

## What "it broke" looks like, and what to do

| Symptom | Meaning | Action |
|---|---|---|
| manifest `degraded: true` | a feed was stale/unreachable; nothing published over old data | usually self-heals next run; check the log |
| exit 1 + a FAILED stage in the manifest | one stage died; earlier stages are checkpointed | re-run the same job — it resumes, not restarts |
| "budget guard: refusing" | monthly API cap nearly spent | wait for the month to roll or check the tally for a runaway |
| "live lock held" | a previous run is still going (or crashed <2h ago) | nothing — the next scheduled run proceeds after the stale-lock window |

Nothing in this loop stakes anything, recommends anything, or needs
babysitting. It accumulates the live, out-of-sample record that the
kill-check (gate 7) will eventually be run against — once, pre-registered.
