# stallionsstallion — weekly run loop spec (Workstream I, deep dive)

**What this is:** the full spec for turning `scripts/auto_weekly.py` +
`scripts/state_store.py` into the loop `hot.md`'s own next-action line
describes: pull new data each week, refit ratings (Workstream B), rerun the
simulation (Workstream C), log closing-line value (Workstream D). BUILD_SPEC
section 13 states the requirement in six sentences; this is the six sentences
worked all the way through, against the code that actually exists today.

**Status when written (2026-07-20):** gates 0-2 (Phase 0, A-ingest, B-ratings)
pass. C (simulation), D (market/CLV), F (explainer) are unbuilt. That means
this spec has a real Phase 1 you can build now and a Phase 2 that is honestly
blocked — see §0. Read `AGENT_BUILD_PROMPT.md` first if you have no prior
context.

---

## 0. Why this can't be built as one shot, and what changes when it can

BUILD_SPEC's stage list for the weekly loop is: refresh ingest → freshness
gate → refit ratings → simulate → pull market → blend → log CLV candidates →
render explainer → notify. Three of those nine stages (simulate, blend, log
CLV) don't have code to call yet. Building the full loop today would mean
stubbing three stages and shipping something that *looks* complete — exactly
the "assumed success without verifying" failure mode `AGENT_BUILD_PROMPT.md`
ranks #1. So this spec is phased on purpose:

- **Phase 1 (buildable now):** refresh ingest → freshness gate → refit
  ratings → **honesty report** (walk-forward MAE and margin-correlation vs.
  the market close, extending `scripts/backtest_b.py`'s method to run
  incrementally week-by-week instead of only as a one-shot historical
  backtest) → notify. No CLV, because there is no D yet to produce it.
- **Phase 2 (after C, D land):** insert simulate + pull market + blend + log
  CLV candidates between "refit ratings" and "notify". The orchestration,
  manifest, locking, and resumability built in Phase 1 do not change shape —
  Phase 2 is new stages slotted into an existing skeleton, not a rewrite.

Building Phase 1 now is what makes "does the re-fit change anything" a
weekly, checked fact instead of a one-off finding from 2026-07-19's session.

---

## 1. What already exists and what's actually wrong with it

`scripts/auto_weekly.py` is a real, working self-scheduling wrapper — ported
whole from fablesfable (NFL). It is not a stub. But it is NFL-shaped in a way
that doesn't just need renaming:

- **Job names encode the NFL's weekly rhythm.** `--job wed` / `--job t90` /
  `--job tuesday` assume one script-open day, one game window (Sun/Mon), one
  grading day (Tuesday). CFB does not have this shape: games run Tuesday
  through Saturday in season, with **Saturday alone carrying roughly
  two-thirds of the week's slate** (~60+ of ~130 FBS games most weeks) and a
  handful scattered Tue-Fri (MACtion, occasional Friday G5/service-academy
  games). A single "tuesday" grading job would grade Saturday's 80 games a
  full 48+ hours after they're final, and a single "t90" job can't cover
  pregame windows that are smeared across five days instead of clustered on
  two.
- **`state_store.py`'s `STATE_GLOBS` lists NFL/prop artifacts that don't
  exist in this repo**: `data/ml_frame.parquet`, `data/weekly_props.json`,
  `data/weights.json`. This repo's actual state lives in
  `data/rating_params.json`, `data/ratings_*.parquet`,
  `data/cfbd/_call_tally.json`, and (once D exists) a CLV log. Left
  unedited, `pack()` silently archives nothing that matters — it would
  "succeed" and back up an empty shell. This is a correctness bug waiting to
  be found the hard way, not a hypothetical.
- **No single-writer lock exists yet.** BUILD_SPEC requires one; nothing in
  `auto_weekly.py` or `state_store.py` currently provides it.

None of this means auto_weekly.py is the wrong starting point — its
self-scheduling design (figures out season/week/games itself, exits clean
code 0 in the offseason, Discord notify already wired) is exactly right and
should be kept. The job *names and cadence* need to be redrawn around CFB's
actual calendar, and `STATE_GLOBS` needs to point at files that exist.

---

## 2. Build — Phase 1

**Redraw the job cadence around CFB's week, not the NFL's:**
- `week-open` (Tuesday, once new week's games post) — freshness-gated ingest
  refresh for the completed week, refit ratings over the full graph to date,
  write the honesty report, notify.
- `pregame` (runs multiple times Tue-Sat, not once) — parameterized by a
  lookback window per day rather than one fixed T-90 check, since kickoffs
  are spread across the week. Reuses the existing t90-style freshness logic,
  called on a schedule dense enough to cover a Tuesday MACtion window and a
  Saturday noon/afternoon/night wave without needing five hand-maintained
  job names.
- `grade` (runs through Saturday night into Sunday, resumable, not a single
  Tuesday pass) — pulls final scores for games that have gone final since
  the last grade pass; idempotent so a partial Saturday-night run and a
  Sunday-morning cleanup run don't double-count.

**Orchestration, Phase 1 stage list:**
```
refresh ingest (nflvalue.ingest, checkpoint=True)
  -> freshness gate (nflvalue.freshness)
  -> refit ratings (nflvalue.ratings, walk-forward, warm-started from last week)
  -> honesty report (extend scripts/backtest_b.py: this week's games only,
     model MAE/corr vs. market close, vs. last week's same numbers)
  -> notify (existing Discord path in auto_weekly.py)
```

**Run manifest**, one JSON per execution, `data/manifests/{season}_wk{week}_{utc_stamp}.json`:
config hash, data-as-of timestamps per source table, git commit sha (now
available — `git rev-parse HEAD`), random seed used in the ratings solve,
CFBD monthly call count before/after, and every stage's pass/fail. Any
number this loop publishes must trace to one of these. `state_store.py`'s
`pack()` should archive the current manifest alongside the state files it
already globs.

**Single-writer lock:** a plain lock file, `data/.weekly.lock`, containing
PID + UTC start timestamp + job name. A run that finds a live lock exits
clean with a logged reason rather than racing. Stale-lock override after a
configurable timeout (default 2 hours — long enough for a real run,
short enough that a crashed process doesn't permanently jam the schedule).
This is a local cron/Cowork-scheduled-task context, not a distributed
system — a file lock is the right amount of engineering, not a queue or a
distributed lock service.

**Idempotent and resumable, by construction, not by hope:** reuse the exact
checkpoint pattern `nflvalue/ingest.py`'s `refresh_season(checkpoint=True)`
already proved out against the sandbox's 45-second kill
(2026-07-19T22:45Z worklog entry) — per-endpoint and per-week checkpoints
written atomically to `data/cfbd/_ckpt/`, promoted on success. The weekly
loop's own stages (refit, honesty report) should checkpoint the same way:
write a stage-complete sentinel after each stage's atomic output write, and
have the loop's entry point check sentinels before re-running a stage that
already finished this week.

**Budget awareness:** before running, read `data/cfbd/_call_tally.json` and
refuse (not degrade — refuse, loud) if the projected cost of this run would
push the monthly count within, say, 50 calls of the 1,000 cap, mirroring
`bootstrap_history.py`'s existing hard-refuse pattern. A regular-season
weekly refresh is cheap (single-digit calls, far under the ~20/season
backfill cost) but the tally is shared across ingest, and later, Odds API
and weather calls — the guard should look at the whole tally, not just this
job's own expected cost.

**Fix `state_store.py`'s `STATE_GLOBS`** to the files this repo actually
produces: `data/rating_params.json`, `data/ratings_*.parquet`,
`data/ratings_units_*.parquet`, `data/cfbd/_call_tally.json`,
`data/manifests/*.json`, and (added when D lands) the CLV log. Drop the
NFL/prop-only entries or gate them behind a check that they exist before
globbing, so a fresh CFB checkout doesn't warn about files it will never
have.

## 3. Build — Phase 2 (unblocked once C and D exist)

Insert between "refit ratings" and "notify": simulate (Workstream C,
drive-level MC off the fresh ratings) → pull market (multi-book open/close)
→ blend (nfelo-style, Workstream D) → log CLV candidates (slice-tagged).
Notify gains the CLV summary; the honesty report becomes the full weekly
scoreboard (see `WEEKLY_SCOREBOARD_SPEC.md`). No change to the lock,
manifest, or checkpoint design — they were built stage-agnostic on purpose.

---

## 4. Hardening

- **Never publish over stale data.** If the freshness gate fails, the loop
  still writes a manifest (marked `degraded: true`) and still notifies, but
  the notification and any dashboard render say so visibly — reuse the
  existing `degraded` badge pattern in `dashboard.py`. A silent skip is
  worse than a loud degraded run.
- **A crash mid-week resumes, not restarts.** Test this directly: kill the
  process mid-refit and assert the next invocation picks up from the last
  completed stage sentinel, not from `week-open` again.
- **Reproducibility:** re-running the same manifest (same config hash, same
  data-as-of, same seed) must reproduce identical ratings output — this is
  already asserted for the ratings solve itself (gate 2's fingerprint test);
  the loop-level test is that two full Phase-1 runs against the same frozen
  cache produce byte-identical `rating_params.json` and honesty-report
  numbers.
- **Notify on failure, not just success.** A stage that fails should push a
  Discord message saying which stage and why, not just go silent — the
  existing `notify.py`/webhook path already exists, this is a call site, not
  new infrastructure.

## 5. Optimization

Warm-start the weekly refit from last week's solved parameters rather than
re-solving from scratch (Workstream B's own optimization note already asks
for this — this loop is the thing that actually calls it weekly). Cache the
ridge design matrix by `(season, through_week)` so a re-render or a retry
after a crash doesn't refit. Profile before assuming the sim (once built) is
the bottleneck; today, ingest's per-week `/plays` calls are the slowest
stage by wall clock, not by API budget.

## 6. Definition of Done

- Phase 1 runs unattended end-to-end on a real week (a scheduled or manually
  triggered `week-open` job followed by `grade`), producing a manifest and a
  Discord notification, with no human intervention.
- A deliberately killed mid-run process resumes cleanly on next invocation —
  demonstrated, not asserted.
- A deliberately stale ingest cache produces a `degraded`-flagged manifest
  and a visibly degraded notification, never a confident-looking normal one.
- Two runs against the same frozen inputs produce byte-identical outputs.
- `state_store.py pack` on a live repo actually contains the files that
  matter — checked by unpacking into a scratch dir and diffing against
  `data/`.
- `data/.weekly.lock` prevents a second concurrent run from racing the
  first — demonstrated with two invocations launched back to back.

## 7. Kickoff prompt

> Implement `WEEKLY_LOOP_SPEC.md` Phase 1. Redraw `scripts/auto_weekly.py`'s
> job cadence around CFB's actual week (Tuesday week-open, a pregame check
> that runs across Tue-Sat rather than once, a grade pass that covers
> Saturday-night-into-Sunday) instead of the inherited NFL wed/t90/tuesday
> names. Wire: freshness-gated ingest refresh → warm-started ratings refit →
> a walk-forward honesty report (extend `backtest_b.py`) → notify. Add a
> single-writer lock file, a per-run manifest (config hash, data-as-of,
> git sha, seed, stage pass/fail), and reuse `ingest.py`'s existing
> checkpoint pattern for resumability. Fix `state_store.py`'s `STATE_GLOBS`
> to this repo's real state files. Do not touch Workstreams C, D, or F. Do
> not log CLV — there is no D yet to produce it. Append to `_worklog.md` and
> update `hot.md` when done, including the exact stage list that ran and
> its manifest path.

**Nothing in this spec authorizes staking real money or implies one.** The
honesty report is a model-vs-market diagnostic, not a betting signal — that
distinction stays true even once Phase 2 adds CLV. Gate 7 (Phase 4
kill-check) is still the only thing that changes that.
