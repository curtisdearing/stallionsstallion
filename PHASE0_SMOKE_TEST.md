# Phase 0 — CFBD smoke test

**Status: built, not yet run live (2026-07-17).** This brief is now backed by
runnable code — `phase0_smoke_test.py` (+ `nflvalue/sources/cfbd.py`) is a
faithful implementation of the steps below. The next data-equipped session just
needs a free CFBD key and one command:

```bash
export CFBD_API_KEY=xxxxx          # or copy config.local.json.example -> config.local.json
python phase0_smoke_test.py        # default slice: Mountain West 2023
python phase0_smoke_test.py --selftest   # offline dry run, no key/network
```

Results write to `data/phase0_report.{md,json}`; exit 0 = PASS/PARTIAL, 1 =
FAIL. Build decisions and the one assumption already corrected (`/plays` is
per-week, not bulk) are in `PHASE0_DECISIONS.md`. The prose steps below remain
the human-readable spec the code implements.

## Objective

Get a free CollegeFootballData (CFBD) API key and pull one season of
play-by-play, betting lines, and talent-composite data for a single
conference. Confirm the three endpoints this project's entire modeling plan
depends on (`/plays`, `/lines`, `/talent`) actually return the fields the
plan assumes. This is a ~30-minute sanity check, not a build — it exists to
de-risk everything downstream before any real engineering time goes into
`nflvalue/ingest.py` or `scripts/bootstrap_history.py`.

Full context: `research/cfb-portability-plan.md`'s "smallest testable
slice" (in its Phased Roadmap section) and `_SETUP.md`'s breakdown of what's
runnable vs. stubbed in this repo right now.

## Prerequisites

1. Sign up for a free CFBD API key at https://collegefootballdata.com — no
   payment info required for the free tier.
2. Free tier is capped at **1,000 calls/month**. Track how many calls this
   smoke test actually burns and report it — the plan assumes most CFBD
   endpoints are bulk season pulls rather than per-game, so this should cost
   far fewer than 1,000, but that's an assumption to confirm, not a given.
3. Either the official `cfbd` Python client or raw `requests` against
   `api.collegefootballdata.com` is fine for this smoke test — don't build
   the production ingest module yet (see "Explicitly out of scope" below).

## Pick one conference and one season

Recommend a conference with **stable membership** for this first pass —
not the rebuilt Pac-12 or any conference mid-realignment, since
`research/conference-tier-and-scheduling.md` already flags those as having
the thinnest cross-conference connectivity and would confound a data-shape
check with a data-*sparsity* problem. A mid-size, membership-stable
conference (e.g. Mountain West, Sun Belt, or the MAC) is a cleaner first
target than a Power Four conference or a freshly-shuffled one. Pick one
season recent enough to have modern PPA/advanced-stats coverage (2019 or
later; PPA/advanced-stats depth varies by era per the data source catalog).

## Steps

1. Pull `/plays` (or the PBP endpoint) for every team in the chosen
   conference for the chosen season. Confirm: PPA (CFBD's EPA equivalent) is
   present and non-null on a meaningful fraction of plays; down/distance/
   yard-line fields are populated; play type is a usable, groupable field
   (needed for standard-downs/passing-downs splits per
   `research/matchup-model-framework.md`).
2. Pull `/lines` for the same games. Confirm: multiple sportsbooks are
   represented per game (needed for de-vig/consensus math in `oddsmath.py`);
   spread, total, and moneyline are all present; check how far back
   real coverage actually starts (the plan notes "confirmed available from
   at least 2018 onward, exact earliest coverage unverified" — this smoke
   test is the place to verify it directly rather than keep citing that
   caveat).
3. Pull `/talent` for the same teams/season. Confirm: the 247 Team Talent
   Composite is present and the scale looks sane (should be roughly
   comparable across years, higher for blue-blood programs).
4. Spot-check one game manually against a public box score (ESPN or
   Sports-Reference) to confirm final score, team names, and game date all
   match what CFBD returns — a basic sanity check before trusting the feed
   for anything downstream.
5. Note anything that *doesn't* match what the plan assumes — field names,
   missing seasons, unexpected nulls, rate-limit behavior — rather than
   silently working around it. Surfacing a wrong assumption here is the
   entire point of this task.

## Explicitly out of scope for this task

- **Do not** rewrite `nflvalue/ingest.py` or `scripts/bootstrap_history.py`
  yet — those are legitimately NFL/nflverse-shaped internally and need a
  real rewrite, which is Phase 0's *next* task, not this smoke test.
- **Do not** touch `build_ratings.py`, `nflvalue/montecarlo.py`, or any
  ratings-engine code — this task is data-shape verification only.
- **Do not** create a GitHub remote or push this repo — that's a separate,
  deliberately-open decision (see the project's `hot.md`); ask Curtis first.
- **Do not** commit the CFBD API key anywhere in this repo. Use an
  environment variable or a local, gitignored config file
  (`nflvalue/config.py` already has a pattern for this from the odds-API
  key — follow it).

## Pass/fail criteria

**Pass:** `/plays`, `/lines`, and `/talent` all return usable data for the
chosen conference/season with the fields above populated, the manual
box-score spot-check matches, and the total call count is logged. Move on to
rewriting `nflvalue/ingest.py`/`bootstrap_history.py` against these
confirmed shapes.

**Fail (or partial):** any endpoint is missing an assumed field, returns
unexpectedly sparse data, or the box-score spot-check doesn't match. Write
up exactly what diverged from the plan's assumptions and where — this
becomes the first correction to `research/cfb-portability-plan.md`'s data
source catalog, not a blocker to route around silently.

## Report back

Whoever runs this: append one line to `_worklog.md` per the project's
worklog discipline (`- <utc> [s:session] KIND: summary || next: … || files:
…`) with the pass/fail result, the actual call count used, and any
field-level surprises. Update `hot.md`'s `pm_next_action` once this is done
— the next action becomes the `ingest.py`/`bootstrap_history.py` rewrite,
scoped by whatever this smoke test actually found.
