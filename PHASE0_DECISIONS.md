# Phase 0 — build decisions log

Companion to `PHASE0_SMOKE_TEST.md` (the *what*) — this is the *why* behind
how the smoke test was built into runnable code on 2026-07-17. Append-only;
one heading per decision so a later session can see what was deliberate vs.
incidental.

## D1 — Runnable code, not just a brief
`PHASE0_SMOKE_TEST.md` was a hand-off prose brief. Turned it into an
executable `phase0_smoke_test.py` so the next data-equipped session runs one
command instead of re-deriving the steps. The brief stays as the human-readable
spec; the script is its faithful implementation (same endpoints, same
pass/fail criteria, same "surface surprises, don't route around them" stance).

## D2 — Raw HTTP, zero third-party deps
Used the standard library (`urllib`) instead of the official `cfbd` PyPI
client, matching the existing NFL `sources/` clients (`_http.py`,
`oddsapi.py`). Keeps the smoke test dependency-free and consistent with the
repo's "stdlib for I/O, pandas/numpy only for the modeling core" convention.
Trade-off: we hand-maintain endpoint shapes instead of getting typed models —
acceptable for a 4-endpoint smoke test, revisit if the production ingest wants
the generated client.

## D3 — Slice: Mountain West, 2023 (Curtis's call, 2026-07-17)
Membership-stable G5 conference the season *before* the July-2026 realignment
that reshuffled the Mountain West / rebuilt Pac-12. 2023 has modern PPA
coverage. This isolates a data-*shape* check from the data-*sparsity* problem a
mid-realignment conference would introduce (per
`research/conference-tier-and-scheduling.md`). Overridable via
`--conference` / `--year`.

## D4 — `/plays` is per-week, not a bulk season pull (assumption CORRECTED)
Verified against the CFBD API: `/plays` requires `year` + `week` +
`seasonType` and returns one week at a time. `/games`, `/lines`, `/talent` are
bulk (1 call/season each). So the portability plan's "most CFBD endpoints are
bulk season pulls" is true for three of four endpoints but **not** for
`/plays`. Budget impact: a full regular season costs ~15 `/plays` calls + 3
bulk calls ≈ **18 calls/season**, still trivially under the 1,000/month free
cap. The client counts every call so this is measured live, not assumed. This
is the first concrete correction the smoke test surfaced — exactly its purpose.

## D5 — Tolerate both field-name casings
CFBD has shipped both snake_case (`home_team`, `play_type`) and camelCase
(`homeTeam`, `playType`) across its API versions. Rather than hard-code one and
break on the other, every field read goes through `pick(row, *aliases)` which
tries known spellings. The smoke test reports which shape it actually saw, so
whoever runs it live can lock the real spelling into the fixtures afterward.

## D6 — Auth + key handling
Base URL `https://api.collegefootballdata.com`, `Authorization: Bearer <key>`
(verified 2026-07-17). Key resolution order: env `CFBD_API_KEY` >
`config.local.json` (gitignored) > `config.json` (tracked, discouraged). This
mirrors the existing `odds_api_key` pattern and honors the brief's "do not
commit the API key" rule. Added `config.local.json.example` as a template and
wired `cfbd_api_key` into `nflvalue/config.py`'s loader.

## D7 — Offline self-test + fixtures
No live key was available the session this was built, so correctness is proven
offline: pure `validate_*()` functions take raw API rows and return findings,
and `--selftest` + `tests/fixtures/cfbd_smoke_fixtures.json` +
`tests/test_phase0_smoke.py` exercise every check with no network. The
fixtures are hand-authored to the *current* camelCase shape and explicitly
flagged as fake — the intent is to overwrite them with the first real recorded
slice so the shape is pinned against future API drift. All 6 tests pass; the
mocked live path was verified to make the expected 18-ish calls.

## D8 — Verdict semantics
Three states, not two: PASS (all checks true), PARTIAL (some checks returned a
warn/None — e.g. sparse G5 moneylines, PPA present but below the "healthy"
threshold), FAIL (any hard-false — an endpoint returned nothing or a required
field is entirely missing). Exit code is 0 for PASS/PARTIAL, 1 for FAIL, so CI
or a scheduled run can gate on it. PARTIAL is deliberately non-fatal because
some G5 sparseness is expected reality, not a broken feed — but it's recorded
so it can't be silently ignored.

## Out of scope (unchanged from the brief, restated so it isn't re-litigated)
No rewrite of `nflvalue/ingest.py` / `scripts/bootstrap_history.py` yet, no
touch to `build_ratings.py` / `montecarlo.py`, no GitHub remote (still Curtis's
open decision), no key committed anywhere.

## What the next session does
Get a free CFBD key → `export CFBD_API_KEY=...` → `python phase0_smoke_test.py`
→ read `data/phase0_report.md` → if PASS/PARTIAL, save the real pulls into the
fixtures file and scope the `ingest.py`/`bootstrap_history.py` rewrite against
the confirmed shapes; if FAIL, the report says exactly which assumption broke.
Append the result to `_worklog.md` and update `hot.md`'s `pm_next_action`.
