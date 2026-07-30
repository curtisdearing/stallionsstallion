# stallionsstallion — weekly scoreboard & recap spec

**What this is:** the spec for a recurring artifact — generated once a
week's games are final — that says plainly how the model did, why, and what
to watch for next time. It extends BUILD_SPEC's F6 (slice-level CLV
scoreboard, a live panel) with something F6 doesn't cover: a dated, written
recap with a memory, in the same spirit as this project's own worklog
discipline applied to football performance instead of engineering sessions.

**Status when written (2026-07-20):** depends on `WEEKLY_LOOP_SPEC.md`
Phase 1 (to run every week) and, for the full version, Workstreams C and D.
Phase 1 of *this* spec is buildable today on top of B alone — see §0.

---

## 0. The one design risk that matters more than the format

Curtis's request was for the scoreboard to "note trends that happened to
follow next time" — i.e., for the recap to carry a memory forward and check
whether what it flagged actually mattered. Built carelessly, that is exactly
the failure mode BUILD_SPEC's non-negotiable #6 exists to prevent: a
situational factor (or a "trend") quietly earning influence over a
probability without ever clearing the n-gate + matched-control + BH q<0.05
bar. A weekly recap that silently nudges next week's numbers because last
week's write-up sounded convincing is an ungated feedback loop wearing a
narrative disguise — worse than a normal ungated factor, because it *looks*
like accountability rather than a shortcut.

So this spec draws a hard line, and it is the load-bearing decision in this
document:

- **The recap is read-only with respect to the model.** It is assembled
  after ranking, from the same frame the ranker used — the `game_notes.py`
  pattern BUILD_SPEC non-negotiable #5 already establishes, applied here to
  a weekly cadence instead of a single game.
- **"Trends that followed" are tracked as a dated hypothesis ledger, not as
  applied adjustments.** Each week's recap (a) proposes zero or more
  candidate hypotheses from that week's biggest misses, logged with a
  timestamp and the evidence that prompted them, and (b) revisits
  previously-logged hypotheses against what actually happened since,
  scoring each as *supported*, *contradicted*, or *still pending* (too few
  observations). Nothing in that scoring changes a rating, a probability, or
  a bet. The ledger is intake for Workstream E (new factor candidates) and
  Workstream H (re-validation candidates) — the only two places a factor is
  allowed to gain real influence, and only after the full gate.
- **A test enforces this architecturally**, mirroring F8's guarantee for the
  explainer: there must be no code path from `data/recaps/` or the
  hypothesis ledger back into `nflvalue/ratings.py`, the simulation, or the
  blend. If that test doesn't exist, this spec isn't done, no matter how
  good the recap looks.

Everything below is built inside that boundary.

---

## 1. What "performance" means before D exists

There is no CLV yet (Workstream D unbuilt), so a scoreboard that only knows
how to report CLV would have nothing to show for months. Phased, same
pattern as `WEEKLY_LOOP_SPEC.md`:

- **Phase 1 (buildable now, rides on B + WEEKLY_LOOP_SPEC Phase 1):**
  weekly model-vs-market-vs-actual recap. For every graded game: the model's
  pre-game margin projection, the market close, the actual margin, model
  error, market error, and whether the model or the market was closer.
  Rolled up: mean absolute error, correlation to actual margin, and the same
  two numbers for the market, so the gap this week is visible against the
  gap `backtest_b.py` already established historically (model MAE 13.16 vs.
  market 12.24, pooled 2021-2023). A week where the model closes that gap,
  or blows past it, is itself a signal worth writing down.
- **Phase 2 (after C, D land):** the full slice-tagged CLV scoreboard per
  F6, Brier score and reliability per Workstream O, and staking-relevant
  language — still gated by the paper-mode banner (F7) and never implying a
  bet before gate 7.

## 2. Build — Phase 1

**Weekly recap generation**, triggered by the `grade` job in
`WEEKLY_LOOP_SPEC.md` once a week's games are final:

1. Pull the week's final scores (already in the ingest cache) and the
   model's pre-game projections from that week's run manifest — never
   recomputed after the fact, so the recap grades what was actually
   published, not a re-run with hindsight.
2. Compute per-game and rolled-up error metrics (model vs. market vs.
   actual).
3. Rank games by |model error| descending. Take the top N (default 5) as
   **this week's misses**.
4. For each miss, pull the situational context already available
   display-only (injuries, coaching changes, weather, bye weeks — whatever
   `game_notes.py`-style context exists at that point in the build) and
   note what was different about that game, in plain language, generated
   from the same values the table shows — same discipline as F3.
5. Write two artifacts:
   - `data/recaps/{season}_wk{week}.md` — the human-readable recap:
     headline numbers, the week's misses with context, and the hypothesis
     ledger update (new entries + revisited entries).
   - `data/recaps/{season}_wk{week}.json` — the same content structured,
     so future weeks' revisits and eventual dashboard rendering read
     structured data rather than re-parsing markdown.

**The hypothesis ledger** — `data/hypothesis_ledger.json`, append-only like
the worklog, one entry per candidate:
```
{
  "id": "2026wk3-backup-qb-miss",
  "logged": "2026-week-3 recap",
  "hypothesis": "3 of this week's 5 biggest misses involved a team missing
                 its starting QB; model didn't know because there's no
                 availability feed yet (Workstream E item 1, unbuilt).",
  "evidence_games": ["<game_id>", "<game_id>", "<game_id>"],
  "status": "pending",
  "revisits": [],
  "promoted_to": null
}
```
On each subsequent week, the recap job re-reads open-status entries and
checks whether the pattern recurred; after a configurable number of
supporting or contradicting observations (default 3), status moves to
`supported` / `contradicted` / stays `pending`. `supported` entries with
enough n are the ones worth handing to Workstream E as a factor candidate or
Workstream H as a re-validation trigger — `promoted_to` records that
hand-off explicitly (a workstream item ID, a research-doc reference, or
`null` if not yet acted on). Promotion is a human/Curtis decision recorded
here, not an automatic action.

**Format:** the markdown recap is meant to be read, not just archived — same
target audience as `PHASE0_SMOKE_TEST.md`'s report-back instruction. Plain
language first, table second, raw numbers last. The JSON twin exists so
Workstream F's eventual dashboard can render a "this week" panel without
re-deriving anything.

## 3. Build — Phase 2 (after C, D)

Extend the recap with: slice-tagged CLV for the week (P4/G5/FCS-buy/
bowl/week0-1/neutral, matching F6's slice list exactly so the weekly recap
and the persistent scoreboard panel never disagree on taxonomy); Brier score
and a reliability check for the week's probability outputs (Workstream O);
and — only for slices where the kill-check (gate 7) has already returned GO
— the same slice's CLV trend, since F6 is the only place in the whole
interface allowed to imply an edge exists. Every other slice's recap stays
diagnostic-only language, no exceptions, regardless of how good a week looks.

## 4. Hardening

- **The "why not confident" framing carries into the recap.** A good week
  is not evidence of a real edge any more than F5's "why not" panel treats
  one confident-looking game as license to act — small-n weekly swings are
  noise until gate 7 says otherwise. The recap's own honesty furniture
  (inherited from `document.py`/`dashboard.py`) should say this explicitly
  in weeks where the numbers look unusually good, not just in bad weeks.
- **Architectural guarantee test** (per §0): assert no import path exists
  from `data/recaps/` generation code or the hypothesis ledger into
  `nflvalue/ratings.py`, the simulation module, or the blend. This is the
  single most important test in this spec.
- **Grade what was published, not what hindsight would pick.** The recap
  must read projections from the week's run manifest (`WEEKLY_LOOP_SPEC.md`
  §2), never regenerate them — regenerating "predictions" after seeing the
  outcome and then grading those is a leakage pattern with a narrative
  costume.
- **Ledger entries never silently expire.** A `pending` hypothesis with too
  few observations stays visible in the ledger indefinitely rather than
  being pruned — an unresolved hypothesis is exactly the kind of finding
  BUILD_SPEC's "rejections get written down and kept" principle (Workstream
  E) already applies to; this ledger is upstream of that, for the same
  reason.

## 5. Optimization

Recap generation is a once-a-week, small-N job — no performance requirement
beyond "finishes in the weekly loop's existing budget." The only thing worth
caching is the ledger read/write, since it's append-only and grows slowly;
no database needed, plain JSON is fine at this scale (revisit if the ledger
exceeds a few thousand entries, which would itself be a sign something's
wrong — a healthy model shouldn't be generating thousands of open
hypotheses).

## 6. Definition of Done

- A real graded week produces both recap artifacts automatically from the
  `grade` job, with no hand-editing.
- The hypothesis ledger contains at least one real entry from a real week,
  correctly carried into the following week's revisit pass.
- The architectural-guarantee test (§0, §4) exists and passes.
- A week with an unusually good model-vs-market gap still renders the "this
  could be noise" framing — checked by a test on a synthetic good week, not
  just eyeballed.
- Every slice label used in the Phase 2 recap matches F6's slice list
  exactly (shared constant, not two hand-maintained lists).

## 7. Kickoff prompt

> Implement `WEEKLY_SCOREBOARD_SPEC.md` Phase 1. Build the weekly recap job
> (triggered by `WEEKLY_LOOP_SPEC.md`'s `grade` stage): model-vs-market-vs-
> actual error metrics for the week, top-N misses with plain-language
> context, and the append-only hypothesis ledger with its revisit logic.
> Write both `data/recaps/{season}_wk{week}.md` and the structured `.json`
> twin. The recap must read projections from the week's run manifest, never
> regenerate them. Build and pass the architectural-guarantee test proving
> no code path exists from the recap or the ledger back into the ratings,
> simulation, or blend — this is the non-negotiable part of the spec, not
> optional hardening. Do not implement Phase 2 (CLV/Brier) — that's gated
> behind Workstreams C and D. Append to `_worklog.md` and update `hot.md`
> when done.

**This is a diagnostic and hypothesis-generation tool, not a performance
claim.** Nothing it produces authorizes staking or implies the model is
"improving" in a load-bearing sense until Workstream H formally re-validates
a promoted hypothesis and gate 7 clears it. A ledger full of `pending` and
`contradicted` entries is a healthy ledger, not a failing one.
