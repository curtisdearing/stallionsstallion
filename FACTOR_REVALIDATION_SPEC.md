# stallionsstallion — factor re-validation cadence spec (Workstream H, deep dive)

**What this is:** the spec for the thing `research/core-modeling-philosophy.md`
named and then admitted, in its own closing line, "isn't scheduled anywhere
yet" — a recurring recheck of every factor sorted into that doc's
"real-but-era-dependent" bucket (Bucket 3), so a factor that stops being true
gets demoted automatically instead of quietly staying influential. BUILD_SPEC
section 12 states the requirement; this is it worked through against the
factors that already have real numbers behind them, three days into the
build.

**Status when written (2026-07-20):** Workstream E (the situational factor
battery itself) hasn't started — nothing is gated in yet. That does not mean
this spec is premature. The registry is the container the factors land in,
built the same way the explainer (F) was deliberately sequenced before the
factors (E): build the thing that keeps the work honest *before* the work
that needs keeping honest, not after.

---

## 0. Why "not scheduled anywhere" is a real gap, not a nice-to-have

`core-modeling-philosophy.md`'s three-bucket framing is the single best idea
this project's research produced: structurally timeless, noise, and
real-but-era-dependent. The third bucket is the interesting one, and it is
exactly the bucket a one-time literature scan cannot close, because its
whole definition is that the *true value drifts*: HFA's mechanism is
permanent but its magnitude has been declining for decades; returning
production's reliability is plausibly drifting as the transfer portal
matures; conference-tier structure gets redrawn most offseasons; postseason
motivation dynamics change under the post-2026 CFP seeding rule. A factor
that was correctly accepted in 2026 can be quietly wrong in 2029 with nobody
having looked. That is the gap this spec closes.

## 1. The registry — one file, not two

BUILD_SPEC asks for two things that would drift apart if built separately:
Workstream E's "excluded-variable list as a file the code reads" and
Workstream H's "factor registry." This spec merges them into one file,
`data/factor_registry.json`, because a factor's *status* — permanently
excluded, gated-active, display-only-pending-more-evidence, or
under-re-test — is one field on one record, not two files that both need
editing every time a factor moves. Schema:

```json
{
  "factor_id": "league_hfa",
  "description": "League-average home-field advantage, points.",
  "bucket": 3,
  "status": "gated_active",
  "effect_size": 2.83,
  "ci": null,
  "n": 1765,
  "source_doc": "research/win-factors-literature-scan.md",
  "source_measurement": "stallionsstallion/data/refit_b_report.txt (2026-07-19 refit)",
  "test": "tests/test_research_claims.py::test_hfa_in_literature_range",
  "last_validated": "2026-07-19",
  "next_validation_due": "2026 season end",
  "revalidation_history": []
}
```

`status` values: `excluded` (Tier 3 per `proven-win-indicators.md`, permanent
until explicitly re-argued), `untested`, `gated_active` (cleared n-gate +
matched control + BH q<0.05, currently influencing a probability),
`display_only` (exists in the explainer, does not touch a number — either
never cleared the gate or was demoted), `under_revalidation`. **Demotion is
the only status transition this spec's scheduled job is allowed to make on
its own** — promotion to `gated_active` always requires the full Workstream E
gate, run and recorded by a person or an agent session, never by the
scheduled job.

## 2. Seeding the registry with what's already measured

The registry should not launch empty. Three real numbers already exist from
gate 2's re-fit (`data/refit_b_report.txt`, 2026-07-19) and belong in it now,
each a small, concrete demonstration of exactly the problem this spec exists
to solve:

- **`league_hfa`** — fit at **2.83 points** on the full 2019-2023 graph
  (n=1,765), inside the literature's cited 2.5-3.5 range. Bucket 3 because
  the literature also documents a multi-decade *decline* in this number —
  next validation should re-fit on the 2024-2026 graph once enough of it
  exists and check the trend sign, not just the range.
- **`per_team_time_varying_hfa`** — the single clearest illustration of why
  this spec exists. At n=42 (the original MWC-only slice), the walk-forward
  search pushed `lambda_hfa` to its maximum, i.e. the data said "one league
  number, no per-team variation earns its place." At n=1,765 (the same
  parameter, re-fit on the real sample), the search landed at `lambda_hfa=10`
  — **interior**, meaning per-team HFA *does* earn its place once there's
  enough data to see it. Recorded status: `gated_active` as of the
  2026-07-19 refit, `revalidation_history` should carry the earlier n=42
  verdict as the reason this factor is tagged `bucket: 3` (era/sample
  dependent) rather than `bucket: 1` — a factor whose own acceptance
  criterion already flipped once under more data is not the kind of thing to
  treat as settled.
- **`explosiveness_stickiness`** — tested and currently a **rejection**, not
  a gap: at both n=42 and the full re-fit (n≈1,190 unit-games), the shrinkage
  search chose maximum shrinkage — no usable week-to-week signal — matching
  the literature's own externally-confirmed finding. Status `excluded` for
  now (it is failing on this project's own data, not just in someone else's
  study), with a note that unit-level ratings generally (not just
  explosiveness) showed *no* factor was sticky at lag-1, including
  efficiency — this deserves a wider re-look once opponent-adjusted (not
  raw) stickiness can be measured, hence `bucket: 3` not `bucket: 2`
  (permanent noise) — it might be a measurement artifact, not a settled
  finding, and Workstream M's falsifiable-test framing is the right place to
  pin that down.
- **`conference_strength_term`** — flat across the shrinkage grid at the
  current sample ("not load-bearing for margin prediction"), not yet a
  rejection because the term may matter for unit-level ratings even where it
  doesn't move margin — status `under_revalidation`, next check tied to
  Workstream G's bakeoff (a different model family might reveal a signal
  ridge's margin objective can't see, the same way explosiveness_shrinkage
  was invisible to margin prediction until measured against a separate
  objective).

Seeding these four now means the very first scheduled run of this spec's job
has real history to diff against, instead of a cold start.

## 3. Build — the scheduled job

**Cadence** (matches BUILD_SPEC section 22's table exactly — this spec does
not introduce a different schedule):
- **Weekly, in season:** schema/field drift on CFBD and the odds feed
  (cheap, mechanical — assert expected columns/types, same discipline as
  Workstream A's schema contracts); calibration-decay monitoring once
  Workstream O exists.
- **Annually + midseason:** re-run each Bucket-3 factor's original test
  (Workstream M's `tests/test_research_claims.py`, once it exists) against
  the newest available data. Midseason matters specifically for this sport:
  a full-season-only cadence would miss a realignment year's connectivity
  break or a CFP-seeding-rule change until the season it actually bites.

**Job logic:**
1. For each `gated_active` registry entry, re-run its cited test against
   current data.
2. Diff the new effect size against the registry's stored value and CI.
3. If the effect crosses out of significance (sign flip, CI now spans zero,
   or falls below whatever bar Workstream E set for that factor type):
   **flip `status` to `display_only` automatically**, append the old and new
   measurements to `revalidation_history`, and stop — do not delete the
   factor, do not re-run until it passes, record the demotion plainly.
4. If the effect holds, update `last_validated` and `next_validation_due`;
   leave `status` untouched.
5. Write a run report (`data/factor_revalidation_{date}.md`) listing every
   factor checked, its verdict, and any demotions — surfaced via the
   existing notify path, same as a failed gate anywhere else in this repo.

**Market-structure monitoring**, same job family: book count per game, limit
sizes, line-move timing, tracked per slice. The project's own "G5/bowl
markets are softer" hypothesis (from `research/week0-data-sources-and-
methods.md`) is explicitly expected to decay as CFB analytics culture
matures — this is the mechanism that would catch that decay happening,
rather than the project finding out from a kill-check going NO_GO with no
explanation years later.

## 4. Hardening

- **Demotion is asymmetric by design and that asymmetry is load-bearing.**
  Getting demoted requires only the scheduled job to notice a crossed
  threshold; getting (re-)promoted requires the full n-gate + matched
  control + BH q<0.05 gate, same as first admission. A factor should never
  be able to talk its way back in on the strength of the same automated job
  that can throw it out.
  **`killcheck.py`'s existing constants (`DEFAULT_MIN_SAMPLE=150`,
  `POSITIVE_RATE_BAR=0.52`) are PROP_SHORTLISTER_SPEC.md's numbers, not this
  project's.** They measure "leans beat the close at a >=52% rate," a
  different question from a Bucket-3 factor's effect-size significance, and
  a different question again from Workstream D's own note that the
  "bets needed to prove an edge" table needs recomputing for CFB volume
  (~800 FBS games/season vs. the NFL's 272). Do not let this spec's
  significance bar silently inherit props-tool constants that were never
  derived for game lines or for factor re-validation.
- **Registry writes are atomic** (temp + rename), same discipline as
  `nflvalue/ingest.py`'s cache writes — a killed mid-write job must not
  leave a half-written registry, since every other workstream reads it.
- **The registry is the single source of truth for exclusion.** Any code
  implementing Workstream E's factor battery reads `status` from this file
  before letting a factor touch a probability; there is no second,
  hand-maintained exclusion list to fall out of sync with it.
- **A demotion is a finding, not an incident.** The run report and notify
  message should say so in those words — this mirrors the project's own
  worklog convention (negative results recorded, not buried) applied to a
  machine-generated report instead of a human one.

## 5. Optimization

The registry is small (dozens of factors at most) and the job runs at most
weekly — no performance work needed beyond keeping each cited test fast
enough to run unattended. If Workstream M's test suite grows large,
parallelize test execution before optimizing anything else; the registry
diff itself is trivial.

## 6. Definition of Done

- `data/factor_registry.json` exists, seeded with the four entries in §2,
  atomic-write tested.
- The scheduled job runs against a synthetic "factor that just crossed out
  of significance" fixture and correctly flips it to `display_only`,
  appends history, and produces a run report — demonstrated with a test, not
  just described.
- A promotion attempt made *by the job itself* (rather than through the
  full Workstream E gate) is impossible by construction — enforced by a
  test, same pattern as F8's architectural guarantee.
- The market-structure monitor produces a report for at least one real week
  once Workstream I's weekly loop is running.
- Workstream E's factor code reads `status` from this registry and nothing
  else — no second exclusion list exists anywhere in the repo.

## 7. Kickoff prompt

> Implement `FACTOR_REVALIDATION_SPEC.md`. Build `data/factor_registry.json`
> with the schema in §1, seeded with the four entries in §2 (league_hfa,
> per_team_time_varying_hfa, explosiveness_stickiness,
> conference_strength_term) pulled from the real `data/refit_b_report.txt`
> numbers. Build the scheduled re-validation job: weekly schema/calibration
> drift checks, annual+midseason Bucket-3 factor re-tests, automatic
> demotion to `display_only` on a crossed significance threshold (never
> automatic promotion), atomic writes, and a run report through the
> existing notify path. Do not let `killcheck.py`'s prop-tool constants
> leak into this spec's significance bar — derive or explicitly parameterize
> CFB-appropriate ones. Build the test proving the job cannot promote a
> factor, only demote one. Append to `_worklog.md` and update `hot.md` when
> done, including which of the four seeded factors (if any) the first run
> flagged.

**This registry does not authorize staking and does not, by itself, make the
model "better."** It makes drift visible instead of silent. The kill-check
in gate 7 is still the only gate that matters for real money, and a factor
sitting at `display_only` is not a bug in this system — it's the system
working.
