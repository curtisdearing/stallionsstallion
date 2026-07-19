# stallionsstallion — master build spec

**What this is:** the full engineering spec for taking this repo from "Phase 0
smoke test built, no data pulled" to a hardened, optimized CFB game-line model
with an interface that explains *why* every number is what it is.

**How to use it:** read the Non-negotiables first — they constrain everything
else. Then work one workstream at a time, in order (A→F); each has a
Definition of Done that gates the next. Section 9 has short, ready-to-paste
prompts for handing a single workstream to a fresh agent session.

**Two parts.** Part I (sections 0–10) is workstreams **A–F**: the initial
build, from raw CFBD data to an explainer interface. Part II (sections 11–22)
is workstreams **G–P**: the bakeoff, drift detection, weekly operations,
season rollover, kill-check execution, red-teaming, research-claims-as-tests,
external baselines, calibration, and vault sync — what keeps the model honest,
running, and maintainable. Kickoff prompts for G–P are in section 21; section
22 says when each runs and what it depends on.

**Cold-start context for an AI agent** — repo landmines, conventions, and
failure modes — is in `AGENT_BUILD_PROMPT.md`. Read that first if you have no
prior context on this project.

**Status when written (2026-07-17):** Phase 0 smoke test is built and
offline-verified but has never run against live CFBD data. Everything below
is unbuilt. Every modeling claim in the `research/` docs is a hypothesis.

---

## 0. Non-negotiables

These are not new rules. They are the guardrails this project already set for
itself (`research/cfb-portability-plan.md`, `nfl-sim/fablesfable/PREMORTEM.md`,
`hot.md`'s `pm_dod`). They are listed first because every optimization below is
allowed to make things faster or sharper but **not** allowed to weaken these.

1. **Paper-only until the Phase 4 kill-check returns GO**, per slice. `hot.md`'s
   own definition of done: "Phase 4 kill-check GO (slice-specific positive CLV,
   n-gated) before any real staking; research/paper-only until then." No
   interface element may imply a bet is recommended before that gate passes.
2. **CLV is the scoreboard, not win rate.** Win rate needs thousands of bets to
   show skill; CLV shows it in ~50–65. Log opening and closing lines from day
   one — retrofitting this was fablesfable's P0 premortem finding.
3. **The model does not get bet straight.** fablesfable's own backtest: pure
   power rating correlates 0.37 with margins, the close correlates 0.43, and
   betting the model into the close lost money at every market (spread −6.1%,
   total −12.2%, ML −14.8%). The model gets blended with the market; the blend
   is the product.
4. **Walk-forward or it doesn't ship.** Every feature is `shift(1)`-then-roll.
   Any same-game statistic used to predict that same game is leakage. The
   poison-a-future-week test (`tests/test_leakage.py`'s technique in
   fablesfable — currently a stub here) must be re-implemented against the new
   engine before any backtest number is believed.
5. **Context is display-only by construction.** Following `game_notes.py`'s
   existing pattern: explanatory content is assembled *after* ranking, from the
   same frame the ranker used, so nothing in the explanation layer can feed
   back and move a score. This is an architectural guarantee, not a convention.
6. **Situational factors must earn their place**: n-gate + matched control +
   Benjamini-Hochberg q<0.05 before touching a probability. Until then they are
   display-only context. The vault has a retraction on record
   (`nfl-sim/factor-combo/`'s TD-cascade correction) from skipping exactly this.
7. **Uncertainty is a first-class output**, not a footnote. Low schedule
   connectivity, thin samples, and early-season ratings widen the band — and a
   wide band is itself a reason to pass on a game, not a caveat printed beside
   a confident-looking number.

### 0.1 On the phrase "overwhelming evidence"

This spec was requested with the goal of surfacing the evidence backing
specific bets. Worth being precise about what the system can and cannot
honestly deliver, because it determines the interface design:

- **What it can do:** decompose a projection into contributions, show how
  strong the evidence behind each contribution is, and quantify how far the
  model's number sits from the market's — with an honest confidence band.
- **What it cannot do:** manufacture conviction. In an efficient-ish market,
  most games have no edge. The realistic ceiling from the project's own
  research is 52.5–55% ATS against a 52.38% break-even. A tool that finds
  "overwhelming evidence" on many games is broken, not good.
- **Therefore:** the explainer must be *equally* capable of rendering "the
  evidence here is thin, pass" as "this is a real edge," and must show the
  reasons against a bet with the same prominence as the reasons for it. An
  interface that only surfaces supporting evidence is a rationalization engine.
  Design requirement F5 below makes this concrete.

A model that has never seen data has zero evidence behind it. Nothing in this
spec produces a defensible bet until Workstream D is backtested and Workstream
E's factors clear their gates.

---

## 1. Architecture and data flow

```
CFBD API ──> A. ingest (parquet cache, walk-forward safe)
                   │
                   ├─> B. ratings engine (ridge solve over game graph)
                   │        └─ per-team off/def, conference term, per-team HFA,
                   │           connectivity confidence
                   │
                   ├─> C. simulation (drive-level Monte Carlo, CFB OT)
                   │        └─ margin distribution -> spread/total/ML probs
                   │
                   ├─> D. market layer (multi-book, de-vig, model<->market
                   │        blend, CLV log)  <-- the actual product
                   │
                   └─> E. situational factors (gated; display-only until proven)
                                    │
                                    v
                          F. explainer interface
                             (why this number, how strong, why not)
```

Existing code to reuse rather than rewrite: `oddsmath.py` (de-vig/consensus),
`killcheck.py` (GO/NO-GO gate), `premortem_mc.py` (Kelly/ruin math),
`clv.py`, `freshness.py`, `db.py`, `dashboard.py` (HTML render pattern),
`game_notes.py` (display-only context pattern), `nflvalue/sources/cfbd.py`
(the Phase 0 client). Reuse is preferred over reimplementation everywhere.

---

## 2. Workstream A — data layer (CFBD ingest)

Replaces the nflverse-shaped `nflvalue/ingest.py` and
`scripts/bootstrap_history.py`, which will not import as-is.

**Build:**
- Season-partitioned parquet cache mirroring the existing pattern: a frozen
  historical base plus one file per season, current season re-fetched on
  refresh. Loaders compose base + per-season into one frame.
- Pull and cache: plays (PPA, down/distance/yard line, play type, clock),
  games/schedule, lines (all books, opening + closing), talent composite,
  returning production, transfer portal, rosters, venues.
- Extend `nflvalue/sources/cfbd.py` rather than writing a second client. Keep
  its call counter; log calls-per-refresh to a running monthly tally against
  the 1,000 cap. **Confirmed:** `/plays` is per-week (~15 calls/season);
  `/games`, `/lines`, `/talent` are bulk (1 each).
- **Garbage-time filter applied at ingest**, before any rate stat is computed —
  a score-differential-and-clock threshold, per standard CFB analytics practice.
  Store both filtered and raw so the threshold stays tunable.
- Conference membership read as a **per-season lookup**, never hardcoded —
  realignment moves constantly.
- FCS opponents tagged and handled as a separate tier, not fed to the FBS fit
  at face value.

**Hardening:** fail-loud staleness (a refresh that can't reach CFBD keeps the
cache and reports `stale=True`, letting the freshness gate decide — never
silent). Schema-contract test that asserts expected columns and types on every
load, so upstream API drift breaks a test rather than corrupting a rating.
Retry with backoff on 5xx; never retry into the rate limit. Idempotent
re-runs. Atomic writes (temp + rename) so an interrupted refresh can't leave a
half-written parquet.

**Optimization:** column pruning at read; categorical dtypes for team/conference;
one bulk pull per season rather than per-game loops; cache the ridge design
matrix keyed by (season, through_week) so a re-render doesn't refit.

**DoD:** leakage test passes (poison a future week, assert no earlier number
moves); schema contract test passes; a full-season refresh reports its exact
call count; loaders return identical frames on repeat runs.

---

## 3. Workstream B — ratings engine

Replaces `build_ratings.py`'s sequential single-game update, which assumes NFL
schedule density that CFB does not have.

**Build:**
- **Regularized least-squares (ridge) solve refit weekly across the full game
  graph to date.** Each game contributes roughly
  `margin ≈ HFA + off[home] − def[away] − (off[away] − def[home])`, stacked and
  solved jointly so information propagates through chains of games rather than
  only through direct or shared opponents.
- **Shrink toward a conference-level mean**, not zero, for teams with few games
  or poor connectivity. Conference strength is an **explicit term**,
  re-estimated as non-conference games accumulate.
- **Connectivity confidence per team** — shortest path through the schedule
  graph to a stable reference set. This is a *model input*, not an annotation:
  it widens the uncertainty band and can veto a bet outright.
- **Per-team (or per-tier) HFA, time-varying**, explicitly zeroed or discounted
  at neutral sites (conference championships, neutral openers, the whole
  bowl/CFP slate).
- **Unit-level ratings** per `research/matchup-model-framework.md`: rush/pass ×
  offense/defense × efficiency/explosiveness = eight numbers, opponent-adjusted
  first, then recency-weighted (EWM; decay is tuned, not guessed).
  Explosiveness gets materially more shrinkage than efficiency — it is
  genuinely associated with winning but near-zero sticky week to week.
- **Preseason prior** = prior-year rating (down-weighted) + returning
  production + recruiting/portal talent composite, decaying in favor of
  in-season performance as games accumulate.

**Hardening:** ridge strength and conference-shrinkage weight are fit by
walk-forward grid search (extend `learn.py`; note `tune_weights.py` here is a
stub of the prop tuner, not this). Seed every stochastic step and assert
run-to-run reproducibility. Assert the design matrix is well-conditioned and
fail loud if a team is disconnected from the graph rather than emitting a
garbage rating.

**Optimization:** sparse matrices for the game graph; warm-start the weekly
refit from last week's solution; cache per-week solutions by content hash.

**DoD:** ratings reproduce exactly across runs; connectivity confidence is
populated and demonstrably lower in September than November; a known blue-blood
vs. known bottom-tier team ordering passes a sanity assertion.

---

## 4. Workstream C — simulation

**Build:**
- Drive-level Monte Carlo: simulate possessions from a rating-derived scoring
  rate model, rather than sampling one margin number.
- **Matchup-conditional drive-outcome rates.** A single league-wide prior will
  systematically under-simulate blowouts and over-simulate competitiveness in
  mismatched games — CFB's talent spread is far wider than the NFL's. Fit
  drive-outcome rates against the rating differential (binned or regressed).
- **A CFB overtime module.** Alternating possessions from the opponent's
  25-yard line, mandatory two-point conversions from the second OT onward.
  This changes totals and moneylines in close games and cannot be ported from
  NFL sudden-death logic.
- Output the **full margin and total distributions**, not point estimates — the
  explainer and the uncertainty band both need the distribution.

**DoD:** simulated margin distribution's variance is in the right neighborhood
for CFB (wider than NFL); OT module unit-tested against hand-worked cases;
calibration (Brier) and margin-correlation-vs-closing-line reported honestly as
the **CFB baseline**, explicitly not graded against fablesfable's NFL numbers.

---

## 5. Workstream D — market layer and CLV

This is the actual product. Everything above is an input to it.

**Build:**
- Multi-book ingest with **opening and closing lines logged from day one**.
  De-vig and consensus via the existing `oddsmath.py`; sharp-book weighting
  already exists in config.
- **nfelo-style error-weighted model↔market blend.** Weight is fit on
  out-of-sample squared error, not chosen. The blend — not the raw model — is
  what generates a projection.
- **CLV log** per bet-candidate: timestamp, book, price taken, closing price,
  CLV in cents and in percent, market slice tag (P4 / G5 / FCS-buy / bowl /
  week 0–1 / neutral site).
- EV and fractional-Kelly staking via `premortem_mc.py`'s existing math —
  quarter-to-half Kelly, hard cap per the config's `max_stake_pct`. Recompute
  the "bets needed to prove an edge" table for CFB volume (~800 FBS games/season
  vs. the NFL's 272) rather than inheriting the NFL number.

**DoD (the Phase 2 gate):** the blend beats both the raw model and the raw
market on out-of-sample squared error. If it doesn't, that is the finding —
write it down plainly, same as fablesfable's own NO-GO convention.

---

## 6. Workstream E — situational factor battery

Every factor here is a hypothesis with a test attached. Order by expected value:

1. **Backup-QB adjustment** — highest-leverage single signal; the CFB QB1→QB2
   cliff is expected steeper than the NFL's measured −8.4% efficiency. Measure,
   don't assume the NFL magnitude transfers.
2. **Offensive line depth** — plausibly underpriced; degrades run game and pass
   protection simultaneously and is less visible than a skill-position injury.
3. **Two-tier injury/availability feed** — P4 + CFP have real (but young,
   low-trust) public availability reports; G5 falls back to depth-chart deltas
   and participation inference. Grade the tiers differently.
4. **Coaching change, conditional** — not a flat flag. Condition on prior team
   quality and years-since-arrival; the pooled average effect is weak-to-null.
5. **Bowl opt-outs** (knowable in advance, publicly announced) kept **separate**
   from **CFP-seeding motivation** under the post-2026 seeding rule.
6. **Weather** — wind on passing/kicking is well supported; the "warm team's
   first cold road game" narrative gets the matched-control treatment before
   it's trusted (the NFL analog of this class of narrative mostly came back null).
7. **Field position, finishing drives, special teams as a fifth unit** — the
   open recommendation from `research/win-factors-literature-scan.md`.

**Explicitly excluded from the model** (per `research/proven-win-indicators.md`
Tier 3): raw time of possession, penalty counts, raw red-zone TD%, raw pressure
rate, raw turnover margin (use turnover *luck* instead), and any same-game stat
used to predict that same game. **Maintain this exclusion list as a file the
code reads**, so an excluded variable can't quietly reappear as someone's new
feature idea.

**DoD:** each factor passes n-gate + matched control + BH q<0.05, or is
recorded as tested-and-rejected. Rejections get written down and kept — the
rejection list is as valuable as the acceptance list.

---

## 7. Workstream F — the explainer interface

The "translatable" layer: make it possible to see, for any game, exactly what
is driving the number and how much to trust it. Extends the existing
self-contained-HTML pattern in `dashboard.py` (no server, no external
libraries, opens by double-clicking, all data inlined).

### F1 — Contribution decomposition
For each game, decompose the projected margin into signed point contributions
that **sum to the projection**. Because the ratings solve is linear, this is
exact rather than an approximation:

```
Projected margin  +6.4  (model)   |  Market  +7.5  |  Blend  +7.1
  ├─ Off/def rating differential      +4.9
  ├─ Home-field advantage             +2.6   (team-specific; league avg +2.9)
  ├─ Pass-offense vs pass-defense     +1.4
  ├─ Rush-offense vs rush-defense     −0.8
  ├─ Conference-strength term         −1.2
  ├─ Preseason prior residual         −0.5    (decaying: week 7 weight 0.18)
  └─ Situational (display-only)        0.0    (QB1 questionable — not applied)
```
Every row links to the underlying data: which games produced it, the sample
size, the opponent adjustment applied.

### F2 — Evidence tier per input
Each contributing variable carries a visible tier, sourced from
`research/proven-win-indicators.md`:
- **Tier 1** — pregame-predictive, externally documented (success rate,
  EPA/play, third-down rate, points per red-zone trip, sack-rate margin,
  turnover luck, returning production, talent composite).
- **Tier 2** — real but matchup-specific (havoc rate by position group,
  pass-rush vs. pass-block win rate, special teams).
- **Tier 3 / excluded** — shown only to explain *why it's excluded*, never as
  support for a bet.
- **Ungated** — a situational factor that hasn't cleared its statistical bar.
  Rendered visually distinct and explicitly labeled as not affecting the number.

### F3 — Plain-language translation
Every numeric row gets a one-sentence English rendering, written for someone
who does not want to read a regression table. "Boise State's pass offense has
been 1.4 points/game better than Fresno's pass defense, adjusted for who each
played — based on 7 games, which is a normal sample for week 9." Generate
these from templates bound to the same values the table shows, so the prose
cannot drift from the numbers.

### F4 — Confidence, not just point estimates
Show the margin distribution, not a single number. Surface explicitly:
schedule-connectivity confidence, games-in-sample, week-of-season, whether
either team is in a freshly-reshuffled conference, and the resulting width of
the band. **A wide band is displayed as a reason to pass**, not as a caveat.

### F5 — The "why not" panel (required, equal prominence)
For every game, render the case *against* acting, with the same visual weight
as the case for. Populated from: thin connectivity, small sample, high
team-to-team variance, market moved against the model since open, slice has no
positive CLV history yet, factor involved is ungated, model-market gap smaller
than historical model error, or kill-check not passed for this slice. **If the
"why not" panel is empty, that is a bug** — there is essentially always a
reason for caution in a market this efficient.

### F6 — Slice-level CLV scoreboard
A persistent panel showing measured CLV per market slice (P4 / G5 / FCS-buy /
bowl / week 0–1 / neutral) with n. This is the only place the interface is
allowed to imply an edge exists, and only where the kill-check has returned GO
for that specific slice.

### F7 — Honesty furniture (inherit from the existing surfaces)
Carry over what `document.py` and `dashboard.py` already do: leans not locks,
sample counts visible, synthetic/estimated values daggered, display-only
context marked as such, 1-800-GAMBLER present. Add a **paper-mode banner** that
is on by default and can only be turned off by a config flag that also asserts
the Phase 4 kill-check passed for the slice being viewed.

### F8 — Architectural guarantee
The explainer reads a serialized `explanation` object produced *after* ranking,
from the same frame the ranker used — the `game_notes.py` pattern. There must
be no code path by which anything in Workstream F can influence a projection.
Enforce with a test.

**DoD:** contributions sum to the projection within floating-point tolerance
(asserted in a test); every displayed number traces to a source; the "why not"
panel is non-empty on every game in a full week's render; plain-language
strings are generated from the same values as the table (property test);
paper-mode banner cannot be disabled without the kill-check assertion passing.

---

## 8. Cross-cutting hardening and optimization

**Testing:** re-implement the poison-a-future-week leakage test against the new
engine (the single most valuable pattern in the source repo, currently a stub
here). Schema-contract tests on every external feed. Reproducibility test —
same inputs, same seed, identical outputs. Property tests on odds math
(de-vigged probabilities sum to 1; Kelly never exceeds the cap). Golden-file
test on a known historical week so a refactor that changes numbers is caught.

**Failure behavior:** fail loud, never silent. A stale feed degrades the
surface visibly (the existing `degraded` badge in `dashboard.py`) rather than
rendering confident numbers over old data. Every fallback path is logged and
surfaced in the UI.

**Secrets:** keys only via env or the gitignored `config.local.json`. Add a
pre-commit-style check that greps for key-shaped strings before any future
`git add`, since this repo has no history yet and a leaked key in commit one
would be permanent.

**Performance:** the full weekly pipeline should complete in minutes, not
hours. Vectorize; avoid per-game Python loops over play-by-play; cache the
ridge design matrix and warm-start refits; keep the Monte Carlo's inner loop in
numpy. Profile before optimizing — the sim is the likely hot spot, not ingest.

**Version control (open decision):** this repo has no `.git` at all. Before
serious build work, `git init` locally at minimum — the code is now valuable
enough that an accidental overwrite has no undo. A GitHub remote remains
Curtis's explicit call (it touches the PAT/credentials state in
`nfl-repos-and-push-access`).

---

## 9. Ready-to-paste session prompts

Hand one of these, plus this file, to a fresh session. Each assumes the prior
workstream's DoD has passed.

**A — ingest:**
> Implement Workstream A of `BUILD_SPEC.md`. Rewrite `nflvalue/ingest.py` and
> `scripts/bootstrap_history.py` against CFBD, extending the existing
> `nflvalue/sources/cfbd.py` client. Season-partitioned parquet cache,
> garbage-time filtering at ingest, per-season conference lookup, FCS tagged as
> a separate tier, fail-loud staleness, atomic writes, schema-contract tests,
> and a monthly API-call tally. Do not touch the ratings engine. Respect the
> Non-negotiables in section 0. Report the exact call count a full-season
> refresh burns. Append to `_worklog.md` and update `hot.md` when done.

**B — ratings:**
> Implement Workstream B of `BUILD_SPEC.md`. Replace `build_ratings.py`'s
> sequential update with a weekly ridge solve over the full game graph:
> conference-mean shrinkage, explicit conference-strength term, per-team
> time-varying HFA with neutral-site zeroing, unit-level ratings
> (rush/pass × off/def × efficiency/explosiveness) with heavier shrinkage on
> explosiveness, connectivity confidence as a real model input, and the
> returning-production + talent-composite preseason prior. Tune ridge strength
> and shrinkage by walk-forward grid search — do not guess constants.

**C — simulation:**
> Implement Workstream C of `BUILD_SPEC.md`. Drive-level Monte Carlo with
> matchup-conditional drive-outcome rates (not one league prior) and a real CFB
> overtime module. Emit full margin/total distributions. Report Brier and
> model-margin-correlation vs. closing-line-correlation as the CFB baseline —
> do not grade against fablesfable's NFL numbers.

**D — market/CLV:**
> Implement Workstream D of `BUILD_SPEC.md`. Multi-book ingest with opening and
> closing lines logged from day one, de-vig via `oddsmath.py`, nfelo-style
> error-weighted model↔market blend with the weight fit out-of-sample, a
> slice-tagged CLV log, and Kelly staking via `premortem_mc.py`. Gate: the
> blend must beat both raw model and raw market on out-of-sample squared error.
> If it doesn't, write that down plainly rather than tuning until it does.

**E — factors:**
> Implement Workstream E of `BUILD_SPEC.md`. Build the factor battery in the
> listed order, each with n-gate + matched control + BH q<0.05 before it
> touches a probability. Implement the excluded-variable list as a file the
> code reads and enforces. Record rejections as carefully as acceptances.

**F — explainer:**
> Implement Workstream F of `BUILD_SPEC.md`. Per-game contribution
> decomposition that sums exactly to the projection, evidence tier per input,
> plain-language translation generated from the same values as the table,
> confidence band driven by connectivity and sample size, a mandatory non-empty
> "why not" panel with equal visual weight, and a slice-level CLV scoreboard.
> Extend the self-contained-HTML pattern in `dashboard.py`. The explainer must
> be architecturally incapable of influencing a projection — enforce with a
> test.

---

## 10. Sequencing and gates

| Order | Workstream | Gate before proceeding |
|---|---|---|
| 0 | Phase 0 smoke test (built) | Run it live; endpoints confirmed |
| 1 | A — ingest | Leakage + schema tests pass |
| 2 | B — ratings | Reproducible; connectivity populated |
| 3 | C — simulation | CFB baseline Brier + margin-corr reported honestly |
| 4 | D — market/CLV | Blend beats model and market out-of-sample |
| 5 | F — explainer (first pass) | Contributions sum; "why not" always populated |
| 6 | E — factors | Each clears n-gate + matched control + BH q<0.05 |
| 7 | Phase 4 kill-check | Slice-specific positive CLV before **any** real staking |

The explainer is deliberately sequenced before the factor battery: it is the
tool that makes the factor work legible, and building it early means the
factors get evaluated in a surface that already shows evidence honestly.

**Part II interleaves with this table** rather than following it — the bakeoff
(G) belongs between C and D, calibration (O) rides along with every backtest,
red-teaming (L) comes before the kill-check, and vault sync (P) runs every
session. See section 22 for the full cadence.

**Nothing in this spec authorizes staking real money.** That decision belongs
to gate 7 and to Curtis, and only per-slice where CLV is positive and
significant. If you are ever unsure whether the gate has passed, it hasn't.

If betting stops feeling like the modeling project it is here, that's worth
noticing — 1-800-GAMBLER.

---

# Part II — beyond the initial build

Workstreams A–F produce a model. **G–P keep it honest, keep it running, and
answer the questions A–F deliberately deferred.** Some are one-time, some
recurring — the cadence in section 23 matters as much as the content.

Two of these close gaps this project's own research opened and never resolved:
**G** (the model bakeoff `research/week0-data-sources-and-methods.md`
recommended) and **H** (the Bucket-3 re-validation cadence
`research/core-modeling-philosophy.md` flagged as unscheduled).

---

## 11. Workstream G — model bakeoff

**Why:** Workstream B specifies ridge. That choice was made on reputation, not
evidence. `research/week0-data-sources-and-methods.md` explicitly recommends
running candidate methods side by side and letting the metric decide.

**Build:** one walk-forward harness, identical folds and features across all
candidates:
- Ridge / regularized least squares (the Workstream B baseline)
- Elo with a football-appropriate K-factor
- G-Elo (margin-of-victory-generalized Elo)
- Glickman-Stern Bayesian state-space — two variance terms (week-to-week and
  season-to-season); the more rigorous cousin of B's EWM recency approach
- Gradient-boosted trees on the Tier-1 feature set
- Random forest, lasso, linear SVM as secondary comparisons
- Play-style clustering as a **feature generator** feeding the above, never as
  a standalone predictor

**Rules:** identical train/test splits for every candidate. **Pre-register the
decision metric before running anything** — Brier plus margin-correlation
versus the closing line. Ensemble only if the ensemble beats the best single
model out-of-sample; do not ensemble by default.

**Trap:** more candidates means more chances to overfit the backtest window.
The pre-registered metric and fixed folds are the defense. If a model wins by a
hair, that's a tie, not a winner.

**DoD:** a results table covering every candidate on identical folds; the
winner chosen by the pre-registered metric; the losers recorded with their
numbers, not deleted.

---

## 12. Workstream H — drift detection and re-validation cadence

**Why:** `research/core-modeling-philosophy.md` sorts factors into structurally
timeless, noise, and **real-but-era-dependent**. That third bucket needs
periodic re-validation, and no cadence was ever scheduled. Realignment,
declining HFA, portal-era returning-production reliability, and post-2026 CFP
seeding all drift.

**Build:**
- A **factor registry** file: each accepted factor with its effect size,
  confidence interval, n, source doc, bucket, and last-validated date.
- A scheduled job that re-runs each Bucket-3 factor's original test on the
  newest data and diffs the effect size against the registry. Alert when an
  accepted factor's effect crosses out of significance — **it gets demoted to
  display-only automatically**, not left in the model pending review.
- Schema/field drift monitoring on CFBD and the odds feed.
- Market-structure monitoring: book count per game, limit sizes, line-move
  timing — the project's "G5/bowl markets are softer" hypothesis is expected to
  decay as CFB analytics culture matures.
- Week-over-week calibration decay monitoring.

**DoD:** the registry exists and is populated; the job fails loud on drift;
demotion is automatic rather than discretionary.

---

## 13. Workstream I — weekly run loop

**Build:** extend `scripts/auto_weekly.py` and `scripts/state_store.py`.
Ordered stages: refresh ingest → freshness gate → refit ratings → simulate →
pull market → blend → log CLV candidates → render explainer → notify.

**Requirements:** idempotent and resumable (a crash mid-week resumes, doesn't
restart). Single-writer lock so two runs can't race. **Never publish over stale
data** — the freshness gate degrades the surface visibly instead. Write a **run
manifest** per execution: config hash, data as-of timestamps, random seed, code
version, and every gate's pass/fail. Any published number must be traceable to
a manifest.

**DoD:** a full week runs unattended end to end; a deliberately stale feed
produces a degraded surface rather than confident numbers; re-running the same
manifest reproduces identical output.

---

## 14. Workstream J — season rollover

**Build:** a script, not a manual edit session. Refresh conference membership
(per-season lookup), handle new and promoted teams including FCS→FBS moves,
rebuild the preseason prior (decayed prior-year rating + returning production +
talent/portal composite), refit HFA, review the exclusion list, and archive the
prior season's ratings and CLV log.

**Traps:** teams change subdivision, not just conference. Rating carryover must
**not** inherit the NFL's 60% — college roster turnover is far higher. A
realignment year breaks connectivity assumptions, so newly-reshuffled
conferences start the season flagged low-confidence by construction.

**DoD:** rollover is scripted and repeatable; a realignment year is handled
without hardcoding a single team or conference.

---

## 15. Workstream K — kill-check execution (Phase 4)

Not building `killcheck.py` — **running it**, correctly, as the final gate
before any real staking.

**Method:** run per market slice (P4, G5, FCS-buy, bowl, week 0–1, neutral
site). Each slice needs its own n. GO only where slice CLV is **positive and
statistically significant**.

**Rules that make this meaningful:**
- **Do not pool slices to reach n.** A blended verdict hides where the edge
  isn't.
- **Do not re-run until it returns GO.** One pre-registered evaluation.
- The result is recorded whichever way it lands. A NO-GO written in plain
  language is the expected outcome for most slices and is a successful session.

**DoD:** a written verdict per slice — n, mean CLV, confidence interval, and
the decision — committed to the repo and summarized in `hot.md`.

---

## 16. Workstream L — red-team

Run by a **separate agent with an adversarial mandate**, not by whoever built
the thing.

**Tasks:** hunt for leakage independently of the built-in tests; re-run results
on a different train/test split; test sensitivity to the backtest window
(does the edge survive dropping the best month?); attempt to reproduce the
headline numbers from scratch; audit the factor battery for p-hacking and
multiplicity handling; check whether the model↔market blend's advantage holds
excluding the largest-edge decile.

**DoD:** a written report where every finding is either fixed or explicitly
accepted with a rationale. "No findings" is not an acceptable report.

---

## 17. Workstream M — research claims into falsifiable tests

**Why:** everything in `research/` is currently prose. Prose can't fail.

**Build:** `tests/test_research_claims.py`, converting each claim into an
assertion checked against this project's own data, each test citing its source
document. Examples:
- "Explosiveness has near-zero week-to-week stickiness" → compute
  autocorrelation, assert below threshold.
- "Success rate is the stickiest of the Five Factors" → rank stickiness,
  assert the ordering.
- "HFA is ~2.5–3.5 points and declining" → fit it, assert range and trend sign.
- "Raw turnover margin is ~2.6% forward-predictive" → measure, assert low.
- "Penalties have near-zero correlation with wins" → measure, assert near zero.

**Critical framing:** a failing test here means **the literature doesn't hold
on your data**. That is a finding to write into the source doc, not a test to
loosen until it passes.

**DoD:** every Tier-1 claim has a test; every failure is documented back into
the originating research doc with the measured value.

---

## 18. Workstream N — external baseline reconciliation

**Build:** compare this project's ratings against the Massey Ratings archive
(a 29-system ensemble) — report correlation and, more usefully, the **biggest
disagreements**. Spot-check CFBD box scores against Sports-Reference.
Reconcile line data against a second source where available.

**Purpose, both directions:** catch silent data corruption, and flag when the
model is confidently different from every public system. The second is the more
valuable signal — occasionally it means an edge, usually it means a bug.

**DoD:** a reconciliation report per season; every disagreement past a
threshold investigated and explained, not just logged.

---

## 19. Workstream O — calibration deep-dive

**Why:** a well-calibrated model that loses to the closing line is a
fundamentally different problem from a miscalibrated one, and the raw Brier
score alone doesn't distinguish them.

**Build:** reliability diagrams; Brier decomposition into reliability,
resolution, and uncertainty; calibration per market slice and per
confidence-bucket; and a **coverage check on the uncertainty band** — does the
stated 80% band actually contain the outcome 80% of the time?

**Rule:** post-hoc calibration correction (isotonic or Platt) is allowed only
if the correction itself survives out-of-sample. Fitting a calibration curve on
the same data you evaluate on is circular.

**DoD:** calibration reported per slice; band coverage tested and honest;
CFB's baseline established on its own terms rather than against NFL numbers.

---

## 20. Workstream P — vault sync

**Recurring housekeeping.** Reconcile `hot.md`, `_SETUP.md`, and the MOC
against the worklog tail; refresh the `pm_*` headers; flag any doc older than
~14 days that contradicts current state; confirm every runnable entry point is
listed in `_SETUP.md` and every research doc is linked from the MOC.

**DoD:** no vault doc contradicts the worklog; a cold agent reading `hot.md`
plus `_SETUP.md` gets an accurate picture without reading the full worklog.

---

## 21. Kickoff prompts for G–P

**G — bakeoff:**
> Implement Workstream G of `BUILD_SPEC.md`. Build one walk-forward harness and
> run ridge, Elo, G-Elo, Glickman-Stern Bayesian state-space, GBDT on Tier-1
> features, random forest, lasso, and linear SVM on identical folds.
> Pre-register Brier + margin-correlation-vs-close as the decision metric
> before running anything. Report every candidate's numbers including losers.
> Ensemble only if it beats the best single model out-of-sample.

**H — drift:**
> Implement Workstream H of `BUILD_SPEC.md`. Build the factor registry
> (effect size, CI, n, bucket, source doc, last-validated date) and the
> scheduled re-validation job. Bucket-3 factors get re-tested on new data and
> auto-demoted to display-only when their effect crosses out of significance.
> Add schema-drift, market-structure, and calibration-decay monitoring.

**I — weekly ops:**
> Implement Workstream I of `BUILD_SPEC.md`. Extend `auto_weekly.py` and
> `state_store.py` into an idempotent, resumable weekly loop with a
> single-writer lock, a freshness gate that degrades the surface rather than
> publishing over stale data, and a per-run manifest making every published
> number reproducible.

**J — season rollover:**
> Implement Workstream J of `BUILD_SPEC.md`. Script the annual rollover:
> per-season conference membership, subdivision changes, preseason prior
> rebuild, HFA refit, exclusion-list review, prior-season archive. Do not
> inherit the NFL's 60% rating carryover. Flag reshuffled conferences as
> low-confidence by construction.

**K — kill-check:**
> Execute Workstream K of `BUILD_SPEC.md`. Run the kill-check per market slice
> with slice-specific n. Do not pool slices. Do not re-run until it returns GO.
> Write the verdict — n, mean CLV, CI, decision — for every slice, including
> and especially the NO-GOs.

**L — red-team:**
> Execute Workstream L of `BUILD_SPEC.md`. You did not build this model and
> your job is to break it. Hunt leakage independently, re-run on different
> splits, test window sensitivity, reproduce headline numbers from scratch,
> audit the factor battery for p-hacking, and check whether the blend's
> advantage survives excluding the largest-edge decile. "No findings" is not an
> acceptable report.

**M — claims to tests:**
> Implement Workstream M of `BUILD_SPEC.md`. Convert every Tier-1 claim in
> `research/` into an assertion in `tests/test_research_claims.py`, each citing
> its source doc. A failing test means the literature doesn't hold on our data
> — document that back into the source doc. Do not loosen a test to make it
> pass.

**N — baselines:**
> Implement Workstream N of `BUILD_SPEC.md`. Reconcile our ratings against the
> Massey archive and our box scores against Sports-Reference. Report the
> biggest disagreements, not just the correlation, and investigate each past
> the threshold.

**O — calibration:**
> Implement Workstream O of `BUILD_SPEC.md`. Reliability diagrams, Brier
> decomposition, per-slice and per-confidence-bucket calibration, and a
> coverage check on the uncertainty band. Any post-hoc calibration correction
> must survive out-of-sample or it doesn't ship.

**P — vault sync:**
> Execute Workstream P of `BUILD_SPEC.md`. Reconcile `hot.md`, `_SETUP.md`, and
> the MOC against the worklog tail. Refresh `pm_*` headers. Flag contradictions
> and stale docs. Confirm every runnable entry point and research doc is
> discoverable.

---

## 22. When each of these runs

| Workstream | Runs | Depends on |
|---|---|---|
| M — claims to tests | Once, early; extended as research grows | A (data) |
| N — baselines | Per season, plus after any ingest change | B |
| G — bakeoff | Once, before committing to a method; revisit yearly | A, B, C |
| O — calibration | With every backtest; per slice | C |
| L — red-team | Before the kill-check, and after any major change | D |
| I — weekly ops | Every week in season | D, F |
| K — kill-check | Once per slice, pre-registered | D, I, L |
| H — drift | Weekly (schema, calibration); annually + midseason (factors) | E, I |
| J — season rollover | Annually, offseason | B, E |
| P — vault sync | Continuously; at minimum every session | — |

**The gate that still governs everything:** no real staking until K returns GO
for the specific slice being bet. G through P make the model better, more
honest, and maintainable. None of them substitute for that gate.
