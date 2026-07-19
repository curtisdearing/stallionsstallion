# Agent handoff — build brief for stallionsstallion

**Purpose:** everything an AI agent needs to build this project correctly,
starting cold. Section 1 is the prompt to paste. Sections 2–10 are the context
that prompt assumes.

**Tip:** if your agent tool auto-loads `AGENTS.md` or `CLAUDE.md`, copy or
rename this file to that name so it loads without being pasted.

---

## 1. PASTE THIS TO START A SESSION

> You are building **stallionsstallion**, a college football game-line
> prediction model (spreads/totals/moneyline). It is a research project, and
> it is **paper-only** — no real money is staked until a specific statistical
> gate passes, which has not happened.
>
> **Before writing any code, read, in this order:**
> 1. `hot.md` — current state and open actions (project root, one level up)
> 2. the last ~5 entries of `_worklog.md` — what just happened, and whether
>    another agent has an open claim on a file you intend to touch
> 3. `stallionsstallion/AGENT_BUILD_PROMPT.md` — this file, in full
> 4. `stallionsstallion/BUILD_SPEC.md` — the engineering spec, especially
>    section 0 (Non-negotiables) and section 10 (sequencing and gates)
> 5. `stallionsstallion/_SETUP.md` — what is runnable vs. stubbed vs.
>    deliberately import-broken. **Read a file's own docstring before
>    concluding it is a bug.** Several files are intentionally non-functional.
>
> **Your task this session:** implement **Workstream [A/B/C/D/E/F]** from
> `BUILD_SPEC.md`. Do that workstream only. Do not start the next one; its
> gate has to pass first.
>
> **Rules that override any instinct to be helpful:**
> - Never weaken a guardrail in `BUILD_SPEC.md` section 0 to make something
>   work or make a number look better.
> - If a gate fails, **write down that it failed**. Do not tune until it
>   passes. A recorded NO-GO is a successful session.
> - Never delete a file in this vault. Overwrite with an explanatory
>   docstring-only stub instead (see section 5 below).
> - Never commit or print an API key.
> - Ask before: creating a git repo or remote, adding a dependency, or
>   changing anything in section 0.
>
> **When done:** append one line to `_worklog.md` in the exact format in
> section 6 below, and update `hot.md`'s `pm_next_action` and `pm_blocked_by`.
> Report what you built, what you verified, what failed, and what surprised
> you — surprises are the most valuable thing you can report.

---

## 2. What this project is, and where it stands

A CFB game-line model ported from **fablesfable**, a sibling NFL betting
project in the same vault. It reuses that project's ratings → Monte Carlo →
market architecture, adapted to college football's sparser schedule,
conference structure, and much wider talent spread. Player props are
explicitly out of scope.

**State as of 2026-07-17:**
- Code is a curated duplicate of fablesfable's game-line layer. **No CFB data
  has ever been pulled. No modeling code has been run.**
- The only thing built for CFB is the **Phase 0 smoke test**
  (`phase0_smoke_test.py` + `nflvalue/sources/cfbd.py`), which is tested
  offline but has never run live — it needs a free CollegeFootballData API key.
- Everything in `research/` is a **hypothesis**, not a validated finding.
  Treat it as a well-sourced literature review, not as ground truth.
- There is **no git repository** anywhere in this project.

---

## 3. The non-negotiables, compressed

Full versions in `BUILD_SPEC.md` section 0. An agent must not trade any of
these away:

1. **Paper-only** until the Phase 4 kill-check returns a slice-specific GO.
   No output may imply a bet is recommended before that.
2. **CLV (closing line value) is the scoreboard, not win rate.** Log opening
   and closing lines from day one.
3. **The model is never bet straight** — it is blended with the market. The
   sibling project's own backtest lost money at every market betting the raw
   model into the close.
4. **Walk-forward or it doesn't ship.** Any same-game statistic used to
   predict that same game is leakage.
5. **Explanatory context is display-only by construction** — assembled after
   ranking, architecturally unable to feed back into a projection.
6. **Situational factors need n-gate + matched control + BH q<0.05** before
   touching a probability. This project has a retraction on record from
   skipping exactly this.
7. **Uncertainty is a first-class output.** A wide band is a reason to pass,
   not a footnote.

---

## 4. Repo map and landmines

Paths are relative to `stallionsstallion/` (the code dir) unless noted.

**Built for CFB, working:**
- `nflvalue/sources/cfbd.py` — CFBD API client, stdlib-only, counts API calls
- `phase0_smoke_test.py` — Phase 0 smoke test; `--selftest` runs offline
- `tests/test_phase0_smoke.py`, `tests/fixtures/cfbd_smoke_fixtures.json`

**Reusable as-is (do not reimplement):**
`oddsmath.py` (de-vig/consensus), `killcheck.py` (GO/NO-GO gate),
`premortem_mc.py` (Kelly/ruin math), `clv.py`, `freshness.py`, `db.py`,
`notify.py`, `document.py`, `dashboard.py` (self-contained HTML pattern),
`game_notes.py` (the display-only-context pattern to copy),
`nflvalue/config.py`, most of `nflvalue/sources/`.

**The core to modify (runs today, NFL-shaped, needs CFB rework):**
`build_ratings.py`, `nflvalue/montecarlo.py`, `backtest.py`,
`nflvalue/learn.py`, `nflvalue/model.py`, `nflvalue/factors.py`.

### Landmines — read this before filing a bug against yourself

- **The package is called `nflvalue`.** That is legacy naming from the NFL
  parent project, not a mistake. Do not rename it mid-build.
- **`nflvalue/ingest.py` and `scripts/bootstrap_history.py` do not import.**
  This is deliberate — they are nflverse/NFL-shaped internally and are
  Workstream A's rewrite target. Not a bug.
- **Several files are docstring-only stubs**, including `tune_weights.py`,
  `nflvalue/report.py`, `tests/conftest.py`, and 8 test files. They were
  player-prop-entangled and out of scope. Each stub's docstring explains
  itself and points at the working original. Not bugs.
- **`tune_weights.py` is a stub of the *prop* weight tuner** — it is not the
  game-line tuner. The game-line analog to extend is `nflvalue/learn.py`.
- **`tests/test_leakage.py` is a stub of the single most valuable pattern in
  the parent repo** (poison a future week, assert nothing earlier moves).
  Re-implement that technique against the new engine. Do not skip it because
  this particular file didn't port.
- **CFBD's `/plays` is per-week**, not a bulk season pull — it needs
  year + week + seasonType. `/games`, `/lines`, `/talent` are bulk. A full
  season costs ~18 calls. The free tier is **1,000 calls/month**; the client
  counts calls, so track and report the burn.
- **Conference membership must be a per-season lookup**, never hardcoded.
  Realignment moves teams constantly, including a rebuilt Pac-12 in 2026.
- **FCS opponents are not FBS data.** Tag them as a separate tier; don't feed
  those blowouts into the FBS rating fit at face value.
- **Garbage time must be filtered before any rate stat is computed.** CFB has
  dozens of 35+ point games weekly; unfiltered fourth-quarter stats will
  visibly corrupt ratings. The NFL parent doesn't need this — you do.

---

## 5. Environment and setup

```bash
cd stallionsstallion
pip install -r requirements.txt        # pandas, pyarrow, numpy, scipy, sklearn
python phase0_smoke_test.py --selftest # verify your environment, no key needed
python -m pytest tests/ -q
```

- **Python standard library is preferred for I/O**; pandas/numpy for modeling.
  Adding a dependency requires asking first.
- **API keys** come from env vars (`CFBD_API_KEY`, `ODDS_API_KEY`) or the
  gitignored `config.local.json` (template: `config.local.json.example`).
  Never put a real key in `config.json` — that file is tracked.
- **There is no git repo.** Nothing is version-controlled, so there is no undo.
  Be conservative with overwrites. Creating a repo or remote requires asking.

---

## 6. Working conventions (load-bearing — this vault depends on them)

**Worklog discipline.** Append one line to `_worklog.md` (project root) after
every meaningful action — a data pull, a code change, a decision, a
retraction. Append-only, **newest at the bottom**. Never hand-edit or reorder
past entries. Exact format:

```
- <utc timestamp> [s:session] KIND: summary || next: … || files: …
```

`KIND` is one of: `start`, `progress`, `done`, `decision`, `claim`,
`release`, `handoff`, `block`. Be specific and dense, not essayistic — a
future agent reads the tail to learn where things stand. Include what
surprised you and what you deliberately did *not* do.

**Claims.** Before editing a file, scan the worklog tail for a `claim` on it
with no later `release`. If one exists, another agent is mid-edit —
coordinate, don't clobber. For long edits, post your own `claim`, then
`release`.

**`hot.md`.** The rolling state snapshot at the project root. Read it first;
update its `pm_next_action`, `pm_blocked_by`, and decisions block at the end
of your session. It is a cache distilled from the worklog, not the source of
truth.

**Never delete a vault file.** If something must go away, overwrite it with a
docstring-only stub explaining why and pointing at any working original. This
is why several stubs exist already.

**Don't re-derive what's in `research/`.** Six planning docs already cover the
modeling content. Read before deciding; cite them in your worklog entry when
they drove a choice.

---

## 7. Execution order and gates

Work one workstream per session, in this order. Each gate must pass before the
next starts. Full detail in `BUILD_SPEC.md` sections 2–7 and 10.

| # | Workstream | Gate before proceeding |
|---|---|---|
| 0 | Phase 0 smoke test (built) | Run live; three endpoints confirmed |
| 1 | **A — ingest** | Leakage + schema-contract tests pass |
| 2 | **B — ratings** | Reproducible run-to-run; connectivity populated |
| 3 | **C — simulation** | CFB baseline Brier + margin-corr reported honestly |
| 4 | **D — market/CLV** | Blend beats raw model *and* raw market out-of-sample |
| 5 | **F — explainer** | Contributions sum; "why not" panel never empty |
| 6 | **E — factors** | Each clears n-gate + matched control + BH q<0.05 |
| 7 | Kill-check | Slice-specific positive CLV before any real staking |

The explainer (F) comes before the factor battery (E) on purpose: it is the
surface that makes factor evidence legible, so factors get judged honestly.

Per-workstream kickoff prompts are in **`BUILD_SPEC.md` section 9** — paste the
one matching your assigned workstream into the placeholder in section 1 above.

---

## 8. Decide autonomously vs. ask first

**Decide yourself:** file and function layout, algorithm implementation
details, test design, refactors inside a workstream, variable naming, caching
strategy, how to render something in the HTML surface.

**Ask first:**
- Creating a git repo or a GitHub remote (touches stored credentials)
- Adding any dependency
- Changing anything in `BUILD_SPEC.md` section 0
- Widening a workstream's scope, or starting the next one early
- Anything that would stake, or appear to recommend staking, real money
- Deleting or repurposing an existing file

**Never guess a constant.** Ridge strength, conference-shrinkage weight,
recency decay, garbage-time threshold, blend weight — all are fit by
walk-forward search, not chosen by intuition. If you can't fit it yet, say so
and leave it parameterized.

---

## 9. What "done" means, and how to report it

A workstream is done when its gate in section 7 passes **and** you can answer
all of these in your worklog entry:

1. What did you build? (files, and what each does)
2. What did you verify, and how? (tests run, with actual results — not "tests
   should pass")
3. What failed or came back negative? (state it plainly; a negative result is
   a result)
4. What assumption in `research/` or `BUILD_SPEC.md` turned out to be wrong?
5. What did you deliberately not do, and why?
6. What is the next action, concretely enough that a cold agent could start?

**Do not report success you did not verify.** Do not describe code as tested
if you did not run the tests. If you could not run something (no API key, no
data), say exactly that — the previous session did precisely this for Phase 0
and it was the right call.

---

## 10. Failure modes to avoid

These are the specific ways this build goes wrong. They are ranked by how
likely they are and how much damage they do.

1. **Tuning until a gate passes.** The Phase 2 gate is "the blend beats both
   inputs out-of-sample." If it doesn't, that is the finding. Repeatedly
   adjusting until a number clears a threshold is how a model gets fooled.
2. **Leakage.** The most common silent failure in sports modeling. Any feature
   built from the game being predicted, or from data that wasn't available
   before kickoff, invalidates every downstream number. Re-implement the
   poison test.
3. **Treating `research/` as validated.** Those documents are a sourced
   literature review. Every claim needs to survive this project's own
   walk-forward backtest.
4. **Assuming NFL magnitudes transfer.** The backup-QB effect, home-field
   advantage, drive-outcome rates, and the bets-needed-to-prove-an-edge table
   were all measured on the NFL. CFB numbers must be re-measured, not inherited.
5. **Building an interface that only argues for betting.** The explainer must
   render the case against with equal weight; an empty "why not" panel is a
   bug. In a market this efficient, most games have no edge — the realistic
   ceiling is ~52.5–55% ATS against a 52.38% break-even.
6. **Silent degradation.** A stale feed or a failed fetch must fail loud and
   show as degraded in the UI, never render confident numbers over old data.
7. **Scope creep into player props.** Out of scope by design. If you find
   prop-related code, leave it stubbed.

---

*This project is research, and paper-only until its own kill-check says
otherwise. If betting stops feeling like the modeling exercise it is here,
that's worth noticing — 1-800-GAMBLER.*
