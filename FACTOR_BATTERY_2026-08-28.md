# Context-layer factor battery — run 1, 2026-08-28

Pre-registered in `data/factor_prereg_2026-08-28.json`, written and appended to
`_worklog.md` **before** the analysis code was executed. One execution.
Nothing was promoted. Nothing was retuned to pass.

## Results

| Hypothesis | Shape | n | Effect | p | BH crit (m=5) | Verdict |
|---|---|---|---|---|---|---|
| `returning_production_diff` | continuous | 2,328 | −3.8335 pts per unit | 2.83e-05 | 0.0100 | **pass** — but read the caveat |
| `special_teams_field_position_diff` | continuous | 2,377 | −0.1212 pts per yard | 0.0011 | 0.0200 | **pass** |
| `rest_days_diff` | continuous | 2,377 | +0.2474 pts per day | 0.0318 | 0.0300 | reject |
| `coach_year2` | event | 835 pairs | +0.5395 pts | 0.5146 | 0.0400 | reject |
| `short_week` | event | 16 treated | — | — | 0.0500 | underpowered (gate 40) |

Evaluation set: `backtest_c.csv` + `backtest_c_2024.csv`, 2021–2024, weeks 4–15,
n = 2,377. Outcome is the signed walk-forward residual `margin − pred_b`.
No line, price, book or CLV column enters any frame — asserted by test.

## The two passes are not two new signals

**`returning_production_diff` is endogenous, and wrong-signed.** `ratings.solve`
uses `preseason_prior` as the ridge **prior mean**, and that prior is built from
talent **+ returning production** with `prior_w_returning = 1.0`. Returning
production is therefore already inside `pred_b` at every week. The registration
failed to declare that — a defect in my own pre-registration, recorded rather
than quietly fixed. Combined with the sign (−3.83 against a hypothesised
positive), the parsimonious reading is not "returning production predicts
residuals" but **"the preseason prior over-weights returning production, and the
residual is correcting it."** That is a model-calibration finding. Re-fitting
`prior_w_returning` needs its own pre-registration and a walk-forward search —
never a guessed constant.

**`special_teams_field_position_diff` overlaps a factor that already passed.**
It correlates with the registered `field_position_diff` (slope +0.6165, passed
2026-07-30), which mixes offence-earned and special-teams-earned field position.
The overlap was declared in advance. Not independent evidence; do not count it
twice. Magnitude is 0.12 pts per yard of net punt field position — a fraction of
a point at realistic differentials.

## A bug I caught in my own run

The first execution computed Benjamini-Hochberg over only the four hypotheses
that produced a p-value, silently dropping the underpowered one and shrinking
m from 5 to 4. The registration says m = 5 and "never dropped to shrink m".
A smaller m loosens every threshold, and it mattered: `rest_days_diff`
(p = 0.031764) clears the m=4 threshold of 0.0375 and **fails** the registered
m=5 threshold of 0.0300. Corrected before anything reached the registry — the
underpowered hypothesis stays in the family at p = 1.0. The correction moved the
result *away* from a pass.

There is also an erratum in the registration itself: the special-teams entry
states a formula and then labels its direction inconsistently with that formula.
The formula is the operative definition and was implemented exactly as
registered; the label was wrong. The test is two-sided, so no verdict changed.
Both are recorded in the script's module docstring rather than edited away.

## Families recorded but not tested

| Family | Why | What would unblock it |
|---|---|---|
| QB availability | **Already adjudicated 2026-07-30**: `backup_qb_pregame` was pre-registered and rejected (p = 0.075, n = 745). Re-running the same test on the same data is a re-report; picking a new definition after seeing that rejection is the variant search this protocol forbids. | An approved live availability feed plus a fresh pre-registration |
| OL / key-unit availability | **No source at all.** The factor cannot be defined, let alone tested. | A named provider through the `unit_availability_feed` gate |
| Weather | No approved forecast provider, and CFBD's weather is **observed** — recorded at or after kickoff — so a pregame weather factor would be same-game leakage by definition. | A forecast provider with a `forecast_issued_at` distinct from fetch time |
| Travel / time zone | **Blocked on the `cfbd_venues` gate.** `research/circadian-timezone-factor-plan.md` already pre-registers a census first and a priced-in test second; this run does not override that. Census fraction counted, **no effect estimated**. | Open the gate (1 bulk call), complete the census, then the priced-in test |
| Portal | Not cached, source unapproved. `returning_production_diff` stood in as a labelled proxy. | Cache the portal feed and approve `cfbd_portal` |

### Travel census — counts only

Of the 2,377 walk-forward evaluation games: 3 kick before noon ET, 1,169 at
12:00–15:59, 841 at 16:00–19:59, 222 at 20:00–21:59, and **142 at 22:00 ET or
later** — one game short of the 143 n-gate a kill-check slice needs. Across the
full 2021–2024 schedule the late slice holds 295. Zones crossed, travel distance
and body-clock hour stay uncountable until venue coordinates are cached.

## Registry and isolation

`data/factor_registry.json` went from 14 to 24 entries. Every new entry —
including all five untested families — landed `display_only`. The
`gated_active` set is byte-identical to what it was before the run:
`league_hfa`, `per_team_time_varying_hfa`.

Requirement 7 is structural, not procedural. All eight forecast-producing
modules (`cfb_sim`, `ratings`, `engine`, `kalman_engine`, `montecarlo`,
`gameline`, `bakeoff`, `cfb_ot`) were checked by AST and text: **none reads the
factor registry at all.** There is no code path by which a rejected, display-only
or stale factor could reach a margin, total or win probability, whatever its
status says. A staleness helper was added so an active factor past its
revalidation date reads as stale rather than silently fresh — and a
non-date placeholder like "2026 season end" reads as *unknown*, never as overdue.

29 tests green, ruff E,F clean.
