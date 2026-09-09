# Final-score verdict — 2026-09-09

**Retain incumbent. No promising development candidate selected, no independently supported improvement, no production replacement, and no shadow activation.** The bounded C1 experiment implemented a new research score capability using the existing ridge solver. Its largest development gain was **0.06459 points of MAE per team**, below the predeclared **0.50** practical floor, with improvement in only **3/5** seasons. This is not an incumbent final-score improvement: that early route has no supported total or team scores.

Canonical evidence: [run manifest](../../data/final_score_program/20260909T024200Z-c1-v1/manifest.json), [decision](../../data/final_score_program/20260909T024200Z-c1-v1/decision.json), [paired comparisons](../../data/final_score_program/20260909T024200Z-c1-v1/paired_comparisons.json), [all metrics](../../data/final_score_program/20260909T024200Z-c1-v1/metrics.json), [independent review](evidence_review.json). The directory name is a run identifier; its manifest's `created_at_utc` records actual execution time. All historical rows are **week-granular development reconstructions**, not proof of contemporaneous pregame availability. Seasons 2021–2025 remain inspected development data.

## What was independently established

The original 106 evidence-layer tests pass, but independent reproductions expose ten unresolved defects/overclaims, preserved as **strict expected failures** in `tests/test_team_evidence_review.py`. They are not silently counted as working functionality. Major findings:

| Finding | Evidence / implication |
|---|---|
| Short-yardage eligibility is wrong | Non-scrimmage rows count in the denominator: 3,425/13,289 eligible-looking 2025 rows, including punts, timeouts, penalties and field goals. Exclude this feature. |
| Missing yards becomes failure | A synthetic missing-yard rush is counted as an unsuccessful play; no such missing rush/pass yards were observed in the inspected 2025/2026 caches. |
| Adjusted-unit lineage is incomplete | Changing Alabama's history from 2026 alone to 2025–2026 changes its adjusted rush efficiency under the same source version. Historical games, plays, membership and parameters are missing from lineage/sample semantics. |
| Week-one population disappears | No past plays can produce an empty profile set even when scheduled teams are known, instead of explicit unavailable observations. |
| FG buckets disappear | Measure IDs emitted with bucket suffixes do not match the profile lookup; a fixture with 120 made FGs has no bucket observation. |
| Drive approval alone cannot unblock | There is no drive computation in the profile builder, and `gated_pending_source` is refused unconditionally. The handoff's “change nothing here” instruction is incorrect. |
| Provenance can be bypassed | Direct `Observation` construction/replacement can carry present NaN values without provenance. Factory validation is insufficient. |
| News source/cutoff gates are incomplete | An approved requested feed can return a fact from an undeclared source; omitting kickoff can admit a future fact. These are latent reuse bugs: no populated approved feed currently enters predictions. |
| Residual conditioning is no guarantee | Conditioning on model margin does not force correlated schedule information to have zero explanatory power. The handoff's “structurally impossible” double-counting claim is false. |

The inherited unit solver also computes decay from week number without season age. Garbage-filtered scrimmage plays include overtime, so the evidence layer's plays/game is not regulation pace. None of these evidence features entered C1. News source-ID blacklists cannot establish that an extraction avoided outcome knowledge. No verified historical extraction corpus, original-document snapshot/receipt trail or independently reviewed extraction sample exists here; historical news contamination is **unassessed**, not cleared. Adding a current `observed_at` timestamp cannot recover historical knowability.

## Supported baselines and the bounded implementation

[supported_baselines.json](../../data/final_score_program/20260909T024200Z-c1-v1/supported_baselines.json) exactly reproduces the prior audit's metrics and source hashes. The early production-table route was also rerun: **all 724 selected margins match exactly**, including all **495** primary P4-involved games. Eleven additional FBS games retain the incumbent's unratable status and are enumerated in [incumbent_parity.json](../../data/final_score_program/20260909T024200Z-c1-v1/incumbent_parity.json); the candidate comparison is conditional on that declared incumbent coverage. No score is manufactured for those refusals.

| Existing record, primary scope where applicable | n | Margin MAE / bias | Team-score and total availability |
|---|---:|---:|---|
| Early selected incumbent | 495 | 13.12084 / −1.21500 | Unavailable; no supported total or team scores |
| Corrected later **unselected candidate diagnostic** | 186 | 13.19842 / −2.28436 | Own derived team means: MAE 9.68466; total MAE 13.38242; not an incumbent benchmark |
| Cross-division, separate population | 599 | 15.01615 / −4.63971 | Unavailable; deferred from primary modeling |

Early selected home-win Brier/log loss are .153832/.464273 using its existing pooled SD. Those are distinct from C1's newly train-fitted uncertainty. Later replay is still unresolved: weekly production and corrected backtest label historical drives through different routing policies, and the usable later basis points to archival evidence. No later incumbent score benchmark was invented.

The numerical implementation lives only in `nflvalue/final_score_research.py` and `scripts/final_score_experiment.py`; no production entry point imports it. It:

1. Reuses `ratings.solve` with two score equations, conference effects, opponent graph, connectivity shrinkage, HFA and season-aware `_row_weights`. An explicit zero net prior makes the new **total** component statistics-only; the unchanged incumbent margin still carries its existing talent/returning prior.
2. Correctly recovers the uncentered score intercept from the existing centered ridge design. The independent reviewer compared every paired fitted training total with the solver's actual `X @ theta`, including asymmetric schedules and conference changes, to floating-point precision. Adding exported `mu` directly to raw coefficients would not be correct.
3. Constructs total from those existing score equations, preserves the selected incumbent margin, and returns `(total ± margin)/2`. `total >= abs(margin)` constrains expected means; it never clips simulated negative scores. No primary candidate row required this constraint.
4. Uses only allowlisted game/membership fields for the total fit. Rankings, prose, news counts, return/turnover rates, prices and market-derived targets are absent. Team names identify score-history effects; relabeling names leaves predictions invariant.

The amendment was frozen in [c1_preregistration.json](c1_preregistration.json), committed as `3e3e725` **before candidate evaluation**; implementation `070dfc9` preceded the immutable run. It deliberately executes only C1, replacing the proposed adapter prior-games grid with the existing solver's shrinkage/decay controls. Six contenders are P4-involved or all-FBS training crossed with one-season, three-season or all-since-2019 history. C2/C3 were not silently added.

Outer seasons are 2021–2025. Inner expanding chronological origins start with 2019 → 2020; each feature fit uses only earlier weeks/seasons. Ridge penalties `[3,30,300,3000]` and per-week decays `[0,.02,.05]` are selected only inside outer training. The one-standard-error rule prefers stronger shrinkage, then the declared decay tie-break. All 30 outer fits selected ridge 3000 / decay .05. SD/covariance come from paired inner chronological residuals; no outer result enters preprocessing, parameter selection or uncertainty. Initial tuning/calibration has only **16 primary 2020 games**, then 109, 202, 300 and 405. The same inner predictions select hyperparameters and estimate covariance, so selection optimism inside training remains a limitation.

## Results and rejected hypotheses

All rows below use the **same 495 games**. Bias is forecast minus actual. The common research reference is the frozen incumbent margin plus the P4 all-history training mean total. It is explicitly a new research reference, never an incumbent score. The simple reference predicts training mean home/away scores without the incumbent margin.

| Model / training | Team MAE | Home MAE | Away MAE | Total MAE | Total bias | Gain vs common research reference |
|---|---:|---:|---:|---:|---:|---:|
| Simple mean home/away | 11.7961 | 11.8461 | 11.7460 | 12.8148 | +2.6597 | −2.5288 |
| Common research reference | 9.2673 | 9.6068 | 8.9279 | 12.8148 | +2.6597 | — |
| C1 P4, one season | 9.2130 | 9.5650 | 8.8610 | 12.6382 | +1.7033 | +.0544 |
| C1 P4, three seasons | 9.2027 | 9.5506 | 8.8549 | 12.6265 | +1.8813 | +.0646 |
| C1 P4, all history | 9.2088 | 9.5576 | 8.8600 | 12.6411 | +1.9375 | +.0585 |
| C1 all FBS, one season | 9.2352 | 9.5717 | 8.8987 | 12.6993 | +2.2593 | +.0321 |
| C1 all FBS, three seasons | 9.2407 | 9.5730 | 8.9084 | 12.7173 | +2.4624 | +.0267 |
| C1 all FBS, all history | 9.2438 | 9.5758 | 8.9118 | 12.7255 | +2.5120 | +.0235 |

Best descriptive gain, C1 P4/three seasons: paired week-block 95% interval **[.01676,.12212]**, simultaneous six-comparison interval **[−.00999,.13917]**, season-block interval **[.00575,.11673]**. The inherited .50 practical criterion fails, as does 4/5 nonnegative seasons. Twenty thousand bootstrap draws use seed 20260908 and season-week blocks; season resampling and leave-one-season-out sensitivity are preserved. With only five seasons these uncertainty estimates are approximate development summaries.

More history did not help materially: all versus three seasons changed team MAE improvement by **−.00605** for P4 training and **−.00314** for all-FBS training, on identical games. Broader training also did not supply a practical gain. No extra variants were added after observing these results. The simplest viable research reference remains competitive.

Weekly total-state adaptation versus freezing only the total fit at week one changes P4/three-season team MAE from 9.21674 to 9.20274 (gain .01399). The incumbent margin updates in both arms; this is not a fully frozen-model test. Strong ridge shrinkage reduces the synthetic single extreme-game influence. The generic one-event rate probe verifies raw sensitivity `1/n` versus `1/(n+k)` under shrinkage; it does **not** validate a real turnover/return feature or show that either improves forecasting.

| Hypothesis / pressure test | Disposition |
|---|---|
| C1 existing offense/defense score level adds useful accuracy | Tested six variants; practical improvement rejected. Removing the added scoring component gives the named research reference; this comparison does not isolate causal contributions of each coefficient block. |
| Longer or broader training helps | Paired tested as above; no practical benefit in this early cohort. |
| Market removal/scrambling and future result/news changes | Positive-schema and poisoned-future tests pass for candidate construction. No market file is opened by the runner; it uses the separately preserved incumbent anchor. Historical availability is still unverified. |
| C2 possessions; C3 success units | Blocked by drive implementation/estimand and unit lineage/season-decay defects. No numerical ablation or no-effect claim. |
| Common-opponent incremental value / correlated families | Not tested numerically; no audited incremental feature exists. Residualizing against margin does not eliminate this obligation. No duplicate common-opponent term was added. |
| Raw versus shrunken real turnovers/returns; explosive events | Deferred pending event attribution, denominators and coverage validation. Only generic influence stress tested; no performance claim for those families. |
| Small-sample, roster-change, missing-news cases | Every early row is small-sample (<4 current games on at least one side). Sample/fallback counts are per game. Verified roster strata unavailable; news uniformly unavailable, not “healthy.” |
| Prominent versus less-covered teams | Name invariance passes. Performance and source-coverage fairness comparison unavailable: no independently defined prominence labels or approved news corpus. Conference pair counts are retained and are not relabeled media coverage. |
| Statistics-only versus verified news additions | Unavailable, not a tie or zero effect. No feed, extraction lineage, or validated numeric personnel adjustment was fabricated. |

## Distribution quality and postmortems

C1 and its common reference preserve margin MAE 13.12084. Their train-fitted Gaussian margin approximation gives Brier .153345 and log loss .461386; these do not change across C1 totals. Calibration intercept/slope diagnostics are −.02352/1.01666 with unclustered Wald intervals explicitly labeled. They are descriptive checks, never fitted on outer outcomes and fed back into a forecast. Reliability bins include counts and Wilson intervals.

For C1 P4/three seasons, home/away/total CRPS are 6.71321/6.25254/9.00295. Home/away/total nominal 95% coverage is .93737/.95354/.96566, with average widths 45.32/44.35/66.42 points. Margin coverage is **.92525**, below nominal .95, with width 60.01. The total remains inflated by **1.8813 points**; home/away biases are +.33316/+1.54816. Home/away 95th-percentile absolute errors are 23.57/21.58. Average Gaussian mass below zero is **1.076% home / 4.868% away**. These are probabilities under an approximation, not observed negative football scores.

The existing simulator is unchanged. Explicit reproduction confirms that adding antithetic continuous margin noise after scoring preserves mean margin and every total, yet can produce fractional/negative implied paired scores. In the deterministic valid 42–0 fixture, all 4,000 implied paths are fractional and 2,000 have a negative score. This demonstrates unsupported literal-score interpretation, not production incidence. Existing discrete simulation and overtime tests remain active. C1's target is official final scores including OT; it adds no second OT increment.

Expected means may be continuous. Literal outcome samples must be nonnegative, discrete, overtime-consistent, and drawn from a specified joint law. We expose no paired score-sample API, exact-score mass, supported joint energy score or discrete PIT here. Gaussian error diagnostics do not satisfy that missing joint-score requirement. Clipping negative draws is not an accepted repair.

[forecasts.csv.gz](../../data/final_score_program/20260909T024200Z-c1-v1/forecasts.csv.gz) preserves 17,376 immutable model/reference rows; [postmortems.csv.gz](../../data/final_score_program/20260909T024200Z-c1-v1/postmortems.csv.gz) preserves all 4,344 candidate-game rows (724 games × six variants), their original anchor/forecast, actual home/away scores, margin/total and all errors. Identical primary game keys are maintained across variants. P4 fallback counts are 71/10/9 for one/three/all-season P4 training and 1/0/0 for all-FBS; fallback to that arm's training reference is explicit and stays in the loss.

Pace, efficiency, field-position and personnel **discrepancies** are explicitly unavailable where no reconciled prediction/observation pair exists. Final-score errors alone cannot distinguish data failure, missing knowable information, relationship error or outcome variation. Default tags therefore say unexplained variation without causal attribution; coverage fallback receives its own data-coverage tag. A future analyst may append independently sourced diagnostics keyed to the original forecast hash; never overwrite the forecast or invent a causal decomposition.

## Verification, artifact convention and Opus handoff

The canonical API specification is [score_confidence_api.json](score_confidence_api.json): 34 fields, each with units, target, timing, model version, availability and uncertainty semantics. It distinguishes expected points, predictive-error SD/covariance, win probability approximation, unavailable literal-score samples, and rating-parameter uncertainty. No qualitative confidence label or prose adjustment is authorized. Machine-readable forecasts/evidence remain canonical; Obsidian and website are derived views.

Test-first evidence: initial research test collection failed because the module did not exist; the implemented research suite passes. Independent review added two direct centered-design checks and ten strict expected failures for unresolved evidence bugs. [model_verification.json](model_verification.json) records final working-tree/clean-checkout/CI checks and versions. Four pre-existing cache-dependent CI assumptions were repaired only in tests: three use explicit self-contained fixtures, one real-data integration is explicitly cache-gated. Production evidence verification still fails closed on missing/tampered declared inputs.

Reproduce from the repository, using the existing environment and exact cache hashes:

```bash
.venv/bin/python scripts/verify_final_score_run.py data/final_score_program/20260909T024200Z-c1-v1
# Above needs no private caches. To verify all 43 producing input hashes too:
.venv/bin/python scripts/verify_final_score_run.py data/final_score_program/20260909T024200Z-c1-v1 --inputs
# Full numerical rerun must target a NEW directory:
.venv/bin/python scripts/final_score_experiment.py --output /tmp/stallions-c1-new-run
.venv/bin/python -m pytest -q
```

CSV gzip files are lossless transport. `artifact_hashes.json` hashes their decompressed canonical bytes; `delivery.json` records transport hashes. The runner recreates plain CSVs. Do not overwrite an existing run. Missing private caches are an explicit rerun blocker, not permission to download replacement data or silently switch versions. Code, protocol, input hashes, anchor parity, all 360 inner hyperparameter summaries, 30 selected fits/covariances, all six contenders, reference/sensitivity views, exclusions, losses, bootstrap output and decision are retained.

**Self-contained next tasks for Opus, in order:**

1. Read `00`–`03`, `evidence_review.json`, the API and immutable decision. Keep incumbent and shadow status unchanged. Do not extend C1's grid after this failed run.
2. Own repairs to `nflvalue/team_evidence/{contracts,measures,profile,news,sources}.py` and matching tests in separate reviewed commits. Turn each strict xfail into a real passing regression by fixing eligibility, missingness, complete lineage, profile population, FG buckets, source-ID matching and mandatory cutoff. Preserve existing generated/private notes and frozen pregame blocks. Extend the news record with original-document hash/locator, publication/retrieval/effective times, extractor version and verification status; never backdate receipts.
3. Own T0 later-route replay instrumentation in new research files first. Prove production/backtest parity and resolve selected-basis coverage before calling anything a later incumbent score benchmark. Do not edit the evidence manifest merely to mark old evidence selected.
4. Specify and validate a discrete joint-score representation using the existing scoring/pace/OT architecture before exposing literal samples. Keep continuous contrast-error diagnostics separately typed; test nonnegative discrete support, score/margin/total identities, ties/OT, sample moments and mass/coverage without clipping. Any model alternative requires a new preregistration.
5. Only after evidence repair, preregister one C2 or C3 family and its removal/common-opponent/turnover-return ablations, preserving nested chronology, identical games, current-season state adaptation, inner uncertainty fits, the inherited practical criteria and no-improvement option. Validate actual source coverage and event denominators before feature activation. News comparison remains blocked until a verified corpus exists.

Ownership this session: modeling lead owns the isolated C1 adapter/runner/verifier, research tests, C1 protocol/API/verdict and run artifacts; independent reviewer owns `evidence_review.json` and `test_team_evidence_review.py`. The four CI test files have scoped fixture edits only. **Untouched:** production engine/ratings/Kalman/simulation/forecast contract, weekly/rendering jobs, frozen forecasts, selected evidence/bases, factor/source approval status, source caches, market files, website and all 99 pre-existing dirty/untracked files. Opus's previously committed display-only layer is included in the draft review ancestry with its defects disclosed. No merge or deployment.
