# Runtime and statistical audit — 2026-09-08 ET / 2026-09-09 UTC

**Conclusion: retain the incumbent. No engine was changed, no challenger shadow log was activated, and this audit establishes no model improvement.** The largest demonstrated final-score gap is missing or uncertified scoring-level evidence, not an absence of modeling infrastructure. Reuse the existing ratings, drive, pace, OT, snapshot and validation interfaces. The implementation contract is [00_contract.md](00_contract.md).

Evidence scope: local repository HEAD `c6c780f521bf955caf84e3fb46287f3a19c2e2ab`, with pre-existing dirty files and untracked research modules explicitly included in the read-only inspection. The hashes in [audit_evidence.json](audit_evidence.json) identify the bytes examined; the artifact is canonical for measurements. Git status was read before code. `early_challenger.py` and `performance_mc.py` are **untracked local work**, not shipped modules. `weekly_loop.py` includes uncommitted grading work. This report does not imply those changes are in a fresh checkout.

## 1. Runtime/dataflow map

```text
CFBD caches: games / season teams / talent / returning / drives
  ingest.load_* -> ratings.game_frame (completed scores, season membership)
  schedule = full ingest.load_games target week, including non-FBS/refusals
  weekly_loop.stage_freshness -> capability-specific health gate
  engine.solve_production_table
    weeks 0–3 -> solve_early_season_ridge_table
      previous_season_net -> previous final ridge table
      ratings.solve(hist < week, preseason_prior(talent, returning, prev_net))
      producers.early_ridge_forecasts -> Normal(margin, registered residual SD)
    weeks 4+ -> configured engine (observed: kalman)
      kalman_engine -> bakeoff.KalmanModel -> net-strength table
      own-week production tables label historical FBS-v-FBS regulation drives
      DriveRateModel + pooled paired PaceModel + past-only OT fit
      cfb_sim.simulate_game -> margin/total arrays -> summary
    FBS-v-non-FBS override, any week -> cross_division ridge / pooled __FCS__
      early: same prior ridge; later: plain ridge solve_cached
      producers.cross_division_forecasts -> Normal margin, separate residual SD
  forecast_contract.Forecast / PredictiveMargin / BasisRegistry
  build_artifact -> schedule-complete rows (numbers OR refusal reason)
  snapshots.freeze_kicked_off -> apply_frozen -> validate
  _publish_week -> immutable stamp -> latest pointer -> convenience predictions.csv
      -> insight JSON cards -> hub/API/Obsidian derived views
      -> post-freeze market board/ledger/picks diagnostics (separate outputs)
      -> later finals -> scoreboard recap/CLV/hypothesis diagnostics
```

### Routes and what is actually published

| Route | Construction / cutoff | Forecast outputs and uncertainty |
|---|---|---|
| Early, weeks 0–3 | `engine.py:solve_production_table`; `producers.py:build_early_week`, `early_ridge_forecasts`; fitted historical score ridge around talent/returning/previous-rating prior, strict earlier weeks | `ridge_preseason_prior`: home margin, Normal-derived home-win p, predictive SD 16.8982 from 724 historical early games. `rating_parameter_sd` is the **sum** of the teams' net bands, a separate technical diagnostic. No simulation, no supported total, no team scores. |
| Later, weeks 4+ | `weekly_loop.py:stage_simulate`; historical tables from `solve_production_table`, `cfb_sim.label_week_drives`; actual config returns Kalman | `kalman_montecarlo`: MC mean margin, SD, p(home win), mean total and separate parameter SD. Full arrays exist in memory but public v2 drops them and total SD. No explicit score columns or score-sample artifact. |
| Cross-division, any week | `cross_division.py:solve_table/predict_margin`; `producers.py:cross_division_forecasts`; exactly one season-listed FBS side, pooled non-FBS side | `ridge_cross_division`: margin Normal with SD 17.8957, distinct basis/version. No total/scores. It cannot distinguish strong and weak FCS teams. Non-FBS-versus-non-FBS is refused. |
| Missing prior, rating, source or basis | Producer errors and contract refusal/degraded/uncertainty-unvalidated states; schedule ID equality checked | No made-up scores or replacement narrow rating band. Missing histories can also abort a stage before per-game refusal assembly; test this path in the future harness. |

The current convenience artifact has 455 rows: 70 priced, 19 frozen, 366 refused; 89 forecast-bearing rows and **zero totals**. Family counts on refused rows are provenance labels, not extra forecasts. It is a mutable convenience view of earlier snapshots, not proof that a currently “priced” game remains pregame. No grading or snapshot refresh was run for this audit.

`mc_forecasts` hardcodes the Kalman MC family. Although `engine.margin_engine` allows a local config override to ridge after week 3, the MC family labeling does not become a distinct ridge-MC family. The observed setting is Kalman; alternate overrides require an explicit route-contract test before use.

### Published surfaces traced to inputs

| Surface | Backward trace / limitation |
|---|---|
| Immutable `data/snapshots/<season_week>/<stamp>/predictions.csv` and snapshot metadata | `_publish_week` → contract rows; pregame values freeze from the newest snapshot strictly before each kickoff. `frozen.json` preserves selected rows. Hash verification and exact schedule coverage are reusable protections. |
| `data/predictions_*.csv` | A reread/copy of the latest immutable stamp, through `FC.read_artifact/write_artifact`. Consumers must retain per-row family/basis and refusal states. |
| `data/insights_*.json` | `_publish_insights` → `insight.build_week` from **published** rows and the same rating table's components. No context/market object is supplied here. Standalone `render_insights.py` re-solves priced-row decompositions; frozen rows omit reconstructed reasoning. It cannot independently certify old source snapshots. |
| `data/explainer_*.html` | Later weekly route builds `explain.build_explanation` from in-memory MC arrays, margin histogram/deciles, exact linear contributions plus simulation adjustment. `market=None` in this route; optional context hook is additive after calculation. Early weekly publication does not refresh this HTML, so a file of that name can be stale. |
| Historical `scripts/render_explainer.py` | Recomputes a historical simulation and joins historical lines for display; **OT still fits 2021–2025 pooled data**, regardless of requested historical cutoff. Do not use this renderer as a historical forecast/evidence producer. |
| `data/site/api/hub.json` | `build_hub_feed.py` reads/validates latest contract rows, cards and recap data; it does not manufacture ridge MC totals. Current local generated feed may lag code/forecasts; no regeneration or live-site verification was performed. |
| `data/site/index.html` | `build_site.py` reads factor registry, CLV files, recaps and `evidence.benchmark_or_notice`. No selected later benchmark means “awaiting revalidation,” not fallback to the archival CSV. |
| Vault `picks/*` MD/CSV | `render_picks.py` reads contract plus insight cards; winner and p use the artifact, margin intervals use a **Normal approximation** mean±z·SD. Compact/value/next-days variants additionally read frozen ledger or cached lines, derive cover/EV/scale-adjusted diagnostics and rank/filter them. These are market-derived downstream views, not training features or final-score forecasts. No total is filled in. |
| Board and board ledger | `build_board.py` joins forecast artifact to market/context; `board_ledger.py` freezes pregame leans and grades them. Both belong downstream of frozen forecasts, and their selections must not determine this program's evaluation population. Existing dirty reconciliation work is preserved. |
| Recaps, CLV and miss review | `scoreboard.py` reads published contract rows + later finals; `stage_clv_capture` joins lines after forecasts; untracked `miss_review.py` reads frozen predictions and finals. Hypothesis/watch-list output does not feed ratings. Current scoreboard still contains historical hardcoded benchmark constants; those are not a final-score baseline. |
| Operational reports and publication | `stage_refit` writes a ratings table; `stage_honesty` reads lines for a diagnostic; neither supplies a new score target. `run_weekly.sh` builds local site after grade and renders vault picks after forecasts. `publish_site.sh` is separate. GitHub Pages triggers master+site-path pushes or manual dispatch; a docs feature push triggers tests, not this Pages workflow. |

The weekly forecast's numerical kernel does not ingest sportsbook lines, but the **process** is not a hermetic market-free job: refresh/honesty read lines, the source fingerprint includes `lines*`, and later simulation reads archival CLV for explanation. Thus “no line in the score equation” is supported; “the forecast construction process cannot access lines” is not yet supported. T0 must enforce the stronger mission boundary.

## 2. Existing football functionality: reuse versus missing work

| Component | Actual implementation | Assessment |
|---|---|---|
| Offense/defense and opponent adjustment | `ratings.solve`: two score rows per game, league scoring intercept, offense and points-allowed defense deviations, conference effects, HFA; weighted ridge with connectivity shrinkage. `game_frame` filters completed games and maps membership by season. | Already exists. Its score target includes all realized points; the production ridge publisher exports margin only. Extending to score means needs correct centered prediction reconstruction and validation, not an assumed `mu+off+def` shortcut. |
| Later strength | `KalmanModel.update`: score-margin residual, opponent variance in denominator; fixed obs SD 14, week drift 0.8, season drift 6, prior SD10 and HFA 2.4. `KalmanTable.off=net/2`, `def_=-net/2`. | **Not learned offensive/defensive strengths.** `e_home + e_away = 0` under this table, including split HFA. Equal net matchups cannot distinguish two elite offenses from two elite defenses on these edges. Total varies through the empirical rate curve/pace/OT, not independent team scoring-level knowledge. |
| Pass/rush units | `ratings.play_units` and `solve_units`: eight opponent-adjusted offense/defense success/explosiveness outputs; explosiveness penalty 128 in parameters; tests/research objectives exist. | No call from production ratings/MC/publish paths was found. Do not rebuild the unit extractor. Audit sack-as-rush semantics and the unit fit's recency calculation: it carries week but omits season from its weight, unlike `_row_weights`. This is a reuse hazard, not a shipped unit-feature performance defect. |
| Pace | `PaceModel.fit/sample`: empirical paired regulation drive counts, minimum six drives per side, ≥50 games, joint pair resampling. | Pair dependence already preserved. No team-specific pace, matchup tempo or game-state drive-count model. Longer historical pooled samples may mix eras. |
| Drive scoring | `classify_drives`: offense/defense score deltas with joint classes `{0,3,6,7,8}×{0,2,6,7,8}`; `DriveRateModel`: quantile edge bins, ≥400/bin, interpolation, endpoint clamp. | Defensive scores, safeties and kicking outcomes enter aggregate scoring. Runtime ignores the classifier's dropped count; future artifacts must expose exclusions and reconcile totals. No explicit field-position transition chain. No Dirichlet rare-cell shrinkage beyond large pooled bins. |
| Overtime | `cfb_ot.fit_ot_params/simulate_ot`; era-specific historical fit, ≥100 standard possessions and ≥5 mandatory two-point attempts; regulation ties trigger OT. | Reuse; do not invent OT rates to fill unfittable historical folds. Its floors caused most corrected early years to be omitted. This audit verifies code behavior, not current NCAA rules against an external rulebook. |
| Turnovers and returns | Realized point targets include their consequences; drive defense-score classes include observed scoring returns. `FiveFactorsWalkForward` can display turnover-luck/field-position/special-teams features. Raw turnover-margin factor is excluded. | No separately learned team turnover/return adjustment in production. Rare events can still move latent ratings through final scores. Adding raw turnover margin, return TD rates or unshrunk explosive effects would count outcome noise again. No measured overreaction magnitude established here. |
| Special teams | Drive field-goal/TD/defensive-score outcomes; context field-position/special-teams diagnostics; factor registry display-only special-teams difference. | Aggregate contribution exists. A punt/kick/return/field-position transition or kicker skill model does not. Check score-delta accounting before claiming all return points are captured. |
| Home field | Ridge fits league and shrunk team HFA; neutral zero. Kalman uses constant 2.4 and no per-team HFA. | Registry `per_team_time_varying_hfa=gated_active` does **not** establish later-route use. Conference status likewise does not remove the existing ridge conference coefficients. |
| Uncertainty | Ridge coefficient variance/connectivity/misfit bands separate from pooled outcome SD. Kalman diagonal posterior variance → square-root sum → MC margin perturbation. Contract checks finite SD, p and approximate Normal coherence. | Useful existing interfaces; not a full team-score uncertainty contract. Posterior covariance between opponents is omitted by the registered diagonal approximation. |

The comment in `cfb_sim` describing clean-play latent strength is broader than runtime: the main ridge and Kalman fits use game scores, **not clean-play rates**. Clean-play garbage-time filtering applies to unit/feature extraction. Simulation deliberately keeps regulation garbage-time drives to represent final scores. Don't “fix” this by removing garbage-time scoring from final-score targets.

## 3. Correlation, duplication and popularity investigations

**Common opponents:** the game graph/score solve and Kalman updates already transmit opponent information. No separate runtime common-opponent bonus was found. Two score rows per game are two legitimate responses but are not two independent games for evaluation. Conference/team coefficients are jointly regularized within ridge, not separately estimated additive votes.

**Repeated prior evidence:** previous-season results inform `previous_season_net`, while the main ridge solve also retains older games with exponential decay. Talent, returning production and prior rating are averaged as standardized prior components with weights1/1/1, not independence-verified signals. This is deliberate shrinkage with overlapping information; it warrants an ablation, not an unsupported claim that it necessarily worsens accuracy. The challenger's returning feature would also overlap the incumbent prior. Rating-derived drive labels and same historical score outcomes form a calibration stage, not independent corroboration. Require conditional/add-one-block value, not several marginal correlations counted as proof.

**Rare-event overreaction:** no live raw-turnover or explosive add-on was found; the old unit implementation already applies strong explosive shrinkage. However all final-score-based ratings can respond to nonpersistent return/turnover outcomes, and drive cells use empirical counts without rare-cell smoothing. Establish sensitivity and paired loss before intervention. Use opportunity denominators, as-of n, pooled priors, and train-only shrunk persistent components. Do not add an independent correction for scoring events already accounted for in drives/field position. Current evidence does not identify a particular turnover/return adjustment as an improvement.

**Popularity:** no AP/CFP ranking, article count, news sentiment or generated-news feature reaches the inspected forecast equation. Team names are dictionary/join identifiers, not embeddings. Recruiting talent and conference fallback can encode measurement/coverage bias; missing talent is correctly treated as unmeasured (service-academy zero sentinel), not as poor quality. Coaching/availability coverage is uneven and retrospective. The optional context panel and generated rhetoric do not currently change predictions. A future news feed must use the structured claim contract, enforce unknown coverage, deduplicate syndication and test team-ID/display-name invariance. Absence of a news feature is not proof that all existing priors are bias-free.

## 4. Continuous margin uncertainty: reproduced, not assumed

`cfb_sim.simulate_game` first creates paired scores and OT-adjusted `(M,T)`, then adds antithetic `epsilon ~ Normal(0,param_sd)` **only to M**. A separate RNG preserves base simulations; pairing keeps mean M unchanged. It returns the shifted margin and unchanged total arrays.

Algebraically `H'=(T+M+epsilon)/2 = H+epsilon/2` and `A'=(T−M−epsilon)/2 = A−epsilon/2`. Holding the total fixed while transferring uncertainty between teams is a coherent **continuous** contrast-error construction. For independent epsilon it adds variance to each team and negative covariance between them, preserving total variance. It is not automatically an incorrect margin approximation. Antithetic shifts have zero sample mean; finite-sample covariance with base margins need not be exactly zero, so SD growth is not guaranteed to equal quadrature exactly.

But it is not a supported joint distribution over literal football final scores. [reproduce_audit.py](reproduce_audit.py) uses the existing simulator, six drives per side, valid deterministic 42–0 scoring, n 4000, parameter SD 3.6 and fixed RNG seeds. Results: mean-margin delta 0; total arrays identical; **4000 fractional-score paths and 2000 negative-score paths**, minimum implied score −7.17055. These are synthetic support violations, **not a claim that half of real games produce negative scores**. The no-shift fixture is integer/nonnegative.

Exposure matters: v2 publishes summary margin/SD/p/mean-total; the explainer shows a continuous margin histogram and deciles. Neither currently publishes a joint home/away score-sample distribution. `p_ot` refers to regulation ties before the shift. Future callers cannot reinterpret returned arrays as literal paired scores. `PredictiveMargin.from_summary` checks p against a Normal approximation within 0.05; it does not validate joint support, total uncertainty or sample identity. Its docstring's “no probability constructor” claim is outdated: `from_summary` accepts p explicitly. No fix was made; T2 requires a separately evaluated representation before any score-distribution exposure.

## 5. Separate baselines (descriptive development evidence)

The following are recomputed from existing CSVs, not newly fitted forecasts. Each row's source hash/status is in `audit_evidence.json`; source file hashes match the manifest where one exists. P4 follows season-specific game metadata and the contract's four-conference definition. All losses are against outcomes; no market data were loaded by the audit probe.

| Source / scope | n | Margin MAE / RMSE / bias | Win Brier / log loss | Margin Gaussian CRPS | 80% / 95% coverage |
|---|---:|---|---|---:|---|
| Early selected record, all FBS | 724 | 13.5665 / 16.9124 / −0.9350 | .1683 / .5025 | 9.5462 | .7997 / .9530 |
| Early selected record, P4 involved | 495 | 13.1208 / 16.5408 / −1.2150 | .1538 / .4643 | 9.3073 | .8061 / .9576 |
| Corrected later **unselected candidate**, all FBS | 429 | 13.1160 / 16.5936 / −0.7549 | .1893 / .5601 | 9.3513 | .7995 / .9417 |
| Corrected later **unselected candidate**, P4 involved | 186 | 13.1984 / 16.7777 / −2.2844 | .1840 / .5455 | 9.4618 | .7957 / .9409 |
| Cross-division, separate basis only | 599 | 15.0161 / 18.4729 / −4.6397 | .0529 / .2001 | 10.5091 | .7780 / .9416 |

The later corrected record covers only 2023 weeks 7–15. It is **not** an incumbent final-score benchmark. The early and cross-division SDs were estimated from the same pooled development residuals on which coverage is summarized; their standardized residual SD≈1 is largely mechanical. Probabilities are home-oriented, not favourite-oriented. Fixed reliability bins with n and Wilson intervals are in the JSON; aggregate Brier/accuracy alone does not establish calibration. For example early P4 mean p(home)=.6556 versus observed .6606, later corrected P4 .5531 versus .5860, and cross-division .8926 versus .9416; uncertainty and subgroup composition still matter.

| Score source / scope | Home MAE / RMSE / bias | Away MAE / RMSE / bias | Mean team MAE | Total MAE / RMSE / bias |
|---|---|---|---:|---|
| Early incumbent, any scope | unavailable | unavailable | unavailable | unavailable |
| Later **certified incumbent** | unavailable: no selected comparable record | unavailable | unavailable | unavailable benchmark |
| Corrected later **candidate diagnostic**, n429 | 10.1528 / 12.7097 / −1.0357 | 9.1756 / 11.3452 / −0.2808 | 9.6642 | 13.9390 / 17.4685 / −1.3165 |
| Same candidate, P4 n186 | 9.6820 / 12.0246 / −1.4695 | 9.6874 / 11.7285 / +0.8149 | 9.6847 | 13.3824 / 16.8167 / −0.6546 |
| Cross-division incumbent | unavailable | unavailable | unavailable | unavailable |

Candidate diagnostic score means are `(mc_total±mc_margin)/2` from that record itself; actual score means use the recorded actual margin/total. No total was grafted onto a margin-only route. Total Gaussian-summary CRPS/NLL/coverage are also in JSON. Joint energy score, team-score CRPS/PIT and exact-score probability are unavailable: persisted paired samples or an adequate joint law were not found. A margin SD and total SD do not identify their covariance. Do not reconstruct score SDs by assuming independence.

## 6. Availability and blockers

| Source / feature | Observed cache and runtime use | Blocker / next evidence |
|---|---|---|
| Games / season teams | 2019–2026 caches exist; used in main routes. Game row counts vary substantially with lower-division coverage. | Historical retrieval/availability is not proven by current file timestamps. Join membership, identity and outcomes explicitly; freeze corrected finals separately. |
| Talent / returning | Both have 2019–2026 caches and enter early prior through loaders; returning 2026 has 136 rows, talent 138. | Historical provenance hashes/as-of records absent in `context_sources`; existing early evidence manifest hashes games/code, not all prior inputs. “Approved” does not certify pregame availability or independent effect. |
| Plays / units | Clean-play caches 2019–2026 (2026:29,252 rows); dormant unit extraction and factor research exist. | Coverage per target team/game, revisability, sack semantics, garbage-time definitions, stable IDs and season-aware decay need verification. No formation/tracking claim. |
| Drives / pace / scoring | 2019–2026 (2026:4,852 rows); runtime rate labels restrict FBS-v-FBS, but **pace and OT receive all cached drives**, not the same division filter. | Changing lower-division cache coverage can change pooled pace/OT. Audit membership and score reconciliation before an all-FBS training comparison. |
| Coaches | 2019–2025 cached; no 2026 file. Local challenger rejects ambiguous multiple coach rows. | Historical preseason as-of proof absent; calendar-year January 1 test in challenger provenance is overly restrictive for actual preseason facts and requires replacement by game/season cutoffs. |
| QB starts / availability | Retrospective `qb_starts` caches in 2021–2024; no approved pregame QB/unit feed. | In-game attempt leader is unavailable pregame. No injury adjustment is justified. |
| Venues / travel / portal / weather | Registry gates not requested for venue/portal and unnamed external weather/availability providers; no matching venue/portal caches found. Board may display cached team geometry, which does not open the venue gate. | No new source activated. Weather must carry forecast issue time; observed weather is not a forecast. |
| Context five factors | Provider expects per-team derived `points_per_trip/start_field_position/special_teams_epa/turnover_luck` frame and earlier current-season weeks. | Raw drives/plays are not that interface. Main weekly cards do not call this provider. Registry entries are not a runtime wire. |
| Baseline evidence | Early selected n 724; corrected later candidate n 429; cross-division CSV/basis exists but has no evidence-manifest role. | `forecast_sd_basis.sim_kalman_wk4_plus` remains usable and cites archival pooled-pace/OT evidence; `BasisRegistry` does not consult `evidence.Manifest`. It permits weeks 4–20 although cited measured weeks are4–15. Resolve before any new claim. |
| Replay parity | Weekly labels past drives via **production policy** (ridge early, Kalman later). `cfb_sim.walk_forward_backtest` labels via `solve_table(...,engine=engine)` in every week. | Corrected Kalman backtest does not exactly reproduce the production training-label route. A new parity instrument is required; the 429 rows cannot simply be relabeled incumbent. |
| Provenance identity | Snapshot contents have SHA256; input source digest uses sizes/mtime and includes lines; config hash covers config.json/rating_params but runtime can read config.local overrides. | Content/as-of/version manifest must cover actual numerical inputs without recording secrets. A git SHA alone does not identify this dirty worktree. |

`early_challenger` uses outer season-forward splits but inner leave-one-season/week-out splits that train on later inner periods. Its fitted residual SD is in-sample; incumbent SD uses earlier outer-training residuals. These are not the nested chronological uncertainty protocol required here. Its `decide` also has a cover-log-loss guardrail, an obsolete objective for this program. The existing report retains all three tested arms and starts no shadow log; this audit does not reverse that decision. `performance_mc` provides useful proper scores and paired block resampling but its `evaluate_oof` contract expects market columns and its simulations are diagnostics, not extra validation data. Reuse primitives, not its full market-oriented evaluation wrapper.

Other operational blockers recorded by prior work include the LSU–Clemson kickoff reconciliation mismatch and launchd permission failures. They were not re-executed in this audit. Keep them separate from statistical uncertainty and do not rewrite frozen records to make a grade pass.

## 7. Verification and handoff

Local runtime: repository `.venv`, Python 3.11.15, NumPy 2.4.6, pandas 3.0.5, pytest 9.1.1, pyarrow 25.0.1, SciPy 1.17.1. System Python 3.14.7 lacks NumPy; use the repo venv. No dependencies were installed. The old vault statement that this local environment cannot run pytest is no longer true for this Mac.

Executed focused suite: **273 passed in 2.86s**. It includes forecast contract, CFB sim, parameter uncertainty, Kalman, leakage, context contracts/leakage, explanation context hook, early challenger, performance MC, evidence/leakage, insight cards and render-picks tests. These are working-tree tests and include untracked research files; they do not certify those files as shipped. The exact invocation is in `verification.json`.

Read-only probe: `.venv/bin/python docs/final_score_program/reproduce_audit.py` emitted the canonical baseline, cache/hash and support evidence. It reads no market file and performs no candidate fitting. Existing tests validate preservation of margin means/total under parameter sampling; they do not test joint literal-score support. The new probe records that distinction rather than changing the engine to satisfy a new test.

GitHub CI on audit commit `3c54d9d` is **red: 4 failed, 770 passed, 14 skipped**. The failures are tests requiring missing gitignored CFBD game/team parquet caches: cross-division real-week reconstruction, selected-evidence input verification, factor-context frame evaluation, and the no-basis producer test. All four failures were reproduced on a clean `git archive c6c780f` using the local venv (4 failed in 0.96s), establishing that they predate this six-file audit. The cache-present focused suite and release-integrity probes test different conditions; their passes do not imply a green full CI suite. Record/fix the clean-checkout fixture or data-provisioning policy as T0 work; do not weaken production evidence checks or bundle private caches to make CI green. Details and the run URL are in `verification.json`.

Release-check, preservation, commit and PR results are recorded in `verification.json` and the final handoff. This audit's scope is documentation plus read-only evidence tools/records. No production model, data, frozen forecast, registry, website, scheduler or dependency changed. Vault updates are derived pointers and an append-only worklog entry.

**Next session for Opus:** T0/T1 only: isolate the market-free evidence inputs, reproduce routing and basis inconsistencies, account for missingness and benchmark coverage, then produce a machine-readable baseline instrument. Do not start a replacement engine, re-run the old challenger to find a winner, activate shadow collection or claim improved final scores. A verified “unavailable / no improvement” result completes the task honestly.
