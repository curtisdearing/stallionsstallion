# Final-score program contract — v1, 2026-09-08 ET

**2026-09-09 status update:** the user authorized a bounded C1 implementation after this design freeze. [03_model_verdict.md](03_model_verdict.md), [c1_preregistration.json](c1_preregistration.json) and the immutable run now record that work. All six variants failed the inherited practical criterion; incumbent retained and no shadow activation. The original design below remains historical and is not rewritten to match results.

Status: audit and implementation specification only. **Incumbent retained; no challenger shadow log activated.** This contract authorizes no production switch, forecast rewrite, merge, deployment, new provider, or bet. Its machine-readable experiment decisions are in [protocol.json](protocol.json); descriptive measurements and source hashes are in [audit_evidence.json](audit_evidence.json). The [architecture audit](01_architecture_audit.md) records what actually runs. No model improvement has been demonstrated by this work.

## 1. Objective and scope

Predict the home and away teams' official final points, including overtime, using measurable football statistics and verified information available at the forecast cutoff. Optimize outcome error, not agreement with a sportsbook. This user instruction supersedes the market-blend/CLV objective in the historical `BUILD_SPEC.md`, `AGENT_BUILD_PROMPT.md`, and vault spread-improvement plan **for this program**. Those files and their historical experiments remain intact.

Primary evaluation population: FBS-versus-FBS games with at least one member of ACC, Big Ten, Big 12 or SEC **in that season**. Freeze the season/team membership lookup and its hashes. Historical Pac-12 games do not become P4 by retroactive team membership; any historical P5 sensitivity is a separately named descriptive slice. Notre Dame is not automatically P4; its opponent can qualify the game. Unknown membership is an explicit exclusion. Primary training uses P4-involved FBS-versus-FBS; broader all-FBS training is one separately declared variant per candidate. Never pool FBS-versus-FCS outcomes into the primary scorecard. Cross-division modeling is deferred.

One game is one paired observation, even though it has two scores, many drives and many simulation draws. Report early (CFBD weeks 0–3), later (4+), season, neutral/home, conference pairing, and missingness slices. Report cohort counts and refusals alongside every metric. Do not silently narrow to ranked teams, televised games, games with news coverage, games with lines, or games on which a candidate agrees with a line.

## 2. Information boundary and canonical records

The forecast builder receives a positive allowlist of numerical football features and verified structured pregame facts. **No opening, current or closing line, odds, market-derived rank, market consensus, residual-to-market target, CLV or cover result is accessible to construction, preprocessing, tuning, calibration, selection or promotion.** Renaming a market column does not make it eligible. Use provenance lineage and isolated inputs, not the existing challenger's substring denylist alone.

After a forecast is committed to an immutable artifact, a different read-only process may join the market by game ID for an explicitly labeled comparison. This process cannot write model parameters, eligibility decisions or forecast artifacts. Forecast generation must work identically with the entire lines directory absent. Perturbing lines must leave the model input digest and forecast values unchanged. Current snapshot digests include lines and file metadata; that operational rerun mechanism does **not** yet satisfy this stricter boundary.

Canonical ordering:

1. Immutable source evidence with content hash, origin and availability times.
2. Versioned feature rows with lineage to that evidence and recorded cutoff.
3. Frozen protocol, membership, folds, preprocessing, fitted parameters and candidate manifest.
4. Immutable pregame forecasts, followed by separately versioned official outcomes.
5. Machine-readable paired loss rows, exclusions, metrics and decisions.
6. Website, explainers, Obsidian and generated prose as derived views only.

Do not make an Obsidian sentence, rendered table or LLM interpretation the numerical source of truth. A narrative correction must never alter a prediction. A changed fact creates a new pre-kickoff version; the old version survives. After kickoff, an evidence correction is an append-only correction record, never a replacement forecast. Reuse `forecast_contract`, `snapshots`, `atomic`, `evidence` and `insight` interfaces where they apply; do not add a second production writer.

### Structured news contract (blocked until source evidence exists)

Each extracted claim needs `claim_id`, `source_id`, canonical source URL, document hash, cited passage/location, entity IDs, `game_id`, fact type, normalized value/status, `published_at_utc`, `retrieved_at_utc`, effective time, parser/model version, confidence, conflict state and verification state. Store evidence under applicable source permissions, with hashes and locators when redistribution is unavailable. No credentials in records or Markdown.

Availability vocabulary: confirmed, probable, doubtful, out, unknown; missing coverage means unknown, never healthy. Keep player status separate from estimated player effect. Resolve contradictory reports explicitly; do not average incompatible statuses. Duplicate or syndicated stories form one underlying claim, not extra evidence. Prefer official releases or independently verified reports; the count or tone of stories is never a strength feature. Rankings, brand, recruiting-news sentiment and team-name embeddings are excluded. A normalized talent composite already exists but its ascertainment and historical availability need auditing before reuse.

An LLM may extract cited facts and explain frozen results. Only a validated deterministic feature transformation with a registered effect can change a numerical forecast. The current `context` package remains display-only. Do not invert its import direction to let narratives call the model. Prospective pregame availability needs its own audited input module and approval record; `qb_starts` inferred from game attempts is prohibited as pregame evidence.

## 3. Baselines and estimands

Bias is forecast minus actual. Margin is home minus away; total is home plus away. Primary loss per game is `(abs(pred_home - home_points) + abs(pred_away - away_points)) / 2`. Report home and away MAE, RMSE and signed bias separately; also margin/total MAE, RMSE and bias. Preserve both team errors in paired loss rows and do not bootstrap them independently.

For win probability use Brier and log loss against `home_points > away_points`; report fixed-width 0.1 reliability bins, counts, Wilson intervals, calibration intercept/slope and uncertainty where estimable. Accuracy alone is not calibration. Probability clipping at 1e-9 is solely numerical for log loss and must be declared.

For distributions report marginal score, margin and total CRPS; 50/80/95% interval coverage **and width**; randomized PIT for discrete outcomes; and joint energy score on paired home/away samples. Exact-score log loss is allowed only with a normalized, validated joint probability mass model; a Gaussian density or a finite simulation's zero cell frequency is not an exact-score probability. Score means may be fractional; simulated final-score realizations must be physically admissible. Report the number of simulations and Monte Carlo error, never treat them as observed games.

Current supported comparisons:

| Route | Honest baseline |
|---|---|
| Early ridge | Existing margin/probability baseline; **team scores and total unavailable**. Pooled residual SD calibration is descriptive on reused development rows. |
| Later Kalman → MC | Runtime supports mean margin and total but no selected corrected benchmark exists. Existing corrected records provide labeled development diagnostics, **not certified incumbent final scores**. No persisted joint score samples. |
| Cross-division ridge | Separate margin-only reference; scores/total unavailable and outside primary scope. |

Do not construct incumbent scores by adding a market total, a league total, a guessed total or a future candidate total. Algebraically deriving score **means** from an existing MC record's own mean margin and mean total is legitimate only with that record's provenance/status, as done in the audit; it does not certify the record or recover a joint distribution.

For early-season capability experiments, create a clearly named **new research reference**: frozen incumbent margin plus training-only mean total, constrained so expected scores are nonnegative. This reference is not the incumbent, must be independently labeled, and cannot establish incumbent score improvement. Compare new score candidates against it and against a training-only direct home/away mean reference. Keep incumbent margin/probability paired comparisons separate. Until a certified score baseline exists, say “new score capability versus research reference,” not “improved incumbent final scores.”

## 4. Fixed chronological experiment protocol

All 2021–2025 seasons have already been inspected and remain development data. 2019–2020 are initialization/development, not an untouched control. No reshuffling, new file name or preregistration retroactively makes any of these seasons virgin validation.

Outer development folds: test 2021, 2022, 2023, 2024, 2025 in order. For season S, only earlier seasons enter initial fitting. Within S, predeclare weekly updates using completed games from weeks strictly before W and available before the forecast cutoff. Parameters governing adaptation are frozen by earlier data; late corrections cannot be retroactively treated as known. All arms use the same cutoff. Report state-frozen-through-prior-season versus weekly-adapted sensitivity separately, not as an extra promotion contender.

Inner folds are **expanding chronological origins**, never leave-one-season/week-out. Train through season s−1, validate season s, restricted to outer training seasons. Initial outer-2021 fit uses 2019→2020 inner validation. In each validation season, reproduce the same weekly update rule. If insufficient origins/data prevent a fit, record an unfit fold, never train on the future or silently substitute shuffled validation. The existing `early_challenger.choose_lambda` must not be copied unchanged.

Fit all scaling, imputation, feature selection, shrinkage, decay, prior weights and calibration inside the inner training data. Fit uncertainty from inner out-of-fold residuals or posterior predictive samples, not the fitted training residual SD. Use only verified as-of snapshots. Missing retrospective issue timestamps block prospective eligibility; file mtime is not historical availability. A synthetic cutoff cannot certify revised historical features.

Each candidate spec fixes its full feature equations, units, grids and missingness before it runs. Frozen default search: ridge penalty `{3, 30, 300, 3000}` after train-only scaling; current-season adaptation prior effective games `{4, 8, 16}`. No expansion based on outer-fold scores. Pick lower inner primary loss; within one inner standard error prefer fewer feature blocks, then stronger shrinkage. Never select the training-population variant on the outer fold. Declare failed/unfit arms in the original multiplicity family.

Paired comparisons join `(season, game_id, cutoff_id)`, verify identities/outcomes and carry both losses on exactly the same games. Publish the overlap and every no-call, plus an all-eligible policy analysis with preregistered research-reference fallback; no candidate may win by refusing difficult games. Whole `(season, week)` blocks are resampled together (20,000 draws, seed 20260908); season-block and leave-one-season-out summaries are dependence sensitivities. Five seasons are few clusters: do not sell a narrow week-bootstrap interval as five independent replications. A simultaneous bootstrap over the declared contender family accompanies development comparisons; historical intervals remain descriptive.

### Practical effect and decision rule

The prospective utility floor is fixed **before any new candidate result** at **0.50 points per team in primary MAE**, equivalent to one point less combined absolute score error per game. This is an explicit engineering utility choice, not a threshold estimated from the audit or a claim that half a point is universally valuable. Changing it requires a new version before prospective collection, not an exception after results.

Select at most one challenger using development results, complexity and availability. Development cannot promote it. It must have nonnegative primary improvement in at least four of five outer folds on the primary scope; missing folds are not passes. Failure, unavailable evidence or insufficient precision yields retain-incumbent/no activation. If all score comparisons are to new references, only new-capability status is available.

Before any future shadow activation, freeze a separate activation manifest with exact first eligible kickoff, candidate fit/hash, selected variant, paired reference, full fixed season window and precision calculation. Evaluate once after all regular-season games in the next **two complete seasons starting after activation** have settled; intervening partial-season collection is descriptive only. Do not peek and stop early, extend the window to rescue a failure, or silently replace a candidate. A train-block simulation must first show at least 80% power for a true 1.00-point/team improvement against the 0.50-point floor at one-sided alpha .025, using the candidate’s development paired-block dispersion and the frozen expected cohort; otherwise record “insufficient design,” not collect indefinitely. The 1.00-point alternative is a predeclared planning effect, not an observed gain. The expected eligible n and block count are recorded before activation.

Prospective acceptance requires the paired 95% lower bound on primary MAE improvement to exceed 0.50; it also requires the 95% upper bounds on increases in margin MAE, total MAE, Brier, log loss and available distribution proper scores to be ≤0. Bias, reliability, interval width and coverage are reported together to identify compensating failures. This deliberately strict no-harm rule can produce no improvement. When incumbent scores are unavailable, primary comparison is to the named research reference; incumbent margin/probability remain guardrails and any approved result is a **new capability**, not measured incumbent-score gain. No automatic deployment follows an acceptance.

## 5. Bounded implementation sequence for Opus

Start with missing/unsupported scoring level and its evidence chain, the weakest demonstrated capability. Do not start with popularity, news sentiment, a nonlinear spread stretch or a new engine. The audit does not rank unbuilt features by their imagined gains.

| Task | Concrete deliverable and acceptance | Owner / proposed paths |
|---|---|---|
| T0: reproducible evidence boundary | Inventory as-of sources, freeze P4 membership and schedule denominator; clean-checkout reproduction; separate line inputs/digests. Reproduce basis-manifest inconsistency, historical-renderer OT leakage, production/backtest label mismatch and score-support fixture before any fix. Document source exclusions. | Opus evidence owner: new `scripts/final_score_audit.py`, `tests/test_final_score_evidence.py`, new `data/final_score_program/<run_id>/` only. Existing publishing code remains protected. |
| T1: baseline instrument | Persist home/away/margin/total losses and calibrated probability/distribution diagnostics, reference status and unavailable fields. Read frozen forecasts; no recomputation masquerading as a forecast. Corrected later benchmark must reproduce production routing, including early-week training-drive labels. Resolve its coverage gap explicitly; do not select the 429-row record just because it exists. | Opus evaluation owner: new `nflvalue/final_score_evaluation.py`, tests and run artifacts. |
| C1: scoring-level candidate | Reuse ridge score equations and prior/table interfaces to estimate home+away scoring level independently of the frozen margin; verify the centered-intercept prediction formula first. Test a shrunk scoring-level model against both research references, in early and later routes. Record existing Kalman antisymmetric-edge limitation. Add no pace/style/news feature yet. | Opus modeling owner: isolated `nflvalue/final_score_candidates.py`, numerical tests; no edits to incumbent `ratings.py`/`engine.py`. |
| C2: possession candidate | One block of team offense/defense possession rates, shrunk to the training league rate with the registered effective-game grid. Reuse paired pace sampling and OT; validate paired counts and count units. Add only if C1 supplies an eligible reference; compare the incremental paired effect. | Same isolated candidate module; reuse `PaceModel` interface. |
| C3: unit-efficiency candidate | One predeclared block of opponent-adjusted pass/rush success-rate matchup terms from existing `play_units`/`solve_units`; account for sacks and fix/replace the unit solver's season-age omission inside the isolated adapter with a reproducer. Residualize/regularize jointly against incumbent strength and pace on inner training only. No explosive-play, turnover or return-rate feature in v1. | Same module, input lineage and correlated-feature ablation tests. |
| T2: score-distribution capability | Before exposing score samples, test a coherent joint scoring representation using existing drive/OT interfaces. A repair is a separately versioned candidate; preserve the old continuous margin approximation for its historical evidence. No rounding/clipping “fix” without mean, probability and proper-score comparisons. | Opus distribution owner: isolated adapter and support tests. Production `cfb_sim.py` stays unchanged until a later reviewed task. |
| T3: display/handoff | Render only committed machine records; every number has run/cutoff/source linkage and status. Create the next-session manifest and concrete decision, including no-improvement. | Opus view owner: new derived report, no current website/picks regeneration. |

C1–C3 each have exactly two training-population variants: primary P4-involved and broader all-FBS. Maximum six contenders; incremental order is fixed. Do not explore all combinations or add a fourth feature family after inspecting results. If an upstream stage fails, mark downstream stages not run; the family remains declared. Joint-support work T2 is a prerequisite for a score-distribution claim, not license for an extra performance search. Numerical distribution alternatives require their own preregistered extension.

Structured QB/unit availability, weather, coaching, talent/returning revisions, finishing drives, field position, returns and turnovers are **deferred** until their source availability and conditional value are demonstrated. Source collection alone does not activate a factor. Common opponents are already in opponent-adjusted ratings: do not add an unshrunk “A beat C, B lost to C” bonus.

## 6. Acceptance tests and protected surface

Before running performance candidates, require tests for:

- Future-week and same-week poisoning across scores, drives, plays, rosters and context; changing data unavailable at cutoff must not change earlier predictions or fitted preprocessing. Strict chronological assertions on every inner and outer fold.
- Market-directory removal/poisoning and renamed market-derived fields; forecast values and model-input digest unchanged. Rendering prose, rankings, team display names, news volume and duplicate stories cannot affect predictions. Team renaming must preserve stable IDs/features and results.
- Joint score support: paired samples finite/nonnegative/integer; margin and total match those samples; total ≥ absolute margin and parity is consistent; OT resolves regulation ties; empirical win probabilities and marginals come from the same paired samples. Fractional **means** are allowed. Validate equal sample lengths and weights.
- Paired games retained under missingness; home/away swap equivariance with venue handling; correct bias signs; metrics hand-checked on tiny fixtures; no duplicated games/drives; distribution scores computed on real games, not simulation rows.
- Train-only prior and rare-event shrinkage; one extreme turnover/return/explosive game cannot silently enter several independent corrections. Provenance groups identify shared games/features; remove-one-block ablations disclose conditional value.
- Snapshot first-write-wins, strictly pre-kickoff selection, hash verification and identity reconciliation. Frozen failures remain failures. No backfill into frozen slots.
- Evidence record agrees with route/version/window; unavailable total implies unavailable scores; a candidate/archival record cannot authorize a production claim. Clean index extraction imports without untracked dependencies.
- Generated explanations reproduce the frozen record and contributions; no narrative can rewrite a feature or a forecast. Acceptance of code is separate from statistical acceptance.

Untouched in this session and protected in implementation unless separately scoped: incumbent engine/rating/simulation/contract files; `data/rating_params.json`, `data/forecast_sd_basis.json`, existing evidence/factor/source registries; all forecasts, snapshot stores, ledgers, calibration/shadow artifacts, current site and picks; local configs/secrets; scheduling and publication scripts; unrelated dirty and untracked files. No new API calls, providers or dependencies are required for T0/T1. Resolve coordination against the existing dirty grading/challenger work before editing shared files.

## 7. Version and session handoff

Use `data/final_score_program/<UTC_run_id>/` for new immutable run directories, never overwrite a prior run. Required manifest fields: schema/protocol ID and hash, git commit and tree, dirty-source hashes if used (otherwise reject dirty runs), source content hashes and as-of proofs, membership hash, cutoff policy, full candidate family and selected variant, seeds, fold boundaries, environment/package versions, input counts, exclusions, fit versions, evaluation status and explicit decision. No private notes or raw licensed caches in a public commit.

Suggested artifacts: `protocol.json`, `source_manifest.json`, `cohort.parquet`, `features.parquet`, `fits.json`, `predictions.parquet`, `outcomes.parquet`, `paired_losses.parquet`, `metrics.json`, `decision.json`, and derived `report.md`. Parquet is globally ignored here; choose a permitted storage location or explicit scoped tracking policy before claiming a fresh clone can reproduce it. Version corrections as new files with `supersedes`, reason and hash; never mutate old forecasts.

At session start read this contract, the audit, protocol/evidence hashes, project instructions, git status and worklog tail. At session end append what/why/UTC/next to the vault worklog, reconcile only changed facts in hot/MOC, and list owned files, exact commands, results, limitations and the next bounded task. Small explicit commits on a feature branch. A docs-only PR should target the existing context branch to avoid presenting its unrelated ancestors as part of this program. No merge or deploy.

**Opus first task:** implement T0 and T1 as isolated read-only instruments and return an evidence manifest plus reproduced findings. Stop with an explicit blocker if pregame availability or later-route parity cannot be established. Do not build C1 until those instruments can distinguish unavailable incumbent scores, new references and valid paired comparisons. Preserve the incumbent and leave shadow activation off.
