# 02 — Data handoff: the team-season evidence layer

**Written** 2026-09-09 · **For** Astra's experiments · **Branch**
`agent/team-season-evidence-2026-09-09`, based on `99fe693`
(`audit/final-score-program-2026-09-08`).

**Read first**, in this order:

1. `00_contract.md` in this directory — the final-score program contract, v1,
   written by the `codex-final-score` session. It governs the program. Nothing in
   this handoff overrides it.
2. `../team_evidence/00_contract.md` §3 (the retrospective block) and §1 (what may
   not be edited) — the contract for the layer described below.
3. `../team_evidence/01_architecture_audit.md` — a narrower audit of the evidence
   surface, supplementary to `01_architecture_audit.md` here.

**Sequencing, stated up front.** The program contract's §5 defers turnovers,
returns, explosive plays, availability and field position "until their source
availability and conditional value are demonstrated", and adds *"source collection
alone does not activate a factor."* Several measures below sit in that list. They
ship as **collection, not activation**: nothing here can move a margin, total or
probability; every measure is `display_only` or blocked; promotion still runs
through `data/factor_registry.json` plus a pre-registration; and the availability
question the deferral turns on is answered negatively and enforced by the block in
§6.1. If the program owner reads the deferral more strictly, the fix is to mark
those registry entries `blocked` — a data change, not a code change.

Program tasks **C2** (possession rates) and **C3** (opponent-adjusted pass/rush
success-rate matchup terms from `play_units`/`solve_units`) consume this surface
directly.

---

## 1. What landed

A new package, `nflvalue/team_evidence/`, plus two scripts and one registry. It
consumes the existing loaders and the existing opponent adjustment and writes
nothing back into either. No hash-pinned file was touched.

| Path | What it is |
|---|---|
| `nflvalue/team_evidence/contracts.py` | `Observation` (two constructors, one of which refuses to build without a full provenance envelope), `EvidenceAvailability`, `Missingness`, `AvailabilityClass`, `assert_retrospective_claim_allowed` |
| `nflvalue/team_evidence/registry.py` | loads `data/team_evidence_registry.json`; refuses overclaiming ids, blocked-without-a-reason, and adjusted-without-a-method |
| `nflvalue/team_evidence/sources.py` | approval gate (delegates to `context.sources.SourceManifest`) + sha256 artifact versioning + `derived_state` for values computed rather than read |
| `nflvalue/team_evidence/measures.py` | `team_game_measures(plays_clean)` → tidy long frame, one row per `(game_id, team, measure_id)` with `value`, `num`, `den` |
| `nflvalue/team_evidence/profile.py` | `build_profiles(season, week, ...)` → as-of-week `TeamSeasonProfile` per team; `common_opponents` (descriptive only) |
| `nflvalue/team_evidence/news.py` | `PersonnelFact`, `deduplicate`, `facts_before_kickoff`, `collect`, and the four refusals |
| `nflvalue/team_evidence/vault.py` | bounded generated sections, immutable pregame record, append-only postgame review |
| `scripts/build_team_evidence.py` | writes `data/team_evidence/{season}_wk{NN}.json` |
| `scripts/render_team_notes.py` | writes `teams/{season}/{slug}.md` and, with `--games`, `games/{season}/wk{NN}/{game_id}--away-at-home.md` |
| `data/team_evidence_registry.json` | 63 measures: 52 emittable, 4 gated on an undeclared source, 7 blocked |

`data/context_sources.json` changed in exactly one way: seven existing entries
gained `"team_season_evidence"` in their `serves` list. No approval status was
altered and no source was added — see §5.

---

## 2. How to run it

The repo's `.venv/` is a **macOS** venv (`.venv/bin/python3.11 →
/opt/homebrew/...`), so `.venv/bin/python` works in Curtis's Terminal and not
inside the device VM. Both are recorded here so neither surprises you.

```bash
# on the Mac
.venv/bin/python scripts/build_team_evidence.py --season 2026 --week 2 --history 1
.venv/bin/python scripts/render_team_notes.py  --season 2026 --week 2 --games

# inside the device VM (no python3.11 there)
~/venvs/ss/bin/python scripts/build_team_evidence.py --season 2026 --week 2 --history 1
```

`--week W` means **strictly earlier than W**. `--history N` is how many prior
seasons feed `ratings.solve_units`; `1` is enough to identify the graph and runs in
about 7 s for 2026 and 16 s for a mid-2025 week.

To see the layer at full sample rather than at week 2 of a season that has played
one week:

```bash
.venv/bin/python scripts/build_team_evidence.py --season 2025 --week 8 --history 1
```

### The artifact

`data/team_evidence/{season}_wk{NN}.json`, atomic write, schema
`stallionsstallion.team_evidence.v1`. Header carries `generated_at_utc`, `commit`,
`registry_path`, `source_states` (per source: availability, version,
`retrieved_at`), `blocked_measures` (id → reason), `availability_note`, and
`artifact_sha256`. Each team carries `team_id`, `is_fbs`, `games_used`,
`opponents_faced`, `neutral_site_games` and the full observation list.

It is a build output and is **not** committed.

---

## 3. Coverage as actually measured

Two real runs, on the caches present 2026-09-09.

| Run | Teams | Observations present | Blocked measures |
|---|---|---|---|
| 2026 wk2 (one completed week) | 287 (138 FBS) | 6,301 / 18,081 | 11 |
| 2025 wk8 (seven completed weeks) | 311 (136 FBS) | 12,110 / 19,593 | 11 |

State breakdown, 2025 wk8: `ok` 12,110 · `unavailable_no_data` 2,987 ·
`insufficient_sample` 2,163 · `blocked_unverified_availability` 2,177 ·
`unavailable_not_approved` 1,244.

Alabama, 2025 as of week 8 (six games), a spot-check of real output:

```
rush_success_rate_off              0.3869   n=168
pass_success_rate_off              0.5419   n=179
rush_success_rate_def              0.3916   n=143
sack_rate_per_dropback_off         0.0625   n=192
rush_stuff_rate_proxy_def          0.1678   n=143
short_yardage_conversion_rate_off  0.5667   n=30
explosive_play_rate_rush_off       0.0655   n=168
takeaway_rate_per_opp_play         0.0301   n=266
giveaway_rate_per_play             0.0028   n=361
plays_per_game_off                60.1667   n=6 games
fg_made_rate                       0.6364   n=11
fg_distance_median                39.0      n=11
fg_distance_iqr                   16.5      n=11
fumble_retention_rate_off              —    n=1   insufficient_sample
```

**Expect week 2 of a season to be almost entirely `insufficient_sample`, and read
that as the layer working.** One game gives Alabama 26 rushes and 17 dropbacks
against a registered gate of 40. A 26-play rate is not a rate. Most gates clear in
weeks 3–4.

The `n` on a rate is the summed denominator across the games used, not the game
count — a rate is a ratio of sums, never a mean of per-game rates
(`test_a_rate_is_a_ratio_of_sums_not_a_mean_of_rates`).

---

## 4. Blocked fields, and exactly why

Seven measures are `blocked` and four are `gated_pending_source`. Each carries its
reason in the registry and in the artifact header.

| Measure | State | Why |
|---|---|---|
| `seconds_per_play`, `time_of_possession_share` | blocked | The play clock is not in the cache. `ingest.normalize_plays` drops CFBD's `clock`; `normalize_drives` drops `elapsed`/`startTime`/`endTime`. Both exist upstream. Re-pulling into the existing parquet paths would break the input hashes of the **selected** benchmark, so this needs its own decision, not a side effect. |
| `coverage_yards_allowed_per_return`, `return_yards_per_return` | blocked | `yards_gained` on special-teams rows is not the return distance — a `Punt` row reads 0 and a `Punt Return Touchdown` reads 22 on a play that plainly gained far more. Canonical `puntReturnYards`/`kickReturnYards` are on `/games/teams`, for which this repo has no loader and no cache. Return and coverage **event rates** (return TDs per punt / per kickoff, both directions) are computed, because attribution is verified. |
| `personnel_status_qb`, `personnel_status_unit`, `roster_changes_verified` | blocked | `qb_availability_feed`, `unit_availability_feed` and `cfbd_portal` are all `not_requested` in `data/context_sources.json`. An unapproved source cannot emit a value. |
| `drives_per_game`, `points_per_drive_off`, `points_per_drive_def`, `avg_start_yards_to_goal_off` | gated | `cfbd_drives` is not declared in `data/context_sources.json` at all — see §5. |

Two measures ship **with a stated caveat rather than blocked**, and the caveat is
attached to every observation:

* `sack_rate_per_dropback_{off,def}` — **a sack rate is not a pressure rate.** CFBD
  types QB scrambles as `Rush` and the play text that would separate them is
  dropped at ingest, so the denominator understates true dropbacks by the scramble
  count.
* `rush_stuff_rate_proxy_{off,def}` — a proxy, and the id says so. CFBD's canonical
  `stuffRate` is on `/stats/game/advanced`, which this repo does not fetch.

---

## 5. The one decision waiting on Curtis

**Declare `cfbd_drives` in `data/context_sources.json`.**

The drives cache is already a production input (`nflvalue/data_health.py` `INPUTS`,
`nflvalue/cfb_sim.py`), needs no new dependency, costs nothing, and requires no
extra CFBD call — the parquet already exists for 2019–2026. But that manifest is
the only place a source may be declared, and it does not mention drives. So this
layer refuses to read them.

Deliberately **not** done in this branch, for two reasons. The manifest's contract
is the context layer's, and `tests/test_context_contracts.py` requires every
declared source to have a matching `FieldSpec` and provider in `nflvalue/context/`
— so declaring it properly is a context-layer change, not a one-line edit. And an
approval record needs a named approver, which is Curtis, not this session.

The unblock, in order:

1. add a `cfbd_drives` entry to `data/context_sources.json` with
   `loader: "nflvalue.ingest.load_drives"` and an approval record;
2. add a matching `FieldSpec` in `nflvalue/context/contracts.py:FIELDS` and a
   provider in `providers.py`, so the context layer's own contract test holds;
3. change nothing here — the four measures light up on their own.

Until then those four render `unavailable_not_approved` with the reason and the
unblock instructions attached. `test_an_undeclared_source_can_only_back_blocked_measures`
asserts the block cannot be forgotten.

---

## 6. Rules your experiments must live inside

### 6.1 The retrospective block

```python
from nflvalue.team_evidence.contracts import assert_retrospective_claim_allowed
assert_retrospective_claim_allowed(obs, "descriptive")   # passes
assert_retrospective_claim_allowed(obs, "predictive")    # raises RetrospectiveClaimBlocked
```

Every observation from the caches is `reconstructed_week_granular`, because nothing
in `data/cfbd/*.parquet` records when a row became knowable
(`01_architecture_audit.md` §2.2). Descriptive work — profiles, notes, exploration,
and building features whose evaluation is itself pre-registered and week-granular —
is fine. A claim that a measure *would have predicted* something is not, and there
is no override flag.

To lift it properly, stamp `df.attrs["observed_at"]` in
`ingest._atomic_write_parquet` at write time. `context/assemble.py:54` already reads
exactly that key and `context/contracts.py:110 kickoff_guard` is already tested
against it. That change belongs in its own reviewed commit because `ingest.py`
participates in the selected benchmark's declared inputs.

### 6.2 Do not add a second strength-of-schedule term

Opponent strength is already inside the ratings three times over: the per-game
opponent column in the design matrix, the conference blocks, and connectivity. The
eight `*_adj` measures come from `ratings.solve_units` and carry that adjustment.
Everything else is `opponent_adjusted: false` and says so.

Score against the **walk-forward residual**, as `factor_battery` and
`early_strength_candidate` already do, and pass `model_margin` as a conditioning
feature. Then schedule information the rating already used has zero explanatory
power left and double-counting is structurally impossible. Score against the
outcome instead and it is near-certain and silent.

`common_opponents()` returns a listing and refuses to return a differential.
`test_common_opponents_is_descriptive_only` asserts it contains no number at all.

**Open question, not resolved here:** whether `pred_b` in the battery frames is the
ridge or the kalman prediction. If a feature is scored against ridge residuals while
production weeks 4+ price with kalman, the "already used up" argument covers only
the ridge's schedule handling. Worth settling before a design freeze.

### 6.3 Promotion path

Nothing here can move a margin, total or probability. To promote a measure:

1. add a `data/factor_registry.json` entry with `status: "display_only"` **before**
   any test — `factor_battery.registry_entry()` raises otherwise;
2. write a pre-registration alongside `data/factor_prereg_2026-08-28.json`: shape,
   definition, `pregame_knowable` justification, n-gate, matching, BH family and a
   frozen `m`, stopping rule;
3. evaluate against `margin - pred_b` with `factor_battery.evaluate_*`;
4. land as `display_only`. Promotion to `gated_active` is a separate human edit, and
   `tests/test_factor_isolation.py` asserts the active set is exactly
   `["league_hfa", "per_team_time_varying_hfa"]` — adding to it breaks that test by
   design.

### 6.4 News

The personnel channel is machinery without a feed today. When one is approved:
facts are bounded records with citations; `deduplicate()` collapses syndication and
the canonical fact keeps `evidence_weight == 1` however many wire copies arrive;
absence maps to `no_report_found_is_unknown_not_healthy`, never to healthy; article
count, poll rank and name familiarity are refused as signals; a fact with no URL or
a model-shaped `source_id` raises `UnsourcedFactError`; and prose that assigns a
point value raises `ProsePricingError`.

---

## 7. Obsidian output

Registered in `../team_evidence/00_contract.md` §5 because the vault had no per-team or per-game
notes, no stable-ID frontmatter and no generated-section markers.

```
teams/{season}/{team_slug}.md
games/{season}/wk{NN}/{game_id}--{away_slug}-at-{home_slug}.md
```

The slug is a filename convenience; the identifier is the frontmatter
`team_id: "{season}:{CFBD school string}"` / `game_id`, which preserves the exact
upstream string (`Hawai'i`, `San José State`). Notes carry `artifact_sha256` and
`commit` — versions and commits, not copied mutable claims.

Generated content lives between markers:

```
<!-- BEGIN GENERATED team-season-profile v1 :: do not edit inside -->
<!-- END GENERATED team-season-profile v1 -->
```

Everything outside survives byte-for-byte, including the `## Notes (hand-written —
never regenerated)` section. A start marker without its end raises rather than
guessing where someone's paragraph ends.

The pregame record is a separate block, copied from
`data/snapshots/{season}_wk{NN}/frozen.json` — the only artifact in this repo that
proves what was published before a kickoff. It is written once; an identical rewrite
is a no-op and a differing one raises `ImmutablePregameError`. Postgame reviews are
appended under `## Postgame reviews (append-only)` and the writer verifies after
the append that the pregame block is byte-identical.

**Run of record, 2026-09-09:** 138 team notes and 203 game notes written, **95 of
the game notes carrying a frozen pregame record**. Nothing was written into
`picks/` — those files belong to the existing generators and to Curtis. Nothing was
published.

---

## 8. Tests

`tests/test_team_evidence_{contracts,measures,leakage,news,vault}.py` — **104 tests,
all green** on the Mac's caches and in a bare CI checkout with no parquet at all.

Fixtures are local to each file. `tests/conftest.py` is a docstring-only stub
inherited from fablesfable and offers nothing usable.

The ones that matter to you:

| Claim | Test |
|---|---|
| a future game cannot move an earlier feature | `test_a_poisoned_future_week_cannot_move_an_earlier_profile`, `test_appending_future_games_cannot_move_an_earlier_profile` |
| …and the cutoff is not simply reading nothing | `test_the_cutoff_actually_bites_in_the_other_direction` |
| the local `<`-predicate has not drifted from the pinned one | `test_the_local_predicate_agrees_with_cfb_sim` |
| a later report cannot alter an earlier feature | `test_a_later_report_cannot_alter_an_earlier_feature` |
| duplicate stories do not multiply | `test_duplicate_stories_do_not_multiply_effects`, `test_syndication_survives_cosmetic_rewriting` |
| missing coverage stays unknown | `test_absence_of_a_report_is_unknown_not_healthy`, `test_missing_caches_fail_closed_rather_than_defaulting`, `test_no_observation_is_ever_silently_omitted` |
| sample counts are right | `test_sample_counts_are_the_summed_denominator_not_a_count_of_games`, `test_a_rate_is_a_ratio_of_sums_not_a_mean_of_rates`, `test_a_denominator_below_the_registered_gate_is_refused_not_rendered` |
| common opponents are not a second adjustment | `test_common_opponents_is_descriptive_only`, `test_no_profile_field_combines_common_opponent_results_into_a_number`, `test_exactly_one_opponent_adjustment_exists_and_it_is_the_existing_one` |
| no second rating engine exists | `test_this_package_defines_no_ridge_or_rating_solver` |
| machine calculations never read generated prose | `test_no_calculation_module_reads_the_vault` (AST, both directions), `test_poisoned_note_prose_cannot_change_a_profile` (functional) |
| the pregame record is immutable | `test_rewriting_a_pregame_record_raises`, `test_postgame_append_never_rewrites_the_pregame_record`, `test_the_pregame_record_echoes_and_invents_nothing` |
| names do not overclaim | `test_registry_names_do_not_overclaim`, `test_the_registry_refuses_an_overclaiming_measure` |
| team-season identity does not bleed | `test_team_season_identity_does_not_bleed_across_seasons` |
| neutral venues are carried | `test_neutral_site_games_are_carried_and_not_counted_as_home` |

### Suite state

Device, everything except the two release-gate files: **985 passed, 0 failed.**

`tests/test_clean_checkout_importability.py` and
`tests/test_legacy_entry_points_fail_closed.py` fail with
`GitError: git write-tree failed (128): Unable to create .git/index.lock` — the
mounted vault forbids `unlink`, so any git operation leaves a lock the next one
trips over. **Pre-existing**: the same 3 failures and 19 errors were present before
this branch, on an untouched tree. Sweep with
`mv .git/*.lock ../_to_delete/` before running them. They pass in CI, where the
filesystem is normal.

The cloud clone shows four additional failures
(`test_cross_division`, `test_evidence_and_leakage::test_select_never_returns_an_archival_record`,
`test_factor_context_leakage`, `test_forecast_contract`) — all of them need the
gitignored parquet caches, and all four fail identically on a pristine checkout of
`c6c780f`.

---

## 9. What this branch did not do

No model promotion. No shadow-collection activation. No modification of frozen picks
or `data/snapshots/`. No publishing. No site deploy. No edit to any hash-pinned file
(`config.json`, `engine.py`, `ratings.py`, `kalman_engine.py`, `cfb_ot.py`,
`cfb_sim.py`, `backtest_early.py`, `backtest_c.py` — all 17 declared hashes still
match). No new dependency. No CFBD calls: the call tally is unchanged at 73 for
2026-09.
