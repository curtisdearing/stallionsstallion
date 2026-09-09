# 00 — Contract: the team-season evidence layer

**Written** 2026-09-09 · **Status** subordinate to
[`../final_score_program/00_contract.md`](../final_score_program/00_contract.md)
· **Supersedes** nothing.

## Relationship to the final-score program contract — read this first

This document governs **one layer**: a display-only team-season evidence surface.
It does not govern the final-score program. That program's contract is
`docs/final_score_program/00_contract.md` v1, written by the `codex-final-score`
session on 2026-09-08 and committed on branch
`audit/final-score-program-2026-09-08` at `99fe693`. Where the two disagree, that
one wins.

**Sequencing note, stated plainly.** That contract's §5 defers "structured QB/unit
availability, weather, coaching, talent/returning revisions, finishing drives,
field position, returns and turnovers … until their source availability and
conditional value are demonstrated", and adds: *"Source collection alone does not
activate a factor."* Several measures in this layer sit inside that deferral list.
They are shipped here as **collection, not activation**:

* nothing in this layer can alter a projected margin, total or win probability;
  `test_no_calculation_module_names_a_forecast_field` and
  `test_the_evidence_layer_is_not_a_forecast_module` assert it structurally;
* every measure is `display_only` or blocked, and promotion runs through
  `data/factor_registry.json` plus a written pre-registration — a separate human
  decision, unchanged by this branch;
* the availability question that deferral turns on is **answered, negatively, and
  enforced**: §3 below blocks every retrospective predictive claim built on these
  measures, because their pregame availability is a reconstruction and not a proof.

So this layer supplies the audited source-availability evidence that deferral asks
for, and it withholds exactly the thing the deferral withholds. If the program
owner reads it otherwise, the correct resolution is to mark the affected registry
entries `blocked` — a data change, not a code change.

Two of the final-score contract's later tasks consume this surface directly: **C2**
(possession rates) and **C3** (opponent-adjusted pass/rush success-rate matchup
terms from `play_units`/`solve_units`). The measures behind both are built and
tested here, unadjusted-and-labelled or adjusted by the existing solver.

---

Below is the contract this evidence layer must satisfy. It is written to be checkable:
every clause below is either enforced by a constructor that refuses to build, or by
a named test in `tests/test_team_evidence_*.py`.

Scope: **evidence only**. This layer produces no projected margin, no total, no win
probability, and no point adjustment. It is not a forecast module and it is not a
rating engine.

---

## 1. Non-negotiables inherited from the repo

1. **Do not edit hash-pinned files.** `config.json`, `nflvalue/engine.py`,
   `nflvalue/ratings.py`, `nflvalue/kalman_engine.py`, `nflvalue/cfb_ot.py`,
   `nflvalue/cfb_sim.py`, `scripts/backtest_early.py`, `scripts/backtest_c.py`.
   They are sha256-pinned by `data/evidence_manifest.json` and back the repo's only
   `selected` benchmark. The layer lives in a new package that *imports* them and
   writes nothing back — the pattern already established by
   `nflvalue/cross_division.py` and `nflvalue/early_strength_candidate.py`.
2. **Do not build a second rating engine.** Opponent adjustment comes from
   `ratings.solve_units`, which already performs it. Anything `solve_units` does not
   cover ships opponent-**un**adjusted and says so in the record.
3. **An unapproved source cannot emit a value.** Gating goes through
   `nflvalue.context.sources.SourceManifest` and `data/context_sources.json`. There
   is no code-only path to a live source.
4. **Missing is a value, never an imputation.** No default-fill, no mean-fill, no
   zero-fill. A number that is not there is represented as an explicit
   unavailability state carrying a reason.
5. **Display-only.** Nothing here may alter a projected margin, total or win
   probability. Promotion of any measure into the model runs through
   `data/factor_registry.json` plus a written pre-registration, and is a separate,
   human decision.

---

## 2. The observation

`nflvalue/team_evidence/contracts.py :: Observation` is the atomic unit. It is a
frozen dataclass with exactly two constructors, mirroring
`context.contracts.ContextValue`:

* `Observation.present(...)` — refuses to construct unless **every** provenance
  field is supplied and non-`None`, and unless `value is not None`.
* `Observation.missing(...)` — cannot carry a value (hardcoded `value=None`),
  refuses `EvidenceAvailability.OK`, and requires a `reason`.

There is no third path, so an unprovenanced or silently-defaulted number cannot
exist in this layer.

### Required provenance envelope

| Field | Meaning | Enforced |
|---|---|---|
| `source_id` | key into `data/context_sources.json` | must be declared; gate consulted before emission |
| `source_url_or_id` | the citation | non-empty |
| `source_version` | sha256 (or explicit `size:mtime_ns` fingerprint) of the artifact the value was computed from | non-empty |
| `retrieved_at` | when *we* obtained the artifact, UTC, tz-aware | non-`None` |
| `effective_through` | the last kickoff whose result is inside this value | non-`None` |
| `availability` | `EvidenceAvailability` | `OK` only via `.present` |
| `availability_class` | see §3 | non-`None` |
| `missingness` | `PRESENT` on `.present`; a specific reason otherwise | non-`None` |
| `n` / `n_games` | the denominator, and how many team-games it spans | non-negative ints |
| `opponent_adjusted` | bool | must be paired with `opponent_adjustment_method` |

`EvidenceAvailability` is a superset of `context.contracts.Availability` — every
`Availability` value is representable by the same string (asserted by
`test_every_context_availability_value_is_representable`) — plus two states the
context layer has no need for:

* `INSUFFICIENT_SAMPLE` — the measure exists but `n` is below its registered gate.
* `BLOCKED_UNVERIFIED_AVAILABILITY` — see §3.

### Naming discipline (the "do not call it that" rule)

Enforced by `data/team_evidence_registry.json` and
`test_registry_names_do_not_overclaim`:

* A **sack rate is not a pressure rate.** The measure is
  `sack_rate_per_dropback`, its registry entry records that QB scrambles are typed
  `Rush` upstream and cannot be separated (`playText` is dropped at ingest), and the
  word `pressure` may not appear in any measure id or label.
* **Team W/L is not an offensive-line ranking.** No measure id may combine a record
  with a unit label.
* **Capacity is not crowd noise.** Venue capacity is not a crowd measure and is not
  emitted as one.
* **A large gain is not a broken assignment.** The measure is
  `explosive_play_rate_*` with a stated yardage threshold; no measure id or label
  may contain `broken`, `blown`, `bust` or `assignment`.
* A **proxy** must say so in its id or its registry `caveat`. `rush_stuff_rate_proxy`
  carries the proxy suffix because the canonical `stuffRate` lives on
  `/stats/game/advanced`, which this repo does not fetch.

---

## 3. Availability classes, and the retrospective block

Every observation carries an `availability_class`:

| Class | Meaning |
|---|---|
| `proven_pregame` | a git-tracked artifact records that this exact value existed before the relevant kickoff (`data/snapshots/` freeze, `board_ledger.csv`) |
| `reconstructed_week_granular` | derived from a cache with no observation timestamp; availability is the assumption "everything from week ≤ W−1 was known by week W" |
| `unverifiable` | availability cannot be established at all |

**Finding (`01_architecture_audit.md` §2.2): no cached CFBD record carries a
retrieval time, an effective time, a version or a hash.** Therefore every measure
this layer computes from `data/cfbd/*.parquet` is
`reconstructed_week_granular`, not proven.

**The block.** `EvidenceAvailability.BLOCKED_UNVERIFIED_AVAILABILITY` and the guard
`contracts.assert_retrospective_claim_allowed(observation, claim_kind)` implement
the brief's rule directly:

> a retrospective **predictive** claim — "this measure would have predicted X" —
> may be made only from observations whose `availability_class` is
> `proven_pregame`.

`reconstructed_week_granular` observations remain fully usable for **descriptive**
work (profiles, notes, exploration, and for building features whose evaluation is
itself pre-registered and week-granular). They are refused for predictive claims.
The guard raises `RetrospectiveClaimBlocked`; there is no flag to override it.

Rationale, stated once so it is not re-argued: week granularity cannot separate a
Thursday game from the following Saturday game inside the same week. The repo
already says so at `cfb_sim.py:347`.

**Forward fix, recommended not implemented.** Setting
`df.attrs["observed_at"]` in `ingest._atomic_write_parquet` would make future pulls
`proven_pregame` and would activate the already-tested `context.contracts.
kickoff_guard`, whose `assemble.Env.observed_at` reads exactly that key. It costs
nothing. It is **not** done here because `ingest.py` participates in the input
hashes of the selected benchmark's declared inputs and the change belongs in its own
reviewed commit.

---

## 4. Temporal rules

1. **Strictly earlier games only.** Every rolling summary and every opponent
   adjustment filters with the repo's standard predicate:
   `(season < S) | ((season == S) & (week < W))`. Never `<=`. The predicate is
   re-implemented as `team_evidence.profile._strictly_before` **as a deliberate copy**
   — `ratings.py` and `cfb_sim.py` are hash-pinned and cannot host a shared helper.
   `test_the_local_predicate_matches_cfb_sim` asserts the copy agrees with
   `cfb_sim._strictly_before` on a shared fixture.
2. **A future game cannot move an earlier feature.** Proven by poisoning: append and
   corrupt later weeks, assert the earlier profile is byte-identical.
3. **A later news item cannot move an earlier feature.** Same technique, on the
   personnel-fact path.
4. **Neutral venues** are read from `games.neutral_site` and carried on every
   game-level record. A neutral game is never counted as a home game for either team.
5. **Team-season identity** is `(season, team)` with the CFBD `school` string
   verbatim, matching `ratings.game_frame`. A team's 2025 rows never merge into its
   2026 profile; `test_team_season_identity_does_not_bleed_across_seasons`.
6. **Roster changes** are represented as *events with provenance*
   (`news.PersonnelFact`), never as a silent recomputation of past games. A
   personnel change dated after a game may not alter that game's measures.

---

## 5. Obsidian layout (registered here because none exists)

`01_architecture_audit.md` §5 establishes that the vault has no per-team and no
per-game notes, no stable-ID frontmatter and no generated-section markers. This
contract registers the layout.

### Paths (stable-ID based)

```
teams/{season}/{team_slug}.md
games/{season}/wk{week:02d}/{game_id}--{away_slug}-at-{home_slug}.md
```

* `team_slug` = `team_evidence.vault.team_slug(team)` — lowercase, non-alphanumerics
  collapsed to `-`. It is a **filename convenience, not an identifier**. The
  identifier is the frontmatter `team_id: "{season}:{CFBD school string}"`, which
  preserves the exact upstream string including apostrophes and accents
  (`Hawai'i`, `San José State`).
* `game_id` is the CFBD id, unchanged. It leads the filename so the note is
  addressable even if the schools are renamed.
* Frontmatter carries `team_id` / `game_id`, `season`, `week`, `generated_by`,
  `artifact_sha256`, `commit`. It follows the `research/` frontmatter shape
  (`type / project / tags / created`), because `picks/` has no frontmatter at all.
* Vault root resolution copies `render_picks.default_out_dir()`: the vault folder
  when the repo sits inside the vault, else `data/vault_preview/` in the repo. This
  layer never writes into `picks/`.

### Bounded generated sections

The vault's first marker convention. Exactly:

```
<!-- BEGIN GENERATED {block_id} v{n} :: do not edit inside -->
...generated body...
<!-- END GENERATED {block_id} v{n} -->
```

`vault.render_bounded(existing_text, block_id, version, body)` replaces **only** the
text between the markers. Everything outside is preserved byte-for-byte, including
a `## Notes` section a human wrote. If the markers are absent the block is appended
and existing content is untouched. If a start marker exists without its end marker
the function raises `MalformedBlockError` rather than guessing where to stop.

`test_manual_prose_survives_regeneration` writes a note, hand-edits prose outside
the block, regenerates, and asserts the prose is byte-identical.

### Immutable pregame, append-only postgame

* `vault.write_pregame_record(...)` writes the block
  `<!-- BEGIN PREGAME RECORD {game_id} :: frozen ... -->` **once**. If a pregame
  block already exists and differs, it raises `ImmutablePregameError`. Identical
  re-writes are a no-op, so the generator is safely idempotent.
* `vault.append_postgame_review(...)` appends under
  `## Postgame reviews (append-only)` and is forbidden from touching the pregame
  block; `test_postgame_append_never_rewrites_the_pregame_record`.
* Notes link to **artifact versions and commits**, not copied mutable claims: the
  frontmatter carries `artifact_sha256` and `commit`, and the body links the
  snapshot stamp rather than restating a number that can drift.

### Machine calculations never read generated prose

Two enforcements:

1. Structural — `measures.py` and `profile.py` may not import `vault`, and no module
   except `vault.py` may reference the vault directory
   (`test_no_calculation_module_reads_the_vault`, AST-based, both directions).
2. Functional — write a note, poison its generated prose with absurd numbers,
   rebuild the profile from canonical data, assert byte-identical
   (`test_poisoned_note_prose_cannot_change_a_profile`).

---

## 6. News and personnel facts

1. **Approved sources only.** Every `PersonnelFact` names a `source_id` that must be
   declared in `data/context_sources.json`; the manifest gate is consulted before
   any fact is emitted. Today `qb_availability_feed`, `unit_availability_feed` and
   `cfbd_portal` are all `not_requested`, so the entire personnel channel emits
   `UNAVAILABLE_NOT_APPROVED`. That is the correct current state, not a defect.
2. **Bounded structured facts.** A fact carries subject identity
   (`subject_type`, `subject_id`, `subject_name`, `team`), `claim` from a closed
   vocabulary, `status` from a closed vocabulary, `status_confidence`
   (`confirmed` / `reported` / `speculative`), `published_at`, `retrieved_at`,
   `source_id`, `source_url`, a `citation_quote`, and an explicit `uncertainty`
   note. Free prose beyond the quoted citation is not a fact.
3. **"No report found" is unknown, not healthy.** Absence maps to
   `UNAVAILABLE_NO_DATA` with reason `no_report_found_is_unknown_not_healthy`. It is
   never rendered as available, healthy, cleared, or full-strength.
   `test_absence_of_a_report_is_unknown_not_healthy`.
4. **Deduplicate syndication.** `news.deduplicate(facts)` collapses on
   `(subject_id, claim, status, published date, normalised quote fingerprint)`.
   Duplicates are attached to the canonical fact as `duplicate_of`; the canonical
   fact's `evidence_weight` stays **exactly 1** however many copies arrive.
   `test_duplicate_stories_do_not_multiply_effects`.
5. **Attention is not quality.** `news.FORBIDDEN_DERIVED_SIGNALS` bans article
   count, mention volume, media attention, AP/coaches poll rank and name
   familiarity from becoming measures. `test_attention_signals_are_refused`.
6. **No unrestricted LLM knowledge.** A fact with no `source_url`, or with a
   `source_id` naming a model rather than a manifest source, raises
   `UnsourcedFactError`. `test_model_recall_cannot_become_a_fact`.
7. **No numerical point adjustments in prose.** `news.assert_no_point_adjustment`
   rejects any fact text matching a point-valuation pattern
   (`worth 3 points`, `-2.5 pts`, `+4 points`). `test_prose_may_not_price_a_fact`.

---

## 7. Handoff obligations

`02_data_handoff.md` records, for Astra: the implemented interfaces, per-measure
coverage, the blocked fields with the exact reason each is blocked, the test list,
and how to run an experiment off this layer without tripping the retrospective
block.

Open question carried forward, not resolved here: whether `pred_b` in the factor
battery frames is the ridge or the kalman prediction. If a feature is scored against
ridge residuals while production weeks 4+ price with kalman, the "the rating already
used up schedule strength" argument holds only for the ridge's schedule handling.
See `01_architecture_audit.md` §4.

---

## 8. Out of scope for this program (explicitly)

No model promotion. No shadow-collection activation. No modification of frozen
picks or `data/snapshots/`. No publishing of private notes. No site deploy. No edit
to any hash-pinned file. No new external dependency.
