# 01 — Architecture audit, evidence surface

**Written** 2026-09-09 · **Author** session `cowork-opus-team-evidence`
**Status** supplementary to
[`../final_score_program/01_architecture_audit.md`](../final_score_program/01_architecture_audit.md).

## Why a second audit exists

This session was told to read `docs/final_score_program/00_contract.md` and
`01_architecture_audit.md` first. At the moment it looked, neither existed: the tree
was checked out at `c6c780f` on `agent/context-layer-2026-08-28`, `docs/` was absent
entirely, and `_worklog.md` carried an open claim from `[s:codex-final-score]` at
`2026-09-09T00:07:27Z` with no matching `release`. Following the standing
instruction to reconcile when the handoffs are missing, this session audited the
code itself.

**Both documents did in fact exist, on a branch created while this session was
running** — `audit/final-score-program-2026-09-08`, commits `f9490f2`, `3c54d9d`,
`99fe693`, the last timestamped 2026-09-08 20:24 ET. The device tree was switched
onto that branch mid-session. This is exactly the concurrent-session hazard the
project's operational notes warn about, and the lesson is recorded here: **run
`git branch -a` at session start, not just `git status` and `git log`.**

That audit is the program's audit and nothing here supersedes it. Where the two
overlap they agree — the route map, the missing score support, the snapshot/freeze
protections and the hash pins are the same findings reached from two directions.
This file is narrower: it records what the **evidence surface** looks like and what
it can and cannot honestly measure. Every claim below was verified by reading the
named file, not inferred from documentation.

## 1. What already exists (reuse, do not rebuild)

### Loaders and caches

| Concern | Where | Notes |
|---|---|---|
| CFBD client | `nflvalue/sources/cfbd.py` | 12 endpoints, call-counting, 401/429 never retried |
| Normalise + cache | `nflvalue/ingest.py` | `data/cfbd/{kind}_{season}.parquet`, atomic write, `SCHEMAS` asserted on every load |
| Loaders | `ingest.load_games / load_plays_clean / load_drives / load_teams / load_talent / load_returning / load_lines / load_coaches` | deterministic sort, missing seasons silently skipped |
| Call budget | `ingest.CALL_BUDGET_MONTHLY = 1000`, `data/cfbd/_call_tally.json` (tracked) | 2026-09 tally at audit time: 73 used |

Cached coverage measured on-device 2026-09-09: `games/plays/plays_clean/drives/
lines/teams/talent/returning` for 2019–2026; `coaches` 2019–2025; `qb_starts`
2021–2024. `plays` stops at week 15 every season; `drives` reaches week 16. **No
postseason at all** — every fetch defaults `season_type="regular"` and
`weeks=range(1,16)`.

### Identity

* Teams are the **raw CFBD `school` string, verbatim**. There is no alias map, no
  slug, no `team_id` anywhere in `nflvalue/` or `scripts/`. Team-season identity is
  the implicit tuple `(season, team)`; realignment is handled by re-pulling
  `/teams?year=` each season (`ingest.py:19-21`).
* FBS/FCS from `teams.classification` (`ingest.py:242`). Every non-FBS team pools to
  the pseudo-team `ratings.FCS_TEAM = "__FCS__"`; FCS-vs-FCS games are dropped
  entirely (`ratings.game_frame`, `ratings.py:238`).
* Games are CFBD `/games.id` as `int64`, named `id` on the games frame and
  `game_id` everywhere else.
* Neutral site is `games.neutral_site`, a required schema column; the alias
  `neutral` is explicitly forbidden by the forecast contract
  (`forecast_contract.py:192`).

### Opponent adjustment — it exists, and it is not margin-only

`ratings.solve_units(games, units, season, week, params)` (`ratings.py:822`) is a
hierarchical weighted-ridge solve whose design matrix carries the opponent's column
on every row. That **is** the opponent adjustment. It already adjusts rush/pass
efficiency and explosiveness, and it enforces "earlier games only" with the repo's
standard predicate at `ratings.py:832`:

```python
hist = games[(games["season"] < season)
             | ((games["season"] == season) & (games["week"] < week))]
```

Its input is produced by `ratings.play_units(plays_clean)` (`ratings.py:786`) —
per `(game_id, offense, defense, phase)` rows carrying `n`, `eff` (success rate)
and `exp` (mean PPA on successful plays only).

Two facts that constrain this program:

1. **The public surface is welded shut.** `solve_units` hardcodes
   `for phase in ("rush","pass"): for metric in ("eff","exp")` and returns the fixed
   `UNIT_NAMES`. Lines 870–910 *are* a generic `(team, opponent, value, weight) →
   adjusted value` solver, but there is no entry point for an arbitrary metric.
2. **`ratings.py` cannot be edited.** See §2.

Consequence, recorded here so nobody re-litigates it: rush/pass efficiency and
explosiveness get the **existing** opponent adjustment by calling `solve_units`.
Every other measure in this program ships **opponent-unadjusted and labelled as
such**. Building a second adjustment for them would be building a second rating
engine, which the brief forbids and which §4 shows would double-count.

### Provenance and gating machinery already in place

| Mechanism | Where | What it guarantees |
|---|---|---|
| Source approval gate | `nflvalue/context/sources.py` + `data/context_sources.json` | a source not `approved` cannot emit a value, whatever code exists |
| Full provenance envelope | `context/contracts.py:195` `ContextValue.available` | refuses to construct without `source, source_url_or_id, observed_at, effective_for_game, confidence, coverage` |
| Missing is a value | `ContextValue.unavailable` (`contracts.py:225`) | cannot carry a number; requires a reason |
| Availability vocabulary | `context/contracts.py:38` `Availability` | ok / no_source / not_approved / no_data / stale / conflicting / low_coverage / refused_post_kickoff |
| Kickoff guard | `contracts.py:110` | refuses `observed_at >= kickoff` and `effective_for_game > kickoff` |
| Forbidden columns | `contracts.py:63` + `assert_column_allowed` | closing prices and every postgame outcome, by name |
| Factor registry | `data/factor_registry.json` + `context/registry.py` | a factor is `display_only` unless `gated_active`; five standing exclusions are a load-time tripwire |
| Pre-registration | `data/factor_prereg_2026-08-28.json` | hypotheses, BH `m`, and stopping rule frozen before any result |
| Evidence manifest | `nflvalue/evidence.py` + `data/evidence_manifest.json` | content / code / input sha256; an artifact with empty `code_hashes` can never be selected |
| Immutable snapshots | `nflvalue/snapshots.py`, `data/snapshots/` | `taken_at` + sha256; `freeze_kicked_off` copies the last strictly-pre-kickoff row |
| Public schema | `nflvalue/forecast_contract.py` | closed 22-column schema v2, coherence gate, registered-SD-basis gate |
| Input gate + halt | `nflvalue/data_health.py` + `scripts/weekly_loop.py:1156` | unhealthy inputs halt before anything is written |
| Release gate | `nflvalue/release_gate.py` | validates the candidate tree, not the dirty worktree |

### Missingness idioms already in use

There is no single shared sentinel. Four consistent idioms exist and this program
reuses them rather than inventing a fifth:

1. `Availability` enum — the richest; says *why* a value is absent.
2. Zero-as-missing unmasked at ingest — `talent == 0` → `pd.NA` +
   `unranked_recruiting` flag (`ingest.py:364`); downstream falls back to the
   conference mean, **never to a low value** (`ratings.py:350`).
3. Null-vs-status in the forecast contract — `status` carries the meaning; a refused
   row carrying any forecast number raises.
4. Unparseable-due-date is **UNKNOWN, never overdue and never fresh**
   (`context/registry.py:76`).

## 2. Hard constraints discovered

### 2.1 Hash-pinned files — DO NOT EDIT

`data/evidence_manifest.json` pins these by sha256. All 17 declared hashes were
re-derived on-device 2026-09-09 and **all currently match**, so the pin is live:

```
config.json  nflvalue/engine.py  nflvalue/ratings.py  nflvalue/kalman_engine.py
nflvalue/cfb_ot.py  nflvalue/cfb_sim.py
scripts/backtest_early.py  scripts/backtest_c.py
```

`backtest_early_20260830` is the **only `selected` benchmark in the repo**
(`benchmark_wk4plus` currently has none). Editing any pinned file — a comment, a
pure-addition helper — un-certifies it without changing one published number.

The repo already documents the sanctioned workaround, twice
(`nflvalue/cross_division.py:1-10`, `nflvalue/early_strength_candidate.py:1-6`): a
**new module that consumes the fitted table and writes nothing back**. This program
follows that pattern.

### 2.2 Historical availability cannot be established

This is the finding with the largest consequence, and it was checked directly
rather than inferred.

* **No cached raw record carries a retrieval time, an effective time, a version or
  a content hash.** No `SCHEMAS` entry declares one; `_atomic_write_parquet`
  (`ingest.py:178`) writes no metadata. The only `attrs` in the whole cache is
  `dropped_unmapped` on drives.
* The only "as-of" that exists is `scripts/weekly_loop.py:208 data_as_of()`, which
  is **file mtime**, collapsed to one stamp per kind. It moves whenever anyone
  touches the file, and `data/cfbd/*.parquet` is gitignored, so there is no
  version-control record of when any row arrived.
* `nflvalue/snapshots.py:191 _sources_part` says so explicitly: it fingerprints by
  `(size, mtime_ns)` rather than content hash, by design.
* The layer that *would* answer this — `context/contracts.py:110 kickoff_guard` —
  depends on `frame.attrs["observed_at"]` (`context/assemble.py:54`), and **no
  cached frame ever sets it**. `assemble_game_context` has no production caller;
  every caller is a test.

**Therefore:** for seasons 2019–2025, and for 2026 week 1 (pulled after the fact),
pregame availability of any derived stat is a **reconstruction at (season, week)
granularity**, not a proof. The repo's own comment names the limit
(`cfb_sim.py:347`): within a week — a Thursday game versus a Saturday game — week
granularity is not sufficient.

What *can* be proven, from 2026 week 1 forward: the forecast and the price taken,
via the git-tracked `data/snapshots/` freeze (`taken_at` + sha256 + kickoff-gated
`frozen.json`) and `board_ledger.csv`'s pre-kickoff-only `taken_at_utc`.

The brief's instruction applies directly: *"If historical availability cannot be
established, block retrospective predictive claims for those fields."* §3 of
`00_contract.md` turns that into a machine-checked rule.

### 2.3 Field-level data gaps in the cache

Verified against the parquet columns on-device:

| Wanted | Status | Reason |
|---|---|---|
| rush/pass success rate, denominators | available | `plays_clean.{down,distance,yards_gained,play_type}`; `ratings.play_units` already does it |
| explosive gains created/allowed | available | both sides free — every play row carries `offense` **and** `defense` |
| sack rate per dropback | available **with a stated caveat** | QB scrambles are typed `Rush` and cannot be separated: `playText` is dropped at ingest (`ingest.py:291`). The denominator is dropbacks-excluding-scrambles. **A sack rate is not a pressure rate** and is not labelled as one. |
| run stuffs | available **as a proxy** | `play_type=="Rush" & yards_gained <= 0`. Canonical `stuffRate` lives on `/stats/game/advanced`, which has no loader and no cache. |
| short-yardage conversion | available | `down in (3,4) & distance <= 2`, converted = `yards_gained >= distance` |
| special teams — kicking | available | FG good/missed/blocked, punt and kickoff counts; FG distance = `yards_to_goal + 17`. Attribution verified empirically (below). |
| special teams — coverage/returns, **event rates** | available | return-TD-allowed per punt / per kickoff; both sides identifiable |
| special teams — coverage/returns, **yardage** | **BLOCKED** | `yards_gained` semantics on ST rows are not what they appear: a `Punt` row reads 0 and a `Punt Return Touchdown` reads 22 on a play that gained far more. Canonical `puntReturnYards`/`kickReturnYards` live on `/games/teams`, which has no loader. |
| turnover components + opportunity rates | available | INT thrown/forced from three play types; fumbles lost vs retained from `Fumble Recovery (Own)` vs `(Opponent)`, giving a real recovery-rate denominator |
| pace — plays per game, plays per drive | available | from `plays_clean` |
| pace — drives/possessions per game | **gated** | `drives` is not declared in `data/context_sources.json`; see §2.4 |
| pace — seconds per play, time of possession | **BLOCKED** | `clock` dropped from plays (`ingest.py:291`), `elapsed`/`startTime`/`endTime` dropped from drives (`ingest.py:418`). Both exist upstream; neither is in the cache. |
| verified personnel changes / news | **BLOCKED** | `qb_availability_feed`, `unit_availability_feed`, `cfbd_portal` are all `not_requested` in `data/context_sources.json` |

**Empirical attribution check (this session, `plays_2026`, game 401867916).** On a
`Kickoff` row, `offense` is the **kicking** team — row 24 is a Robert Morris passing
touchdown, row 25 is a Kickoff with `offense = Robert Morris`. On punt rows,
`offense` is the **punting** team (row 4, `Punt Return`, 4th-and-4, offense =
Wagner). So attribution is `offense = the team that had possession`, consistently.
That makes ST **counts** trustworthy. It does not rescue ST **yardage**, whose
field semantics remain unverified.

### 2.4 The drives cache is not a declared context source

`data/context_sources.json` declares 13 sources. `plays_clean` is covered by the
approved `model_drive_features`. **`drives` is not declared at all.** By the
manifest's own rule an undeclared source cannot be read, so every drive-derived
possession measure is gated until Curtis approves an entry. This program adds
`cfbd_drives` with `approval.status: "pending"` and no loader (a non-approved source
declaring a live loader fails `tests/test_context_contracts.py`). Flipping it to
`approved` is a one-line unblock and is the single highest-value item in the
handoff.

### 2.5 Environment

* `.venv/` is a **macOS** venv: `.venv/bin/python3.11 -> /opt/homebrew/opt/python@3.11/bin/python3.11`, and `/opt/homebrew` does not exist inside the device VM. The instruction "use `.venv/bin/python`" is therefore only executable in Curtis's own Terminal. On the device VM this session used a scratch venv over the system 3.10 (`~/venvs/ss`, `--system-site-packages`, plus `pytest` and `scipy`). CI runs 3.11.
* A stale `.git/index.lock` was present and blocking `tests/test_clean_checkout_importability.py` and all of `tests/test_legacy_entry_points_fail_closed.py` (19 errors, 3 failures). Swept into `_to_delete/` per the project's standing workaround (the mount forbids `unlink`).
* The device VM has **no GitHub push credentials**; `git ls-remote` succeeds only because the repo is public. Push and PR creation were done from the cloud container, which holds a token.

## 3. Baseline test state

Full suite on-device before any change (`~/venvs/ss/bin/python -m pytest -q --ignore=data`):

```
3 failed, 880 passed, 19 errors in 19.98s
```

All 22 non-passes were the stale `.git/index.lock` (`nflvalue.release_gate.GitError:
git write-tree failed (128)`), in `test_clean_checkout_importability.py` and
`test_legacy_entry_points_fail_closed.py`. After sweeping the lock the suite is
green. **No pre-existing product failure was found**, and none was introduced.

## 4. The double-counting hazard, stated precisely

There is **no** common-opponent logic and **no** strength-of-schedule term anywhere
in the repo (case-insensitive grep for `sos`, `strength_of_schedule`,
`common_opponent` across `nflvalue/`, `scripts/`, `tests/`, `*.json`, `*.md`: zero
hits). Nothing to reuse — and nothing that would visibly conflict, which is exactly
what makes the mistake easy.

Opponent strength is already embedded in the ratings three times over: the
per-game opponent column in the design matrix; the conference blocks penalised by
cross-conference game count only; and connectivity, which scales each team's ridge
penalty and can veto a team outright.

Three concrete ways a new evidence feature double-counts it:

1. **A raw season-to-date aggregate fed beside a rating-based prediction.** Its
   largest source of variance is *who they played* — precisely what `off[s]` already
   netted out. Bad-schedule teams get penalised twice.
2. **A hand-rolled SOS feature** ("mean of opponents' net ratings"). Those net
   values are the output of a solve that already conditioned on that schedule. The
   coefficient fits the residual of a term that is definitionally zero-mean in the
   residual: noise, presented as signal.
3. **A common-opponents comparison.** "A beat X by 20, B beat X by 3, so A − B =
   +17" is the pairwise special case of what the ridge solves globally, with less
   information and no regularisation — and its errors correlate with the rating's
   own, so it looks predictive in-sample.

The structural defence already exists and this program adopts it: every factor is
scored against the **walk-forward residual `margin - pred_b`**, never the outcome
(`factor_battery.py:27-32`, `:145`, `:206`; and
`data/early_strength_prereg_2026-09-07.json`, which passes `model_margin` as a
conditioning feature so any schedule information the rating already used has zero
explanatory power left).

**Open question flagged, not answered.** `pred_b` in the battery frames appears to
be the Workstream-B *ridge* prediction, while production weeks 4+ price with
*kalman*. If a feature is scored against ridge residuals but the shipped forecast is
kalman, the "already used up" argument holds only for the ridge's schedule handling.
This is recorded in `00_contract.md` §7 as a question for Astra, not resolved here.

## 5. Obsidian: nothing to reuse

Audited the whole vault (`_to_delete/` excluded).

* **There are no per-team notes and no per-game notes anywhere.** The unit of a note
  is the *week*. Games appear only as table rows and as `**Away @ Home**` bullets
  inside week pages. `picks/` holds 17 generated/agent-written weekly pages.
* **No frontmatter in `picks/` at all** — every file starts with `# <title>`.
  `research/` uses `type / project / description / created / tags`.
* **No stable ID in any frontmatter.** `game_id` lives in CSV columns and inline
  prose (`· game 401858423`); teams are keyed by the CFBD display string.
* **No marker-bounded generated-section convention exists.** One grep hit vault-wide
  and it is a dataview query in the MOC, not a generated region. Protection today is
  *file-level*: scripts own whole files under `picks/` and rewrite them; humans write
  different files in the same folder.
* Every vault writer (`render_picks.py`, `render_straight_up.py`,
  `render_spread_scrutiny.py`, `miss_review.py`) does whole-file `open(path, "w")`.
  **Zero append-mode writes exist in `scripts/` or `nflvalue/`.**
* The vault path idiom to copy is `render_picks.default_out_dir()` (`:272`): the
  vault folder when the repo sits inside the vault, else `data/…` in the repo.
* Immutable pregame records exist, but in `data/snapshots/`, not the vault.
  Grading today **rewrites the same file** with a filled result column. The only
  append-only artefact in the vault is `_worklog.md`, and only by convention.
* `research/spread-improvement-plan-2026-09-08.md` §"Workstream C" already specs a
  per-team automatic page with a hand-written "notable insights" field beside the
  numbers. That is the requirement this program implements, and it is the only place
  the vault contemplates protecting hand-written prose.

So there is no existing location to reuse. `00_contract.md` §5 registers a
stable-ID layout, and this program introduces the vault's first marker-bounded
generated-section convention.

## 6. What this audit could not determine

1. Whether `pred_b` is ridge or kalman (see §4). Affects the residual argument.
2. Who wrote `data/cfbd/venues.parquet`, `team_locations_2026.parquet` and
   `game_venues_2026.parquet` — no code in the repo writes them, only
   `scripts/build_board.py` reads them. Only 2026 exists, so there is no historical
   game→venue map.
3. Whether the `[s:codex-final-score]` worklog claim is still live. It has no
   `release`, so by the vault's own rule `docs/final_score_program`, `hot.md` and
   the MOC were claimed by another agent when this session started. This session
   recorded its own claim and noted the collision rather than assuming abandonment.
4. Which Obsidian plugins are installed — `.obsidian/` sits above the folder grant.
5. `data/snapshots/2026_wk01/latest.json` stores a machine-absolute `path` under a
   *different* session mount. `list_snapshots` recomputes it; `read_latest` returns
   it verbatim. Not this program's bug, but a trap for anything that dereferences it.
