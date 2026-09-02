# stallionsstallion

**One supported product: college-football paper-only forecasts.**

Everything else in this repository is historical source, kept for provenance
and explicitly unsupported. If you are looking for what this repo *does*, it is
the section immediately below and nothing else.

---

## The supported product

A weekly college-football forecast, published on paper only.

The model produces distributions and honesty reporting for scheduled FBS games.
Its output is a research forecast and is **not actionable**: no wager, stake,
bankroll allocation, or recommendation is produced, implied, or supported. Rows
that cannot be forecast honestly are published as explicit refusals with a
reason rather than filled in.

### Supported operational entry point

`scripts/weekly_loop.py` — the only supported runnable forecast path.

```
python3 scripts/weekly_loop.py --job week-open   # Tue: ingest refresh, refit, simulate, render
python3 scripts/weekly_loop.py --job pregame     # Tue-Sat: re-simulate only if the cache moved
python3 scripts/weekly_loop.py --job grade       # Sun: finals, honesty report, recap + CLV for the week before
```

`--week N` forces the week the loop is in. `week-open` and `pregame` price
week N; `grade` recaps and captures CLV for week N-1, and its manifest names
that week as `target_week`. Without `--week`, the loop's week is the lowest
week with a kickoff still in the future — so a week that ends with a Monday
game is still "current" on the Sunday before it, and that Sunday's unforced
grade run targets the week before. In week 1 that means grading nothing; in a
later week it means re-grading the previous week and deferring the one that
just finished. Grade such a week by hand with `--job grade --week N+1`.

`scripts/run_weekly.sh` wraps these jobs for scheduled local runs; it adds no
forecasting logic of its own and runs no git command. Nothing else in this
repository is a supported way to produce a forecast.

### Engine policy

One dispatch point decides which model prices a published week:
`nflvalue.engine.production_margin_engine(week)`.

| Weeks | Engine | Why |
| --- | --- | --- |
| **0–3 (early season)** | `ridge` + `preseason_prior` | The ridge path is the only one carrying a prior-year component. The kalman filter has no preseason prior at all — before it has seen games it would be pricing from nothing, which is exactly when the prior matters most. |
| **4 and later (in season)** | `config.json` `margin_engine`, registered default `kalman` | Registered on bakeoff-G evidence (pre-registered 2026-07-30): kalman 254.71 vs ridge 281.64 pooled MSE, 2021–2024 wk≥4, n=2,377; confirmed on the virgin 2025 holdout, 234.87 vs 291.13, n=617. |

`ridge` stays fully wired as the drift-comparison baseline; flipping the
`margin_engine` config key back is the entire rollback story.

Early-season rows carry their own status string
(`nflvalue.engine.EARLY_SEASON_STATUS`) saying the preseason prior still
dominates and the number is not actionable. Registered evidence windows for the
backtests, calibration and CLV logs cover **weeks 4–15 only** — weeks 0–3 are
outside every validated window, and the early-season status string exists to
keep that visible on the face of each row.

### Data

Cached only, under `data/`. The weekly loop is budget-guarded (monthly CFBD
call cap) and every job starts with a dependency-aware input health check
(`nflvalue/data_health.py`). A failed check halts the run before anything is
published: exit 3, a manifest with `halted: true` and the reason, and a
non-numeric `data/pipeline_health.json`. The `latest` pointer does not move
and the published artifact is not overwritten. Publishing is manual
(`scripts/publish_site.sh`). Details in `RUNBOOK_WEEKLY.md` and
`ops/LOAD_ME.md`.

---

## Unsupported legacy

This repository was ported from an NFL project (fablesfable, 2026-07-17) and
still carries that project's wagering-oriented tooling. It is retained as
historical source because the worklog and decision records reference it. **It
is not maintained, not validated, and cannot run.**

| Legacy entry point | Status |
| --- | --- |
| `run.py` | Refuses. Was the NFL value-slate builder and dashboard. |
| `weekly.py` | Refuses. Was the NFL weekly board graded against closing lines. |
| `scripts/auto_weekly.py` | Refuses. Was the NFL wed/t90/tuesday scheduling wrapper. |

Each refuses at the entry point with a `LEGACY ENTRY POINT - UNSUPPORTED,
REFUSED` message on stderr, an empty stdout, and exit status **78** — before
any pricing, bankroll, staking, player-prop or notification code path can
execute. The guard lives in `nflvalue/legacy_guard.py` and is stdlib-only, so
it cannot be defeated by a missing dependency.

The NFL/player-prop stack these entry points belong to (`nflvalue/pipeline.py`,
`nflvalue/montecarlo.py`, `nflvalue/dashboard.py`, the odds and Discord
clients, and the surrounding historical scripts) is likewise unsupported. Files
are intact on purpose; mass deletion is a separate, later decision.

---

## Release integrity

```
python3 scripts/release_check.py
```

Validates the **candidate tree** — HEAD plus the staged index, materialised
with `git write-tree` and extracted clean — not the working tree. It checks
that the release files are actually tracked, that `scripts/weekly_loop.py`
imports from a clean tree, that this README names one product, and that every
legacy entry point fails closed. The working tree can import modules that ship
to nobody; the candidate tree cannot.

The same invariants are asserted by the suite in
`tests/test_clean_checkout_importability.py` and
`tests/test_legacy_entry_points_fail_closed.py`.

## Tests

```
python3.11 -m pytest tests/ -q
```

Python 3.11. Runtime dependencies for the supported path are in
`requirements-weekly.txt` (`requirements.txt` is the inherited file and lists
excluded NFL/prop dependencies).
