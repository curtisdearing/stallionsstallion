"""Obsidian generation: bounded generated sections, immutable pregame records.

THIS MODULE IS WRITE-ONLY WITH RESPECT TO CALCULATION
-----------------------------------------------------
It reads an existing note for exactly one reason: to preserve the human's prose
while replacing the generated block. No number ever travels from a note back into
a calculation. `measures.py` and `profile.py` do not import this module, and no
other module references the vault directory —
`tests/test_team_evidence_vault.py::test_no_calculation_module_reads_the_vault`
asserts both directions with an AST scan, and
`test_poisoned_note_prose_cannot_change_a_profile` proves it functionally.

LAYOUT (registered in 00_contract.md section 5 because the vault had none)
--------------------------------------------------------------------------
    teams/{season}/{team_slug}.md
    games/{season}/wk{week:02d}/{game_id}--{away_slug}-at-{home_slug}.md

`team_slug` is a filename convenience. The identifier is the frontmatter
`team_id: "{season}:{CFBD school string}"`, which preserves the exact upstream
string — apostrophes, accents and all — because that string is the join key
everywhere else in this repo.

THE FIRST MARKER CONVENTION IN THIS VAULT
-----------------------------------------
The vault has no generated-section markers anywhere (one grep hit vault-wide and
it is a dataview query). Every existing writer does a whole-file
`open(path, "w")`, which is fine while a script owns a whole file and fatal the
moment a human wants to write beside the numbers. `research/
spread-improvement-plan-2026-09-08.md` already asks for exactly that: a free-text
"notable insights" field stored beside the generated ones. So:

    <!-- BEGIN GENERATED {block_id} v{n} :: do not edit inside -->
    ...
    <!-- END GENERATED {block_id} v{n} -->

Only the text between the markers is replaced. Everything else survives
byte-for-byte.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .contracts import Observation, TeamSeasonProfile

BLOCK_VERSION = 1

GENERATED_START = "<!-- BEGIN GENERATED {block_id} v{version} :: do not edit inside -->"
GENERATED_END = "<!-- END GENERATED {block_id} v{version} -->"

PREGAME_START = "<!-- BEGIN PREGAME RECORD {game_id} :: frozen, never rewritten -->"
PREGAME_END = "<!-- END PREGAME RECORD {game_id} -->"

POSTGAME_HEADING = "## Postgame reviews (append-only)"
MANUAL_HEADING = "## Notes (hand-written — never regenerated)"


class VaultError(RuntimeError):
    pass


class MalformedBlockError(VaultError):
    """A start marker without its end marker. Guessing where the block stops is
    how a generator eats a paragraph of someone's writing."""


class ImmutablePregameError(VaultError):
    """A pregame record already exists and differs. The original rationale is not
    rewritten after the result is known — that is the whole point of freezing it."""


# --------------------------------------------------------------------------- #
# Paths and identity
# --------------------------------------------------------------------------- #
def team_slug(team: str) -> str:
    """Filename convenience, NOT an identifier. The identifier is `team_id`."""
    t = unicodedata.normalize("NFKD", str(team))
    t = t.encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "unknown"


def team_id(season: int, team: str) -> str:
    return f"{int(season)}:{team}"


def vault_root(repo_root: str) -> str:
    """The vault folder when the repo sits inside it, else a repo-local preview dir.

    Same idiom as `scripts/render_picks.default_out_dir()`. This layer never writes
    into `picks/` — those files are owned by other generators and by Curtis.
    """
    parent = os.path.dirname(os.path.abspath(repo_root))
    if os.path.isdir(os.path.join(parent, "picks")):
        return parent
    return os.path.join(repo_root, "data", "vault_preview")


def team_note_path(root: str, season: int, team: str) -> str:
    return os.path.join(root, "teams", str(int(season)), f"{team_slug(team)}.md")


def game_note_path(root: str, season: int, week: int, game_id: int,
                   home: str, away: str) -> str:
    name = f"{int(game_id)}--{team_slug(away)}-at-{team_slug(home)}.md"
    return os.path.join(root, "games", str(int(season)), f"wk{int(week):02d}", name)


# --------------------------------------------------------------------------- #
# Bounded generated sections
# --------------------------------------------------------------------------- #
def _markers(block_id: str, version: int) -> Tuple[str, str]:
    return (GENERATED_START.format(block_id=block_id, version=version),
            GENERATED_END.format(block_id=block_id, version=version))


def render_bounded(existing: str, block_id: str, body: str, *,
                   version: int = BLOCK_VERSION) -> str:
    """Replace only the text between this block's markers.

    Absent markers: the block is appended and existing content is untouched.
    A start without an end: raise, rather than guess where the block ends.
    """
    start, end = _markers(block_id, version)
    block = f"{start}\n{body.rstrip()}\n{end}"
    existing = existing or ""
    has_start, has_end = start in existing, end in existing
    if has_start != has_end:
        raise MalformedBlockError(
            f"note has {'a start' if has_start else 'an end'} marker for "
            f"{block_id!r} without its counterpart; refusing to guess the "
            "boundary of someone else's writing")
    if not has_start:
        sep = "" if not existing or existing.endswith("\n\n") else (
            "\n" if existing.endswith("\n") else "\n\n")
        return f"{existing}{sep}{block}\n"
    head, _, rest = existing.partition(start)
    _, _, tail = rest.partition(end)
    return f"{head}{block}{tail}"


def extract_bounded(text: str, block_id: str, *,
                    version: int = BLOCK_VERSION) -> Optional[str]:
    start, end = _markers(block_id, version)
    if start not in text or end not in text:
        return None
    return text.partition(start)[2].partition(end)[0].strip("\n")


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _write(path: str, text: str) -> str:
    """Truncate-in-place, matching every existing vault writer.

    Deliberately not temp-file + os.replace: the mounted vault forbids `unlink`,
    and a rename over an existing file is exactly the operation that fails there.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(obs: Observation) -> str:
    if obs.is_present and obs.value is not None:
        v = (f"{obs.value:.3f}" if obs.unit in ("rate", "ppa", "rating")
             else f"{obs.value:.1f}")
        n = f"n={obs.n}" if obs.n else "n=—"
        adj = " · opponent-adjusted" if obs.opponent_adjusted else ""
        return f"| {obs.label} | {v} | {n} | {obs.n_games} | ok{adj} |"
    reason = (obs.reason or "").replace("|", "/")
    if len(reason) > 150:
        reason = reason[:147] + "..."
    return (f"| {obs.label} | — | n={obs.n} | {obs.n_games} | "
            f"{obs.availability.value}: {reason} |")


def _frontmatter(pairs: Sequence[Tuple[str, str]]) -> str:
    lines = ["---"]
    for key, value in pairs:
        v = str(value)
        lines.append(f'{key}: "{v}"' if any(c in v for c in ":#[]{}") else f"{key}: {v}")
    lines.append("---")
    return "\n".join(lines)


def team_profile_body(profile: TeamSeasonProfile, *, artifact_sha256: str,
                      commit: str, families: Optional[Sequence[str]] = None,
                      registry=None) -> str:
    """The generated body of a team-season note. Numbers, denominators, refusals.

    Every row shows its denominator and its state. A blocked measure is printed
    WITH its reason rather than omitted, because a silently absent row is
    indistinguishable from a forgotten one.
    """
    out: List[str] = []
    out.append(f"*As of {profile.season} week {profile.as_of_week}, built from "
               f"{len(profile.games_used)} completed game(s) strictly earlier than "
               f"that week.*")
    out.append("")
    out.append(f"- **team_id** `{team_id(profile.season, profile.team)}`")
    out.append(f"- **generated** {profile.generated_at_utc.isoformat()}")
    out.append(f"- **artifact** `sha256:{artifact_sha256[:16]}`")
    out.append(f"- **commit** `{commit}`")
    out.append(f"- **games used** {', '.join(str(g) for g in profile.games_used) or '—'}")
    out.append(f"- **opponents faced** {', '.join(profile.opponents_faced) or '—'}")
    if profile.neutral_site_games:
        out.append("- **neutral-site games** "
                   + ", ".join(str(g) for g in profile.neutral_site_games))
    out.append("")

    by_family: Dict[str, List[Observation]] = {}
    for obs in profile.observations:
        fam = "other"
        if registry is not None:
            try:
                fam = registry.get(obs.measure_id).family
            except Exception:                                   # noqa: BLE001
                fam = "other"
        by_family.setdefault(fam, []).append(obs)

    for fam in sorted(by_family):
        if families and fam not in families:
            continue
        out.append(f"### {fam.replace('_', ' ')}")
        out.append("")
        out.append("| measure | value | denominator | games | state |")
        out.append("|---|---|---|---|---|")
        for obs in sorted(by_family[fam], key=lambda o: o.measure_id):
            out.append(_fmt(obs))
        out.append("")

    caveats = sorted({o.caveat for o in profile.observations if o.caveat})
    if caveats:
        out.append("### Caveats carried by these measures")
        out.append("")
        for c in caveats:
            out.append(f"- {c}")
        out.append("")

    out.append("### Availability")
    out.append("")
    out.append("Every value above is `reconstructed_week_granular`: it was derived "
               "from a cache that carries no observation timestamp, so its pregame "
               "availability is the assumption that everything from week "
               f"{max(profile.as_of_week - 1, 0)} and earlier was knowable at week "
               f"{profile.as_of_week}. Retrospective *predictive* claims from these "
               "values are blocked — see `docs/team_evidence/00_contract.md` "
               "section 3.")
    return "\n".join(out)


def write_team_note(root: str, profile: TeamSeasonProfile, *,
                    artifact_sha256: str, commit: str, registry=None) -> str:
    """Write or refresh a team-season note, preserving everything outside the block."""
    path = team_note_path(root, profile.season, profile.team)
    existing = _read(path)
    if not existing:
        existing = "\n".join([
            _frontmatter([
                ("type", "generated-team-profile"),
                ("project", "stallionsstallion"),
                ("team_id", team_id(profile.season, profile.team)),
                ("team", profile.team),
                ("season", str(profile.season)),
                ("generated_by", "nflvalue.team_evidence.vault"),
                ("tags", "[team-profile, cfb, generated]"),
                ("created", profile.generated_at_utc.date().isoformat()),
            ]),
            "",
            f"# {profile.team} — {profile.season} team profile",
            "",
            MANUAL_HEADING,
            "",
            "*Anything you write in this section is preserved across regenerations.*",
            "",
        ])
    body = team_profile_body(profile, artifact_sha256=artifact_sha256,
                             commit=commit, registry=registry)
    return _write(path, render_bounded(existing, "team-season-profile", body))


def game_log_body(rows: Sequence[Dict], *, artifact_sha256: str,
                  commit: str) -> str:
    """One game, both teams, every measure with its numerator and denominator.

    Sorted by team then measure so a regeneration is byte-stable and a diff shows
    only what actually changed.
    """
    out = ["| team | opponent | measure | value | numerator | denominator |",
           "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (str(x["team"]), str(x["measure_id"]))):
        raw = r.get("value")
        val = "—" if raw is None or raw != raw else f"{float(raw):.3f}"
        out.append(f"| {r['team']} | {r['opponent']} | {r['measure_id']} | {val} | "
                   f"{float(r['num']):.0f} | {float(r['den']):.0f} |")
    out.append("")
    out.append(f"*artifact `sha256:{artifact_sha256[:16]}` · commit `{commit}`*")
    return "\n".join(out)


def pregame_body(frozen_row: Dict, *, snapshot_stamp: Optional[str] = None) -> str:
    """Render a FROZEN forecast row as the immutable pregame record.

    The row comes from `data/snapshots/{season}_wk{NN}/frozen.json`, which
    `nflvalue.snapshots.freeze_kicked_off` wrote from the newest snapshot taken
    STRICTLY BEFORE that game's kickoff. That freeze is the only artifact in this
    repo that proves what was knowable pregame, which is why the pregame record is
    copied from it rather than recomputed.
    """
    keys = ("status", "model_family", "model_version", "projected_home_margin",
            "forecast_margin_sd", "p_home_win", "projected_total",
            "forecast_sd_basis", "refusal_reason", "generated_at_utc",
            "snapshot_id", "source_digest", "config_hash", "frozen_from_stamp")
    out = ["*Frozen before kickoff. Never rewritten — postgame goes below, "
           "as an append.*", "",
           "| field | value |", "|---|---|"]
    for key in keys:
        if key in frozen_row and frozen_row[key] not in (None, ""):
            out.append(f"| {key} | {frozen_row[key]} |")
    if snapshot_stamp:
        out.append(f"| snapshot_stamp | {snapshot_stamp} |")
    return "\n".join(out)


def write_game_note(root: str, season: int, week: int, game_id: int, home: str,
                    away: str, rows: Sequence[Dict], *, artifact_sha256: str,
                    commit: str, kickoff_utc: Optional[str] = None,
                    neutral_site: bool = False,
                    pregame: Optional[Dict] = None,
                    pregame_stamp: Optional[str] = None) -> str:
    """Write a game log, and the immutable pregame record when one is available.

    Two blocks with different rules in one file: the log is regenerated on every
    run; the pregame record is written once and then refuses to change.
    """
    path = game_note_path(root, season, week, game_id, home, away)
    if not _read(path):
        _write(path, "\n".join([
            _frontmatter([
                ("type", "generated-game-log"),
                ("project", "stallionsstallion"),
                ("game_id", str(int(game_id))),
                ("season", str(int(season))),
                ("week", str(int(week))),
                ("home_team", home),
                ("away_team", away),
                ("neutral_site", "true" if neutral_site else "false"),
                ("kickoff_utc", kickoff_utc or "unknown"),
                ("generated_by", "nflvalue.team_evidence.vault"),
                ("tags", "[game-log, cfb, generated]"),
            ]),
            "",
            f"# {away} at {home} — {season} week {week}"
            + (" (neutral site)" if neutral_site else ""),
            "",
        ]))
    if pregame:
        write_pregame_record(path, game_id,
                             pregame_body(pregame, snapshot_stamp=pregame_stamp))
    body = game_log_body(rows, artifact_sha256=artifact_sha256, commit=commit)
    return _write(path, render_bounded(_read(path), f"game-log-{int(game_id)}", body))


# --------------------------------------------------------------------------- #
# Immutable pregame record, append-only postgame review
# --------------------------------------------------------------------------- #
def write_pregame_record(path: str, game_id: int, body: str, *,
                         header: Optional[str] = None) -> str:
    """Write the pregame block exactly once.

    Identical content is a no-op, so the generator is safely idempotent. Different
    content raises: the rationale recorded before kickoff is not edited after the
    result is known, and a generator that "refreshes" it has destroyed the only
    thing that made it evidence.
    """
    start = PREGAME_START.format(game_id=int(game_id))
    end = PREGAME_END.format(game_id=int(game_id))
    existing = _read(path)
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in existing:
        if end not in existing:
            raise MalformedBlockError(
                f"{path}: pregame start marker without an end marker")
        current = existing.partition(start)[2].partition(end)[0].strip("\n")
        if current.strip() == body.rstrip().strip():
            return path
        raise ImmutablePregameError(
            f"{path}: a pregame record for game {game_id} already exists and "
            "differs. Immutable by contract — append a postgame review instead of "
            "rewriting what was said before kickoff.")
    text = existing
    if not text:
        text = (header or f"# Game {game_id}") + "\n\n"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return _write(path, f"{text}{sep}{block}\n")


def append_postgame_review(path: str, game_id: int, review: str, *,
                           reviewed_at: Optional[dt.datetime] = None,
                           artifact_sha256: str = "", commit: str = "") -> str:
    """Append under the postgame heading. Never touches the pregame block."""
    stamp = (reviewed_at or dt.datetime.now(dt.timezone.utc)).isoformat()
    existing = _read(path)
    if not existing:
        raise VaultError(
            f"{path}: no note to append to. Write the pregame record first — a "
            "postgame review with no pregame record is a result with no prediction.")
    pregame_start = PREGAME_START.format(game_id=int(game_id))
    before = existing.partition(pregame_start)[2] if pregame_start in existing else ""

    entry = [f"### Review {stamp}", "", review.rstrip()]
    if artifact_sha256 or commit:
        entry += ["", f"*artifact `sha256:{artifact_sha256[:16]}` · commit `{commit}`*"]
    entry_text = "\n".join(entry)

    if POSTGAME_HEADING in existing:
        text = existing.rstrip("\n") + "\n\n" + entry_text + "\n"
    else:
        text = (existing.rstrip("\n") + "\n\n" + POSTGAME_HEADING + "\n\n"
                + entry_text + "\n")
    out = _write(path, text)

    # Structural guarantee, checked rather than asserted in a comment.
    after = _read(path).partition(pregame_start)[2] if pregame_start in existing else ""
    if pregame_start in existing:
        end = PREGAME_END.format(game_id=int(game_id))
        if before.partition(end)[0] != after.partition(end)[0]:
            raise ImmutablePregameError(
                f"{path}: appending a review altered the pregame block")
    return out
