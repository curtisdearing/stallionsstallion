"""Fail-closed guard for this repository's legacy, unsupported entry points.

stallionsstallion ships exactly ONE supported product: college-football
paper-only forecasts, produced by ``scripts/weekly_loop.py``.

The repository was ported from an NFL project (fablesfable, 2026-07-17) and
still carries that project's wagering-oriented tooling as historical source:
bankroll and staking math, player-prop projections, a value-slate dashboard.
None of it is maintained, validated, or in scope here, and none of it may run.

Deleting that source outright would destroy provenance for decisions the
worklog and the MOC still reference, so it is kept intact and made inert
instead: every legacy entry point calls :func:`refuse` *before* it can execute
any of that logic.  The code survives as a record; the program refuses to
produce output.

Two properties make this a real gate rather than a warning:

* **stdlib only.**  A guard that could itself fail to import (missing pandas,
  an unreadable config) would hand control straight back to the code paths it
  exists to stop.  This module imports nothing outside the standard library.
* **Nothing on stdout.**  The refusal is written to stderr and the process
  exits non-zero.  A refused entry point therefore emits an empty stdout,
  which is what the regression tests assert: not "no wagering output was
  observed", but "no output was produced at all".
"""

from __future__ import annotations

import sys
from typing import NoReturn, Optional, TextIO

#: The one product this repository supports.
SUPPORTED_PRODUCT = "college-football paper-only forecasts"

#: The one operational entry point that produces it.
SUPPORTED_ENTRY_POINT = "scripts/weekly_loop.py"

#: A concrete, copy-pasteable invocation of that entry point.
SUPPORTED_INVOCATION = "python3 scripts/weekly_loop.py --job pregame"

#: Distinctive exit status.  argparse already owns 2, so a refusal has to be
#: distinguishable from a usage error by both callers and tests.
EXIT_CODE = 78

#: Stable, greppable first line.  Tests and scripts/release_check.py match on
#: this exact string; keep it ASCII so subprocess capture never re-encodes it.
REFUSAL_MARKER = "LEGACY ENTRY POINT - UNSUPPORTED, REFUSED"


def message(entry_point: str, *, capability: str = "") -> str:
    """The refusal text for ``entry_point``.

    ``capability`` names, for a human reader, what this file used to do -- so
    someone who ran it on purpose learns why it stopped rather than guessing.
    """
    lines = [
        REFUSAL_MARKER,
        "",
        f"  {entry_point} is legacy source retained for historical reference.",
        "  It is not part of this repository's supported product and will not",
        "  run.",
    ]
    if capability:
        lines += ["", f"  What it was: {capability}."]
    lines += [
        "",
        f"  This repository supports exactly one product: {SUPPORTED_PRODUCT},",
        f"  produced by {SUPPORTED_ENTRY_POINT}.",
        "",
        f"  Supported:  {SUPPORTED_INVOCATION}",
        "  Details:    README.md",
        "",
        "  The inherited NFL wagering stack (bankroll sizing, staking math,",
        "  player-prop projections, value-slate dashboard) is unmaintained and",
        "  unvalidated here.  It is refused before it can execute, not after.",
    ]
    return "\n".join(lines)


def refuse(entry_point: str, *, capability: str = "",
           stream: Optional[TextIO] = None) -> NoReturn:
    """Write the refusal to stderr and exit ``EXIT_CODE``.  Never returns.

    Raising ``SystemExit`` (rather than calling ``os._exit``) keeps the guard
    usable at module scope: an interpreter running the file as a script exits
    with the code and prints nothing further, while a test that imports the
    module sees an ordinary, catchable exception.
    """
    out = sys.stderr if stream is None else stream
    print(message(entry_point, capability=capability), file=out)
    try:
        out.flush()
    except Exception:  # noqa: BLE001 - flushing must never mask the refusal
        pass
    raise SystemExit(EXIT_CODE)
