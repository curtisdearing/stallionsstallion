"""The game context panel: used by the model / display-only / excluded /
missing, kept visibly apart.

Rendered from a finished `GameContext`, into a fragment the week explainer
concatenates. It is deliberately dumb: no computation, no defaulting, no
rounding that could imply precision the source does not have. A field with
no usable value prints why, never a number.
"""
from __future__ import annotations

import html as _html
from typing import Iterable, List

from .assemble import GameContext
from .contracts import FAMILY_LABELS, Availability, ContextValue, Usage

BUCKET_HEADINGS = {
    Usage.USED_BY_MODEL: ("Used by the model",
                          "Cleared the factor-registry gate AND is consumed "
                          "by the shipped engine."),
    Usage.DISPLAY_ONLY: ("Display-only",
                         "Shown as context. None of it entered the margin, "
                         "total or win probability."),
    Usage.EXCLUDED: ("Excluded",
                     "Argued and settled against. Rendered so the absence is "
                     "explained rather than silent."),
    Usage.MISSING: ("Missing",
                    "No usable value. Unavailable, stale, conflicting, "
                    "low-coverage or refused — never imputed."),
}

CSS = """
.ctx{background:#161c22;border:1px solid #2a333c;border-radius:10px;
padding:12px 14px;margin:12px 0}
.ctx h3{margin:0 0 2px;font-size:15px}
.ctx h4{margin:14px 0 4px;font-size:13px;letter-spacing:.02em}
.ctx .sub{color:#93a1ae;font-size:12px;margin:0 0 8px}
.ctx table{width:100%;border-collapse:collapse;font-size:12px}
.ctx td,.ctx th{border-top:1px solid #232c34;padding:4px 6px;
vertical-align:top;text-align:left}
.ctx th{color:#93a1ae;font-weight:600}
.ctx .v{font-variant-numeric:tabular-nums;white-space:nowrap}
.ctx .na{color:#c98a6b}
.ctx .prov{color:#7d8b98;font-size:11px}
.ctx .fam{color:#93a1ae}
.ctx .b-used{border-left:3px solid #4f9d69;padding-left:8px}
.ctx .b-disp{border-left:3px solid #4a7ea8;padding-left:8px}
.ctx .b-excl{border-left:3px solid #6b6b6b;padding-left:8px}
.ctx .b-miss{border-left:3px solid #a86a4a;padding-left:8px}
.ctx .gate{color:#c98a6b}
"""

_BUCKET_CLASS = {Usage.USED_BY_MODEL: "b-used", Usage.DISPLAY_ONLY: "b-disp",
                 Usage.EXCLUDED: "b-excl", Usage.MISSING: "b-miss"}


def _e(x) -> str:
    return _html.escape("" if x is None else str(x))


def _prov(v: ContextValue) -> str:
    if v.availability is Availability.OK:
        bits = [f"source {_e(v.source)}", f"id {_e(v.source_url_or_id)}",
                f"observed {_e(v.observed_at.isoformat())}",
                f"effective for {_e(v.effective_for_game.isoformat())}",
                f"confidence {_e(v.confidence)}",
                f"coverage {v.coverage:.0%}"]
    else:
        bits = [f"state {_e(v.availability.value)}"]
        if v.source:
            bits.append(f"source {_e(v.source)}")
        if v.observed_at:
            bits.append(f"observed {_e(v.observed_at.isoformat())}")
        if v.note:
            bits.append(_e(v.note))
    return " · ".join(bits)


def _rows(values: Iterable[ContextValue]) -> List[str]:
    out = []
    for v in values:
        cls = "v na" if v.availability is not Availability.OK else "v"
        reg = (f" <span class='fam'>[registry: {_e(v.registry_status)}]</span>"
               if v.registry_status else "")
        note = (f"<div class='prov'>{_e(v.note)}</div>"
                if v.note and v.availability is Availability.OK else "")
        out.append(
            f"<tr data-field=\"{_e(v.key)}\">"
            f"<td>{_e(v.label)}{reg}"
            f"<div class='prov'>{_prov(v)}</div>{note}</td>"
            f"<td class='{cls}'>{_e(v.display)}</td></tr>")
    return out


def render_context_panel(ctx: GameContext) -> str:
    counts = ctx.counts()
    p: List[str] = ["<div class='ctx'>"]
    p.append(f"<h3>Pregame context — {_e(ctx.away)} at {_e(ctx.home)}</h3>")
    p.append("<p class='sub'>Kickoff "
             f"{_e(ctx.kickoff_et.strftime('%Y-%m-%d %-I:%M %p'))} ET · "
             f"assembled {_e(ctx.generated_at.isoformat())} · "
             f"{counts['used_by_model']} used by the model, "
             f"{counts['display_only']} display-only, "
             f"{counts['excluded']} excluded, {counts['missing']} missing. "
             "This layer cannot move a projection: it is unable to import the "
             "modelling modules and they are unable to import it.</p>")

    for usage in (Usage.USED_BY_MODEL, Usage.DISPLAY_ONLY, Usage.MISSING):
        heading, blurb = BUCKET_HEADINGS[usage]
        vals = ctx.by_usage(usage)
        p.append(f"<div class='{_BUCKET_CLASS[usage]}'>")
        p.append(f"<h4>{_e(heading)} ({len(vals)})</h4>")
        p.append(f"<p class='sub'>{_e(blurb)}</p>")
        if not vals:
            p.append("<p class='sub'>Nothing in this bucket.</p></div>")
            continue
        p.append("<table><tr><th>Field</th><th>Value</th></tr>")
        seen = set()
        for fam in [f for f in FAMILY_LABELS if any(v.family == f for v in vals)]:
            p.append(f"<tr><th colspan='2' class='fam'>"
                     f"{_e(FAMILY_LABELS[fam])}</th></tr>")
            fam_vals = [v for v in vals if v.family == fam]
            seen.update(v.key for v in fam_vals)
            p.extend(_rows(fam_vals))
        p.append("</table></div>")

    heading, blurb = BUCKET_HEADINGS[Usage.EXCLUDED]
    p.append(f"<div class='{_BUCKET_CLASS[Usage.EXCLUDED]}'>")
    p.append(f"<h4>{_e(heading)} ({len(ctx.excluded_rows)})</h4>")
    p.append(f"<p class='sub'>{_e(blurb)} Status comes from "
             "data/factor_registry.json, which is the only thing that can "
             "change it.</p>")
    p.append("<table><tr><th>Factor</th><th>Why it is absent</th></tr>")
    for r in ctx.excluded_rows:
        p.append(f"<tr data-field=\"{_e(r['factor_id'])}\">"
                 f"<td>{_e(r['factor_id'])}<div class='prov'>"
                 f"{_e(r.get('description'))}</div></td>"
                 f"<td>{_e(r.get('why'))}<div class='prov'>"
                 f"source: {_e(r.get('source_doc'))} · last validated "
                 f"{_e(r.get('last_validated'))}</div></td></tr>")
    for v in ctx.by_usage(Usage.EXCLUDED):
        p.append(f"<tr data-field=\"{_e(v.key)}\"><td>{_e(v.label)}</td>"
                 f"<td class='na'>{_e(v.display)}</td></tr>")
    p.append("</table></div>")

    p.append("<div class='b-miss'><h4>Approval gates</h4>")
    p.append("<p class='sub'>A source that is not approved cannot emit a "
             "value, whatever code exists for it. Opening a gate means "
             "editing data/context_sources.json with a named approver, a date "
             "and a decision record.</p>")
    p.append("<table><tr><th>Source</th><th>State</th><th>What an approval "
             "needs</th></tr>")
    for s in ctx.source_states:
        cls = "" if s["approval"] == "approved" else "gate"
        req = ", ".join(s.get("requires") or []) or "—"
        p.append(f"<tr><td>{_e(s['source'])}<div class='prov'>"
                 f"{_e(s.get('provider') or 'no provider')}"
                 f"{' · external' if s['external'] else ''}</div></td>"
                 f"<td class='{cls}'>{_e(s['approval'])}</td>"
                 f"<td>{_e(req)}<div class='prov'>"
                 f"{_e(s['approval_record'])}</div></td></tr>")
    p.append("</table></div>")
    p.append("</div>")
    return "".join(p)
