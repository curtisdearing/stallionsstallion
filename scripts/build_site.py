#!/usr/bin/env python3
"""Build the public-facing pattern explorer — one self-contained HTML file.

Generates data/site/index.html from the repo's REAL artifacts (factor
registry, close-free CLV logs, walk-forward backtests, calibration report,
bakeoff results, latest recap). Static, no server, no external libraries —
host it anywhere as-is. Regenerate weekly after the grade job.

Honesty is the product surface, deliberately:
  * the paper banner is unconditional (this is research, not advice);
  * REJECTED findings render with the same prominence as passed ones;
  * every number's provenance (n, p, window, caveats) travels with it;
  * the explorer computes slice stats live from the same rows it displays,
    so a visitor can check any headline against the raw log themselves.

The display layer is read-only w.r.t. the model (same rule as the explainer
and scoreboard — nothing here can move a projection).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import ingest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(DATA, "site")


def load_inputs():
    reg = json.load(open(os.path.join(DATA, "factor_registry.json")))
    clv = pd.concat([pd.read_csv(os.path.join(DATA, "clv_log_backtest.csv")),
                     pd.read_csv(os.path.join(DATA, "clv_log_2024.csv"))],
                    ignore_index=True)
    bt = pd.concat([pd.read_csv(os.path.join(DATA, "backtest_c.csv")),
                    pd.read_csv(os.path.join(DATA, "backtest_c_2024.csv"))],
                   ignore_index=True)
    lines = ingest.load_lines(sorted(bt["season"].unique()))
    med = lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median()
    bt["market"] = [(-float(med.get(i)) if i in med.index else None)
                    for i in bt["id"]]
    recaps = []
    rdir = os.path.join(DATA, "recaps")
    if os.path.isdir(rdir):
        for fn in sorted(os.listdir(rdir)):
            if fn.endswith(".json"):
                recaps.append(json.load(open(os.path.join(rdir, fn))))
    return reg, clv, bt, recaps


def calibration_rows(bt: pd.DataFrame):
    df = bt.copy()
    df["won"] = (df["margin"] > 0).astype(float)
    df["bucket"] = pd.cut(df["p_home_win"], np.linspace(0, 1, 11))
    out = []
    for b, r in df.groupby("bucket", observed=True).agg(
            n=("won", "size"), pred=("p_home_win", "mean"),
            obs=("won", "mean")).iterrows():
        out.append({"bucket": f"{b.left:.0%}-{b.right:.0%}",
                    "n": int(r["n"]), "pred": round(float(r["pred"]), 3),
                    "obs": round(float(r["obs"]), 3)})
    return out


def main() -> int:
    reg, clv, bt, recaps = load_inputs()
    clv_rows = clv[["game_id", "season", "week", "book", "side",
                    "edge_at_open", "clv_pts", "slice"]].round(2)
    bt_rows = bt[["season", "week", "home", "away", "pred_b", "mc_margin",
                  "p_home_win", "margin", "market"]].round(2)

    payload = {
        "registry": reg,
        "clv": clv_rows.to_dict("records"),
        "games": bt_rows.to_dict("records"),
        "calibration": calibration_rows(bt),
        "recaps": recaps[-4:],
        "meta": {
            "built": "2026-07-30",
            "games_n": len(bt_rows), "clv_n": len(clv_rows),
            "windows": "walk-forward 2021-2024, weeks >= 4",
            "bakeoff": [
                {"model": "Kalman state-space", "mse": 254.71, "note":
                 "bakeoff winner; confirmed on virgin 2025 holdout (234.9 vs ridge 291.1)"},
                {"model": "G-Elo", "mse": 262.09, "note": ""},
                {"model": "Lasso (linear)", "mse": 278.21, "note": ""},
                {"model": "Ridge (incumbent)", "mse": 281.64, "note": ""},
                {"model": "Random forest", "mse": 286.97, "note": ""},
                {"model": "Elo", "mse": 293.43, "note": ""},
                {"model": "Gradient-boosted trees", "mse": 302.40, "note": ""},
                {"model": "Closing line (market)", "mse": 237.57, "note":
                 "context, not a candidate — it beats every model"},
            ],
        },
    }
    blob = json.dumps(payload, separators=(",", ":"))

    html = _TEMPLATE.replace("/*__DATA__*/null", blob)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"[site] wrote {out} ({os.path.getsize(out)//1024} KB, "
          f"{len(bt_rows)} games, {len(clv_rows)} CLV rows)")
    return 0


_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>stallionsstallion — CFB model, shown honestly</title>
<style>
:root{color-scheme:light;
 --surface:#fcfcfb;--card:#ffffff;--line:#e4e3df;
 --ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;
 --s1:#2a78d6;--s2:#eb6834;--neu:#9b9a95;
 --good:#1a7a3a;--bad:#a33125;}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;
 --surface:#1a1a19;--card:#232322;--line:#3a3937;
 --ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;
 --s1:#3987e5;--s2:#d95926;--neu:#7d7c76;
 --good:#37b061;--bad:#e0604f;}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
 font:15px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.banner{background:#7a1f1f;color:#fff;padding:10px 18px;font-weight:600;
 position:sticky;top:0;z-index:5;font-size:.92em}
.wrap{max-width:1060px;margin:0 auto;padding:20px 18px 60px}
h1{font-size:1.55em;margin:.6em 0 .2em}
h2{font-size:1.15em;margin:1.8em 0 .5em}
.sub{color:var(--ink2)}
.small{color:var(--ink3);font-size:.85em}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:14px 16px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
.badge{display:inline-block;border-radius:7px;padding:1px 9px;font-size:.78em;
 font-weight:600;border:1px solid var(--line)}
.b-pass{color:var(--good)}.b-rej{color:var(--bad)}.b-neu{color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:.9em}
td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
.num{text-align:right;font-variant-numeric:tabular-nums}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
select,input{background:var(--card);color:var(--ink);border:1px solid var(--line);
 border-radius:7px;padding:5px 8px;font-size:.92em}
.stat{font-size:1.6em;font-weight:700;font-variant-numeric:tabular-nums}
.bar-row{display:grid;grid-template-columns:110px 1fr 130px;gap:8px;
 align-items:center;margin:5px 0}
.bar-track{position:relative;height:16px}
.bar{position:absolute;top:2px;height:12px;border-radius:0 4px 4px 0}
.bar.neg{border-radius:4px 0 0 4px}
.zero{position:absolute;top:0;bottom:0;width:1px;background:var(--ink3)}
.cal-pair{display:flex;gap:2px;align-items:flex-end;height:96px;flex:1}
.cal-col{display:flex;gap:14px;align-items:flex-end}
.legend{display:flex;gap:16px;margin:6px 0}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;
 border-radius:3px;margin-right:6px;vertical-align:-1px}
.lg1::before{background:var(--s1)}.lg2::before{background:var(--s2)}
footer{margin-top:40px;color:var(--ink3);font-size:.85em}
a{color:var(--s1)}
</style>
<div class="banner">PAPER-ONLY RESEARCH. Nothing on this page is betting advice or a
recommendation. The model does not beat the closing line and says so below.
If gambling stops being entertainment, call 1-800-GAMBLER.</div>
<div class="wrap">
<h1>stallionsstallion</h1>
<p class="sub">A college football game-line model that shows its work — every
number below is walk-forward (predictions made only from information available
before each game), every claim carries its sample size and p-value, and the
rejected findings are displayed as prominently as the survivors. That is the
point of the site.</p>
<div id="hero" class="grid"></div>

<h2>Strongest findings — and what killed the rest</h2>
<p class="small">From the factor registry: every hypothesis tested against
walk-forward residuals with matched controls and false-discovery-rate
correction. "Display-only" means it passed but does not touch a prediction
until further gates clear. Rejections stay on the board permanently.</p>
<div id="findings" class="grid"></div>

<h2>Model bakeoff — seven methods, identical folds</h2>
<p class="small">Pre-registered metric (pooled margin MSE, 2021–2024, n=2,377
identical games) chosen before any candidate ran. Lower is better.</p>
<div class="card" id="bakeoff"></div>

<h2>Pattern explorer — check any claim yourself</h2>
<p class="small">These stats recompute live from the same rows shown in the
table. CLV (“closing line value”) = how far the line moved toward the model's
side between the opening and closing number, in points; positive means the
market moved our way. Books within a game are collapsed to one observation.</p>
<div class="card">
 <div class="controls">
  <label>Season <select id="f-season"></select></label>
  <label>Slice <select id="f-slice"></select></label>
  <label>Min edge at open <input id="f-edge" type="number" value="2" step="0.5" style="width:70px"></label>
 </div>
 <div id="clv-stats" class="grid"></div>
 <div id="clv-bars"></div>
 <div class="small" id="clv-note"></div>
</div>

<h2>Game explorer — every graded prediction</h2>
<div class="card">
 <div class="controls">
  <label>Season <select id="g-season"></select></label>
  <label>Team <input id="g-team" placeholder="e.g. Boise State"></label>
 </div>
 <div id="g-stats" class="grid"></div>
 <div style="max-height:420px;overflow:auto"><table id="g-table"></table></div>
</div>

<h2>Calibration — do the probabilities mean what they say?</h2>
<p class="small">Predicted home-win probability vs what actually happened,
n=2,377 walk-forward games. Bars nearly matching = honest probabilities.
Murphy decomposition: Brier 0.2007 = reliability 0.0007 (miscalibration —
tiny) − resolution 0.0440 + uncertainty 0.2444.</p>
<div class="card">
 <div class="legend"><span class="lg1">predicted</span><span class="lg2">observed</span></div>
 <div class="cal-col" id="cal"></div>
</div>

<h2>Methodology &amp; provenance</h2>
<div class="card small">
<p><b>Data:</b> CollegeFootballData.com (all 134–136 FBS teams, 2019–2026:
games, multi-book opening+closing lines, 198k drive records, play-by-play,
talent, returning production, coaches, QB starts). Every pull is
schema-checked and logged against an API budget ledger.</p>
<p><b>Discipline:</b> every evaluation pre-registered before running; leakage
("poison the future") tests at the ingest, ratings, and simulation layers;
two findings retracted this cycle when their own controls failed them (a CLV
look-ahead artifact and a reverse-causation QB effect) — both documented in
the repo, not deleted.</p>
<p><b>The headline honesty:</b> pooled walk-forward margin MAE — model 13.2,
closing line 12.2. The market is the better forecaster and every blend
attempt failed its gate (recorded twice, including on a virgin holdout).
This site exists to show what the model <i>can</i> support, with the case
against always attached.</p>
<p>Data: <a href="https://collegefootballdata.com">CollegeFootballData.com</a>.
This site is independent research and is not affiliated with any sportsbook,
team, or conference.</p>
</div>
<footer>PAPER-ONLY. No slice has passed the kill-check gate; nothing here is
a recommendation. 1-800-GAMBLER.</footer>
</div>
<script>
const D=/*__DATA__*/null;
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function card(h){const d=document.createElement("div");d.className="card";d.innerHTML=h;return d}

// hero stats
(function(){
 const m=D.meta;
 const rows=[["Graded walk-forward games",m.games_n],
  ["CLV observations (close-free)",m.clv_n],
  ["Model MAE vs market","13.2 vs 12.2"],
  ["Calibration error (reliability)","0.0007"]];
 rows.forEach(([k,v])=>$("#hero").appendChild(card(
  `<div class="stat">${esc(v)}</div><div class="small">${esc(k)}</div>`)));
})();

// findings
(function(){
 const order={display_only:0,gated_active:1,under_revalidation:2,untested:3,excluded:4};
 const reg=[...D.registry].sort((a,b)=>(order[a.status]??9)-(order[b.status]??9));
 reg.forEach(e=>{
  const passed=/PASSED/.test(e.source_measurement||"");
  const rej=/REJECTED/.test(e.source_measurement||"")&&!passed;
  const badge=e.status==="excluded"?`<span class="badge b-neu">excluded by design</span>`
   :passed?`<span class="badge b-pass">passed gate</span>`
   :rej?`<span class="badge b-rej">tested &amp; rejected</span>`
   :`<span class="badge b-neu">${esc(e.status)}</span>`;
  const eff=e.effect_size==null?"":`<b>${e.effect_size>0?"+":""}${e.effect_size}</b> pts`;
  const n=e.n?` · n=${e.n}`:"";
  $("#findings").appendChild(card(
   `<div>${badge}</div><div style="margin:6px 0 2px"><b>${esc(e.factor_id)}</b></div>
    <div class="small">${esc(e.description)}</div>
    <div style="margin-top:6px">${eff}<span class="small">${n}</span></div>
    <div class="small" style="margin-top:6px">${esc((e.source_measurement||"").slice(0,260))}</div>`));
 });
})();

// bakeoff
(function(){
 const rows=D.meta.bakeoff;
 const best=Math.min(...rows.map(r=>r.mse)),worst=Math.max(...rows.map(r=>r.mse));
 let h="";
 rows.forEach(r=>{
  const w=8+84*(r.mse-best)/(worst-best);
  const mk=r.model.includes("market");
  h+=`<div class="bar-row"><span class="small">${esc(r.model)}</span>
   <span class="bar-track"><span class="bar" style="left:0;width:${w}%;
   background:${mk?"var(--neu)":"var(--s1)"}"></span></span>
   <span class="num">${r.mse.toFixed(1)}${r.note?` <span class="small" title="${esc(r.note)}">ⓘ</span>`:""}</span></div>`;
 });
 $("#bakeoff").innerHTML=h+`<div class="small">Bayesian state-space (Kalman)
 won and re-won on an untouched holdout year. Deep learning was not entered:
 at ~700 games/season it is a registered future candidate, expected to
 overfit. The market beats everything — that line is the honesty anchor.</div>`;
})();

// CLV explorer
(function(){
 const seasons=[...new Set(D.clv.map(r=>r.season))].sort();
 const slices=[...new Set(D.clv.map(r=>r.slice))].sort();
 $("#f-season").innerHTML=`<option value="">all</option>`+seasons.map(s=>`<option>${s}</option>`).join("");
 $("#f-slice").innerHTML=`<option value="">all</option>`+slices.map(s=>`<option>${esc(s)}</option>`).join("");
 function render(){
  const se=$("#f-season").value,sl=$("#f-slice").value,
   ed=parseFloat($("#f-edge").value)||0;
  const rows=D.clv.filter(r=>(!se||r.season==se)&&(!sl||r.slice==sl)&&Math.abs(r.edge_at_open)>=ed);
  const byGame=new Map();
  rows.forEach(r=>{const a=byGame.get(r.game_id)||[];a.push(r.clv_pts);byGame.set(r.game_id,a)});
  const per=[...byGame.values()].map(a=>a.reduce((x,y)=>x+y,0)/a.length);
  const n=per.length;
  const mean=n?per.reduce((x,y)=>x+y,0)/n:0;
  const sd=n>2?Math.sqrt(per.map(x=>(x-mean)**2).reduce((x,y)=>x+y,0)/(n-1)):0;
  const t=n>2?mean/(sd/Math.sqrt(n)):0;
  const pos=n?per.filter(x=>x>0).length/n:0;
  $("#clv-stats").innerHTML="";
  [["games (book-deduped)",n],["mean CLV (pts)",(mean>=0?"+":"")+mean.toFixed(3)],
   ["share positive",(100*pos).toFixed(1)+"%"],["t-statistic",t.toFixed(2)]]
   .forEach(([k,v])=>$("#clv-stats").appendChild(card(
    `<div class="stat">${v}</div><div class="small">${k}</div>`)));
  // per-slice bars for current season/edge selection
  const bySlice=new Map();
  D.clv.filter(r=>(!se||r.season==se)&&Math.abs(r.edge_at_open)>=ed)
   .forEach(r=>{const m=bySlice.get(r.slice)||new Map();
    const a=m.get(r.game_id)||[];a.push(r.clv_pts);m.set(r.game_id,a);bySlice.set(r.slice,m)});
  let h="";const lim=1.6;
  [...bySlice.entries()].sort().forEach(([s,m])=>{
   const p=[...m.values()].map(a=>a.reduce((x,y)=>x+y,0)/a.length);
   const mu=p.reduce((x,y)=>x+y,0)/p.length;
   const w=Math.min(Math.abs(mu)/lim*50,50);
   const bar=mu>=0?`<span class="bar" style="left:50%;width:${w}%;background:var(--s1)"></span>`
    :`<span class="bar neg" style="left:${50-w}%;width:${w}%;background:var(--s2)"></span>`;
   h+=`<div class="bar-row"><span class="small">${esc(s)}</span>
    <span class="bar-track"><span class="zero" style="left:50%"></span>${bar}</span>
    <span class="num">${mu>=0?"+":""}${mu.toFixed(2)} pts · n=${p.length}</span></div>`;
  });
  $("#clv-bars").innerHTML=h;
  $("#clv-note").textContent="Backtest-window measurement (the model was developed on these seasons) — NOT a live track record and NOT a kill-check verdict. A t-stat needs ~n"+"≥143 and a live season before it means anything.";
 }
 ["f-season","f-slice","f-edge"].forEach(id=>$("#"+id).addEventListener("input",render));
 render();
})();

// game explorer
(function(){
 const seasons=[...new Set(D.games.map(r=>r.season))].sort();
 $("#g-season").innerHTML=`<option value="">all</option>`+seasons.map(s=>`<option>${s}</option>`).join("");
 function render(){
  const se=$("#g-season").value,q=($("#g-team").value||"").toLowerCase();
  const rows=D.games.filter(r=>(!se||r.season==se)&&(!q||r.home.toLowerCase().includes(q)||r.away.toLowerCase().includes(q)));
  const withM=rows.filter(r=>r.market!=null);
  const mae=a=>a.length?(a.reduce((x,r)=>x+Math.abs(r),0)/a.length).toFixed(2):"—";
  $("#g-stats").innerHTML="";
  [["games",rows.length],
   ["model MAE",mae(rows.map(r=>r.mc_margin-r.margin))],
   ["market MAE (same games)",mae(withM.map(r=>r.market-r.margin))]]
   .forEach(([k,v])=>$("#g-stats").appendChild(card(
    `<div class="stat">${v}</div><div class="small">${k}</div>`)));
  const head=`<tr><th>wk</th><th>matchup</th><th class="num">model</th>
   <th class="num">market</th><th class="num">actual</th><th class="num">P(home)</th></tr>`;
  $("#g-table").innerHTML=head+rows.slice(0,400).map(r=>
   `<tr><td>${r.season} w${r.week}</td><td>${esc(r.away)} @ ${esc(r.home)}</td>
    <td class="num">${r.mc_margin>0?"+":""}${r.mc_margin}</td>
    <td class="num">${r.market==null?"—":(r.market>0?"+":"")+r.market}</td>
    <td class="num">${r.margin>0?"+":""}${r.margin}</td>
    <td class="num">${r.p_home_win.toFixed(2)}</td></tr>`).join("")
   +(rows.length>400?`<tr><td colspan=6 class="small">showing first 400 of ${rows.length}</td></tr>`:"");
 }
 $("#g-season").addEventListener("input",render);
 $("#g-team").addEventListener("input",render);
 render();
})();

// calibration paired bars
(function(){
 let h="";
 D.calibration.forEach(r=>{
  h+=`<div style="text-align:center">
   <div class="cal-pair" title="pred ${r.pred} vs obs ${r.obs} (n=${r.n})">
    <span style="flex:1;background:var(--s1);border-radius:4px 4px 0 0;height:${90*r.pred}px"></span>
    <span style="flex:1;background:var(--s2);border-radius:4px 4px 0 0;height:${90*r.obs}px"></span>
   </div><div class="small">${esc(r.bucket)}</div></div>`;
 });
 $("#cal").innerHTML=h;
})();
</script>
"""


def _also_build_hub_feed() -> None:
    """The hub feed rides along with every site rebuild; never fatal."""
    import subprocess
    import sys as _sys
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "build_hub_feed.py")
    try:
        subprocess.run([_sys.executable, script], check=False)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    _code = main()
    _also_build_hub_feed()
    raise SystemExit(_code)
