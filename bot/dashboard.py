"""Generate a self-contained HTML performance dashboard from trades.csv.

Reads the closed-trade ledger, embeds it as JSON into a single standalone HTML
file (no server needed — just open it in a browser), and renders:
  * period filter buttons: today / yesterday / 2d / 3d / week / month / all / custom
  * KPI cards: net P&L, win rate, profit factor, trades, expectancy, avg R, max DD
  * an equity curve (cumulative P&L) and a P&L-by-day bar chart
  * a sortable trades table

All filtering and metrics are computed client-side in JavaScript so the buttons
are instant and mirror bot/report.py exactly. Chart.js is loaded from a CDN; if
offline, the charts simply won't render but the cards and table still work.

CLI:
  uv run python -m bot.dashboard                  # writes logs/dashboard.html
  uv run python -m bot.dashboard --open           # also open it in your browser
  uv run python -m bot.dashboard --file path.csv --out report.html
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .report import load_trades


def _rows_for_web(csv_path) -> List[dict]:
    df = load_trades(csv_path)
    if df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        ct = r.get("close_time_utc")
        ot = r.get("open_time_utc")
        out.append({
            "close_time": ct.isoformat() if ct is not None and not _isnat(ct) else "",
            "open_time": ot.isoformat() if ot is not None and not _isnat(ot) else "",
            "symbol": str(r.get("symbol", "")),
            "side": str(r.get("side", "")),
            "lots": _num(r.get("lots")),
            "entry": _num(r.get("entry")),
            "exit": _num(r.get("exit")),
            "pnl": _num(r.get("pnl")),
            "commission": _num(r.get("commission")),
            "swap": _num(r.get("swap")),
            "r_multiple": _num(r.get("r_multiple")),
        })
    return out


def _isnat(v) -> bool:
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except Exception:
        return False


def _num(v) -> float:
    try:
        f = float(v)
        return f if f == f else 0.0  # filter NaN
    except (TypeError, ValueError):
        return 0.0


def build_html(csv_path) -> str:
    rows = _rows_for_web(csv_path)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trades": rows,
    }
    blob = json.dumps(payload).replace("</", "<\\/")
    return _TEMPLATE.replace("/*__DATA__*/", blob)


def _default_csv() -> Path:
    return Path(__file__).resolve().parent.parent / "logs" / "trades.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML performance dashboard.")
    parser.add_argument("--file", default=str(_default_csv()), help="path to trades.csv")
    parser.add_argument("--out", default=None, help="output HTML path (default: logs/dashboard.html)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="open the dashboard in your default browser after writing")
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else (_default_csv().parent / "dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_for_web(args.file)
    html = build_html(args.file)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard -> {out}  ({len(rows)} trades embedded)")
    if args.open_browser:
        webbrowser.open(out.resolve().as_uri())
    return 0


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Forex Bot — Performance Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0e1117; --panel:#161b22; --panel2:#1c2230; --line:#2a3140;
    --txt:#e6edf3; --muted:#8b949e; --green:#2ea043; --red:#f85149; --accent:#58a6ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;
         align-items:baseline;gap:14px;flex-wrap:wrap}
  header h1{font-size:18px;margin:0;font-weight:600}
  header .sub{color:var(--muted);font-size:12px}
  .wrap{padding:20px 24px;max-width:1200px;margin:0 auto}
  .filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px;align-items:center}
  .btn{background:var(--panel2);border:1px solid var(--line);color:var(--txt);
       padding:7px 13px;border-radius:8px;cursor:pointer;font-size:13px}
  .btn:hover{border-color:var(--accent)}
  .btn.active{background:var(--accent);color:#04101f;border-color:var(--accent);font-weight:600}
  .custom{display:flex;gap:6px;align-items:center;margin-left:auto;color:var(--muted);font-size:12px}
  .custom input{background:var(--panel2);border:1px solid var(--line);color:var(--txt);
                padding:6px;border-radius:6px;font-size:12px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .card .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  .card .val{font-size:22px;font-weight:700;margin-top:6px}
  .pos{color:var(--green)} .neg{color:var(--red)}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .panel h3{margin:0 0 10px;font-size:13px;font-weight:600;color:var(--muted)}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
  th{color:var(--muted);font-weight:600;cursor:pointer;user-select:none}
  tbody tr:hover{background:var(--panel2)}
  .empty{color:var(--muted);text-align:center;padding:40px;font-size:14px}
  @media(max-width:820px){.charts{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Forex Bot — Performance Dashboard</h1>
  <span class="sub" id="meta"></span>
</header>
<div class="wrap">
  <div class="filters" id="filters">
    <button class="btn" data-p="today">Today</button>
    <button class="btn" data-p="yesterday">Yesterday</button>
    <button class="btn" data-p="2d">2 days</button>
    <button class="btn" data-p="3d">3 days</button>
    <button class="btn" data-p="week">Week</button>
    <button class="btn" data-p="month">Month</button>
    <button class="btn active" data-p="all">All</button>
    <span class="custom">
      <label>From <input type="date" id="from"/></label>
      <label>To <input type="date" id="to"/></label>
      <button class="btn" data-p="custom">Apply</button>
    </span>
  </div>

  <div class="cards" id="cards"></div>

  <div class="charts">
    <div class="panel"><h3>Equity curve (cumulative net P&amp;L)</h3><canvas id="equity" height="200"></canvas></div>
    <div class="panel"><h3>P&amp;L by day</h3><canvas id="daily" height="200"></canvas></div>
  </div>

  <div class="panel">
    <h3>Trades</h3>
    <div id="tableWrap"></div>
  </div>
</div>

<script>
const PAYLOAD = /*__DATA__*/;
const TRADES = (PAYLOAD.trades || []).map(t => ({...t, _close: t.close_time ? new Date(t.close_time) : null}));
let equityChart=null, dailyChart=null, sortKey="close_time", sortDir=-1, currentPeriod="all";

document.getElementById("meta").textContent =
  `${TRADES.length} trades · generated ${PAYLOAD.generated_utc} · times in UTC`;

function startOfTodayUTC(now){return new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),now.getUTCDate()));}

function filterTrades(period){
  const now=new Date();
  if(period==="all") return TRADES.slice();
  if(period==="today"){const s=startOfTodayUTC(now);return TRADES.filter(t=>t._close&&t._close>=s);}
  if(period==="yesterday"){const s=startOfTodayUTC(now);const y=new Date(s.getTime()-864e5);
    return TRADES.filter(t=>t._close&&t._close>=y&&t._close<s);}
  if(period==="week"||period==="7d"){const s=new Date(now.getTime()-7*864e5);return TRADES.filter(t=>t._close&&t._close>=s);}
  if(period==="month"||period==="30d"){const s=new Date(now.getTime()-30*864e5);return TRADES.filter(t=>t._close&&t._close>=s);}
  const m=/^(\d+)d$/.exec(period);
  if(m){const s=new Date(now.getTime()-parseInt(m[1])*864e5);return TRADES.filter(t=>t._close&&t._close>=s);}
  if(period==="custom"){
    const f=document.getElementById("from").value, tt=document.getElementById("to").value;
    let rows=TRADES.slice();
    if(f){const s=new Date(f+"T00:00:00Z");rows=rows.filter(t=>t._close&&t._close>=s);}
    if(tt){const e=new Date(new Date(tt+"T00:00:00Z").getTime()+864e5);rows=rows.filter(t=>t._close&&t._close<e);}
    return rows;
  }
  return TRADES.slice();
}

function summarize(rows){
  const n=rows.length;
  const wins=rows.filter(r=>r.pnl>=0), losses=rows.filter(r=>r.pnl<0);
  const gp=wins.reduce((a,r)=>a+r.pnl,0), gl=-losses.reduce((a,r)=>a+r.pnl,0);
  const net=rows.reduce((a,r)=>a+r.pnl,0);
  const pf= gl>0 ? gp/gl : (gp>0?Infinity:0);
  const avgR= n? rows.reduce((a,r)=>a+(r.r_multiple||0),0)/n : 0;
  // max drawdown on cumulative pnl in row order (by close time)
  const ordered=rows.slice().sort((a,b)=>(a._close?-a._close: 0) - (b._close? -b._close:0));
  let cum=0,peak=-Infinity,dd=0;
  rows.slice().sort((a,b)=>(a._close||0)-(b._close||0)).forEach(r=>{cum+=r.pnl;peak=Math.max(peak,cum);dd=Math.max(dd,peak-cum);});
  return {n,wins:wins.length,losses:losses.length,winRate:n?wins.length/n:0,
          net,gp,gl,pf,avgR,expectancy:n?net/n:0,
          best:n?Math.max(...rows.map(r=>r.pnl)):0, worst:n?Math.min(...rows.map(r=>r.pnl)):0,
          maxDD:dd};
}

const money=v=> (v>=0?"+":"")+v.toFixed(2);
const cls=v=> v>0?"pos":(v<0?"neg":"");

function renderCards(s){
  const pf = s.pf===Infinity?"∞":s.pf.toFixed(2);
  const verdict = s.net>0?"PROFIT":(s.net<0?"LOSS":"FLAT");
  const cards=[
    ["Result", `<span class="${cls(s.net)}">${verdict}</span>`],
    ["Net P&L", `<span class="${cls(s.net)}">${money(s.net)}</span>`],
    ["Trades", `${s.n} <span style="font-size:13px;color:var(--muted)">(${s.wins}W/${s.losses}L)</span>`],
    ["Win rate", (s.winRate*100).toFixed(1)+"%"],
    ["Profit factor", pf],
    ["Expectancy", `<span class="${cls(s.expectancy)}">${money(s.expectancy)}</span>`],
    ["Avg R", `<span class="${cls(s.avgR)}">${(s.avgR>=0?"+":"")+s.avgR.toFixed(2)}</span>`],
    ["Max drawdown", s.maxDD.toFixed(2)],
    ["Best / Worst", `<span class="pos">${money(s.best)}</span> / <span class="neg">${money(s.worst)}</span>`],
  ];
  document.getElementById("cards").innerHTML = cards.map(
    ([l,v])=>`<div class="card"><div class="label">${l}</div><div class="val">${v}</div></div>`).join("");
}

function renderCharts(rows){
  const sorted=rows.slice().filter(r=>r._close).sort((a,b)=>a._close-b._close);
  let cum=0; const eqLabels=[], eqData=[];
  sorted.forEach(r=>{cum+=r.pnl; eqLabels.push(r._close.toISOString().slice(0,16).replace("T"," ")); eqData.push(+cum.toFixed(2));});
  const byDay={};
  sorted.forEach(r=>{const d=r._close.toISOString().slice(0,10); byDay[d]=(byDay[d]||0)+r.pnl;});
  const dayLabels=Object.keys(byDay).sort(), dayData=dayLabels.map(d=>+byDay[d].toFixed(2));

  if(window.Chart){
    if(equityChart)equityChart.destroy();
    equityChart=new Chart(document.getElementById("equity"),{type:"line",
      data:{labels:eqLabels,datasets:[{data:eqData,borderColor:"#58a6ff",backgroundColor:"rgba(88,166,255,.12)",
        fill:true,tension:.2,pointRadius:0,borderWidth:2}]},
      options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#8b949e",maxTicksLimit:6},grid:{color:"#2a3140"}},
        y:{ticks:{color:"#8b949e"},grid:{color:"#2a3140"}}}}});
    if(dailyChart)dailyChart.destroy();
    dailyChart=new Chart(document.getElementById("daily"),{type:"bar",
      data:{labels:dayLabels,datasets:[{data:dayData,
        backgroundColor:dayData.map(v=>v>=0?"#2ea043":"#f85149")}]},
      options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#8b949e",maxTicksLimit:8},grid:{display:false}},
        y:{ticks:{color:"#8b949e"},grid:{color:"#2a3140"}}}}});
  }
}

function renderTable(rows){
  if(!rows.length){document.getElementById("tableWrap").innerHTML='<div class="empty">No trades in this period.</div>';return;}
  const cols=[["close_time","Close (UTC)"],["symbol","Symbol"],["side","Side"],["lots","Lots"],
    ["entry","Entry"],["exit","Exit"],["pnl","Net P&L"],["r_multiple","R"]];
  const sorted=rows.slice().sort((a,b)=>{
    let x=a[sortKey],y=b[sortKey];
    if(sortKey==="close_time"){x=a._close?a._close.getTime():0;y=b._close?b._close.getTime():0;}
    if(typeof x==="string"){return sortDir*x.localeCompare(y);}
    return sortDir*((x||0)-(y||0));
  });
  const head="<tr>"+cols.map(([k,l])=>`<th data-k="${k}">${l}${sortKey===k?(sortDir>0?" ▲":" ▼"):""}</th>`).join("")+"</tr>";
  const body=sorted.map(r=>{
    const c=r._close?r._close.toISOString().slice(0,16).replace("T"," "):"";
    return `<tr><td>${c}</td><td>${r.symbol}</td><td>${r.side}</td><td>${r.lots}</td>`+
           `<td>${r.entry}</td><td>${r.exit}</td>`+
           `<td class="${cls(r.pnl)}">${money(r.pnl)}</td>`+
           `<td class="${cls(r.r_multiple)}">${(r.r_multiple>=0?"+":"")+(r.r_multiple||0).toFixed(2)}</td></tr>`;
  }).join("");
  document.getElementById("tableWrap").innerHTML=`<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(sortKey===k)sortDir*=-1; else{sortKey=k;sortDir=-1;} renderTable(rows);
  });
}

function update(period){
  currentPeriod=period;
  const rows=filterTrades(period);
  renderCards(summarize(rows));
  renderCharts(rows);
  renderTable(rows);
}

document.querySelectorAll(".btn[data-p]").forEach(b=>b.onclick=()=>{
  if(b.dataset.p!=="custom"){
    document.querySelectorAll(".btn[data-p]").forEach(x=>x.classList.remove("active"));
    b.classList.add("active");
  }
  update(b.dataset.p);
});

update("all");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
