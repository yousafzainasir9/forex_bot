# Forex Trading Bot — v1 (M5 EMA/RSI/ATR, DEMO-first)

A transparent, modular intraday forex bot for **MetaTrader 5 (MT5)**. It trades
one pair on the M5 (5-minute) timeframe using an EMA-crossover signal, an RSI filter, and
ATR-based stops, with a risk manager that can veto any trade.

> ⚠️ **Educational software for DEMO accounts.** Most retail forex bots lose
> money. This is a learning baseline, not a money-maker. Never run against real
> funds until you have months of reviewed demo results and understand every line.
> This is engineering guidance, not financial advice.

---

## Contents
1. [How the strategy works (plain English)](#how-the-strategy-works-plain-english)
2. [Key terms explained (glossary)](#key-terms-explained-glossary)
3. [Architecture](#architecture)
4. [Prerequisites (one-time)](#1-prerequisites-one-time)
5. [Open a broker demo account](#2-open-a-broker-demo-account)
6. [Install the bot](#3-install-the-bot)
7. [Configure `.env`](#4-configure-env)
8. [Run it](#5-run-it-in-this-order)
9. [`.env` key reference](#env-key-reference)
10. [Profit / loss tracking & reports](#profit--loss-tracking--reports)
11. [Live web app (dashboard + controls)](#live-web-app-dashboard--controls)
12. [Tuning for M5](#tuning-for-m5-and-how-not-to-fool-yourself)
13. [Troubleshooting](#troubleshooting)
14. [Backtesting & walk-forward](#backtesting--walk-forward-analysis)
15. [Safety model](#safety-model)
16. [Tests](#tests)

---

## How the strategy works (plain English)

New to trading? Start here. The bot watches one currency pair (default EUR/USD)
and checks it once every 5 minutes (each "M5 candle"). On each check it does
one of four things: **BUY**, **SELL**, **CLOSE**, or **HOLD** (do nothing).

It decides using three simple tools, layered together:

1. **Direction — two moving averages crossing.** A moving average is just the
   average price over the last N bars, redrawn each bar so it forms a smooth line.
   The bot tracks a **fast** line (last 9 bars) and a **slow** line (last 21 bars).
   When the fast line crosses *above* the slow line, price is turning up → a BUY
   idea. When it crosses *below*, price is turning down → a SELL idea. (This kind
   of crossing is the classic "EMA crossover".)

2. **Momentum filter — RSI.** RSI is a 0–100 gauge of how stretched price is.
   Near 70+ the market is "overbought" (already ran up a lot); near 30- it's
   "oversold" (already fell a lot). The bot only BUYs if RSI is below 70 (not
   already overbought) and only SELLs if RSI is above 30 (not already oversold).
   This stops it from chasing a move that may be exhausted.

3. **Safety net — ATR-based stop and target.** ATR measures how much the pair
   typically moves in a bar (its "normal wiggle"). Every trade gets a
   **stop-loss** (auto-close if price goes against you) placed 1.5 × ATR away, and
   a **take-profit** (auto-close in profit) placed 1.5 × further than the stop — so
   a winner aims to earn 1.5× what a loser costs (a 1 : 1.5 "risk/reward").

The bot also **exits early**: if you're in a BUY and the lines cross back down (or
vice-versa), it closes the trade rather than waiting for the stop. And if nothing
lines up, it simply **HOLDs** — sitting out is a valid, deliberate choice.

In one sentence: *buy when the trend turns up and isn't overbought, sell when it
turns down and isn't oversold, always with a protective stop, and otherwise wait.*

### The exact rules (the precise version)

- **BUY (go long)** when the fast EMA (9) crosses **above** the slow EMA (21)
  **and** RSI(14) is below 70.
- **SELL (go short)** when the fast EMA crosses **below** the slow EMA **and**
  RSI is above 30.
- **Stop-loss** = entry price ± (ATR(14) × 1.5).
- **Take-profit** = a 1 : 1.5 risk/reward distance from entry.
- **Exit early** on the opposite EMA cross.
- Otherwise **HOLD**.

### Money rules (how much it risks)

- Risks **1% of your account per trade** — the position size is calculated so that
  if the stop-loss is hit, you lose about 1%, no more.
- **At most 1 trade open at a time.**
- If losses reach **3% of the day's starting balance**, it **stops trading for the
  rest of the day** and resumes tomorrow.

> Important: this is a deliberately simple, transparent *learning* strategy, not a
> proven money-maker. Its job is to teach you the mechanics safely on a demo
> account. Most simple retail strategies do not beat the market.

> **Why M5 (5-minute candles)?** It's the default here — it reacts faster and gives
> more trading opportunities than higher timeframes. The trade-off, in plain terms:
> faster timeframes mean **more signals but more noise**, and because the ATR-based
> stop is smaller in absolute price, the broker's **spread and commission eat a
> bigger share of each trade** — so costs matter more and the 3% daily-loss halt is
> reached sooner on a bad run. You can switch to a calmer timeframe any time by
> setting `TIMEFRAME=M15` (or `H1`) in `.env`; everything else works identically.

## Key terms explained (glossary)

Every term the bot and dashboard use, in beginner language. (The web dashboard
also shows these as hover tooltips.)

**Account & money**

- **Balance** — settled cash from trades that have *already closed*. Only changes
  when a trade closes (or you deposit/withdraw).
- **Equity** — your account's value *right now*: balance plus the floating profit
  or loss of any open trades. If you have no open trades, equity = balance.
- **Floating (unrealised) P&L** — the profit/loss of a trade that's still open. It
  moves with every price tick and isn't "real" until the trade closes.
- **Realised P&L** — profit/loss locked in once a trade closes. This is what lands
  in the trade log and the reports.
- **Lot** — trade size. 1.00 lot = 100,000 units of the base currency; 0.10 = a
  "mini" lot, 0.01 = a "micro" lot. The bot picks the lot size for you from your
  1% risk and the stop distance.
- **Pip / spread** — a pip is the small standard price increment for a pair. The
  **spread** is the gap between the buy price (ask) and sell price (bid) — a cost
  you pay on entry.

**Prices & orders**

- **Bid / Ask (live price)** — bid is the price you can **sell** at; ask is the
  price you can **buy** at. The dashboard shows them as "bid / ask".
- **LONG vs SHORT** — LONG means you bought and profit if price **rises**; SHORT
  means you sold and profit if price **falls**.
- **Entry / Exit** — the price your trade opened at / closed at.
- **Stop-loss (SL)** — a price where the trade auto-closes to **cap your loss**.
- **Take-profit (TP)** — a price where the trade auto-closes to **bank your gain**.

**Indicators (how it decides)**

- **EMA (Exponential Moving Average)** — a smoothed average of recent prices that
  reacts faster to new prices than a plain average. Fast EMA = 9 bars, slow = 21.
- **EMA crossover** — the moment the fast EMA crosses the slow EMA; the bot's
  trigger for a possible trade.
- **RSI (Relative Strength Index)** — a 0–100 momentum gauge. >70 = overbought,
  <30 = oversold. Used as a filter so the bot doesn't chase tired moves.
- **ATR (Average True Range)** — the pair's typical move per bar ("normal
  volatility"). Used to size the stop-loss so it adapts to calm vs choppy markets.

**Performance metrics (the reports & dashboard cards)**

- **Net P&L** — total profit minus losses over a period, after costs. The bottom line.
- **Win rate** — percentage of trades that finished in profit. High win rate alone
  doesn't equal profit — the *size* of wins vs losses matters too.
- **Profit factor** — gross profit ÷ gross loss. Above 1.0 = winners outweigh
  losers; 2.0 = you made twice what you lost; below 1.0 = losing.
- **Expectancy** — average net P&L **per trade**. Positive = the typical trade makes
  money.
- **R / R-multiple** — result measured in units of risk. 1R is what you risked on a
  trade; +2R means you made twice your risk, −1R means you lost the full risk.
- **Risk/reward (R:R)** — how much you aim to win versus lose on a trade. 1 : 1.5
  means risking 1 to try to make 1.5.
- **Max drawdown** — the largest peak-to-trough drop in your cumulative P&L — your
  worst losing streak, in money. Lower is better.
- **Daily loss vs cap** — how much of the 3% daily-loss limit today has used. At
  100% the bot halts new trades until tomorrow.

## Architecture

```
Config → Data → Indicators → Strategy → Risk Manager → Execution → MT5 (DEMO)
                                            │
                       Logger / Monitor  ◄──┴── (records every decision)
```

| File | Role |
|---|---|
| `bot/config.py` | Settings, DEMO/LIVE flag (defaults DEMO), kill switch |
| `bot/data.py` | MT5 connect, closed candles, ticks, account, symbol spec |
| `bot/indicators.py` | Causal EMA / RSI / ATR (no look-ahead) |
| `bot/strategy.py` | BUY / SELL / HOLD / CLOSE signal engine |
| `bot/risk.py` | Position sizing, stops/targets, daily halt, veto |
| `bot/execution.py` | Place / modify / close orders, position tracking |
| `bot/monitor.py` | Logging, trade CSV, running stats, daily summary |
| `bot/report.py` | Performance reports with time filters over trades.csv |
| `bot/dashboard.py` | Self-contained HTML dashboard (filters, charts, table) |
| `bot/backtest.py` | Walk-forward backtester (same pipeline, realistic costs) |
| `bot/control.py` | Runtime kill-switch flag shared with the web app |
| `webapp/app.py` | Flask web app: live API + start/stop/kill controls |
| `webapp/templates/dashboard.html` | Single-page live dashboard UI |
| `main.py` | Runner — one pipeline pass per closed candle |

---

## 1. Prerequisites (one-time)

You need **Windows** — the `MetaTrader5` Python library only works on Windows with
the MT5 terminal installed. (On macOS/Linux you can still run the unit tests, but
not connect to a broker.)

| What | Where | Notes |
|---|---|---|
| Python 3.10+ | https://www.python.org/downloads/ | Tick **"Add Python to PATH"** during install. |
| MetaTrader 5 terminal | https://www.metatrader5.com/en/download | Or download from your broker's site. |
| `uv` (package manager) | https://docs.astral.sh/uv/ | Install command below. |
| A broker **demo** account | see step 2 | Free, no funding required. |

Install `uv` in **PowerShell**, then reopen PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Open a broker demo account

Pick the broker you'd realistically go live with one day, and demo on that exact
one so spreads and symbol names match. Both below are free with no funding.

### Option A — IC Markets  (best for algo / tight spreads; non-US clients)

- **Sign up:** https://www.icmarkets.com/global/en/open-trading-account/demo
- Choose the **MetaTrader 5** platform and the **Raw Spread** account type.
- From the welcome email, note: **login**, **password** (the *master* password,
  not the investor one), and **server**.
- Common demo servers (confirm yours in MT5): `ICMarkets-Demo`,
  `ICMarketsSC-Demo`, `ICMarketsEU-Demo`.
- Symbol is usually plain `EURUSD` (confirm in Market Watch).

### Option B — OANDA  (best regulation breadth; works for US clients)

- **Sign up:** https://www.oanda.com/  →  Open Account → **Demo**.
- In the OANDA dashboard, enable the **MetaTrader 5** platform. OANDA issues a
  **separate MT5 login/password** — distinct from your oanda.com web login. Use
  the MT5 one here.
- Common demo servers (confirm yours in MT5): `OANDA-Demo-1`,
  `OANDA-GlobalMarkets-Demo`.
- Symbol uses an **underscore**: `EUR_USD` (confirm in Market Watch).
- US accounts follow FIFO / no-hedging rules — fine here, since the bot holds at
  most one position.

### After signing up

1. Open the MT5 terminal and log into your demo account once manually
   (File → Login to Trade Account) so it remembers the connection. **Leave the
   terminal running** whenever the bot runs.
2. Enable algo trading: Tools → Options → Expert Advisors →
   ✅ "Allow algorithmic trading".
3. Make sure your symbol is visible: in Market Watch, right-click → Show All, and
   note the **exact** symbol name.

> The `MT5_SERVER` and `SYMBOL` strings must match the terminal **exactly**.
> These two are the most common cause of connection failures.

---

## 3. Install the bot

Open PowerShell in the project folder and sync dependencies:

```powershell
cd "C:\Users\LENOVO\Desktop\Trader-M5\forex_bot"
uv sync
```

This creates a `.venv` and installs pandas, numpy, MetaTrader5, etc.

---

## 4. Configure `.env`

Two ready-made templates ship with the project — just copy the one for your
broker to `.env`, then edit the three credential lines:

**IC Markets:**
```powershell
copy .env.icmarkets .env
notepad .env
```

**OANDA:**
```powershell
copy .env.oanda .env
notepad .env
```

(There is also a generic `.env.example` if you use a different broker.)

Fill in only these three, from your broker welcome email / MT5 login dialog:

```ini
MT5_LOGIN=51234567
MT5_PASSWORD=your-demo-master-password
MT5_SERVER=ICMarketsSC-Demo
```

Leave everything else at its defaults (DEMO mode, 1% risk, 3% daily stop). See the
[key reference](#env-key-reference) below for what each setting does.

---

## 5. Run it — in this order

**Phase 1 — confirm the data connection** (no trading, just prints a price):
```powershell
uv run python -m bot.data --check
```
You should see your account equity (marked `DEMO`), a live tick, and the last 10
candles. If this works, the MT5 connection is good.

**Phase 3 — dry run, log only** (evaluates the strategy, places **no orders**):
```powershell
uv run python main.py --once
```
Check `logs\bot.log` for the signal and what the risk manager *would* decide.

**Phase 5 — the live demo loop** (places real trades on your **demo** account):
```powershell
uv run python main.py
```
Runs continuously, acting once per closed M5 candle. Press **Ctrl-C** to stop
gracefully (writes a daily summary). Trades are logged to `logs\trades.csv`.

**Run the test suite** (works on any OS, no broker needed):
```powershell
uv run pytest -q
```

---

## `.env` key reference

| Key | Default | Meaning |
|---|---|---|
| `MT5_LOGIN` | — | **Required.** Your demo account number. |
| `MT5_PASSWORD` | — | **Required.** The *master/trading* password (not investor). |
| `MT5_SERVER` | — | **Required.** Exact server name from the MT5 login dialog. |
| `MT5_TERMINAL_PATH` | auto | Optional path to `terminal64.exe` if not auto-detected. |
| `TRADING_MODE` | `DEMO` | **Leave as DEMO.** The safe mode. |
| `I_UNDERSTAND_LIVE_TRADING_RISKS` | *(empty)* | Leave empty. Required only to ever go live. |
| `KILL_SWITCH` | `0` | Set to `1` to block all new orders instantly. |
| `SYMBOL` | `EURUSD` | Pair to trade. IC Markets: `EURUSD`; OANDA: `EUR_USD`. |
| `TIMEFRAME` | `M5` | Candle timeframe (`M1/M5/M15/M30/H1/H4/D1`). |
| `RISK_PER_TRADE` | `0.01` | Risk 1% of equity per trade (used in `fixed` mode). |
| `RISK_MODE` | `fixed` | `fixed` = 1% for all balances; `tiered` = scale risk up on small accounts (aggressive, validated against hard caps). |
| `MAX_DAILY_LOSS` | `0.03` | Halt for the day after losing 3% (used in `fixed` mode). |
| `REQUIRE_HTF_ALIGN` | `true` | Hard gate: only trade when the M5 signal agrees with the `HTF_TIMEFRAME` trend. |
| `SESSION_FILTER` | `true` | Only open new trades inside the session-hour window (open trades still managed any time). |
| `SESSION_START_HOUR` / `SESSION_END_HOUR` | `7` / `16` | UTC trading window (may wrap past midnight). |
| `EMA_FAST` / `EMA_SLOW` | `9` / `21` | Fast & slow moving-average lengths (fast must be < slow). |
| `RSI_PERIOD` | `14` | RSI lookback. |
| `RSI_MODE` | `confirm` | `confirm` = RSI must agree with the cross (≥/≤ `RSI_MIDLINE`); `filter` = legacy overbought/oversold veto. |
| `RSI_MIDLINE` | `50` | Confirm-mode threshold: long needs RSI ≥ this, short ≤ this. |
| `RSI_OVERBOUGHT` / `RSI_OVERSOLD` | `70` / `30` | RSI gates used only when `RSI_MODE=filter`. |
| `ADX_PERIOD` / `ADX_MIN` | `14` / `20` | ADX lookback and minimum trend strength to open a trade. |
| `REQUIRE_ADX` | `true` | Hard gate: only enter when ADX ≥ `ADX_MIN` (skip chop). |
| `MAX_TOTAL_DRAWDOWN` | `0.15` | Halt new entries if peak→equity drawdown exceeds this (0 = off; resets on restart). |
| `MAX_CONSECUTIVE_LOSSES` | `6` | Halt new entries after this many losses in a row (0 = off). |
| `MAX_PER_CURRENCY` | `1` | Cap concurrent positions sharing a currency (correlation guard; top_n/any only). |
| `MAX_SPREAD_POINTS` | `0` | Execution refuses a fill if live spread exceeds this many points (0 = off). |
| `TRAIL_AFTER_TP` / `TRAIL_ATR_MULT` | `true` / `1.5` | After target, trail the stop by ATR × mult (on top of break-even). |
| `PARTIAL_TP_ENABLED` | `true` | Bank part of the position at the partial target, ride the rest. |
| `PARTIAL_TP_FRACTION` / `PARTIAL_TP_R` | `0.5` / `1.0` | Fraction to bank, and the partial target distance in R (stop multiples). |
| `ATR_PERIOD` | `14` | ATR lookback for stop sizing. |
| `ATR_MULTIPLIER` | `1.5` | Stop distance = ATR × this. Higher = wider stop, smaller size. |
| `RISK_REWARD` | `1.5` | Take-profit distance as a multiple of the stop distance. |
| `HISTORY_BARS` | `600` | Candles pulled each cycle (~50h on M5). |

---

## Tuning for M5 (and how not to fool yourself)

All strategy knobs are exposed in `.env` (see the table above) so you can
experiment without touching code. The shipped values are the **standard M5
baseline**, chosen to be transparent rather than optimised:

- **EMA 9 / 21** — the conventional fast/slow pair for 5-minute charts. If M5
  feels too choppy (lots of quick in-and-out trades), slow it down with
  `EMA_FAST=12` `EMA_SLOW=26`, or `21/55` for a calmer, trend-only feel. For more
  reactivity, `5/13`. Fast must always be **less than** slow.
- **RSI 14, `RSI_MODE=confirm` (midline 50)** — by default RSI must *agree* with
  the cross: a long needs RSI ≥ 50, a short needs RSI ≤ 50, so momentum has to back
  the signal. Set `RSI_MIDLINE=55/45` to demand stronger momentum (fewer trades).
  Set `RSI_MODE=filter` for the old overbought/oversold veto (`RSI_OVERBOUGHT`/
  `RSI_OVERSOLD`), which rarely triggers at a fresh cross.
- **ATR 14 × 1.5** — sets the stop distance. On M5 the spread is a larger share of
  a small ATR, so many M5 traders widen this to **`ATR_MULTIPLIER=2.0`** so normal
  noise and the spread don't stop them out prematurely. A wider stop means a
  *smaller* position for the same 1% risk — your risk per trade is unchanged.
- **HISTORY_BARS 600** — ~50 hours of M5 data: comfortably past indicator warmup,
  spans weekends, and gives the charts more context. Raise it if you want longer
  history in the dashboard.

> **The honest caveat (read this):** changing these numbers until past results look
> good is **overfitting** — it produces settings tuned to history that usually fall
> apart live. Treat the values above as sensible starting points, change **one**
> thing at a time, and judge it on **forward** demo performance over weeks, not on a
> single good-looking afternoon. A backtest can rule out disasters but cannot prove
> an edge. If you want to compare settings properly, ask and I'll add a
> walk-forward backtester that replays history through this exact pipeline.

### Whipsaw filters (regime + timing)

An EMA crossover bleeds when it trades in a range or in thin hours. Two **structural**
filters fight that — both default ON, both env-tunable, and both honoured in the
backtester so your results reflect them:

- **Higher-timeframe trend gate (`REQUIRE_HTF_ALIGN=true`).** Only take an M5 long
  when the `HTF_TIMEFRAME` (default M15) trend is UP, and a short only when it's
  DOWN. Counter-trend crosses — the bulk of whipsaw — are skipped. If the HTF trend
  can't be read, entries are skipped (safe). This is the single most effective
  filter; set `REQUIRE_HTF_ALIGN=false` to disable.
- **Session filter (`SESSION_FILTER=true`).** Only *open* new trades inside
  `SESSION_START_HOUR`–`SESSION_END_HOUR` (UTC, default 07–16 = London + the
  London/NY overlap). Open trades are still managed and closed any time. The window
  may wrap past midnight (e.g. start 22, end 6). Set `SESSION_FILTER=false` for 24h.
- **ADX trend-strength gate (`REQUIRE_ADX=true`, `ADX_MIN=20`).** The HTF gate says
  *which way* the trend is; ADX says *whether there's a trend at all*. Entries are
  skipped when ADX is below `ADX_MIN` (i.e. the market is ranging) — the most direct
  filter against the chop where a crossover bleeds. Set `REQUIRE_ADX=false` to disable.

Recommended first pass for cutting whipsaw: keep both of these on, use
`RSI_MODE=confirm`, leave EMA periods at 9/21, and judge changes on **out-of-sample**
walk-forward results — not in-sample. Backtest both gates explicitly with
`python -m bot.backtest --demo --htf-gate --session` (or `--no-htf-gate` /
`--no-session` to compare).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MT5 initialize() failed` | The MT5 terminal isn't running, or "Allow algorithmic trading" is off. |
| `Could not select symbol` | Wrong `SYMBOL`; use the exact Market Watch name (e.g. `EUR_USD`, `EURUSD.a`). |
| Login error / "Invalid account" | You used the *investor* password — use the **master** one; recheck `MT5_SERVER` spelling. |
| Connected as `REAL`, bot refuses to run | Good — it's protecting you. Use a **demo** account. |
| `MetaTrader5` won't install | You're not on Windows; the trading layer is Windows-only. `uv run pytest` still works. |

---

## Profit / loss tracking & reports

Every **closed** trade is reconciled from MT5's deal history and appended to
`logs/trades.csv` — including exits the broker fills on its own (stop-loss /
take-profit), not just bot-initiated closes. Each row records the realized **net
P&L** (profit + commission + swap), the R-multiple, entry/exit, stop/target, and
the open reason. The daily-loss halt uses this realized P&L, so the 3% stop is now
actually enforced.

`logs/trades.csv` columns:

```
close_time_utc, open_time_utc, symbol, side, lots, entry, exit,
stop_loss, take_profit, pnl, commission, swap, r_multiple, position_id,
reason_open, reason_close
```

### Performance report with time filters

Run the report tool to see whether you're up or down over any window:

```powershell
uv run python -m bot.report --period today
uv run python -m bot.report --period yesterday
uv run python -m bot.report --period 2d         # rolling last 2 days
uv run python -m bot.report --period 3d         # rolling last 3 days
uv run python -m bot.report --period week        # last 7 days
uv run python -m bot.report --period month       # last 30 days
uv run python -m bot.report --period all
uv run python -m bot.report --period custom --from 2026-05-01 --to 2026-05-15
uv run python -m bot.report --period today --csv  # also list the individual trades
```

Each report shows: PROFIT/LOSS verdict and net P&L, trade count and W/L split,
win rate, profit factor, expectancy per trade, average R-multiple, gross
profit/loss, best/worst trade, average win/loss, max drawdown, and total
commission/swap. Example:

```
Performance report — today
====================================================
  Result:            PROFIT  (net P&L +83.00)
  Trades:            2  (1W / 1L)
  Win rate:          50.0%
  Profit factor:     2.38
  Expectancy/trade:  +41.50
  Avg R-multiple:    +0.41
  ...
```

> Period windows filter on each trade's **close time** (UTC). `today`/`yesterday`
> are UTC calendar days; `2d`/`3d`/`week`/`month` are rolling windows back from now.

### Visual HTML dashboard

For an at-a-glance view instead of the CLI, generate a self-contained HTML
dashboard and open it in any browser:

```powershell
uv run python -m bot.dashboard            # writes logs/dashboard.html
uv run python -m bot.dashboard --open     # writes it and opens your browser
```

The dashboard has clickable period filters (Today / Yesterday / 2d / 3d / Week /
Month / All / custom date range), KPI cards (net P&L, win rate, profit factor,
expectancy, avg R, max drawdown, best/worst), an equity curve, a P&L-by-day bar
chart, and a sortable trades table. All filtering is instant and runs in the
browser — the trade data is embedded in the file, so it works offline (the charts
need a one-time internet load of Chart.js from a CDN; cards and table work either
way). The bot also refreshes `logs/dashboard.html` automatically on shutdown, so
after any run you can just open that file.

> A pre-filled sample, `logs/dashboard_sample.html`, ships with the project so you
> can see the layout before you have any live trades.

## Live web app (dashboard + controls)

A Flask web app gives you a real-time browser view with controls. Start it
(in a second terminal, alongside the bot):

```powershell
uv run python -m webapp.app          # http://127.0.0.1:5000
uv run python -m webapp.app --port 8000
```

Then open http://127.0.0.1:5000. The page shows:

- **Live M5 candlestick chart** — recent candles with the EMA 9/21 overlays the
  strategy trades on; the forming (current) candle updates in real time from the
  price tick. Built with TradingView Lightweight Charts; data comes from
  `logs/candles.json`, which the bot publishes each loop, served via `/api/candles`.
- **Live status** — equity, balance, current bid/ask, daily P&L vs the daily-loss
  cap, open positions with floating P&L, and the latest signal. It reads the
  `status.json` snapshot the bot writes each loop, and (on Windows with MT5 up)
  augments it with a real-time account read so equity stays current between loops.
- **Controls** — Start / Stop the bot (launched as a subprocess) and a live
  **Kill switch** toggle that blocks new entries immediately, without restarting
  the bot. The kill switch is shared via `logs/control.json`, which the bot polls
  every loop.
- **Performance** — the same period filters as the CLI (Today / Yesterday / 2d /
  3d / Week / Month / All / custom range), KPI cards, equity curve, P&L-by-day
  chart, and a sortable trades table.
- **Activity log** — a live tail of `bot.log`.

There are also one-click Windows launchers: `run_bot.bat` and `run_webapp.bat`.

> ⚠️ **Security:** the app binds to `127.0.0.1` (localhost only) by default and has
> **no authentication**. The control endpoints can affect live trading — do not
> expose it to a network or the internet without adding auth and TLS. Keep it
> local. As always, DEMO first.

### How the pieces talk

```
            logs/status.json   (bot writes each loop  → web reads)
 bot loop ─►logs/control.json  (web writes kill switch → bot reads each loop)
            logs/trades.csv    (bot appends on close   → web reads for reports)
            logs/bot.log       (bot writes             → web tails)
```

## Backtesting & walk-forward analysis

Before trusting any settings, replay them against history. The backtester
(`bot/backtest.py`) runs candles through the **exact same** indicator, strategy,
and risk-manager code the live bot uses — so it reflects the real bot, not a
look-alike. It models a spread and commission, prevents look-ahead (a signal is
decided on a closed bar and entered at the **next** bar's open), and on an
ambiguous bar assumes the **stop** filled before the target (worst case).

Quick try with built-in synthetic data (no broker or files needed):

```powershell
uv run python -m bot.backtest --demo                       # single run
uv run python -m bot.backtest --demo --walk-forward        # walk-forward
```

On your own data or live MT5 history:

```powershell
# CSV needs columns: time, open, high, low, close
uv run python -m bot.backtest --csv data\eurusd_m5.csv --walk-forward
# Pull history straight from MT5 (Windows, terminal running):
uv run python -m bot.backtest --symbol EURUSD --timeframe M5 --bars 8000 --walk-forward
# Model your broker's costs:
uv run python -m bot.backtest --csv data.csv --spread 0.00012 --commission 7
```

It writes `logs/backtest_trades.csv` (single run) or `logs/backtest_oos_trades.csv`
(walk-forward), which you can open in the same dashboard/report tools.

### Why *walk-forward*, not a plain backtest

A single backtest where you tweak settings until the curve looks good is
**overfitting** — you've fitted the parameters to that exact history, and it
usually collapses live. Walk-forward fights this: it splits the data into folds,
**picks the best parameters on an in-sample slice, then scores them on the next,
unseen out-of-sample slice**, and rolls forward. The aggregated **out-of-sample**
number is the honest estimate. The tool also prints an "overfit benchmark" (best
params fitted on *all* data) right next to it so you can see the gap — the
out-of-sample result is almost always worse, and that's the realistic one.

> Even a good walk-forward result is not a promise. It can rule out disasters and
> expose fragile settings; it cannot prove a live edge. The only real test is weeks
> of forward performance on a **demo** account. Change one parameter at a time.

### Visual report + robustness heatmap

Add `--report` to a single run to write a self-contained `logs/backtest_report.html`
— summary cards, the equity curve, a **parameter robustness heatmap**, and the
trade list, all openable in a browser:

```powershell
uv run python -m bot.backtest --demo --report
uv run python -m bot.backtest --csv data.csv --report --metric profit_factor
```

The heatmap runs the backtest across a grid of `ema_fast` × `ema_slow` and colours
each cell by the chosen metric (green = better, red = worse). **What you want to
see is a broad coloured region**, not one lone green cell in a sea of red — a
single isolated winner almost always means an overfit fluke that won't survive
live. A wide stable plateau is the sign of a setting that's robust to small changes.


## Safety model

- **DEMO is the default.** Going LIVE requires `TRADING_MODE=LIVE` **and**
  `I_UNDERSTAND_LIVE_TRADING_RISKS=I_UNDERSTAND_LIVE_TRADING_RISKS` in `.env`.
  Execution double-checks and refuses a REAL account while in DEMO mode.
- **Kill switch:** `KILL_SWITCH=1` refuses all new orders (closes still allowed).
- **Risk rules beat signals, always** — the risk manager can veto any trade.

## Tests

73 unit tests cover indicators (incl. a no-look-ahead causality check), the
strategy's cross/filter/exit logic, and the risk manager's sizing and every veto
path, the deal-to-trade reconciliation builder, and the report period filters, and the dashboard data injection, the runtime control flags, and the web API endpoints, and the backtest engine (hand-verified TP/SL fills, next-open entry, costs, walk-forward, grid-search heatmap, and HTML report). Run `uv run pytest -q`.

> Broker references are practical setup guidance, not financial advice or an
> endorsement. Confirm each broker's current regulatory status and suitability
> for your jurisdiction yourself. A good backtest does not mean a profitable bot —
> soak on demo for weeks and review the logs before considering anything live.
