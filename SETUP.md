# Quick Setup — Forex Bot (M5, DEMO)

Everything needed to run the app and review the web dashboard. Full detail is in `README.md`.

> Runs on a **demo account only** by default. Educational software — not financial advice.

---

## A. Install (one-time)

| What | Where |
|---|---|
| Python 3.10+ | https://www.python.org/downloads/ (tick "Add Python to PATH") |
| MetaTrader 5 terminal | https://www.metatrader5.com/en/download |
| `uv` (package manager) | PowerShell: `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` then reopen PowerShell |

Then install the project dependencies:

```powershell
cd "C:\Users\LENOVO\Desktop\Trader-M5\forex_bot"
uv sync
```

---

## B. Get a demo account (gets you the 3 credentials)

Easiest way — inside the MT5 terminal, no website needed:

1. **File → Open an Account.**
2. Choose **"Open a demo account to trade virtual money without risk"** → **Next**.
3. Pick a server (e.g. `MetaQuotes-Demo`) → fill name/email → **Next**.
4. The final screen shows **Login**, **Password**, **Server** — write them down (shown once).
5. **Finish.** You're now logged in (bottom-right shows a live connection).

Enable trading: **Tools → Options → Expert Advisors → ✅ Allow algorithmic trading.** Leave MT5 running.

---

## C. Configure `.env`

```powershell
copy .env.icmarkets .env      # any template is fine; or .env.oanda
notepad .env
```

Set these three lines (from step B), then save:

```ini
MT5_LOGIN=10011122379
MT5_PASSWORD=your-master-password
MT5_SERVER=MetaQuotes-Demo
```

Leave the rest as-is (DEMO mode, M5, EURUSD, 1% risk).

---

## D. Run the bot

In a PowerShell window (MT5 must be open):

```powershell
cd "C:\Users\LENOVO\Desktop\Trader-M5\forex_bot"
uv run python -m bot.data --check     # 1) confirm connection (prints your balance + a price)
uv run python main.py --once          # 2) dry run — logs one signal, places no orders
uv run python main.py                 # 3) live demo loop (Ctrl-C to stop)
```

Leave `main.py` running — it acts once every 5 minutes on each closed candle.

---

## E. Run the WEB DASHBOARD (frontend review)

Open a **second** PowerShell window (keep the bot running in the first):

```powershell
cd "C:\Users\LENOVO\Desktop\Trader-M5\forex_bot"
uv run python -m webapp.app
```

Then open this in your browser:

### 👉 http://127.0.0.1:5000

You'll see: a **live M5 candlestick chart** with the EMA 9/21 overlays (the last
candle updates in real time from the price tick), live equity/balance/price, open
positions, the latest signal, Start/Stop and Kill-switch controls, performance KPIs
with Today/Week/Month/custom filters, the equity curve, and a live log. Hover any figure for a plain-English definition; click
**"How this bot works"** at the top for the strategy explanation.

Stop the web app with **Ctrl-C**. Change the port with `--port 8000` if 5000 is busy.

> One-click alternative: double-click **`run_webapp.bat`** (dashboard) and **`run_bot.bat`** (bot).

---

## F. Review WITHOUT MetaTrader (demo/preview only)

No MT5 or broker? You can still see the frontend and reports using bundled sample data:

```powershell
cd "C:\Users\LENOVO\Desktop\Trader-M5\forex_bot"
uv sync

# Static dashboard from a sample ledger — opens an HTML file directly:
start logs\dashboard_sample.html

# Backtest with built-in synthetic data + a visual HTML report:
uv run python -m bot.backtest --demo --report
start logs\backtest_report.html
```

The live web app (section E) also starts without MT5 — it just shows "snapshot/—"
for the live figures until the bot has run once, but the page, controls, filters,
and charts all work for review.

---

## If something fails

| Symptom | Fix |
|---|---|
| `MT5 initialize() failed` | MT5 isn't running, or algo trading is off (Tools → Options → Expert Advisors). |
| `Could not select symbol` | Set `SYMBOL` in `.env` to the exact Market Watch name (`EURUSD`, or `EUR_USD` on OANDA). |
| Login error | Use the **master** password (not investor); recheck `MT5_SERVER` spelling. |
| Web page won't load | Make sure `uv run python -m webapp.app` is still running; use the printed URL. |
| Port 5000 in use | `uv run python -m webapp.app --port 8000` → open http://127.0.0.1:8000 |

See `README.md` for reports, backtesting, tuning, and how it all works.
