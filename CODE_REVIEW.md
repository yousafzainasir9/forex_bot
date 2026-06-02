# Code Review — Forex Bot (M5 EMA/RSI/ATR, MT5)

Reviewed: 2026-06-02. Scope: full `bot/`, `main.py`, `webapp/`, `tests/`, config.
Verdict: **well-architected and unusually honest about its own limits, but the
default risk configuration and the backtester quietly disagree with how the bot
actually trades.** Two of those gaps materially weaken the "DEMO-first, sized at
1%, backtest reflects live" story the README sells. Fix the HIGH items before
trusting any backtest number or running on a funded account.

The grading is from an engineering + risk standpoint, not a promise about edge.
EMA-crossover scalping on M5 is, as the README admits, a coin-flip-with-costs
baseline; nothing below changes that.

---

## What's good (keep it)

- **Clean layering.** Config → Data → Indicators → Strategy → Scanner → Risk →
  Execution → Monitor are genuinely separable; the pure modules (indicators,
  strategy, risk, scanner scoring, report) take plain numbers and are unit-testable
  offline. This is the right shape.
- **Look-ahead discipline is real** where it counts: indicators are causal
  (`adjust=False`, Wilder RMA, no `shift(-1)`), `get_candles` drops the forming
  bar, the strategy reads only `iloc[-1]/[-2]`, and the backtest enters at the
  *next* bar's open with worst-case intrabar fills (stop assumed first when a bar
  spans both). That's more rigor than most retail bots.
- **Walk-forward + overfit benchmark** (`walk_forward` vs `full_in_sample`) and the
  robustness heatmap show the author understands overfitting, not just metrics.
- **Safety defaults**: DEMO default, LIVE needs two aligned env values, REAL-account
  connection is refused in DEMO, kill switch blocks new orders but still allows
  closes, every order carries SL/TP, idempotent reconciliation deduped by
  position id. Good defense-in-depth.
- **Test breadth** ~100 tests across all modules.

---

## HIGH — fix before funded use

### H1. The default risk mode silently bypasses the safety guardrails
`RISK_MODE` defaults to **`"tiered"`** (`config.py:211`). In tiered mode
`RiskManager.effective_risk()` ignores `risk_per_trade`/`max_daily_loss` entirely
and returns the tier table (`risk.py:46-53`), which for a sub-$100 account allows
**10–20% risk per trade and 25–40% daily loss**.

But `assert_safe_to_run()` only validates the *fixed* fields —
`risk_per_trade ≤ 0.05` and `max_daily_loss ≤ 0.20` (`config.py:236-243`) — and
never inspects the tier table. So the "sane range" guard that the code advertises
is **inert in the default configuration**, and the README's "risk 1% per trade"
framing is not what ships. A small account flipped to LIVE would trade far more
aggressively than any line of documentation implies.

Also cosmetic-but-misleading: `status.json` reports `"max_daily_loss":
s.max_daily_loss` (the 3% fixed value) while the actually-enforced limit is the
tier value (`main.py:215`).

**Fix:** either default `RISK_MODE=fixed`, or make `assert_safe_to_run()` validate
every tier in the active table against the same ceilings, and surface the
*effective* limits everywhere they're displayed.

### H2. The backtest does not model trend-riding — but trend-riding is ON by default
`RIDE_TREND_AFTER_TP` defaults to **`True`** (`config.py:208`). Live, that means
execution sends **no take-profit to the broker** (`execution.py:246`,
`broker_tp = 0`) and the position is exited by `_trend_ride_check` (rides past the
target, closes on the first bar that fails a higher-high / lower-low).

The backtester does none of this. `run_backtest` hard-codes a fixed TP and exits
intrabar the moment price touches it (`backtest.py:214-238, 178-196`). So the
backtest's exits structurally differ from live for the default config — directly
contradicting the module's headline claim that it "reuses the exact same code the
live bot uses" and "reflects how the bot would actually have behaved."

**Impact:** every backtest/walk-forward number (expectancy, PF, drawdown, the
"honest OOS estimate") describes a strategy you are not running. **Fix:** port
`trend_ride_exit` into the backtest loop and gate it on the same setting, or at
minimum print a loud warning that the backtester models fixed-TP only.

### H3. Trend-riding removes the TP but never trails the stop
When riding, the broker TP is dropped and the **only** downside protection left is
the *original* ATR stop set at entry (`execution.py:246`; `_trend_ride_check`
never calls `modify_sl_tp`). Between hitting target and the next turning bar, the
stop does not ratchet toward break-even. A trade can run to +3R, reverse, and exit
all the way back at the original ≈ −1R on a fast move/gap. That is a real,
asymmetric give-back that the fixed-TP backtest (H2) will never show.

**Fix:** when TP is first reached, move the SL to at least break-even (ideally
trail it by ATR), so riding can't surrender the whole gain.

---

## MEDIUM

### M1. Backtest sizing also diverges from live
`run_backtest` builds `RiskManager(...)` without `risk_mode`, so it sizes at the
**fixed** 1% (`backtest.py:138-142`) while live defaults to tiered. Lot sizes —
and therefore P&L, drawdown, and the daily-loss halt — won't match live. Pass the
configured `risk_mode`/tiers into the backtester.

### M2. `trend_ride_exit` doc says "closes," code uses bar extremes
`strategy.py:209-224` decides the turn on **highs/lows** (`last_high < prev_high`),
but the `main.py` docstrings and the `run_pipeline` comment describe exiting when a
bar **closes** against the trade (`main.py:360-361, 416-420`). Behaviour and docs
disagree; using extremes makes the exit hair-triggered (one lower high ends the
ride). Reconcile the two and pick deliberately.

### M3. Two contradictory position-count caps
`RiskManager.max_open_positions` is built from `MAX_OPEN_POSITIONS` (default **1**,
`main.py:65-73`) and vetoes any entry past that count (`risk.py:198`). But
`scan_mode=top_n` caps at `scan_top_n` (default 3) and `any` at
`max_open_positions`. So a user who sets `SCAN_MODE=top_n` without also raising
`MAX_OPEN_POSITIONS` gets silently vetoed after the first position — a foot-gun
with two sources of truth. Derive one from the other or validate consistency in
`assert_safe_to_run`.

### M4. Broker→UTC offset is inferred from a single tick vs wall clock
`_utc_offset_sec()` rounds `(tick.time − now)/900` to detect the server offset
(`data.py:150-171`). If the first read happens while the market is closed (stale
tick) the offset can be wrong, mis-stamping every candle's UTC time — which then
shifts the daily rollover, the `today`/`yesterday` report filters, and the
daily-loss day boundary. Prefer the explicit `BROKER_UTC_OFFSET_HOURS` for
anything important, and consider validating the auto value against a known session.

### M5. Break-even trades count as wins
`pnl >= 0` is treated as a win in `report.py:189`, `monitor.py:146`, and
`backtest.py:96`. Zero/near-zero P&L trades inflate win rate. Use `> 0`, or report
scratches separately.

---

## LOW / notes

- **L1.** If the bot starts mid-day after losses already occurred, `DailyState`
  baselines to *current* equity (`main.py` `start()`), so the daily-loss halt
  forgets earlier damage until the next UTC rollover.
- **L2.** The RSI filter (`BUY only if RSI<70`, `SELL only if RSI>30`) almost never
  blocks a *fresh* EMA cross — RSI is rarely extreme right at a cross — so it adds
  little. If it's meant to be a filter, tighten the thresholds or use slope/divergence.
- **L3.** EMA(9/21) crossover on M5 is whipsaw- and cost-heavy; expect many small
  losses. The README is honest about this — just don't let a flattering (and, per
  H2, unrepresentative) backtest override that.
- **L4.** `.env.icmarkets` and `.env.oanda` are committed (passwords are clearly
  placeholders, good) but carry 7–8 digit `MT5_LOGIN` values — confirm those are
  example numbers, not real demo accounts, before pushing anywhere public.
- **L5.** Web control endpoints (`/api/control/bot|kill|close`) are unauthenticated;
  fine while bound to `127.0.0.1` as documented — never expose without auth.
- **L6.** I could not execute the test suite in this environment: the Linux sandbox
  received truncated / null-byte-corrupted copies of several source files (a
  file-sync artifact — the real files read clean and parse fine). Run
  `uv run pytest -q` locally to confirm green; coverage looks strong on inspection.

---

## Suggested order of attack
1. H1 — default to `fixed` risk *or* validate tiers in `assert_safe_to_run`; fix the
   status display.
2. H2 + M1 — make the backtester model trend-riding and the configured risk mode, or
   warn loudly. Re-run walk-forward afterward; treat prior results as void.
3. H3 — move SL to break-even on TP touch when riding.
4. M2–M5 then the LOW items.

## What to validate after fixing
Re-run walk-forward with the *actual* live config (tiered sizing + trend-ride +
break-even stop), realistic spread and commission. Compare in-sample vs OOS
expectancy and PF; confirm OOS doesn't collapse. Then forward-test on demo for
weeks and reconcile bot-reported P&L against the broker statement before drawing
any conclusion.

*Educational/engineering review only — not financial advice. Most retail FX bots
lose money; nothing here implies an edge.*

---

## Resolution log (fixes applied 2026-06-02)

All HIGH/MEDIUM items and the actionable LOW item were fixed:

- **H1** `config.py` — `RISK_MODE` now defaults to **`fixed`** (matches the
  documented 1%); `assert_safe_to_run` now **validates every tier** against hard
  caps (risk ≤ 25%, daily ≤ 50%) so an aggressive/typo'd table can't silently run;
  `banner()` prints `risk TIERED (scales by equity)` instead of a misleading fixed
  number. `main.py` status now reports the **effective** daily-loss limit.
  *(Your `.env` keeps `RISK_MODE=tiered`; the default change only affects fresh
  installs. The shipped tiers pass the new caps, so your bot still starts.)*
- **H2 / M1** `backtest.py` — `run_backtest` now takes `ride_trend_after_tp`,
  `risk_mode`, and `tiers`, models trend-ride exits via the live
  `trend_ride_exit`, and sizes with the configured risk mode. The CLI defaults
  both from `.env` (override with `--ride-trend/--no-ride-trend`, `--risk-mode`)
  and prints the active backtest config. Walk-forward + grid inherit it.
- **H3** Break-even lock-in — when the target is first reached the stop moves to
  entry: live (`main.py._trend_ride_check` calls `modify_sl_tp`) and in the
  backtest. Riding can no longer round-trip a winner to a full loss.
- **M2** `main.py` `_trend_ride_check` docstring corrected to describe candle
  **extremes** (highs/lows), matching the tested code.
- **M3** `main.py` — risk manager's `max_open_positions` is raised to `scan_top_n`
  in `top_n` mode so the two caps can't contradict.
- **M4** `data.py` — broker→UTC offset rounds to 30 min and is **rejected** if
  outside the −12h..+14h band (stale-tick guard); set `BROKER_UTC_OFFSET_HOURS`
  to be certain.
- **M5** Win rate now counts only `pnl > 0` (break-even is no longer a "win") in
  `report.py`, `monitor.py`, and `backtest.py`.
- **L1** `main.py.start()` seeds the daily-loss baseline with today's realized P&L
  so a mid-session restart can't forget earlier losses.

Not code-changed: **L2** (weak RSI filter) and **L3** (M5 crossover whipsaw) are
strategy-design calls — left as-is pending your direction. **L4** the committed
`.env.icmarkets/.env.oanda` use placeholder passwords (safe), just confirm the
login numbers are examples. **L5** web controls remain localhost-only by design.

Verification note: edits were confirmed via the authoritative file tools; the
Linux sandbox mount was frozen on a stale snapshot this session, so run
`uv run pytest -q` locally to get a green confirmation.
