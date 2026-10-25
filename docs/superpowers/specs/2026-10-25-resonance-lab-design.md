# Resonance Lab — Design

**Date:** 2026-10-25 · **Owner:** Bella Marco · **Status:** draft for review

## Goal

Find a tradable edge by testing Tesla-inspired, numerology, personal-number and
pattern-recognition signals against real market data, with the same rigor used for
ordinary quant signals. Priority: make money > learn > content.

**Rule:** every idea is a hypothesis and the backtest decides. A signal is traded only if
it survives the tests below. If nothing survives, the email says so.

## Scope and phases

| Phase | Output | Needs |
|---|---|---|
| 1. Swing battery | Scoreboard of all swing hypotheses on daily data | Free daily data (yfinance) |
| 2. Day-trade battery | Same on 1-minute data, plus intraday-only hypotheses | Alpaca paper account keys (data only) |
| 3. Daily email | GitHub Actions emails picks from survivors each weekday | Private GitHub repo, Gmail app password |
| 4. Live | Bella places picks manually in a $1,000 Fidelity "lab" cash account | Lab account opened |

Alpaca auto-trading is out of scope (later, once phase 4 has a track record).

## Universe and data

- ~60 symbols: SPY, QQQ, IWM, DIA plus ~50 large, liquid US stocks (AAPL, MSFT, NVDA,
  AMZN, GOOGL, META, ...). Fixed list, written down before testing.
- Daily: adjusted OHLCV from yfinance, as far back as each symbol allows (20+ years for most).
- Intraday: Alpaca 1-minute IEX bars, 2016 onward.
- VOO launched 2010, so the buy-and-hold benchmark uses SPY before 2010 and VOO after.

## Hypotheses (pre-registered — this list is fixed before any run)

All swing hypotheses enter at the close on signal day and exit at the close after the hold.

**Tesla group**
- **T1 · 3-6-9 days:** the day of the month's digit root is 3, 6 or 9. Hold 1 day.
- **T2 · 3-6-9 holds:** after a 20-day-high breakout, holds of 3/6/9 days beat holds of 4/5/7/8.
- **T3 · 3-6-9 price levels:** when the close crosses below a whole-dollar level whose digit root is 3, 6 or 9, buy for a bounce. Hold 3 days. Control: levels with other digit roots.
- **T4 · 3-6-9 streaks:** after exactly 3, 6 or 9 consecutive down closes, buy the reversal. Hold 3 days.
- **T5 · Frequency:** take the dominant cycle (3–60 days) from an FFT of the last 252 days of detrended price; buy at the projected trough and hold half a cycle.
- **T6 · Resonance:** the 3-, 6- and 9-day band-pass cycle components all turn up on the same day. Hold 3 days.
- **T7 · Oscillation:** the 9-day z-score is below −2; buy and exit on reversion to the mean or after 9 days.

**Numerology / personal (life path 9; master number 22; 222)**
- **N1 · Date root:** the digit root of the full date (month + day + year) is 3, 6 or 9. Hold 1 day.
- **N2 · 22 days:** the 22nd of any month, plus 2/22. Hold 1 day.
- **N3 · Personal day:** Bella's personal day number (birth month + birth day + current date, reduced) is 9 or 22. Hold 1 day.
- **N4 · 222 prices:** the close's cents are .22, or the price contains "222". Hold 1 day.

**Pattern recognition**
- **P1 · Echo:** take the last 9 days of normalized returns and find the 9 most similar past windows (past only). Buy if their average next-3-day return is positive and at least 6 of the 9 agree. Hold 3 days.

**Intraday only (phase 2)**
- **I1 · 2:22 PM:** buy at 2:22 PM ET and sell at the close, compared against every other minute.
- **I2–I4:** T4, T6 and T7 rebuilt on 3/6/9-minute bars, exiting the same day.

**Controls (not hypotheses — the bars to clear)**
- **C0:** buy and hold SPY/VOO.
- **C1:** random entries with the same frequency and hold as each hypothesis (the permutation test).
- **C2:** classic 12-1 momentum (a known real effect, to show the harness can detect an edge).

## Pass criteria (all required)

1. **Holdout locked.** Discovery uses data through 2023-10-31. 2023-11-01 → today is touched once, at the end.
2. **Net of costs:** 5 bp round trip on stocks, 2 bp on ETFs.
3. **Luck test:** permutation p-value against C1 (1,000 shuffles), pooled across the universe, is below **0.05 / number of hypotheses** (Bonferroni; about 0.003 for 15 tests).
4. **Walk-forward:** positive in a majority of the 5-year-train / 2-year-test folds.
5. **Holdout:** same sign and at least half the discovery-period effect size.
6. **Beats VOO risk-adjusted:** as a $1K portfolio, its Sharpe beats C0 over the same period.

Every hypothesis is reported, pass or fail, in `results/scoreboard.csv`, with a one-page
plain-English `results/report.md`.

## Daily email (phase 3)

- GitHub Actions cron on weekdays at 12:45 UTC (8:45 ET in summer, 7:45 in winter).
- Runs only the survivors on the latest daily data.
- Sizing: the $1,000 lab account has at most 4 positions of $250. A holding period of at least 1 day keeps every buy on settled cash (no Good Faith Violations).
- Email to isabellarosemarco@gmail.com, sent via Gmail SMTP with an app password stored as a repo secret:
  - **Buy today:** symbol, $ amount, exit date, stop (−5%), signal name.
  - **Sell today:** exits that fall due today. These are computed from entry date + hold, so no stored state is needed.
  - **No trades today** when there are none. If no hypothesis survives, the email reports that and stays off.

## Structure

```
resonance_lab/
  lab/data.py         # daily + Alpaca loaders, caching (adapted from ~/quant_alpha/data_loader.py)
  lab/numerology.py   # digit root, personal day, date/price predicates (pure functions)
  lab/cycles.py       # FFT dominant cycle, band-pass, resonance detection
  lab/hypotheses.py   # one function per hypothesis -> entry dates + hold
  lab/stats.py        # permutation test, HAC t, walk-forward, Bonferroni (adapted from structural_anomalies_tester.py)
  lab/portfolio.py    # $1K / 4-slot simulation vs VOO
  run_battery.py      # phase 1/2 runner -> results/
  daily_email.py      # phase 3
  .github/workflows/daily.yml
  tests/              # unit tests for numerology, cycles, stats, no-lookahead checks
```

## Error handling and safety

- **No lookahead:** every signal uses only data available at the close of the signal day. A test shifts the data and asserts the signals don't change.
- If the daily-email job fails to fetch data, it emails "data fetch failed — no picks today" rather than staying silent.
- **Lab cap:** $1,000. It's only ever added to from its own profits.
- No API keys or passwords in the repo. They live in GitHub secrets and a local `.env` that's gitignored.
